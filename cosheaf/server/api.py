"""Read-only local HTTP API for website preview."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from difflib import unified_diff
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Literal, cast
from urllib.parse import parse_qs, unquote, urlparse

from cosheaf.agent.context_pack import PACK_FILENAMES
from cosheaf.app import CosheafApp
from cosheaf.core.artifact import BaseArtifact
from cosheaf.core.ids import validate_artifact_id
from cosheaf.core.status import ArtifactStatus, ArtifactType, is_preaccepted_status
from cosheaf.forge import (
    FORGE_AUTHORITY_WARNING,
    FORGE_PREVIEW_AUTHORITY_WARNING,
    ForgeActionError,
    ForgeActionResult,
    ForgeCredentialProvider,
)
from cosheaf.issues import ISSUE_AUTHORITY_NOTICE, LocalIssueError
from cosheaf.server.auth import (
    HostedAuthProvider,
    HostedIdentity,
    hosted_action_allowed,
    hosted_required_role,
)
from cosheaf.services import PROMOTION_CONFIRMATION_PHRASES, DraftWriteServiceError
from cosheaf.site import REQUIRED_SITE_EXPORT_FILES, SITE_EXPORT_AUTHORITY_NOTICE
from cosheaf.storage.loader import LoadedRecord, LoadError, load_artifacts
from cosheaf.storage.writer import dump_yaml_deterministic
from cosheaf.web_actions import (
    WEB_ACTION_AUDIT_PATH,
    WebActionAuditEntry,
    WebActionError,
    WebActionKind,
    WebActionMode,
    append_web_action_audit,
)

READONLY_SERVER_HOST = "127.0.0.1"
READONLY_SERVER_PORT = 8765
READONLY_SERVER_SCHEMA_VERSION = 1
CONTEXT_AUTHORITY_NOTICE = (
    "Context packs are retrieval context only; they are not proof, verifier "
    "pass, gate pass, human review, accepted status, or promotion authority."
)
CHECK_AUTHORITY_NOTICE = (
    "Validation and gate output are workflow context only; skipped is not pass, "
    "and gate pass is not accepted status or promotion authority."
)
ARTIFACT_WRITE_AUTHORITY_NOTICE = (
    "Web artifact writes can create or edit draft/pre-accepted records and "
    "attach source/evidence metadata only. They do not create proof, accepted "
    "knowledge, human review, verifier pass, gate pass, or promotion authority."
)
REVIEW_PACKET_AUTHORITY_NOTICE = (
    "Review packets are informational context for human reviewers. They do not "
    "create proof, mark human review complete, pass gates, set accepted status, "
    "or grant promotion authority."
)
REVIEW_DECISION_AUTHORITY_NOTICE = (
    "Human review decisions record reviewer judgment and may update artifact "
    "review state. They do not create accepted status, gate pass, verifier "
    "pass, or promotion authority. AI/Codex cannot be recorded as reviewer."
)
PROMOTION_AUTHORITY_NOTICE = (
    "Promotion readiness and preview output are advisory workflow context. "
    "They do not write accepted artifacts, change artifact status, pass gates, "
    "mutate verifier evidence, or grant promotion authority."
)
LOCAL_ACTOR_NOTICE = (
    "Local actor records who operated this local Workbench session. It is not "
    "authentication, authorization, or cryptographic identity."
)
_EXPORT_ENDPOINTS = {
    "/api/workspace": "workspace.json",
    "/api/artifacts": "artifacts.json",
    "/api/issues": "issues.json",
    "/api/graph": "graph.json",
    "/api/gates": "gates.json",
}
_PREVIEW_ENDPOINTS = {
    "/api/forge/branch/preview",
    "/api/forge/commit/preview",
    "/api/forge/local-issues/preview",
    "/api/forge/issues/preview",
    "/api/forge/pr/preview",
    "/api/forge/prs/preview",
    "/api/forge/push/preview",
    "/api/forge/review-packets/preview",
    "/api/issues/preview-create",
    "/api/reviews/packets/preview",
    "/api/reviews/decisions/preview",
}
_CREATE_ENDPOINTS = {
    "/api/forge/branch/create",
    "/api/forge/commit/create",
    "/api/forge/issues/create",
    "/api/forge/pr/create",
    "/api/forge/prs/create",
    "/api/forge/push/create",
    "/api/issues/create",
    "/api/artifacts/create",
    "/api/reviews/packets/create",
    "/api/reviews/decisions/create",
}
_RUN_ENDPOINTS = {
    "/api/validate/run",
    "/api/gate/run",
}
_ISSUE_ACTION_SUFFIXES = {
    "preview-update",
    "update",
    "preview-close",
    "close",
}
_ARTIFACT_ACTION_SUFFIXES = {
    "preview-evidence",
    "evidence",
    "preview-source",
    "source",
    "preview-update",
    "update",
}
_ARTIFACT_PREVIEW_ENDPOINTS = {
    "/api/artifacts/preview-create",
}


@dataclass(frozen=True)
class ApiResponse:
    """JSON response returned by the read-only local API router."""

    status: int
    payload: dict[str, Any]

    def body(self) -> bytes:
        """Serialize this response body deterministically."""
        return (
            json.dumps(self.payload, ensure_ascii=True, indent=2, sort_keys=True)
            + "\n"
        ).encode("utf-8")


class ReadOnlySiteApi:
    """Route read-only website API requests through the app facade."""

    def __init__(
        self,
        app: CosheafApp,
        *,
        host: str = READONLY_SERVER_HOST,
        credential_provider: ForgeCredentialProvider | None = None,
        local_actor: str | None = None,
        web_action_mode: WebActionMode = WebActionMode.LOCAL,
        hosted_auth_provider: HostedAuthProvider | None = None,
        strict_readonly: bool = False,
    ) -> None:
        self.app = app
        self.host = host
        self.credential_provider = credential_provider
        self.local_actor = _normalize_local_actor(local_actor)
        self.web_action_mode = WebActionMode(web_action_mode)
        self.hosted_auth_provider = hosted_auth_provider
        self.strict_readonly = strict_readonly
        self._artifact_write_plans: dict[tuple[str, str], dict[str, Any]] = {}

    def handle(
        self,
        method: str,
        raw_path: str,
        body: str | bytes = b"",
    ) -> ApiResponse:
        """Return a JSON response for one HTTP-like request."""
        normalized_method = method.upper()
        path = _normalize_path(raw_path)
        if normalized_method == "GET":
            return self._handle_get(path, raw_path)
        if normalized_method == "POST":
            return self._handle_post(path, body)
        return _error(
            405,
            "method_not_allowed",
            "The local website API is read-only and only accepts GET or "
            "preview-only POST requests.",
        )

    def _handle_get(self, path: str, raw_path: str) -> ApiResponse:
        if path in _PREVIEW_ENDPOINTS:
            return _error(
                405,
                "method_not_allowed",
                "Preview endpoints only accept POST.",
            )
        if path == "/api/health":
            return ApiResponse(200, self._health_payload())
        if path == "/api/workspace/live":
            return self._live_workspace_payload()
        if path == "/api/status":
            return self._live_status_payload()
        if path == "/api/issues/live":
            return self._live_issues_payload()
        if path.startswith("/api/issues/"):
            return self._live_issue_payload(path.removeprefix("/api/issues/"))
        if path == "/api/artifacts/live":
            return self._live_artifacts_payload()
        artifact_promotion = _parse_artifact_promotion_path(path)
        if artifact_promotion is not None:
            artifact_id, action = artifact_promotion
            if action == "readiness":
                return self._artifact_promotion_readiness(artifact_id)
        if path.startswith("/api/artifacts/"):
            return self._live_artifact_payload(path.removeprefix("/api/artifacts/"))
        if path.startswith("/api/context/") and path.endswith("/latest"):
            issue_id = path.removeprefix("/api/context/").removesuffix("/latest")
            return self._live_context_latest_payload(issue_id)
        if path == "/api/gates/latest":
            return self._live_gate_latest_payload()
        if path == "/api/audit/recent":
            return self._live_audit_recent_payload()
        if path == "/api/forge/pr-status":
            return self._forge_pr_status_payload(_query_params(raw_path))
        if path in _EXPORT_ENDPOINTS:
            return ApiResponse(200, self._payload_file(_EXPORT_ENDPOINTS[path]))
        if path.startswith("/api/context/"):
            issue_id = path.removeprefix("/api/context/")
            return self._context_payload(issue_id)
        return _error(404, "not_found", f"Unknown endpoint: {path}")

    def _handle_post(self, path: str, body: str | bytes) -> ApiResponse:
        issue_action = _parse_issue_action_path(path)
        artifact_action = _parse_artifact_action_path(path)
        artifact_promotion = _parse_artifact_promotion_path(path)
        context_action = _parse_context_action_path(path)
        if (
            path not in _PREVIEW_ENDPOINTS
            and path not in _ARTIFACT_PREVIEW_ENDPOINTS
            and path not in _CREATE_ENDPOINTS
            and path not in _RUN_ENDPOINTS
            and issue_action is None
            and artifact_action is None
            and artifact_promotion is None
            and context_action is None
        ):
            return _error(
                405,
                "method_not_allowed",
                "POST is only available for preview or authenticated create "
                "endpoints.",
            )
        payload = _json_body(body)
        if isinstance(payload, ApiResponse):
            return payload
        route_action = _post_route_action(
            path,
            issue_action=issue_action,
            artifact_action=artifact_action,
            artifact_promotion=artifact_promotion,
            context_action=context_action,
        )
        blocked = self._blocked_hosted_action(
            route_action,
            payload,
            preview_only=_post_route_is_preview(path, artifact_promotion),
        )
        if blocked is not None:
            return blocked
        if (
            self.strict_readonly
            and not _post_route_is_preview(path, artifact_promotion)
            and path != "/api/validate/run"
        ):
            self._write_audit(
                action=route_action,
                result_status="readonly_mode_writes_disabled",
                explicit_confirm=bool(payload.get("confirm")),
                preview_only=False,
                authority_warnings=_authority_warnings_for_action(route_action),
            )
            return _error(
                403,
                "readonly_mode_writes_disabled",
                "Confirmed write endpoints are disabled in strict read-only mode.",
            )
        try:
            if path == "/api/artifacts/preview-create":
                return self._preview_artifact_create(payload)
            if path == "/api/artifacts/create":
                return self._create_artifact(payload)
            if artifact_action is not None:
                artifact_id, action = artifact_action
                if action == "preview-source":
                    return self._preview_artifact_source(artifact_id, payload)
                if action == "source":
                    return self._add_artifact_source(artifact_id, payload)
                if action == "preview-evidence":
                    return self._preview_artifact_evidence(artifact_id, payload)
                if action == "evidence":
                    return self._add_artifact_evidence(artifact_id, payload)
                if action == "preview-update":
                    return self._preview_artifact_update(artifact_id, payload)
                if action == "update":
                    return self._update_artifact(artifact_id, payload)
            if artifact_promotion is not None:
                artifact_id, action = artifact_promotion
                if action == "preview":
                    return self._preview_artifact_promotion(artifact_id, payload)
                if action == "confirm":
                    return self._confirm_artifact_promotion(artifact_id, payload)
            if path == "/api/issues/preview-create":
                return self._preview_issue_create(payload)
            if path == "/api/issues/create":
                return self._create_issue(payload)
            if issue_action is not None:
                issue_id, action = issue_action
                if action == "preview-update":
                    return self._preview_issue_update(issue_id, payload)
                if action == "update":
                    return self._update_issue(issue_id, payload)
                if action == "preview-close":
                    return self._preview_issue_close(issue_id, payload)
                if action == "close":
                    return self._close_issue(issue_id, payload)
            if context_action is not None:
                issue_id, action = context_action
                if action == "preview-build":
                    return self._preview_context_build(issue_id, payload)
                if action == "build":
                    return self._build_context(issue_id, payload)
            if path == "/api/validate/run":
                return self._run_validate()
            if path == "/api/gate/run":
                return self._run_gate()
            if path == "/api/reviews/packets/preview":
                return self._preview_review_packet(payload)
            if path == "/api/reviews/packets/create":
                return self._create_review_packet(payload)
            if path == "/api/reviews/decisions/preview":
                return self._preview_review_decision(payload)
            if path == "/api/reviews/decisions/create":
                return self._create_review_decision(payload)
            if path == "/api/forge/local-issues/preview":
                return self._preview_local_issue(payload)
            if path == "/api/forge/issues/preview":
                return self._preview_github_issue(payload)
            if path == "/api/forge/branch/preview":
                return self._preview_forge_branch(payload)
            if path == "/api/forge/commit/preview":
                return self._preview_forge_commit(payload)
            if path == "/api/forge/push/preview":
                return self._preview_forge_push(payload)
            if path == "/api/forge/pr/preview":
                return self._preview_github_pr(payload)
            if path == "/api/forge/prs/preview":
                return self._preview_github_pr(payload)
            if path == "/api/forge/review-packets/preview":
                return self._preview_forge_review_packet(payload)
            if path == "/api/forge/branch/create":
                return self._create_forge_branch(payload)
            if path == "/api/forge/commit/create":
                return self._create_forge_commit(payload)
            if path == "/api/forge/push/create":
                return self._create_forge_push(payload)
            if path == "/api/forge/issues/create":
                return self._create_github_issue(payload)
            if path == "/api/forge/pr/create":
                return self._submit_github_pr(payload)
            if path == "/api/forge/prs/create":
                return self._create_github_pr(payload)
        except ForgeActionError as exc:
            return _error(400, exc.code, "forge action failed; see backend audit")
        except LocalIssueError as exc:
            return _error(400, "issue_action_failed", str(exc))
        except DraftWriteServiceError as exc:
            return _error(400, exc.code, str(exc))
        except ValueError as exc:
            return _error(400, "preview_invalid_input", str(exc))
        return _error(404, "not_found", f"Unknown endpoint: {path}")

    def _health_payload(self) -> dict[str, Any]:
        confirmed_writes_enabled = not self.strict_readonly
        return {
            "schema_version": READONLY_SERVER_SCHEMA_VERSION,
            "status": "ok",
            "readonly": True,
            "strict_readonly": self.strict_readonly,
            "confirmed_writes_enabled": confirmed_writes_enabled,
            "capabilities": {
                "read": True,
                "preview": True,
                "confirmed_local_writes": confirmed_writes_enabled,
                "confirmed_forge_actions": (
                    confirmed_writes_enabled and self.credential_provider is not None
                ),
                "direct_accepted_writes": False,
            },
            "host": self.host,
            "repo_root": str(self.app.context.repo_root),
            "local_actor": self.local_actor,
            "local_actor_configured": self.local_actor is not None,
            "local_actor_is_auth": False,
            "local_actor_notice": LOCAL_ACTOR_NOTICE,
            "web_action_mode": self.web_action_mode,
            "hosted_auth_configured": self.hosted_auth_provider is not None,
            "hosted_auth_production_ready": False,
            "authority_notice": SITE_EXPORT_AUTHORITY_NOTICE,
        }

    def _live_workspace_payload(self) -> ApiResponse:
        return _live_response(
            "workspace_live",
            workspace=_workspace_info_payload(self.app.workspace_info()),
        )

    def _live_status_payload(self) -> ApiResponse:
        validation = self.app.validate_repository()
        return _live_response(
            "repository_status",
            status="ok" if validation.ok else "validation_failed",
            workspace=_workspace_info_payload(self.app.workspace_info()),
            validation=_validation_payload(validation),
        )

    def _live_issues_payload(self) -> ApiResponse:
        result = self.app.list_issues()
        issues = [
            {
                **issue.model_dump(mode="json"),
                "path": path.as_posix(),
                "context_pack": _issue_context_pack_summary(
                    self.app.context.repo_root,
                    issue.id,
                ),
            }
            for issue, path in zip(result.issues, result.paths, strict=True)
        ]
        return _live_response(
            "issues_live",
            issues=issues,
            count=len(issues),
            authority_notice=ISSUE_AUTHORITY_NOTICE,
        )

    def _live_issue_payload(self, issue_id: str) -> ApiResponse:
        try:
            result = self.app.show_issue(issue_id)
        except (LocalIssueError, ValueError):
            return _error(
                404,
                "issue_not_found",
                f"No repository-local issue exists for: {issue_id}",
            )
        issue = result.issue.model_dump(mode="json")
        issue["path"] = result.relative_path.as_posix()
        return _live_response(
            "issue_live",
            issue=issue,
            authority_notice=result.authority_notice,
        )

    def _live_artifacts_payload(self) -> ApiResponse:
        loaded = self._live_artifact_records()
        if isinstance(loaded, ApiResponse):
            return loaded
        artifacts = [_artifact_payload(record) for record in loaded]
        return _live_response(
            "artifacts_live",
            artifacts=artifacts,
            count=len(artifacts),
        )

    def _live_artifact_payload(self, artifact_id: str) -> ApiResponse:
        try:
            normalized_id = validate_artifact_id(artifact_id)
        except ValueError:
            return _error(
                404,
                "artifact_not_found",
                f"No lifecycle artifact exists for: {artifact_id}",
            )
        loaded = self._live_artifact_records()
        if isinstance(loaded, ApiResponse):
            return loaded
        for record in loaded:
            if record.id == normalized_id:
                return _live_response(
                    "artifact_live",
                    artifact=_artifact_payload(record),
                )
        return _error(
            404,
            "artifact_not_found",
            f"No lifecycle artifact exists for: {normalized_id}",
        )

    def _artifact_promotion_readiness(self, artifact_id: str) -> ApiResponse:
        try:
            normalized_id = validate_artifact_id(artifact_id)
            report = self.app.promotion_readiness(artifact_id=normalized_id)
        except ValueError as exc:
            return _error(400, "promotion_readiness_failed", str(exc))
        payload = report.to_dict()
        if not payload["artifacts"] and any(
            reason["code"] == "artifact_not_found" for reason in payload["reasons"]
        ):
            return _error(
                404,
                "artifact_not_found",
                f"No lifecycle artifact exists for: {normalized_id}",
            )
        return _live_response(
            "promotion_readiness",
            promotion_readiness=payload,
            ready=payload["ready"],
            accepted_write_performed=False,
            promotion_performed=False,
            authority_notice=PROMOTION_AUTHORITY_NOTICE,
        )

    def _live_context_latest_payload(self, issue_id: str) -> ApiResponse:
        if not issue_id:
            return _error(404, "issue_not_found", "No issue id was provided.")
        try:
            self.app.show_issue(issue_id)
        except (LocalIssueError, ValueError):
            return _error(
                404,
                "issue_not_found",
                f"No repository-local issue exists for: {issue_id}",
            )
        task_dir = self.app.context.resolve(Path("context") / "TASKS" / issue_id)
        files = _repo_files_under(self.app.context.repo_root, task_dir)
        audit = _read_json_file(task_dir / "RETRIEVAL_AUDIT.json")
        return _live_response(
            "context_pack_latest",
            issue_id=issue_id,
            context_pack={
                "exists": task_dir.is_dir(),
                "task_dir": _repo_relative_or_string(
                    self.app.context.repo_root,
                    task_dir,
                ),
                "files": files,
                "retrieval_audit": audit,
                "authority_notice": (
                    CONTEXT_AUTHORITY_NOTICE
                ),
            },
        )

    def _live_gate_latest_payload(self) -> ApiResponse:
        reports_dir = self.app.context.resolve(Path(".cosheaf") / "reports")
        candidates = sorted(reports_dir.glob("*-gate-report.json"))
        if not candidates:
            return _live_response(
                "gate_latest",
                gate_report={
                    "exists": False,
                    "path": None,
                    "report": None,
                    "verdict": "not_run",
                    "note": "No gate report exists under .cosheaf/reports.",
                },
            )
        latest = candidates[-1]
        report = _read_json_file(latest)
        if report is None:
            return _error(
                500,
                "gate_report_unreadable",
                "Latest gate report could not be read as JSON.",
            )
        return _live_response(
            "gate_latest",
            gate_report={
                "exists": True,
                "path": _repo_relative_or_string(self.app.context.repo_root, latest),
                "report": report,
                "verdict": report.get("verdict"),
            },
        )

    def _live_audit_recent_payload(self) -> ApiResponse:
        audit_path = self.app.context.resolve(WEB_ACTION_AUDIT_PATH)
        entries = _read_jsonl_file(audit_path)
        if isinstance(entries, ApiResponse):
            return entries
        recent = entries[-50:]
        return _live_response(
            "web_action_audit_recent",
            audit={
                "exists": audit_path.is_file(),
                "path": WEB_ACTION_AUDIT_PATH.as_posix(),
            },
            entries=recent,
            count=len(recent),
        )

    def _live_artifact_records(self) -> tuple[LoadedRecord, ...] | ApiResponse:
        try:
            records = tuple(load_artifacts(self.app.context))
        except LoadError as exc:
            return _error(
                500,
                "repository_load_failed",
                f"Cannot load repository records: {exc}",
            )
        return tuple(
            record for record in records if isinstance(record.record, BaseArtifact)
        )

    def _forge_pr_status_payload(self, query: dict[str, str]) -> ApiResponse:
        number_text = _optional_text(query, "number")
        number: int | None = None
        if number_text is not None:
            try:
                number = int(number_text)
            except ValueError:
                return _error(400, "invalid_pr_number", "PR number must be an integer")
            if number <= 0:
                return _error(400, "invalid_pr_number", "PR number must be positive")
        head = _optional_text(query, "head") or ""
        if number is None and not head:
            return _error(
                400,
                "pr_selector_required",
                "PR number or head branch is required",
            )
        result = self.app.forge_github_pr_status(
            number=number,
            base=_optional_text(query, "base") or "",
            head=head,
        )
        result_payload = result.model_dump(mode="json")
        return _live_response(
            "github_pr_status",
            pr_status=result_payload,
            github_auth_available=result.github_auth_available,
            network_calls_performed=result.network_calls_performed,
            git_writes_performed=False,
            github_writes_performed=False,
            authority_notice=FORGE_AUTHORITY_WARNING,
        )

    def _context_payload(self, issue_id: str) -> ApiResponse:
        payload = self._payload_file("context_packs.json")
        for context_pack in payload["context_packs"]:
            if context_pack["issue_id"] == issue_id:
                return ApiResponse(
                    200,
                    {
                        "schema_version": READONLY_SERVER_SCHEMA_VERSION,
                        "kind": "context_pack",
                        "issue_id": issue_id,
                        "context_pack": context_pack,
                        "authority_notice": SITE_EXPORT_AUTHORITY_NOTICE,
                    },
                )
        return _error(
            404,
            "context_pack_not_found",
            f"No exported context pack summary for issue: {issue_id}",
        )

    def _preview_local_issue(self, payload: dict[str, Any]) -> ApiResponse:
        result = self.app.preview_issue(
            issue_id=_required_text(payload, "issue_id"),
            title=_required_text(payload, "title"),
            summary=_optional_text(payload, "summary"),
            authors=_text_list(payload, "authors"),
            labels=_text_list(payload, "labels"),
            related_artifacts=_text_list(payload, "related_artifacts"),
            related_sources=_text_list(payload, "related_sources"),
            scope=_scope(payload.get("scope", "private")),
        )
        planned_file = result.relative_path.as_posix()
        self._write_audit(
            action="local_issue_preview",
            result_status="preview",
            explicit_confirm=False,
            preview_only=True,
            planned_files=[planned_file],
        )
        return _preview_response(
            "local_issue_preview",
            planned_actions=["create repository-local issue YAML preview"],
            planned_files=[planned_file],
            issue=result.issue.model_dump(mode="json"),
        )

    def _preview_github_issue(self, payload: dict[str, Any]) -> ApiResponse:
        source_path = _required_text(payload, "source_path")
        result = self.app.forge_issue_preview(source_path)
        result_payload = result.model_dump(mode="json")
        self._write_audit(
            action="github_issue_preview",
            result_status="preview",
            explicit_confirm=False,
            preview_only=True,
            planned_files=[source_path],
        )
        return _preview_response(
            "github_issue_preview",
            planned_actions=["create GitHub issue preview"],
            planned_files=[source_path],
            forge_preview=result_payload,
            github_issue_plan=result_payload.get("github_issue_plan"),
        )

    def _preview_forge_branch(self, payload: dict[str, Any]) -> ApiResponse:
        branch = _required_text(payload, "branch")
        allow_dirty = _bool(payload, "allow_dirty", default=False)
        self._write_audit(
            action="branch_preview",
            result_status="preview",
            explicit_confirm=False,
            preview_only=True,
            branch=branch,
        )
        action = f"create and switch branch {branch}"
        if allow_dirty:
            action += " carrying current worktree changes"
        return _preview_response(
            "branch_preview",
            planned_actions=[action],
            planned_files=[],
            branch=branch,
            allow_dirty=allow_dirty,
        )

    def _preview_forge_commit(self, payload: dict[str, Any]) -> ApiResponse:
        message = _required_text(payload, "message")
        stage_all = _bool(payload, "stage_all", default=False)
        self._write_audit(
            action="commit_preview",
            result_status="preview",
            explicit_confirm=False,
            preview_only=True,
        )
        actions = ["run validation, run gate, commit staged changes"]
        if stage_all:
            actions.insert(0, "stage all repository changes")
        return _preview_response(
            "commit_preview",
            planned_actions=actions,
            planned_files=[],
            commit_message=message,
            stage_all=stage_all,
        )

    def _preview_forge_push(self, payload: dict[str, Any]) -> ApiResponse:
        head = _required_text(payload, "head")
        self._write_audit(
            action="push_preview",
            result_status="preview",
            explicit_confirm=False,
            preview_only=True,
            head=head,
        )
        return _preview_response(
            "push_preview",
            planned_actions=[f"git push -u origin {head}"],
            planned_files=[],
            head=head,
        )

    def _preview_github_pr(self, payload: dict[str, Any]) -> ApiResponse:
        base = _required_text(payload, "base")
        head = _required_text(payload, "head")
        result = self.app.forge_pr_preview(
            base=base,
            head=head,
        )
        result_payload = result.model_dump(mode="json")
        self._write_audit(
            action="github_pr_preview",
            result_status="preview",
            explicit_confirm=False,
            preview_only=True,
            planned_files=[],
            base=base,
            head=head,
        )
        return _preview_response(
            "github_pr_preview",
            planned_actions=["create GitHub pull request preview"],
            planned_files=[],
            forge_preview=result_payload,
            github_pr_plan=result_payload.get("github_pr_plan"),
        )

    def _preview_forge_review_packet(self, payload: dict[str, Any]) -> ApiResponse:
        issue_id = _required_text(payload, "issue_id")
        issue = self.app.show_issue(issue_id).issue
        planned_file = f"reviews/website/{issue_id}-review-packet.md"
        self._write_audit(
            action="review_packet_preview",
            result_status="preview",
            explicit_confirm=False,
            preview_only=True,
            planned_files=[planned_file],
        )
        return _preview_response(
            "review_packet_preview",
            planned_actions=["create review packet markdown preview"],
            planned_files=[planned_file],
            issue=issue.model_dump(mode="json"),
            review_packet={
                "issue_id": issue_id,
                "title": issue.title,
                "sections": [
                    "issue summary",
                    "related artifacts",
                    "context pack",
                    "authority-boundary checklist",
                ],
            },
        )

    def _preview_review_packet(self, payload: dict[str, Any]) -> ApiResponse:
        try:
            packet = _build_review_packet(self, payload)
            result = self.app.write_review_request(packet["request"], dry_run=True)
        except (DraftWriteServiceError, LocalIssueError, LoadError, ValueError) as exc:
            return _error(400, "review_packet_failed", str(exc))
        planned_file = result.relative_path.as_posix()
        self._write_audit(
            action="review_packet_create",
            result_status="preview",
            explicit_confirm=False,
            preview_only=True,
            planned_files=[planned_file],
            authority_warnings=[REVIEW_PACKET_AUTHORITY_NOTICE],
        )
        return _preview_response(
            "review_packet_preview",
            planned_actions=["create draft informational review packet preview"],
            planned_files=[planned_file],
            review_packet=packet["packet"],
            review_request=packet["request"],
            path=planned_file,
            authority_warning=REVIEW_PACKET_AUTHORITY_NOTICE,
            authority_notice=REVIEW_PACKET_AUTHORITY_NOTICE,
        )

    def _create_review_packet(self, payload: dict[str, Any]) -> ApiResponse:
        action = "review_packet_create"
        blocked = self._blocked_confirm(
            action,
            payload,
            authority_warnings=[REVIEW_PACKET_AUTHORITY_NOTICE],
        )
        if blocked is not None:
            return blocked
        try:
            packet = _build_review_packet(self, payload)
            result = self.app.write_review_request(packet["request"], dry_run=False)
        except (DraftWriteServiceError, LocalIssueError, LoadError, ValueError) as exc:
            self._write_audit(
                action=action,
                result_status="review_packet_failed",
                explicit_confirm=True,
                preview_only=False,
                authority_warnings=[REVIEW_PACKET_AUTHORITY_NOTICE],
            )
            return _error(400, "review_packet_failed", str(exc))
        planned_file = result.relative_path.as_posix()
        self._write_audit(
            action=action,
            result_status="success",
            explicit_confirm=True,
            preview_only=False,
            planned_files=[planned_file],
            repo_writes_performed=True,
            result={"action_performed": True},
            authority_warnings=[REVIEW_PACKET_AUTHORITY_NOTICE],
        )
        return ApiResponse(
            200,
            {
                "schema_version": READONLY_SERVER_SCHEMA_VERSION,
                "kind": "review_packet_create",
                "action_performed": True,
                "repo_writes_performed": True,
                "git_writes_performed": False,
                "github_writes_performed": False,
                "network_calls_performed": False,
                "audit_logged": True,
                "planned_files": [planned_file],
                "written_files": [path.as_posix() for path in result.written_paths],
                "path": planned_file,
                "review_id": result.record_id,
                "review_packet": packet["packet"],
                "review_request": packet["request"],
                "authority_warning": REVIEW_PACKET_AUTHORITY_NOTICE,
                "authority_notice": REVIEW_PACKET_AUTHORITY_NOTICE,
            },
        )

    def _preview_review_decision(self, payload: dict[str, Any]) -> ApiResponse:
        action = "review_decision_create"
        try:
            result = self.app.write_review_decision(payload, dry_run=True)
        except DraftWriteServiceError as exc:
            self._write_audit(
                action=action,
                result_status=exc.code,
                explicit_confirm=False,
                preview_only=True,
                authority_warnings=[REVIEW_DECISION_AUTHORITY_NOTICE],
            )
            return _error(400, exc.code, str(exc))
        planned_files = _review_decision_planned_files(result)
        self._write_audit(
            action=action,
            result_status="preview",
            explicit_confirm=False,
            preview_only=True,
            planned_files=planned_files,
            authority_warnings=[REVIEW_DECISION_AUTHORITY_NOTICE],
        )
        return _preview_response(
            "review_decision_preview",
            planned_actions=["record explicit human review decision preview"],
            planned_files=planned_files,
            review_decision=result.review,
            artifact_update=result.artifact.model_dump(mode="json"),
            artifact_review_state_changed=result.artifact_updated,
            accepted_write_performed=False,
            promotion_performed=False,
            path=result.relative_path.as_posix(),
            authority_warning=REVIEW_DECISION_AUTHORITY_NOTICE,
            authority_notice=REVIEW_DECISION_AUTHORITY_NOTICE,
        )

    def _create_review_decision(self, payload: dict[str, Any]) -> ApiResponse:
        action = "review_decision_create"
        blocked = self._blocked_confirm(
            action,
            payload,
            authority_warnings=[REVIEW_DECISION_AUTHORITY_NOTICE],
        )
        if blocked is not None:
            return blocked
        actor, actor_blocked = self._require_confirm_actor(
            action,
            authority_warnings=[REVIEW_DECISION_AUTHORITY_NOTICE],
        )
        if actor_blocked is not None:
            return actor_blocked
        try:
            result = self.app.write_review_decision(payload, dry_run=False)
        except DraftWriteServiceError as exc:
            self._write_audit(
                action=action,
                result_status=exc.code,
                explicit_confirm=True,
                preview_only=False,
                actor=actor,
                authority_warnings=[REVIEW_DECISION_AUTHORITY_NOTICE],
            )
            return _error(400, exc.code, str(exc))
        planned_files = _review_decision_planned_files(result)
        self._write_audit(
            action=action,
            result_status="success",
            explicit_confirm=True,
            preview_only=False,
            actor=actor,
            planned_files=planned_files,
            repo_writes_performed=True,
            result={"action_performed": True},
            authority_warnings=[REVIEW_DECISION_AUTHORITY_NOTICE],
        )
        return ApiResponse(
            200,
            {
                "schema_version": READONLY_SERVER_SCHEMA_VERSION,
                "kind": "review_decision_create",
                "action_performed": True,
                "repo_writes_performed": True,
                "git_writes_performed": False,
                "github_writes_performed": False,
                "network_calls_performed": False,
                "audit_logged": True,
                "planned_files": planned_files,
                "written_files": [path.as_posix() for path in result.written_paths],
                "path": result.relative_path.as_posix(),
                "review_id": result.record_id,
                "review_decision": result.review,
                "artifact": result.artifact.model_dump(mode="json"),
                "artifact_review_state_changed": result.artifact_updated,
                "accepted_write_performed": False,
                "promotion_performed": False,
                "authority_warning": REVIEW_DECISION_AUTHORITY_NOTICE,
                "authority_notice": REVIEW_DECISION_AUTHORITY_NOTICE,
            },
        )

    def _preview_artifact_promotion(
        self,
        artifact_id: str,
        payload: dict[str, Any],
    ) -> ApiResponse:
        action = "promotion_preview"
        normalized_id = validate_artifact_id(artifact_id)
        target_state = _promotion_target_state(payload.get("target_state", "accepted"))
        actor = self._preview_actor(payload)
        report = self.app.promotion_readiness(artifact_id=normalized_id)
        readiness = report.to_dict()
        missing = _promotion_missing_requirements(readiness)
        blocked = bool(missing)
        summary = (
            "ready for promotion"
            if not blocked
            else f"blocked by {len(missing)} missing requirement(s)"
        )
        promotion_plan = {
            "artifact_id": normalized_id,
            "target_state": target_state,
            "readiness_summary": summary,
            "required_confirmation": _promotion_required_confirmation(target_state),
            "validation_required": True,
            "gate_required": True,
            "human_review_required": True,
            "promotion_performed": False,
        }
        planned_files: list[str] = []
        yaml_diff = ""
        review_record_preview: dict[str, Any] = {}
        validation_summary = None
        gate_summary = None
        if not blocked:
            try:
                result = self.app.preview_promote_artifact(
                    normalized_id,
                    target_state=target_state,
                    actor=actor,
                )
            except DraftWriteServiceError as exc:
                blocked = True
                missing = [
                    {
                        "code": exc.code,
                        "severity": "blocking",
                        "message": str(exc),
                    }
                ]
                promotion_plan["readiness_summary"] = "blocked by promotion policy"
            else:
                planned_files = [path.as_posix() for path in result.planned_files]
                yaml_diff = result.yaml_diff
                review_record_preview = result.review_record_preview()
                validation_summary = result.validation_summary
                gate_summary = result.gate_summary
        self._write_audit(
            action=action,
            result_status="preview_blocked" if blocked else "preview",
            explicit_confirm=False,
            preview_only=True,
            planned_files=planned_files,
            authority_warnings=[PROMOTION_AUTHORITY_NOTICE],
        )
        return _preview_response(
            "promotion_preview",
            planned_actions=[f"preview {target_state} promotion readiness"],
            planned_files=planned_files,
            promotion_plan=promotion_plan,
            promotion_readiness=readiness,
            promotion_blocked=blocked,
            missing_requirements=missing,
            yaml_diff=yaml_diff,
            review_record_preview=review_record_preview,
            validation_summary=validation_summary,
            gate_summary=gate_summary,
            accepted_write_performed=False,
            promotion_performed=False,
            authority_warning=PROMOTION_AUTHORITY_NOTICE,
            authority_notice=PROMOTION_AUTHORITY_NOTICE,
        )

    def _confirm_artifact_promotion(
        self,
        artifact_id: str,
        payload: dict[str, Any],
    ) -> ApiResponse:
        action = "promotion_confirm"
        normalized_id = validate_artifact_id(artifact_id)
        target_state = _promotion_target_state(payload.get("target_state", "accepted"))
        blocked = self._blocked_confirm(
            action,
            payload,
            authority_warnings=[PROMOTION_AUTHORITY_NOTICE],
        )
        if blocked is not None:
            return blocked
        actor, actor_blocked = self._require_confirm_actor(
            action,
            authority_warnings=[PROMOTION_AUTHORITY_NOTICE],
        )
        if actor_blocked is not None:
            return actor_blocked
        required = _promotion_required_confirmation(target_state)
        typed = _required_text(payload, "typed_confirmation")
        if typed != required:
            self._write_audit(
                action=action,
                result_status="typed_confirmation_required",
                explicit_confirm=True,
                preview_only=False,
                actor=actor,
                authority_warnings=[PROMOTION_AUTHORITY_NOTICE],
            )
            return _error(
                400,
                "typed_confirmation_required",
                f'typed_confirmation must exactly match "{required}"',
            )
        justification = payload.get("promotion_justification")
        if not isinstance(justification, str) or not justification.strip():
            self._write_audit(
                action=action,
                result_status="promotion_justification_required",
                explicit_confirm=True,
                preview_only=False,
                actor=actor,
                authority_warnings=[PROMOTION_AUTHORITY_NOTICE],
            )
            return _error(
                400,
                "promotion_justification_required",
                "promotion_justification must be a non-empty string",
            )
        promotion_justification = justification.strip()
        try:
            result = self.app.promote_artifact(
                normalized_id,
                target_state=target_state,
                actor=actor,
            )
        except DraftWriteServiceError as exc:
            self._write_audit(
                action=action,
                result_status=exc.code,
                explicit_confirm=True,
                preview_only=False,
                actor=actor,
                authority_warnings=[PROMOTION_AUTHORITY_NOTICE],
            )
            return _error(400, exc.code, str(exc))
        planned_files = [path.as_posix() for path in result.planned_files]
        written_files = [path.as_posix() for path in result.written_paths]
        self._write_audit(
            action=action,
            result_status="success",
            explicit_confirm=True,
            preview_only=False,
            actor=actor,
            planned_files=planned_files,
            written_files=written_files,
            repo_writes_performed=True,
            result={
                "action_performed": True,
                "validation_performed": True,
                "gate_performed": True,
            },
            authority_warnings=[PROMOTION_AUTHORITY_NOTICE],
            operator_notes=promotion_justification,
        )
        return ApiResponse(
            200,
            {
                "schema_version": READONLY_SERVER_SCHEMA_VERSION,
                "kind": "promotion_confirm",
                "action_performed": True,
                "repo_writes_performed": True,
                "git_writes_performed": False,
                "github_writes_performed": False,
                "network_calls_performed": False,
                "audit_logged": True,
                "planned_files": planned_files,
                "written_files": written_files,
                "promotion_plan": {
                    "artifact_id": normalized_id,
                    "target_state": target_state,
                    "readiness_summary": "promotion confirmed",
                    "required_confirmation": required,
                    "validation_required": True,
                    "gate_required": True,
                    "human_review_required": True,
                    "promotion_performed": result.promotion_performed,
                },
                "review_record_preview": result.review_record_preview(),
                "validation_summary": result.validation_summary,
                "gate_summary": result.gate_summary,
                "yaml_diff": result.yaml_diff,
                "artifact": result.updated_artifact.model_dump(mode="json"),
                "old_path": result.old_relative_path.as_posix(),
                "path": result.new_relative_path.as_posix(),
                "accepted_write_performed": result.accepted_write_performed,
                "promotion_performed": result.promotion_performed,
                "promotion_justification_recorded": True,
                "authority_warning": PROMOTION_AUTHORITY_NOTICE,
                "authority_notice": PROMOTION_AUTHORITY_NOTICE,
            },
        )

    def _preview_issue_create(self, payload: dict[str, Any]) -> ApiResponse:
        result = self.app.preview_issue(**_issue_write_args(payload, require_id=True))
        planned_file = result.relative_path.as_posix()
        self._write_audit(
            action="issue_create",
            result_status="preview",
            explicit_confirm=False,
            preview_only=True,
            planned_files=[planned_file],
            authority_warnings=[ISSUE_AUTHORITY_NOTICE],
        )
        return _preview_response(
            "issue_create_preview",
            planned_actions=["create repository-local issue YAML preview"],
            planned_files=[planned_file],
            issue=result.issue.model_dump(mode="json"),
            path=planned_file,
            diff=_diff_text(
                "",
                result.yaml_text(),
                fromfile="/dev/null",
                tofile=planned_file,
            ),
            authority_warning=ISSUE_AUTHORITY_NOTICE,
            authority_notice=ISSUE_AUTHORITY_NOTICE,
        )

    def _preview_artifact_create(self, payload: dict[str, Any]) -> ApiResponse:
        action = "artifact_create"
        try:
            write_args = _artifact_write_args(payload, require_id=True)
            result = self.app.preview_create_artifact(**write_args)
        except DraftWriteServiceError as exc:
            return self._artifact_error(action, exc)
        planned_file = result.relative_path.as_posix()
        yaml_text = result.yaml_text()
        write_plan = _build_artifact_write_plan(
            action=action,
            artifact_id=str(write_args["artifact_id"]),
            write_args=write_args,
            expected_revision=None,
            planned_content_revision=result.artifact.content_revision,
            planned_files=[planned_file],
        )
        self._artifact_write_plans[(action, str(write_args["artifact_id"]))] = (
            write_plan
        )
        self._write_audit(
            action=action,
            result_status="preview",
            explicit_confirm=False,
            preview_only=True,
            planned_files=[planned_file],
            authority_warnings=[ARTIFACT_WRITE_AUTHORITY_NOTICE],
        )
        return _preview_response(
            "artifact_create_preview",
            planned_actions=["create draft artifact YAML preview"],
            planned_files=[planned_file],
            artifact=result.artifact.model_dump(mode="json"),
            path=planned_file,
            yaml=yaml_text,
            write_plan=write_plan,
            diff=_diff_text(
                "",
                yaml_text,
                fromfile="/dev/null",
                tofile=planned_file,
            ),
            authority_warning=ARTIFACT_WRITE_AUTHORITY_NOTICE,
            authority_notice=ARTIFACT_WRITE_AUTHORITY_NOTICE,
        )

    def _create_artifact(self, payload: dict[str, Any]) -> ApiResponse:
        action = "artifact_create"
        blocked = self._blocked_confirm(
            action,
            payload,
            authority_warnings=[ARTIFACT_WRITE_AUTHORITY_NOTICE],
        )
        if blocked is not None:
            return blocked
        try:
            write_args = _artifact_write_args(payload, require_id=True)
        except DraftWriteServiceError as exc:
            return self._artifact_error(action, exc)
        artifact_id = str(write_args["artifact_id"])
        plan_error = self._verify_artifact_write_plan(
            action=action,
            artifact_id=artifact_id,
            write_args=write_args,
            current_revision=None,
            payload=payload,
        )
        if plan_error is not None:
            return plan_error
        try:
            result = self.app.create_artifact(**write_args)
        except DraftWriteServiceError as exc:
            return self._artifact_error(action, exc)
        self._artifact_write_plans.pop((action, artifact_id), None)
        planned_file = result.relative_path.as_posix()
        validation = self.app.validate_artifact_file(result.relative_path)
        self._write_audit(
            action=action,
            result_status="success",
            explicit_confirm=True,
            preview_only=False,
            planned_files=[planned_file],
            repo_writes_performed=True,
            result={
                "action_performed": True,
                "validation_performed": True,
            },
            authority_warnings=[ARTIFACT_WRITE_AUTHORITY_NOTICE],
        )
        return _artifact_action_response(
            "artifact_create",
            result,
            planned_files=[planned_file],
            diff=_diff_text(
                "",
                result.yaml_text(),
                fromfile="/dev/null",
                tofile=planned_file,
            ),
            validation=_validation_payload(validation),
        )

    def _preview_artifact_update(
        self,
        artifact_id: str,
        payload: dict[str, Any],
    ) -> ApiResponse:
        action = "artifact_update"
        try:
            before = _artifact_snapshot(self, artifact_id)
            write_args = _artifact_write_args(payload, require_id=False)
            result = self.app.preview_update_artifact(
                artifact_id,
                **write_args,
            )
        except DraftWriteServiceError as exc:
            return self._artifact_error(action, exc)
        planned_file = result.relative_path.as_posix()
        normalized_id = str(before["artifact"].get("id", artifact_id))
        write_plan = _build_artifact_write_plan(
            action=action,
            artifact_id=normalized_id,
            write_args=write_args,
            expected_revision=_optional_str(
                before["artifact"].get("content_revision")
            ),
            planned_content_revision=result.artifact.content_revision,
            planned_files=[planned_file],
        )
        self._artifact_write_plans[(action, normalized_id)] = write_plan
        self._write_audit(
            action=action,
            result_status="preview",
            explicit_confirm=False,
            preview_only=True,
            planned_files=[planned_file],
            authority_warnings=[ARTIFACT_WRITE_AUTHORITY_NOTICE],
        )
        return _preview_response(
            "artifact_update_preview",
            planned_actions=["update draft artifact YAML preview"],
            planned_files=[planned_file],
            before_artifact=before["artifact"],
            artifact=result.artifact.model_dump(mode="json"),
            path=planned_file,
            yaml=result.yaml_text(),
            write_plan=write_plan,
            diff=_diff_text(
                before["yaml"],
                result.yaml_text(),
                fromfile=planned_file,
                tofile=planned_file,
            ),
            authority_warning=ARTIFACT_WRITE_AUTHORITY_NOTICE,
            authority_notice=ARTIFACT_WRITE_AUTHORITY_NOTICE,
        )

    def _update_artifact(
        self,
        artifact_id: str,
        payload: dict[str, Any],
    ) -> ApiResponse:
        action = "artifact_update"
        blocked = self._blocked_confirm(
            action,
            payload,
            authority_warnings=[ARTIFACT_WRITE_AUTHORITY_NOTICE],
        )
        if blocked is not None:
            return blocked
        try:
            before = _artifact_snapshot(self, artifact_id)
            write_args = _artifact_write_args(payload, require_id=False)
        except DraftWriteServiceError as exc:
            return self._artifact_error(action, exc)
        normalized_id = str(before["artifact"].get("id", artifact_id))
        plan_error = self._verify_artifact_write_plan(
            action=action,
            artifact_id=normalized_id,
            write_args=write_args,
            current_revision=_optional_str(
                before["artifact"].get("content_revision")
            ),
            payload=payload,
        )
        if plan_error is not None:
            return plan_error
        try:
            result = self.app.update_artifact(
                artifact_id,
                **write_args,
            )
        except DraftWriteServiceError as exc:
            return self._artifact_error(action, exc)
        self._artifact_write_plans.pop((action, normalized_id), None)
        planned_file = result.relative_path.as_posix()
        validation = self.app.validate_artifact_file(result.relative_path)
        self._write_audit(
            action=action,
            result_status="success",
            explicit_confirm=True,
            preview_only=False,
            planned_files=[planned_file],
            repo_writes_performed=True,
            result={
                "action_performed": True,
                "validation_performed": True,
            },
            authority_warnings=[ARTIFACT_WRITE_AUTHORITY_NOTICE],
        )
        return _artifact_action_response(
            "artifact_update",
            result,
            planned_files=[planned_file],
            diff=_diff_text(
                before["yaml"],
                result.yaml_text(),
                fromfile=planned_file,
                tofile=planned_file,
            ),
            validation=_validation_payload(validation),
        )

    def _verify_artifact_write_plan(
        self,
        *,
        action: str,
        artifact_id: str,
        write_args: dict[str, Any],
        current_revision: str | None,
        payload: dict[str, Any],
    ) -> ApiResponse | None:
        raw_plan = payload.get("write_plan")
        stored_plan = self._artifact_write_plans.get((action, artifact_id))
        if raw_plan is not None and not isinstance(raw_plan, dict):
            return _error(
                400,
                "invalid_write_plan",
                "write_plan must be a JSON object when provided.",
            )
        candidate_plan = (
            cast(dict[str, Any], raw_plan)
            if isinstance(raw_plan, dict)
            else stored_plan
        )
        if candidate_plan is None:
            return None
        current_request_hash = _artifact_write_request_hash(
            action=action,
            artifact_id=artifact_id,
            write_args=write_args,
        )
        expected_revision = _optional_str(candidate_plan.get("expected_revision"))
        if expected_revision != current_revision:
            self._write_audit(
                action=action,
                result_status="stale_artifact_revision",
                explicit_confirm=True,
                preview_only=False,
                authority_warnings=[ARTIFACT_WRITE_AUTHORITY_NOTICE],
            )
            return _error(
                400,
                "stale_artifact_revision",
                (
                    "Target artifact revision changed since preview; preview "
                    "again before confirming."
                ),
            )
        if _optional_str(candidate_plan.get("request_hash")) != current_request_hash:
            self._write_audit(
                action=action,
                result_status="write_plan_mismatch",
                explicit_confirm=True,
                preview_only=False,
                authority_warnings=[ARTIFACT_WRITE_AUTHORITY_NOTICE],
            )
            return _error(
                400,
                "write_plan_mismatch",
                "Confirmed artifact payload does not match previewed WritePlan.",
            )
        return None

    def _preview_artifact_source(
        self,
        artifact_id: str,
        payload: dict[str, Any],
    ) -> ApiResponse:
        action = "source_attach"
        try:
            before = _artifact_snapshot(self, artifact_id)
            result = self.app.preview_append_source_metadata(artifact_id, payload)
        except DraftWriteServiceError as exc:
            return self._artifact_error(action, exc)
        planned_file = result.relative_path.as_posix()
        self._write_audit(
            action=action,
            result_status="preview",
            explicit_confirm=False,
            preview_only=True,
            planned_files=[planned_file],
            authority_warnings=[ARTIFACT_WRITE_AUTHORITY_NOTICE],
        )
        return _preview_response(
            "artifact_source_preview",
            planned_actions=["append source metadata preview"],
            planned_files=[planned_file],
            artifact=result.artifact.model_dump(mode="json"),
            path=planned_file,
            yaml=result.yaml_text(),
            diff=_diff_text(
                before["yaml"],
                result.yaml_text(),
                fromfile=planned_file,
                tofile=planned_file,
            ),
            warnings=[],
            authority_warning=ARTIFACT_WRITE_AUTHORITY_NOTICE,
            authority_notice=ARTIFACT_WRITE_AUTHORITY_NOTICE,
        )

    def _add_artifact_source(
        self,
        artifact_id: str,
        payload: dict[str, Any],
    ) -> ApiResponse:
        action = "source_attach"
        blocked = self._blocked_confirm(
            action,
            payload,
            authority_warnings=[ARTIFACT_WRITE_AUTHORITY_NOTICE],
        )
        if blocked is not None:
            return blocked
        try:
            before = _artifact_snapshot(self, artifact_id)
            result = self.app.append_source_metadata(artifact_id, payload)
        except DraftWriteServiceError as exc:
            return self._artifact_error(action, exc)
        planned_file = result.relative_path.as_posix()
        validation = self.app.validate_artifact_file(result.relative_path)
        self._write_audit(
            action=action,
            result_status="success",
            explicit_confirm=True,
            preview_only=False,
            planned_files=[planned_file],
            repo_writes_performed=True,
            result={
                "action_performed": True,
                "validation_performed": True,
            },
            authority_warnings=[ARTIFACT_WRITE_AUTHORITY_NOTICE],
        )
        return _artifact_action_response(
            "artifact_source",
            result,
            planned_files=[planned_file],
            diff=_diff_text(
                before["yaml"],
                result.yaml_text(),
                fromfile=planned_file,
                tofile=planned_file,
            ),
            validation=_validation_payload(validation),
        )

    def _preview_artifact_evidence(
        self,
        artifact_id: str,
        payload: dict[str, Any],
    ) -> ApiResponse:
        action = "evidence_attach"
        try:
            before = _artifact_snapshot(self, artifact_id)
            result = self.app.preview_append_evidence_metadata(artifact_id, payload)
        except DraftWriteServiceError as exc:
            return self._artifact_error(action, exc)
        planned_file = result.relative_path.as_posix()
        warnings = _evidence_path_warnings(self, payload)
        self._write_audit(
            action=action,
            result_status="preview",
            explicit_confirm=False,
            preview_only=True,
            planned_files=[planned_file],
            authority_warnings=[ARTIFACT_WRITE_AUTHORITY_NOTICE],
        )
        return _preview_response(
            "artifact_evidence_preview",
            planned_actions=["append evidence metadata preview"],
            planned_files=[planned_file],
            artifact=result.artifact.model_dump(mode="json"),
            path=planned_file,
            yaml=result.yaml_text(),
            diff=_diff_text(
                before["yaml"],
                result.yaml_text(),
                fromfile=planned_file,
                tofile=planned_file,
            ),
            warnings=warnings,
            authority_warning=ARTIFACT_WRITE_AUTHORITY_NOTICE,
            authority_notice=ARTIFACT_WRITE_AUTHORITY_NOTICE,
        )

    def _add_artifact_evidence(
        self,
        artifact_id: str,
        payload: dict[str, Any],
    ) -> ApiResponse:
        action = "evidence_attach"
        blocked = self._blocked_confirm(
            action,
            payload,
            authority_warnings=[ARTIFACT_WRITE_AUTHORITY_NOTICE],
        )
        if blocked is not None:
            return blocked
        try:
            before = _artifact_snapshot(self, artifact_id)
            result = self.app.append_evidence_metadata(artifact_id, payload)
        except DraftWriteServiceError as exc:
            return self._artifact_error(action, exc)
        planned_file = result.relative_path.as_posix()
        validation = self.app.validate_artifact_file(result.relative_path)
        self._write_audit(
            action=action,
            result_status="success",
            explicit_confirm=True,
            preview_only=False,
            planned_files=[planned_file],
            repo_writes_performed=True,
            result={
                "action_performed": True,
                "validation_performed": True,
            },
            authority_warnings=[ARTIFACT_WRITE_AUTHORITY_NOTICE],
        )
        return _artifact_action_response(
            "artifact_evidence",
            result,
            planned_files=[planned_file],
            diff=_diff_text(
                before["yaml"],
                result.yaml_text(),
                fromfile=planned_file,
                tofile=planned_file,
            ),
            validation=_validation_payload(validation),
        )

    def _create_issue(self, payload: dict[str, Any]) -> ApiResponse:
        action = "issue_create"
        blocked = self._blocked_confirm(action, payload)
        if blocked is not None:
            return blocked
        result = self.app.create_issue(**_issue_write_args(payload, require_id=True))
        planned_file = result.relative_path.as_posix()
        self._write_audit(
            action=action,
            result_status="success",
            explicit_confirm=True,
            preview_only=False,
            planned_files=[planned_file],
            repo_writes_performed=True,
            result={"action_performed": True},
            authority_warnings=[ISSUE_AUTHORITY_NOTICE],
        )
        return _issue_action_response(
            "issue_create",
            result,
            planned_files=[planned_file],
            diff=_diff_text(
                "",
                result.yaml_text(),
                fromfile="/dev/null",
                tofile=planned_file,
            ),
        )

    def _preview_issue_update(
        self,
        issue_id: str,
        payload: dict[str, Any],
    ) -> ApiResponse:
        before = self.app.show_issue(issue_id)
        result = self.app.preview_update_issue(
            issue_id,
            **_issue_write_args(payload, require_id=False),
        )
        planned_file = result.relative_path.as_posix()
        self._write_audit(
            action="issue_update",
            result_status="preview",
            explicit_confirm=False,
            preview_only=True,
            planned_files=[planned_file],
            authority_warnings=[ISSUE_AUTHORITY_NOTICE],
        )
        return _preview_response(
            "issue_update_preview",
            planned_actions=["update repository-local issue YAML preview"],
            planned_files=[planned_file],
            before_issue=before.issue.model_dump(mode="json"),
            issue=result.issue.model_dump(mode="json"),
            path=planned_file,
            diff=_diff_text(
                before.yaml_text(),
                result.yaml_text(),
                fromfile=planned_file,
                tofile=planned_file,
            ),
            authority_warning=ISSUE_AUTHORITY_NOTICE,
            authority_notice=ISSUE_AUTHORITY_NOTICE,
        )

    def _update_issue(self, issue_id: str, payload: dict[str, Any]) -> ApiResponse:
        action = "issue_update"
        blocked = self._blocked_confirm(action, payload)
        if blocked is not None:
            return blocked
        before = self.app.show_issue(issue_id)
        result = self.app.update_issue(
            issue_id,
            **_issue_write_args(payload, require_id=False),
        )
        planned_file = result.relative_path.as_posix()
        self._write_audit(
            action=action,
            result_status="success",
            explicit_confirm=True,
            preview_only=False,
            planned_files=[planned_file],
            repo_writes_performed=True,
            result={"action_performed": True},
            authority_warnings=[ISSUE_AUTHORITY_NOTICE],
        )
        return _issue_action_response(
            "issue_update",
            result,
            planned_files=[planned_file],
            diff=_diff_text(
                before.yaml_text(),
                result.yaml_text(),
                fromfile=planned_file,
                tofile=planned_file,
            ),
        )

    def _preview_issue_close(
        self,
        issue_id: str,
        payload: dict[str, Any],
    ) -> ApiResponse:
        before = self.app.show_issue(issue_id)
        result = self.app.preview_close_issue(
            issue_id,
            reason=_required_text(payload, "reason"),
        )
        planned_file = result.relative_path.as_posix()
        self._write_audit(
            action="issue_close",
            result_status="preview",
            explicit_confirm=False,
            preview_only=True,
            planned_files=[planned_file],
            authority_warnings=[ISSUE_AUTHORITY_NOTICE],
        )
        return _preview_response(
            "issue_close_preview",
            planned_actions=["close repository-local issue YAML preview"],
            planned_files=[planned_file],
            before_issue=before.issue.model_dump(mode="json"),
            issue=result.issue.model_dump(mode="json"),
            path=planned_file,
            artifact_status_changed=result.artifact_status_changed,
            diff=_diff_text(
                before.yaml_text(),
                result.yaml_text(),
                fromfile=before.relative_path.as_posix(),
                tofile=planned_file,
            ),
            authority_warning=ISSUE_AUTHORITY_NOTICE,
            authority_notice=ISSUE_AUTHORITY_NOTICE,
        )

    def _close_issue(self, issue_id: str, payload: dict[str, Any]) -> ApiResponse:
        action = "issue_close"
        blocked = self._blocked_confirm(action, payload)
        if blocked is not None:
            return blocked
        before = self.app.show_issue(issue_id)
        result = self.app.close_issue(
            issue_id,
            reason=_required_text(payload, "reason"),
        )
        planned_file = result.relative_path.as_posix()
        self._write_audit(
            action=action,
            result_status="success",
            explicit_confirm=True,
            preview_only=False,
            planned_files=[planned_file],
            repo_writes_performed=True,
            result={"action_performed": True},
            authority_warnings=[ISSUE_AUTHORITY_NOTICE],
        )
        return _issue_action_response(
            "issue_close",
            result,
            planned_files=[planned_file],
            diff=_diff_text(
                before.yaml_text(),
                result.yaml_text(),
                fromfile=before.relative_path.as_posix(),
                tofile=planned_file,
            ),
        )

    def _preview_context_build(
        self,
        issue_id: str,
        payload: dict[str, Any],
    ) -> ApiResponse:
        self.app.show_issue(issue_id)
        planned_files = _context_pack_planned_files(issue_id)
        self._write_audit(
            action="context_build",
            result_status="preview",
            explicit_confirm=False,
            preview_only=True,
            planned_files=planned_files,
            authority_warnings=[CONTEXT_AUTHORITY_NOTICE],
        )
        return _preview_response(
            "context_build_preview",
            planned_actions=["build issue context pack preview"],
            planned_files=planned_files,
            context_pack={
                "issue_id": issue_id,
                "role": _context_role(payload),
                "public_only": _bool(payload, "public_only", default=False),
                "max_cards": _int(payload, "max_cards", default=20, minimum=1),
                "max_full_artifacts": _int(
                    payload,
                    "max_full_artifacts",
                    default=0,
                    minimum=0,
                ),
                "authority_notice": CONTEXT_AUTHORITY_NOTICE,
            },
            authority_warning=CONTEXT_AUTHORITY_NOTICE,
            authority_notice=CONTEXT_AUTHORITY_NOTICE,
        )

    def _build_context(self, issue_id: str, payload: dict[str, Any]) -> ApiResponse:
        action = "context_build"
        blocked = self._blocked_confirm(action, payload)
        if blocked is not None:
            return blocked
        result = self.app.build_context(
            issue_id,
            role=_context_role(payload),
            public_only=_bool(payload, "public_only", default=False),
            max_cards=_int(payload, "max_cards", default=20, minimum=1),
            max_full_artifacts=_int(
                payload,
                "max_full_artifacts",
                default=0,
                minimum=0,
            ),
        )
        planned_files = [
            _repo_relative_or_string(self.app.context.repo_root, path)
            for path in result.files
        ]
        self._write_audit(
            action=action,
            result_status="success",
            explicit_confirm=True,
            preview_only=False,
            planned_files=planned_files,
            repo_writes_performed=True,
            result={"action_performed": True},
            authority_warnings=[CONTEXT_AUTHORITY_NOTICE],
        )
        return ApiResponse(
            200,
            {
                "schema_version": READONLY_SERVER_SCHEMA_VERSION,
                "kind": "context_build",
                "action_performed": True,
                "repo_writes_performed": True,
                "git_writes_performed": False,
                "github_writes_performed": False,
                "network_calls_performed": False,
                "audit_logged": True,
                "planned_files": planned_files,
                "written_files": planned_files,
                "context_pack": _context_pack_result_payload(
                    self.app.context.repo_root,
                    result.issue_id,
                    result.task_dir,
                    result.files,
                ),
                "authority_warning": CONTEXT_AUTHORITY_NOTICE,
                "authority_notice": CONTEXT_AUTHORITY_NOTICE,
            },
        )

    def _run_validate(self) -> ApiResponse:
        validation = self.app.validate_repository()
        self._write_audit(
            action="validate_run",
            result_status="success",
            explicit_confirm=True,
            preview_only=False,
            result={"action_performed": True},
            authority_warnings=[CHECK_AUTHORITY_NOTICE],
        )
        return ApiResponse(
            200,
            {
                "schema_version": READONLY_SERVER_SCHEMA_VERSION,
                "kind": "validate_run",
                "action_performed": True,
                "repo_writes_performed": False,
                "git_writes_performed": False,
                "github_writes_performed": False,
                "network_calls_performed": False,
                "audit_logged": True,
                "validation": _validation_payload(validation),
                "accepted_status_changed": False,
                "authority_warning": CHECK_AUTHORITY_NOTICE,
                "authority_notice": CHECK_AUTHORITY_NOTICE,
            },
        )

    def _run_gate(self) -> ApiResponse:
        result = self.app.run_gate()
        report = result.report.to_dict()
        planned_files = [
            _repo_relative_or_string(self.app.context.repo_root, result.json_path),
            _repo_relative_or_string(self.app.context.repo_root, result.markdown_path),
        ]
        self._write_audit(
            action="gate_run",
            result_status="success",
            explicit_confirm=True,
            preview_only=False,
            planned_files=planned_files,
            repo_writes_performed=True,
            result={"action_performed": True},
            authority_warnings=[CHECK_AUTHORITY_NOTICE],
        )
        return ApiResponse(
            200,
            {
                "schema_version": READONLY_SERVER_SCHEMA_VERSION,
                "kind": "gate_run",
                "action_performed": True,
                "repo_writes_performed": True,
                "git_writes_performed": False,
                "github_writes_performed": False,
                "network_calls_performed": False,
                "audit_logged": True,
                "planned_files": planned_files,
                "written_files": planned_files,
                "gate": _gate_run_payload(report, planned_files),
                "accepted_status_changed": False,
                "authority_warning": CHECK_AUTHORITY_NOTICE,
                "authority_notice": CHECK_AUTHORITY_NOTICE,
            },
        )

    def _create_forge_branch(self, payload: dict[str, Any]) -> ApiResponse:
        action = "branch_create"
        branch = _required_text(payload, "branch")
        allow_dirty = _bool(payload, "allow_dirty", default=False)
        blocked = self._blocked_confirm(action, payload)
        if blocked is not None:
            return blocked
        try:
            result = self.app.forge_branch_create(
                branch,
                confirm=True,
                allow_dirty=allow_dirty,
            )
        except ForgeActionError as exc:
            self._write_action_failure(
                action=action,
                result_status=exc.code,
                planned_files=[],
                head=branch,
            )
            raise
        return self._action_response(
            "branch_create",
            result,
            planned_files=[],
            repo_writes_performed=False,
        )

    def _create_forge_commit(self, payload: dict[str, Any]) -> ApiResponse:
        action = "commit"
        message = _required_text(payload, "message")
        stage_all = _bool(payload, "stage_all", default=False)
        blocked = self._blocked_confirm(action, payload)
        if blocked is not None:
            return blocked
        try:
            result = self.app.forge_commit(
                message=message,
                confirm=True,
                stage_all=stage_all,
            )
        except ForgeActionError as exc:
            self._write_action_failure(
                action=action,
                result_status=exc.code,
                planned_files=[],
            )
            raise
        return self._action_response(
            "commit",
            result,
            planned_files=[],
            repo_writes_performed=False,
        )

    def _create_forge_push(self, payload: dict[str, Any]) -> ApiResponse:
        action = "push"
        head = _required_text(payload, "head")
        blocked = self._blocked_confirm(action, payload)
        if blocked is not None:
            return blocked
        try:
            result = self.app.forge_push(
                branch=head,
                remote=_optional_text(payload, "remote") or "origin",
                confirm=True,
            )
        except ForgeActionError as exc:
            self._write_action_failure(
                action=action,
                result_status=exc.code,
                planned_files=[],
                head=head,
            )
            raise
        return self._action_response(
            "push",
            result,
            planned_files=[],
            repo_writes_performed=False,
        )

    def _create_github_issue(self, payload: dict[str, Any]) -> ApiResponse:
        action = "github_issue_create"
        source_path = _required_text(payload, "source_path")
        blocked = self._blocked_create(action, payload)
        if blocked is not None:
            return blocked
        try:
            result = self.app.forge_github_issue_create(
                source_path,
                confirm=True,
            )
        except ForgeActionError as exc:
            self._write_action_failure(
                action=action,
                result_status=exc.code,
                planned_files=[source_path],
            )
            raise
        return self._action_response(
            "github_issue_create",
            result,
            planned_files=[source_path],
            repo_writes_performed=True,
        )

    def _create_github_pr(self, payload: dict[str, Any]) -> ApiResponse:
        action = "github_pr_create"
        base = _required_text(payload, "base")
        head = _required_text(payload, "head")
        blocked = self._blocked_create(action, payload)
        if blocked is not None:
            return blocked
        try:
            result = self.app.forge_github_pr_create(
                base=base,
                head=head,
                draft=_bool(payload, "draft", default=False),
                confirm=True,
            )
        except ForgeActionError as exc:
            self._write_action_failure(
                action=action,
                result_status=exc.code,
                planned_files=[],
                base=base,
                head=head,
            )
            raise
        return self._action_response(
            "github_pr_create",
            result,
            planned_files=[],
            repo_writes_performed=False,
        )

    def _submit_github_pr(self, payload: dict[str, Any]) -> ApiResponse:
        action = "github_pr_submit"
        base = _required_text(payload, "base")
        head = _required_text(payload, "head")
        blocked = self._blocked_create(action, payload)
        if blocked is not None:
            return blocked
        try:
            result = self.app.forge_github_pr_submit(
                base=base,
                head=head,
                draft=_bool(payload, "draft", default=True),
                remote=_optional_text(payload, "remote") or "origin",
                confirm=True,
            )
        except ForgeActionError as exc:
            self._write_action_failure(
                action=action,
                result_status=exc.code,
                planned_files=[],
                base=base,
                head=head,
            )
            raise
        return self._action_response(
            "github_pr_submit",
            result,
            planned_files=[],
            repo_writes_performed=False,
        )

    def _current_hosted_identity(self) -> HostedIdentity | None:
        if self.hosted_auth_provider is None:
            return None
        return self.hosted_auth_provider.current_identity()

    def _preview_actor(self, payload: dict[str, Any]) -> str:
        if self.web_action_mode is WebActionMode.HOSTED:
            identity = self._current_hosted_identity()
            if identity is not None:
                return identity.subject
            return "hosted.anonymous"
        return self.local_actor or _optional_text(payload, "actor") or "local.web"

    def _blocked_hosted_action(
        self,
        action: WebActionKind,
        payload: dict[str, Any],
        *,
        preview_only: bool,
    ) -> ApiResponse | None:
        if self.web_action_mode is not WebActionMode.HOSTED:
            return None
        identity = self._current_hosted_identity()
        authority_warnings = _authority_warnings_for_action(action)
        explicit_confirm = payload.get("confirm") is True
        if identity is None:
            self._write_audit(
                action=action,
                result_status="hosted_auth_required",
                explicit_confirm=explicit_confirm,
                preview_only=preview_only,
                actor="hosted.anonymous",
                authority_warnings=authority_warnings,
            )
            return _error(
                403,
                "hosted_auth_required",
                "hosted mode requires an authenticated hosted identity",
            )
        if not hosted_action_allowed(identity, action):
            required = hosted_required_role(action)
            self._write_audit(
                action=action,
                result_status="hosted_action_denied",
                explicit_confirm=explicit_confirm,
                preview_only=preview_only,
                actor=identity.subject,
                authority_warnings=authority_warnings,
            )
            return _error(
                403,
                "hosted_action_denied",
                f"hosted action requires role: {required}",
            )
        return None

    def _blocked_confirm(
        self,
        action: str | WebActionKind,
        payload: dict[str, Any],
        *,
        authority_warnings: list[str] | None = None,
    ) -> ApiResponse | None:
        explicit_confirm = payload.get("confirm") is True
        if not explicit_confirm:
            self._write_audit(
                action=action,
                result_status="confirm_required",
                explicit_confirm=False,
                preview_only=False,
                authority_warnings=authority_warnings or [ISSUE_AUTHORITY_NOTICE],
            )
            return _error(400, "confirm_required", "write requires confirm: true")
        return None

    def _require_confirm_actor(
        self,
        action: str,
        *,
        authority_warnings: list[str] | None = None,
    ) -> tuple[str, ApiResponse | None]:
        if self.web_action_mode is WebActionMode.HOSTED:
            identity = self._current_hosted_identity()
            if identity is not None:
                return identity.subject, None
            self._write_audit(
                action=action,
                result_status="hosted_auth_required",
                explicit_confirm=True,
                preview_only=False,
                actor="hosted.anonymous",
                authority_warnings=authority_warnings or [SITE_EXPORT_AUTHORITY_NOTICE],
            )
            return (
                "",
                _error(
                    403,
                    "hosted_auth_required",
                    "hosted mode requires an authenticated hosted identity",
                ),
            )
        return self._require_local_actor(
            action,
            authority_warnings=authority_warnings,
        )

    def _require_local_actor(
        self,
        action: str,
        *,
        authority_warnings: list[str] | None = None,
    ) -> tuple[str, ApiResponse | None]:
        actor = self.local_actor
        if actor is not None:
            return actor, None
        self._write_audit(
            action=action,
            result_status="local_actor_required",
            explicit_confirm=True,
            preview_only=False,
            authority_warnings=authority_warnings or [LOCAL_ACTOR_NOTICE],
        )
        return (
            "",
            _error(
                400,
                "local_actor_required",
                "local mode requires server option --local-actor <name> before "
                "confirming human review or promotion actions",
            ),
        )

    def _artifact_error(
        self,
        action: str,
        exc: DraftWriteServiceError,
    ) -> ApiResponse:
        self._write_audit(
            action=action,
            result_status=exc.code,
            explicit_confirm=False,
            preview_only=False,
            authority_warnings=[ARTIFACT_WRITE_AUTHORITY_NOTICE],
        )
        return _error(400, exc.code, str(exc))

    def _blocked_create(
        self,
        action: str,
        payload: dict[str, Any],
    ) -> ApiResponse | None:
        explicit_confirm = payload.get("confirm") is True
        if not explicit_confirm:
            self._write_audit(
                action=action,
                result_status="confirm_required",
                explicit_confirm=False,
                preview_only=False,
            )
            return _error(400, "confirm_required", "create requires confirm: true")
        provider = self.credential_provider
        if provider is None or not provider.has_github_token():
            self._write_audit(
                action=action,
                result_status="auth_required",
                explicit_confirm=True,
                preview_only=False,
                credential_provider=self._credential_provider_name(),
            )
            return _error(
                401,
                "auth_required",
                "authenticated forge create requires backend GitHub credentials",
            )
        return None

    def _action_response(
        self,
        kind: str,
        result: ForgeActionResult,
        *,
        planned_files: list[str],
        repo_writes_performed: bool,
    ) -> ApiResponse:
        payload = result.model_dump(mode="json")
        self._write_audit(
            action=result.action,
            result_status="success",
            explicit_confirm=True,
            preview_only=False,
            credential_provider=self._credential_provider_name(),
            planned_files=planned_files,
            repo_writes_performed=repo_writes_performed,
            result=payload,
        )
        return ApiResponse(
            200,
            {
                "schema_version": READONLY_SERVER_SCHEMA_VERSION,
                "kind": kind,
                "action_performed": result.action_performed,
                "repo_writes_performed": repo_writes_performed,
                "git_writes_performed": result.git_writes_performed,
                "github_writes_performed": result.github_writes_performed,
                "network_calls_performed": result.network_calls_performed,
                "audit_logged": True,
                "forge_action": payload,
                "authority_warning": FORGE_AUTHORITY_WARNING,
                "authority_notice": SITE_EXPORT_AUTHORITY_NOTICE,
            },
        )

    def _write_action_failure(
        self,
        *,
        action: str,
        result_status: str,
        planned_files: list[str],
        base: str | None = None,
        head: str | None = None,
    ) -> None:
        self._write_audit(
            action=action,
            result_status=result_status,
            explicit_confirm=True,
            preview_only=False,
            credential_provider=self._credential_provider_name(),
            planned_files=planned_files,
            base=base,
            head=head,
        )

    def _credential_provider_name(self) -> str | None:
        return (
            self.credential_provider.provider_name()
            if self.credential_provider
            else None
        )

    def _write_audit(
        self,
        *,
        action: str,
        result_status: str,
        explicit_confirm: bool,
        preview_only: bool,
        credential_provider: str | None = None,
        planned_files: list[str] | None = None,
        written_files: list[str] | None = None,
        repo_writes_performed: bool = False,
        base: str | None = None,
        head: str | None = None,
        branch: str | None = None,
        result: dict[str, Any] | None = None,
        authority_warnings: list[str] | None = None,
        operator_notes: str | None = None,
        actor: str | None = None,
    ) -> None:
        result = result or {}
        audit_actor = (
            actor
            or self.local_actor
            or (
                "hosted.anonymous"
                if self.web_action_mode is WebActionMode.HOSTED
                else "local.web"
            )
        )
        github_url = result.get("github_issue_url") or result.get("github_pr_url")
        performed = result_status == "success" and bool(
            result.get("action_performed", False)
        )
        error_code = None if result_status in {"preview", "success"} else result_status
        errors = []
        if error_code is not None:
            errors.append(
                WebActionError(
                    code=error_code,
                    message="web action did not complete",
                    remediation="Inspect the API response and backend credentials.",
                    blocking=True,
                )
            )
        append_web_action_audit(
            self.app.context,
            WebActionAuditEntry(
                timestamp=datetime.now(UTC).replace(microsecond=0),
                actor=audit_actor,
                action=action
                if isinstance(action, WebActionKind)
                else _web_action_kind(action),
                mode=self.web_action_mode,
                repo_root=str(self.app.context.repo_root),
                branch=branch or _optional_str(result.get("branch")),
                base=base or _optional_str(result.get("base")),
                head=head or _optional_str(result.get("head")),
                commit_hash=_optional_str(result.get("commit_hash")),
                preview_only=preview_only,
                confirm_required=result_status == "confirm_required",
                confirmed=explicit_confirm,
                explicit_confirm=explicit_confirm,
                performed=performed,
                repo_writes_performed=repo_writes_performed,
                git_writes_performed=bool(result.get("git_writes_performed", False)),
                github_writes_performed=bool(
                    result.get("github_writes_performed", False)
                ),
                network_calls_performed=bool(
                    result.get("network_calls_performed", False)
                ),
                planned_files=planned_files or [],
                written_files=(
                    written_files
                    if written_files is not None
                    else (planned_files or []) if repo_writes_performed else []
                ),
                validation_summary=_performed_summary(result, "validation_performed"),
                gate_summary=_performed_summary(result, "gate_performed"),
                github_urls=[github_url] if isinstance(github_url, str) else [],
                credential_provider=credential_provider,
                result_status=result_status,
                authority_warnings=authority_warnings or [FORGE_AUTHORITY_WARNING],
                operator_notes=operator_notes,
                error_code=error_code,
                errors=errors,
            ),
        )

    def _payload_file(self, filename: str) -> dict[str, Any]:
        payloads = self._site_payloads()
        return payloads[filename]

    def _site_payloads(self) -> dict[str, dict[str, Any]]:
        with TemporaryDirectory(prefix="cosheaf-site-api-") as temp_dir:
            out = Path(temp_dir)
            self.app.export_site_data(out)
            return {
                filename: json.loads((out / filename).read_text(encoding="utf-8"))
                for filename in REQUIRED_SITE_EXPORT_FILES
            }


def make_handler(api: ReadOnlySiteApi) -> type[BaseHTTPRequestHandler]:
    """Create an HTTP handler class bound to a read-only API router."""

    class ReadOnlySiteRequestHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            self._send(api.handle("GET", self.path))

        def do_POST(self) -> None:  # noqa: N802
            self._send(api.handle("POST", self.path, self._read_body()))

        def do_OPTIONS(self) -> None:  # noqa: N802
            self.send_response(204)
            self._send_cors_headers()
            self.send_header("Content-Length", "0")
            self.end_headers()

        def do_PUT(self) -> None:  # noqa: N802
            self._send(api.handle("PUT", self.path))

        def do_PATCH(self) -> None:  # noqa: N802
            self._send(api.handle("PATCH", self.path))

        def do_DELETE(self) -> None:  # noqa: N802
            self._send(api.handle("DELETE", self.path))

        def log_message(self, format: str, *args: object) -> None:
            return

        def _send(self, response: ApiResponse) -> None:
            body = response.body()
            self.send_response(response.status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self._send_cors_headers()
            self.end_headers()
            self.wfile.write(body)

        def _read_body(self) -> bytes:
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                length = 0
            return self.rfile.read(length)

        def _send_cors_headers(self) -> None:
            origin = self.headers.get("Origin")
            if origin is not None and _allowed_cors_origin(origin):
                self.send_header("Access-Control-Allow-Origin", origin)
                self.send_header("Vary", "Origin")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")

    return ReadOnlySiteRequestHandler


def serve_readonly_api(
    app: CosheafApp,
    *,
    host: str = READONLY_SERVER_HOST,
    port: int = READONLY_SERVER_PORT,
    local_actor: str | None = None,
    strict_readonly: bool = False,
) -> None:
    """Serve the read-only local website API until interrupted."""
    api = ReadOnlySiteApi(
        app,
        host=host,
        local_actor=local_actor,
        strict_readonly=strict_readonly,
    )
    server = HTTPServer((host, port), make_handler(api))
    try:
        server.serve_forever()
    finally:
        server.server_close()


def _normalize_path(raw_path: str) -> str:
    path = unquote(urlparse(raw_path).path)
    if path != "/":
        path = path.rstrip("/")
    return path or "/"


def _query_params(raw_path: str) -> dict[str, str]:
    parsed = parse_qs(urlparse(raw_path).query, keep_blank_values=False)
    return {
        key: values[-1]
        for key, values in parsed.items()
        if values and isinstance(values[-1], str)
    }


def _post_route_action(
    path: str,
    *,
    issue_action: tuple[str, str] | None,
    artifact_action: tuple[str, str] | None,
    artifact_promotion: tuple[str, str] | None,
    context_action: tuple[str, str] | None,
) -> WebActionKind:
    direct = {
        "/api/artifacts/preview-create": WebActionKind.ARTIFACT_CREATE,
        "/api/artifacts/create": WebActionKind.ARTIFACT_CREATE,
        "/api/issues/preview-create": WebActionKind.ISSUE_CREATE,
        "/api/issues/create": WebActionKind.ISSUE_CREATE,
        "/api/validate/run": WebActionKind.VALIDATE_RUN,
        "/api/gate/run": WebActionKind.GATE_RUN,
        "/api/reviews/packets/preview": WebActionKind.REVIEW_PACKET_CREATE,
        "/api/reviews/packets/create": WebActionKind.REVIEW_PACKET_CREATE,
        "/api/reviews/decisions/preview": WebActionKind.REVIEW_DECISION_CREATE,
        "/api/reviews/decisions/create": WebActionKind.REVIEW_DECISION_CREATE,
        "/api/forge/local-issues/preview": WebActionKind.ISSUE_CREATE,
        "/api/forge/issues/preview": WebActionKind.ISSUE_PUBLISH_GITHUB,
        "/api/forge/issues/create": WebActionKind.ISSUE_PUBLISH_GITHUB,
        "/api/forge/branch/preview": WebActionKind.FORGE_BRANCH_CREATE,
        "/api/forge/branch/create": WebActionKind.FORGE_BRANCH_CREATE,
        "/api/forge/commit/preview": WebActionKind.FORGE_COMMIT_CREATE,
        "/api/forge/commit/create": WebActionKind.FORGE_COMMIT_CREATE,
        "/api/forge/push/preview": WebActionKind.FORGE_PUSH_CREATE,
        "/api/forge/push/create": WebActionKind.FORGE_PUSH_CREATE,
        "/api/forge/pr/preview": WebActionKind.FORGE_PR_CREATE,
        "/api/forge/pr/create": WebActionKind.FORGE_PR_CREATE,
        "/api/forge/prs/preview": WebActionKind.FORGE_PR_CREATE,
        "/api/forge/prs/create": WebActionKind.FORGE_PR_CREATE,
        "/api/forge/review-packets/preview": WebActionKind.REVIEW_PACKET_CREATE,
    }
    if path in direct:
        return direct[path]
    if issue_action is not None:
        _, action = issue_action
        return {
            "preview-update": WebActionKind.ISSUE_UPDATE,
            "update": WebActionKind.ISSUE_UPDATE,
            "preview-close": WebActionKind.ISSUE_CLOSE,
            "close": WebActionKind.ISSUE_CLOSE,
        }[action]
    if artifact_action is not None:
        _, action = artifact_action
        return {
            "preview-source": WebActionKind.SOURCE_ATTACH,
            "source": WebActionKind.SOURCE_ATTACH,
            "preview-evidence": WebActionKind.EVIDENCE_ATTACH,
            "evidence": WebActionKind.EVIDENCE_ATTACH,
            "preview-update": WebActionKind.ARTIFACT_UPDATE,
            "update": WebActionKind.ARTIFACT_UPDATE,
        }[action]
    if artifact_promotion is not None:
        _, action = artifact_promotion
        return {
            "preview": WebActionKind.PROMOTION_PREVIEW,
            "confirm": WebActionKind.PROMOTION_CONFIRM,
        }[action]
    if context_action is not None:
        return WebActionKind.CONTEXT_BUILD
    raise ValueError(f"unmapped POST endpoint: {path}")


def _post_route_is_preview(
    path: str,
    artifact_promotion: tuple[str, str] | None,
) -> bool:
    if path in _PREVIEW_ENDPOINTS or path in _ARTIFACT_PREVIEW_ENDPOINTS:
        return True
    if "/preview-" in path or path.endswith("/preview"):
        return True
    return artifact_promotion is not None and artifact_promotion[1] == "preview"


def _authority_warnings_for_action(action: WebActionKind) -> list[str]:
    if action in {
        WebActionKind.ISSUE_CREATE,
        WebActionKind.ISSUE_UPDATE,
        WebActionKind.ISSUE_CLOSE,
        WebActionKind.ISSUE_PUBLISH_GITHUB,
    }:
        return [ISSUE_AUTHORITY_NOTICE]
    if action in {
        WebActionKind.ARTIFACT_CREATE,
        WebActionKind.ARTIFACT_UPDATE,
        WebActionKind.SOURCE_ATTACH,
        WebActionKind.EVIDENCE_ATTACH,
    }:
        return [ARTIFACT_WRITE_AUTHORITY_NOTICE]
    if action is WebActionKind.CONTEXT_BUILD:
        return [CONTEXT_AUTHORITY_NOTICE]
    if action in {WebActionKind.VALIDATE_RUN, WebActionKind.GATE_RUN}:
        return [CHECK_AUTHORITY_NOTICE]
    if action is WebActionKind.REVIEW_PACKET_CREATE:
        return [REVIEW_PACKET_AUTHORITY_NOTICE]
    if action is WebActionKind.REVIEW_DECISION_CREATE:
        return [REVIEW_DECISION_AUTHORITY_NOTICE]
    if action in {WebActionKind.PROMOTION_PREVIEW, WebActionKind.PROMOTION_CONFIRM}:
        return [PROMOTION_AUTHORITY_NOTICE]
    if action in {
        WebActionKind.FORGE_BRANCH_CREATE,
        WebActionKind.FORGE_COMMIT_CREATE,
        WebActionKind.FORGE_PUSH_CREATE,
        WebActionKind.FORGE_PR_CREATE,
    }:
        return [FORGE_AUTHORITY_WARNING]
    return [SITE_EXPORT_AUTHORITY_NOTICE]


def _parse_issue_action_path(path: str) -> tuple[str, str] | None:
    if not path.startswith("/api/issues/"):
        return None
    parts = path.removeprefix("/api/issues/").split("/")
    if len(parts) != 2:
        return None
    issue_id, action = parts
    if not issue_id or action not in _ISSUE_ACTION_SUFFIXES:
        return None
    return issue_id, action


def _parse_artifact_action_path(path: str) -> tuple[str, str] | None:
    if not path.startswith("/api/artifacts/"):
        return None
    parts = path.removeprefix("/api/artifacts/").split("/")
    if len(parts) != 2:
        return None
    artifact_id, action = parts
    if not artifact_id or action not in _ARTIFACT_ACTION_SUFFIXES:
        return None
    return artifact_id, action


def _parse_artifact_promotion_path(path: str) -> tuple[str, str] | None:
    if not path.startswith("/api/artifacts/"):
        return None
    parts = path.removeprefix("/api/artifacts/").split("/")
    if len(parts) == 2:
        artifact_id, action = parts
        if artifact_id and action == "promotion-readiness":
            return artifact_id, "readiness"
    if len(parts) == 3:
        artifact_id, group, action = parts
        if artifact_id and group == "promotion" and action in {"preview", "confirm"}:
            return artifact_id, action
    return None


def _parse_context_action_path(path: str) -> tuple[str, str] | None:
    if not path.startswith("/api/context/"):
        return None
    parts = path.removeprefix("/api/context/").split("/")
    if len(parts) != 2:
        return None
    issue_id, action = parts
    if not issue_id or action not in {"preview-build", "build"}:
        return None
    return issue_id, action


def _allowed_cors_origin(origin: str | None) -> bool:
    if not origin:
        return False
    parsed = urlparse(origin)
    return parsed.scheme in {"http", "https"} and parsed.hostname in {
        "localhost",
        "127.0.0.1",
        "::1",
    }


def _json_body(body: str | bytes) -> dict[str, Any] | ApiResponse:
    try:
        text = body.decode("utf-8") if isinstance(body, bytes) else body
        payload = json.loads(text or "{}")
    except json.JSONDecodeError as exc:
        return _error(400, "invalid_json", str(exc))
    if not isinstance(payload, dict):
        return _error(400, "invalid_json", "Expected a JSON object request body.")
    return cast(dict[str, Any], payload)


def _required_text(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return value.strip()


def _optional_text(payload: dict[str, Any], key: str) -> str | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string when provided")
    return value.strip()


def _bool(payload: dict[str, Any], key: str, *, default: bool) -> bool:
    value = payload.get(key, default)
    if not isinstance(value, bool):
        raise ValueError(f"{key} must be a boolean")
    return value


def _int(
    payload: dict[str, Any],
    key: str,
    *,
    default: int,
    minimum: int,
) -> int:
    value = payload.get(key, default)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{key} must be an integer")
    if value < minimum:
        raise ValueError(f"{key} must be at least {minimum}")
    return value


def _text_list(payload: dict[str, Any], key: str) -> list[str]:
    value = payload.get(key, [])
    if not isinstance(value, list):
        raise ValueError(f"{key} must be a list of strings")
    result: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"{key} must be a list of non-empty strings")
        result.append(item.strip())
    return result


def _scope(value: object) -> Literal["private", "public"]:
    if not isinstance(value, str):
        raise ValueError("scope must be private or public")
    normalized = value.strip()
    if normalized not in {"private", "public"}:
        raise ValueError("scope must be private or public")
    return cast(Literal["private", "public"], normalized)


def _promotion_target_state(value: object) -> str:
    target_state = str(value).strip()
    if target_state not in {"accepted", "refuted", "obsolete"}:
        raise ValueError("target_state must be accepted, refuted, or obsolete")
    return target_state


def _promotion_required_confirmation(target_state: str) -> str:
    return PROMOTION_CONFIRMATION_PHRASES[target_state]


def _promotion_missing_requirements(readiness: dict[str, Any]) -> list[dict[str, Any]]:
    reasons: list[dict[str, Any]] = []
    for reason in readiness.get("reasons", []):
        if isinstance(reason, dict) and reason.get("severity") == "blocking":
            reasons.append(reason)
    for artifact in readiness.get("artifacts", []):
        if not isinstance(artifact, dict):
            continue
        for reason in artifact.get("reasons", []):
            if isinstance(reason, dict) and reason.get("severity") == "blocking":
                reasons.append(reason)
    return reasons


def _issue_write_args(
    payload: dict[str, Any],
    *,
    require_id: bool,
) -> dict[str, Any]:
    args: dict[str, Any] = {
        "title": _required_text(payload, "title"),
        "summary": _optional_text(payload, "summary"),
        "authors": _text_list(payload, "authors"),
        "labels": _text_list(payload, "labels"),
        "related_artifacts": _text_list(payload, "related_artifacts"),
        "related_sources": _text_list(payload, "related_sources"),
        "scope": _scope(payload.get("scope", "private")),
    }
    if require_id:
        args["issue_id"] = _required_text(payload, "issue_id")
    return args


def _artifact_write_args(
    payload: dict[str, Any],
    *,
    require_id: bool,
) -> dict[str, Any]:
    try:
        artifact_type = ArtifactType(_required_text(payload, "artifact_type"))
        status = ArtifactStatus(_required_text(payload, "status"))
    except ValueError as exc:
        raise DraftWriteServiceError(
            str(exc),
            code="artifact_model_validation_failed",
            remediation="Use a supported artifact type and lifecycle status.",
        ) from exc
    if not is_preaccepted_status(status):
        raise DraftWriteServiceError(
            "web artifact writes cannot target accepted or terminal statuses",
            code="accepted_write_forbidden"
            if status is ArtifactStatus.ACCEPTED
            else "terminal_status_forbidden",
            remediation=(
                "Use draft/pre-accepted statuses only. Accepted, refuted, "
                "obsolete, and superseded states require lifecycle workflows."
            ),
        )
    args: dict[str, Any] = {
        "artifact_type": artifact_type,
        "title": _required_text(payload, "title"),
        "domain": _text_list(payload, "domain"),
        "status": status,
        "statement": _required_text(payload, "statement"),
        "authors": _text_list(payload, "authors"),
        "tags": _text_list(payload, "tags"),
        "depends_on": _text_list(payload, "depends_on"),
        "supersedes": _text_list(payload, "supersedes"),
    }
    if require_id:
        args["artifact_id"] = _required_text(payload, "artifact_id")
    return args


def _artifact_write_request_hash(
    *,
    action: str,
    artifact_id: str,
    write_args: dict[str, Any],
) -> str:
    canonical = {
        "action": action,
        "artifact_id": artifact_id,
        "artifact_type": str(write_args["artifact_type"]),
        "title": str(write_args["title"]),
        "domain": list(write_args["domain"]),
        "status": str(write_args["status"]),
        "statement": str(write_args["statement"]),
        "authors": list(write_args["authors"]),
        "tags": list(write_args["tags"]),
        "depends_on": list(write_args["depends_on"]),
        "supersedes": list(write_args["supersedes"]),
    }
    serialized = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
    return f"sha256:{hashlib.sha256(serialized.encode('utf-8')).hexdigest()}"


def _build_artifact_write_plan(
    *,
    action: str,
    artifact_id: str,
    write_args: dict[str, Any],
    expected_revision: str | None,
    planned_content_revision: str | None,
    planned_files: list[str],
) -> dict[str, Any]:
    request_hash = _artifact_write_request_hash(
        action=action,
        artifact_id=artifact_id,
        write_args=write_args,
    )
    plan_seed = json.dumps(
        {
            "action": action,
            "artifact_id": artifact_id,
            "request_hash": request_hash,
            "expected_revision": expected_revision,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    plan_digest = hashlib.sha256(plan_seed.encode("utf-8")).hexdigest()[:16]
    return {
        "plan_id": f"write-plan.{action}.{plan_digest}",
        "action": action,
        "artifact_id": artifact_id,
        "request_hash": request_hash,
        "expected_revision": expected_revision,
        "planned_content_revision": planned_content_revision,
        "planned_files": list(planned_files),
    }


def _issue_context_pack_summary(repo_root: Path, issue_id: str) -> dict[str, Any]:
    task_dir = repo_root / "context" / "TASKS" / issue_id
    audit = _read_json_file(task_dir / "RETRIEVAL_AUDIT.json")
    audit_dict = audit if isinstance(audit, dict) else {}
    raw_card_count = audit_dict.get("card_count", 0)
    card_count = (
        raw_card_count
        if isinstance(raw_card_count, int) and not isinstance(raw_card_count, bool)
        else 0
    )
    return {
        "issue_id": issue_id,
        "exists": task_dir.is_dir(),
        "card_count": card_count if task_dir.is_dir() else 0,
        "public_only": bool(audit_dict.get("public_only", False)),
        "private_context_included": bool(
            audit_dict.get("private_context_included", False)
        ),
    }


def _review_decision_planned_files(result: Any) -> list[str]:
    planned = [result.relative_path.as_posix()]
    if result.artifact_updated:
        planned.append(result.artifact_relative_path.as_posix())
    return planned


def _build_review_packet(
    api: ReadOnlySiteApi,
    payload: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    issue_id = _required_text(payload, "issue_id")
    issue = api.app.show_issue(issue_id).issue
    artifact_id = _optional_text(payload, "artifact_id")
    artifacts = _review_packet_artifacts(api, issue.related_artifacts, artifact_id)
    target = artifact_id or issue_id
    sections = _review_packet_sections(api, artifacts)
    review_id = _review_packet_id(issue_id, artifact_id)
    authority_checklist = sections["authority_checklist"]
    findings = [
        f"Gate state: {sections['gate_state']}",
        *[
            f"Reviewer question: {question}"
            for question in sections["reviewer_questions"]
        ],
        *[f"Authority: {item}" for item in authority_checklist],
    ]
    request = {
        "review_id": review_id,
        "title": f"Review packet: {issue.title}",
        "status": "draft",
        "authors": [],
        "target": target,
        "summary": f"Review packet for {issue.id}: {issue.summary}",
        "findings": findings,
        "decision": "informational",
    }
    packet = {
        "review_id": review_id,
        "title": request["title"],
        "target": target,
        "issue_id": issue.id,
        "artifact_ids": [artifact.id for artifact in artifacts],
        "summary": request["summary"],
        "findings": findings,
        "decision": "informational",
        "status": "draft",
        "sections": sections,
        "authority_notice": REVIEW_PACKET_AUTHORITY_NOTICE,
    }
    return {"request": request, "packet": packet}


def _review_packet_id(issue_id: str, artifact_id: str | None) -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%d%H%M%S%f")
    parts = ["review", "packet", *issue_id.split(".")]
    if artifact_id:
        parts.extend(artifact_id.split("."))
    parts.append(timestamp)
    return ".".join(parts)


def _review_packet_artifacts(
    api: ReadOnlySiteApi,
    related_artifact_ids: list[str],
    artifact_id: str | None,
) -> list[BaseArtifact]:
    requested_ids = [artifact_id] if artifact_id else related_artifact_ids
    records = tuple(load_artifacts(api.app.context))
    by_id = {
        record.id: record.record
        for record in records
        if isinstance(record.record, BaseArtifact)
    }
    artifacts: list[BaseArtifact] = []
    for requested_id in requested_ids:
        if not requested_id:
            continue
        artifact = by_id.get(requested_id)
        if artifact is None and artifact_id:
            raise ValueError(f"review packet artifact not found: {requested_id}")
        if artifact is not None:
            artifacts.append(artifact)
    return artifacts


def _review_packet_sections(
    api: ReadOnlySiteApi,
    artifacts: list[BaseArtifact],
) -> dict[str, Any]:
    focus = artifacts[0] if len(artifacts) == 1 else None
    dependencies = sorted(
        {
            dependency
            for artifact in artifacts
            for dependency in artifact.depends_on
        }
    )
    return {
        "artifact_statement": focus.statement if focus else "",
        "dependencies": dependencies,
        "sources": [
            _source_label(artifact, source)
            for artifact in artifacts
            for source in artifact.sources
        ],
        "evidence": [
            _evidence_label(artifact, evidence)
            for artifact in artifacts
            for evidence in artifact.evidence
        ],
        "known_failures": [
            _failure_label(artifact, failure)
            for artifact in artifacts
            for failure in artifact.failure_log
        ],
        "gate_state": _latest_gate_state(api),
        "reviewer_questions": [
            "Have dependencies been checked against accepted/draft boundaries?",
            "Are sources sufficient for the intended review scope?",
            "Does evidence support review discussion without claiming verifier pass?",
        ],
        "authority_checklist": [
            "This packet is not human review.",
            "This packet is not proof.",
            "This packet is not gate pass or verifier pass.",
            "This packet does not grant accepted status or promotion authority.",
        ],
    }


def _source_label(artifact: BaseArtifact, source: Any) -> str:
    details = [source.kind]
    if source.year is not None:
        details.append(str(source.year))
    if source.theorem_number:
        details.append(source.theorem_number)
    if source.page:
        details.append(f"p. {source.page}")
    suffix = f" ({', '.join(details)})" if details else ""
    title = source.title or "Untitled source"
    return f"{artifact.id}: {title}{suffix}"


def _evidence_label(artifact: BaseArtifact, evidence: Any) -> str:
    return f"{artifact.id}: {evidence.kind} {evidence.path} - {evidence.summary}"


def _failure_label(artifact: BaseArtifact, failure: Any) -> str:
    return (
        f"{artifact.id}: {failure.summary} - {failure.failed_because} "
        f"({failure.status})"
    )


def _latest_gate_state(api: ReadOnlySiteApi) -> str:
    reports_dir = api.app.context.resolve(Path(".cosheaf") / "reports")
    candidates = sorted(reports_dir.glob("*-gate-report.json"))
    if not candidates:
        return "not_run"
    report = _read_json_file(candidates[-1])
    if not isinstance(report, dict):
        return "unreadable"
    verdict = report.get("verdict")
    return str(verdict) if verdict is not None else "unknown"


def _context_role(payload: dict[str, Any]) -> str:
    return _optional_text(payload, "role") or "orchestrator"


def _context_pack_planned_files(issue_id: str) -> list[str]:
    return [
        (Path("context") / "TASKS" / issue_id / filename).as_posix()
        for filename in PACK_FILENAMES
    ]


def _context_pack_result_payload(
    repo_root: Path,
    issue_id: str,
    task_dir: Path,
    files: tuple[Path, ...],
) -> dict[str, Any]:
    return {
        "issue_id": issue_id,
        "exists": task_dir.is_dir(),
        "task_dir": _repo_relative_or_string(repo_root, task_dir),
        "files": [_repo_relative_or_string(repo_root, path) for path in files],
        "retrieval_audit": _read_json_file(task_dir / "RETRIEVAL_AUDIT.json"),
        "authority_notice": CONTEXT_AUTHORITY_NOTICE,
    }


def _preview_response(
    kind: str,
    *,
    planned_actions: list[str],
    planned_files: list[str],
    **extra: Any,
) -> ApiResponse:
    return ApiResponse(
        200,
        {
            "schema_version": READONLY_SERVER_SCHEMA_VERSION,
            "kind": kind,
            "dry_run_only": True,
            "repo_writes_performed": False,
            "git_writes_performed": False,
            "github_writes_performed": False,
            "network_calls_performed": False,
            "planned_actions": planned_actions,
            "planned_files": planned_files,
            "authority_warning": FORGE_PREVIEW_AUTHORITY_WARNING,
            "authority_notice": SITE_EXPORT_AUTHORITY_NOTICE,
            **extra,
        },
    )


def _issue_action_response(
    kind: str,
    result: Any,
    *,
    planned_files: list[str],
    diff: str,
) -> ApiResponse:
    return ApiResponse(
        200,
        {
            "schema_version": READONLY_SERVER_SCHEMA_VERSION,
            "kind": kind,
            "action_performed": True,
            "repo_writes_performed": True,
            "git_writes_performed": False,
            "github_writes_performed": False,
            "network_calls_performed": False,
            "audit_logged": True,
            "planned_files": planned_files,
            "written_files": planned_files,
            "issue": result.issue.model_dump(mode="json"),
            "path": result.relative_path.as_posix(),
            "artifact_status_changed": result.artifact_status_changed,
            "diff": diff,
            "authority_warning": ISSUE_AUTHORITY_NOTICE,
            "authority_notice": ISSUE_AUTHORITY_NOTICE,
        },
    )


def _artifact_action_response(
    kind: str,
    result: Any,
    *,
    planned_files: list[str],
    diff: str,
    validation: dict[str, Any],
) -> ApiResponse:
    return ApiResponse(
        200,
        {
            "schema_version": READONLY_SERVER_SCHEMA_VERSION,
            "kind": kind,
            "action_performed": True,
            "repo_writes_performed": True,
            "git_writes_performed": False,
            "github_writes_performed": False,
            "network_calls_performed": False,
            "audit_logged": True,
            "planned_files": planned_files,
            "written_files": planned_files,
            "artifact": result.artifact.model_dump(mode="json"),
            "path": result.relative_path.as_posix(),
            "yaml": result.yaml_text(),
            "diff": diff,
            "validation": validation,
            "accepted_write_performed": False,
            "authority_warning": ARTIFACT_WRITE_AUTHORITY_NOTICE,
            "authority_notice": ARTIFACT_WRITE_AUTHORITY_NOTICE,
        },
    )


def _diff_text(
    before: str,
    after: str,
    *,
    fromfile: str,
    tofile: str,
) -> str:
    return "".join(
        unified_diff(
            before.splitlines(keepends=True),
            after.splitlines(keepends=True),
            fromfile=fromfile,
            tofile=tofile,
            lineterm="",
        )
    )


def _web_action_kind(action: str) -> WebActionKind:
    mapping = {
        "local_issue_preview": WebActionKind.ISSUE_CREATE,
        "issue_create": WebActionKind.ISSUE_CREATE,
        "issue_update": WebActionKind.ISSUE_UPDATE,
        "issue_close": WebActionKind.ISSUE_CLOSE,
        "artifact_create": WebActionKind.ARTIFACT_CREATE,
        "artifact_update": WebActionKind.ARTIFACT_UPDATE,
        "source_attach": WebActionKind.SOURCE_ATTACH,
        "evidence_attach": WebActionKind.EVIDENCE_ATTACH,
        "review_packet_create": WebActionKind.REVIEW_PACKET_CREATE,
        "review_decision_create": WebActionKind.REVIEW_DECISION_CREATE,
        "promotion_preview": WebActionKind.PROMOTION_PREVIEW,
        "promotion_confirm": WebActionKind.PROMOTION_CONFIRM,
        "github_issue_preview": WebActionKind.ISSUE_PUBLISH_GITHUB,
        "github_issue_create": WebActionKind.ISSUE_PUBLISH_GITHUB,
        "branch_preview": WebActionKind.FORGE_BRANCH_CREATE,
        "branch_create": WebActionKind.FORGE_BRANCH_CREATE,
        "commit_preview": WebActionKind.FORGE_COMMIT_CREATE,
        "commit": WebActionKind.FORGE_COMMIT_CREATE,
        "push_preview": WebActionKind.FORGE_PUSH_CREATE,
        "push": WebActionKind.FORGE_PUSH_CREATE,
        "github_pr_preview": WebActionKind.FORGE_PR_CREATE,
        "github_pr_create": WebActionKind.FORGE_PR_CREATE,
        "github_pr_submit": WebActionKind.FORGE_PR_CREATE,
        "review_packet_preview": WebActionKind.REVIEW_PACKET_CREATE,
        "review_decision_preview": WebActionKind.REVIEW_DECISION_CREATE,
        "context_build": WebActionKind.CONTEXT_BUILD,
        "validate_run": WebActionKind.VALIDATE_RUN,
        "gate_run": WebActionKind.GATE_RUN,
    }
    try:
        return mapping[action]
    except KeyError as exc:
        raise ValueError(f"unknown web action audit kind: {action}") from exc


def _optional_str(value: object) -> str | None:
    return value if isinstance(value, str) and value.strip() else None


def _normalize_local_actor(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip()
    return normalized or None


def _performed_summary(result: dict[str, Any], key: str) -> str | None:
    if result.get(key) is True:
        return "performed"
    return None


def _live_response(kind: str, **payload: Any) -> ApiResponse:
    authority_notice = payload.pop("authority_notice", SITE_EXPORT_AUTHORITY_NOTICE)
    return ApiResponse(
        200,
        {
            "schema_version": READONLY_SERVER_SCHEMA_VERSION,
            "kind": kind,
            "source_of_truth": "repository",
            "authority_notice": authority_notice,
            **payload,
        },
    )


def _workspace_info_payload(info: Any) -> dict[str, Any]:
    return {
        "name": info.name,
        "repo_root": str(info.repo_root),
        "mode": info.mode,
        "kb_roots": [
            {
                "name": root.name,
                "path": root.path,
                "readonly": root.readonly,
                "priority": root.priority,
            }
            for root in info.kb_roots
        ],
    }


def _validation_payload(report: Any) -> dict[str, Any]:
    return {
        "ok": report.ok,
        "checked_count": report.checked_count,
        "failures": [
            {
                "gate": failure.gate,
                "source_path": failure.source_path,
                "artifact_id": failure.artifact_id,
                "message": failure.message,
            }
            for failure in report.failures
        ],
    }


def _gate_run_payload(
    report: dict[str, Any],
    report_paths: list[str],
) -> dict[str, Any]:
    raw_summary = report.get("summary")
    summary: dict[Any, Any] = raw_summary if isinstance(raw_summary, dict) else {}
    return {
        "verdict": report.get("verdict", "not_run"),
        "report": report,
        "report_paths": report_paths,
        "gates": report.get("gates", []),
        "blocking_issues": report.get("blocking_issues", []),
        "nonblocking_issues": report.get("nonblocking_issues", []),
        "pass_count": _summary_count(summary, "gates_passed"),
        "fail_count": _summary_count(summary, "gates_failed"),
        "skipped_count": _summary_count(summary, "gates_skipped"),
        "skipped_is_pass": False,
        "gate_pass_is_accepted_authority": False,
    }


def _summary_count(summary: dict[Any, Any], key: str) -> int:
    value = summary.get(key, 0)
    return value if isinstance(value, int) else 0


def _artifact_payload(record: LoadedRecord) -> dict[str, Any]:
    artifact = cast(BaseArtifact, record.record)
    payload = artifact.model_dump(mode="json")
    payload["content_revision"] = artifact.content_revision
    payload["path"] = record.source_path.as_posix()
    payload["kb_root"] = record.kb_root_name
    payload["kb_root_readonly"] = record.kb_root_readonly
    if record.kb_relative_path is not None:
        payload["kb_relative_path"] = record.kb_relative_path.as_posix()
    return payload


def _artifact_snapshot(api: ReadOnlySiteApi, artifact_id: str) -> dict[str, Any]:
    normalized_id = validate_artifact_id(artifact_id)
    loaded = api._live_artifact_records()
    if isinstance(loaded, ApiResponse):
        raise DraftWriteServiceError(
            "cannot load repository records",
            code="repository_load_failed",
            remediation="Fix repository load errors before editing artifacts.",
        )
    for record in loaded:
        if record.id == normalized_id:
            artifact = cast(BaseArtifact, record.record)
            artifact_dict = artifact.model_dump(mode="json")
            artifact_dict["content_revision"] = artifact.content_revision
            return {
                "artifact": artifact_dict,
                "yaml": dump_yaml_deterministic(artifact),
                "path": record.source_path.as_posix(),
            }
    raise DraftWriteServiceError(
        f"artifact not found: {normalized_id}",
        code="artifact_not_found",
        remediation="Check the artifact id and retry.",
        details={"artifact_id": normalized_id},
    )


def _evidence_path_warnings(
    api: ReadOnlySiteApi,
    payload: dict[str, Any],
) -> list[str]:
    raw_path = str(payload.get("path", "")).strip().replace("\\", "/")
    if not raw_path or raw_path.lower().startswith("external:"):
        return []
    evidence_path = Path(raw_path)
    resolved = (
        evidence_path.resolve()
        if evidence_path.is_absolute()
        else api.app.context.resolve(evidence_path)
    )
    try:
        resolved.relative_to(api.app.context.repo_root)
    except ValueError:
        return [f"evidence path escapes repository: {raw_path}"]
    if resolved.exists():
        return []
    return [f"missing local evidence path: {raw_path}"]


def _repo_files_under(repo_root: Path, root: Path) -> list[str]:
    if not root.is_dir():
        return []
    return [
        _repo_relative_or_string(repo_root, path)
        for path in sorted(root.iterdir())
        if path.is_file()
    ]


def _read_json_file(path: Path) -> Any:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _read_jsonl_file(path: Path) -> list[dict[str, Any]] | ApiResponse:
    if not path.is_file():
        return []
    entries: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return _error(
            500,
            "audit_log_unreadable",
            "Web action audit log could not be read.",
        )
    for index, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError:
            return _error(
                500,
                "audit_log_unreadable",
                f"Web action audit log line {index} is not valid JSON.",
            )
        if isinstance(raw, dict):
            entries.append(cast(dict[str, Any], raw))
    return entries


def _repo_relative_or_string(repo_root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return str(path)


def _error(status: int, code: str, message: str) -> ApiResponse:
    return ApiResponse(
        status,
        {
            "schema_version": READONLY_SERVER_SCHEMA_VERSION,
            "code": code,
            "message": message,
            "blocking": True,
            "authority_notice": SITE_EXPORT_AUTHORITY_NOTICE,
        },
    )
