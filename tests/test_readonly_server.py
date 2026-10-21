from __future__ import annotations

import json
import subprocess
from http.server import HTTPServer
from pathlib import Path
from threading import Thread
from typing import Any
from urllib.request import Request, urlopen

import yaml  # type: ignore[import-untyped]
from typer.testing import CliRunner

from cosheaf.app import open_app
from cosheaf.cli import app
from cosheaf.server import READONLY_SERVER_HOST, ReadOnlySiteApi, make_handler

runner = CliRunner()


class _FakeCredentialProvider:
    def __init__(self, *, has_token: bool = True) -> None:
        self.has_token = has_token
        self.calls = 0
        self.secret = "secret-token-value"

    def provider_name(self) -> str:
        return "fake-provider"

    def has_github_token(self) -> bool:
        self.calls += 1
        return self.has_token


def _write_yaml(repo_root: Path, relative_path: str, data: dict[str, Any]) -> None:
    path = repo_root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def _artifact_data() -> dict[str, Any]:
    return {
        "id": "claim.fixture.readonly-server",
        "type": "claim",
        "title": "Read-only server fixture claim",
        "domain": ["testing"],
        "status": "draft",
        "created_at": "2026-06-19T00:00:00Z",
        "updated_at": "2026-06-19T00:00:00Z",
        "authors": ["tester"],
        "depends_on": [],
        "supersedes": [],
        "tags": ["readonly-server"],
        "statement": "The local server can expose read-only site payloads.",
        "evidence": [],
        "review": {"state": "requested", "notes": "Fixture review."},
        "risk": {"level": "low", "notes": "Fixture risk."},
    }


def _issue_data() -> dict[str, Any]:
    return {
        "id": "issue.fixture.readonly-server",
        "type": "issue",
        "title": "Read-only server fixture issue",
        "status": "open",
        "summary": "Exercise read-only local API routing.",
        "created_at": "2026-06-19T00:00:00Z",
        "updated_at": "2026-06-19T00:00:00Z",
        "authors": ["tester"],
        "labels": ["readonly-server"],
        "related_artifacts": ["claim.fixture.readonly-server"],
        "related_sources": [],
        "scope": "public",
    }


def _fixture_workspace(repo_root: Path) -> None:
    _write_yaml(
        repo_root,
        "kb/draft/claims/readonly-server.yaml",
        _artifact_data(),
    )
    _write_yaml(
        repo_root,
        "issues/open/readonly-server.yaml",
        _issue_data(),
    )


def _audit_entries(repo_root: Path) -> list[dict[str, Any]]:
    audit_path = repo_root / ".cosheaf" / "audit" / "web-actions.jsonl"
    return [
        json.loads(line)
        for line in audit_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _repo_file_snapshot(repo_root: Path) -> dict[str, str]:
    return {
        path.relative_to(repo_root).as_posix(): path.read_text(encoding="utf-8")
        for path in sorted(repo_root.rglob("*"))
        if path.is_file()
    }


def _artifact_statuses(repo_root: Path) -> dict[str, tuple[str, str]]:
    statuses: dict[str, tuple[str, str]] = {}
    for path in sorted((repo_root / "kb").rglob("*.yaml")):
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        statuses[str(data["id"])] = (
            str(data.get("status", "")),
            str(data.get("review", {}).get("state", "")),
        )
    return statuses


def test_readonly_site_api_routes_export_payloads_without_cli_subprocess(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    _fixture_workspace(tmp_path)

    def fail_subprocess_run(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("read-only server must not shell out to CLI")

    monkeypatch.setattr(subprocess, "run", fail_subprocess_run)

    api = ReadOnlySiteApi(open_app(tmp_path))

    health = api.handle("GET", "/api/health")
    assert health.status == 200
    assert health.payload["readonly"] is True
    assert health.payload["host"] == "127.0.0.1"

    workspace = api.handle("GET", "/api/workspace")
    assert workspace.status == 200
    assert workspace.payload["kind"] == "workspace"

    artifacts = api.handle("GET", "/api/artifacts")
    assert artifacts.status == 200
    assert artifacts.payload["artifacts"][0]["id"] == "claim.fixture.readonly-server"

    context = api.handle("GET", "/api/context/issue.fixture.readonly-server")
    assert context.status == 200
    assert context.payload["issue_id"] == "issue.fixture.readonly-server"
    assert context.payload["context_pack"]["related_artifacts"] == [
        "claim.fixture.readonly-server"
    ]

    missing = api.handle("GET", "/api/context/issue.missing")
    assert missing.status == 404
    assert missing.payload["code"] == "context_pack_not_found"

    refused_write = api.handle("POST", "/api/artifacts")
    assert refused_write.status == 405
    assert refused_write.payload["code"] == "method_not_allowed"


def test_live_repository_api_reads_current_repo_state_without_writes(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    _fixture_workspace(tmp_path)
    context_dir = tmp_path / "context" / "TASKS" / "issue.fixture.readonly-server"
    context_dir.mkdir(parents=True)
    (context_dir / "CONTEXT.md").write_text("# Fixture context\n", encoding="utf-8")
    (context_dir / "RETRIEVAL_AUDIT.json").write_text(
        json.dumps({"card_count": 1}, indent=2) + "\n",
        encoding="utf-8",
    )
    report_dir = tmp_path / ".cosheaf" / "reports"
    report_dir.mkdir(parents=True)
    (report_dir / "20260619T000000000000Z-gate-report.json").write_text(
        json.dumps(
            {
                "verdict": "fail",
                "summary": {"records_checked": 2},
                "blocking_issues": [{"message": "fixture block"}],
                "nonblocking_issues": [],
                "gates": [],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    audit_dir = tmp_path / ".cosheaf" / "audit"
    audit_dir.mkdir(parents=True)
    audit_dir.joinpath("web-actions.jsonl").write_text(
        json.dumps(
            {
                "timestamp": "2026-06-19T00:00:00Z",
                "actor": "local.web",
                "action": "forge.pr_create",
                "result_status": "preview",
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    def fail_subprocess_run(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("live read endpoints must not shell out to CLI")

    monkeypatch.setattr(subprocess, "run", fail_subprocess_run)

    before = _repo_file_snapshot(tmp_path)
    api = ReadOnlySiteApi(open_app(tmp_path))

    workspace = api.handle("GET", "/api/workspace/live")
    assert workspace.status == 200
    assert workspace.payload["kind"] == "workspace_live"
    assert workspace.payload["source_of_truth"] == "repository"
    assert workspace.payload["workspace"]["mode"] == "legacy"

    status = api.handle("GET", "/api/status")
    assert status.status == 200
    assert status.payload["kind"] == "repository_status"
    assert status.payload["validation"]["ok"] is True
    assert status.payload["source_of_truth"] == "repository"

    issues = api.handle("GET", "/api/issues/live")
    assert issues.status == 200
    assert issues.payload["count"] == 1
    assert issues.payload["issues"][0]["id"] == "issue.fixture.readonly-server"

    issue = api.handle("GET", "/api/issues/issue.fixture.readonly-server")
    assert issue.status == 200
    assert issue.payload["issue"]["path"] == "issues/open/readonly-server.yaml"

    artifacts = api.handle("GET", "/api/artifacts/live")
    assert artifacts.status == 200
    assert artifacts.payload["count"] == 1
    assert artifacts.payload["artifacts"][0]["id"] == "claim.fixture.readonly-server"
    assert artifacts.payload["artifacts"][0]["path"] == (
        "kb/draft/claims/readonly-server.yaml"
    )

    artifact = api.handle("GET", "/api/artifacts/claim.fixture.readonly-server")
    assert artifact.status == 200
    assert artifact.payload["artifact"]["id"] == "claim.fixture.readonly-server"

    latest_context = api.handle(
        "GET",
        "/api/context/issue.fixture.readonly-server/latest",
    )
    assert latest_context.status == 200
    assert latest_context.payload["context_pack"]["exists"] is True
    assert latest_context.payload["context_pack"]["files"] == [
        "context/TASKS/issue.fixture.readonly-server/CONTEXT.md",
        "context/TASKS/issue.fixture.readonly-server/RETRIEVAL_AUDIT.json",
    ]
    assert latest_context.payload["context_pack"]["retrieval_audit"] == {
        "card_count": 1
    }

    latest_gate = api.handle("GET", "/api/gates/latest")
    assert latest_gate.status == 200
    assert latest_gate.payload["gate_report"]["path"] == (
        ".cosheaf/reports/20260619T000000000000Z-gate-report.json"
    )
    assert latest_gate.payload["gate_report"]["report"]["verdict"] == "fail"

    audit = api.handle("GET", "/api/audit/recent")
    assert audit.status == 200
    assert audit.payload["count"] == 1
    assert audit.payload["entries"][0]["action"] == "forge.pr_create"

    missing_issue = api.handle("GET", "/api/issues/issue.missing")
    assert missing_issue.status == 404
    assert missing_issue.payload["code"] == "issue_not_found"

    assert _repo_file_snapshot(tmp_path) == before


def test_context_build_preview_then_confirm_writes_context_pack(
    tmp_path: Path,
) -> None:
    _fixture_workspace(tmp_path)
    api = ReadOnlySiteApi(open_app(tmp_path))
    payload = {
        "role": "orchestrator",
        "public_only": True,
        "max_cards": 5,
        "max_full_artifacts": 0,
    }

    preview = api.handle(
        "POST",
        "/api/context/issue.fixture.readonly-server/preview-build",
        json.dumps(payload),
    )

    assert preview.status == 200
    assert preview.payload["kind"] == "context_build_preview"
    assert preview.payload["repo_writes_performed"] is False
    assert "context/TASKS/issue.fixture.readonly-server/CONTEXT.md" in (
        preview.payload["planned_files"]
    )
    assert "context/TASKS/issue.fixture.readonly-server/RETRIEVAL_AUDIT.json" in (
        preview.payload["planned_files"]
    )
    assert "context/TASKS/issue.fixture.readonly-server/COMMANDS.md" in (
        preview.payload["planned_files"]
    )
    assert not (tmp_path / "context" / "TASKS").exists()

    blocked = api.handle(
        "POST",
        "/api/context/issue.fixture.readonly-server/build",
        json.dumps(payload),
    )
    assert blocked.status == 400
    assert blocked.payload["code"] == "confirm_required"

    built = api.handle(
        "POST",
        "/api/context/issue.fixture.readonly-server/build",
        json.dumps({**payload, "confirm": True}),
    )

    assert built.status == 200
    assert built.payload["kind"] == "context_build"
    assert built.payload["repo_writes_performed"] is True
    assert built.payload["context_pack"]["files"] == preview.payload["planned_files"]
    assert (
        tmp_path
        / "context"
        / "TASKS"
        / "issue.fixture.readonly-server"
        / "CONTEXT.md"
    ).is_file()

    latest = api.handle(
        "GET",
        "/api/context/issue.fixture.readonly-server/latest",
    )
    assert latest.status == 200
    assert latest.payload["context_pack"]["exists"] is True
    assert latest.payload["context_pack"]["retrieval_audit"]["issue_id"] == (
        "issue.fixture.readonly-server"
    )

    entries = _audit_entries(tmp_path)
    assert [entry["action"] for entry in entries] == [
        "context.build",
        "context.build",
        "context.build",
    ]
    assert entries[0]["preview_only"] is True
    assert entries[1]["result_status"] == "confirm_required"
    assert entries[2]["repo_writes_performed"] is True


def test_validate_and_gate_run_actions_are_audited_without_acceptance_changes(
    tmp_path: Path,
) -> None:
    _fixture_workspace(tmp_path)
    api = ReadOnlySiteApi(open_app(tmp_path))
    before = _artifact_statuses(tmp_path)

    validate_run = api.handle("POST", "/api/validate/run", "{}")

    assert validate_run.status == 200
    assert validate_run.payload["kind"] == "validate_run"
    assert validate_run.payload["validation"]["ok"] is True
    assert validate_run.payload["repo_writes_performed"] is False
    assert validate_run.payload["accepted_status_changed"] is False
    assert _artifact_statuses(tmp_path) == before

    gate_run = api.handle("POST", "/api/gate/run", "{}")

    assert gate_run.status == 200
    assert gate_run.payload["kind"] == "gate_run"
    assert gate_run.payload["gate"]["verdict"] == "pass"
    assert gate_run.payload["gate"]["skipped_count"] >= 0
    assert gate_run.payload["gate"]["skipped_is_pass"] is False
    assert gate_run.payload["gate"]["gate_pass_is_accepted_authority"] is False
    assert gate_run.payload["repo_writes_performed"] is True
    assert gate_run.payload["accepted_status_changed"] is False
    assert _artifact_statuses(tmp_path) == before

    latest = api.handle("GET", "/api/gates/latest")
    assert latest.status == 200
    assert latest.payload["gate_report"]["exists"] is True
    assert latest.payload["gate_report"]["verdict"] == "pass"

    entries = _audit_entries(tmp_path)
    assert [entry["action"] for entry in entries] == [
        "validate.run",
        "gate.run",
    ]
    assert entries[0]["repo_writes_performed"] is False
    assert entries[1]["repo_writes_performed"] is True


def test_preview_endpoints_plan_actions_without_repo_or_github_writes(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    _fixture_workspace(tmp_path)

    def fail_subprocess_run(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("preview endpoints must not run git, gh, or CLI")

    monkeypatch.setattr(subprocess, "run", fail_subprocess_run)

    api = ReadOnlySiteApi(open_app(tmp_path))

    local_issue = api.handle(
        "POST",
        "/api/forge/local-issues/preview",
        json.dumps(
            {
                "issue_id": "issue.fixture.web-preview",
                "title": "Preview a local issue",
                "summary": "Show the exact local issue file before writing.",
                "authors": ["tester"],
                "labels": ["website-preview"],
                "related_artifacts": ["claim.fixture.readonly-server"],
                "related_sources": [],
                "scope": "private",
            }
        ),
    )
    assert local_issue.status == 200
    assert local_issue.payload["kind"] == "local_issue_preview"
    assert local_issue.payload["dry_run_only"] is True
    assert local_issue.payload["repo_writes_performed"] is False
    assert local_issue.payload["github_writes_performed"] is False
    assert local_issue.payload["planned_files"] == [
        "issues/open/issue.fixture.web-preview.yaml"
    ]
    assert not (tmp_path / "issues/open/issue.fixture.web-preview.yaml").exists()

    github_issue = api.handle(
        "POST",
        "/api/forge/issues/preview",
        json.dumps({"source_path": "issues/open/readonly-server.yaml"}),
    )
    assert github_issue.status == 200
    assert github_issue.payload["kind"] == "github_issue_preview"
    assert github_issue.payload["network_calls_performed"] is False
    assert github_issue.payload["github_writes_performed"] is False
    assert github_issue.payload["planned_files"] == [
        "issues/open/readonly-server.yaml"
    ]

    github_pr = api.handle(
        "POST",
        "/api/forge/prs/preview",
        json.dumps({"base": "main", "head": "website-preview-actions"}),
    )
    assert github_pr.status == 200
    assert github_pr.payload["kind"] == "github_pr_preview"
    assert github_pr.payload["planned_actions"] == [
        "create GitHub pull request preview"
    ]
    assert github_pr.payload["github_pr_plan"]["head"] == "website-preview-actions"

    review_packet = api.handle(
        "POST",
        "/api/forge/review-packets/preview",
        json.dumps({"issue_id": "issue.fixture.readonly-server"}),
    )
    assert review_packet.status == 200
    assert review_packet.payload["kind"] == "review_packet_preview"
    assert review_packet.payload["planned_files"] == [
        "reviews/website/issue.fixture.readonly-server-review-packet.md"
    ]
    assert review_packet.payload["repo_writes_performed"] is False
    assert not (
        tmp_path
        / "reviews/website/issue.fixture.readonly-server-review-packet.md"
    ).exists()


def test_preview_endpoints_do_not_use_backend_credentials(tmp_path: Path) -> None:
    _fixture_workspace(tmp_path)
    credentials = _FakeCredentialProvider()
    api = ReadOnlySiteApi(open_app(tmp_path), credential_provider=credentials)

    response = api.handle(
        "POST",
        "/api/forge/prs/preview",
        json.dumps({"base": "main", "head": "website-authenticated-forge-actions"}),
    )

    assert response.status == 200
    assert response.payload["kind"] == "github_pr_preview"
    assert credentials.calls == 0


def test_b281_forge_flow_preview_endpoints_are_read_only(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    def fail_subprocess_run(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("B2.8.1 preview endpoints must not run git or gh")

    monkeypatch.setattr(subprocess, "run", fail_subprocess_run)

    api = ReadOnlySiteApi(open_app(tmp_path))
    requests = [
        (
            "/api/forge/branch/preview",
            {"branch": "feature.web-pr"},
            "branch_preview",
        ),
        (
            "/api/forge/commit/preview",
            {"message": "Add web PR flow"},
            "commit_preview",
        ),
        (
            "/api/forge/push/preview",
            {"head": "feature.web-pr"},
            "push_preview",
        ),
        (
            "/api/forge/pr/preview",
            {"base": "main", "head": "feature.web-pr"},
            "github_pr_preview",
        ),
    ]

    for endpoint, payload, kind in requests:
        response = api.handle("POST", endpoint, json.dumps(payload))
        assert response.status == 200, response.payload
        assert response.payload["kind"] == kind
        assert response.payload["dry_run_only"] is True
        assert response.payload["repo_writes_performed"] is False
        assert response.payload["git_writes_performed"] is False
        assert response.payload["github_writes_performed"] is False
        assert response.payload["network_calls_performed"] is False


def test_b281_forge_push_and_pr_submit_endpoints_return_pr_url(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    credentials = _FakeCredentialProvider()
    calls: list[list[str]] = []

    def fake_subprocess_run(
        args: list[str],
        **_kwargs: object,
    ) -> subprocess.CompletedProcess[str]:
        calls.append(args)
        if args == ["git", "branch", "--show-current"]:
            stdout = "feature.web-pr\n"
        elif args == ["git", "status", "--porcelain=v1"]:
            stdout = ""
        elif args in (
            ["git", "rev-parse", "HEAD"],
            ["git", "rev-parse", "feature.web-pr"],
        ):
            stdout = "cafebabe0123456789\n"
        elif args[:2] == ["git", "push"]:
            stdout = ""
        elif args[:3] == ["gh", "pr", "create"]:
            stdout = "https://github.com/CheemsaDoge/tcs-cosheaf/pull/1004\n"
        else:
            raise AssertionError(f"unexpected subprocess call: {args}")
        return subprocess.CompletedProcess(
            args=args,
            returncode=0,
            stdout=stdout,
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", fake_subprocess_run)

    api = ReadOnlySiteApi(open_app(tmp_path), credential_provider=credentials)
    push_response = api.handle(
        "POST",
        "/api/forge/push/create",
        json.dumps({"head": "feature.web-pr", "confirm": True}),
    )
    pr_response = api.handle(
        "POST",
        "/api/forge/pr/create",
        json.dumps(
            {
                "base": "main",
                "head": "feature.web-pr",
                "draft": True,
                "confirm": True,
            }
        ),
    )

    assert push_response.status == 200, push_response.payload
    assert push_response.payload["kind"] == "push"
    assert push_response.payload["forge_action"]["push_performed"] is True
    assert pr_response.status == 200, pr_response.payload
    assert pr_response.payload["kind"] == "github_pr_submit"
    assert pr_response.payload["forge_action"]["validation_performed"] is True
    assert pr_response.payload["forge_action"]["gate_performed"] is True
    assert pr_response.payload["forge_action"]["push_performed"] is True
    assert pr_response.payload["forge_action"]["github_pr_created"] is True
    assert pr_response.payload["forge_action"]["commit_hash"] == "cafebabe0123456789"
    assert pr_response.payload["forge_action"]["github_pr_url"].endswith("/pull/1004")
    assert credentials.calls == 1
    assert calls[0] == ["git", "push", "-u", "origin", "feature.web-pr"]
    assert ["git", "push", "-u", "origin", "feature.web-pr"] in calls[1:]
    assert calls[-1][:3] == ["gh", "pr", "create"]

    entries = _audit_entries(tmp_path)
    assert [entry["action"] for entry in entries] == [
        "forge.push_create",
        "forge.pr_create",
    ]
    assert entries[1]["commit_hash"] == "cafebabe0123456789"
    assert entries[1]["github_urls"] == [
        "https://github.com/CheemsaDoge/tcs-cosheaf/pull/1004"
    ]


def test_b282_pr_status_reads_github_metadata_without_review_import(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    _fixture_workspace(tmp_path)
    before = _repo_file_snapshot(tmp_path)
    calls: list[list[str]] = []
    gh_payload = {
        "number": 570,
        "title": "Add PR review status",
        "state": "OPEN",
        "isDraft": False,
        "mergeStateStatus": "CLEAN",
        "mergeable": "MERGEABLE",
        "headRefName": "web-pr-review-status",
        "baseRefName": "main",
        "url": "https://github.com/CheemsaDoge/tcs-cosheaf/pull/570",
        "author": {"login": "CheemsaDoge"},
        "reviewDecision": "APPROVED",
        "body": "- [x] make test\n- [ ] reviewer sign-off\n",
        "updatedAt": "2026-06-19T21:00:00Z",
        "closingIssuesReferences": [
            {
                "number": 570,
                "title": "Add PR review status and comments in web",
                "state": "OPEN",
                "url": "https://github.com/CheemsaDoge/tcs-cosheaf/issues/570",
            }
        ],
        "statusCheckRollup": [
            {
                "name": "lint",
                "status": "COMPLETED",
                "conclusion": "SUCCESS",
                "detailsUrl": "https://example.invalid/lint",
            },
            {
                "name": "gate",
                "status": "COMPLETED",
                "conclusion": "SUCCESS",
                "detailsUrl": "https://example.invalid/gate",
            },
        ],
        "reviews": [
            {
                "author": {"login": "reviewer"},
                "state": "APPROVED",
                "submittedAt": "2026-06-19T21:01:00Z",
                "body": "Looks ready.",
            }
        ],
        "comments": [
            {
                "author": {"login": "reviewer"},
                "createdAt": "2026-06-19T21:02:00Z",
                "url": "https://github.com/CheemsaDoge/tcs-cosheaf/pull/570#x",
                "body": "One non-blocking note.",
            }
        ],
    }

    def fake_subprocess_run(
        args: list[str],
        **_kwargs: object,
    ) -> subprocess.CompletedProcess[str]:
        calls.append(args)
        return subprocess.CompletedProcess(
            args=args,
            returncode=0,
            stdout=json.dumps(gh_payload),
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", fake_subprocess_run)

    api = ReadOnlySiteApi(open_app(tmp_path))
    response = api.handle(
        "GET",
        "/api/forge/pr-status?number=570&base=main&head=web-pr-review-status",
    )

    assert response.status == 200, response.payload
    payload = response.payload["pr_status"]
    assert response.payload["kind"] == "github_pr_status"
    assert response.payload["github_auth_available"] is True
    assert response.payload["git_writes_performed"] is False
    assert response.payload["github_writes_performed"] is False
    assert payload["source_of_truth"] == "github"
    assert payload["pr"]["number"] == 570
    assert payload["linked_issue"]["number"] == 570
    assert payload["checklist"]["completed"] == 1
    assert payload["checklist"]["total"] == 2
    assert payload["ci"]["status"] == "success"
    assert payload["gate"]["status"] == "success"
    assert payload["gate"]["skipped_is_pass"] is False
    assert payload["cosheaf_review"]["status"] == "not_imported"
    assert payload["github_reviews"][0]["state"] == "approved"
    assert payload["review_comments"][0]["body"] == "One non-blocking note."
    assert "imported" in payload["warnings"][0]
    assert calls[0][:4] == ["gh", "pr", "view", "570"]
    assert _repo_file_snapshot(tmp_path) == before


def test_b282_pr_status_degrades_when_github_is_unavailable(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    _fixture_workspace(tmp_path)

    def fake_subprocess_run(
        args: list[str],
        **_kwargs: object,
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args=args,
            returncode=1,
            stdout="",
            stderr="gh auth required",
        )

    monkeypatch.setattr(subprocess, "run", fake_subprocess_run)

    api = ReadOnlySiteApi(open_app(tmp_path))
    response = api.handle(
        "GET",
        "/api/forge/pr-status?number=570&base=main&head=web-pr-review-status",
    )

    assert response.status == 200, response.payload
    payload = response.payload["pr_status"]
    assert payload["source_of_truth"] == "degraded"
    assert payload["github_auth_available"] is False
    assert payload["network_calls_performed"] is False
    assert payload["pr"]["number"] == 570
    assert payload["pr"]["base"] == "main"
    assert payload["pr"]["head"] == "web-pr-review-status"
    assert payload["ci"]["status"] == "unknown"
    assert payload["gate"]["status"] == "unknown"
    assert payload["cosheaf_review"]["status"] == "not_imported"
    assert payload["github_reviews"] == []
    assert "unavailable" in payload["warnings"][0]


def test_preview_endpoint_writes_web_action_audit_without_repo_file(
    tmp_path: Path,
) -> None:
    _fixture_workspace(tmp_path)
    api = ReadOnlySiteApi(open_app(tmp_path))

    response = api.handle(
        "POST",
        "/api/forge/local-issues/preview",
        json.dumps(
            {
                "issue_id": "issue.fixture.web-preview",
                "title": "Preview a local issue",
                "summary": "Show the exact local issue file before writing.",
                "authors": ["tester"],
                "labels": ["website-preview"],
                "related_artifacts": ["claim.fixture.readonly-server"],
                "related_sources": [],
                "scope": "private",
            }
        ),
    )

    assert response.status == 200
    assert not (tmp_path / "issues/open/issue.fixture.web-preview.yaml").exists()
    entries = _audit_entries(tmp_path)
    assert len(entries) == 1
    assert entries[0]["action"] == "issue.create"
    assert entries[0]["actor"] == "local.web"
    assert entries[0]["preview_only"] is True
    assert entries[0]["confirmed"] is False
    assert entries[0]["performed"] is False
    assert entries[0]["planned_files"] == [
        "issues/open/issue.fixture.web-preview.yaml"
    ]


def test_health_endpoint_reports_configured_local_actor(tmp_path: Path) -> None:
    _fixture_workspace(tmp_path)
    api = ReadOnlySiteApi(open_app(tmp_path), local_actor="Ada Local")

    response = api.handle("GET", "/api/health")

    assert response.status == 200
    assert response.payload["local_actor"] == "Ada Local"
    assert response.payload["local_actor_configured"] is True
    assert response.payload["local_actor_is_auth"] is False


def test_authenticated_create_endpoints_require_auth_and_confirm(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    _fixture_workspace(tmp_path)

    def fail_subprocess_run(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("unauthorized or unconfirmed create must not run gh")

    monkeypatch.setattr(subprocess, "run", fail_subprocess_run)

    no_auth_api = ReadOnlySiteApi(open_app(tmp_path))
    no_auth = no_auth_api.handle(
        "POST",
        "/api/forge/prs/create",
        json.dumps(
            {
                "base": "main",
                "head": "website-authenticated-forge-actions",
                "confirm": True,
            }
        ),
    )
    assert no_auth.status == 401
    assert no_auth.payload["code"] == "auth_required"

    credentials = _FakeCredentialProvider()
    authed_api = ReadOnlySiteApi(open_app(tmp_path), credential_provider=credentials)
    no_confirm = authed_api.handle(
        "POST",
        "/api/forge/prs/create",
        json.dumps({"base": "main", "head": "website-authenticated-forge-actions"}),
    )
    assert no_confirm.status == 400
    assert no_confirm.payload["code"] == "confirm_required"

    audit = _audit_entries(tmp_path)
    assert [entry["result_status"] for entry in audit] == [
        "auth_required",
        "confirm_required",
    ]


def test_authenticated_create_endpoints_call_forge_and_write_redacted_audit(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    _fixture_workspace(tmp_path)
    credentials = _FakeCredentialProvider()
    calls: list[list[str]] = []

    def fake_subprocess_run(
        args: list[str],
        **_kwargs: object,
    ) -> subprocess.CompletedProcess[str]:
        calls.append(args)
        if args[:3] == ["gh", "issue", "create"]:
            stdout = "https://github.com/CheemsaDoge/tcs-cosheaf/issues/1001\n"
        elif args[:3] == ["gh", "pr", "create"]:
            stdout = "https://github.com/CheemsaDoge/tcs-cosheaf/pull/1002\n"
        else:
            raise AssertionError(f"unexpected subprocess call: {args}")
        return subprocess.CompletedProcess(args=args, returncode=0, stdout=stdout)

    monkeypatch.setattr(subprocess, "run", fake_subprocess_run)

    api = ReadOnlySiteApi(open_app(tmp_path), credential_provider=credentials)
    issue_response = api.handle(
        "POST",
        "/api/forge/issues/create",
        json.dumps(
            {
                "source_path": "issues/open/readonly-server.yaml",
                "confirm": True,
            }
        ),
    )
    pr_response = api.handle(
        "POST",
        "/api/forge/prs/create",
        json.dumps(
            {
                "base": "main",
                "head": "website-authenticated-forge-actions",
                "draft": True,
                "confirm": True,
            }
        ),
    )

    assert issue_response.status == 200
    assert issue_response.payload["kind"] == "github_issue_create"
    assert issue_response.payload["forge_action"]["github_issue_created"] is True
    assert pr_response.status == 200
    assert pr_response.payload["kind"] == "github_pr_create"
    assert pr_response.payload["forge_action"]["github_pr_created"] is True
    assert credentials.calls == 2
    assert calls[0][:3] == ["gh", "issue", "create"]
    assert calls[1][:3] == ["gh", "pr", "create"]

    response_text = json.dumps(
        [issue_response.payload, pr_response.payload],
        sort_keys=True,
    )
    assert credentials.secret not in response_text
    assert "token" not in response_text.lower()

    audit_path = tmp_path / ".cosheaf" / "audit" / "web-actions.jsonl"
    audit_text = audit_path.read_text(encoding="utf-8")
    assert credentials.secret not in audit_text
    entries = _audit_entries(tmp_path)
    assert [entry["action"] for entry in entries] == [
        "issue.publish_github",
        "forge.pr_create",
    ]
    assert [entry["credential_provider"] for entry in entries] == [
        "fake-provider",
        "fake-provider",
    ]
    assert all(entry["explicit_confirm"] is True for entry in entries)
    assert all(entry["github_writes_performed"] is True for entry in entries)
    assert entries[0]["github_urls"] == [
        "https://github.com/CheemsaDoge/tcs-cosheaf/issues/1001"
    ]
    assert entries[1]["github_urls"] == [
        "https://github.com/CheemsaDoge/tcs-cosheaf/pull/1002"
    ]
    assert entries[1]["base"] == "main"
    assert entries[1]["head"] == "website-authenticated-forge-actions"


def test_authenticated_create_failures_write_redacted_audit(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    _fixture_workspace(tmp_path)
    credentials = _FakeCredentialProvider()

    def fail_subprocess_run(
        args: list[str],
        **_kwargs: object,
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args=args,
            returncode=1,
            stdout="",
            stderr=f"gh failed with token {credentials.secret}",
        )

    monkeypatch.setattr(subprocess, "run", fail_subprocess_run)

    api = ReadOnlySiteApi(open_app(tmp_path), credential_provider=credentials)
    response = api.handle(
        "POST",
        "/api/forge/prs/create",
        json.dumps(
            {
                "base": "main",
                "head": "website-authenticated-forge-actions",
                "confirm": True,
            }
        ),
    )

    assert response.status == 400
    assert response.payload["code"] == "forge_github_failed"
    response_text = json.dumps(response.payload, sort_keys=True)
    assert credentials.secret not in response_text
    assert "token" not in response_text.lower()

    entries = _audit_entries(tmp_path)
    assert len(entries) == 1
    assert entries[0]["action"] == "forge.pr_create"
    assert entries[0]["result_status"] == "forge_github_failed"
    assert entries[0]["credential_provider"] == "fake-provider"
    assert entries[0]["explicit_confirm"] is True
    assert entries[0]["github_writes_performed"] is False


def test_http_preview_endpoints_allow_localhost_browser_preflight(
    tmp_path: Path,
) -> None:
    _fixture_workspace(tmp_path)
    api = ReadOnlySiteApi(open_app(tmp_path))
    server = HTTPServer((READONLY_SERVER_HOST, 0), make_handler(api))
    port = server.server_port
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        preflight = Request(
            f"http://{READONLY_SERVER_HOST}:{port}/api/forge/prs/preview",
            method="OPTIONS",
            headers={
                "Origin": "http://localhost:4321",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "content-type",
            },
        )
        with urlopen(preflight, timeout=5) as response:
            assert response.status == 204
            assert response.headers["Access-Control-Allow-Origin"] == (
                "http://localhost:4321"
            )
            assert "POST" in response.headers["Access-Control-Allow-Methods"]

        request = Request(
            f"http://{READONLY_SERVER_HOST}:{port}/api/forge/prs/preview",
            data=json.dumps(
                {"base": "main", "head": "website-preview-actions"}
            ).encode("utf-8"),
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Origin": "http://localhost:4321",
            },
        )
        with urlopen(request, timeout=5) as response:
            payload = json.loads(response.read().decode("utf-8"))
            assert response.status == 200
            assert response.headers["Access-Control-Allow-Origin"] == (
                "http://localhost:4321"
            )
            assert payload["kind"] == "github_pr_preview"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_server_cli_registers_readonly_localhost_command(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    fake_app = object()
    seen: dict[str, object] = {}

    def fake_open_app(repo_root: Path) -> object:
        seen["repo_root"] = repo_root
        return fake_app

    def fake_serve_readonly_api(
        app_obj: object,
        *,
        host: str,
        port: int,
        local_actor: str | None = None,
    ) -> None:
        seen["app"] = app_obj
        seen["host"] = host
        seen["port"] = port
        seen["local_actor"] = local_actor

    monkeypatch.setattr("cosheaf.server.cli.open_app", fake_open_app)
    monkeypatch.setattr(
        "cosheaf.server.cli.serve_readonly_api",
        fake_serve_readonly_api,
    )
    result = runner.invoke(
        app,
        [
            "server",
            "serve",
            "--readonly",
            "--port",
            "8765",
            "--repo-root",
            str(tmp_path),
            "--local-actor",
            "Ada Local",
        ],
    )

    assert result.exit_code == 0, result.output
    assert seen == {
        "repo_root": tmp_path,
        "app": fake_app,
        "host": "127.0.0.1",
        "port": 8765,
        "local_actor": "Ada Local",
    }


def test_server_cli_requires_readonly_guard(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "server",
            "serve",
            "--port",
            "8765",
            "--repo-root",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 1
    assert "pass --readonly" in result.output


def test_health_capabilities_and_strict_readonly_refuses_confirmed_writes(
    tmp_path: Path,
) -> None:
    _fixture_workspace(tmp_path)
    default_api = ReadOnlySiteApi(open_app(tmp_path))
    default_health = default_api.handle("GET", "/api/health")
    assert default_health.status == 200
    assert default_health.payload["strict_readonly"] is False
    assert default_health.payload["confirmed_writes_enabled"] is True
    assert default_health.payload["capabilities"]["confirmed_local_writes"] is True

    strict_api = ReadOnlySiteApi(open_app(tmp_path), strict_readonly=True)
    strict_health = strict_api.handle("GET", "/api/health")
    assert strict_health.status == 200
    assert strict_health.payload["strict_readonly"] is True
    assert strict_health.payload["confirmed_writes_enabled"] is False
    assert strict_health.payload["capabilities"]["confirmed_local_writes"] is False

    preview = strict_api.handle(
        "POST",
        "/api/context/issue.fixture.readonly-server/preview-build",
        json.dumps({"role": "orchestrator", "public_only": True}),
    )
    assert preview.status == 200

    blocked_write = strict_api.handle(
        "POST",
        "/api/context/issue.fixture.readonly-server/build",
        json.dumps(
            {
                "role": "orchestrator",
                "public_only": True,
                "confirm": True,
            }
        ),
    )
    assert blocked_write.status == 403
    assert blocked_write.payload["code"] == "readonly_mode_writes_disabled"

