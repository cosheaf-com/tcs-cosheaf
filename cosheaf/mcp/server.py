"""Minimal read-only MCP JSON-RPC surface for TCS-Cosheaf.

This module intentionally avoids an MCP SDK dependency for the first stdio
surface. It implements the protocol-level request shapes needed by tests and
external agents while delegating repository behavior to the shared service
layer.
"""

from __future__ import annotations

import json
import re
import sys
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TextIO

from pydantic import ValidationError

from cosheaf import __version__
from cosheaf.agent.context_pack import ContextPackError
from cosheaf.agent.orchestrator_planner import OrchestratorPlannerError
from cosheaf.agent.worker_bundle_v2 import WorkerBundleV2Error
from cosheaf.evals import (
    DEFAULT_RESEARCH_RUN_LOOP_EVAL_CASES,
    DEFAULT_STRATEGY_PLANNER_EVAL_CASES,
    ResearchRunLoopEvalError,
    StrategyPlannerEvalError,
    load_research_run_loop_eval_suite,
    load_strategy_planner_eval_suite,
    resolve_research_run_loop_eval_case_path,
    resolve_strategy_planner_eval_case_path,
    run_research_run_loop_eval_suite,
    run_strategy_planner_eval_suite,
)
from cosheaf.gates.gatekeeper import GatekeeperRunResult, ValidationReport
from cosheaf.memory import (
    ArtifactCard,
    ArtifactCardStatus,
    MemoryCardError,
    MemoryRootScope,
)
from cosheaf.memory import MemorySearchError as MemorySearchServiceError
from cosheaf.operator_session import (
    OperatorPolicyMode,
    OperatorSession,
    OperatorSessionError,
    OperatorToolCallRecord,
    OperatorToolCallStatus,
    append_operator_session_event,
    load_operator_session,
)
from cosheaf.research.run import (
    ResearchRunError,
    append_artifact_to_research_run,
    append_command_to_research_run,
    append_output_to_research_run,
    build_research_run_evidence_report,
    export_research_run_review,
    finalize_research_run,
    load_research_run,
    start_research_run,
)
from cosheaf.services import (
    BundleValidationService,
    ContextPackService,
    ControlledWriteResult,
    DraftWriteService,
    DraftWriteServiceError,
    GateService,
    MemorySearchService,
    OrchestratorPlanService,
    ServiceError,
    ValidationService,
    WorkspaceInfoResult,
    WorkspaceService,
)
from cosheaf.services.models import DraftArtifactWriteRequest, WorkerBundleSubmitRequest
from cosheaf.storage.loader import IssueRecord, LoadedRecord, LoadError, load_artifacts
from cosheaf.storage.repo import RepoContext
from cosheaf.strategy.models import (
    STRATEGY_AUTHORITY_NOTICE,
    StrategyError,
    StrategyPlan,
)
from cosheaf.strategy.planner import build_strategy_plan
from cosheaf.strategy.storage import (
    attach_context_reference,
    export_strategy_review,
    load_strategy_plan,
    update_strategy_plan_from_run,
    write_strategy_plan,
)
from cosheaf.verification.counterexample_evidence import (
    CHECKED_COUNTEREXAMPLE_AUTHORITY_NOTICE,
    CheckedCounterexampleEvidenceError,
    stage_checked_counterexample_evidence,
    validate_checked_counterexample_evidence_payload,
)

READ_ONLY_TOOL_NAMES = (
    "workspace_info",
    "validate",
    "gate",
    "gate_pr_checklist",
    "gate_run",
    "memory_cards",
    "memory_search",
    "context_build",
    "context_show",
    "strategy_plan",
    "strategy_show",
    "strategy_graph",
    "strategy_next",
    "run_show",
    "run_evidence_report",
    "eval_strategy_planner",
    "eval_research_run_loop",
    "draft_artifact_create_or_update",
    "source_note_draft_create",
    "worker_bundle_validate",
    "worker_bundle_stage",
    "review_request_from_bundle",
    "checked_counterexample_evidence_validate",
    "checked_counterexample_evidence_stage",
    "failure_log_add_draft",
    "research_run_start",
    "research_run_append_command",
    "research_run_append_artifact",
    "research_run_append_output",
    "research_run_finalize",
    "research_run_export_review_dry_run",
    "research_run_export_review",
    "strategy_update_from_run",
    "strategy_export_review_dry_run",
    "strategy_export_review",
    "orchestrator_plan",
)
READ_ONLY_PROMPT_NAMES = (
    "start_issue_work",
    "reason_about_issue",
    "verify_draft",
    "prepare_review_bundle",
    "public_kb_contribution_check",
)
PRIVATE_SCOPE_REMEDIATION = (
    "Private MCP resources require an explicit private research policy mode "
    "and operator consent; the current M.3 public surface does not grant that "
    "permission."
)
MCP_CONTROLLED_WRITE_AUTHORITY_NOTICE = (
    "Controlled MCP writes may create only draft, proposal, review-context, "
    "or runtime records allowed by Cosheaf policy. They do not write accepted "
    "knowledge, create human review, verify proofs, or promote artifacts."
)


class McpServerError(ValueError):
    """Expected JSON-RPC error for the read-only MCP surface."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        remediation: str,
        blocking: bool = True,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.remediation = remediation
        self.blocking = blocking
        self.details = dict(details or {})

    def to_error_data(self) -> dict[str, Any]:
        """Return deterministic machine-readable error details."""
        data: dict[str, Any] = {
            "code": self.code,
            "message": self.message,
            "remediation": self.remediation,
            "blocking": self.blocking,
        }
        if self.details:
            data["details"] = self.details
        return data


class ReadOnlyMcpServer:
    """Protocol-level read-only MCP handler over Cosheaf services."""

    def __init__(self, context: RepoContext) -> None:
        self.context = context
        self._tool_handlers: dict[
            str,
            Callable[[Mapping[str, Any]], dict[str, Any]],
        ] = {
            "workspace_info": self._tool_workspace_info,
            "validate": self._tool_validate,
            "gate": self._tool_gate,
            "gate_pr_checklist": self._tool_gate_pr_checklist,
            "gate_run": self._tool_gate_run,
            "memory_cards": self._tool_memory_cards,
            "memory_search": self._tool_memory_search,
            "context_build": self._tool_context_build,
            "context_show": self._tool_context_show,
            "strategy_plan": self._tool_strategy_plan,
            "strategy_show": self._tool_strategy_show,
            "strategy_graph": self._tool_strategy_graph,
            "strategy_next": self._tool_strategy_next,
            "run_show": self._tool_run_show,
            "run_evidence_report": self._tool_run_evidence_report,
            "eval_strategy_planner": self._tool_eval_strategy_planner,
            "eval_research_run_loop": self._tool_eval_research_run_loop,
            "draft_artifact_create_or_update": (
                self._tool_draft_artifact_create_or_update
            ),
            "source_note_draft_create": self._tool_source_note_draft_create,
            "worker_bundle_validate": self._tool_worker_bundle_validate,
            "worker_bundle_stage": self._tool_worker_bundle_stage,
            "review_request_from_bundle": self._tool_review_request_from_bundle,
            "checked_counterexample_evidence_validate": (
                self._tool_checked_counterexample_evidence_validate
            ),
            "checked_counterexample_evidence_stage": (
                self._tool_checked_counterexample_evidence_stage
            ),
            "failure_log_add_draft": self._tool_failure_log_add_draft,
            "research_run_start": self._tool_research_run_start,
            "research_run_append_command": self._tool_research_run_append_command,
            "research_run_append_artifact": self._tool_research_run_append_artifact,
            "research_run_append_output": self._tool_research_run_append_output,
            "research_run_finalize": self._tool_research_run_finalize,
            "research_run_export_review_dry_run": (
                self._tool_research_run_export_review_dry_run
            ),
            "research_run_export_review": self._tool_research_run_export_review,
            "strategy_update_from_run": self._tool_strategy_update_from_run,
            "strategy_export_review_dry_run": (
                self._tool_strategy_export_review_dry_run
            ),
            "strategy_export_review": self._tool_strategy_export_review,
            "orchestrator_plan": self._tool_orchestrator_plan,
        }

    def handle(self, request: Mapping[str, Any]) -> dict[str, Any]:
        """Handle one JSON-RPC request mapping and return one response mapping."""
        request_id = request.get("id")
        try:
            method = _required_string(request, "method")
            params = _optional_mapping(request.get("params", {}), field_name="params")
            result: dict[str, Any]
            if method == "initialize":
                result = {
                    "protocolVersion": "2024-11-05",
                    "serverInfo": {
                        "name": "tcs-cosheaf-readonly",
                        "version": __version__,
                    },
                    "capabilities": {
                        "tools": {},
                        "resources": {},
                    },
                }
            elif method == "tools/list":
                result = {"tools": tool_definitions()}
            elif method == "resources/list":
                result = {"resources": resource_definitions()}
            elif method == "prompts/list":
                result = {"prompts": prompt_definitions()}
            elif method == "prompts/get":
                result = self._handle_prompt_get(params)
            elif method == "tools/call":
                result = self._handle_tool_call(params)
            elif method == "resources/read":
                result = self._handle_resource_read(params)
            else:
                raise McpServerError(
                    "method_not_found",
                    f"unsupported MCP method: {method}",
                    remediation=(
                        "Use tools/list, tools/call, resources/list, "
                        "resources/read, prompts/list, or prompts/get."
                    ),
                )
            return _success_response(request_id, result)
        except McpServerError as exc:
            return _error_response(request_id, exc)
        except Exception as exc:
            error = McpServerError(
                "internal_error",
                "read-only MCP request failed",
                remediation="Inspect the repository state and rerun the request.",
                details={"exception": exc.__class__.__name__},
            )
            return _error_response(request_id, error)

    def _handle_tool_call(self, params: Mapping[str, Any]) -> dict[str, Any]:
        name = _required_string(params, "name")
        raw_arguments = _optional_mapping(
            params.get("arguments", {}),
            field_name="arguments",
        )
        session_id = _optional_string(
            raw_arguments.get("session_id"),
            field_name="session_id",
        )
        arguments = {
            str(key): value
            for key, value in raw_arguments.items()
            if key != "session_id"
        }
        session = self._load_recording_session(session_id) if session_id else None
        handler = self._tool_handlers.get(name)
        if handler is None:
            error = McpServerError(
                "tool_not_found",
                f"tool is not exposed by the read-only MCP server: {name}",
                remediation=(
                    "Call tools/list and use one of the whitelisted tool names."
                ),
            )
            if session is not None:
                self._record_mcp_tool_call(
                    session=session,
                    tool_name=name,
                    arguments=arguments,
                    status=OperatorToolCallStatus.DENIED,
                    result_summary=_mcp_error_summary(
                        name,
                        error,
                        status=OperatorToolCallStatus.DENIED,
                    ),
                    warning_codes=(error.code,),
                )
            raise error
        try:
            payload = handler(arguments)
            if session is not None:
                self._record_mcp_tool_call(
                    session=session,
                    tool_name=name,
                    arguments=arguments,
                    status=OperatorToolCallStatus.COMPLETED,
                    result_summary=_mcp_success_summary(name, payload),
                    warning_codes=_mcp_warning_codes(payload),
                )
            return _tool_result(payload)
        except McpServerError as exc:
            if session is not None:
                status = _mcp_error_status(exc)
                self._record_mcp_tool_call(
                    session=session,
                    tool_name=name,
                    arguments=arguments,
                    status=status,
                    result_summary=_mcp_error_summary(name, exc, status=status),
                    warning_codes=(exc.code,),
                )
            return _tool_error_result(exc)
        except Exception:
            if session is not None:
                self._record_mcp_tool_call(
                    session=session,
                    tool_name=name,
                    arguments=arguments,
                    status=OperatorToolCallStatus.ERROR,
                    result_summary=f"error tool call: {name}; code=internal_error",
                    warning_codes=("internal_error",),
                )
            raise

    def _load_recording_session(self, session_id: str) -> OperatorSession:
        try:
            return load_operator_session(self.context, session_id).session
        except OperatorSessionError as exc:
            raise McpServerError(
                exc.code,
                str(exc),
                remediation=exc.remediation,
                details=exc.details,
            ) from exc

    def _record_mcp_tool_call(
        self,
        *,
        session: OperatorSession,
        tool_name: str,
        arguments: Mapping[str, Any],
        status: OperatorToolCallStatus,
        result_summary: str,
        warning_codes: tuple[str, ...] = (),
    ) -> None:
        recorded_at = datetime.now(UTC).replace(microsecond=0)
        record = OperatorToolCallRecord(
            event_id=_mcp_event_id(tool_name, recorded_at),
            tool_name=tool_name,
            status=status,
            recorded_at=recorded_at,
            input_metadata=_mcp_input_metadata(
                session=session,
                arguments=arguments,
            ),
            result_summary=result_summary,
            warning_codes=warning_codes,
        )
        try:
            append_operator_session_event(
                self.context,
                session_id=session.session_id,
                event=record,
            )
        except OperatorSessionError as exc:
            raise McpServerError(
                exc.code,
                str(exc),
                remediation=exc.remediation,
                details=exc.details,
            ) from exc

    def _handle_prompt_get(self, params: Mapping[str, Any]) -> dict[str, Any]:
        name = _required_string(params, "name")
        arguments = _optional_mapping(
            params.get("arguments", {}),
            field_name="arguments",
        )
        if name not in READ_ONLY_PROMPT_NAMES:
            raise McpServerError(
                "prompt_not_found",
                f"prompt is not exposed by the read-only MCP server: {name}",
                remediation=(
                    "Call prompts/list and use one of the whitelisted prompt names."
                ),
            )
        issue_id = _optional_string(arguments.get("issue_id"), field_name="issue_id")
        artifact_id = _optional_string(
            arguments.get("artifact_id"),
            field_name="artifact_id",
        )
        return _prompt_get_result(
            name,
            issue_id=issue_id,
            artifact_id=artifact_id,
        )

    def _handle_resource_read(self, params: Mapping[str, Any]) -> dict[str, Any]:
        uri = _required_string(params, "uri")
        if uri == "cosheaf://workspace":
            return _resource_result(
                uri,
                _workspace_info_payload(WorkspaceService(self.context).info()),
            )
        if uri.startswith("cosheaf://issues/"):
            issue_id = uri.removeprefix("cosheaf://issues/")
            if "/" in issue_id or not issue_id:
                raise _resource_not_found(uri)
            return _resource_result(uri, self._issue_payload(issue_id))
        if uri.startswith("cosheaf://artifacts/public/") and uri.endswith("/card"):
            artifact_id = (
                uri.removeprefix("cosheaf://artifacts/public/")
                .removesuffix("/card")
            )
            if "/" in artifact_id or not artifact_id:
                raise _resource_not_found(uri)
            return _resource_result(
                uri,
                self._public_artifact_card_payload(artifact_id),
            )
        if uri.startswith("cosheaf://artifacts/private/") and uri.endswith("/card"):
            raise _private_resource_denied()
        if uri.startswith("cosheaf://artifacts/") and uri.endswith("/card"):
            artifact_id = uri.removeprefix("cosheaf://artifacts/").removesuffix("/card")
            if "/" in artifact_id or not artifact_id:
                raise _resource_not_found(uri)
            return _resource_result(
                uri,
                self._public_artifact_card_payload(artifact_id),
            )
        if uri.startswith("cosheaf://context/public/"):
            issue_id = uri.removeprefix("cosheaf://context/public/")
            if "/" in issue_id or not issue_id:
                raise _resource_not_found(uri)
            rendered = ContextPackService(self.context).show(issue_id, public_only=True)
            return _text_resource_result(uri, rendered, mime_type="text/markdown")
        if uri.startswith("cosheaf://context/private/"):
            raise _private_resource_denied()
        if uri.startswith("cosheaf://context/"):
            issue_id = uri.removeprefix("cosheaf://context/")
            if "/" in issue_id or not issue_id:
                raise _resource_not_found(uri)
            rendered = ContextPackService(self.context).show(issue_id, public_only=True)
            return _text_resource_result(uri, rendered, mime_type="text/markdown")
        if uri == "cosheaf://gate/latest":
            return _resource_result(uri, self._latest_gate_payload())
        raise _resource_not_found(uri)

    def _tool_workspace_info(self, _arguments: Mapping[str, Any]) -> dict[str, Any]:
        return _workspace_info_payload(WorkspaceService(self.context).info())

    def _tool_validate(self, _arguments: Mapping[str, Any]) -> dict[str, Any]:
        report = ValidationService(self.context).validate_repository()
        return _validation_payload(report)

    def _tool_gate(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        return self._run_gate_tool(arguments)

    def _tool_gate_pr_checklist(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        pr_checklist = _required_string(arguments, "pr_checklist")
        return self._run_gate_tool({"pr_checklist": pr_checklist})

    def _tool_gate_run(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        return self._run_gate_tool(arguments)

    def _run_gate_tool(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        pr_checklist = arguments.get("pr_checklist")
        if pr_checklist is not None and not isinstance(pr_checklist, str):
            raise McpServerError(
                "invalid_arguments",
                "pr_checklist must be a string path when provided",
                remediation="Pass a repository-local PR checklist path or omit it.",
            )
        result = GateService(self.context).run(pr_checklist_path=pr_checklist)
        return _gate_payload(self.context, result)

    def _tool_memory_cards(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        issue_id = _optional_string(arguments.get("issue_id"), field_name="issue_id")
        status = _optional_string(arguments.get("status"), field_name="status")
        status_filter = None
        if status is not None:
            try:
                status_filter = ArtifactCardStatus(status)
            except ValueError as exc:
                raise McpServerError(
                    "invalid_arguments",
                    f"unsupported artifact-card status: {status}",
                    remediation="Pass a supported artifact status or omit status.",
                ) from exc
        try:
            cards = MemorySearchService(self.context).cards(
                issue_id=issue_id,
                status=status_filter,
            )
        except MemoryCardError as exc:
            raise McpServerError(
                "memory_cards_failed",
                str(exc),
                remediation=(
                    "Check the issue ID, status filter, and repository records."
                ),
            ) from exc
        return {
            "cards": [_artifact_card_payload(card) for card in cards],
            "card_count": len(cards),
            "public_only": True,
            "accepted_writes": False,
        }

    def _tool_memory_search(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        query = _required_string(arguments, "query")
        issue_id = _optional_string(arguments.get("issue_id"), field_name="issue_id")
        max_cards = _optional_positive_int(
            arguments.get("max_cards"),
            field_name="max_cards",
            default=20,
        )
        try:
            result = MemorySearchService(self.context).search(
                query,
                max_cards=max_cards,
            )
        except MemorySearchServiceError as exc:
            raise McpServerError(
                "memory_search_failed",
                str(exc),
                remediation="Check the query, issue ID, and repository records.",
            ) from exc
        payload = result.to_dict()
        payload["public_only"] = True
        if issue_id is not None:
            payload["requested_issue_id"] = issue_id
            payload["issue_conditioning"] = "disabled_for_read_only_public_mcp"
        return payload

    def _tool_context_build(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        issue_id = _required_string(arguments, "issue_id")
        max_cards = _optional_positive_int(
            arguments.get("max_cards"),
            field_name="max_cards",
            default=20,
        )
        max_full_artifacts = _optional_non_negative_int(
            arguments.get("max_full_artifacts"),
            field_name="max_full_artifacts",
            default=0,
        )
        try:
            result = ContextPackService(self.context).build(
                issue_id,
                max_cards=max_cards,
                max_full_artifacts=max_full_artifacts,
                public_only=True,
            )
        except ContextPackError as exc:
            raise McpServerError(
                "context_build_failed",
                str(exc),
                remediation="Check the issue ID and repository records.",
            ) from exc
        return {
            "issue_id": result.issue_id,
            "task_dir": _repo_relative_or_string(self.context, result.task_dir),
            "files": [
                _repo_relative_or_string(self.context, path)
                for path in result.files
            ],
            "public_only": True,
            "accepted_writes": False,
        }

    def _tool_context_show(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        issue_id = _required_string(arguments, "issue_id")
        try:
            rendered = ContextPackService(self.context).show(issue_id, public_only=True)
        except ContextPackError as exc:
            raise McpServerError(
                "context_show_failed",
                str(exc),
                remediation="Check the issue ID and repository records.",
            ) from exc
        return {
            "issue_id": issue_id,
            "public_only": True,
            "content": rendered,
        }

    def _tool_strategy_plan(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        issue_id = _required_string(arguments, "issue_id")
        from_context = _optional_string(
            arguments.get("from_context"),
            field_name="from_context",
        )
        try:
            built = build_strategy_plan(self.context, issue_id)
            plan = (
                attach_context_reference(self.context, built.plan, Path(from_context))
                if from_context is not None
                else built.plan
            )
            public_plan = _public_strategy_plan(self.context, plan)
            result = write_strategy_plan(self.context, public_plan)
        except (StrategyError, LoadError, ValueError) as exc:
            raise McpServerError(
                _strategy_error_code(exc),
                str(exc),
                remediation=_strategy_error_remediation(exc),
            ) from exc
        payload = _strategy_plan_payload(result.plan, result.relative_path)
        payload["public_only"] = True
        return payload

    def _tool_strategy_show(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        plan_id = _required_string(arguments, "plan_id")
        try:
            result = load_strategy_plan(self.context, plan_id)
            plan = _public_strategy_plan(self.context, result.plan)
        except (StrategyError, ValueError) as exc:
            raise McpServerError(
                _strategy_error_code(exc),
                str(exc),
                remediation=_strategy_error_remediation(exc),
            ) from exc
        payload = _strategy_plan_payload(plan, result.relative_path)
        payload["public_only"] = True
        return payload

    def _tool_strategy_graph(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        plan_id = _required_string(arguments, "plan_id")
        try:
            result = load_strategy_plan(self.context, plan_id)
            plan = _public_strategy_plan(self.context, result.plan)
        except (StrategyError, ValueError) as exc:
            raise McpServerError(
                _strategy_error_code(exc),
                str(exc),
                remediation=_strategy_error_remediation(exc),
            ) from exc
        return {
            "schema_version": 1,
            "kind": "strategy_task_graph",
            "plan_id": plan.plan_id,
            "accepted_write_performed": False,
            "authority_notice": STRATEGY_AUTHORITY_NOTICE,
            "graph": plan.graph.to_dict(),
            "public_only": True,
        }

    def _tool_strategy_next(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        plan_id = _required_string(arguments, "plan_id")
        try:
            result = load_strategy_plan(self.context, plan_id)
            plan = _public_strategy_plan(self.context, result.plan)
        except (StrategyError, ValueError) as exc:
            raise McpServerError(
                _strategy_error_code(exc),
                str(exc),
                remediation=_strategy_error_remediation(exc),
            ) from exc
        return {
            "schema_version": 1,
            "kind": "strategy_next_steps",
            "plan_id": plan.plan_id,
            "accepted_write_performed": False,
            "authority_notice": STRATEGY_AUTHORITY_NOTICE,
            "next_steps": [step.to_dict() for step in plan.next_steps],
            "public_only": True,
        }

    def _tool_run_show(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        run_id = _required_string(arguments, "run_id")
        try:
            result = load_research_run(self.context, run_id)
        except (ResearchRunError, ValueError) as exc:
            raise McpServerError(
                _research_run_error_code(exc),
                str(exc),
                remediation=_research_run_error_remediation(exc),
            ) from exc
        payload = result.to_dict()
        payload["read_only"] = True
        return payload

    def _tool_run_evidence_report(
        self,
        arguments: Mapping[str, Any],
    ) -> dict[str, Any]:
        run_id = _required_string(arguments, "run_id")
        try:
            result = load_research_run(self.context, run_id)
        except (ResearchRunError, ValueError) as exc:
            raise McpServerError(
                _research_run_error_code(exc),
                str(exc),
                remediation=_research_run_error_remediation(exc),
            ) from exc
        payload = build_research_run_evidence_report(result.record)
        payload["read_only"] = True
        return payload

    def _tool_eval_strategy_planner(
        self,
        arguments: Mapping[str, Any],
    ) -> dict[str, Any]:
        cases = _optional_string(arguments.get("cases"), field_name="cases")
        case_path = (
            Path(cases) if cases is not None else DEFAULT_STRATEGY_PLANNER_EVAL_CASES
        )
        try:
            resolved = resolve_strategy_planner_eval_case_path(self.context, case_path)
            suite = load_strategy_planner_eval_suite(resolved)
            report = run_strategy_planner_eval_suite(self.context, suite)
        except StrategyPlannerEvalError as exc:
            raise McpServerError(
                "strategy_planner_eval_failed",
                str(exc),
                remediation="Check the repository-local strategy-planner eval cases.",
            ) from exc
        payload = report.model_dump(mode="json")
        payload["accepted_write_performed"] = False
        return payload

    def _tool_eval_research_run_loop(
        self,
        arguments: Mapping[str, Any],
    ) -> dict[str, Any]:
        cases = _optional_string(arguments.get("cases"), field_name="cases")
        case_path = (
            Path(cases) if cases is not None else DEFAULT_RESEARCH_RUN_LOOP_EVAL_CASES
        )
        try:
            resolved = resolve_research_run_loop_eval_case_path(self.context, case_path)
            suite = load_research_run_loop_eval_suite(resolved)
            report = run_research_run_loop_eval_suite(self.context, suite)
        except ResearchRunLoopEvalError as exc:
            raise McpServerError(
                "research_run_loop_eval_failed",
                str(exc),
                remediation="Check the repository-local research-run eval cases.",
            ) from exc
        payload = report.model_dump(mode="json")
        payload["accepted_write_performed"] = False
        return payload

    def _tool_draft_artifact_create_or_update(
        self,
        arguments: Mapping[str, Any],
    ) -> dict[str, Any]:
        request = dict(_required_mapping(arguments, "request"))
        dry_run = _optional_bool(arguments.get("dry_run"), field_name="dry_run")
        if str(request.get("status", "")).strip() == "accepted":
            raise McpServerError(
                "accepted_write_forbidden",
                "draft artifact MCP writes cannot target accepted knowledge",
                remediation=(
                    "Use draft status and the explicit promotion workflow after "
                    "review and gates."
                ),
            )
        try:
            parsed = DraftArtifactWriteRequest.model_validate(request)
            result = DraftWriteService(self.context).write_artifact_request(
                parsed,
                dry_run=dry_run,
            )
        except (DraftWriteServiceError, ValidationError) as exc:
            raise _mcp_error_from_exception(
                exc,
                default_code="draft_write_failed",
                default_remediation="Fix the draft artifact request and retry.",
            ) from exc
        return _controlled_write_payload(result)

    def _tool_source_note_draft_create(
        self,
        arguments: Mapping[str, Any],
    ) -> dict[str, Any]:
        request = dict(_required_mapping(arguments, "request"))
        dry_run = _optional_bool(arguments.get("dry_run"), field_name="dry_run")
        try:
            result = DraftWriteService(self.context).write_source_note(
                request,
                dry_run=dry_run,
            )
        except DraftWriteServiceError as exc:
            raise _mcp_error_from_exception(
                exc,
                default_code="source_note_write_failed",
                default_remediation="Fix the source-note request and retry.",
            ) from exc
        return _controlled_write_payload(result)

    def _tool_worker_bundle_validate(
        self,
        arguments: Mapping[str, Any],
    ) -> dict[str, Any]:
        bundle_path = _required_string(arguments, "bundle_path")
        try:
            bundle = BundleValidationService(self.context).validate(bundle_path)
        except WorkerBundleV2Error as exc:
            raise McpServerError(
                "worker_bundle_validate_failed",
                str(exc),
                remediation=(
                    "Fix the WorkerBundle v2 manifest and keep outputs "
                    "review-only."
                ),
            ) from exc
        return {
            "schema_version": 1,
            "kind": "worker_bundle_validation",
            "bundle_id": bundle.bundle_id,
            "task_id": bundle.task_id,
            "accepted_write_performed": False,
            "authority_notice": MCP_CONTROLLED_WRITE_AUTHORITY_NOTICE,
            "bundle": bundle.to_dict(),
        }

    def _tool_worker_bundle_stage(
        self,
        arguments: Mapping[str, Any],
    ) -> dict[str, Any]:
        payload = {
            "task_id": _required_string(arguments, "task_id"),
            "bundle_path": _required_string(arguments, "bundle_path"),
            "complete_task": _optional_bool(
                arguments.get("complete_task"),
                field_name="complete_task",
            ),
        }
        dry_run = _optional_bool(arguments.get("dry_run"), field_name="dry_run")
        try:
            request = WorkerBundleSubmitRequest.model_validate(payload)
            result = BundleValidationService(self.context).submit(
                request,
                dry_run=dry_run,
            )
        except (ServiceError, ValidationError) as exc:
            raise _mcp_error_from_exception(
                exc,
                default_code="bundle_submit_failed",
                default_remediation="Fix the WorkerBundle stage request and retry.",
            ) from exc
        response = result.to_dict()
        response["accepted_write_performed"] = False
        response["authority_notice"] = MCP_CONTROLLED_WRITE_AUTHORITY_NOTICE
        response["dry_run"] = dry_run
        return response

    def _tool_review_request_from_bundle(
        self,
        arguments: Mapping[str, Any],
    ) -> dict[str, Any]:
        bundle_path = _required_string(arguments, "bundle_path")
        dry_run = _optional_bool(arguments.get("dry_run"), field_name="dry_run")
        try:
            result = DraftWriteService(
                self.context
            ).write_review_request_from_bundle(
                bundle_path,
                dry_run=dry_run,
            )
        except DraftWriteServiceError as exc:
            raise _mcp_error_from_exception(
                exc,
                default_code="review_request_failed",
                default_remediation="Fix the WorkerBundle before review staging.",
            ) from exc
        write = result.write_result
        payload = _controlled_write_payload(write)
        payload.update(
            {
                "bundle_id": result.bundle.bundle_id,
                "task_id": result.bundle.task_id,
                "review_id": write.record_id,
                "generated_request": dict(result.request),
            }
        )
        return payload

    def _tool_checked_counterexample_evidence_validate(
        self,
        arguments: Mapping[str, Any],
    ) -> dict[str, Any]:
        evidence = dict(_required_mapping(arguments, "evidence"))
        try:
            record = validate_checked_counterexample_evidence_payload(evidence)
        except (CheckedCounterexampleEvidenceError, ValidationError, ValueError) as exc:
            raise _mcp_checked_evidence_error(exc) from exc
        return {
            "schema_version": 1,
            "kind": "checked_counterexample_evidence_validation",
            "valid": True,
            "accepted_write_performed": False,
            "authority_notice": CHECKED_COUNTEREXAMPLE_AUTHORITY_NOTICE,
            "evidence": record.to_dict(),
        }

    def _tool_checked_counterexample_evidence_stage(
        self,
        arguments: Mapping[str, Any],
    ) -> dict[str, Any]:
        evidence = dict(_required_mapping(arguments, "evidence"))
        dry_run = _optional_bool(arguments.get("dry_run"), field_name="dry_run")
        try:
            result = stage_checked_counterexample_evidence(
                self.context,
                evidence,
                dry_run=dry_run,
            )
        except (CheckedCounterexampleEvidenceError, ValidationError, ValueError) as exc:
            raise _mcp_checked_evidence_error(exc) from exc
        return result.to_dict()

    def _tool_failure_log_add_draft(
        self,
        arguments: Mapping[str, Any],
    ) -> dict[str, Any]:
        artifact_id = _required_string(arguments, "artifact_id")
        entry = dict(_required_mapping(arguments, "entry"))
        dry_run = _optional_bool(arguments.get("dry_run"), field_name="dry_run")
        try:
            result = DraftWriteService(self.context).append_failure_log_entry(
                artifact_id,
                entry,
                dry_run=dry_run,
            )
        except DraftWriteServiceError as exc:
            raise _mcp_error_from_exception(
                exc,
                default_code="draft_write_failed",
                default_remediation="Fix the failure-log entry and target artifact.",
            ) from exc
        return _controlled_write_payload(result)

    def _tool_research_run_start(
        self,
        arguments: Mapping[str, Any],
    ) -> dict[str, Any]:
        try:
            result = start_research_run(
                self.context,
                issue_id=_required_string(arguments, "issue_id"),
                operator_kind=_required_string(arguments, "operator_kind"),
                operator_label=_required_string(arguments, "operator_label"),
                run_id=_optional_string(arguments.get("run_id"), field_name="run_id"),
            )
        except (ResearchRunError, ValueError) as exc:
            raise _mcp_research_run_error(exc) from exc
        return result.to_dict()

    def _tool_research_run_append_command(
        self,
        arguments: Mapping[str, Any],
    ) -> dict[str, Any]:
        run_id = _required_string(arguments, "run_id")
        command = dict(_required_mapping(arguments, "command"))
        try:
            result = append_command_to_research_run(
                self.context,
                run_id=run_id,
                payload=command,
            )
        except (ResearchRunError, ValueError) as exc:
            raise _mcp_research_run_error(exc) from exc
        return result.to_dict()

    def _tool_research_run_append_artifact(
        self,
        arguments: Mapping[str, Any],
    ) -> dict[str, Any]:
        try:
            result = append_artifact_to_research_run(
                self.context,
                run_id=_required_string(arguments, "run_id"),
                artifact_id=_required_string(arguments, "artifact_id"),
                mode=_required_string(arguments, "mode"),
            )
        except (ResearchRunError, ValueError) as exc:
            raise _mcp_research_run_error(exc) from exc
        return result.to_dict()

    def _tool_research_run_append_output(
        self,
        arguments: Mapping[str, Any],
    ) -> dict[str, Any]:
        run_id = _required_string(arguments, "run_id")
        output = dict(_required_mapping(arguments, "output"))
        try:
            result = append_output_to_research_run(
                self.context,
                run_id=run_id,
                payload=output,
            )
        except (ResearchRunError, ValueError) as exc:
            raise _mcp_research_run_error(exc) from exc
        return result.to_dict()

    def _tool_research_run_finalize(
        self,
        arguments: Mapping[str, Any],
    ) -> dict[str, Any]:
        try:
            result = finalize_research_run(
                self.context,
                run_id=_required_string(arguments, "run_id"),
                status=_required_string(arguments, "status"),
                stop_reason=_required_string(arguments, "stop_reason"),
            )
        except (ResearchRunError, ValueError) as exc:
            raise _mcp_research_run_error(exc) from exc
        return result.to_dict()

    def _tool_research_run_export_review_dry_run(
        self,
        arguments: Mapping[str, Any],
    ) -> dict[str, Any]:
        return self._research_run_export_review(arguments, dry_run=True)

    def _tool_research_run_export_review(
        self,
        arguments: Mapping[str, Any],
    ) -> dict[str, Any]:
        return self._research_run_export_review(arguments, dry_run=False)

    def _research_run_export_review(
        self,
        arguments: Mapping[str, Any],
        *,
        dry_run: bool,
    ) -> dict[str, Any]:
        try:
            result = export_research_run_review(
                self.context,
                run_id=_required_string(arguments, "run_id"),
                dry_run=dry_run,
            )
        except (ResearchRunError, ValueError) as exc:
            raise _mcp_research_run_error(exc) from exc
        return result.to_dict()

    def _tool_strategy_update_from_run(
        self,
        arguments: Mapping[str, Any],
    ) -> dict[str, Any]:
        try:
            result = update_strategy_plan_from_run(
                self.context,
                plan_id=_required_string(arguments, "plan_id"),
                run_id=_required_string(arguments, "run_id"),
            )
        except (StrategyError, ValueError) as exc:
            raise McpServerError(
                _strategy_error_code(exc),
                str(exc),
                remediation=_strategy_error_remediation(exc),
            ) from exc
        payload = result.to_dict()
        payload["plan"] = _public_strategy_plan(self.context, result.plan).to_dict()
        payload["public_only"] = True
        return payload

    def _tool_strategy_export_review_dry_run(
        self,
        arguments: Mapping[str, Any],
    ) -> dict[str, Any]:
        return self._strategy_export_review(arguments, dry_run=True)

    def _tool_strategy_export_review(
        self,
        arguments: Mapping[str, Any],
    ) -> dict[str, Any]:
        return self._strategy_export_review(arguments, dry_run=False)

    def _strategy_export_review(
        self,
        arguments: Mapping[str, Any],
        *,
        dry_run: bool,
    ) -> dict[str, Any]:
        try:
            result = export_strategy_review(
                self.context,
                plan_id=_required_string(arguments, "plan_id"),
                dry_run=dry_run,
            )
        except (StrategyError, ValueError) as exc:
            raise McpServerError(
                _strategy_error_code(exc),
                str(exc),
                remediation=_strategy_error_remediation(exc),
            ) from exc
        payload = result.to_dict()
        payload["plan"] = _public_strategy_plan(self.context, result.plan).to_dict()
        payload["public_only"] = True
        return payload

    def _tool_orchestrator_plan(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        issue_id = _required_string(arguments, "issue_id")
        try:
            plan = OrchestratorPlanService(self.context).plan_for_issue(issue_id)
        except OrchestratorPlannerError as exc:
            raise McpServerError(
                "orchestrator_plan_failed",
                str(exc),
                remediation="Check the issue ID and repository records.",
            ) from exc
        return {
            **plan.to_dict(),
            "execution_performed": False,
            "hosted_provider_called": False,
            "accepted_writes": False,
        }

    def _issue_payload(self, issue_id: str) -> dict[str, Any]:
        issue = _find_issue(self.context, issue_id)
        public_card_ids = {
            card.id
            for card in MemorySearchService(self.context).cards(issue_id=issue_id)
            if card.root_scope is not MemoryRootScope.PRIVATE
        }
        related_artifacts = [
            artifact_id
            for artifact_id in issue.related_artifacts
            if artifact_id in public_card_ids
        ]
        return {
            "id": issue.id,
            "type": issue.type,
            "title": issue.title,
            "status": issue.status,
            "severity": issue.severity,
            "description": issue.description,
            "related_artifacts": related_artifacts,
            "tags": issue.tags,
            "public_only": True,
        }

    def _public_artifact_card_payload(self, artifact_id: str) -> dict[str, Any]:
        try:
            cards = MemorySearchService(self.context).cards()
        except MemoryCardError as exc:
            raise McpServerError(
                "artifact_card_failed",
                str(exc),
                remediation="Fix repository records and retry.",
            ) from exc
        matches = [card for card in cards if card.id == artifact_id]
        if matches:
            return _artifact_card_payload(matches[0])
        if _artifact_exists_with_private_scope(self.context, artifact_id):
            raise McpServerError(
                "private_resource_denied",
                "private artifact card is not available through read-only "
                "public MCP resources",
                remediation=(
                    "Use an explicit private research policy mode in a later "
                    "controlled MCP surface; M.2 exposes public resources only."
                ),
            )
        raise McpServerError(
            "resource_not_found",
            "artifact card resource was not found",
            remediation="Check the artifact ID and visible public KB scope.",
        )

    def _latest_gate_payload(self) -> dict[str, Any]:
        reports_dir = self.context.resolve(Path(".cosheaf") / "reports")
        candidates = sorted(reports_dir.glob("*-gate-report.json"))
        if not candidates:
            raise McpServerError(
                "gate_report_not_found",
                "no gate report exists under .cosheaf/reports",
                remediation="Run the gate_run tool or cosheaf gate run first.",
                blocking=False,
            )
        latest = candidates[-1]
        try:
            payload = json.loads(latest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise McpServerError(
                "gate_report_unreadable",
                "latest gate report could not be read",
                remediation="Rerun gate_run to regenerate the report.",
                details={"path": _repo_relative_or_string(self.context, latest)},
            ) from exc
        return {
            "path": _repo_relative_or_string(self.context, latest),
            "report": payload,
        }


def tool_definitions() -> list[dict[str, Any]]:
    """Return deterministic read-only MCP tool definitions."""
    descriptions = {
        "workspace_info": "Return configured workspace and KB root metadata.",
        "validate": "Run repository validation and return structured failures.",
        "gate": "Run gatekeeper and return report metadata.",
        "gate_pr_checklist": "Run gatekeeper with a repository-local PR checklist.",
        "gate_run": "Run gatekeeper and return report metadata.",
        "memory_cards": "List public compact artifact cards.",
        "memory_search": (
            "Search public artifact cards with deterministic local scoring."
        ),
        "context_build": "Build a public-only issue context pack.",
        "context_show": "Build and return public-only issue context text.",
        "strategy_plan": "Build a public-scoped runtime strategy plan.",
        "strategy_show": "Read one public-scoped runtime strategy plan.",
        "strategy_graph": "Read one public-scoped strategy task graph.",
        "strategy_next": "Read public-scoped strategy next-step guidance.",
        "run_show": "Read one research-run provenance record.",
        "run_evidence_report": "Return read-only evidence counts for a run.",
        "eval_strategy_planner": "Run deterministic strategy-planner eval smoke.",
        "eval_research_run_loop": "Run deterministic research-run loop eval smoke.",
        "draft_artifact_create_or_update": (
            "Create or preview one controlled draft artifact write."
        ),
        "source_note_draft_create": (
            "Create or preview one draft source-note record."
        ),
        "worker_bundle_validate": (
            "Validate one WorkerBundle v2 manifest for review."
        ),
        "worker_bundle_stage": (
            "Stage a WorkerBundle v2 manifest for review without promotion."
        ),
        "review_request_from_bundle": (
            "Create or preview a draft review request from a WorkerBundle."
        ),
        "checked_counterexample_evidence_validate": (
            "Validate checked counterexample evidence without writing."
        ),
        "checked_counterexample_evidence_stage": (
            "Stage or preview checked counterexample evidence for review."
        ),
        "failure_log_add_draft": (
            "Append or preview one failure-log entry on a draft artifact."
        ),
        "research_run_start": "Start one repository-local research-run record.",
        "research_run_append_command": (
            "Append one command provenance record to a research run."
        ),
        "research_run_append_artifact": (
            "Append one artifact read/touched marker to a research run."
        ),
        "research_run_append_output": (
            "Append one output/reference record to a research run."
        ),
        "research_run_finalize": "Finalize one repository-local research run.",
        "research_run_export_review_dry_run": (
            "Preview a research-run review export without writing."
        ),
        "research_run_export_review": (
            "Export one research run as non-authoritative review context."
        ),
        "strategy_update_from_run": (
            "Update a runtime strategy plan from research-run provenance."
        ),
        "strategy_export_review_dry_run": (
            "Preview a strategy-plan review export without writing."
        ),
        "strategy_export_review": (
            "Export one strategy plan as non-authoritative review context."
        ),
        "orchestrator_plan": "Create a deterministic plan without executing workers.",
    }
    schemas: dict[str, dict[str, Any]] = {
        "workspace_info": {"type": "object", "additionalProperties": False},
        "validate": {"type": "object", "additionalProperties": False},
        "gate": {
            "type": "object",
            "properties": {"pr_checklist": {"type": "string"}},
            "additionalProperties": False,
        },
        "gate_pr_checklist": {
            "type": "object",
            "properties": {"pr_checklist": {"type": "string"}},
            "required": ["pr_checklist"],
            "additionalProperties": False,
        },
        "gate_run": {
            "type": "object",
            "properties": {"pr_checklist": {"type": "string"}},
            "additionalProperties": False,
        },
        "memory_cards": {
            "type": "object",
            "properties": {
                "issue_id": {"type": "string"},
                "status": {"type": "string"},
            },
            "additionalProperties": False,
        },
        "memory_search": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "issue_id": {"type": "string"},
                "max_cards": {"type": "integer", "minimum": 1},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
        "context_build": {
            "type": "object",
            "properties": {
                "issue_id": {"type": "string"},
                "max_cards": {"type": "integer", "minimum": 1},
                "max_full_artifacts": {"type": "integer", "minimum": 0},
            },
            "required": ["issue_id"],
            "additionalProperties": False,
        },
        "context_show": {
            "type": "object",
            "properties": {"issue_id": {"type": "string"}},
            "required": ["issue_id"],
            "additionalProperties": False,
        },
        "strategy_plan": {
            "type": "object",
            "properties": {
                "issue_id": {"type": "string"},
                "from_context": {"type": "string"},
            },
            "required": ["issue_id"],
            "additionalProperties": False,
        },
        "strategy_show": {
            "type": "object",
            "properties": {"plan_id": {"type": "string"}},
            "required": ["plan_id"],
            "additionalProperties": False,
        },
        "strategy_graph": {
            "type": "object",
            "properties": {"plan_id": {"type": "string"}},
            "required": ["plan_id"],
            "additionalProperties": False,
        },
        "strategy_next": {
            "type": "object",
            "properties": {"plan_id": {"type": "string"}},
            "required": ["plan_id"],
            "additionalProperties": False,
        },
        "run_show": {
            "type": "object",
            "properties": {"run_id": {"type": "string"}},
            "required": ["run_id"],
            "additionalProperties": False,
        },
        "run_evidence_report": {
            "type": "object",
            "properties": {"run_id": {"type": "string"}},
            "required": ["run_id"],
            "additionalProperties": False,
        },
        "eval_strategy_planner": {
            "type": "object",
            "properties": {"cases": {"type": "string"}},
            "additionalProperties": False,
        },
        "eval_research_run_loop": {
            "type": "object",
            "properties": {"cases": {"type": "string"}},
            "additionalProperties": False,
        },
        "draft_artifact_create_or_update": {
            "type": "object",
            "properties": {
                "request": {"type": "object"},
                "dry_run": {"type": "boolean"},
            },
            "required": ["request"],
            "additionalProperties": False,
        },
        "source_note_draft_create": {
            "type": "object",
            "properties": {
                "request": {"type": "object"},
                "dry_run": {"type": "boolean"},
            },
            "required": ["request"],
            "additionalProperties": False,
        },
        "worker_bundle_validate": {
            "type": "object",
            "properties": {"bundle_path": {"type": "string"}},
            "required": ["bundle_path"],
            "additionalProperties": False,
        },
        "worker_bundle_stage": {
            "type": "object",
            "properties": {
                "task_id": {"type": "string"},
                "bundle_path": {"type": "string"},
                "complete_task": {"type": "boolean"},
                "dry_run": {"type": "boolean"},
            },
            "required": ["task_id", "bundle_path"],
            "additionalProperties": False,
        },
        "review_request_from_bundle": {
            "type": "object",
            "properties": {
                "bundle_path": {"type": "string"},
                "dry_run": {"type": "boolean"},
            },
            "required": ["bundle_path"],
            "additionalProperties": False,
        },
        "checked_counterexample_evidence_validate": {
            "type": "object",
            "properties": {"evidence": {"type": "object"}},
            "required": ["evidence"],
            "additionalProperties": False,
        },
        "checked_counterexample_evidence_stage": {
            "type": "object",
            "properties": {
                "evidence": {"type": "object"},
                "dry_run": {"type": "boolean"},
            },
            "required": ["evidence"],
            "additionalProperties": False,
        },
        "failure_log_add_draft": {
            "type": "object",
            "properties": {
                "artifact_id": {"type": "string"},
                "entry": {"type": "object"},
                "dry_run": {"type": "boolean"},
            },
            "required": ["artifact_id", "entry"],
            "additionalProperties": False,
        },
        "research_run_start": {
            "type": "object",
            "properties": {
                "issue_id": {"type": "string"},
                "operator_kind": {"type": "string"},
                "operator_label": {"type": "string"},
                "run_id": {"type": "string"},
            },
            "required": ["issue_id", "operator_kind", "operator_label"],
            "additionalProperties": False,
        },
        "research_run_append_command": {
            "type": "object",
            "properties": {
                "run_id": {"type": "string"},
                "command": {"type": "object"},
            },
            "required": ["run_id", "command"],
            "additionalProperties": False,
        },
        "research_run_append_artifact": {
            "type": "object",
            "properties": {
                "run_id": {"type": "string"},
                "artifact_id": {"type": "string"},
                "mode": {"type": "string"},
            },
            "required": ["run_id", "artifact_id", "mode"],
            "additionalProperties": False,
        },
        "research_run_append_output": {
            "type": "object",
            "properties": {
                "run_id": {"type": "string"},
                "output": {"type": "object"},
            },
            "required": ["run_id", "output"],
            "additionalProperties": False,
        },
        "research_run_finalize": {
            "type": "object",
            "properties": {
                "run_id": {"type": "string"},
                "status": {"type": "string"},
                "stop_reason": {"type": "string"},
            },
            "required": ["run_id", "status", "stop_reason"],
            "additionalProperties": False,
        },
        "research_run_export_review_dry_run": {
            "type": "object",
            "properties": {"run_id": {"type": "string"}},
            "required": ["run_id"],
            "additionalProperties": False,
        },
        "research_run_export_review": {
            "type": "object",
            "properties": {"run_id": {"type": "string"}},
            "required": ["run_id"],
            "additionalProperties": False,
        },
        "strategy_update_from_run": {
            "type": "object",
            "properties": {
                "plan_id": {"type": "string"},
                "run_id": {"type": "string"},
            },
            "required": ["plan_id", "run_id"],
            "additionalProperties": False,
        },
        "strategy_export_review_dry_run": {
            "type": "object",
            "properties": {"plan_id": {"type": "string"}},
            "required": ["plan_id"],
            "additionalProperties": False,
        },
        "strategy_export_review": {
            "type": "object",
            "properties": {"plan_id": {"type": "string"}},
            "required": ["plan_id"],
            "additionalProperties": False,
        },
        "orchestrator_plan": {
            "type": "object",
            "properties": {"issue_id": {"type": "string"}},
            "required": ["issue_id"],
            "additionalProperties": False,
        },
    }
    return [
        {
            "name": name,
            "description": descriptions[name],
            "inputSchema": _tool_schema_with_optional_session_id(schemas[name]),
        }
        for name in READ_ONLY_TOOL_NAMES
    ]


def prompt_definitions() -> list[dict[str, Any]]:
    """Return deterministic governance-safe MCP prompt definitions."""
    descriptions = {
        "start_issue_work": "Start bounded issue work from public context.",
        "reason_about_issue": "Reason about an issue without changing knowledge state.",
        "verify_draft": "Check a draft or proposal before review.",
        "prepare_review_bundle": "Prepare review material without claiming review.",
        "public_kb_contribution_check": (
            "Check whether a public KB contribution is policy-ready."
        ),
    }
    return [
        {
            "name": name,
            "description": descriptions[name],
            "arguments": [
                {
                    "name": "issue_id",
                    "description": "Optional issue ID to keep work scoped.",
                    "required": False,
                },
                {
                    "name": "artifact_id",
                    "description": "Optional artifact ID to keep work scoped.",
                    "required": False,
                },
            ],
        }
        for name in READ_ONLY_PROMPT_NAMES
    ]


def resource_definitions() -> list[dict[str, str]]:
    """Return deterministic read-only MCP resource examples."""
    return [
        {
            "uri": "cosheaf://workspace",
            "name": "workspace",
            "mimeType": "application/json",
        },
        {
            "uri": "cosheaf://issues/{issue_id}",
            "name": "issue",
            "mimeType": "application/json",
        },
        {
            "uri": "cosheaf://artifacts/public/{artifact_id}/card",
            "name": "public artifact card",
            "mimeType": "application/json",
        },
        {
            "uri": "cosheaf://artifacts/private/{artifact_id}/card",
            "name": "private artifact card",
            "mimeType": "application/json",
        },
        {
            "uri": "cosheaf://context/public/{issue_id}",
            "name": "public context pack",
            "mimeType": "text/markdown",
        },
        {
            "uri": "cosheaf://context/private/{issue_id}",
            "name": "private context pack",
            "mimeType": "text/markdown",
        },
        {
            "uri": "cosheaf://gate/latest",
            "name": "latest gate report",
            "mimeType": "application/json",
        },
    ]


def _tool_schema_with_optional_session_id(schema: Mapping[str, Any]) -> dict[str, Any]:
    """Return a copied tool schema with optional session recording metadata."""
    result = dict(schema)
    properties = dict(result.get("properties", {}))
    properties["session_id"] = {
        "type": "string",
        "description": "Optional operator-session ID for bounded tool-call recording.",
    }
    result["properties"] = properties
    return result


def _prompt_get_result(
    name: str,
    *,
    issue_id: str | None,
    artifact_id: str | None,
) -> dict[str, Any]:
    text = _render_prompt(name, issue_id=issue_id, artifact_id=artifact_id)
    return {
        "description": f"TCS-Cosheaf prompt: {name}",
        "messages": [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": text,
                },
            }
        ],
    }


def _render_prompt(
    name: str,
    *,
    issue_id: str | None,
    artifact_id: str | None,
) -> str:
    scope_lines = [
        f"- Issue ID: {issue_id}" if issue_id else "- Issue ID: not provided",
        f"- Artifact ID: {artifact_id}"
        if artifact_id
        else "- Artifact ID: not provided",
    ]
    base_rules = [
        "Use artifact IDs when referring to repository knowledge.",
        "Keep accepted and draft knowledge distinct.",
        "Accepted knowledge requires validation, gates, and human review.",
        "Drafts, proposals, proof attempts, and worker outputs are not "
        "accepted knowledge.",
        "Do not write accepted knowledge.",
        "Do not promote artifacts or mark human review complete.",
        "Do not include private KB content unless an explicit private policy "
        "permits it.",
        "Formal links are metadata unless a real checker verifies them.",
        "Before final handoff, run make test, make validate, and make gate.",
    ]
    task_guidance = {
        "start_issue_work": [
            "Build or request a bounded context pack before reasoning broadly.",
            "Prefer public accepted artifacts, and label drafts explicitly.",
            "Record uncertainties and required follow-up checks.",
        ],
        "reason_about_issue": [
            "State assumptions and dependencies by artifact ID.",
            "Separate conjecture, proof attempt, counterexample, and source evidence.",
            "Do not treat model output as proof or human review.",
        ],
        "verify_draft": [
            "Check schema, dependency, source, and evidence expectations.",
            "Report skipped verifier results as skipped, not pass.",
            "Do not upgrade draft status without explicit review workflow.",
        ],
        "prepare_review_bundle": [
            "Prepare review context and commands; do not perform human review.",
            "List changed artifact IDs and source metadata that require inspection.",
            "Keep validation/gate results separate from human-review decisions.",
        ],
        "public_kb_contribution_check": [
            "Confirm the material is public, citable, and source-reviewed.",
            "Reject private conjectures, unpublished ideas, and LLM-only content.",
            "Require source metadata and a human review record for accepted "
            "public KB artifacts.",
        ],
    }
    lines = [
        f"# {name}",
        "",
        "Scope:",
        *scope_lines,
        "",
        "Governance rules:",
        *(f"- {line}" for line in base_rules),
        "",
        "Task guidance:",
        *(f"- {line}" for line in task_guidance[name]),
    ]
    return "\n".join(lines) + "\n"


def serve_stdio(
    context: RepoContext,
    *,
    input_stream: TextIO | None = None,
    output_stream: TextIO | None = None,
) -> None:
    """Serve line-delimited JSON-RPC requests over stdio streams."""
    server = ReadOnlyMcpServer(context)
    input_handle = input_stream or sys.stdin
    output_handle = output_stream or sys.stdout
    for line in input_handle:
        stripped = line.strip()
        if not stripped:
            continue
        try:
            request = json.loads(stripped)
        except json.JSONDecodeError as exc:
            response = _error_response(
                None,
                McpServerError(
                    "invalid_json",
                    "request line is not valid JSON",
                    remediation="Send one JSON-RPC request object per line.",
                    details={"error": str(exc)},
                ),
            )
        else:
            if not isinstance(request, dict):
                response = _error_response(
                    None,
                    McpServerError(
                        "invalid_request",
                        "JSON-RPC request must be an object",
                        remediation="Send one JSON-RPC request object per line.",
                    ),
                )
            else:
                response = server.handle(request)
        output_handle.write(json.dumps(response, ensure_ascii=True) + "\n")
        output_handle.flush()


def _mcp_input_metadata(
    *,
    session: OperatorSession,
    arguments: Mapping[str, Any],
) -> dict[str, str]:
    keys = tuple(sorted(str(key) for key in arguments))
    metadata = {
        "argument_count": str(len(keys)),
        "argument_names": ",".join(keys) if keys else "none",
        "mcp_mode": "public_adapter",
        "session_mode": session.policy_mode.value,
    }
    if session.policy_mode is OperatorPolicyMode.PUBLIC_ONLY:
        return metadata
    for key in keys:
        value = arguments.get(key)
        if _mcp_metadata_value_is_safe(key, value):
            metadata[key] = str(value).strip()
    return metadata


def _mcp_metadata_value_is_safe(key: str, value: Any) -> bool:
    if isinstance(value, bool):
        return True
    if isinstance(value, int) and not isinstance(value, bool):
        return True
    if not isinstance(value, str):
        return False
    normalized = value.strip()
    if not normalized:
        return False
    lowered_key = key.replace("-", "_").lower()
    if lowered_key in {"query", "summary", "description", "statement", "notes"}:
        return False
    if any(part in lowered_key for part in ("payload", "request", "evidence")):
        return False
    return bool(re.fullmatch(r"[A-Za-z0-9_./:-]+", normalized))


def _mcp_success_summary(name: str, _payload: Mapping[str, Any]) -> str:
    return f"completed tool call: {name}"


def _mcp_error_summary(
    name: str,
    error: McpServerError,
    *,
    status: OperatorToolCallStatus,
) -> str:
    return f"{status.value} tool call: {name}; code={error.code}"


def _mcp_warning_codes(payload: Mapping[str, Any]) -> tuple[str, ...]:
    raw_warnings = payload.get("warning_codes")
    if raw_warnings is None:
        raw_warnings = payload.get("warnings")
    if isinstance(raw_warnings, str):
        return (raw_warnings,)
    if isinstance(raw_warnings, list | tuple):
        return tuple(str(item) for item in raw_warnings if str(item).strip())
    return ()


def _mcp_error_status(error: McpServerError) -> OperatorToolCallStatus:
    code = error.code.lower()
    if (
        "denied" in code
        or "forbidden" in code
        or "not_allowed" in code
        or code == "tool_not_found"
    ):
        return OperatorToolCallStatus.DENIED
    return OperatorToolCallStatus.FAILED


def _mcp_event_id(tool_name: str, recorded_at: datetime) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", tool_name.lower()).strip("-") or "tool"
    return f"event.{slug}.s{recorded_at:%Y%m%dt%H%M%Sz}"


def _success_response(request_id: Any, result: Mapping[str, Any]) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "result": dict(result)}


def _error_response(request_id: Any, error: McpServerError) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "error": {
            "code": -32000,
            "message": error.message,
            "data": error.to_error_data(),
        },
    }


def _tool_result(payload: Mapping[str, Any]) -> dict[str, Any]:
    text = json.dumps(payload, ensure_ascii=True, indent=2)
    return {
        "content": [{"type": "text", "text": text}],
        "structuredContent": dict(payload),
        "isError": False,
    }


def _tool_error_result(error: McpServerError) -> dict[str, Any]:
    payload = error.to_error_data()
    return {
        "content": [
            {
                "type": "text",
                "text": json.dumps(payload, ensure_ascii=True, indent=2),
            }
        ],
        "structuredContent": payload,
        "isError": True,
    }


def _resource_result(uri: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    return _text_resource_result(
        uri,
        json.dumps(payload, ensure_ascii=True, indent=2),
        mime_type="application/json",
    )


def _text_resource_result(
    uri: str,
    text: str,
    *,
    mime_type: str,
) -> dict[str, Any]:
    return {
        "contents": [
            {
                "uri": uri,
                "mimeType": mime_type,
                "text": text,
            }
        ]
    }


def _workspace_info_payload(info: WorkspaceInfoResult) -> dict[str, Any]:
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


def _validation_payload(report: ValidationReport) -> dict[str, Any]:
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


def _gate_payload(context: RepoContext, result: GatekeeperRunResult) -> dict[str, Any]:
    return {
        "verdict": result.report.verdict,
        "json_path": _repo_relative_or_string(context, result.json_path),
        "markdown_path": _repo_relative_or_string(context, result.markdown_path),
        "accepted_writes": False,
        "blocking_issues": [
            issue.to_dict() for issue in result.report.blocking_issues
        ],
        "nonblocking_issues": [
            issue.to_dict() for issue in result.report.nonblocking_issues
        ],
        "summary": result.report.summary,
        "gates": [gate.to_dict() for gate in result.report.gates],
    }


def _strategy_plan_payload(plan: StrategyPlan, relative_path: Path) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "kind": "strategy_plan",
        "plan_id": plan.plan_id,
        "path": relative_path.as_posix(),
        "accepted_write_performed": False,
        "authority_notice": STRATEGY_AUTHORITY_NOTICE,
        "plan": plan.to_dict(),
    }


def _public_strategy_plan(context: RepoContext, plan: StrategyPlan) -> StrategyPlan:
    """Return a public-scoped strategy plan for MCP public mode."""
    public_artifact_ids = {
        card.id for card in MemorySearchService(context).cards()
    }
    raw = plan.to_dict()
    problem = dict(raw["problem"])
    problem["target_artifacts"] = [
        artifact_id
        for artifact_id in problem.get("target_artifacts", [])
        if artifact_id in public_artifact_ids
    ]
    problem["tags"] = []
    problem["public_private_scope_labels"] = [
        scope
        for scope in problem.get("public_private_scope_labels", [])
        if scope != "private"
    ] or ["workspace"]
    raw["problem"] = problem

    original_nodes = [
        dict(node)
        for node in raw["graph"]["nodes"]
        if node.get("scope") != "private"
    ]
    kept_node_ids = {node["node_id"] for node in original_nodes}
    nodes = []
    for node in original_nodes:
        node["depends_on"] = [
            node_id
            for node_id in node.get("depends_on", [])
            if node_id in kept_node_ids
        ]
        node["blocked_by"] = [
            node_id
            for node_id in node.get("blocked_by", [])
            if node_id in kept_node_ids
        ]
        node["related_artifacts"] = [
            artifact_id
            for artifact_id in node.get("related_artifacts", [])
            if artifact_id in public_artifact_ids
        ]
        nodes.append(node)

    graph = dict(raw["graph"])
    graph["nodes"] = nodes
    graph["edges"] = [
        edge
        for edge in graph.get("edges", [])
        if edge.get("from_node") in kept_node_ids
        and edge.get("to_node") in kept_node_ids
    ]
    raw["graph"] = graph
    raw["next_steps"] = [
        step
        for step in raw.get("next_steps", [])
        if step.get("node_id") in kept_node_ids
    ]
    return StrategyPlan.model_validate(raw)


def _strategy_error_code(exc: BaseException) -> str:
    return getattr(exc, "code", "strategy_tool_failed")


def _strategy_error_remediation(exc: BaseException) -> str:
    return getattr(
        exc,
        "remediation",
        "Check the issue ID, plan ID, context path, and strategy runtime files.",
    )


def _research_run_error_code(exc: BaseException) -> str:
    return getattr(exc, "code", "research_run_tool_failed")


def _research_run_error_remediation(exc: BaseException) -> str:
    return getattr(
        exc,
        "remediation",
        "Check the run ID and repository-local research-run runtime files.",
    )


def _artifact_card_payload(card: ArtifactCard) -> dict[str, Any]:
    payload = card.to_dict()
    payload["artifact_id"] = card.id
    return payload


def _find_issue(context: RepoContext, issue_id: str) -> IssueRecord:
    try:
        records = tuple(load_artifacts(context))
    except LoadError as exc:
        raise McpServerError(
            "issue_load_failed",
            str(exc),
            remediation="Fix repository records and retry.",
        ) from exc
    matches = [
        record.record
        for record in records
        if isinstance(record.record, IssueRecord) and record.record.id == issue_id
    ]
    if not matches:
        raise McpServerError(
            "resource_not_found",
            "issue resource was not found",
            remediation="Check the issue ID.",
        )
    return sorted(matches, key=lambda issue: issue.id)[0]


def _artifact_exists_with_private_scope(context: RepoContext, artifact_id: str) -> bool:
    try:
        records = tuple(load_artifacts(context))
    except LoadError as exc:
        raise McpServerError(
            "artifact_load_failed",
            str(exc),
            remediation="Fix repository records and retry.",
        ) from exc
    return any(
        record.id == artifact_id
        and _loaded_record_scope(record) is MemoryRootScope.PRIVATE
        for record in records
    )


def _loaded_record_scope(record: LoadedRecord) -> MemoryRootScope:
    if record.is_private_scope:
        return MemoryRootScope.PRIVATE
    root_name = (record.kb_root_name or "").lower()
    if root_name == "framework":
        return MemoryRootScope.FRAMEWORK
    if record.is_public_kb_root or root_name == "public":
        return MemoryRootScope.PUBLIC
    return MemoryRootScope.WORKSPACE


def _resource_not_found(uri: str) -> McpServerError:
    return McpServerError(
        "resource_not_found",
        "MCP resource was not found",
        remediation="Use resources/list and a supported cosheaf:// URI.",
        details={"uri": uri},
    )


def _private_resource_denied() -> McpServerError:
    return McpServerError(
        "private_resource_denied",
        "private resources are not available through public read-only MCP mode",
        remediation=PRIVATE_SCOPE_REMEDIATION,
    )


def _repo_relative_or_string(context: RepoContext, path: Path) -> str:
    try:
        return path.resolve().relative_to(context.repo_root).as_posix()
    except ValueError:
        return str(path)


def _controlled_write_payload(result: ControlledWriteResult) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "kind": result.kind,
        "path": result.relative_path.as_posix(),
        "written_paths": [path.as_posix() for path in result.written_paths],
        "dry_run": result.dry_run,
        "accepted_write_performed": result.accepted_write_performed,
        "record_id": result.record_id,
        "authority_notice": MCP_CONTROLLED_WRITE_AUTHORITY_NOTICE,
    }


def _mcp_error_from_exception(
    exc: BaseException,
    *,
    default_code: str,
    default_remediation: str,
) -> McpServerError:
    if isinstance(exc, ServiceError):
        return McpServerError(
            exc.code,
            str(exc),
            remediation=exc.remediation,
            blocking=exc.blocking,
            details=exc.details,
        )
    if isinstance(exc, ValidationError):
        return McpServerError(
            default_code,
            _pydantic_error_message(exc),
            remediation=default_remediation,
        )
    return McpServerError(
        default_code,
        str(exc),
        remediation=default_remediation,
    )


def _mcp_checked_evidence_error(exc: BaseException) -> McpServerError:
    if isinstance(exc, CheckedCounterexampleEvidenceError):
        return McpServerError(
            exc.code,
            str(exc),
            remediation=exc.remediation,
            details=exc.details,
        )
    if isinstance(exc, ValidationError):
        return McpServerError(
            "checked_evidence_validation_failed",
            _pydantic_error_message(exc),
            remediation=(
                "Fix the checked counterexample evidence fields and retry. "
                "Checked evidence is review evidence only."
            ),
        )
    return McpServerError(
        "checked_evidence_validation_failed",
        str(exc),
        remediation=(
            "Fix the checked counterexample evidence fields and retry. "
            "Checked evidence is review evidence only."
        ),
    )


def _mcp_research_run_error(exc: BaseException) -> McpServerError:
    if isinstance(exc, ResearchRunError):
        return McpServerError(
            exc.code,
            str(exc),
            remediation=exc.remediation,
            details=exc.details,
        )
    if isinstance(exc, ValidationError):
        return McpServerError(
            "research_run_validation_failed",
            _pydantic_error_message(exc),
            remediation=(
                "Fix the research-run payload and retry. "
                "Research runs are provenance only."
            ),
        )
    return McpServerError(
        "research_run_validation_failed",
        str(exc),
        remediation=(
            "Fix the research-run payload and retry. "
            "Research runs are provenance only."
        ),
    )


def _pydantic_error_message(exc: ValidationError) -> str:
    messages = []
    for error in exc.errors():
        loc = ".".join(str(part) for part in error.get("loc", ()))
        message = str(error.get("msg", "invalid value"))
        messages.append(f"{loc}: {message}" if loc else message)
    return "; ".join(messages) if messages else str(exc)


def _required_string(mapping: Mapping[str, Any], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise McpServerError(
            "invalid_arguments",
            f"{key} must be a non-empty string",
            remediation="Pass the required string argument.",
    )
    return value.strip()


def _required_mapping(mapping: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = mapping.get(key)
    if not isinstance(value, Mapping):
        raise McpServerError(
            "invalid_arguments",
            f"{key} must be an object",
            remediation="Pass the required JSON object argument.",
        )
    return value


def _optional_bool(
    value: Any,
    *,
    field_name: str,
    default: bool = False,
) -> bool:
    if value is None:
        return default
    if not isinstance(value, bool):
        raise McpServerError(
            "invalid_arguments",
            f"{field_name} must be a boolean when provided",
            remediation="Pass true/false or omit the argument.",
        )
    return value


def _optional_string(value: Any, *, field_name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise McpServerError(
            "invalid_arguments",
            f"{field_name} must be a non-empty string when provided",
            remediation="Pass a string value or omit the argument.",
        )
    return value.strip()


def _optional_mapping(value: Any, *, field_name: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise McpServerError(
            "invalid_arguments",
            f"{field_name} must be an object",
            remediation="Pass a JSON object for this field.",
        )
    return value


def _optional_positive_int(value: Any, *, field_name: str, default: int) -> int:
    if value is None:
        return default
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise McpServerError(
            "invalid_arguments",
            f"{field_name} must be a positive integer",
            remediation="Pass a positive integer value.",
        )
    return value


def _optional_non_negative_int(value: Any, *, field_name: str, default: int) -> int:
    if value is None:
        return default
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise McpServerError(
            "invalid_arguments",
            f"{field_name} must be a non-negative integer",
            remediation="Pass a non-negative integer value.",
        )
    return value
