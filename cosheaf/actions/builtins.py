"""Built-in local action implementations for the deterministic action registry.

Each action is a function that receives (request, policy, repo_root) and returns
a LocalActionResult. Actions must not write accepted KB, call hosted providers,
require network, or execute arbitrary shell by default.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from cosheaf.actions.registry import (
    ActionFunc,
    LocalActionError,
    LocalActionInputRefKind,
    LocalActionPolicy,
    LocalActionRegistry,
    LocalActionResult,
    LocalActionRunRequest,
    LocalActionSpec,
    LocalActionStatus,
)
from cosheaf.core.paths import repo_relative_posix
from cosheaf.storage.repo import RepoContext

ScannerStatus = Literal["clean", "blocked", "not_applicable"]
_ExecReturn = tuple[
    LocalActionStatus,
    dict[str, str],
    str,
    LocalActionError | None,
    ScannerStatus,
]


def _run_typed_service(
    action_id: str,
    request: LocalActionRunRequest,
    fn: Callable[[], _ExecReturn],
) -> LocalActionResult:
    started_at = datetime.now(UTC)
    try:
        status, output_refs, snippet, error, scanner_status = fn()
        finished_at = datetime.now(UTC)
        return LocalActionResult(
            action_id=action_id,
            status=status,
            input_refs=dict(request.input_refs),
            output_refs=output_refs,
            started_at=started_at,
            finished_at=finished_at,
            duration_seconds=(finished_at - started_at).total_seconds(),
            stdout_snippet=snippet[:500] if snippet else "",
            stderr_snippet=error.message[:500] if error is not None else "",
            error=error,
            scanner_status=scanner_status,
        )
    except Exception as exc:
        finished_at = datetime.now(UTC)
        code = getattr(exc, "code", "EXECUTION_EXCEPTION")
        if not isinstance(code, str) or not code.strip():
            code = "EXECUTION_EXCEPTION"
        return LocalActionResult(
            action_id=action_id,
            status=LocalActionStatus.FAILED,
            input_refs=dict(request.input_refs),
            started_at=started_at,
            finished_at=finished_at,
            duration_seconds=(finished_at - started_at).total_seconds(),
            stderr_snippet=str(exc)[:500],
            error=LocalActionError(error_code=code, message=str(exc)[:500]),
            scanner_status="not_applicable",
        )


def _json_snippet(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=True, sort_keys=True)[:500]


def _action_workspace_info(
    request: LocalActionRunRequest,
    policy: LocalActionPolicy,
    repo_root: Path,
) -> LocalActionResult:
    def _exec() -> _ExecReturn:
        from cosheaf.services import WorkspaceService

        info = WorkspaceService(RepoContext(repo_root)).info()
        payload = {
            "name": info.name,
            "mode": info.mode,
            "kb_root_count": len(info.kb_roots),
        }
        return (
            LocalActionStatus.SUCCESS,
            {"workspace_name": info.name, "mode": info.mode},
            _json_snippet(payload),
            None,
            "not_applicable",
        )

    return _run_typed_service("workspace.info", request, _exec)


def _action_validate(
    request: LocalActionRunRequest,
    policy: LocalActionPolicy,
    repo_root: Path,
) -> LocalActionResult:
    def _exec() -> _ExecReturn:
        from cosheaf.services import ValidationService

        report = ValidationService(RepoContext(repo_root)).validate_repository()
        if not report.ok:
            msg = (
                "; ".join(failure.message for failure in report.failures)
                or "Validation failed"
            )
            return (
                LocalActionStatus.FAILED,
                {"checked_files": str(report.checked_count)},
                msg,
                LocalActionError(error_code="VALIDATION_FAILED", message=msg[:500]),
                "not_applicable",
            )
        return (
            LocalActionStatus.SUCCESS,
            {"checked_files": str(report.checked_count)},
            f"Validation passed: checked {report.checked_count} YAML record(s).",
            None,
            "not_applicable",
        )

    return _run_typed_service("validate.run", request, _exec)


def _action_gate(
    request: LocalActionRunRequest,
    policy: LocalActionPolicy,
    repo_root: Path,
) -> LocalActionResult:
    def _exec() -> _ExecReturn:
        from cosheaf.services import GateService

        context = RepoContext(repo_root)
        result = GateService(context).run()
        output_refs: dict[str, str] = {}
        if result.json_path is not None:
            output_refs["json_report"] = repo_relative_posix(
                context.repo_root, result.json_path
            )
        if result.report.verdict != "pass":
            msg = f"Gate verdict: {result.report.verdict}"
            return (
                LocalActionStatus.FAILED,
                output_refs,
                msg,
                LocalActionError(error_code="GATE_FAILED", message=msg),
                "not_applicable",
            )
        return (
            LocalActionStatus.SUCCESS,
            output_refs,
            f"Gate verdict: {result.report.verdict}",
            None,
            "not_applicable",
        )

    return _run_typed_service("gate.run", request, _exec)


def _action_index_rebuild(
    request: LocalActionRunRequest,
    policy: LocalActionPolicy,
    repo_root: Path,
) -> LocalActionResult:
    def _exec() -> _ExecReturn:
        from cosheaf.storage.index import rebuild_index

        context = RepoContext(repo_root)
        result = rebuild_index(context)
        summary = (
            f"Index rebuilt ({result.artifact_count} artifacts, "
            f"{result.edge_count} edges)."
        )
        return (
            LocalActionStatus.SUCCESS,
            {
                "artifact_count": str(result.artifact_count),
                "edge_count": str(result.edge_count),
            },
            summary,
            None,
            "not_applicable",
        )

    return _run_typed_service("index.rebuild", request, _exec)


def _action_memory_search(
    request: LocalActionRunRequest,
    policy: LocalActionPolicy,
    repo_root: Path,
) -> LocalActionResult:
    def _exec() -> _ExecReturn:
        from cosheaf.services import MemorySearchService

        issue_id = request.input_refs.get("issue_id", "").strip()
        query = request.input_refs.get("query", "").strip() or issue_id
        if not query:
            return (
                LocalActionStatus.ERROR,
                {},
                "",
                LocalActionError(
                    error_code="MISSING_QUERY",
                    message="memory.search requires a non-empty query or issue_id",
                ),
                "not_applicable",
            )
        result = MemorySearchService(RepoContext(repo_root)).search(
            query,
            issue_id=issue_id or None,
        )
        return (
            LocalActionStatus.SUCCESS,
            {"card_count": str(len(result.cards))},
            _json_snippet(result.to_dict()),
            None,
            "not_applicable",
        )

    return _run_typed_service("memory.search", request, _exec)


def _action_context_build(
    request: LocalActionRunRequest,
    policy: LocalActionPolicy,
    repo_root: Path,
) -> LocalActionResult:
    def _exec() -> _ExecReturn:
        from cosheaf.services import ContextPackService

        issue_id = request.input_refs.get("issue_id", "").strip()
        context = RepoContext(repo_root)
        result = ContextPackService(context).build(
            issue_id,
            public_only=policy.mode == "public_only",
        )
        return (
            LocalActionStatus.SUCCESS,
            {
                "context_pack": repo_relative_posix(
                    context.repo_root, result.task_dir
                )
            },
            f"Built context pack for {issue_id}",
            None,
            "not_applicable",
        )

    return _run_typed_service("context.build", request, _exec)


def _action_strategy_next(
    request: LocalActionRunRequest,
    policy: LocalActionPolicy,
    repo_root: Path,
) -> LocalActionResult:
    def _exec() -> _ExecReturn:
        from cosheaf.strategy.models import StrategyError
        from cosheaf.strategy.planner import build_strategy_plan
        from cosheaf.strategy.storage import load_strategy_plan

        context = RepoContext(repo_root)
        plan_or_issue = (
            request.input_refs.get("strategy_id")
            or request.input_refs.get("issue_id", "")
        ).strip()
        try:
            loaded = load_strategy_plan(context, plan_or_issue)
            plan = loaded.plan
        except StrategyError:
            built = build_strategy_plan(context, plan_or_issue)
            plan = built.plan
        payload = {
            "plan_id": plan.plan_id,
            "next_steps": [step.to_dict() for step in plan.next_steps],
        }
        return (
            LocalActionStatus.SUCCESS,
            {"plan_id": plan.plan_id, "step_count": str(len(plan.next_steps))},
            _json_snippet(payload),
            None,
            "not_applicable",
        )

    return _run_typed_service("strategy.next", request, _exec)


def _action_research_loop_scan(
    request: LocalActionRunRequest,
    policy: LocalActionPolicy,
    repo_root: Path,
) -> LocalActionResult:
    def _exec() -> _ExecReturn:
        from cosheaf.research.loop import scan_research_loop

        loop_id = request.input_refs.get("loop_id", "").strip()
        result = scan_research_loop(RepoContext(repo_root), loop_id)
        if result.handoff_blocked:
            return (
                LocalActionStatus.BLOCKED,
                {"report_path": result.report_path},
                _json_snippet(result.to_dict()),
                LocalActionError(
                    error_code="SCANNER_BLOCKED",
                    message=(
                        f"research loop {loop_id} has "
                        f"{result.blocking_finding_count} blocking scan finding(s)"
                    ),
                    scanner_blocker=True,
                ),
                "blocked",
            )
        return (
            LocalActionStatus.SUCCESS,
            {"report_path": result.report_path},
            _json_snippet(result.to_dict()),
            None,
            "clean",
        )

    return _run_typed_service("research_loop.scan", request, _exec)


def _action_operator_session_scan(
    request: LocalActionRunRequest,
    policy: LocalActionPolicy,
    repo_root: Path,
) -> LocalActionResult:
    def _exec() -> _ExecReturn:
        from cosheaf.operator_session import scan_operator_session

        session_id = request.input_refs.get("session_id", "").strip()
        result = scan_operator_session(RepoContext(repo_root), session_id)
        if result.handoff_blocked:
            return (
                LocalActionStatus.BLOCKED,
                {"report_path": result.report_path.as_posix()},
                _json_snippet(result.to_dict()),
                LocalActionError(
                    error_code="SCANNER_BLOCKED",
                    message=(
                        f"operator session {session_id} has "
                        f"{result.blocking_finding_count} blocking scan finding(s)"
                    ),
                    scanner_blocker=True,
                ),
                "blocked",
            )
        return (
            LocalActionStatus.SUCCESS,
            {"report_path": result.report_path.as_posix()},
            _json_snippet(result.to_dict()),
            None,
            "clean",
        )

    return _run_typed_service("operator_session.scan", request, _exec)


def _action_operator_handoff_preview(
    request: LocalActionRunRequest,
    policy: LocalActionPolicy,
    repo_root: Path,
) -> LocalActionResult:
    def _exec() -> _ExecReturn:
        from cosheaf.operator_session import export_operator_handoff

        handoff_id = request.input_refs.get("handoff_id", "").strip()
        result = export_operator_handoff(
            RepoContext(repo_root),
            handoff_id=handoff_id,
            dry_run=True,
        )
        return (
            LocalActionStatus.SUCCESS,
            {"handoff_bundle": result.target_path},
            _json_snippet(result.to_dict()),
            None,
            "not_applicable",
        )

    return _run_typed_service("operator_handoff.preview", request, _exec)


def _action_research_run_summary(
    request: LocalActionRunRequest,
    policy: LocalActionPolicy,
    repo_root: Path,
) -> LocalActionResult:
    def _exec() -> _ExecReturn:
        from cosheaf.research.run import load_research_run

        run_id = request.input_refs.get("run_id", "").strip()
        result = load_research_run(RepoContext(repo_root), run_id)
        return (
            LocalActionStatus.SUCCESS,
            {"run_id": result.record.run_id, "status": result.record.status.value},
            _json_snippet(result.to_dict()),
            None,
            "not_applicable",
        )

    return _run_typed_service("research_run.summary", request, _exec)


def _action_checked_evidence_summary(
    request: LocalActionRunRequest,
    policy: LocalActionPolicy,
    repo_root: Path,
) -> LocalActionResult:
    def _exec() -> _ExecReturn:
        from cosheaf.verification.counterexample_evidence import (
            show_checked_counterexample_evidence,
        )

        evidence_ref = request.input_refs.get("artifact_id", "").strip()
        result = show_checked_counterexample_evidence(
            RepoContext(repo_root), evidence_ref
        )
        return (
            LocalActionStatus.SUCCESS,
            {
                "evidence_id": result.record.evidence_id,
                "checked_result": result.record.checked_result.value,
            },
            _json_snippet(result.to_dict()),
            None,
            "not_applicable",
        )

    return _run_typed_service("checked_evidence.summary", request, _exec)


def _action_failure_memory_summary(
    request: LocalActionRunRequest,
    policy: LocalActionPolicy,
    repo_root: Path,
) -> LocalActionResult:
    def _exec() -> _ExecReturn:
        from cosheaf.research.loop import load_attempt_memory_index

        issue_id = request.input_refs.get("issue_id", "").strip()
        index = load_attempt_memory_index(RepoContext(repo_root))
        matching = [
            entry.to_dict()
            for entry in index.entries
            if not issue_id or entry.issue_id == issue_id
        ]
        return (
            LocalActionStatus.SUCCESS,
            {"entry_count": str(len(matching))},
            _json_snippet({"issue_id": issue_id, "entries": matching}),
            None,
            "not_applicable",
        )

    return _run_typed_service("failure_memory.summary", request, _exec)


def _action_eval_research_loop(
    request: LocalActionRunRequest,
    policy: LocalActionPolicy,
    repo_root: Path,
) -> LocalActionResult:
    def _exec() -> _ExecReturn:
        from cosheaf.evals import (
            DEFAULT_RESEARCH_LOOP_EVAL_CASES,
            load_research_loop_eval_suite,
            resolve_research_loop_eval_case_path,
            run_research_loop_eval_suite,
        )

        context = RepoContext(repo_root)
        case_path = resolve_research_loop_eval_case_path(
            context, DEFAULT_RESEARCH_LOOP_EVAL_CASES
        )
        suite = load_research_loop_eval_suite(case_path)
        report = run_research_loop_eval_suite(context, suite)
        status = (
            LocalActionStatus.SUCCESS if report.passed else LocalActionStatus.FAILED
        )
        error = (
            None
            if report.passed
            else LocalActionError(
                error_code="EVAL_FAILED",
                message="research-loop eval suite did not pass",
            )
        )
        return (
            status,
            {
                "case_count": str(report.case_count),
                "passed": str(report.passed).lower(),
            },
            _json_snippet(report.to_dict()),
            error,
            "not_applicable",
        )

    return _run_typed_service("eval.research_loop", request, _exec)


# Action spec definitions for the registry
_BUILTIN_SPECS: list[LocalActionSpec] = [
    LocalActionSpec(
        action_id="workspace.info",
        description="Inspect the configured workspace and KB roots",
        allowed_input_refs=[],
        max_timeout_seconds=30,
    ),
    LocalActionSpec(
        action_id="validate.run",
        description="Run artifact validation against loaded KB roots",
        allowed_input_refs=[],
        max_timeout_seconds=60,
    ),
    LocalActionSpec(
        action_id="gate.run",
        description="Run the gatekeeper check",
        allowed_input_refs=[],
        max_timeout_seconds=60,
    ),
    LocalActionSpec(
        action_id="index.rebuild",
        description="Rebuild the deterministic SQLite/manifest index",
        allowed_input_refs=[],
        max_timeout_seconds=120,
    ),
    LocalActionSpec(
        action_id="memory.search",
        description="Search the memory graph for related artifacts",
        allowed_input_refs=[LocalActionInputRefKind.ISSUE_ID],
        max_timeout_seconds=60,
    ),
    LocalActionSpec(
        action_id="context.build",
        description="Build a ranked context pack for an issue",
        allowed_input_refs=[LocalActionInputRefKind.ISSUE_ID],
        max_timeout_seconds=120,
    ),
    LocalActionSpec(
        action_id="strategy.next",
        description="Get the next strategy action for an issue",
        allowed_input_refs=[LocalActionInputRefKind.ISSUE_ID],
        max_timeout_seconds=60,
    ),
    LocalActionSpec(
        action_id="research_loop.scan",
        description="Scan a research loop for leaks and authority violations",
        allowed_input_refs=[LocalActionInputRefKind.LOOP_ID],
        max_timeout_seconds=60,
    ),
    LocalActionSpec(
        action_id="operator_session.scan",
        description="Scan an operator session for leaks and authority violations",
        allowed_input_refs=[LocalActionInputRefKind.SESSION_ID],
        max_timeout_seconds=60,
    ),
    LocalActionSpec(
        action_id="operator_handoff.preview",
        description="Preview an operator handoff export (dry run)",
        allowed_input_refs=[LocalActionInputRefKind.HANDOFF_ID],
        max_timeout_seconds=60,
    ),
    LocalActionSpec(
        action_id="research_run.summary",
        description="Show a research run summary",
        allowed_input_refs=[LocalActionInputRefKind.RUN_ID],
        max_timeout_seconds=30,
    ),
    LocalActionSpec(
        action_id="checked_evidence.summary",
        description="Show checked evidence summary for an artifact",
        allowed_input_refs=[LocalActionInputRefKind.ARTIFACT_ID],
        max_timeout_seconds=30,
    ),
    LocalActionSpec(
        action_id="failure_memory.summary",
        description="Show failure memory summary for an issue",
        allowed_input_refs=[LocalActionInputRefKind.ISSUE_ID],
        max_timeout_seconds=30,
    ),
    LocalActionSpec(
        action_id="eval.research_loop",
        description="Run the deterministic research-loop eval",
        allowed_input_refs=[],
        max_timeout_seconds=120,
    ),
]


_BUILTIN_FUNCS: dict[str, ActionFunc] = {
    "workspace.info": _action_workspace_info,
    "validate.run": _action_validate,
    "gate.run": _action_gate,
    "index.rebuild": _action_index_rebuild,
    "memory.search": _action_memory_search,
    "context.build": _action_context_build,
    "strategy.next": _action_strategy_next,
    "research_loop.scan": _action_research_loop_scan,
    "operator_session.scan": _action_operator_session_scan,
    "operator_handoff.preview": _action_operator_handoff_preview,
    "research_run.summary": _action_research_run_summary,
    "checked_evidence.summary": _action_checked_evidence_summary,
    "failure_memory.summary": _action_failure_memory_summary,
    "eval.research_loop": _action_eval_research_loop,
}


def build_default_registry() -> LocalActionRegistry:
    """Build a LocalActionRegistry with all built-in actions registered."""
    registry = LocalActionRegistry()
    for spec in _BUILTIN_SPECS:
        func = _BUILTIN_FUNCS.get(spec.action_id)
        if func is None:
            raise KeyError(
                f"No function registered for built-in action {spec.action_id!r}"
            )
        registry.register(spec, func)
    return registry
