"""Persistent reviewable research workflow engine.

Workflow records connect existing deterministic surfaces into one runtime
review packet. They are review context only and never create proof, review,
accepted status, verifier results, gate results, or promotion authority.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from cosheaf.actions.builtins import build_default_registry
from cosheaf.actions.registry import (
    LocalActionPolicy,
    LocalActionRegistry,
    LocalActionResult,
    LocalActionRunRequest,
)
from cosheaf.core.ids import validate_artifact_id
from cosheaf.storage.repo import RepoContext

WORKFLOW_AUTHORITY_NOTICE = (
    "Research workflow records are review context only; they are not proof, "
    "verifier pass, gate pass, human review, accepted status, accepted "
    "refutation, or promotion authority."
)
WORKFLOW_RUNTIME_ROOT = Path(".cosheaf") / "workflows"
DEFAULT_WORKFLOW_ACTIONS = (
    "workspace.info",
    "validate.run",
    "gate.run",
    "context.build",
)

_ACCEPTED_PATH_PATTERN = re.compile(
    r"(?i)(?:^|[^A-Za-z0-9_-])kb[/\\][^\"'\s,}]*accepted[/\\]"
)
_PRIVATE_PATH_PATTERN = re.compile(
    r"(?i)(?:^|[/\\])kb[/\\]private[/\\]|(?:^|[/\\])private[/\\]"
)
_AUTHORITY_PATTERN = re.compile(
    r"(?i)(human_reviewed|human review|verifier_pass|gate_pass|"
    r"promotion_authority|accepted_status|accepted refutation)"
)


def _utc_now() -> datetime:
    return datetime.now(UTC).replace(microsecond=0)


def _json_text(payload: Any) -> str:
    if isinstance(payload, BaseModel):
        payload = payload.model_dump(mode="json")
    return json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n"


def _ensure_repo_local(context: RepoContext, target: Path) -> None:
    root = context.repo_root.resolve()
    resolved = target.resolve()
    if resolved != root and root not in resolved.parents:
        raise WorkflowError(
            f"workflow path escapes repository root: {target}",
            code="PATH_ESCAPES_REPOSITORY",
            remediation="Use a repository-local workflow runtime path.",
        )


def _safe_id(value: str, *, field_name: str) -> str:
    try:
        return validate_artifact_id(value.strip())
    except Exception as exc:
        raise WorkflowError(
            f"invalid {field_name}: {value!r}",
            code="INVALID_WORKFLOW_ID",
            remediation=(
                "Use dot-separated lowercase slugs, for example "
                "'workflow.example.1'."
            ),
        ) from exc


def _default_workflow_id(
    issue_id: str,
    *,
    context: RepoContext | None = None,
) -> str:
    issue = _safe_id(issue_id, field_name="issue_id")
    stamp = _utc_now().strftime("%Y%m%d%H%M%S")
    base_id = validate_artifact_id(f"workflow.{issue}.{stamp}")
    if context is None:
        return base_id
    if not context.resolve(workflow_root(base_id)).exists():
        return base_id
    seq = 2
    while True:
        candidate = validate_artifact_id(f"{base_id}.{seq:04d}")
        if not context.resolve(workflow_root(candidate)).exists():
            return candidate
        seq += 1


def _has_accepted_path(value: object) -> bool:
    return bool(_ACCEPTED_PATH_PATTERN.search(str(value)))


def _has_private_path(value: object) -> bool:
    return bool(_PRIVATE_PATH_PATTERN.search(str(value)))


def _has_authority_overclaim(value: object) -> bool:
    text = str(value)
    normalized = text.lower()
    allowed_diagnostics = (
        "accepted_write_blocked",
        "accepted-write permission",
        "accepted-write reference detected",
    )
    if any(marker in normalized for marker in allowed_diagnostics):
        return False
    if _AUTHORITY_PATTERN.search(text):
        return True
    return (
        "accepted" in normalized
        and (
            "status" in normalized
            or "write" in normalized
            or "promotion" in normalized
        )
    )


class WorkflowError(ValueError):
    """Expected workflow service failure."""

    def __init__(
        self,
        message: str,
        *,
        code: str,
        remediation: str,
        details: dict[str, str] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.remediation = remediation
        self.details = dict(details or {})


class WorkflowStatus(StrEnum):
    CREATED = "created"
    RUNNING = "running"
    READY_FOR_DRAFT = "ready_for_draft"
    BLOCKED = "blocked"
    FINALIZED = "finalized"
    FAILED = "failed"


class ReadinessClass(StrEnum):
    READY = "ready_for_draft_proposal"
    BLOCKED_GATE = "blocked_by_gate"
    BLOCKED_SCANNER = "blocked_by_scanner"
    BLOCKED_EVIDENCE = "blocked_by_missing_evidence"
    BLOCKED_LEAK = "blocked_by_private_leak_risk"
    BLOCKED_COUNTEREXAMPLE = "blocked_by_unchecked_counterexample"
    INCONCLUSIVE = "inconclusive"


class WorkflowModel(BaseModel):
    """Strict base model for workflow runtime records."""

    model_config = ConfigDict(extra="forbid", use_enum_values=False)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

    def to_json(self) -> str:
        return _json_text(self)


class WorkflowInput(WorkflowModel):
    issue_id: str
    query: str = ""
    authority_notice: str = WORKFLOW_AUTHORITY_NOTICE

    @field_validator("issue_id")
    @classmethod
    def _issue_id(cls, value: str) -> str:
        return _safe_id(value, field_name="issue_id")


class WorkflowOutput(WorkflowModel):
    output_refs: dict[str, str] = Field(default_factory=dict)
    authority_notice: str = WORKFLOW_AUTHORITY_NOTICE


class WorkflowEvidenceRef(WorkflowModel):
    kind: str
    ref: str
    checked: bool = False
    authority_notice: str = WORKFLOW_AUTHORITY_NOTICE


class WorkflowFailureSummary(WorkflowModel):
    failure_count: int = 0
    blocker_details: list[str] = Field(default_factory=list)
    authority_notice: str = WORKFLOW_AUTHORITY_NOTICE


class WorkflowAuthorityNotice(WorkflowModel):
    notice: str = WORKFLOW_AUTHORITY_NOTICE


class WorkflowReadinessSummary(WorkflowModel):
    classification: ReadinessClass
    blocker_details: list[str] = Field(default_factory=list)
    authority_notice: str = WORKFLOW_AUTHORITY_NOTICE


class WorkflowStep(WorkflowModel):
    step_number: int
    action: str
    status: str = "success"
    input_refs: dict[str, str] = Field(default_factory=dict)
    output_refs: dict[str, str] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    authority_notice: str = WORKFLOW_AUTHORITY_NOTICE

    @model_validator(mode="after")
    def _reject_authority_violations(self) -> WorkflowStep:
        values = {
            "input_refs": self.input_refs,
            "output_refs": self.output_refs,
            "warnings": self.warnings,
        }
        if _has_accepted_path(values):
            raise ValueError("workflow steps cannot reference accepted KB paths")
        if _has_authority_overclaim(values):
            raise ValueError(
                "workflow steps cannot claim accepted, human-review, "
                "verifier, gate, or promotion authority"
            )
        return self


class WorkflowRecord(WorkflowModel):
    workflow_id: str
    issue_id: str = ""
    query: str = ""
    revision: int = Field(default=1, ge=1)
    status: WorkflowStatus = WorkflowStatus.CREATED
    steps: list[WorkflowStep] = Field(default_factory=list)
    readiness: ReadinessClass | None = None
    created_at: datetime
    updated_at: datetime
    input: WorkflowInput | None = None
    output: WorkflowOutput = Field(default_factory=WorkflowOutput)
    evidence_refs: list[WorkflowEvidenceRef] = Field(default_factory=list)
    failure_summary: WorkflowFailureSummary = Field(
        default_factory=WorkflowFailureSummary
    )
    authority_notice: str = WORKFLOW_AUTHORITY_NOTICE

    @field_validator("workflow_id")
    @classmethod
    def _workflow_id(cls, value: str) -> str:
        return _safe_id(value, field_name="workflow_id")

    @field_validator("issue_id")
    @classmethod
    def _issue_id(cls, value: str) -> str:
        return _safe_id(value, field_name="issue_id") if value else value


class WorkflowReadinessReport(WorkflowModel):
    workflow_id: str
    classification: ReadinessClass
    blocker_details: list[str] = Field(default_factory=list)
    completed_steps: int = 0
    recommendations: list[str] = Field(default_factory=list)
    authority_notice: str = WORKFLOW_AUTHORITY_NOTICE


class WorkflowWriteResult(WorkflowModel):
    workflow: WorkflowRecord
    relative_path: Path
    events_path: Path
    librarian_path: Path
    fsm_path: Path
    loop_path: Path
    readiness_path: Path
    authority_notice: str = WORKFLOW_AUTHORITY_NOTICE

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def to_dict(self) -> dict[str, Any]:
        return {
            "workflow": self.workflow.to_dict(),
            "relative_path": self.relative_path.as_posix(),
            "events_path": self.events_path.as_posix(),
            "librarian_path": self.librarian_path.as_posix(),
            "fsm_path": self.fsm_path.as_posix(),
            "loop_path": self.loop_path.as_posix(),
            "readiness_path": self.readiness_path.as_posix(),
            "authority_notice": self.authority_notice,
        }


class WorkflowStepResult(WorkflowModel):
    workflow: WorkflowRecord
    step: WorkflowStep
    event_written: bool
    events_path: Path
    action_result: LocalActionResult | None = None
    authority_notice: str = WORKFLOW_AUTHORITY_NOTICE

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def to_dict(self) -> dict[str, Any]:
        return {
            "workflow": self.workflow.to_dict(),
            "step": self.step.to_dict(),
            "event_written": self.event_written,
            "events_path": self.events_path.as_posix(),
            "action_result": (
                self.action_result.model_dump(mode="json")
                if self.action_result is not None
                else None
            ),
            "authority_notice": self.authority_notice,
        }


class WorkflowRunResult(WorkflowModel):
    workflow: WorkflowRecord
    steps_requested: int
    steps_executed: int
    execute_local_actions: bool
    events_path: Path
    readiness: WorkflowReadinessReport
    authority_notice: str = WORKFLOW_AUTHORITY_NOTICE

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def to_dict(self) -> dict[str, Any]:
        return {
            "workflow": self.workflow.to_dict(),
            "steps_requested": self.steps_requested,
            "steps_executed": self.steps_executed,
            "execute_local_actions": self.execute_local_actions,
            "events_path": self.events_path.as_posix(),
            "readiness": self.readiness.to_dict(),
            "authority_notice": self.authority_notice,
        }


def workflow_root(workflow_id: str) -> Path:
    return WORKFLOW_RUNTIME_ROOT / _safe_id(workflow_id, field_name="workflow_id")


def workflow_path(workflow_id: str) -> Path:
    return workflow_root(workflow_id) / "workflow.json"


def workflow_events_path(workflow_id: str) -> Path:
    return workflow_root(workflow_id) / "events.jsonl"


def workflow_librarian_path(workflow_id: str) -> Path:
    return workflow_root(workflow_id) / "librarian.json"


def workflow_fsm_path(workflow_id: str) -> Path:
    return workflow_root(workflow_id) / "fsm.json"


def workflow_loop_path(workflow_id: str) -> Path:
    return workflow_root(workflow_id) / "loop.json"


def workflow_readiness_path(workflow_id: str) -> Path:
    return workflow_root(workflow_id) / "readiness.json"


def _write_text(context: RepoContext, relative_path: Path, text: str) -> None:
    target = context.resolve(relative_path)
    _ensure_repo_local(context, target)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8", newline="\n")


def _write_json(context: RepoContext, relative_path: Path, payload: Any) -> None:
    _write_text(context, relative_path, _json_text(payload))


def _ensure_component_files(context: RepoContext, workflow: WorkflowRecord) -> None:
    components: dict[Path, dict[str, Any]] = {
        workflow_librarian_path(workflow.workflow_id): {
            "workflow_id": workflow.workflow_id,
            "issue_id": workflow.issue_id,
            "query": workflow.query,
            "status": "planned",
            "authority_notice": WORKFLOW_AUTHORITY_NOTICE,
        },
        workflow_fsm_path(workflow.workflow_id): {
            "workflow_id": workflow.workflow_id,
            "status": "planned",
            "authority_notice": WORKFLOW_AUTHORITY_NOTICE,
        },
        workflow_loop_path(workflow.workflow_id): {
            "workflow_id": workflow.workflow_id,
            "status": "not_started",
            "authority_notice": WORKFLOW_AUTHORITY_NOTICE,
        },
    }
    for relative_path, payload in components.items():
        target = context.resolve(relative_path)
        _ensure_repo_local(context, target)
        if not target.exists():
            _write_json(context, relative_path, payload)


def _persist_workflow_files(
    context: RepoContext,
    workflow: WorkflowRecord,
) -> WorkflowWriteResult:
    relative_path = workflow_path(workflow.workflow_id)
    events_path = workflow_events_path(workflow.workflow_id)
    _write_json(context, relative_path, workflow)
    events_target = context.resolve(events_path)
    _ensure_repo_local(context, events_target)
    events_target.parent.mkdir(parents=True, exist_ok=True)
    if not events_target.exists():
        events_target.write_text("", encoding="utf-8", newline="\n")
    _ensure_component_files(context, workflow)
    readiness = assess_readiness(workflow)
    _write_json(context, workflow_readiness_path(workflow.workflow_id), readiness)
    return WorkflowWriteResult(
        workflow=workflow,
        relative_path=relative_path,
        events_path=events_path,
        librarian_path=workflow_librarian_path(workflow.workflow_id),
        fsm_path=workflow_fsm_path(workflow.workflow_id),
        loop_path=workflow_loop_path(workflow.workflow_id),
        readiness_path=workflow_readiness_path(workflow.workflow_id),
    )


def create_workflow(
    context: RepoContext,
    workflow: WorkflowRecord,
) -> WorkflowWriteResult:
    candidates = (
        workflow_path(workflow.workflow_id),
        workflow_events_path(workflow.workflow_id),
        workflow_librarian_path(workflow.workflow_id),
        workflow_fsm_path(workflow.workflow_id),
        workflow_loop_path(workflow.workflow_id),
        workflow_readiness_path(workflow.workflow_id),
    )
    for rel in candidates:
        target = context.resolve(rel)
        _ensure_repo_local(context, target)
        if target.exists():
            raise WorkflowError(
                f"workflow already exists: {workflow.workflow_id}",
                code="WORKFLOW_ALREADY_EXISTS",
                remediation=(
                    "Use a unique workflow ID or run `cosheaf workflow step` / "
                    "`cosheaf workflow run` on the existing workflow."
                ),
                details={"workflow_id": workflow.workflow_id},
            )
    return _persist_workflow_files(context, workflow)


def update_workflow(
    context: RepoContext,
    workflow: WorkflowRecord,
    *,
    expected_revision: int | None = None,
) -> WorkflowWriteResult:
    existing = load_workflow(context, workflow.workflow_id)
    if expected_revision is not None and existing.revision != expected_revision:
        raise WorkflowError(
            f"workflow revision conflict for {workflow.workflow_id}: "
            f"expected {expected_revision}, found {existing.revision}",
            code="WORKFLOW_REVISION_CONFLICT",
            remediation="Reload the workflow record before applying updates.",
            details={
                "workflow_id": workflow.workflow_id,
                "expected_revision": str(expected_revision),
                "current_revision": str(existing.revision),
            },
        )
    if (
        existing.issue_id
        and workflow.issue_id
        and existing.issue_id != workflow.issue_id
    ):
        raise WorkflowError(
            f"workflow {workflow.workflow_id} cannot change issue_id "
            f"from {existing.issue_id!r} to {workflow.issue_id!r}",
            code="WORKFLOW_CONFLICT",
            remediation="Start a new workflow with a distinct workflow ID.",
            details={"workflow_id": workflow.workflow_id},
        )
    if (
        len(workflow.steps) < len(existing.steps)
        or workflow.steps[: len(existing.steps)] != existing.steps
    ):
        raise WorkflowError(
            f"workflow {workflow.workflow_id} step history is append-only and "
            "cannot be truncated or rewritten",
            code="WORKFLOW_HISTORY_CONFLICT",
            remediation="Append new steps to the existing workflow history.",
            details={"workflow_id": workflow.workflow_id},
        )
    next_revision = max(workflow.revision, existing.revision + 1)
    updated = workflow.model_copy(update={"revision": next_revision})
    return _persist_workflow_files(context, updated)


def write_workflow(
    context: RepoContext,
    workflow: WorkflowRecord,
    *,
    expected_revision: int | None = None,
) -> WorkflowWriteResult:
    target = context.resolve(workflow_path(workflow.workflow_id))
    _ensure_repo_local(context, target)
    if target.exists():
        return update_workflow(
            context,
            workflow,
            expected_revision=expected_revision,
        )
    return create_workflow(context, workflow)


def load_workflow(context: RepoContext, workflow_id: str) -> WorkflowRecord:
    resolved = _safe_id(workflow_id, field_name="workflow_id")
    relative_path = workflow_path(resolved)
    target = context.resolve(relative_path)
    _ensure_repo_local(context, target)
    if not target.exists():
        raise WorkflowError(
            f"workflow not found: {resolved}",
            code="WORKFLOW_NOT_FOUND",
            remediation="Run `cosheaf workflow start --issue <issue-id>` first.",
            details={"workflow_id": resolved},
        )
    try:
        text = target.read_text(encoding="utf-8-sig")
        return WorkflowRecord.model_validate_json(text)
    except Exception as exc:
        raise WorkflowError(
            f"workflow JSON could not be parsed: {relative_path.as_posix()}",
            code="WORKFLOW_JSON_INVALID",
            remediation="Inspect or regenerate the workflow runtime record.",
        ) from exc


def append_workflow_event(
    context: RepoContext,
    *,
    workflow_id: str,
    event_kind: str,
    payload: dict[str, Any],
) -> Path:
    resolved = _safe_id(workflow_id, field_name="workflow_id")
    relative_path = workflow_events_path(resolved)
    target = context.resolve(relative_path)
    _ensure_repo_local(context, target)
    target.parent.mkdir(parents=True, exist_ok=True)
    line = {
        "workflow_id": resolved,
        "event_kind": event_kind,
        "created_at": _utc_now().isoformat(),
        "payload": payload,
        "authority_notice": WORKFLOW_AUTHORITY_NOTICE,
    }
    with target.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(line, ensure_ascii=True, sort_keys=True) + "\n")
    return relative_path


def start_workflow(
    context: RepoContext,
    *,
    issue_id: str,
    query: str = "",
    workflow_id: str | None = None,
    id_generator: Callable[[], str] | None = None,
) -> WorkflowWriteResult:
    now = _utc_now()
    if workflow_id is not None:
        resolved_workflow_id = _safe_id(workflow_id, field_name="workflow_id")
    elif id_generator is not None:
        resolved_workflow_id = _safe_id(id_generator(), field_name="workflow_id")
    else:
        resolved_workflow_id = _default_workflow_id(issue_id, context=context)
    workflow_input = WorkflowInput(issue_id=issue_id, query=query)
    workflow = WorkflowRecord(
        workflow_id=resolved_workflow_id,
        issue_id=workflow_input.issue_id,
        query=workflow_input.query,
        created_at=now,
        updated_at=now,
        input=workflow_input,
    )
    result = create_workflow(context, workflow)
    append_workflow_event(
        context,
        workflow_id=workflow.workflow_id,
        event_kind="workflow_started",
        payload={"issue_id": workflow.issue_id, "query": workflow.query},
    )
    return result


def append_step(record: WorkflowRecord, step: WorkflowStep) -> WorkflowRecord:
    steps = list(record.steps)
    numbered = step.model_copy(update={"step_number": len(steps) + 1})
    steps.append(numbered)
    status = WorkflowStatus.RUNNING
    if numbered.status == "blocked":
        status = WorkflowStatus.BLOCKED
    elif numbered.status == "error":
        status = WorkflowStatus.FAILED
    return record.model_copy(
        update={
            "steps": steps,
            "status": status,
            "updated_at": _utc_now(),
            "revision": record.revision + 1,
        }
    )


def _input_refs_for_action(workflow: WorkflowRecord, action_id: str) -> dict[str, str]:
    refs: dict[str, str] = {}
    if action_id in {"context.build", "strategy.next", "failure_memory.summary"}:
        refs["issue_id"] = workflow.issue_id
    elif action_id == "memory.search":
        refs["query"] = workflow.query
        refs["issue_id"] = workflow.issue_id
    elif action_id == "research_loop.scan":
        refs["loop_id"] = f"loop.{workflow.issue_id}"
    return refs


def _step_from_action_result(
    workflow: WorkflowRecord,
    *,
    action_id: str,
    action_result: LocalActionResult,
) -> WorkflowStep:
    output_refs = {
        "action_status": action_result.status.value,
        "scanner_status": action_result.scanner_status,
    }
    warnings: list[str] = []
    if action_result.error is not None:
        output_refs["error_code"] = action_result.error.error_code
        warnings.append(action_result.error.message)
    if action_result.stdout_snippet:
        output_refs["stdout_snippet"] = action_result.stdout_snippet
    if action_result.stderr_snippet:
        output_refs["stderr_snippet"] = action_result.stderr_snippet
    status = action_result.status.value
    if action_result.error is not None and action_result.error.error_code in {
        "UNKNOWN_ACTION",
        "ACCEPTED_WRITE_BLOCKED",
        "NETWORK_BLOCKED",
        "PROVIDER_BLOCKED",
        "SHELL_BLOCKED",
    }:
        status = "blocked"
    return WorkflowStep(
        step_number=len(workflow.steps) + 1,
        action=action_id,
        status=status,
        input_refs=action_result.input_refs,
        output_refs=output_refs,
        warnings=warnings,
    )


def step_workflow(
    context: RepoContext,
    workflow_id: str,
    *,
    action_id: str | None = None,
    execute_local_action: bool = False,
    registry: LocalActionRegistry | None = None,
) -> WorkflowStepResult:
    workflow = load_workflow(context, workflow_id)
    action = action_id or DEFAULT_WORKFLOW_ACTIONS[
        len(workflow.steps) % len(DEFAULT_WORKFLOW_ACTIONS)
    ]
    action_result: LocalActionResult | None = None
    if execute_local_action:
        active_registry = registry or build_default_registry()
        input_refs = _input_refs_for_action(workflow, action)
        action_result = active_registry.run(
            LocalActionRunRequest(action_id=action, input_refs=input_refs),
            LocalActionPolicy(
                allow_accepted_writes=False,
                allow_network=False,
                allow_hosted_provider=False,
                allow_shell=False,
            ),
            context.repo_root,
        )
        step = _step_from_action_result(
            workflow,
            action_id=action,
            action_result=action_result,
        )
    else:
        step = WorkflowStep(
            step_number=len(workflow.steps) + 1,
            action=action,
            status="planned",
            input_refs=_input_refs_for_action(workflow, action),
        )
    updated = append_step(workflow, step)
    readiness = assess_readiness(updated)
    updated = updated.model_copy(update={"readiness": readiness.classification})
    update_workflow(context, updated, expected_revision=workflow.revision)
    events_path = append_workflow_event(
        context,
        workflow_id=updated.workflow_id,
        event_kind="workflow_step",
        payload={
            "step": step.to_dict(),
            "execute_local_action": execute_local_action,
        },
    )
    return WorkflowStepResult(
        workflow=updated,
        step=step,
        event_written=True,
        events_path=events_path,
        action_result=action_result,
    )


def run_workflow(
    context: RepoContext,
    workflow_id: str,
    *,
    max_steps: int,
    execute_local_actions: bool = False,
) -> WorkflowRunResult:
    if max_steps < 1:
        raise WorkflowError(
            "max_steps must be at least 1",
            code="INVALID_MAX_STEPS",
            remediation="Pass `--max-steps` with a positive integer.",
        )
    executed = 0
    current = load_workflow(context, workflow_id)
    for _ in range(max_steps):
        action = DEFAULT_WORKFLOW_ACTIONS[
            len(current.steps) % len(DEFAULT_WORKFLOW_ACTIONS)
        ]
        result = step_workflow(
            context,
            workflow_id,
            action_id=action,
            execute_local_action=execute_local_actions,
        )
        current = result.workflow
        executed += 1
    readiness = assess_readiness(current)
    _write_json(context, workflow_readiness_path(current.workflow_id), readiness)
    return WorkflowRunResult(
        workflow=current,
        steps_requested=max_steps,
        steps_executed=executed,
        execute_local_actions=execute_local_actions,
        events_path=workflow_events_path(current.workflow_id),
        readiness=readiness,
    )


def assess_readiness(record: WorkflowRecord) -> WorkflowReadinessReport:
    completed = len(record.steps)
    blockers: list[str] = []
    classification = ReadinessClass.READY

    for step in record.steps:
        status = step.status.lower()
        if status in {"failed", "error"}:
            blockers.append(f"Step {step.step_number} ({step.action}) failed")
            classification = ReadinessClass.BLOCKED_EVIDENCE
        if status == "blocked":
            blockers.append(f"Step {step.step_number} ({step.action}) blocked")
            classification = ReadinessClass.BLOCKED_EVIDENCE
        if status == "planned" and classification == ReadinessClass.READY:
            blockers.append(f"Step {step.step_number} ({step.action}) is planned")
            classification = ReadinessClass.INCONCLUSIVE
        if status == "skipped" and classification == ReadinessClass.READY:
            blockers.append(f"Step {step.step_number} ({step.action}) skipped")
            classification = ReadinessClass.INCONCLUSIVE
        if _has_accepted_path(step.output_refs) or (
            step.output_refs.get("error_code") == "ACCEPTED_WRITE_BLOCKED"
        ):
            blockers.append(
                f"Step {step.step_number}: accepted-write reference detected"
            )
            classification = ReadinessClass.BLOCKED_SCANNER
        if _has_private_path(step.warnings) or _has_private_path(step.output_refs):
            blockers.append(f"Step {step.step_number}: private leakage risk detected")
            classification = ReadinessClass.BLOCKED_LEAK

    if completed == 0:
        classification = ReadinessClass.INCONCLUSIVE
    elif completed < 2 and classification == ReadinessClass.READY:
        classification = ReadinessClass.INCONCLUSIVE

    recommendations: list[str] = []
    if completed < 2:
        recommendations.append("Run more steps to gather evidence")
    if classification != ReadinessClass.READY:
        recommendations.append("Resolve blockers before drafting a proposal")

    return WorkflowReadinessReport(
        workflow_id=record.workflow_id,
        classification=classification,
        blocker_details=blockers,
        completed_steps=completed,
        recommendations=recommendations,
    )


__all__ = [
    "DEFAULT_WORKFLOW_ACTIONS",
    "WORKFLOW_AUTHORITY_NOTICE",
    "WORKFLOW_RUNTIME_ROOT",
    "ReadinessClass",
    "WorkflowAuthorityNotice",
    "WorkflowEvidenceRef",
    "WorkflowFailureSummary",
    "WorkflowInput",
    "WorkflowOutput",
    "WorkflowReadinessReport",
    "WorkflowReadinessSummary",
    "WorkflowRecord",
    "WorkflowRunResult",
    "WorkflowStatus",
    "WorkflowStep",
    "WorkflowStepResult",
    "WorkflowWriteResult",
    "append_step",
    "append_workflow_event",
    "assess_readiness",
    "create_workflow",
    "load_workflow",
    "run_workflow",
    "start_workflow",
    "step_workflow",
    "update_workflow",
    "workflow_events_path",
    "workflow_fsm_path",
    "workflow_librarian_path",
    "workflow_loop_path",
    "workflow_path",
    "workflow_readiness_path",
    "workflow_root",
    "write_workflow",
]
