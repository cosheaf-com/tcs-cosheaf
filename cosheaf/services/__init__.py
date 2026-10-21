"""Typed service layer shared by CLI and future agent-access surfaces.

Services are thin boundaries over existing repository logic. They return typed
results and enforce the same local path, public/private KB, gate, review, and
promotion boundaries as the CLI.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from difflib import unified_diff
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

import yaml  # type: ignore[import-untyped]
from pydantic import ValidationError
from yaml import YAMLError

from cosheaf.agent.context_pack import (
    CONTEXT_MAX_CARDS,
    ContextPackResult,
    build_context_pack,
    show_context_pack,
)
from cosheaf.agent.local_runner import (
    LocalWorkerRunConfig,
    LocalWorkerRunner,
    LocalWorkerRunResult,
)
from cosheaf.agent.orchestrator_planner import plan_for_issue
from cosheaf.agent.orchestrator_state import Plan, ReducerResult
from cosheaf.agent.orchestrator_stub import OrchestratorStub, TaskCompletionResult
from cosheaf.agent.task import AgentTask, WorkerType
from cosheaf.agent.worker_bundle_v2 import (
    WorkerBundleV2,
    WorkerBundleV2Error,
    reduce_worker_bundle_v2,
    validate_worker_bundle_v2,
    worker_bundle_review_warnings,
)
from cosheaf.config.workspace import KbRootConfig
from cosheaf.core.artifact import (
    BaseArtifact,
    Evidence,
    FailureLogEntry,
    SourceMetadata,
    evidence_revision_mismatches,
    is_external_dependency_ref,
    review_revision_mismatch,
)
from cosheaf.core.ids import validate_artifact_id
from cosheaf.core.paths import lifecycle_artifact_path, normalize_repo_path
from cosheaf.core.status import (
    ArtifactStatus,
    ArtifactType,
    expected_status_for_path,
    is_preaccepted_status,
)
from cosheaf.gates.dependency_gate import evaluate_status_transition
from cosheaf.gates.gatekeeper import (
    GatekeeperRunResult,
    ValidationReport,
    run_gatekeeper,
    validate_artifact_file,
    validate_repository,
)
from cosheaf.gates.promotion_readiness import promotion_status_blocker
from cosheaf.gates.source_metadata_gate import missing_required_source_metadata
from cosheaf.memory import (
    ArtifactCard,
    ArtifactCardStatus,
    RetrievalResult,
    RetrievalRole,
    build_artifact_cards,
    search_artifact_cards,
)
from cosheaf.services.context_policy import ContextSendPolicyService
from cosheaf.services.models import (
    DraftArtifactWriteRequest,
    ErrorResult,
    WorkerBundleSubmitRequest,
    WorkerBundleSubmitResult,
)
from cosheaf.storage.loader import LoadedRecord, LoadError, ReviewRecord, load_artifacts
from cosheaf.storage.repo import RepoContext
from cosheaf.storage.writer import dump_yaml_deterministic, write_yaml_deterministic


@dataclass(frozen=True)
class KbRootInfo:
    """One KB root exposed through the service boundary."""

    name: str
    path: str
    readonly: bool
    priority: int


@dataclass(frozen=True)
class WorkspaceInfoResult:
    """Typed workspace information for machine and CLI callers."""

    name: str
    repo_root: Path
    mode: str
    kb_roots: tuple[KbRootInfo, ...]


@dataclass(frozen=True)
class ArtifactWriteResult:
    """Result of a controlled draft/pre-accepted artifact write."""

    artifact: BaseArtifact
    relative_path: Path

    def yaml_text(self) -> str:
        """Return the deterministic YAML for this artifact."""
        return dump_yaml_deterministic(self.artifact)


@dataclass(frozen=True)
class ControlledWriteResult:
    """Result of a controlled staging write or dry-run preview."""

    kind: str
    relative_path: Path
    written_paths: tuple[Path, ...]
    dry_run: bool
    accepted_write_performed: bool = False
    record_id: str = ""


@dataclass(frozen=True)
class ReviewDecisionWriteResult:
    """Result of a human review decision preview or write."""

    review: Mapping[str, Any]
    artifact: BaseArtifact
    relative_path: Path
    artifact_relative_path: Path
    written_paths: tuple[Path, ...]
    dry_run: bool
    record_id: str
    artifact_updated: bool


@dataclass(frozen=True)
class PromotionActionResult:
    """Result of a promotion preview or confirmed lifecycle write."""

    artifact: BaseArtifact
    updated_artifact: BaseArtifact
    target_state: ArtifactStatus
    old_status: ArtifactStatus
    old_relative_path: Path
    new_relative_path: Path
    planned_files: tuple[Path, ...]
    written_paths: tuple[Path, ...]
    dry_run: bool
    actor: str
    yaml_diff: str
    validation_summary: str
    gate_summary: str

    @property
    def accepted_write_performed(self) -> bool:
        return (
            not self.dry_run
            and self.target_state is ArtifactStatus.ACCEPTED
            and bool(self.written_paths)
        )

    @property
    def promotion_performed(self) -> bool:
        return not self.dry_run and bool(self.written_paths)

    def review_record_preview(self) -> dict[str, str]:
        """Return the existing review state used by this promotion action."""
        return {
            "actor": self.actor,
            "artifact_id": self.artifact.id,
            "target_state": self.target_state.value,
            "existing_review_state": self.artifact.review.state,
        }


@dataclass(frozen=True)
class ReviewRequestFromBundleResult:
    """Generated draft review request plus controlled write metadata."""

    bundle: WorkerBundleV2
    request: Mapping[str, Any]
    write_result: ControlledWriteResult


@dataclass(frozen=True)
class FailureLogFromBundlePlanResult:
    """Proposed artifact failure-log entries derived from a WorkerBundle."""

    bundle: WorkerBundleV2
    artifact_id: str
    relative_path: Path
    entries: tuple[FailureLogEntry, ...]


@dataclass(frozen=True)
class FailureLogFromBundleWriteResult:
    """Controlled write result for WorkerBundle-derived failure-log entries."""

    plan: FailureLogFromBundlePlanResult
    write_result: ControlledWriteResult


class ServiceError(ValueError):
    """Expected service-layer failure with a stable machine-readable code."""

    def __init__(
        self,
        message: str,
        *,
        code: str,
        remediation: str,
        blocking: bool = True,
        details: dict[str, str] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.remediation = remediation
        self.blocking = blocking
        self.details = dict(details or {})

    def to_error_result(self) -> ErrorResult:
        """Convert this service error to the public agent-access error DTO."""
        return ErrorResult(
            code=self.code,
            message=str(self),
            remediation=self.remediation,
            blocking=self.blocking,
            details=self.details,
        )


class DraftWriteServiceError(ServiceError):
    """Raised for expected draft-write service failures."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "draft_write_failed",
        remediation: str = (
            "Fix the draft artifact request and retry. Accepted knowledge must "
            "enter through explicit review, gates, and promotion."
        ),
        blocking: bool = True,
        details: dict[str, str] | None = None,
    ) -> None:
        super().__init__(
            message,
            code=code,
            remediation=remediation,
            blocking=blocking,
            details=details,
        )


class WorkspaceService:
    """Service for workspace inspection."""

    def __init__(self, context: RepoContext) -> None:
        self.context = context

    def info(self) -> WorkspaceInfoResult:
        """Return active workspace metadata and ordered KB roots."""
        config = self.context.workspace_config
        return WorkspaceInfoResult(
            name=config.name,
            repo_root=self.context.repo_root,
            mode="configured" if config.configured else "legacy",
            kb_roots=tuple(
                KbRootInfo(
                    name=root.name,
                    path=root.path,
                    readonly=root.readonly,
                    priority=root.priority,
                )
                for root in config.ordered_kb
            ),
        )


class ValidationService:
    """Service for repository and artifact validation."""

    def __init__(self, context: RepoContext) -> None:
        self.context = context

    def validate_repository(self) -> ValidationReport:
        """Validate repository YAML records and invariants."""
        return validate_repository(self.context)

    def validate_artifact_file(self, path: str | Path) -> ValidationReport:
        """Validate one repository-local artifact YAML file."""
        return validate_artifact_file(self.context, Path(path))


class GateService:
    """Service for gatekeeper execution."""

    def __init__(self, context: RepoContext) -> None:
        self.context = context

    def run(
        self,
        *,
        persist_review: bool = False,
        pr_checklist_path: str | Path | None = None,
        timestamp: str | None = None,
    ) -> GatekeeperRunResult:
        """Run gatekeeper checks and return written report metadata."""
        return run_gatekeeper(
            self.context,
            persist_review=persist_review,
            pr_checklist_path=Path(pr_checklist_path)
            if pr_checklist_path is not None
            else None,
            timestamp=timestamp,
        )


class MemorySearchService:
    """Service for deterministic artifact-card search."""

    def __init__(self, context: RepoContext) -> None:
        self.context = context

    def cards(
        self,
        *,
        issue_id: str | None = None,
        status: ArtifactCardStatus | None = None,
    ) -> tuple[ArtifactCard, ...]:
        """Build compact artifact cards from existing repository metadata."""
        return build_artifact_cards(
            self.context,
            issue_id=issue_id,
            status=status,
        )

    def search(
        self,
        query: str,
        *,
        issue_id: str | None = None,
        status: ArtifactCardStatus | None = None,
        max_cards: int = CONTEXT_MAX_CARDS,
        seed_artifacts: Sequence[str] = (),
        pinned_artifacts: Sequence[str] = (),
        include_refuted: bool = False,
        include_obsolete: bool = False,
    ) -> RetrievalResult:
        """Search artifact cards with deterministic local scoring."""
        return search_artifact_cards(
            self.context,
            query=query,
            issue_id=issue_id,
            status=status,
            max_cards=max_cards,
            seed_artifacts=tuple(seed_artifacts),
            pinned_artifacts=tuple(pinned_artifacts),
            include_refuted=include_refuted,
            include_obsolete=include_obsolete,
        )


class ContextPackService:
    """Service for bounded issue-scoped context packs."""

    def __init__(self, context: RepoContext) -> None:
        self.context = context

    def build(
        self,
        issue_id: str,
        *,
        role: RetrievalRole | str = RetrievalRole.ORCHESTRATOR,
        max_cards: int = CONTEXT_MAX_CARDS,
        max_full_artifacts: int | None = None,
        public_only: bool = False,
    ) -> ContextPackResult:
        """Build a deterministic context pack for an issue."""
        return build_context_pack(
            self.context,
            issue_id,
            role=role,
            max_cards=max_cards,
            max_full_artifacts=max_full_artifacts,
            public_only=public_only,
        )

    def show(
        self,
        issue_id: str,
        *,
        role: RetrievalRole | str = RetrievalRole.ORCHESTRATOR,
        max_cards: int = CONTEXT_MAX_CARDS,
        max_full_artifacts: int | None = None,
        public_only: bool = False,
    ) -> str:
        """Build and return the rendered main context document."""
        return show_context_pack(
            self.context,
            issue_id,
            role=role,
            max_cards=max_cards,
            max_full_artifacts=max_full_artifacts,
            public_only=public_only,
        )


class OrchestratorPlanService:
    """Service for deterministic issue-scoped orchestrator plans."""

    def __init__(self, context: RepoContext) -> None:
        self.context = context

    def plan_for_issue(self, issue_id: str) -> Plan:
        """Create a deterministic plan for an existing issue without execution."""
        return plan_for_issue(self.context, issue_id)


class TaskService:
    """Service for local task records and explicit local worker runs."""

    def __init__(self, context: RepoContext) -> None:
        self.context = context
        self.orchestrator = OrchestratorStub(context)

    def create_task(
        self,
        *,
        issue_id: str,
        worker_type: WorkerType | str,
    ) -> AgentTask:
        """Create an open local task for an existing issue."""
        return self.orchestrator.create_task(
            issue_id=issue_id,
            worker_type=worker_type,
        )

    def list_tasks(self) -> tuple[AgentTask, ...]:
        """List local tasks in deterministic order."""
        return self.orchestrator.list_tasks()

    def complete_task(
        self,
        *,
        task_id: str,
        bundle_path: str | Path,
    ) -> TaskCompletionResult:
        """Validate a worker output bundle and complete a task."""
        return self.orchestrator.complete_task(
            task_id=task_id,
            bundle_path=bundle_path,
        )

    def run_task(
        self,
        task_id: str,
        *,
        command: Sequence[str],
        timeout_seconds: int = 60,
        cwd: str | Path | None = None,
        bundle_path: str | Path | None = None,
    ) -> LocalWorkerRunResult:
        """Run an explicit local command for an existing task."""
        return LocalWorkerRunner(self.context).run_task(
            task_id,
            LocalWorkerRunConfig(
                command=command,
                timeout_seconds=timeout_seconds,
                cwd=cwd,
                bundle_path=bundle_path,
            ),
        )


class BundleValidationService:
    """Service for worker bundle v2 validation and reduction."""

    def __init__(self, context: RepoContext) -> None:
        self.context = context

    def validate(self, bundle_path: str | Path) -> WorkerBundleV2:
        """Load and validate a worker bundle v2 manifest."""
        return validate_worker_bundle_v2(self.context, bundle_path)

    def reduce(self, bundle_path: str | Path, *, reducer_id: str) -> ReducerResult:
        """Validate and reduce a worker bundle v2 manifest."""
        return reduce_worker_bundle_v2(
            self.context,
            bundle_path,
            reducer_id=reducer_id,
        )

    def submit(
        self,
        request: WorkerBundleSubmitRequest,
        *,
        dry_run: bool = False,
    ) -> WorkerBundleSubmitResult:
        """Validate a worker bundle for review without promotion or task closure."""
        if request.complete_task:
            raise ServiceError(
                "bundle submit does not complete tasks in this controlled surface",
                code="bundle_complete_forbidden",
                remediation=(
                    "Submit the bundle for review first. Complete task records "
                    "through the explicit task workflow when that is intended."
                ),
                blocking=True,
            )

        try:
            bundle = self.validate(request.bundle_path)
        except WorkerBundleV2Error as exc:
            raise ServiceError(
                str(exc),
                code="bundle_submit_failed",
                remediation=(
                    "Fix the worker bundle v2 manifest and keep proposed outputs "
                    "under draft/proposal paths."
                ),
                blocking=True,
            ) from exc

        if bundle.task_id != request.task_id:
            raise ServiceError(
                "worker bundle task_id does not match submit request",
                code="bundle_submit_failed",
                remediation="Use a submit request whose task_id matches the bundle.",
                blocking=True,
                details={
                    "request_task_id": request.task_id,
                    "bundle_task_id": bundle.task_id,
                },
            )

        warnings = worker_bundle_review_warnings(bundle)
        if dry_run:
            warnings.append("dry-run: bundle validated; no task state was changed")

        return WorkerBundleSubmitResult(
            task_id=bundle.task_id,
            bundle_id=bundle.bundle_id,
            accepted_for_review=True,
            output_paths=[
                proposed.path for proposed in bundle.proposed_artifacts
            ],
            warnings=warnings,
        )


class DraftWriteService:
    """Service for controlled draft/pre-accepted lifecycle artifact writes."""

    def __init__(self, context: RepoContext) -> None:
        self.context = context

    def create_artifact(
        self,
        *,
        artifact_id: str,
        artifact_type: ArtifactType,
        title: str,
        domain: Sequence[str],
        status: ArtifactStatus,
        statement: str,
        authors: Sequence[str],
        tags: Sequence[str],
        depends_on: Sequence[str],
        supersedes: Sequence[str],
        created_at: str | None = None,
        dry_run: bool = False,
    ) -> ArtifactWriteResult:
        """Create a deterministic draft/pre-accepted artifact YAML record."""
        if not domain:
            raise DraftWriteServiceError(
                "at least one --domain value is required",
                code="missing_required_domain",
                remediation="Provide at least one domain for the draft artifact.",
            )
        if status is ArtifactStatus.ACCEPTED:
            raise DraftWriteServiceError(
                "accepted artifacts must be promoted through a dedicated "
                "gate/review workflow",
                code="accepted_write_forbidden",
                remediation=(
                    "Create a draft or pre-accepted artifact first, then use the "
                    "explicit promotion workflow after review and gates pass."
                ),
            )
        if not is_preaccepted_status(status):
            raise DraftWriteServiceError(
                "web draft artifact writes cannot target terminal statuses",
                code="terminal_status_forbidden",
                remediation=(
                    "Create or edit draft/pre-accepted artifacts only. Refuted, "
                    "obsolete, and superseded states require lifecycle workflows."
                ),
            )

        try:
            validate_artifact_id(artifact_id)
        except ValueError as exc:
            raise DraftWriteServiceError(
                str(exc),
                code="invalid_artifact_id",
                remediation="Use a dot-separated lowercase artifact id.",
            ) from exc

        timestamp = _parse_artifact_timestamp(created_at)
        try:
            relative_path = _workspace_lifecycle_artifact_path(
                context=self.context,
                artifact_type=artifact_type,
                status=status,
                artifact_id=artifact_id,
            )
        except ValueError as exc:
            raise DraftWriteServiceError(
                str(exc),
                code="invalid_artifact_target_path",
                remediation=(
                    "Choose a status and artifact type with a valid draft path."
                ),
            ) from exc

        _ensure_artifact_id_is_available(self.context, artifact_id)
        target_path = self.context.resolve(relative_path)
        if target_path.exists():
            raise DraftWriteServiceError(
                f"artifact path already exists: {relative_path.as_posix()}",
                code="artifact_path_exists",
                remediation="Choose a new artifact id or inspect the existing path.",
                details={"path": relative_path.as_posix()},
            )

        try:
            artifact = BaseArtifact.model_validate(
                {
                    "id": artifact_id,
                    "type": artifact_type,
                    "title": title,
                    "domain": list(domain),
                    "status": status,
                    "created_at": timestamp,
                    "updated_at": timestamp,
                    "authors": list(authors),
                    "depends_on": list(depends_on),
                    "supersedes": list(supersedes),
                    "tags": list(tags),
                    "statement": statement,
                    "evidence": [],
                    "review": {"state": "requested", "notes": "Created by CLI."},
                    "risk": {"level": "low", "notes": ""},
                }
            )
        except ValidationError as exc:
            raise DraftWriteServiceError(
                _format_pydantic_errors(exc),
                code="artifact_model_validation_failed",
                remediation="Fix the draft artifact fields and retry.",
            ) from exc

        result = ArtifactWriteResult(artifact=artifact, relative_path=relative_path)
        if dry_run:
            return result

        write_yaml_deterministic(target_path, artifact)
        report = validate_artifact_file(self.context, relative_path)
        if report.ok:
            return result

        target_path.unlink(missing_ok=True)
        raise DraftWriteServiceError(
            _format_report_failures(report),
            code="artifact_file_validation_failed",
            remediation="Fix validation failures before writing the draft artifact.",
        )

    def update_artifact(
        self,
        artifact_id: str,
        *,
        artifact_type: ArtifactType,
        title: str,
        domain: Sequence[str],
        status: ArtifactStatus,
        statement: str,
        authors: Sequence[str],
        tags: Sequence[str],
        depends_on: Sequence[str],
        supersedes: Sequence[str],
        dry_run: bool = False,
    ) -> ArtifactWriteResult:
        """Update editable fields on a draft/pre-accepted artifact in place."""
        if not domain:
            raise DraftWriteServiceError(
                "at least one domain value is required",
                code="missing_required_domain",
                remediation="Provide at least one domain for the draft artifact.",
            )
        if not is_preaccepted_status(status):
            raise DraftWriteServiceError(
                "web artifact updates cannot target accepted or terminal statuses",
                code="accepted_write_forbidden"
                if status is ArtifactStatus.ACCEPTED
                else "terminal_status_forbidden",
                remediation=(
                    "Use draft/pre-accepted status here. Accepted and terminal "
                    "states require review, gates, and lifecycle workflows."
                ),
            )

        loaded = _find_unique_artifact(self.context, artifact_id)
        if loaded.kb_root_readonly:
            raise DraftWriteServiceError(
                f"readonly KB root cannot be modified: {loaded.kb_root_name}",
                code="readonly_kb_root",
                remediation="Choose an artifact in a writable private KB root.",
                details={
                    "kb_root": loaded.kb_root_name or "",
                    "path": loaded.source_path.as_posix(),
                },
            )
        _reject_accepted_path(loaded.source_path)
        _ensure_target_writable(self.context, loaded.source_path)
        artifact = loaded.record
        if not isinstance(artifact, BaseArtifact):
            raise AssertionError("unreachable non-artifact update target")
        if artifact.status is ArtifactStatus.ACCEPTED:
            raise DraftWriteServiceError(
                "accepted artifacts cannot be edited by the draft web editor",
                code="accepted_write_forbidden",
                remediation=(
                    "Accepted artifacts require explicit review and promotion "
                    "workflow changes, not direct web edits."
                ),
                details={
                    "artifact_id": artifact.id,
                    "path": loaded.source_path.as_posix(),
                },
            )
        if artifact.type is not artifact_type:
            raise DraftWriteServiceError(
                "artifact type changes are not supported by this draft editor",
                code="artifact_type_change_forbidden",
                remediation=(
                    "Keep the existing artifact type. Use lifecycle tooling for "
                    "moves that would change the canonical path."
                ),
                details={"artifact_id": artifact.id},
            )

        previous_revision = artifact.content_revision
        payload = artifact.model_dump(mode="json")
        payload.update(
            {
                "type": artifact_type,
                "title": title,
                "domain": list(domain),
                "status": status,
                "updated_at": datetime.now(UTC).replace(microsecond=0),
                "authors": list(authors),
                "depends_on": list(depends_on),
                "supersedes": list(supersedes),
                "tags": list(tags),
                "statement": statement,
            }
        )
        try:
            updated = BaseArtifact.model_validate(payload)
        except ValidationError as exc:
            raise DraftWriteServiceError(
                _format_pydantic_errors(exc),
                code="artifact_model_validation_failed",
                remediation="Fix the draft artifact fields and retry.",
            ) from exc

        if updated.content_revision != previous_revision:
            if (
                artifact.review.state in PROMOTION_REVIEW_STATES
                and updated.review.reviewed_revision is None
            ):
                updated = updated.model_copy(
                    update={
                        "review": updated.review.model_copy(
                            update={"reviewed_revision": previous_revision}
                        )
                    }
                )
            if (
                artifact.alignment is not None
                and artifact.alignment.status == "human_reviewed"
                and updated.alignment is not None
                and updated.alignment.reviewed_revision is None
            ):
                updated = updated.model_copy(
                    update={
                        "alignment": updated.alignment.model_copy(
                            update={"reviewed_revision": previous_revision}
                        )
                    }
                )

        result = ArtifactWriteResult(artifact=updated, relative_path=loaded.source_path)
        if dry_run:
            return result

        target_path = self.context.resolve(loaded.source_path)
        original_text = target_path.read_text(encoding="utf-8")
        write_yaml_deterministic(target_path, updated)
        report = validate_artifact_file(self.context, loaded.source_path)
        if report.ok:
            return result

        target_path.write_text(original_text, encoding="utf-8", newline="\n")
        raise DraftWriteServiceError(
            _format_report_failures(report),
            code="artifact_file_validation_failed",
            remediation="Fix validation failures before updating the draft artifact.",
        )

    def write_artifact_request(
        self,
        request: DraftArtifactWriteRequest,
        *,
        dry_run: bool = False,
    ) -> ControlledWriteResult:
        """Write or preview a controlled draft artifact request."""
        if request.status is ArtifactStatus.ACCEPTED:
            raise DraftWriteServiceError(
                "accepted artifacts cannot be written by draft write commands",
                code="accepted_write_forbidden",
                remediation=(
                    "Use draft status here. Accepted artifacts require review, "
                    "gates, and explicit promotion."
                ),
            )

        relative_path = _workspace_lifecycle_artifact_path(
            context=self.context,
            artifact_type=request.artifact_type,
            status=request.status,
            artifact_id=request.artifact_id,
        )
        _reject_accepted_path(relative_path)
        _ensure_target_writable(self.context, relative_path)

        if dry_run:
            self._validate_artifact_request_without_write(request, relative_path)
            return ControlledWriteResult(
                kind="draft_artifact",
                relative_path=relative_path,
                written_paths=(),
                dry_run=True,
                record_id=request.artifact_id,
            )

        result = self.create_artifact(
            artifact_id=request.artifact_id,
            artifact_type=request.artifact_type,
            title=request.title,
            domain=request.domain,
            status=request.status,
            statement=request.statement,
            authors=request.authors,
            tags=request.tags,
            depends_on=request.depends_on,
            supersedes=request.supersedes,
        )
        return ControlledWriteResult(
            kind="draft_artifact",
            relative_path=result.relative_path,
            written_paths=(result.relative_path,),
            dry_run=False,
            record_id=result.artifact.id,
        )

    def write_source_note(
        self,
        request: Mapping[str, Any],
        *,
        dry_run: bool = False,
    ) -> ControlledWriteResult:
        """Write or preview a staged draft source note."""
        source_id = _required_text(request, "source_id")
        try:
            validate_artifact_id(source_id)
        except ValueError as exc:
            raise DraftWriteServiceError(
                str(exc),
                code="invalid_artifact_id",
                remediation="Use a dot-separated lowercase source note id.",
            ) from exc

        target_path = _repo_local_staging_path(
            request.get("target_path", f"sources/notes/{source_id}.yaml")
        )
        _reject_accepted_path(target_path)
        _ensure_target_writable(self.context, target_path)

        source = _source_metadata_from_request(request)
        now = _now_iso()
        payload = {
            "id": source_id,
            "type": "source_note",
            "status": "draft",
            "created_at": now,
            "updated_at": now,
            "authors": _optional_text_list(request, "authors"),
            "source": source.model_dump(mode="json"),
            "notes": str(request.get("notes", "")).strip(),
        }
        _validate_source_note_payload(payload)

        return _write_controlled_yaml(
            context=self.context,
            kind="source_note",
            relative_path=target_path,
            payload=payload,
            dry_run=dry_run,
            record_id=source_id,
            error_code="source_note_write_failed",
            validator=_validate_source_note_file,
        )

    def append_source_metadata(
        self,
        artifact_id: str,
        request: Mapping[str, Any],
        *,
        dry_run: bool = False,
    ) -> ArtifactWriteResult:
        """Append source metadata to a writable draft/pre-accepted artifact."""
        source = _source_metadata_from_request(request)
        return self._append_artifact_metadata(
            artifact_id,
            source=source,
            evidence=None,
            dry_run=dry_run,
        )

    def append_evidence_metadata(
        self,
        artifact_id: str,
        request: Mapping[str, Any],
        *,
        dry_run: bool = False,
    ) -> ArtifactWriteResult:
        """Append evidence metadata to a writable draft/pre-accepted artifact."""
        evidence_payload: dict[str, Any] = {
            "kind": _required_text(request, "kind"),
            "path": _required_text(request, "path"),
            "summary": _required_text(request, "summary"),
        }
        target_revision = request.get("target_revision")
        if target_revision is not None:
            evidence_payload["target_revision"] = target_revision
        try:
            evidence = Evidence.model_validate(evidence_payload)
        except ValidationError as exc:
            raise DraftWriteServiceError(
                _format_pydantic_errors(exc),
                code="artifact_model_validation_failed",
                remediation="Fix evidence metadata fields and retry.",
            ) from exc
        return self._append_artifact_metadata(
            artifact_id,
            source=None,
            evidence=evidence,
            dry_run=dry_run,
        )

    def write_review_request(
        self,
        request: Mapping[str, Any],
        *,
        dry_run: bool = False,
    ) -> ControlledWriteResult:
        """Write or preview a draft review-request record."""
        review_id = _required_text(request, "review_id")
        status = str(request.get("status", "draft")).strip()
        if status in {"human_reviewed", "accepted"}:
            raise DraftWriteServiceError(
                "controlled review requests cannot mark human review complete",
                code="human_review_forbidden",
                remediation=(
                    "Use status=draft and decision=informational. Human review "
                    "must be recorded by the explicit review workflow."
                ),
            )
        if status != "draft":
            raise DraftWriteServiceError(
                f"unsupported review request status: {status}",
                code="review_request_failed",
                remediation="Use status=draft for controlled review requests.",
            )

        target_path = _repo_local_staging_path(
            f"reviews/requests/{review_id}.yaml"
        )
        _reject_accepted_path(target_path)
        _ensure_target_writable(self.context, target_path)

        now = _now_iso()
        payload = {
            "id": review_id,
            "type": "review",
            "title": _required_text(request, "title"),
            "status": "draft",
            "created_at": now,
            "updated_at": now,
            "authors": _optional_text_list(request, "authors"),
            "target": _required_text(request, "target"),
            "summary": _required_text(request, "summary"),
            "findings": _optional_text_list(request, "findings"),
            "decision": str(request.get("decision", "informational")).strip(),
        }
        try:
            ReviewRecord.model_validate(payload)
        except ValidationError as exc:
            raise DraftWriteServiceError(
                _format_pydantic_errors(exc),
                code="review_request_failed",
                remediation="Fix the review request fields and retry.",
            ) from exc
        if payload["decision"] != "informational":
            raise DraftWriteServiceError(
                "controlled review requests must be informational",
                code="human_review_forbidden",
                remediation=(
                    "Use decision=informational. Approval, rejection, or "
                    "changes-requested decisions require human review."
                ),
            )

        return _write_controlled_yaml(
            context=self.context,
            kind="review_request",
            relative_path=target_path,
            payload=payload,
            dry_run=dry_run,
            record_id=review_id,
            error_code="review_request_failed",
            validator=_validate_review_request_file,
        )

    def write_review_request_from_bundle(
        self,
        bundle_path: str | Path,
        *,
        dry_run: bool = False,
    ) -> ReviewRequestFromBundleResult:
        """Generate and write or preview a draft request from a worker bundle."""
        try:
            bundle = validate_worker_bundle_v2(self.context, bundle_path)
        except WorkerBundleV2Error as exc:
            raise DraftWriteServiceError(
                str(exc),
                code="review_request_failed",
                remediation=(
                    "Fix the worker bundle before generating a review request. "
                    "Bundles cannot claim accepted or human-reviewed authority."
                ),
            ) from exc

        request = _review_request_from_bundle(bundle)
        result = self.write_review_request(request, dry_run=dry_run)
        return ReviewRequestFromBundleResult(
            bundle=bundle,
            request=request,
            write_result=result,
        )

    def append_failure_log_entry(
        self,
        artifact_id: str,
        request: Mapping[str, Any],
        *,
        dry_run: bool = False,
    ) -> ControlledWriteResult:
        """Append or preview one failure-log entry on a writable artifact."""
        _reject_failure_log_authority_claims(request)
        try:
            validate_artifact_id(artifact_id)
        except ValueError as exc:
            raise DraftWriteServiceError(
                str(exc),
                code="invalid_artifact_id",
                remediation="Use a dot-separated lowercase artifact id.",
            ) from exc

        try:
            entry = FailureLogEntry.model_validate(request)
        except ValidationError as exc:
            raise DraftWriteServiceError(
                _format_pydantic_errors(exc),
                code="artifact_model_validation_failed",
                remediation=(
                    "Fix the failure-log entry fields. Failure memory cannot "
                    "claim proof, review, verifier pass, or accepted status."
                ),
            ) from exc

        return self._append_failure_log_entries(
            artifact_id,
            (entry,),
            kind="artifact_failure_log_entry",
            record_id=entry.failure_id,
            dry_run=dry_run,
        )

    def _append_artifact_metadata(
        self,
        artifact_id: str,
        *,
        source: SourceMetadata | None,
        evidence: Evidence | None,
        dry_run: bool,
    ) -> ArtifactWriteResult:
        loaded = _find_unique_artifact(self.context, artifact_id)
        if loaded.kb_root_readonly:
            raise DraftWriteServiceError(
                f"readonly KB root cannot be modified: {loaded.kb_root_name}",
                code="readonly_kb_root",
                remediation="Choose an artifact in a writable private KB root.",
                details={
                    "kb_root": loaded.kb_root_name or "",
                    "path": loaded.source_path.as_posix(),
                },
            )
        _reject_accepted_path(loaded.source_path)
        _ensure_target_writable(self.context, loaded.source_path)
        artifact = loaded.record
        if not isinstance(artifact, BaseArtifact):
            raise AssertionError("unreachable non-artifact metadata target")
        if not is_preaccepted_status(artifact.status):
            raise DraftWriteServiceError(
                "web metadata writes cannot target accepted or terminal artifacts",
                code="accepted_write_forbidden"
                if artifact.status is ArtifactStatus.ACCEPTED
                else "terminal_status_forbidden",
                remediation=(
                    "Attach metadata to draft/pre-accepted artifacts only. "
                    "Accepted and terminal states require lifecycle workflows."
                ),
            )

        payload = artifact.model_dump(mode="json")
        payload["updated_at"] = datetime.now(UTC).replace(microsecond=0)
        if source is not None:
            payload["sources"] = [
                *payload.get("sources", []),
                source.model_dump(mode="json"),
            ]
        if evidence is not None:
            payload["evidence"] = [
                *payload.get("evidence", []),
                evidence.model_dump(mode="json"),
            ]
        try:
            updated = BaseArtifact.model_validate(payload)
        except ValidationError as exc:
            raise DraftWriteServiceError(
                _format_pydantic_errors(exc),
                code="artifact_model_validation_failed",
                remediation="Fix metadata fields and retry.",
            ) from exc

        result = ArtifactWriteResult(artifact=updated, relative_path=loaded.source_path)
        if dry_run:
            return result

        target_path = self.context.resolve(loaded.source_path)
        original_text = target_path.read_text(encoding="utf-8")
        write_yaml_deterministic(target_path, updated)
        report = validate_artifact_file(self.context, loaded.source_path)
        if report.ok:
            return result

        target_path.write_text(original_text, encoding="utf-8", newline="\n")
        raise DraftWriteServiceError(
            _format_report_failures(report),
            code="artifact_file_validation_failed",
            remediation="Fix validation failures before attaching metadata.",
        )

    def plan_failure_log_entries_from_bundle(
        self,
        bundle_path: str | Path,
        *,
        target_artifact_id: str,
    ) -> FailureLogFromBundlePlanResult:
        """Plan failure-log entries from WorkerBundle failed attempts."""
        bundle = _validate_worker_bundle_for_failure_log(self.context, bundle_path)
        loaded = self._failure_log_target(target_artifact_id)
        entries = _failure_log_entries_from_bundle(
            bundle,
            target_artifact_id=target_artifact_id,
        )
        return FailureLogFromBundlePlanResult(
            bundle=bundle,
            artifact_id=target_artifact_id,
            relative_path=loaded.source_path,
            entries=entries,
        )

    def append_failure_log_entries_from_bundle(
        self,
        bundle_path: str | Path,
        *,
        target_artifact_id: str,
        dry_run: bool = False,
    ) -> FailureLogFromBundleWriteResult:
        """Append or preview WorkerBundle-derived failure-log entries."""
        plan = self.plan_failure_log_entries_from_bundle(
            bundle_path,
            target_artifact_id=target_artifact_id,
        )
        result = self._append_failure_log_entries(
            target_artifact_id,
            plan.entries,
            kind="artifact_failure_log_bundle_entries",
            record_id=plan.bundle.bundle_id,
            dry_run=dry_run,
        )
        return FailureLogFromBundleWriteResult(plan=plan, write_result=result)

    def _append_failure_log_entries(
        self,
        artifact_id: str,
        entries: Sequence[FailureLogEntry],
        *,
        kind: str,
        record_id: str,
        dry_run: bool,
    ) -> ControlledWriteResult:
        loaded = self._failure_log_target(artifact_id)
        artifact = loaded.record
        if not isinstance(artifact, BaseArtifact):
            raise AssertionError("unreachable non-artifact failure-log target")

        if dry_run or not entries:
            return ControlledWriteResult(
                kind=kind,
                relative_path=loaded.source_path,
                written_paths=(),
                dry_run=dry_run,
                record_id=record_id,
            )

        target_path = self.context.resolve(loaded.source_path)
        original_text = target_path.read_text(encoding="utf-8")
        updated = artifact.model_copy(
            update={
                "updated_at": datetime.now(UTC).replace(microsecond=0),
                "failure_log": [*artifact.failure_log, *entries],
            }
        )
        write_yaml_deterministic(target_path, updated)
        report = validate_artifact_file(self.context, loaded.source_path)
        if not report.ok:
            target_path.write_text(original_text, encoding="utf-8", newline="\n")
            raise DraftWriteServiceError(
                _format_report_failures(report),
                code="artifact_file_validation_failed",
                remediation="Fix validation failures before writing failure memory.",
            )

        return ControlledWriteResult(
            kind=kind,
            relative_path=loaded.source_path,
            written_paths=(loaded.source_path,),
            dry_run=False,
            record_id=record_id,
        )

    def _failure_log_target(self, artifact_id: str) -> LoadedRecord:
        loaded = _find_unique_artifact_for_failure_log(self.context, artifact_id)
        artifact = loaded.record
        if not isinstance(artifact, BaseArtifact):
            raise AssertionError("unreachable non-artifact failure-log target")

        _reject_accepted_path(loaded.source_path)
        _ensure_target_writable(self.context, loaded.source_path)
        if artifact.status is ArtifactStatus.ACCEPTED:
            raise DraftWriteServiceError(
                "failure-log writes cannot mutate accepted artifacts directly",
                code="accepted_write_forbidden",
                remediation=(
                    "Record failure memory on draft/pre-accepted artifacts. "
                    "Accepted artifacts require ordinary review and promotion "
                    "discipline for any content change."
                ),
                details={
                    "artifact_id": artifact.id,
                    "path": loaded.source_path.as_posix(),
                },
            )
        return loaded

    def _validate_artifact_request_without_write(
        self,
        request: DraftArtifactWriteRequest,
        relative_path: Path,
    ) -> None:
        if not request.domain:
            raise DraftWriteServiceError(
                "at least one domain value is required",
                code="missing_required_domain",
                remediation="Provide at least one domain for the draft artifact.",
            )
        _ensure_artifact_id_is_available(self.context, request.artifact_id)
        if self.context.resolve(relative_path).exists():
            raise DraftWriteServiceError(
                f"artifact path already exists: {relative_path.as_posix()}",
                code="artifact_path_exists",
                remediation="Choose a new artifact id or inspect the existing path.",
                details={"path": relative_path.as_posix()},
            )
        timestamp = datetime.now(UTC).replace(microsecond=0)
        try:
            BaseArtifact.model_validate(
                {
                    "id": request.artifact_id,
                    "type": request.artifact_type,
                    "title": request.title,
                    "domain": request.domain,
                    "status": request.status,
                    "created_at": timestamp,
                    "updated_at": timestamp,
                    "authors": request.authors,
                    "depends_on": request.depends_on,
                    "supersedes": request.supersedes,
                    "tags": request.tags,
                    "statement": request.statement,
                    "evidence": [],
                    "review": {
                        "state": "requested",
                        "notes": "Created by CLI.",
                    },
                    "risk": {"level": "low", "notes": ""},
                }
            )
        except ValidationError as exc:
            raise DraftWriteServiceError(
                _format_pydantic_errors(exc),
                code="artifact_model_validation_failed",
                remediation="Fix the draft artifact fields and retry.",
            ) from exc


HUMAN_REVIEW_DECISIONS = frozenset(
    {
        "accept_for_private_use",
        "accept_for_public_candidate",
        "changes_requested",
        "keep_draft",
        "refute_candidate",
        "mark_obsolete",
    }
)
_ARTIFACT_REVIEW_STATE_BY_DECISION = {
    "accept_for_private_use": "human_reviewed",
    "accept_for_public_candidate": "human_reviewed",
    "changes_requested": "changes_requested",
}
_FORBIDDEN_REVIEWER_PATTERN = re.compile(
    r"\b(ai|agent|claude|codex|gemini|gpt|llm|model|provider|verifier)\b",
    re.IGNORECASE,
)


class ReviewDecisionService:
    """Service for explicit human review decisions."""

    def __init__(self, context: RepoContext) -> None:
        self.context = context

    def write_review_decision(
        self,
        request: Mapping[str, Any],
        *,
        dry_run: bool = False,
    ) -> ReviewDecisionWriteResult:
        """Write or preview one human review decision."""
        artifact_id = _required_text(request, "artifact_id")
        try:
            validate_artifact_id(artifact_id)
        except ValueError as exc:
            raise DraftWriteServiceError(
                str(exc),
                code="invalid_artifact_id",
                remediation="Use a valid artifact id for the review target.",
            ) from exc
        reviewer = _required_text_with_code(
            request,
            "reviewer",
            code="review_reviewer_required",
            remediation="Provide the human reviewer identity.",
        )
        if _FORBIDDEN_REVIEWER_PATTERN.search(reviewer):
            raise DraftWriteServiceError(
                "AI, Codex, provider, agent, or verifier output cannot be "
                "recorded as a human reviewer",
                code="review_reviewer_forbidden",
                remediation="Use an explicit human reviewer identity.",
            )
        decision = _review_decision_value(request.get("decision"))
        notes = _required_text_with_code(
            request,
            "review_notes",
            code="review_notes_required",
            remediation="Record non-empty human review notes.",
        )
        scope = _review_scope(request.get("scope", "private"))
        limitations = str(request.get("limitations", "")).strip()
        if request.get("explicit_human_confirmation") is not True:
            raise DraftWriteServiceError(
                "explicit human review confirmation is required",
                code="explicit_human_confirmation_required",
                remediation=(
                    "Check the box confirming this is a human review decision."
                ),
            )

        loaded = _find_unique_artifact(self.context, artifact_id)
        if loaded.kb_root_readonly:
            raise DraftWriteServiceError(
                f"readonly KB root cannot be modified: {loaded.kb_root_name}",
                code="readonly_kb_root",
                remediation="Choose an artifact in a writable private KB root.",
                details={
                    "kb_root": loaded.kb_root_name or "",
                    "path": loaded.source_path.as_posix(),
                },
            )
        _reject_accepted_path(loaded.source_path)
        _ensure_target_writable(self.context, loaded.source_path)
        artifact = loaded.record
        if not isinstance(artifact, BaseArtifact):
            raise AssertionError("unreachable non-artifact review target")
        if not is_preaccepted_status(artifact.status):
            raise DraftWriteServiceError(
                "human review decisions from the web cannot mutate terminal artifacts",
                code="terminal_status_forbidden",
                remediation=(
                    "Record web human reviews on draft/pre-accepted artifacts. "
                    "Accepted, refuted, obsolete, and superseded changes require "
                    "the lifecycle/promotion workflow."
                ),
            )

        review_id = _review_decision_id(artifact_id)
        relative_path = _repo_local_staging_path(
            f"reviews/decisions/{review_id}.yaml"
        )
        _reject_accepted_path(relative_path)
        _ensure_target_writable(self.context, relative_path)
        target_path = self.context.resolve(relative_path)
        if target_path.exists():
            raise DraftWriteServiceError(
                f"review decision path already exists: {relative_path.as_posix()}",
                code="review_decision_path_exists",
                remediation="Retry to generate a fresh review decision id.",
            )

        now = _now_iso()
        checks = _review_decision_checks(request)
        reviewed_revision = artifact.content_revision
        review_payload = {
            "id": review_id,
            "type": "review",
            "title": f"Human review decision: {artifact.title}",
            "status": "human_reviewed",
            "created_at": now,
            "updated_at": now,
            "authors": [reviewer],
            "target": artifact.id,
            "reviewed_revision": reviewed_revision,
            "summary": f"{reviewer} recorded {decision} for {artifact.id}.",
            "findings": _review_decision_findings(
                decision=decision,
                notes=notes,
                scope=scope,
                limitations=limitations,
                checks=checks,
            ),
            "decision": decision,
        }
        try:
            ReviewRecord.model_validate(review_payload)
        except ValidationError as exc:
            raise DraftWriteServiceError(
                _format_pydantic_errors(exc),
                code="review_decision_failed",
                remediation="Fix the human review decision fields and retry.",
            ) from exc

        updated_artifact = _artifact_after_review_decision(
            artifact,
            decision=decision,
            reviewer=reviewer,
            notes=notes,
            limitations=limitations,
        )
        artifact_updated = updated_artifact != artifact
        if dry_run:
            return ReviewDecisionWriteResult(
                review=review_payload,
                artifact=updated_artifact,
                relative_path=relative_path,
                artifact_relative_path=loaded.source_path,
                written_paths=(),
                dry_run=True,
                record_id=review_id,
                artifact_updated=artifact_updated,
            )

        written_paths = [relative_path]
        artifact_path = self.context.resolve(loaded.source_path)
        original_artifact_text = artifact_path.read_text(encoding="utf-8")
        write_yaml_deterministic(target_path, review_payload)
        try:
            _validate_review_decision_file(self.context, relative_path)
            if artifact_updated:
                write_yaml_deterministic(artifact_path, updated_artifact)
                report = validate_artifact_file(self.context, loaded.source_path)
                if not report.ok:
                    raise DraftWriteServiceError(
                        _format_report_failures(report),
                        code="artifact_file_validation_failed",
                        remediation=(
                            "Fix validation failures before recording review state."
                        ),
                    )
                written_paths.append(loaded.source_path)
        except Exception:
            target_path.unlink(missing_ok=True)
            artifact_path.write_text(
                original_artifact_text,
                encoding="utf-8",
                newline="\n",
            )
            raise

        return ReviewDecisionWriteResult(
            review=review_payload,
            artifact=updated_artifact,
            relative_path=relative_path,
            artifact_relative_path=loaded.source_path,
            written_paths=tuple(written_paths),
            dry_run=False,
            record_id=review_id,
            artifact_updated=artifact_updated,
        )


PROMOTION_REVIEW_STATES = frozenset({"human_reviewed", "accepted"})
PROMOTION_TARGET_STATUSES = {
    "accepted": ArtifactStatus.ACCEPTED,
    "refuted": ArtifactStatus.REFUTED,
    "obsolete": ArtifactStatus.OBSOLETE,
}
PROMOTION_CONFIRMATION_PHRASES = {
    "accepted": "PROMOTE TO ACCEPTED",
    "refuted": "MARK REFUTED",
    "obsolete": "MARK OBSOLETE",
}
_BLOCKING_VERIFIER_STATUSES = frozenset({"fail", "error"})


class PromotionActionService:
    """Service for web-controlled artifact promotion actions."""

    def __init__(self, context: RepoContext) -> None:
        self.context = context

    def preview(
        self,
        artifact_id: str,
        *,
        target_state: str,
        actor: str = "local.web",
    ) -> PromotionActionResult:
        """Preview a target-state promotion without writing repository files."""
        return self._build_result(
            artifact_id,
            target_state=target_state,
            actor=actor,
            dry_run=True,
        )

    def confirm(
        self,
        artifact_id: str,
        *,
        target_state: str,
        actor: str,
    ) -> PromotionActionResult:
        """Execute a target-state promotion through lifecycle policy checks."""
        return self._build_result(
            artifact_id,
            target_state=target_state,
            actor=actor,
            dry_run=False,
        )

    def _build_result(
        self,
        artifact_id: str,
        *,
        target_state: str,
        actor: str,
        dry_run: bool,
    ) -> PromotionActionResult:
        actor = _promotion_actor(actor)
        target_status = _promotion_target_status(target_state)
        validation_report = validate_repository(self.context)
        if not validation_report.ok:
            raise DraftWriteServiceError(
                "repository validation failed before promotion: "
                f"{_format_report_failures(validation_report)}",
                code="promotion_validation_failed",
                remediation="Fix repository validation failures before promotion.",
            )

        loaded = _find_unique_promotion_artifact(validation_report.records, artifact_id)
        artifact = loaded.record
        if not isinstance(artifact, BaseArtifact):
            raise AssertionError("unreachable non-artifact promotion target")
        _ensure_promotion_lifecycle_target(loaded, artifact)
        _ensure_preterminal_for_promotion(artifact)
        transition_decision = evaluate_status_transition(
            validation_report.records,
            artifact,
            target_status,
        )
        if not transition_decision.allowed:
            raise DraftWriteServiceError(
                "; ".join(transition_decision.blocking_messages),
                code="promotion_blocked",
                remediation=(
                    "Update or withdraw downstream accepted artifacts before "
                    "changing this artifact's status."
                ),
                details={
                    "artifact_id": artifact.id,
                    "impacted_dependents": ", ".join(
                        transition_decision.impacted_accepted_dependents
                    ),
                },
            )

        gatekeeper_result = run_gatekeeper(self.context)
        _ensure_gatekeeper_allows_promotion(gatekeeper_result, artifact.id)
        _ensure_artifact_reviewed_for_promotion(
            artifact,
            records=validation_report.records,
        )
        if target_status is ArtifactStatus.ACCEPTED:
            _ensure_promotion_dependencies_accepted(validation_report.records, artifact)
            _ensure_source_metadata_for_public_promotion(self.context, loaded, artifact)

        updated_artifact = artifact.model_copy(
            update={
                "status": target_status,
                "updated_at": datetime.now(UTC).replace(microsecond=0),
            }
        )
        new_relative_path = _workspace_status_move_path(loaded, artifact, target_status)
        old_relative_path = loaded.source_path
        planned_files = tuple(
            dict.fromkeys(
                [old_relative_path, new_relative_path],
            )
        )
        source_path = self.context.resolve(old_relative_path)
        before_text = source_path.read_text(encoding="utf-8")
        after_text = dump_yaml_deterministic(updated_artifact)
        result = PromotionActionResult(
            artifact=artifact,
            updated_artifact=updated_artifact,
            target_state=target_status,
            old_status=artifact.status,
            old_relative_path=old_relative_path,
            new_relative_path=new_relative_path,
            planned_files=planned_files,
            written_paths=(),
            dry_run=True,
            actor=actor,
            yaml_diff=_unified_diff_text(
                before_text,
                after_text,
                fromfile=old_relative_path.as_posix(),
                tofile=new_relative_path.as_posix(),
            ),
            validation_summary="validation passed",
            gate_summary=(
                "gate passed"
                if gatekeeper_result.report.verdict == "pass"
                else "gate failed"
            ),
        )
        if dry_run:
            return result

        self._write_promotion(updated_artifact, old_relative_path, new_relative_path)
        return PromotionActionResult(
            artifact=result.artifact,
            updated_artifact=result.updated_artifact,
            target_state=result.target_state,
            old_status=result.old_status,
            old_relative_path=result.old_relative_path,
            new_relative_path=result.new_relative_path,
            planned_files=result.planned_files,
            written_paths=planned_files,
            dry_run=False,
            actor=result.actor,
            yaml_diff=result.yaml_diff,
            validation_summary=result.validation_summary,
            gate_summary=result.gate_summary,
        )

    def _write_promotion(
        self,
        updated_artifact: BaseArtifact,
        source_relative_path: Path,
        target_relative_path: Path,
    ) -> None:
        source_path = self.context.resolve(source_relative_path)
        target_path = self.context.resolve(target_relative_path)
        if target_path.exists() and source_path != target_path:
            raise DraftWriteServiceError(
                "target artifact path already exists: "
                f"{target_relative_path.as_posix()}",
                code="promotion_target_exists",
                remediation="Inspect the target lifecycle path before retrying.",
                details={"path": target_relative_path.as_posix()},
            )

        original_text = source_path.read_text(encoding="utf-8")
        write_yaml_deterministic(source_path, updated_artifact)
        try:
            if source_path != target_path:
                target_path.parent.mkdir(parents=True, exist_ok=True)
                source_path.rename(target_path)
            report = validate_artifact_file(self.context, target_relative_path)
            if not report.ok:
                raise DraftWriteServiceError(
                    _format_report_failures(report),
                    code="artifact_file_validation_failed",
                    remediation="Fix validation failures before promotion.",
                )
            repo_report = validate_repository(self.context)
            if not repo_report.ok:
                raise DraftWriteServiceError(
                    _format_report_failures(repo_report),
                    code="promotion_validation_failed",
                    remediation="Fix repository validation failures before promotion.",
                )
        except Exception:
            if source_path != target_path and target_path.exists():
                target_path.rename(source_path)
            source_path.write_text(original_text, encoding="utf-8", newline="\n")
            raise


def _parse_artifact_timestamp(value: str | None) -> datetime:
    if value is None:
        return datetime.now(UTC).replace(microsecond=0)
    try:
        normalized = value.replace("Z", "+00:00") if value.endswith("Z") else value
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise DraftWriteServiceError(
            f"invalid --created-at timestamp: {value}",
            code="invalid_timestamp",
            remediation="Use an ISO 8601 timestamp with timezone information.",
        ) from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise DraftWriteServiceError(
            "--created-at must include timezone information",
            code="timestamp_missing_timezone",
            remediation="Use an ISO 8601 timestamp with a timezone or trailing Z.",
        )
    return parsed.astimezone(UTC).replace(microsecond=0)


def _now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _required_text(request: Mapping[str, Any], key: str) -> str:
    value = request.get(key)
    if not isinstance(value, str) or not value.strip():
        raise DraftWriteServiceError(
            f"missing required text field: {key}",
            code="draft_write_failed",
            remediation=f"Provide a non-empty `{key}` field in the input JSON.",
        )
    return value.strip()


def _required_text_with_code(
    request: Mapping[str, Any],
    key: str,
    *,
    code: str,
    remediation: str,
) -> str:
    value = request.get(key)
    if not isinstance(value, str) or not value.strip():
        raise DraftWriteServiceError(
            f"missing required text field: {key}",
            code=code,
            remediation=remediation,
        )
    return value.strip()


def _review_decision_value(value: object) -> str:
    if not isinstance(value, str):
        raise DraftWriteServiceError(
            "review decision must be a string",
            code="review_decision_invalid",
            remediation="Choose one of the supported human review decisions.",
        )
    normalized = value.strip()
    if normalized not in HUMAN_REVIEW_DECISIONS:
        raise DraftWriteServiceError(
            f"unsupported review decision: {normalized}",
            code="review_decision_invalid",
            remediation="Choose one of the supported human review decisions.",
            details={"decision": normalized},
        )
    return normalized


def _review_scope(value: object) -> str:
    normalized = str(value).strip() if value is not None else "private"
    if normalized not in {"private", "public"}:
        raise DraftWriteServiceError(
            "review scope must be private or public",
            code="review_scope_invalid",
            remediation="Use scope=private or scope=public.",
        )
    return normalized


def _review_decision_checks(request: Mapping[str, Any]) -> dict[str, bool]:
    return {
        "dependencies_checked": request.get("dependencies_checked") is True,
        "sources_checked": request.get("sources_checked") is True,
        "evidence_checked": request.get("evidence_checked") is True,
        "gate_state_acknowledged": request.get("gate_state_acknowledged") is True,
    }


def _review_decision_findings(
    *,
    decision: str,
    notes: str,
    scope: str,
    limitations: str,
    checks: Mapping[str, bool],
) -> list[str]:
    findings = [
        f"decision={decision}",
        f"scope={scope}",
        f"review_notes={notes}",
        *[f"{key}={value}" for key, value in checks.items()],
        "authority=human review decision; not accepted status, gate pass, "
        "verifier pass, or promotion authority",
    ]
    if limitations:
        findings.append(f"limitations={limitations}")
    return findings


def _artifact_after_review_decision(
    artifact: BaseArtifact,
    *,
    decision: str,
    reviewer: str,
    notes: str,
    limitations: str,
) -> BaseArtifact:
    next_state = _ARTIFACT_REVIEW_STATE_BY_DECISION.get(decision)
    if next_state is None:
        return artifact
    review_notes = f"{reviewer} recorded {decision}: {notes}"
    if limitations:
        review_notes = f"{review_notes} Limitations: {limitations}"
    payload = artifact.model_dump(mode="json")
    payload["updated_at"] = datetime.now(UTC).replace(microsecond=0)
    payload["review"] = {
        "state": next_state,
        "notes": review_notes,
        "reviewed_revision": artifact.content_revision,
    }
    try:
        return BaseArtifact.model_validate(payload)
    except ValidationError as exc:
        raise DraftWriteServiceError(
            _format_pydantic_errors(exc),
            code="artifact_model_validation_failed",
            remediation="Fix the generated artifact review-state update.",
        ) from exc


def _review_decision_id(artifact_id: str) -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%d%H%M%S%f")
    return ".".join(["review", "decision", *artifact_id.split("."), timestamp])


def _optional_text_list(request: Mapping[str, Any], key: str) -> list[str]:
    value = request.get(key, [])
    if value is None:
        return []
    if not isinstance(value, list):
        raise DraftWriteServiceError(
            f"`{key}` must be a list of strings",
            code="draft_write_failed",
            remediation=f"Use a JSON array of strings for `{key}`.",
        )
    result: list[str] = []
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, str):
            raise DraftWriteServiceError(
                f"`{key}` must contain only strings",
                code="draft_write_failed",
                remediation=f"Use a JSON array of strings for `{key}`.",
            )
        normalized = item.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result.append(normalized)
    return result


def _repo_local_staging_path(value: Any) -> Path:
    if not isinstance(value, str | Path):
        raise DraftWriteServiceError(
            "target path must be a repository-local path",
            code="invalid_staging_path",
            remediation="Use a relative repository path ending in .yaml or .yml.",
        )
    raw = str(value)
    normalized = normalize_repo_path(raw)
    is_absolute = Path(raw).is_absolute() or PureWindowsPath(raw).is_absolute()
    parts = PurePosixPath(normalized).parts
    if (
        not normalized
        or normalized == "."
        or is_absolute
        or normalized == ".."
        or normalized.startswith("../")
        or ".." in parts
    ):
        raise DraftWriteServiceError(
            f"invalid repository-local staging path: {raw}",
            code="invalid_staging_path",
            remediation="Use a relative path inside the repository.",
        )
    path = Path(normalized)
    if path.suffix.lower() not in {".yaml", ".yml"}:
        raise DraftWriteServiceError(
            f"staging path must be YAML: {normalized}",
            code="invalid_staging_path",
            remediation="Use a .yaml or .yml staging path.",
        )
    return path


def _reject_accepted_path(path: str | Path) -> None:
    normalized = normalize_repo_path(path)
    parts = PurePosixPath(normalized).parts
    if parts and parts[0] == "kb" and "accepted" in parts:
        raise DraftWriteServiceError(
            f"controlled write cannot target accepted knowledge: {normalized}",
            code="accepted_write_forbidden",
            remediation=(
                "Write draft/proposal/staging outputs only. Accepted knowledge "
                "requires review, gates, and explicit promotion."
            ),
            details={"path": normalized},
        )


def _ensure_target_writable(context: RepoContext, relative_path: Path) -> None:
    try:
        resolved = context.resolve(relative_path)
        resolved.relative_to(context.repo_root)
    except ValueError:
        raise DraftWriteServiceError(
            f"target path is outside the repository: {relative_path}",
            code="invalid_staging_path",
            remediation="Use a repository-local target path.",
        ) from None

    root = context.kb_root_for_path(relative_path)
    if root is not None and root.readonly:
        raise DraftWriteServiceError(
            f"readonly KB root cannot be modified: {root.name}",
            code="readonly_kb_root",
            remediation=(
                "Choose a writable private KB root or staging path outside the "
                "readonly root."
            ),
            details={"kb_root": root.name, "path": relative_path.as_posix()},
        )


def _source_metadata_from_request(request: Mapping[str, Any]) -> SourceMetadata:
    source_payload = {
        "kind": _required_text(request, "kind"),
        "title": str(request.get("title", "")).strip(),
        "authors": _optional_text_list(request, "authors"),
        "year": request.get("year"),
        "doi": str(request.get("doi", "")).strip(),
        "arxiv": str(request.get("arxiv", "")).strip(),
        "url": str(request.get("url", "")).strip(),
        "theorem_number": str(request.get("theorem_number", "")).strip(),
        "page": str(request.get("page", "")).strip(),
        "notes": str(request.get("notes", "")).strip(),
    }
    try:
        return SourceMetadata.model_validate(source_payload)
    except ValidationError as exc:
        raise DraftWriteServiceError(
            _format_pydantic_errors(exc),
            code="source_note_write_failed",
            remediation="Fix source metadata fields and retry.",
        ) from exc


def _validate_source_note_payload(payload: Mapping[str, Any]) -> None:
    source_id = payload.get("id")
    if not isinstance(source_id, str):
        raise DraftWriteServiceError(
            "source note id must be a string",
            code="source_note_write_failed",
            remediation="Provide a valid source_id.",
        )
    try:
        validate_artifact_id(source_id)
        SourceMetadata.model_validate(payload.get("source"))
    except (TypeError, ValueError, ValidationError) as exc:
        raise DraftWriteServiceError(
            str(exc),
            code="source_note_write_failed",
            remediation="Fix the staged source-note metadata and retry.",
        ) from exc
    if payload.get("type") != "source_note" or payload.get("status") != "draft":
        raise DraftWriteServiceError(
            "source note staging records must be type=source_note and status=draft",
            code="source_note_write_failed",
            remediation="Use the controlled source-note request shape.",
        )


def _write_controlled_yaml(
    *,
    context: RepoContext,
    kind: str,
    relative_path: Path,
    payload: Mapping[str, Any],
    dry_run: bool,
    record_id: str,
    error_code: str,
    validator: Callable[[RepoContext, Path], None],
) -> ControlledWriteResult:
    target_path = context.resolve(relative_path)
    if target_path.exists():
        raise DraftWriteServiceError(
            f"target path already exists: {relative_path.as_posix()}",
            code=error_code,
            remediation="Choose a new staging id or inspect the existing file.",
            details={"path": relative_path.as_posix()},
        )
    if dry_run:
        return ControlledWriteResult(
            kind=kind,
            relative_path=relative_path,
            written_paths=(),
            dry_run=True,
            record_id=record_id,
        )

    write_yaml_deterministic(target_path, dict(payload))
    try:
        validator(context, relative_path)
    except DraftWriteServiceError:
        target_path.unlink(missing_ok=True)
        raise
    except Exception as exc:
        target_path.unlink(missing_ok=True)
        raise DraftWriteServiceError(
            str(exc),
            code=error_code,
            remediation="Fix the staged YAML fields and retry.",
        ) from exc

    return ControlledWriteResult(
        kind=kind,
        relative_path=relative_path,
        written_paths=(relative_path,),
        dry_run=False,
        record_id=record_id,
    )


def _read_yaml_mapping(context: RepoContext, relative_path: Path) -> dict[str, Any]:
    path = context.resolve(relative_path)
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except YAMLError as exc:
        raise DraftWriteServiceError(
            f"invalid YAML: {exc}",
            code="draft_write_failed",
            remediation="Fix the generated YAML and retry.",
        ) from exc
    if not isinstance(raw, dict):
        raise DraftWriteServiceError(
            "generated YAML must be a mapping",
            code="draft_write_failed",
            remediation="Fix the generated YAML and retry.",
        )
    return raw


def _validate_source_note_file(context: RepoContext, relative_path: Path) -> None:
    _validate_source_note_payload(_read_yaml_mapping(context, relative_path))


def _validate_review_request_file(context: RepoContext, relative_path: Path) -> None:
    raw = _read_yaml_mapping(context, relative_path)
    try:
        record = ReviewRecord.model_validate(raw)
    except ValidationError as exc:
        raise DraftWriteServiceError(
            _format_pydantic_errors(exc),
            code="review_request_failed",
            remediation="Fix the staged review request and retry.",
        ) from exc
    if record.status != "draft" or record.decision != "informational":
        raise DraftWriteServiceError(
            "review requests must remain draft informational records",
            code="human_review_forbidden",
            remediation="Do not mark human review or approval through this command.",
        )


def _validate_review_decision_file(context: RepoContext, relative_path: Path) -> None:
    raw = _read_yaml_mapping(context, relative_path)
    try:
        record = ReviewRecord.model_validate(raw)
    except ValidationError as exc:
        raise DraftWriteServiceError(
            _format_pydantic_errors(exc),
            code="review_decision_failed",
            remediation="Fix the human review decision record and retry.",
        ) from exc
    if (
        record.status != "human_reviewed"
        or record.decision not in HUMAN_REVIEW_DECISIONS
    ):
        raise DraftWriteServiceError(
            "review decisions must be explicit human-reviewed decision records",
            code="review_decision_failed",
            remediation="Use the web human review decision workflow.",
        )


def _review_request_from_bundle(bundle: WorkerBundleV2) -> dict[str, Any]:
    review_id = f"review.request.{bundle.bundle_id}"
    findings = _review_findings_from_bundle(bundle)
    return {
        "review_id": review_id,
        "title": f"Review request for {bundle.bundle_id}",
        "status": "draft",
        "authors": ["cosheaf-cli"],
        "target": bundle.task_id,
        "summary": (
            f"Draft informational review request generated from WorkerBundle "
            f"{bundle.bundle_id}: {bundle.summary}"
        ),
        "findings": findings,
        "decision": "informational",
    }


def _review_findings_from_bundle(bundle: WorkerBundleV2) -> list[str]:
    findings = [
        "review_request_limitation: Generated from WorkerBundle output; this is "
        "not human review, verifier evidence, accepted knowledge, or promotion "
        "authority.",
        f"worker_role: {bundle.worker_role.value}",
        f"confidence: {bundle.confidence.value}",
    ]
    findings.extend(f"used_artifact: {item}" for item in bundle.used_artifacts)
    findings.extend(f"used_source: {item}" for item in bundle.used_sources)
    findings.extend(f"claim: {item}" for item in bundle.claims)
    findings.extend(
        f"proposed_artifact: {artifact.path} | {artifact.summary}"
        for artifact in bundle.proposed_artifacts
    )
    findings.extend(f"assumption: {item}" for item in bundle.assumptions)
    findings.extend(f"uncertainty: {item}" for item in bundle.uncertainty)
    findings.extend(
        f"verification_request: {item}" for item in bundle.verification_requests
    )
    findings.extend(f"failed_attempt: {item}" for item in bundle.failed_attempts)
    findings.extend(
        f"counterexample_candidate: {item}" for item in bundle.counterexamples
    )
    findings.extend(
        f"counterexample_candidate: {candidate.candidate_id} "
        f"status={candidate.status.value} "
        f"target={candidate.target_claim or 'none'} "
        f"evidence_paths={','.join(candidate.evidence_paths) or 'none'} "
        f"verifier_request_ids={','.join(candidate.verifier_request_ids) or 'none'} "
        f"summary={candidate.construction_summary} "
        f"limitations={candidate.limitations}"
        for candidate in bundle.counterexample_candidates
    )
    findings.extend(bundle.failures_or_counterexamples)
    findings.extend(
        f"dependency_question: {item}" for item in bundle.dependency_questions
    )
    findings.extend(f"risk: {item}" for item in bundle.risk_flags)
    findings.extend(f"next_step: {item}" for item in bundle.next_steps)
    return _dedupe_preserving_order(findings)


def _reject_failure_log_authority_claims(request: Mapping[str, Any]) -> None:
    forbidden_claims = {
        "accepted",
        "accepted_status",
        "artifact_status",
        "checked_counterexample",
        "counterexample_checked",
        "counterexample_status",
        "human_review",
        "human_reviewed",
        "review",
        "review_state",
        "verifier_pass",
        "verifier_result",
        "verifier_status",
    }
    present = sorted(forbidden_claims.intersection(request.keys()))
    if "status" in request and str(request["status"]).strip() == "accepted":
        present.append("status")
    if not present:
        return
    raise DraftWriteServiceError(
        "failure-log input cannot claim review, accepted, verifier, or checked "
        "counterexample authority",
        code="authority_claim_forbidden",
        remediation=(
            "Keep failure-log entries to failed-attempt memory. Record human "
            "review, verifier pass, checked counterexamples, and accepted status "
            "through their dedicated evidence and promotion workflows."
        ),
        details={"forbidden_fields": ",".join(present)},
    )


def _find_unique_artifact_for_failure_log(
    context: RepoContext,
    artifact_id: str,
) -> LoadedRecord:
    return _find_unique_artifact(context, artifact_id)


def _find_unique_artifact(
    context: RepoContext,
    artifact_id: str,
) -> LoadedRecord:
    records = _load_records_for_lifecycle(context)
    matches = [record for record in records if record.id == artifact_id]
    if not matches:
        raise DraftWriteServiceError(
            f"artifact not found: {artifact_id}",
            code="artifact_not_found",
            remediation="Check the artifact id and rerun the command.",
            details={"artifact_id": artifact_id},
        )
    if len(matches) > 1:
        paths = ", ".join(sorted(record.source_path.as_posix() for record in matches))
        raise DraftWriteServiceError(
            f"duplicate artifact id {artifact_id}: {paths}",
            code="repository_load_failed",
            remediation="Fix duplicate artifact ids before writing failure memory.",
            details={"artifact_id": artifact_id},
        )
    loaded = matches[0]
    if not isinstance(loaded.record, BaseArtifact):
        raise DraftWriteServiceError(
            f"record is not an artifact: {artifact_id}",
            code="artifact_model_validation_failed",
            remediation="Choose a lifecycle artifact id.",
            details={"artifact_id": artifact_id},
        )
    return loaded


def _validate_worker_bundle_for_failure_log(
    context: RepoContext,
    bundle_path: str | Path,
) -> WorkerBundleV2:
    try:
        return validate_worker_bundle_v2(context, bundle_path)
    except WorkerBundleV2Error as exc:
        message = str(exc)
        code = "failure_log_from_bundle_failed"
        if "accepted knowledge" in message or "human_reviewed" in message:
            code = "accepted_write_forbidden"
        raise DraftWriteServiceError(
            message,
            code=code,
            remediation=(
                "Fix the WorkerBundle before deriving artifact failure memory. "
                "Bundles cannot claim accepted knowledge, human review, verifier "
                "passes, or checked refutations through this path."
            ),
        ) from exc


def _failure_log_entries_from_bundle(
    bundle: WorkerBundleV2,
    *,
    target_artifact_id: str,
) -> tuple[FailureLogEntry, ...]:
    candidate_ids = [
        candidate.candidate_id for candidate in bundle.counterexample_candidates
    ]
    next_directions = list(bundle.next_steps)
    entries: list[FailureLogEntry] = []
    for index, failed_attempt in enumerate(bundle.failed_attempts, start=1):
        entries.append(
            FailureLogEntry.model_validate(
                {
                    "failure_id": f"failure.{bundle.bundle_id}.{index:04d}",
                    "attempted_at": bundle.created_at,
                    "recorded_by": (
                        f"worker_bundle:{bundle.bundle_id}:"
                        f"{bundle.worker_role.value}"
                    ),
                    "origin": "imported_bundle",
                    "attempt_kind": _attempt_kind_for_worker_role(
                        bundle.worker_role
                    ),
                    "target": target_artifact_id,
                    "direction": failed_attempt,
                    "summary": (
                        f"WorkerBundle {bundle.bundle_id} recorded failed "
                        f"attempt: {failed_attempt}"
                    ),
                    "failed_because": failed_attempt,
                    "evidence_paths": [],
                    "related_verifier_results": [],
                    "related_counterexample_candidates": candidate_ids,
                    "next_possible_directions": next_directions,
                    "status": "open",
                    "limitations": (
                        f"Imported from WorkerBundle {bundle.bundle_id}; this "
                        "failure memory is not proof, verifier success, checked "
                        "counterexample evidence, human review, gate success, "
                        "accepted status, or promotion evidence."
                    ),
                }
            )
        )
    return tuple(entries)


def _attempt_kind_for_worker_role(worker_role: WorkerType) -> str:
    if worker_role is WorkerType.REASONER:
        return "proof_attempt"
    if worker_role is WorkerType.VERIFIER:
        return "verifier_attempt"
    if worker_role is WorkerType.COUNTEREXAMPLER:
        return "counterexample_search"
    if worker_role is WorkerType.CONSTRUCTION_SEARCHER:
        return "construction_attempt"
    if worker_role is WorkerType.FORMALIZER:
        return "formalization_attempt"
    if worker_role is WorkerType.LITERATURE_SCOUT:
        return "retrieval_attempt"
    return "other"


def _dedupe_preserving_order(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _promotion_actor(actor: str) -> str:
    normalized = actor.strip()
    if not normalized:
        raise DraftWriteServiceError(
            "promotion actor is required",
            code="promotion_actor_required",
            remediation="Provide the human actor identity confirming promotion.",
        )
    if _FORBIDDEN_REVIEWER_PATTERN.search(normalized):
        raise DraftWriteServiceError(
            "AI, agent, Claude, Codex, Gemini, GPT, LLM, model, provider, or "
            "verifier output cannot be recorded as a promotion actor",
            code="promotion_actor_forbidden",
            remediation="Use an explicit human actor identity.",
        )
    return normalized


def _promotion_target_status(target_state: str) -> ArtifactStatus:
    normalized = target_state.strip()
    try:
        return PROMOTION_TARGET_STATUSES[normalized]
    except KeyError as exc:
        raise DraftWriteServiceError(
            "target_state must be accepted, refuted, or obsolete",
            code="invalid_promotion_target_state",
            remediation="Choose accepted, refuted, or obsolete.",
        ) from exc


def _find_unique_promotion_artifact(
    records: tuple[LoadedRecord, ...],
    artifact_id: str,
) -> LoadedRecord:
    try:
        normalized_id = validate_artifact_id(artifact_id)
    except ValueError as exc:
        raise DraftWriteServiceError(
            str(exc),
            code="invalid_artifact_id",
            remediation="Use a valid artifact id for promotion.",
        ) from exc
    matches = [record for record in records if record.id == normalized_id]
    if not matches:
        raise DraftWriteServiceError(
            f"artifact not found: {normalized_id}",
            code="artifact_not_found",
            remediation="Check the artifact id before promotion.",
            details={"artifact_id": normalized_id},
        )
    if len(matches) > 1:
        paths = ", ".join(sorted(record.source_path.as_posix() for record in matches))
        raise DraftWriteServiceError(
            f"duplicate artifact id {normalized_id}: {paths}",
            code="repository_load_failed",
            remediation="Fix duplicate artifact ids before promotion.",
            details={"artifact_id": normalized_id},
        )
    loaded = matches[0]
    if not isinstance(loaded.record, BaseArtifact):
        raise DraftWriteServiceError(
            f"record is not a promotable lifecycle artifact: {normalized_id}",
            code="promotion_blocked",
            remediation="Choose a lifecycle artifact id.",
            details={"artifact_id": normalized_id},
        )
    try:
        lifecycle_artifact_path(
            loaded.record.type,
            ArtifactStatus.ACCEPTED,
            loaded.record.id,
        )
    except ValueError as exc:
        raise DraftWriteServiceError(
            f"record is not a promotable lifecycle artifact: {normalized_id}",
            code="promotion_blocked",
            remediation="Choose a lifecycle artifact type.",
            details={"artifact_id": normalized_id},
        ) from exc
    return loaded


def _ensure_promotion_lifecycle_target(
    loaded: LoadedRecord,
    artifact: BaseArtifact,
) -> None:
    if loaded.kb_relative_path is None:
        raise DraftWriteServiceError(
            "promotion is only supported for records under a KB root",
            code="promotion_blocked",
            remediation="Move the record under a lifecycle KB root before promotion.",
            details={"path": loaded.source_path.as_posix()},
        )
    _ensure_promotion_path_status_consistent(loaded, artifact)
    if loaded.kb_root_readonly:
        raise DraftWriteServiceError(
            f"readonly KB root cannot be modified: {loaded.kb_root_name}",
            code="readonly_kb_root",
            remediation="Choose an artifact in a writable KB root.",
            details={
                "kb_root": loaded.kb_root_name or "",
                "path": loaded.source_path.as_posix(),
            },
        )


def _ensure_promotion_path_status_consistent(
    loaded: LoadedRecord,
    artifact: BaseArtifact,
) -> None:
    allowed = expected_status_for_path(_status_path_for_loaded_record(loaded))
    if artifact.status in allowed:
        return
    expected = ", ".join(sorted(status.value for status in allowed))
    raise DraftWriteServiceError(
        "status/path mismatch: "
        f"{loaded.source_path.as_posix()} has status {artifact.status.value}; "
        f"expected one of {expected}",
        code="promotion_blocked",
        remediation="Fix the lifecycle path/status mismatch before promotion.",
        details={"path": loaded.source_path.as_posix()},
    )


def _ensure_preterminal_for_promotion(artifact: BaseArtifact) -> None:
    blocker = promotion_status_blocker(artifact)
    if blocker is None:
        return
    code, _ = blocker
    if code == "already_accepted" or artifact.status is ArtifactStatus.ACCEPTED:
        raise DraftWriteServiceError(
            f"artifact is already accepted: {artifact.id}",
            code="promotion_blocked",
            remediation="Choose a pre-accepted artifact for promotion.",
        )
    if not is_preaccepted_status(artifact.status):
        raise DraftWriteServiceError(
            "only pre-accepted lifecycle artifacts may be promoted: "
            f"{artifact.id} has status {artifact.status.value}",
            code="promotion_blocked",
            remediation="Choose a draft or pre-accepted artifact for promotion.",
        )


def _ensure_gatekeeper_allows_promotion(
    result: GatekeeperRunResult,
    artifact_id: str,
) -> None:
    target_blockers = _target_verifier_blockers(result, artifact_id)
    if target_blockers:
        raise DraftWriteServiceError(
            "target verifier result blocks promotion: "
            f"{'; '.join(target_blockers)}",
            code="promotion_blocked",
            remediation="Fix failed or errored target verifier results first.",
        )
    if result.report.blocking_issues:
        raise DraftWriteServiceError(
            "gatekeeper blocking issues prevent promotion: "
            f"{_format_gatekeeper_blocking_issues(result)}",
            code="promotion_blocked",
            remediation="Fix blocking gate issues before promotion.",
        )


def _target_verifier_blockers(
    result: GatekeeperRunResult,
    artifact_id: str,
) -> list[str]:
    blockers: list[str] = []
    for gate in result.report.gates:
        if gate.gate_id != "G6":
            continue
        for detail in gate.details:
            if detail.get("artifact_id") != artifact_id:
                continue
            status = detail.get("status")
            if status not in _BLOCKING_VERIFIER_STATUSES:
                continue
            verifier = str(detail.get("verifier", "verifier"))
            message = str(detail.get("message", "")).strip()
            rendered = f"{verifier} {status}"
            if message:
                rendered = f"{rendered}: {message}"
            blockers.append(rendered)
    if blockers:
        return blockers
    return [
        issue.message
        for issue in result.report.blocking_issues
        if issue.gate_id == "G6" and issue.artifact_id == artifact_id
    ]


def _format_gatekeeper_blocking_issues(result: GatekeeperRunResult) -> str:
    return "; ".join(
        f"{issue.gate_id} | {issue.source_path or '-'} | "
        f"{issue.artifact_id or '-'} | {issue.message}"
        for issue in result.report.blocking_issues
    )


def _ensure_artifact_reviewed_for_promotion(
    artifact: BaseArtifact,
    *,
    records: tuple[LoadedRecord, ...] = (),
) -> None:
    if artifact.review.state not in PROMOTION_REVIEW_STATES:
        raise DraftWriteServiceError(
            "missing_review: review.state must be human_reviewed or accepted "
            f"before promotion; {artifact.id} has {artifact.review.state}",
            code="promotion_blocked",
            remediation="Record an explicit human review before promotion.",
            details={"artifact_id": artifact.id, "review_state": artifact.review.state},
        )
    mismatch = review_revision_mismatch(artifact, records)
    if mismatch is not None:
        raise DraftWriteServiceError(
            f"stale_review: {mismatch}",
            code="promotion_blocked",
            remediation=(
                "Record a fresh human review for the current artifact revision "
                "before promotion."
            ),
            details={
                "artifact_id": artifact.id,
                "review_state": artifact.review.state,
                "current_revision": artifact.content_revision,
                "reviewed_revision": artifact.review.reviewed_revision or "",
            },
        )
    evidence_mismatches = evidence_revision_mismatches(artifact)
    if evidence_mismatches:
        raise DraftWriteServiceError(
            f"stale_evidence: {'; '.join(msg for _, msg in evidence_mismatches)}",
            code="promotion_blocked",
            remediation=(
                "Re-run verification or attach evidence matching the current "
                "artifact revision before promotion."
            ),
            details={
                "artifact_id": artifact.id,
                "current_revision": artifact.content_revision,
            },
        )


def _ensure_promotion_dependencies_accepted(
    records: tuple[LoadedRecord, ...],
    artifact: BaseArtifact,
) -> None:
    artifacts_by_id = {
        record.id: record
        for record in records
        if isinstance(record.record, BaseArtifact)
    }
    for dependency_id in artifact.depends_on:
        if is_external_dependency_ref(dependency_id):
            continue
        dependency = artifacts_by_id.get(dependency_id)
        if dependency is None or not isinstance(dependency.record, BaseArtifact):
            raise DraftWriteServiceError(
                f"dependency is missing: {dependency_id}",
                code="promotion_blocked",
                remediation="Resolve missing dependencies before promotion.",
                details={"dependency_id": dependency_id},
            )
        if dependency.record.status is ArtifactStatus.ACCEPTED:
            continue
        raise DraftWriteServiceError(
            "dependency is not accepted: "
            f"{dependency_id} has status {dependency.record.status.value} "
            f"at {dependency.source_path.as_posix()}",
            code="promotion_blocked",
            remediation=(
                "Promote or externalize dependencies before accepted promotion."
            ),
            details={"dependency_id": dependency_id},
        )


def _ensure_source_metadata_for_public_promotion(
    context: RepoContext,
    loaded: LoadedRecord,
    artifact: BaseArtifact,
) -> None:
    if not context.workspace_config.policy.accepted_requires_source:
        return
    if loaded.kb_root_name != "public":
        return
    missing = missing_required_source_metadata(artifact)
    if not missing:
        return
    if missing == ("sources",):
        message = (
            "missing_source_metadata: accepted public artifact requires source "
            f"metadata before promotion: {artifact.id}"
        )
    else:
        message = (
            "missing_source_metadata: accepted public artifact has incomplete "
            "source metadata before promotion: " + ", ".join(missing)
        )
    raise DraftWriteServiceError(
        message,
        code="promotion_blocked",
        remediation="Add reviewed source metadata before public accepted promotion.",
        details={"artifact_id": artifact.id, "missing": ",".join(missing)},
    )


def _workspace_lifecycle_artifact_path(
    *,
    context: RepoContext,
    artifact_type: ArtifactType,
    status: ArtifactStatus,
    artifact_id: str,
) -> Path:
    legacy_path = lifecycle_artifact_path(artifact_type, status, artifact_id)
    if not context.workspace_config.configured:
        return legacy_path
    root = _default_writable_kb_root(context)
    return Path(root.path) / _strip_legacy_kb_prefix(legacy_path)


def _workspace_status_move_path(
    loaded: LoadedRecord,
    artifact: BaseArtifact,
    new_status: ArtifactStatus,
) -> Path:
    legacy_path = lifecycle_artifact_path(artifact.type, new_status, artifact.id)
    if loaded.kb_root_path is None:
        return legacy_path
    return loaded.kb_root_path / _strip_legacy_kb_prefix(legacy_path)


def _status_path_for_loaded_record(loaded: LoadedRecord) -> str:
    if loaded.kb_relative_path is None:
        return loaded.source_path.as_posix()
    relative = loaded.kb_relative_path.as_posix()
    return "kb" if not relative else f"kb/{relative}"


def _unified_diff_text(
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


def _strip_legacy_kb_prefix(path: Path) -> Path:
    parts = path.parts
    if not parts or parts[0] != "kb":
        return path
    return Path(*parts[1:])


def _default_writable_kb_root(context: RepoContext) -> KbRootConfig:
    for root in context.workspace_config.ordered_kb:
        if not root.readonly and root.name == "private":
            return root
    for root in context.workspace_config.ordered_kb:
        if not root.readonly:
            return root
    raise DraftWriteServiceError(
        "no writable KB root is configured",
        code="no_writable_kb_root",
        remediation="Configure a writable private KB root before writing drafts.",
    )


def _ensure_artifact_id_is_available(context: RepoContext, artifact_id: str) -> None:
    records = _load_records_for_lifecycle(context)
    if any(record.id == artifact_id for record in records):
        raise DraftWriteServiceError(
            f"artifact already exists: {artifact_id}",
            code="artifact_id_exists",
            remediation="Choose a new artifact id or update the existing draft.",
            details={"artifact_id": artifact_id},
        )


def _load_records_for_lifecycle(context: RepoContext) -> tuple[LoadedRecord, ...]:
    try:
        return tuple(load_artifacts(context))
    except LoadError as exc:
        raise DraftWriteServiceError(
            f"cannot load repository records: {exc}",
            code="repository_load_failed",
            remediation="Fix repository load errors before writing new drafts.",
        ) from exc


def _format_pydantic_errors(exc: ValidationError) -> str:
    return "; ".join(
        f"{'.'.join(str(part) for part in error['loc'])}: {error['msg']}"
        for error in exc.errors()
    )


def _format_report_failures(report: ValidationReport) -> str:
    return "; ".join(
        f"{failure.gate} | {failure.source_path or '-'} | "
        f"{failure.artifact_id or '-'} | {failure.message}"
        for failure in report.failures
    )


__all__ = [
    "ArtifactWriteResult",
    "BundleValidationService",
    "ContextPackService",
    "ContextSendPolicyService",
    "ControlledWriteResult",
    "DraftWriteService",
    "DraftWriteServiceError",
    "FailureLogFromBundlePlanResult",
    "FailureLogFromBundleWriteResult",
    "GateService",
    "KbRootInfo",
    "MemorySearchService",
    "OrchestratorPlanService",
    "PROMOTION_CONFIRMATION_PHRASES",
    "PromotionActionResult",
    "PromotionActionService",
    "ReviewDecisionService",
    "ReviewDecisionWriteResult",
    "ReviewRequestFromBundleResult",
    "ServiceError",
    "TaskService",
    "ValidationService",
    "WorkspaceInfoResult",
    "WorkspaceService",
]
