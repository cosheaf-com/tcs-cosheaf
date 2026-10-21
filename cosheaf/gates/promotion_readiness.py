"""Read-only promotion readiness reporting."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from cosheaf.core.artifact import (
    BaseArtifact,
    evidence_revision_mismatches,
    is_external_dependency_ref,
    review_revision_mismatch,
)
from cosheaf.core.paths import repo_relative_posix
from cosheaf.core.status import ArtifactStatus
from cosheaf.gates.gatekeeper import (
    GateIssue,
    GatekeeperRunResult,
    run_gatekeeper,
    validate_repository,
)
from cosheaf.gates.source_metadata_gate import missing_required_source_metadata
from cosheaf.storage.loader import IssueRecord, LoadedRecord
from cosheaf.storage.repo import RepoContext
from cosheaf.strategy.models import StrategyTaskStatus
from cosheaf.strategy.storage import load_strategy_plans
from cosheaf.verification.counterexample_evidence import (
    load_checked_counterexample_evidence,
)

PROMOTION_REVIEW_STATES = frozenset({"human_reviewed", "accepted"})
BLOCKING_VERIFIER_STATUSES = frozenset({"fail", "error"})
UNRESOLVED_FAILURE_MEMORY_STATUSES = frozenset({"open"})
DRAFT_STATUSES = frozenset({ArtifactStatus.RAW, ArtifactStatus.DRAFT})
READY_PREACCEPTED_STATUSES = frozenset(
    {
        ArtifactStatus.LOCALLY_TESTED,
        ArtifactStatus.ADVERSARIALLY_TESTED,
        ArtifactStatus.MACHINE_CHECKED,
        ArtifactStatus.HUMAN_REVIEWED,
    }
)

ReasonSeverity = Literal["blocking", "warning"]


@dataclass(frozen=True)
class PromotionReadinessReason:
    """One readiness reason for a target artifact or issue."""

    code: str
    severity: ReasonSeverity
    message: str
    artifact_id: str = ""
    source_path: str = ""
    gate_id: str = ""
    verifier: str = ""
    status: str = ""

    @property
    def blocking(self) -> bool:
        return self.severity == "blocking"

    def to_dict(self) -> dict[str, str]:
        return {
            "code": self.code,
            "severity": self.severity,
            "message": self.message,
            "artifact_id": self.artifact_id,
            "source_path": self.source_path,
            "gate_id": self.gate_id,
            "verifier": self.verifier,
            "status": self.status,
        }


@dataclass(frozen=True)
class ArtifactPromotionReadiness:
    """Read-only readiness report for one artifact."""

    artifact_id: str
    source_path: str
    artifact_type: str
    status: str
    kb_root: str
    kb_root_readonly: bool
    review_state: str
    checker_required: bool
    source_metadata_required: bool
    missing_source_metadata: tuple[str, ...]
    gate_verdict: str
    verifier_results: tuple[dict[str, Any], ...]
    reasons: tuple[PromotionReadinessReason, ...]

    @property
    def ready(self) -> bool:
        return not any(reason.blocking for reason in self.reasons)

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "source_path": self.source_path,
            "type": self.artifact_type,
            "status": self.status,
            "kb_root": self.kb_root,
            "kb_root_readonly": self.kb_root_readonly,
            "review_state": self.review_state,
            "checker_required": self.checker_required,
            "source_metadata_required": self.source_metadata_required,
            "missing_source_metadata": list(self.missing_source_metadata),
            "gate_verdict": self.gate_verdict,
            "verifier_results": list(self.verifier_results),
            "ready": self.ready,
            "reasons": [reason.to_dict() for reason in self.reasons],
        }


@dataclass(frozen=True)
class PromotionReadinessReport:
    """Read-only promotion readiness report."""

    target_mode: Literal["artifact", "issue"]
    artifact_id: str
    issue_id: str
    gate_report_json_path: str
    gate_report_markdown_path: str
    artifacts: tuple[ArtifactPromotionReadiness, ...]
    reasons: tuple[PromotionReadinessReason, ...] = ()
    accepted_write_performed: Literal[False] = False
    schema_version: int = 1

    @property
    def ready(self) -> bool:
        return bool(self.artifacts) and not any(
            reason.blocking
            for reason in (
                list(self.reasons)
                + [reason for item in self.artifacts for reason in item.reasons]
            )
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "target": {
                "mode": self.target_mode,
                "artifact_id": self.artifact_id,
                "issue_id": self.issue_id,
            },
            "ready": self.ready,
            "accepted_write_performed": self.accepted_write_performed,
            "gate_report_json_path": self.gate_report_json_path,
            "gate_report_markdown_path": self.gate_report_markdown_path,
            "reasons": [reason.to_dict() for reason in self.reasons],
            "artifacts": [artifact.to_dict() for artifact in self.artifacts],
        }


def build_promotion_readiness_report(
    context: RepoContext,
    *,
    artifact_id: str | None = None,
    issue_id: str | None = None,
) -> PromotionReadinessReport:
    """Build a read-only promotion readiness report."""
    if (artifact_id is None) == (issue_id is None):
        raise ValueError("provide exactly one of artifact_id or issue_id")

    validation = validate_repository(context)
    gatekeeper = run_gatekeeper(context)
    records = validation.records

    if artifact_id is not None:
        artifacts = _artifact_reports(
            context,
            records=records,
            gatekeeper=gatekeeper,
            artifact_ids=(artifact_id,),
        )
        top_reasons = _missing_target_reasons(records, (artifact_id,))
        return PromotionReadinessReport(
            target_mode="artifact",
            artifact_id=artifact_id,
            issue_id="",
            gate_report_json_path=_repo_path(context, gatekeeper.json_path),
            gate_report_markdown_path=_repo_path(context, gatekeeper.markdown_path),
            artifacts=artifacts,
            reasons=top_reasons,
        )

    assert issue_id is not None
    issue = _find_issue(records, issue_id)
    if issue is None:
        return PromotionReadinessReport(
            target_mode="issue",
            artifact_id="",
            issue_id=issue_id,
            gate_report_json_path=_repo_path(context, gatekeeper.json_path),
            gate_report_markdown_path=_repo_path(context, gatekeeper.markdown_path),
            artifacts=(),
            reasons=(
                PromotionReadinessReason(
                    code="issue_not_found",
                    severity="blocking",
                    message=f"issue not found: {issue_id}",
                ),
            ),
        )

    related_artifacts = tuple(issue.related_artifacts)
    if not related_artifacts:
        return PromotionReadinessReport(
            target_mode="issue",
            artifact_id="",
            issue_id=issue_id,
            gate_report_json_path=_repo_path(context, gatekeeper.json_path),
            gate_report_markdown_path=_repo_path(context, gatekeeper.markdown_path),
            artifacts=(),
            reasons=(
                PromotionReadinessReason(
                    code="issue_has_no_related_artifacts",
                    severity="blocking",
                    message=(
                        "issue has no related_artifacts to evaluate for "
                        "promotion readiness"
                    ),
                ),
            ),
        )

    artifacts = _artifact_reports(
        context,
        records=records,
        gatekeeper=gatekeeper,
        artifact_ids=related_artifacts,
    )
    return PromotionReadinessReport(
        target_mode="issue",
        artifact_id="",
        issue_id=issue_id,
        gate_report_json_path=_repo_path(context, gatekeeper.json_path),
        gate_report_markdown_path=_repo_path(context, gatekeeper.markdown_path),
        artifacts=artifacts,
        reasons=_missing_target_reasons(records, related_artifacts),
    )


def _artifact_reports(
    context: RepoContext,
    *,
    records: tuple[LoadedRecord, ...],
    gatekeeper: GatekeeperRunResult,
    artifact_ids: tuple[str, ...],
) -> tuple[ArtifactPromotionReadiness, ...]:
    reports: list[ArtifactPromotionReadiness] = []
    for target_id in artifact_ids:
        loaded = _find_artifact(records, target_id)
        if loaded is None:
            continue
        artifact = loaded.record
        if not isinstance(artifact, BaseArtifact):
            continue
        reports.append(
            _artifact_report(
                context,
                records=records,
                gatekeeper=gatekeeper,
                loaded=loaded,
                artifact=artifact,
            )
        )
    return tuple(sorted(reports, key=lambda report: report.artifact_id))


def _artifact_report(
    context: RepoContext,
    *,
    records: tuple[LoadedRecord, ...],
    gatekeeper: GatekeeperRunResult,
    loaded: LoadedRecord,
    artifact: BaseArtifact,
) -> ArtifactPromotionReadiness:
    checker_required = artifact.verification_policy.require_lean_check
    source_metadata_required = _source_metadata_required(context, loaded)
    missing_source_metadata = (
        missing_required_source_metadata(artifact)
        if source_metadata_required
        else ()
    )
    verifier_results = _target_verifier_results(gatekeeper, artifact.id)
    reasons = (
        _status_reasons(artifact)
        + _review_reasons(artifact, records)
        + _evidence_revision_reasons(artifact)
        + _readonly_reasons(loaded)
        + _source_metadata_reasons(
            artifact,
            source_metadata_required=source_metadata_required,
            missing=missing_source_metadata,
        )
        + _dependency_reasons(records, loaded, artifact)
        + _verifier_reasons(
            artifact,
            verifier_results=verifier_results,
            checker_required=checker_required,
        )
        + _failure_memory_reasons(artifact)
        + _checked_counterexample_evidence_reasons(context, artifact)
        + _strategy_open_blocker_reasons(context, artifact)
        + _repository_gate_reasons(gatekeeper, artifact.id)
        + _target_gate_reasons(gatekeeper, artifact.id)
    )
    return ArtifactPromotionReadiness(
        artifact_id=artifact.id,
        source_path=loaded.source_path.as_posix(),
        artifact_type=artifact.type.value,
        status=artifact.status.value,
        kb_root=loaded.kb_root_name or "",
        kb_root_readonly=loaded.kb_root_readonly,
        review_state=artifact.review.state,
        checker_required=checker_required,
        source_metadata_required=source_metadata_required,
        missing_source_metadata=missing_source_metadata,
        gate_verdict=gatekeeper.report.verdict,
        verifier_results=verifier_results,
        reasons=_dedupe_reasons(reasons),
    )


def promotion_status_blocker(artifact: BaseArtifact) -> tuple[str, str] | None:
    """Shared promotion status policy used by readiness, preview, and execution."""
    if artifact.status in READY_PREACCEPTED_STATUSES:
        return None
    if artifact.status in DRAFT_STATUSES:
        if artifact.review.state in PROMOTION_REVIEW_STATES:
            return None
        return (
            "draft_status",
            (
                "artifact is still raw/draft; readiness requires a "
                "review-grade pre-accepted status before accepted promotion"
            ),
        )
    if artifact.status is ArtifactStatus.ACCEPTED:
        return (
            "already_accepted",
            "artifact is already accepted; no promotion readiness remains",
        )
    return (
        "not_promotable_status",
        f"artifact status is not promotable: {artifact.status.value}",
    )


def _status_reasons(artifact: BaseArtifact) -> tuple[PromotionReadinessReason, ...]:
    blocker = promotion_status_blocker(artifact)
    if blocker is None:
        return ()
    code, message = blocker
    return (
        PromotionReadinessReason(
            code=code,
            severity="blocking",
            artifact_id=artifact.id,
            status=artifact.status.value,
            message=message,
        ),
    )


def _review_reasons(
    artifact: BaseArtifact,
    records: tuple[LoadedRecord, ...] = (),
) -> tuple[PromotionReadinessReason, ...]:
    if artifact.review.state not in PROMOTION_REVIEW_STATES:
        return (
            PromotionReadinessReason(
                code="missing_review",
                severity="blocking",
                artifact_id=artifact.id,
                status=artifact.review.state,
                message=(
                    "review.state must be human_reviewed or accepted; AI/provider "
                    "output cannot satisfy the human review requirement"
                ),
            ),
        )
    mismatch = review_revision_mismatch(artifact, records)
    if mismatch is not None:
        return (
            PromotionReadinessReason(
                code="stale_review",
                severity="blocking",
                artifact_id=artifact.id,
                status=artifact.review.state,
                message=mismatch,
            ),
        )
    return ()


def _evidence_revision_reasons(
    artifact: BaseArtifact,
) -> tuple[PromotionReadinessReason, ...]:
    return tuple(
        PromotionReadinessReason(
            code="stale_evidence",
            severity="blocking",
            artifact_id=artifact.id,
            source_path=evidence_path,
            message=mismatch,
        )
        for evidence_path, mismatch in evidence_revision_mismatches(artifact)
    )


def _readonly_reasons(loaded: LoadedRecord) -> tuple[PromotionReadinessReason, ...]:
    if not loaded.kb_root_readonly:
        return ()
    return (
        PromotionReadinessReason(
            code="readonly_kb_root",
            severity="blocking",
            artifact_id=loaded.id,
            source_path=loaded.source_path.as_posix(),
            message=(
                "artifact is loaded from a readonly KB root; promotion would "
                "need to occur in a writable lifecycle root"
            ),
        ),
    )


def _source_metadata_required(context: RepoContext, loaded: LoadedRecord) -> bool:
    return (
        context.workspace_config.policy.accepted_requires_source
        and loaded.kb_root_name == "public"
    )


def _source_metadata_reasons(
    artifact: BaseArtifact,
    *,
    source_metadata_required: bool,
    missing: tuple[str, ...],
) -> tuple[PromotionReadinessReason, ...]:
    if not source_metadata_required or not missing:
        return ()
    message = (
        "accepted public artifact requires source metadata before promotion"
        if missing == ("sources",)
        else "accepted public artifact has incomplete source metadata: "
        + ", ".join(missing)
    )
    return (
        PromotionReadinessReason(
            code="missing_source_metadata",
            severity="blocking",
            artifact_id=artifact.id,
            message=message,
        ),
    )


def _dependency_reasons(
    records: tuple[LoadedRecord, ...],
    loaded: LoadedRecord,
    artifact: BaseArtifact,
) -> tuple[PromotionReadinessReason, ...]:
    artifacts_by_id = {
        record.id: record
        for record in records
        if isinstance(record.record, BaseArtifact)
    }
    reasons: list[PromotionReadinessReason] = []
    for dependency_id in artifact.depends_on:
        if is_external_dependency_ref(dependency_id):
            continue
        dependency = artifacts_by_id.get(dependency_id)
        if dependency is None or not isinstance(dependency.record, BaseArtifact):
            reasons.append(
                PromotionReadinessReason(
                    code="dependency_risk",
                    severity="blocking",
                    artifact_id=artifact.id,
                    message=f"dependency is missing: {dependency_id}",
                )
            )
            continue
        if loaded.kb_root_name == "public" and dependency.kb_root_name == "private":
            reasons.append(
                PromotionReadinessReason(
                    code="private_dependency",
                    severity="blocking",
                    artifact_id=artifact.id,
                    source_path=dependency.source_path.as_posix(),
                    message=(
                        "public artifact depends on private artifact: "
                        f"{dependency_id}"
                    ),
                )
            )
        if dependency.record.status is ArtifactStatus.ACCEPTED:
            continue
        reasons.append(
            PromotionReadinessReason(
                code="dependency_risk",
                severity="blocking",
                artifact_id=artifact.id,
                source_path=dependency.source_path.as_posix(),
                status=dependency.record.status.value,
                message=(
                    "dependency is not accepted: "
                    f"{dependency_id} has status {dependency.record.status.value}"
                ),
            )
        )
    return tuple(reasons)


def _verifier_reasons(
    artifact: BaseArtifact,
    *,
    verifier_results: tuple[dict[str, Any], ...],
    checker_required: bool,
) -> tuple[PromotionReadinessReason, ...]:
    reasons: list[PromotionReadinessReason] = []
    for result in verifier_results:
        status = str(result.get("status", ""))
        verifier = str(result.get("verifier", "verifier"))
        message = str(result.get("message", "")).strip()
        if status in BLOCKING_VERIFIER_STATUSES:
            reasons.append(
                PromotionReadinessReason(
                    code="failed_verifier",
                    severity="blocking",
                    artifact_id=artifact.id,
                    verifier=verifier,
                    status=status,
                    message=f"{verifier} {status}: {message}".strip(),
                )
            )
        if status == "skipped":
            severity: ReasonSeverity = "blocking" if checker_required else "warning"
            suffix = (
                "Skipped verifier evidence is not a pass; checker-required "
                "readiness remains unsatisfied."
                if checker_required
                else "Skipped verifier evidence is not a pass."
            )
            rendered_message = f"{verifier} skipped: {message}".strip()
            reasons.append(
                PromotionReadinessReason(
                    code="skipped_verifier",
                    severity=severity,
                    artifact_id=artifact.id,
                    verifier=verifier,
                    status=status,
                    message=f"{rendered_message} {suffix}",
                )
            )
    return tuple(reasons)


def _failure_memory_reasons(
    artifact: BaseArtifact,
) -> tuple[PromotionReadinessReason, ...]:
    reasons: list[PromotionReadinessReason] = []
    for entry in sorted(
        artifact.failure_log,
        key=lambda item: (item.attempted_at, item.failure_id),
        reverse=True,
    ):
        if entry.status not in UNRESOLVED_FAILURE_MEMORY_STATUSES:
            continue
        next_text = (
            "; ".join(entry.next_possible_directions)
            if entry.next_possible_directions
            else "-"
        )
        reasons.append(
            PromotionReadinessReason(
                code="unresolved_failure_memory",
                severity="warning",
                artifact_id=artifact.id,
                status=entry.status,
                message=(
                    "unresolved artifact failure memory only; not verifier "
                    "evidence and not a promotion blocker by itself: "
                    f"{entry.direction}; failed_because={entry.failed_because}; "
                    f"origin={entry.origin}; kind={entry.attempt_kind}; "
                    f"next={next_text}"
                ),
            )
        )
    return tuple(reasons)


def _checked_counterexample_evidence_reasons(
    context: RepoContext,
    artifact: BaseArtifact,
) -> tuple[PromotionReadinessReason, ...]:
    reasons: list[PromotionReadinessReason] = []
    for loaded in load_checked_counterexample_evidence(context):
        evidence = loaded.record
        if evidence.target_artifact_id != artifact.id:
            continue
        support = []
        if evidence.verifier_evidence_ids:
            support.append(
                "verifier_evidence="
                + ",".join(sorted(evidence.verifier_evidence_ids))
            )
        if evidence.review_record_paths:
            support.append(
                "review_records=" + ",".join(sorted(evidence.review_record_paths))
            )
        if evidence.evidence_paths:
            support.append(
                "evidence_paths=" + ",".join(sorted(evidence.evidence_paths))
            )
        reasons.append(
            PromotionReadinessReason(
                code="checked_counterexample_evidence",
                severity="warning",
                artifact_id=artifact.id,
                source_path=loaded.relative_path.as_posix(),
                status=evidence.checked_result.value,
                message=(
                    "checked counterexample evidence for review only; not human "
                    "review, accepted refutation, accepted status, or promotion "
                    "authority, and not a promotion blocker by itself: "
                    f"{evidence.evidence_id}; result={evidence.checked_result.value}; "
                    f"candidate={evidence.candidate_id}; support="
                    f"{'; '.join(support) if support else '-'}"
                ),
            )
        )
    return tuple(reasons)


def _strategy_open_blocker_reasons(
    context: RepoContext,
    artifact: BaseArtifact,
) -> tuple[PromotionReadinessReason, ...]:
    reasons: list[PromotionReadinessReason] = []
    for loaded in load_strategy_plans(context):
        plan = loaded.plan
        if artifact.id not in set(plan.problem.target_artifacts):
            related_to_artifact = any(
                artifact.id in set(node.related_artifacts)
                for node in plan.graph.nodes
            )
            if not related_to_artifact:
                continue
        for node in plan.graph.nodes:
            if node.status not in {
                StrategyTaskStatus.BLOCKED,
                StrategyTaskStatus.FAILED,
            }:
                continue
            related = set(node.related_artifacts)
            targets = set(plan.problem.target_artifacts)
            if artifact.id not in related and artifact.id not in targets:
                continue
            reasons.append(
                PromotionReadinessReason(
                    code="strategy_open_blocker",
                    severity="warning",
                    artifact_id=artifact.id,
                    source_path=loaded.relative_path.as_posix(),
                    status=node.status.value,
                    message=(
                        "advisory strategy blocker only; strategy plans are "
                        "review context and not a promotion blocker by itself: "
                        f"{plan.plan_id} {node.node_id} {node.title}"
                    ),
                )
            )
    return tuple(reasons)


def _target_gate_reasons(
    gatekeeper: GatekeeperRunResult,
    artifact_id: str,
) -> tuple[PromotionReadinessReason, ...]:
    reasons: list[PromotionReadinessReason] = []
    handled_gate_ids = {"G4", "G6", "G9"}
    for issue in gatekeeper.report.blocking_issues:
        if issue.artifact_id != artifact_id:
            continue
        if issue.gate_id in handled_gate_ids:
            continue
        reasons.append(_reason_from_gate_issue(issue))
    return tuple(reasons)


def _repository_gate_reasons(
    gatekeeper: GatekeeperRunResult,
    artifact_id: str,
) -> tuple[PromotionReadinessReason, ...]:
    reasons: list[PromotionReadinessReason] = []
    for issue in gatekeeper.report.blocking_issues:
        if issue.artifact_id == artifact_id:
            continue
        location = issue.artifact_id or issue.source_path or "repository"
        reasons.append(
            PromotionReadinessReason(
                code="repository_gate_blocker",
                severity="blocking",
                artifact_id=artifact_id,
                source_path=issue.source_path,
                gate_id=issue.gate_id,
                message=(
                    "repository gatekeeper blocker prevents promotion: "
                    f"{issue.gate_id} {location}: {issue.message}"
                ),
            )
        )
    return tuple(reasons)


def _reason_from_gate_issue(issue: GateIssue) -> PromotionReadinessReason:
    code = "gate_blocker"
    if issue.gate_id == "G10":
        code = "formal_link_policy"
    return PromotionReadinessReason(
        code=code,
        severity="blocking",
        artifact_id=issue.artifact_id,
        source_path=issue.source_path,
        gate_id=issue.gate_id,
        message=issue.message,
    )


def _target_verifier_results(
    gatekeeper: GatekeeperRunResult,
    artifact_id: str,
) -> tuple[dict[str, Any], ...]:
    details: list[dict[str, Any]] = []
    for gate in gatekeeper.report.gates:
        if gate.gate_id != "G6":
            continue
        for detail in gate.details:
            if detail.get("artifact_id") == artifact_id:
                details.append(dict(detail))
    return tuple(
        sorted(
            details,
            key=lambda detail: (
                str(detail.get("verifier", "")),
                str(detail.get("status", "")),
                str(detail.get("message", "")),
            ),
        )
    )


def _missing_target_reasons(
    records: tuple[LoadedRecord, ...],
    artifact_ids: tuple[str, ...],
) -> tuple[PromotionReadinessReason, ...]:
    available_ids = {record.id for record in records}
    reasons = [
        PromotionReadinessReason(
            code="artifact_not_found",
            severity="blocking",
            artifact_id=artifact_id,
            message=f"artifact not found: {artifact_id}",
        )
        for artifact_id in artifact_ids
        if artifact_id not in available_ids
    ]
    return tuple(reasons)


def _find_artifact(
    records: tuple[LoadedRecord, ...],
    artifact_id: str,
) -> LoadedRecord | None:
    matches = [
        record
        for record in records
        if record.id == artifact_id and isinstance(record.record, BaseArtifact)
    ]
    if not matches:
        return None
    return sorted(matches, key=lambda record: record.source_path.as_posix())[0]


def _find_issue(
    records: tuple[LoadedRecord, ...],
    issue_id: str,
) -> IssueRecord | None:
    matches = [
        record.record
        for record in records
        if record.id == issue_id and isinstance(record.record, IssueRecord)
    ]
    if not matches:
        return None
    return sorted(matches, key=lambda record: record.id)[0]


def _dedupe_reasons(
    reasons: tuple[PromotionReadinessReason, ...],
) -> tuple[PromotionReadinessReason, ...]:
    seen: set[tuple[str, str, str, str, str]] = set()
    deduped: list[PromotionReadinessReason] = []
    for reason in reasons:
        key = (
            reason.code,
            reason.severity,
            reason.message,
            reason.source_path,
            reason.status,
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(reason)
    return tuple(
        sorted(
            deduped,
            key=lambda reason: (
                reason.severity != "blocking",
                reason.code,
                reason.artifact_id,
                reason.source_path,
                reason.message,
            ),
        )
    )


def _repo_path(context: RepoContext, path: Path) -> str:
    try:
        return repo_relative_posix(context.repo_root, path)
    except ValueError:
        return path.as_posix()
