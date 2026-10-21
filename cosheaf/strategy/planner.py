"""Deterministic strategy planner over repository-local research state."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from cosheaf.core.artifact import BaseArtifact
from cosheaf.core.status import ArtifactStatus
from cosheaf.research.run import ResearchRunRecord
from cosheaf.storage.loader import IssueRecord, LoadedRecord, load_artifacts
from cosheaf.storage.repo import RepoContext
from cosheaf.strategy.models import (
    STRATEGY_AUTHORITY_NOTICE,
    StrategyEdgeKind,
    StrategyError,
    StrategyNextStep,
    StrategyPlan,
    StrategyProblem,
    StrategyTaskEdge,
    StrategyTaskGraph,
    StrategyTaskNode,
    StrategyTaskNodeKind,
    StrategyTaskScope,
    StrategyTaskStatus,
)
from cosheaf.verification.counterexample_evidence import (
    CheckedCounterexampleEvidenceLoadResult,
    load_checked_counterexample_evidence,
)


@dataclass(frozen=True)
class StrategyPlanBuildResult:
    """Result from building a strategy plan."""

    plan: StrategyPlan

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "kind": "strategy_plan_build",
            "plan_id": self.plan.plan_id,
            "accepted_write_performed": False,
            "authority_notice": STRATEGY_AUTHORITY_NOTICE,
            "plan": self.plan.to_dict(),
        }


def build_strategy_plan(
    context: RepoContext,
    issue_id: str,
) -> StrategyPlanBuildResult:
    """Build a deterministic strategy plan for one issue."""
    records = tuple(load_artifacts(context))
    issues = {
        loaded.record.id: loaded.record
        for loaded in records
        if isinstance(loaded.record, IssueRecord)
    }
    issue = issues.get(issue_id)
    if issue is None:
        raise StrategyError(
            f"issue not found: {issue_id}",
            code="strategy_issue_not_found",
            remediation="Create the issue record first or pass an existing issue ID.",
            details={"issue_id": issue_id},
        )

    artifacts = {
        loaded.record.id: loaded
        for loaded in records
        if isinstance(loaded.record, BaseArtifact)
    }
    related_ids = _issue_related_artifact_ids(issue.related_artifacts, artifacts)
    dependency_ids = _one_hop_dependency_ids(related_ids, artifacts)
    graph_artifact_ids = tuple(dict.fromkeys((*related_ids, *dependency_ids)))
    related_artifacts = _base_artifacts_for_ids(graph_artifact_ids, artifacts)
    failure_entries = _failure_entries(related_artifacts)
    candidate_ids = _candidate_counterexample_ids(failure_entries)
    checked_evidence = _checked_evidence_for_issue(context, related_ids, candidate_ids)
    research_runs = _research_runs_for_issue(context, issue.id)

    nodes = [
        _context_node(issue.id),
        _validate_node(),
        _gate_node(),
        *_artifact_nodes(graph_artifact_ids, artifacts),
        _failure_review_node(failure_entries),
        _counterexample_review_node(candidate_ids, checked_evidence),
        _research_run_review_node(research_runs),
        _proof_attempt_node(related_ids, failure_entries),
    ]
    graph = StrategyTaskGraph(nodes=tuple(nodes), edges=_edges(nodes))
    next_steps = _rank_next_steps(graph.nodes, failure_entries)
    plan = StrategyPlan(
        plan_id=f"strategy.{issue.id}.plan",
        issue_id=issue.id,
        created_at=issue.updated_at.astimezone(UTC),
        problem=StrategyProblem(
            issue_id=issue.id,
            title=issue.title,
            description=issue.description,
            domains=_domains(related_artifacts),
            tags=tuple(sorted(set(issue.tags))),
            target_artifacts=related_ids,
            known_constraints=(
                STRATEGY_AUTHORITY_NOTICE,
                "Do not write accepted knowledge or bypass review, validation, "
                "gates, or promotion.",
            ),
            public_private_scope_labels=_scopes_for_loaded(
                tuple(artifacts[item] for item in graph_artifact_ids)
            ),
        ),
        graph=graph,
        next_steps=next_steps,
    )
    return StrategyPlanBuildResult(plan=plan)


def _issue_related_artifact_ids(
    related_artifacts: list[str],
    artifacts: dict[str, LoadedRecord],
) -> tuple[str, ...]:
    return tuple(item for item in related_artifacts if item in artifacts)


def _one_hop_dependency_ids(
    artifact_ids: tuple[str, ...],
    artifacts: dict[str, LoadedRecord],
) -> tuple[str, ...]:
    deps: list[str] = []
    for artifact_id in artifact_ids:
        record = artifacts[artifact_id].record
        if not isinstance(record, BaseArtifact):
            continue
        for dependency in record.depends_on:
            if dependency in artifacts and dependency not in deps:
                deps.append(dependency)
    return tuple(deps)


def _base_artifacts_for_ids(
    artifact_ids: tuple[str, ...],
    artifacts: dict[str, LoadedRecord],
) -> tuple[BaseArtifact, ...]:
    records: list[BaseArtifact] = []
    for artifact_id in artifact_ids:
        record = artifacts[artifact_id].record
        if isinstance(record, BaseArtifact):
            records.append(record)
    return tuple(records)


def _failure_entries(
    artifacts: tuple[BaseArtifact, ...],
) -> tuple[tuple[str, BaseArtifact, Any], ...]:
    entries: list[tuple[str, BaseArtifact, Any]] = []
    for artifact in artifacts:
        for entry in artifact.failure_log:
            entries.append((entry.failure_id, artifact, entry))
    return tuple(sorted(entries, key=lambda item: (item[0], item[1].id)))


def _candidate_counterexample_ids(
    failure_entries: tuple[tuple[str, BaseArtifact, Any], ...],
) -> tuple[str, ...]:
    candidates: list[str] = []
    for _failure_id, _artifact, entry in failure_entries:
        for candidate_id in entry.related_counterexample_candidates:
            if candidate_id not in candidates:
                candidates.append(candidate_id)
    return tuple(candidates)


def _checked_evidence_for_issue(
    context: RepoContext,
    target_artifact_ids: tuple[str, ...],
    candidate_ids: tuple[str, ...],
) -> tuple[CheckedCounterexampleEvidenceLoadResult, ...]:
    try:
        records = load_checked_counterexample_evidence(context)
    except (OSError, ValueError, ValidationError):
        return ()
    target_set = set(target_artifact_ids)
    candidate_set = set(candidate_ids)
    return tuple(
        loaded
        for loaded in records
        if loaded.record.target_artifact_id in target_set
        or loaded.record.candidate_id in candidate_set
    )


def _research_runs_for_issue(
    context: RepoContext,
    issue_id: str,
) -> tuple[ResearchRunRecord, ...]:
    root = context.resolve(Path(".cosheaf") / "runs")
    if not root.exists():
        return ()
    runs: list[ResearchRunRecord] = []
    for path in sorted(root.glob("*/run.json")):
        try:
            raw = json.loads(path.read_text(encoding="utf-8-sig"))
            record = ResearchRunRecord.model_validate(raw)
        except (OSError, json.JSONDecodeError, ValueError, ValidationError):
            continue
        if record.issue_id == issue_id:
            runs.append(record)
    return tuple(sorted(runs, key=lambda run: run.run_id))


def _context_node(issue_id: str) -> StrategyTaskNode:
    return StrategyTaskNode(
        node_id="task.context-build",
        kind=StrategyTaskNodeKind.RETRIEVAL_CONTEXT,
        title="Build bounded issue context",
        status=StrategyTaskStatus.READY,
        scope=StrategyTaskScope.WORKSPACE,
        description="Build a bounded context pack before changing research state.",
        expected_evidence_kinds=("context_pack", "retrieval_audit"),
        command=("cosheaf", "context", "build", issue_id),
        notes=("Context build is guidance input only, not review or proof.",),
    )


def _validate_node() -> StrategyTaskNode:
    return StrategyTaskNode(
        node_id="task.validate",
        kind=StrategyTaskNodeKind.VALIDATION,
        title="Run repository validation",
        status=StrategyTaskStatus.READY,
        scope=StrategyTaskScope.WORKSPACE,
        expected_evidence_kinds=("validation_report",),
        command=("cosheaf", "validate"),
        notes=("Validation success is not human review.",),
    )


def _gate_node() -> StrategyTaskNode:
    return StrategyTaskNode(
        node_id="task.gate",
        kind=StrategyTaskNodeKind.GATE,
        title="Run gatekeeper",
        status=StrategyTaskStatus.READY,
        scope=StrategyTaskScope.WORKSPACE,
        expected_evidence_kinds=("gate_report",),
        command=("cosheaf", "gate", "run"),
        notes=("Skipped gate rows are not passes.",),
    )


def _artifact_nodes(
    artifact_ids: tuple[str, ...],
    artifacts: dict[str, LoadedRecord],
) -> tuple[StrategyTaskNode, ...]:
    nodes: list[StrategyTaskNode] = []
    for artifact_id in artifact_ids:
        loaded = artifacts[artifact_id]
        artifact = loaded.record
        if not isinstance(artifact, BaseArtifact):
            continue
        nodes.append(
            StrategyTaskNode(
                node_id=f"artifact.{artifact.id}",
                kind=StrategyTaskNodeKind.UNDERSTAND,
                title=f"Read {artifact.id}",
                status=StrategyTaskStatus.READY,
                scope=_scope_for_loaded(loaded),
                description=artifact.title,
                expected_evidence_kinds=("operator_note",),
                related_artifacts=(artifact.id,),
                input_paths=(loaded.source_path.as_posix(),),
                notes=(
                    f"{artifact.status.value} artifact; use as scoped context only.",
                ),
            )
        )
    return tuple(nodes)


def _failure_review_node(
    failure_entries: tuple[tuple[str, BaseArtifact, Any], ...],
) -> StrategyTaskNode:
    failure_ids = tuple(item[0] for item in failure_entries)
    notes = tuple(
        f"{entry.direction}: known failed direction; avoid retry unless retryable."
        for _failure_id, _artifact, entry in failure_entries
        if entry.status == "open"
    )
    return StrategyTaskNode(
        node_id="task.review-failures",
        kind=StrategyTaskNodeKind.REVIEW_DECISION,
        title="Review known failed directions",
        status=StrategyTaskStatus.READY if failure_ids else StrategyTaskStatus.DEFERRED,
        scope=StrategyTaskScope.WORKSPACE,
        description="Inspect failure memory before selecting a next proof route.",
        expected_evidence_kinds=("failure_log_review",),
        related_failure_log_entries=failure_ids,
        related_artifacts=tuple(item[1].id for item in failure_entries),
        notes=notes
        or ("No issue-related failure memory was found in the current workspace.",),
    )


def _counterexample_review_node(
    candidate_ids: tuple[str, ...],
    checked_evidence: tuple[CheckedCounterexampleEvidenceLoadResult, ...],
) -> StrategyTaskNode:
    checked_ids = tuple(loaded.record.evidence_id for loaded in checked_evidence)
    return StrategyTaskNode(
        node_id="task.counterexample-review",
        kind=StrategyTaskNodeKind.COUNTEREXAMPLE_SEARCH,
        title="Review counterexample candidates and checked evidence",
        status=StrategyTaskStatus.READY
        if candidate_ids or checked_ids
        else StrategyTaskStatus.DEFERRED,
        scope=StrategyTaskScope.WORKSPACE,
        expected_evidence_kinds=(
            "candidate_counterexample",
            "checked_counterexample_evidence",
        ),
        related_candidate_counterexamples=candidate_ids,
        related_checked_counterexample_evidence=checked_ids,
        notes=(
            "Candidate counterexamples are candidate only.",
            "Checked counterexample records are checked evidence only.",
            "Neither label creates accepted refutation or promotion authority.",
        ),
    )


def _research_run_review_node(
    runs: tuple[ResearchRunRecord, ...],
) -> StrategyTaskNode:
    run_ids = tuple(run.run_id for run in runs)
    return StrategyTaskNode(
        node_id="task.review-runs",
        kind=StrategyTaskNodeKind.RUN_REVIEW,
        title="Review related research-run provenance",
        status=StrategyTaskStatus.READY if run_ids else StrategyTaskStatus.DEFERRED,
        scope=StrategyTaskScope.WORKSPACE,
        expected_evidence_kinds=("research_run_record",),
        related_research_run_ids=run_ids,
        notes=("Research runs are provenance only, not proof or review.",),
    )


def _proof_attempt_node(
    related_artifact_ids: tuple[str, ...],
    failure_entries: tuple[tuple[str, BaseArtifact, Any], ...],
) -> StrategyTaskNode:
    has_open_failure = any(
        entry.status == "open" for _fid, _artifact, entry in failure_entries
    )
    note = (
        "Known failed direction present; choose a new route before retrying."
        if has_open_failure
        else "No directly related open failed direction was found."
    )
    return StrategyTaskNode(
        node_id="task.proof-attempt",
        kind=StrategyTaskNodeKind.PROOF_ATTEMPT,
        title="Choose a bounded proof attempt",
        status=(
            StrategyTaskStatus.BLOCKED
            if has_open_failure
            else StrategyTaskStatus.READY
        ),
        scope=StrategyTaskScope.PRIVATE,
        description="Attempt only after reviewing context, gates, and known failures.",
        blocked_by=("task.review-failures",) if has_open_failure else (),
        expected_evidence_kinds=("proof_note", "review_request"),
        related_artifacts=related_artifact_ids,
        notes=(note,),
    )


def _edges(nodes: list[StrategyTaskNode]) -> tuple[StrategyTaskEdge, ...]:
    ids = {node.node_id for node in nodes}
    edges: list[StrategyTaskEdge] = []
    for node in nodes:
        for dependency in node.depends_on:
            if dependency in ids:
                edges.append(
                    StrategyTaskEdge(
                        from_node=dependency,
                        to_node=node.node_id,
                        kind=StrategyEdgeKind.PREREQUISITE,
                        reason="task dependency",
                    )
                )
        for blocker in node.blocked_by:
            if blocker in ids:
                edges.append(
                    StrategyTaskEdge(
                        from_node=blocker,
                        to_node=node.node_id,
                        kind=StrategyEdgeKind.BLOCKED_BY,
                        reason="blocked until reviewed",
                    )
                )
    return tuple(edges)


def _rank_next_steps(
    nodes: tuple[StrategyTaskNode, ...],
    failure_entries: tuple[tuple[str, BaseArtifact, Any], ...],
) -> tuple[StrategyNextStep, ...]:
    scored: list[tuple[float, StrategyTaskNode, tuple[str, ...]]] = []
    for node in nodes:
        if not node.node_id.startswith("task."):
            continue
        score = _base_score(node)
        reasons = [f"{node.kind.value} task"]
        if node.command:
            reasons.append("first-class CLI command")
        if node.related_failure_log_entries:
            reasons.append("surfaces known failed directions")
        if node.related_candidate_counterexamples:
            reasons.append("preserves candidate counterexample labels")
        if node.related_checked_counterexample_evidence:
            reasons.append("preserves checked evidence labels")
        if node.related_research_run_ids:
            reasons.append("uses research-run provenance")
        if node.node_id == "task.proof-attempt" and failure_entries:
            score = min(score, 20.0)
            reasons.append("known failed direction must be reviewed before retry")
        scored.append((score, node, tuple(reasons)))
    ordered = sorted(scored, key=lambda item: (-item[0], item[1].node_id))
    return tuple(
        StrategyNextStep(
            rank=index,
            node_id=node.node_id,
            score=score,
            reasons=reasons,
            command=node.command,
        )
        for index, (score, node, reasons) in enumerate(ordered, start=1)
    )


def _base_score(node: StrategyTaskNode) -> float:
    if node.node_id == "task.context-build":
        return 100.0
    if node.node_id == "task.validate":
        return 95.0
    if node.node_id == "task.gate":
        return 90.0
    if node.node_id == "task.review-failures":
        return 82.0 if node.related_failure_log_entries else 30.0
    if node.node_id == "task.counterexample-review":
        return (
            78.0
            if node.related_candidate_counterexamples
            or node.related_checked_counterexample_evidence
            else 28.0
        )
    if node.node_id == "task.review-runs":
        return 70.0 if node.related_research_run_ids else 25.0
    if node.node_id == "task.proof-attempt":
        return 74.0
    return 10.0


def _domains(artifacts: tuple[BaseArtifact, ...]) -> tuple[str, ...]:
    return tuple(
        sorted({domain for artifact in artifacts for domain in artifact.domain})
    )


def _scopes_for_loaded(
    records: tuple[LoadedRecord, ...],
) -> tuple[StrategyTaskScope, ...]:
    scopes = tuple(dict.fromkeys(_scope_for_loaded(record) for record in records))
    return scopes or (StrategyTaskScope.WORKSPACE,)


def _scope_for_loaded(loaded: LoadedRecord) -> StrategyTaskScope:
    if loaded.is_private_scope:
        return StrategyTaskScope.PRIVATE
    source = loaded.source_path.as_posix()
    if "/public/" in source or source.startswith("kb/public/"):
        return StrategyTaskScope.PUBLIC
    if loaded.is_public_kb_root:
        return StrategyTaskScope.PUBLIC
    if (
        isinstance(loaded.record, BaseArtifact)
        and loaded.record.status is ArtifactStatus.ACCEPTED
    ):
        return StrategyTaskScope.WORKSPACE
    return StrategyTaskScope.WORKSPACE


__all__ = ["StrategyPlanBuildResult", "build_strategy_plan"]
