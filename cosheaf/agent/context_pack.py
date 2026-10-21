"""Context pack generation for bounded Codex task handoffs."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from textwrap import shorten
from typing import Any

from cosheaf.core.artifact import BaseArtifact
from cosheaf.core.paths import repo_relative_posix
from cosheaf.core.status import (
    ArtifactStatus,
    is_accepted_status,
    is_preaccepted_status,
)
from cosheaf.memory import (
    ArtifactCard,
    FullArtifactPull,
    MemoryRootScope,
    RetrievalResult,
    RetrievalRole,
    RetrievedArtifactCard,
    search_artifact_cards,
)
from cosheaf.memory.models import ArtifactCardStatus
from cosheaf.storage.loader import IssueRecord, LoadedRecord, load_artifacts
from cosheaf.storage.repo import RepoContext
from cosheaf.strategy.models import StrategyPlan, StrategyTaskScope, StrategyTaskStatus
from cosheaf.strategy.storage import StrategyPlanStorageResult, load_strategy_plans
from cosheaf.verification.counterexample_evidence import (
    CHECKED_COUNTEREXAMPLE_AUTHORITY_NOTICE,
    CheckedCounterexampleEvidenceLoadResult,
    load_checked_counterexample_evidence,
)

PACK_FILENAMES = (
    "CONTEXT.md",
    "ACCEPTANCE.md",
    "RELEVANT_ARTIFACTS.md",
    "KNOWN_FAILURES.md",
    "FULL_ARTIFACTS.md",
    "RETRIEVAL_AUDIT.json",
    "COMMANDS.md",
)

DEFAULT_COMMANDS = (
    "make lint",
    "make typecheck",
    "make test",
    "make validate",
    "make gate",
)

KNOWN_FAILURE_STATUSES = frozenset(
    {
        ArtifactStatus.REFUTED,
        ArtifactStatus.OBSOLETE,
        ArtifactStatus.SUPERSEDED,
    }
)
KNOWN_FAILURE_CARD_STATUSES = frozenset(
    {
        ArtifactCardStatus.REFUTED,
        ArtifactCardStatus.OBSOLETE,
        ArtifactCardStatus.SUPERSEDED,
    }
)
CONTEXT_MAX_CARDS = 20
ALL_CONTEXT_SCOPES = (
    MemoryRootScope.PUBLIC,
    MemoryRootScope.PRIVATE,
    MemoryRootScope.WORKSPACE,
    MemoryRootScope.FRAMEWORK,
)
PUBLIC_CONTEXT_SCOPES = (
    MemoryRootScope.PUBLIC,
    MemoryRootScope.WORKSPACE,
    MemoryRootScope.FRAMEWORK,
)


class ContextPackError(ValueError):
    """Raised when a context pack cannot be generated from repository data."""


@dataclass(frozen=True)
class ContextPackResult:
    """Written context pack files for an issue."""

    issue_id: str
    task_dir: Path
    files: tuple[Path, ...]


@dataclass(frozen=True)
class RankedArtifact:
    """A selected artifact plus deterministic ranking reasons."""

    record: LoadedRecord
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class ContextPackCard:
    """A retrieved card plus issue-local relevance labels."""

    retrieved: RetrievedArtifactCard
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class ContextFailureEntry:
    """One visible failed-attempt memory entry for context packs."""

    artifact_id: str
    artifact_path: str
    root_scope: str
    failure_id: str
    direction: str
    failed_because: str
    status: str
    next_possible_directions: tuple[str, ...]
    origin: str
    attempt_kind: str
    source_label: str

    def to_dict(self) -> dict[str, object]:
        return {
            "artifact_id": self.artifact_id,
            "artifact_path": self.artifact_path,
            "root_scope": self.root_scope,
            "failure_id": self.failure_id,
            "direction": self.direction,
            "failed_because": self.failed_because,
            "status": self.status,
            "next_possible_directions": list(self.next_possible_directions),
            "origin": self.origin,
            "attempt_kind": self.attempt_kind,
            "source_label": self.source_label,
        }


@dataclass(frozen=True)
class ContextCheckedCounterexampleEvidenceEntry:
    """One visible checked counterexample evidence record for context packs."""

    evidence_id: str
    target_artifact_id: str
    candidate_id: str
    candidate_source: str
    check_method: str
    checked_result: str
    checker: str
    source_path: str
    verifier_evidence_ids: tuple[str, ...]
    review_record_paths: tuple[str, ...]
    evidence_paths: tuple[str, ...]
    limitations: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "evidence_id": self.evidence_id,
            "target_artifact_id": self.target_artifact_id,
            "candidate_id": self.candidate_id,
            "candidate_source": self.candidate_source,
            "check_method": self.check_method,
            "checked_result": self.checked_result,
            "checker": self.checker,
            "source_path": self.source_path,
            "verifier_evidence_ids": list(self.verifier_evidence_ids),
            "review_record_paths": list(self.review_record_paths),
            "evidence_paths": list(self.evidence_paths),
            "limitations": list(self.limitations),
            "authority_notice": CHECKED_COUNTEREXAMPLE_AUTHORITY_NOTICE,
        }


@dataclass(frozen=True)
class ContextStrategyPlanEntry:
    """One visible strategy-plan summary for context packs."""

    plan_id: str
    path: str
    issue_id: str
    visible_node_count: int
    open_blocker_count: int
    next_steps: tuple[str, ...]
    private_content_excluded: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "plan_id": self.plan_id,
            "path": self.path,
            "issue_id": self.issue_id,
            "visible_node_count": self.visible_node_count,
            "open_blocker_count": self.open_blocker_count,
            "next_steps": list(self.next_steps),
            "private_content_excluded": self.private_content_excluded,
        }


def build_context_pack(
    context: RepoContext,
    issue_id: str,
    *,
    role: RetrievalRole | str = RetrievalRole.ORCHESTRATOR,
    max_cards: int = CONTEXT_MAX_CARDS,
    max_full_artifacts: int | None = None,
    public_only: bool = False,
) -> ContextPackResult:
    """Build a deterministic bounded context pack for an issue."""
    records = tuple(load_artifacts(context))
    issue = _find_issue(records, issue_id)
    role_value = RetrievalRole(role)
    full_artifact_budget = _context_full_artifact_budget(
        role=role_value,
        max_full_artifacts=max_full_artifacts,
    )
    retrieval = _retrieve_context_cards(
        context,
        issue,
        role=role_value,
        max_cards=max_cards,
        max_full_artifacts=full_artifact_budget,
        public_only=public_only,
    )
    reasons_by_artifact = _issue_relevance_reasons(records, issue)
    cards = tuple(
        ContextPackCard(
            retrieved=retrieved,
            reasons=reasons_by_artifact.get(retrieved.card.id, ()),
        )
        for retrieved in retrieval.cards
        if retrieved.card.id in reasons_by_artifact
    )
    if len(cards) != len(retrieval.cards):
        filtered_count = len(retrieval.cards) - len(cards)
        retrieval = retrieval.model_copy(
            update={
                "cards": [card.retrieved for card in cards],
                "audit": retrieval.audit.model_copy(
                    update={
                        "warnings": [
                            *retrieval.audit.warnings,
                            "context-pack issue-local relevance filter removed "
                            f"{filtered_count} retrieval card(s)",
                        ],
                    }
                ),
            }
        )
    artifact_records = _artifact_records_by_id(records)
    failure_entries = _context_failure_entries(cards, artifact_records)
    checked_counterexample_evidence = _context_checked_counterexample_evidence(
        context,
        records=records,
        issue=issue,
        cards=cards,
        public_only=public_only,
    )
    strategy_plans = _context_strategy_plan_entries(
        context,
        issue=issue,
        public_only=public_only,
    )
    full_artifact_pulls = _pull_full_artifacts(
        context,
        _ordered_context_cards(cards),
        max_full_artifacts=full_artifact_budget,
        role=role_value,
        public_only=public_only,
    )
    retrieval = retrieval.model_copy(
        update={"full_artifact_pulls": list(full_artifact_pulls)}
    )
    task_dir = context.resolve(Path("context") / "TASKS" / issue_id)
    task_dir.mkdir(parents=True, exist_ok=True)

    contents = {
        "CONTEXT.md": _render_context(
            context,
            issue,
            cards,
            artifact_records,
            failure_entries,
            checked_counterexample_evidence,
            strategy_plans,
        ),
        "ACCEPTANCE.md": _render_acceptance(issue),
        "RELEVANT_ARTIFACTS.md": _render_relevant_artifacts(cards, artifact_records),
        "KNOWN_FAILURES.md": _render_known_failures(
            cards,
            artifact_records,
            failure_entries,
            checked_counterexample_evidence,
        ),
        "FULL_ARTIFACTS.md": _render_full_artifacts(context, full_artifact_pulls),
        "RETRIEVAL_AUDIT.json": _render_retrieval_audit(
            issue=issue,
            retrieval=retrieval,
            failure_entries=failure_entries,
            checked_counterexample_evidence=checked_counterexample_evidence,
            strategy_plans=strategy_plans,
            query=_context_query(issue, include_related_artifacts=not public_only),
            role=role_value,
            max_cards=max_cards,
            max_full_artifacts=full_artifact_budget,
            public_only=public_only,
        ),
        "COMMANDS.md": _render_commands(),
    }
    files: list[Path] = []
    for filename in PACK_FILENAMES:
        path = task_dir / filename
        path.write_text(contents[filename], encoding="utf-8")
        files.append(path)

    return ContextPackResult(
        issue_id=issue.id,
        task_dir=task_dir,
        files=tuple(files),
    )


def show_context_pack(
    context: RepoContext,
    issue_id: str,
    *,
    role: RetrievalRole | str = RetrievalRole.ORCHESTRATOR,
    max_cards: int = CONTEXT_MAX_CARDS,
    max_full_artifacts: int | None = None,
    public_only: bool = False,
) -> str:
    """Build a context pack if needed and return its main context document."""
    result = build_context_pack(
        context,
        issue_id,
        role=role,
        max_cards=max_cards,
        max_full_artifacts=max_full_artifacts,
        public_only=public_only,
    )
    return (result.task_dir / "CONTEXT.md").read_text(encoding="utf-8")


def _find_issue(records: tuple[LoadedRecord, ...], issue_id: str) -> IssueRecord:
    issues = [
        record.record
        for record in records
        if isinstance(record.record, IssueRecord) and record.record.id == issue_id
    ]
    if not issues:
        raise ContextPackError(f"issue not found: {issue_id}")
    return sorted(issues, key=lambda issue: issue.id)[0]


def _retrieve_context_cards(
    context: RepoContext,
    issue: IssueRecord,
    *,
    role: RetrievalRole,
    max_cards: int,
    max_full_artifacts: int,
    public_only: bool,
) -> RetrievalResult:
    if max_cards <= 0:
        raise ContextPackError("max_cards must be positive")
    if max_full_artifacts < 0:
        raise ContextPackError("max_full_artifacts must be non-negative")
    try:
        return search_artifact_cards(
            context,
            query=_context_query(issue, include_related_artifacts=not public_only),
            issue_id=issue.id,
            max_cards=max_cards,
            allowed_scopes=PUBLIC_CONTEXT_SCOPES if public_only else ALL_CONTEXT_SCOPES,
            pinned_artifacts=()
            if public_only
            else tuple(issue.related_artifacts),
            include_refuted=True,
            include_obsolete=True,
            role=role,
            max_full_artifacts=max_full_artifacts,
        )
    except ValueError as exc:
        raise ContextPackError(str(exc)) from exc


def _context_query(
    issue: IssueRecord,
    *,
    include_related_artifacts: bool = True,
) -> str:
    related = tuple(issue.related_artifacts) if include_related_artifacts else ()
    parts = [issue.title, issue.description, *issue.tags, *related]
    query = " ".join(part for part in parts if part.strip()).strip()
    return query or issue.id


def _context_full_artifact_budget(
    *,
    role: RetrievalRole,
    max_full_artifacts: int | None,
) -> int:
    if max_full_artifacts is not None:
        return max_full_artifacts
    if role is RetrievalRole.ORCHESTRATOR:
        return 0
    return 0


def _select_relevant_artifacts(
    records: tuple[LoadedRecord, ...],
    issue: IssueRecord,
) -> tuple[RankedArtifact, ...]:
    artifact_records = [
        record for record in records if isinstance(record.record, BaseArtifact)
    ]
    artifact_by_id = {record.id: record for record in artifact_records}
    direct_ids = set(issue.related_artifacts)
    dependency_ids: set[str] = set()

    for artifact_id in sorted(issue.related_artifacts):
        record = artifact_by_id.get(artifact_id)
        if isinstance(record, LoadedRecord) and isinstance(record.record, BaseArtifact):
            dependency_ids.update(record.record.depends_on)

    issue_terms = _issue_terms(issue)
    issue_tags = {_normalize_term(tag) for tag in issue.tags}
    selected: list[RankedArtifact] = []

    for record in artifact_records:
        artifact = _as_artifact(record)
        reasons = _relevance_reasons(
            artifact,
            direct_ids=direct_ids,
            dependency_ids=dependency_ids,
            issue_terms=issue_terms,
            issue_tags=issue_tags,
        )
        if reasons:
            selected.append(RankedArtifact(record=record, reasons=reasons))

    return tuple(sorted(selected, key=_artifact_sort_key))


def _issue_relevance_reasons(
    records: tuple[LoadedRecord, ...],
    issue: IssueRecord,
) -> dict[str, tuple[str, ...]]:
    artifacts = _select_relevant_artifacts(records, issue)
    return {artifact.record.id: artifact.reasons for artifact in artifacts}


def _relevance_reasons(
    artifact: BaseArtifact,
    *,
    direct_ids: set[str],
    dependency_ids: set[str],
    issue_terms: set[str],
    issue_tags: set[str],
) -> tuple[str, ...]:
    reasons: list[str] = []
    if artifact.id in direct_ids:
        reasons.append("direct reference")
    if artifact.id in dependency_ids:
        reasons.append("dependency neighbor")
    if _artifact_domain_matches_issue(artifact, issue_terms):
        reasons.append("domain match")
    if issue_tags.intersection({_normalize_term(tag) for tag in artifact.tags}):
        reasons.append("tag match")
    return tuple(reasons)


def _artifact_domain_matches_issue(
    artifact: BaseArtifact,
    issue_terms: set[str],
) -> bool:
    for domain in artifact.domain:
        terms = _term_variants(domain)
        if terms.intersection(issue_terms):
            return True
        parts = _term_parts(domain)
        if parts and parts.issubset(issue_terms):
            return True
    return False


def _artifact_sort_key(record: RankedArtifact) -> tuple[int, int, str, str]:
    artifact = _as_artifact(record.record)
    return (
        _reason_priority(record.reasons),
        _status_priority(artifact.status),
        artifact.id,
        record.record.source_path.as_posix(),
    )


def _reason_priority(reasons: tuple[str, ...]) -> int:
    priorities = {
        "direct reference": 0,
        "dependency neighbor": 1,
        "domain match": 2,
        "tag match": 3,
    }
    return min(priorities[reason] for reason in reasons)


def _issue_terms(issue: IssueRecord) -> set[str]:
    terms: set[str] = set()
    for value in (issue.title, issue.description, *issue.tags):
        terms.update(_term_variants(value))
    return terms


def _term_variants(value: str) -> set[str]:
    normalized = _normalize_term(value)
    variants = {normalized}
    variants.update(
        match.group(0)
        for match in re.finditer(r"[a-z0-9]+(?:-[a-z0-9]+)*", normalized)
    )
    return {variant for variant in variants if variant}


def _term_parts(value: str) -> set[str]:
    normalized = _normalize_term(value)
    return set(re.findall(r"[a-z0-9]+", normalized))


def _normalize_term(value: str) -> str:
    return value.strip().lower().replace("_", "-")


def _status_priority(status: ArtifactStatus) -> int:
    if is_accepted_status(status):
        return 0
    if is_preaccepted_status(status):
        return 1
    if status in KNOWN_FAILURE_STATUSES:
        return 2
    return 3


def _artifact_records_by_id(
    records: tuple[LoadedRecord, ...],
) -> dict[str, LoadedRecord]:
    artifact_records = [
        record for record in records if isinstance(record.record, BaseArtifact)
    ]
    return {record.id: record for record in artifact_records}


def _render_context(
    context: RepoContext,
    issue: IssueRecord,
    cards: tuple[ContextPackCard, ...],
    artifact_records: dict[str, LoadedRecord],
    failure_entries: tuple[ContextFailureEntry, ...],
    checked_counterexample_evidence: tuple[
        ContextCheckedCounterexampleEvidenceEntry, ...
    ],
    strategy_plans: tuple[ContextStrategyPlanEntry, ...],
) -> str:
    ordered_cards = _ordered_context_cards(cards)
    accepted = [
        record
        for record in ordered_cards
        if record.retrieved.card.status is ArtifactCardStatus.ACCEPTED
    ]
    drafts = [
        record
        for record in ordered_cards
        if record.retrieved.card.status
        in {
            ArtifactCardStatus.RAW,
            ArtifactCardStatus.DRAFT,
            ArtifactCardStatus.LOCALLY_TESTED,
            ArtifactCardStatus.ADVERSARIALLY_TESTED,
            ArtifactCardStatus.MACHINE_CHECKED,
            ArtifactCardStatus.HUMAN_REVIEWED,
        }
    ]
    known_failures = [
        record
        for record in ordered_cards
        if record.retrieved.card.status in KNOWN_FAILURE_CARD_STATUSES
    ]

    lines = [
        f"# Context Pack: {issue.id}",
        "",
        "## Issue Summary",
        "",
        f"- Title: {issue.title}",
        f"- Status: {issue.status}",
        f"- Severity: {issue.severity}",
        f"- Source: {_issue_source_path(context, issue.id)}",
        f"- Summary: {_one_line(issue.description)}",
        "",
        "## Relevant Artifact Cards",
        "",
        (
            "Default context uses compact ArtifactCard entries. Full artifact "
            "YAML is not included unless explicitly pulled."
        ),
        "",
        "## Relevant Accepted Artifacts",
        "",
    ]
    lines.extend(_artifact_lines(accepted, artifact_records))
    lines.extend(["", "## Relevant Draft Artifacts", ""])
    lines.extend(_artifact_lines(drafts, artifact_records))
    lines.extend(["", "## Relevant Known Failures", ""])
    lines.extend(_artifact_lines(known_failures, artifact_records))
    if failure_entries:
        lines.extend(["", "## Known Failed Directions", ""])
        lines.extend(_failure_entry_lines(failure_entries))
    if checked_counterexample_evidence:
        lines.extend(["", "## Checked Counterexample Evidence", ""])
        lines.extend(_checked_counterexample_evidence_lines(checked_counterexample_evidence))
    if strategy_plans:
        lines.extend(["", "## Strategy Plan Summary", ""])
        lines.extend(_strategy_plan_lines(strategy_plans))
    lines.extend(
        [
            "",
            "## Retrieval Audit",
            "",
            (
                "See `RETRIEVAL_AUDIT.json` for filters, score metadata, "
                "exclusions, warnings, and any full artifact pulls."
            ),
            "",
            "## Current Project State",
            "",
            _read_markdown_excerpt(context.resolve("context/PROJECT_STATE.md")),
            "",
            "## Relevant Interfaces",
            "",
            _read_markdown_excerpt(
                context.resolve("context/INTERFACE_REGISTRY.md"),
                keywords=("context", "RepoContext", "load_artifacts", "IssueRecord"),
            ),
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def _render_acceptance(issue: IssueRecord) -> str:
    task_criteria = _extract_acceptance_criteria(issue.description)
    lines = [
        f"# Acceptance: {issue.id}",
        "",
        "## Task-Specific Acceptance Criteria",
        "",
    ]
    if task_criteria:
        lines.extend(f"- {criterion}" for criterion in task_criteria)
    else:
        lines.append(
            "- No task-specific acceptance criteria recorded in the issue file."
        )

    lines.extend(
        [
            "",
            "## Default Engineering Acceptance Criteria",
            "",
            "- New behavior includes tests.",
            "- Public interface changes update `context/INTERFACE_REGISTRY.md`.",
            "- Workflow or behavior changes update relevant docs.",
            "- Verification failures are reported, not hidden.",
            "- Generated outputs are deterministic.",
        ]
    )
    return "\n".join(lines) + "\n"


def _render_relevant_artifacts(
    cards: tuple[ContextPackCard, ...],
    artifact_records: dict[str, LoadedRecord],
) -> str:
    lines = ["# Relevant Artifact Cards", ""]
    ordered_cards = _ordered_context_cards(cards)
    if ordered_cards:
        lines.extend(
            _artifact_reference_line(record, artifact_records)
            for record in ordered_cards
        )
    else:
        lines.append("- None")
    return "\n".join(lines) + "\n"


def _render_known_failures(
    cards: tuple[ContextPackCard, ...],
    artifact_records: dict[str, LoadedRecord],
    failure_entries: tuple[ContextFailureEntry, ...],
    checked_counterexample_evidence: tuple[
        ContextCheckedCounterexampleEvidenceEntry, ...
    ],
) -> str:
    known_failures = [
        record
        for record in _ordered_context_cards(cards)
        if record.retrieved.card.status in KNOWN_FAILURE_CARD_STATUSES
    ]
    lines = ["# Known Failures", ""]
    if known_failures:
        lines.extend(
            _artifact_reference_line(record, artifact_records)
            for record in known_failures
        )
    if failure_entries:
        lines.extend(["", "## Known Failed Directions", ""])
        lines.extend(_failure_entry_lines(failure_entries))
    if checked_counterexample_evidence:
        lines.extend(["", "## Checked Counterexample Evidence", ""])
        lines.extend(
            _checked_counterexample_evidence_lines(checked_counterexample_evidence)
        )
    if (
        not known_failures
        and not failure_entries
        and not checked_counterexample_evidence
    ):
        lines.append("- None")
    return "\n".join(lines) + "\n"


def _render_full_artifacts(
    context: RepoContext,
    pulls: tuple[FullArtifactPull, ...],
) -> str:
    lines = ["# Full Artifacts", ""]
    if not pulls:
        lines.append(
            "No full artifacts pulled. The default context pack is cards-only."
        )
        return "\n".join(lines) + "\n"

    for pull in pulls:
        path = context.resolve(pull.path)
        lines.extend(
            [
                f"## {pull.artifact_id}",
                "",
                f"- Source: {pull.path}",
                f"- Reason: {pull.reason}",
                "",
                "```yaml",
                path.read_text(encoding="utf-8").rstrip(),
                "```",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def _render_retrieval_audit(
    *,
    issue: IssueRecord,
    retrieval: RetrievalResult,
    failure_entries: tuple[ContextFailureEntry, ...],
    checked_counterexample_evidence: tuple[
        ContextCheckedCounterexampleEvidenceEntry, ...
    ],
    strategy_plans: tuple[ContextStrategyPlanEntry, ...],
    query: str,
    role: RetrievalRole,
    max_cards: int,
    max_full_artifacts: int,
    public_only: bool,
) -> str:
    context_payload: dict[str, object] = {
        "card_count": len(retrieval.cards),
        "full_artifact_count": len(retrieval.full_artifact_pulls),
        "failure_entry_count": len(failure_entries),
        "checked_counterexample_evidence_count": len(
            checked_counterexample_evidence
        ),
        "content_mode": _context_payload_mode(retrieval),
    }
    if strategy_plans:
        context_payload["strategy_plan_count"] = len(strategy_plans)

    payload = {
        "schema_version": 1,
        "issue_id": issue.id,
        "request": {
            "query": query,
            "issue_id": issue.id,
            "role": role.value,
            "max_cards": max_cards,
            "max_full_artifacts": max_full_artifacts,
            "public_only": public_only,
        },
        "retrieval": {
            "request_id": retrieval.request_id,
            "generated_at": retrieval.generated_at.isoformat(),
            "index_fingerprint": retrieval.index_fingerprint,
            "cards": [
                {
                    "artifact_id": hit.card.id,
                    "path": hit.card.path,
                    "root_scope": hit.card.root_scope.value,
                    "status": hit.card.status.value,
                    "score_breakdown": hit.score_breakdown.to_dict(),
                    "failure_count": hit.card.failure_count,
                    "recent_failure_directions": hit.card.recent_failure_directions,
                    "why_relevant": hit.why_relevant,
                }
                for hit in retrieval.cards
            ],
        },
        "full_artifact_pulls": [
            pull.to_dict() for pull in retrieval.full_artifact_pulls
        ],
        "context_payload": context_payload,
        "failure_memory": [entry.to_dict() for entry in failure_entries],
        "checked_counterexample_evidence": [
            entry.to_dict() for entry in checked_counterexample_evidence
        ],
        "strategy_plans": [entry.to_dict() for entry in strategy_plans],
        "audit": retrieval.audit.to_dict(),
    }
    return json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n"


def _render_commands() -> str:
    lines = ["# Commands", "", "Run these commands before handoff:", ""]
    lines.extend(f"- `{command}`" for command in DEFAULT_COMMANDS)
    return "\n".join(lines) + "\n"


def _artifact_lines(
    records: list[ContextPackCard],
    artifact_records: dict[str, LoadedRecord],
) -> list[str]:
    if not records:
        return ["- None"]
    lines = []
    for record in records:
        lines.extend(_artifact_reference_lines(record, artifact_records))
    return lines


def _artifact_reference_line(
    record: ContextPackCard,
    artifact_records: dict[str, LoadedRecord],
) -> str:
    return "\n".join(_artifact_reference_lines(record, artifact_records))


def _artifact_reference_lines(
    record: ContextPackCard,
    artifact_records: dict[str, LoadedRecord],
) -> list[str]:
    card = record.retrieved.card
    prefix = _card_status_prefix(card.status)
    reasons = _combined_reasons(record)
    lines = [
        f"- {prefix}{card.id} | {card.title} | "
        f"{card.status.value} | {card.path} | "
        f"score: {record.retrieved.score_breakdown.total:.6f} | "
        f"root_scope={card.root_scope.value} | "
        f"{_failure_card_summary(card)}"
        f"reasons: {_format_reasons(reasons)}"
    ]
    loaded = artifact_records.get(card.id)
    if loaded is not None and isinstance(loaded.record, BaseArtifact):
        lines.extend(_formal_metadata_lines(loaded.record))
    return lines


def _context_failure_entries(
    cards: tuple[ContextPackCard, ...],
    artifact_records: dict[str, LoadedRecord],
) -> tuple[ContextFailureEntry, ...]:
    entries: list[ContextFailureEntry] = []
    for card_record in _ordered_context_cards(cards):
        card = card_record.retrieved.card
        loaded = artifact_records.get(card.id)
        if loaded is None or not isinstance(loaded.record, BaseArtifact):
            continue
        artifact = loaded.record
        failure_log = sorted(
            artifact.failure_log,
            key=lambda entry: (entry.attempted_at, entry.failure_id),
            reverse=True,
        )
        for entry in failure_log:
            root_scope = card.root_scope.value
            origin = entry.origin
            entries.append(
                ContextFailureEntry(
                    artifact_id=artifact.id,
                    artifact_path=card.path,
                    root_scope=root_scope,
                    failure_id=entry.failure_id,
                    direction=entry.direction,
                    failed_because=entry.failed_because,
                    status=entry.status,
                    next_possible_directions=tuple(entry.next_possible_directions),
                    origin=origin,
                    attempt_kind=entry.attempt_kind,
                    source_label=f"{root_scope}:{origin}",
                )
            )
    return tuple(entries)


def _context_checked_counterexample_evidence(
    context: RepoContext,
    *,
    records: tuple[LoadedRecord, ...],
    issue: IssueRecord,
    cards: tuple[ContextPackCard, ...],
    public_only: bool,
) -> tuple[ContextCheckedCounterexampleEvidenceEntry, ...]:
    visible_target_ids = {record.retrieved.card.id for record in cards}
    visible_target_ids.update(
        _directly_visible_issue_targets(
            records,
            issue=issue,
            public_only=public_only,
        )
    )
    entries = [
        _checked_counterexample_evidence_entry(result)
        for result in load_checked_counterexample_evidence(context)
        if result.record.target_artifact_id in visible_target_ids
    ]
    return tuple(
        sorted(
            entries,
            key=lambda entry: (
                entry.target_artifact_id,
                entry.evidence_id,
                entry.source_path,
            ),
        )
    )


def _directly_visible_issue_targets(
    records: tuple[LoadedRecord, ...],
    *,
    issue: IssueRecord,
    public_only: bool,
) -> set[str]:
    if not issue.related_artifacts:
        return set()
    records_by_id = {
        record.id: record
        for record in records
        if isinstance(record.record, BaseArtifact)
    }
    targets: set[str] = set()
    for artifact_id in issue.related_artifacts:
        loaded = records_by_id.get(artifact_id)
        if loaded is None:
            continue
        scope = _root_scope_for_loaded_record(loaded)
        if public_only and scope not in PUBLIC_CONTEXT_SCOPES:
            continue
        targets.add(artifact_id)
    return targets


def _checked_counterexample_evidence_entry(
    result: CheckedCounterexampleEvidenceLoadResult,
) -> ContextCheckedCounterexampleEvidenceEntry:
    record = result.record
    return ContextCheckedCounterexampleEvidenceEntry(
        evidence_id=record.evidence_id,
        target_artifact_id=record.target_artifact_id,
        candidate_id=record.candidate_id,
        candidate_source=record.candidate_source.value,
        check_method=record.check_method.value,
        checked_result=record.checked_result.value,
        checker=record.checker,
        source_path=result.relative_path.as_posix(),
        verifier_evidence_ids=record.verifier_evidence_ids,
        review_record_paths=record.review_record_paths,
        evidence_paths=record.evidence_paths,
        limitations=record.limitations,
    )


def _checked_counterexample_evidence_lines(
    entries: tuple[ContextCheckedCounterexampleEvidenceEntry, ...],
) -> list[str]:
    lines = [
        (
            "Checked counterexample evidence is evidence for review only; it "
            "does not create human review, accepted refutation, accepted "
            "status, or promotion authority."
        )
    ]
    for entry in entries:
        support = _checked_counterexample_support_text(entry)
        limitations = "; ".join(entry.limitations) if entry.limitations else "-"
        lines.extend(
            [
                (
                    f"- {entry.evidence_id} | target: {entry.target_artifact_id} "
                    f"| result: {entry.checked_result} | path: {entry.source_path}"
                ),
                f"  - candidate: {entry.candidate_id} ({entry.candidate_source})",
                f"  - method: {entry.check_method}; checker: {entry.checker}",
                f"  - support: {support}",
                f"  - limitations: {limitations}",
            ]
        )
    return lines


def _checked_counterexample_support_text(
    entry: ContextCheckedCounterexampleEvidenceEntry,
) -> str:
    parts = []
    if entry.verifier_evidence_ids:
        parts.append(
            "verifier_evidence="
            + ",".join(sorted(entry.verifier_evidence_ids))
        )
    if entry.review_record_paths:
        parts.append("review_records=" + ",".join(sorted(entry.review_record_paths)))
    if entry.evidence_paths:
        parts.append("evidence_paths=" + ",".join(sorted(entry.evidence_paths)))
    return "; ".join(parts) if parts else "-"


def _context_strategy_plan_entries(
    context: RepoContext,
    *,
    issue: IssueRecord,
    public_only: bool,
) -> tuple[ContextStrategyPlanEntry, ...]:
    entries = [
        _strategy_plan_entry(loaded, public_only=public_only)
        for loaded in load_strategy_plans(context)
        if loaded.plan.issue_id == issue.id
    ]
    return tuple(entry for entry in entries if entry is not None)


def _strategy_plan_entry(
    loaded: StrategyPlanStorageResult,
    *,
    public_only: bool,
) -> ContextStrategyPlanEntry | None:
    plan = loaded.plan
    visible_nodes = _visible_strategy_nodes(plan, public_only=public_only)
    private_excluded = len(visible_nodes) != len(plan.graph.nodes)
    if not visible_nodes:
        return None
    visible_ids = {node.node_id for node in visible_nodes}
    next_steps = tuple(
        f"{step.rank}. {step.node_id}"
        for step in plan.next_steps
        if step.node_id in visible_ids
    )[:5]
    blockers = [
        node
        for node in visible_nodes
        if node.status in {StrategyTaskStatus.BLOCKED, StrategyTaskStatus.FAILED}
    ]
    return ContextStrategyPlanEntry(
        plan_id=plan.plan_id,
        path=loaded.relative_path.as_posix(),
        issue_id=plan.issue_id,
        visible_node_count=len(visible_nodes),
        open_blocker_count=len(blockers),
        next_steps=next_steps,
        private_content_excluded=private_excluded,
    )


def _visible_strategy_nodes(
    plan: StrategyPlan,
    *,
    public_only: bool,
) -> tuple[Any, ...]:
    if not public_only:
        return plan.graph.nodes
    return tuple(
        node for node in plan.graph.nodes if node.scope is not StrategyTaskScope.PRIVATE
    )


def _strategy_plan_lines(
    entries: tuple[ContextStrategyPlanEntry, ...],
) -> list[str]:
    lines = [
        (
            "Strategy plans are guidance for review only; they are not proof, "
            "checked evidence, gate pass, human review, accepted status, or "
            "promotion authority."
        )
    ]
    for entry in entries:
        next_steps = "; ".join(entry.next_steps) if entry.next_steps else "-"
        lines.extend(
            [
                (
                    f"- {entry.plan_id} | path: {entry.path} | "
                    f"visible_nodes: {entry.visible_node_count} | "
                    f"open_blockers: {entry.open_blocker_count}"
                ),
                f"  - next: {next_steps}",
            ]
        )
        if entry.private_content_excluded:
            lines.append("  - private strategy content excluded by public-only scope")
    return lines


def _failure_entry_lines(
    failure_entries: tuple[ContextFailureEntry, ...],
) -> list[str]:
    lines = [
        (
            "Failure memory is failed/unresolved attempt context only; "
            "not proof, refutation, verifier pass, or human review."
        )
    ]
    for entry in failure_entries:
        next_text = (
            "; ".join(entry.next_possible_directions)
            if entry.next_possible_directions
            else "-"
        )
        lines.extend(
            [
                (
                    f"- {entry.artifact_id} | source: {entry.source_label} | "
                    f"kind: {entry.attempt_kind} | path: {entry.artifact_path}"
                ),
                f"  - direction: {entry.direction}",
                f"  - failed_because: {entry.failed_because}",
                f"  - status: {entry.status}",
                f"  - next: {next_text}",
                f"  - origin: {entry.origin}",
            ]
        )
    return lines


def _root_scope_for_loaded_record(loaded: LoadedRecord) -> MemoryRootScope:
    if loaded.is_private_scope:
        return MemoryRootScope.PRIVATE
    kb_root_name = (loaded.kb_root_name or "").lower()
    if kb_root_name == "framework":
        return MemoryRootScope.FRAMEWORK
    if loaded.is_public_kb_root or kb_root_name == "public":
        return MemoryRootScope.PUBLIC
    return MemoryRootScope.WORKSPACE


def _combined_reasons(record: ContextPackCard) -> tuple[str, ...]:
    reasons = list(record.reasons)
    reasons.extend(record.retrieved.why_relevant)
    return tuple(dict.fromkeys(reason for reason in reasons if reason))


def _failure_card_summary(card: ArtifactCard) -> str:
    failure_count = card.failure_count
    if not failure_count:
        return ""
    directions = card.recent_failure_directions
    if directions:
        return (
            f"failures: {failure_count}; recent failed directions: "
            f"{'; '.join(directions)} | "
        )
    return f"failures: {failure_count} | "


def _ordered_context_cards(
    cards: tuple[ContextPackCard, ...],
) -> tuple[ContextPackCard, ...]:
    return tuple(sorted(cards, key=_context_card_sort_key))


def _context_card_sort_key(record: ContextPackCard) -> tuple[int, int, float, str, str]:
    card = record.retrieved.card
    return (
        _reason_priority(record.reasons or ("tag match",)),
        _card_status_priority(card.status),
        -record.retrieved.score_breakdown.total,
        card.id,
        card.path,
    )


def _card_status_priority(status: ArtifactCardStatus) -> int:
    if status is ArtifactCardStatus.ACCEPTED:
        return 0
    if status in {
        ArtifactCardStatus.RAW,
        ArtifactCardStatus.DRAFT,
        ArtifactCardStatus.LOCALLY_TESTED,
        ArtifactCardStatus.ADVERSARIALLY_TESTED,
        ArtifactCardStatus.MACHINE_CHECKED,
        ArtifactCardStatus.HUMAN_REVIEWED,
    }:
        return 1
    if status in KNOWN_FAILURE_CARD_STATUSES:
        return 2
    return 3


def _formal_metadata_lines(artifact: BaseArtifact) -> list[str]:
    if not _has_formal_metadata(artifact):
        return []

    lines: list[str] = []
    if artifact.formalizations:
        lines.append("  - Formal links:")
        for ref in sorted(artifact.formalizations, key=lambda item: item.id):
            lines.append(
                f"    - {ref.library}@{ref.library_ref}:"
                f"{ref.import_path}#{ref.symbol} "
                f"[{ref.declaration_kind}, {ref.status}, {ref.check_mode}]"
            )
    else:
        lines.append("  - Formal links: none")

    reviewer = artifact.alignment.reviewer or "-"
    policy = artifact.verification_policy
    lines.extend(
        [
            f"  - Alignment: {artifact.alignment.status}; reviewer={reviewer}",
            "  - Verification policy: "
            f"{policy.level}; formal_link={_bool_text(policy.require_formal_link)}; "
            f"lean_check={_bool_text(policy.require_lean_check)}; "
            f"alignment_review={_bool_text(policy.require_alignment_review)}",
            f"  - G10-relevant: yes; {'; '.join(_formal_hints(artifact))}",
        ]
    )
    return lines


def _has_formal_metadata(artifact: BaseArtifact) -> bool:
    policy = artifact.verification_policy
    return any(
        (
            artifact.formalizations,
            artifact.alignment.status != "none",
            policy.level != "source_reviewed",
            policy.require_formal_link,
            policy.require_lean_check,
            policy.require_alignment_review,
        )
    )


def _formal_hints(artifact: BaseArtifact) -> list[str]:
    hints: list[str] = []
    policy = artifact.verification_policy
    if policy.require_formal_link:
        hints.append("requires formal link")
    if policy.require_lean_check:
        hints.append("requires Lean check")
    if policy.require_alignment_review:
        hints.append("requires alignment review")
    if artifact.alignment.status == "rejected":
        hints.append("alignment rejected")
    if any(ref.status == "planned" for ref in artifact.formalizations):
        hints.append("planned formalization")
    if any(ref.status == "broken" for ref in artifact.formalizations):
        hints.append("broken formalization")
    if any(ref.status == "deprecated" for ref in artifact.formalizations):
        hints.append("deprecated formalization")
    return hints or ["metadata present"]


def _bool_text(value: bool) -> str:
    return str(value).lower()


def _status_prefix(status: ArtifactStatus) -> str:
    if is_preaccepted_status(status):
        return "[DRAFT] "
    if status in KNOWN_FAILURE_STATUSES:
        return f"[{status.value.upper()}] "
    return ""


def _card_status_prefix(status: ArtifactCardStatus) -> str:
    if status in {
        ArtifactCardStatus.RAW,
        ArtifactCardStatus.DRAFT,
        ArtifactCardStatus.LOCALLY_TESTED,
        ArtifactCardStatus.ADVERSARIALLY_TESTED,
        ArtifactCardStatus.MACHINE_CHECKED,
        ArtifactCardStatus.HUMAN_REVIEWED,
    }:
        return "[DRAFT] "
    if status in KNOWN_FAILURE_CARD_STATUSES:
        return f"[{status.value.upper()}] "
    return ""


def _format_reasons(reasons: tuple[str, ...]) -> str:
    return ", ".join(reasons)


def _as_artifact(record: LoadedRecord) -> BaseArtifact:
    if not isinstance(record.record, BaseArtifact):
        raise TypeError(f"loaded record is not an artifact: {record.id}")
    return record.record


def _extract_acceptance_criteria(description: str) -> list[str]:
    criteria: list[str] = []
    for raw_line in description.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("- [ ] "):
            criteria.append(line.removeprefix("- [ ] ").strip())
        elif line.startswith("- "):
            criteria.append(line.removeprefix("- ").strip())
        elif line.startswith("* "):
            criteria.append(line.removeprefix("* ").strip())
        elif _starts_numbered_item(line):
            criteria.append(line.split(".", 1)[1].strip())
    return criteria


def _starts_numbered_item(line: str) -> bool:
    prefix, separator, rest = line.partition(".")
    return bool(separator and rest.strip() and prefix.isdecimal())


def _read_markdown_excerpt(
    path: Path,
    *,
    keywords: tuple[str, ...] = (),
    max_lines: int = 18,
    max_chars: int = 1600,
) -> str:
    if not path.exists():
        return f"{path.name} is not available."

    lines = path.read_text(encoding="utf-8").splitlines()
    selected = _select_excerpt_lines(lines, keywords=keywords, max_lines=max_lines)
    excerpt = "\n".join(selected).strip()
    if len(excerpt) > max_chars:
        return excerpt[: max_chars - 4].rstrip() + "\n..."
    return excerpt or "No excerpt available."


def _select_excerpt_lines(
    lines: list[str],
    *,
    keywords: tuple[str, ...],
    max_lines: int,
) -> list[str]:
    if not keywords:
        return lines[:max_lines]

    lowered_keywords = tuple(keyword.lower() for keyword in keywords)
    selected = [
        line
        for line in lines
        if any(keyword in line.lower() for keyword in lowered_keywords)
    ]
    if selected:
        return selected[:max_lines]
    return lines[:max_lines]


def _one_line(value: str) -> str:
    return shorten(" ".join(value.split()), width=240, placeholder="...")


def _issue_source_path(context: RepoContext, issue_id: str) -> str:
    for record in load_artifacts(context):
        if isinstance(record.record, IssueRecord) and record.record.id == issue_id:
            return record.source_path.as_posix()
    return "-"


def _pull_full_artifacts(
    context: RepoContext,
    cards: tuple[ContextPackCard, ...],
    *,
    max_full_artifacts: int,
    role: RetrievalRole,
    public_only: bool,
) -> tuple[FullArtifactPull, ...]:
    if max_full_artifacts <= 0:
        return ()
    pulls: list[FullArtifactPull] = []
    for record in cards:
        if len(pulls) >= max_full_artifacts:
            break
        card = record.retrieved.card
        artifact_path = context.resolve(card.path)
        try:
            relative_path = repo_relative_posix(context.repo_root, artifact_path)
        except ValueError:
            continue
        if not artifact_path.is_file():
            continue
        pulls.append(
            FullArtifactPull(
                artifact_id=card.id,
                path=relative_path,
                reason=_full_artifact_pull_reason(
                    role=role,
                    public_only=public_only,
                    max_full_artifacts=max_full_artifacts,
                ),
            )
        )
    return tuple(pulls)


def _context_payload_mode(
    retrieval: RetrievalResult,
) -> str:
    if retrieval.full_artifact_pulls:
        return "cards_with_full_artifacts"
    return "cards_only"


def _full_artifact_pull_reason(
    *,
    role: RetrievalRole,
    public_only: bool,
    max_full_artifacts: int,
) -> str:
    policy_scope = "public_only" if public_only else "workspace"
    return (
        "explicit context-pack full-artifact budget; "
        f"role={role.value}; policy_scope={policy_scope}; "
        f"max_full_artifacts={max_full_artifacts}"
    )
