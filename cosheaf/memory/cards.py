"""Deterministic artifact-card construction from loaded repository records."""

from __future__ import annotations

from collections.abc import Iterable

from cosheaf.core.artifact import BaseArtifact, SourceMetadata
from cosheaf.core.ids import validate_artifact_id
from cosheaf.memory.models import (
    ArtifactCard,
    ArtifactCardStatus,
    ArtifactCardType,
    MemoryRootScope,
)
from cosheaf.storage.loader import IssueRecord, LoadedRecord, LoadError, load_artifacts
from cosheaf.storage.repo import RepoContext

DEFAULT_CARD_SCOPES = frozenset(
    {
        MemoryRootScope.PUBLIC,
        MemoryRootScope.WORKSPACE,
        MemoryRootScope.FRAMEWORK,
    }
)


class MemoryCardError(ValueError):
    """Raised for expected memory-card construction failures."""


def build_artifact_cards(
    context: RepoContext,
    *,
    issue_id: str | None = None,
    status: ArtifactCardStatus | str | None = None,
    allowed_scopes: Iterable[MemoryRootScope | str] = DEFAULT_CARD_SCOPES,
) -> tuple[ArtifactCard, ...]:
    """Build deterministic artifact cards from existing repository records.

    This reads current YAML metadata only. It does not rank, search, write
    sidecars, pull full artifact text, or modify repository records.
    """
    try:
        records = tuple(load_artifacts(context))
    except LoadError as exc:
        raise MemoryCardError(f"cannot load repository records: {exc}") from exc

    related_artifacts = _related_artifacts_for_issue(records, issue_id)
    status_filter = ArtifactCardStatus(status) if status is not None else None
    scope_filter = {MemoryRootScope(scope) for scope in allowed_scopes}

    cards = []
    for loaded in records:
        if not isinstance(loaded.record, BaseArtifact):
            continue
        if related_artifacts is not None and loaded.record.id not in related_artifacts:
            continue

        card = artifact_card_from_loaded_record(loaded)
        if card.root_scope not in scope_filter:
            continue
        if status_filter is not None and card.status is not status_filter:
            continue
        cards.append(card)

    return tuple(sorted(cards, key=lambda card: (card.id, card.path)))


def artifact_card_from_loaded_record(loaded: LoadedRecord) -> ArtifactCard:
    """Return one compact card for a loaded lifecycle artifact record."""
    artifact = loaded.record
    if not isinstance(artifact, BaseArtifact):
        raise MemoryCardError(f"record is not an artifact: {loaded.id}")

    root_scope = _root_scope_for_loaded_record(loaded)
    return ArtifactCard(
        id=artifact.id,
        path=loaded.source_path.as_posix(),
        root_scope=root_scope,
        type=ArtifactCardType(artifact.type.value),
        status=ArtifactCardStatus(artifact.status.value),
        title=artifact.title,
        summary=_card_summary(artifact),
        domain=artifact.domain,
        tags=artifact.tags,
        depends_on=artifact.depends_on,
        sources=_source_labels(artifact.sources),
        review_state=artifact.review.state,
        verifier_state="not_run",
        formalization_state=_formalization_state(artifact),
        failure_count=len(artifact.failure_log),
        recent_failure_directions=_recent_failure_directions(artifact),
        trust_score=_trust_score(artifact.status.value),
        retrieval_score=0.0,
        why_relevant="repository artifact metadata",
        risk_flags=_risk_flags(artifact, root_scope),
        can_pull_full=False,
    )


def _related_artifacts_for_issue(
    records: tuple[LoadedRecord, ...],
    issue_id: str | None,
) -> frozenset[str] | None:
    if issue_id is None:
        return None
    validated_issue_id = validate_artifact_id(issue_id)
    matches = [
        record.record
        for record in records
        if isinstance(record.record, IssueRecord)
        and record.record.id == validated_issue_id
    ]
    if not matches:
        raise MemoryCardError(f"issue not found: {validated_issue_id}")
    if len(matches) > 1:
        raise MemoryCardError(f"duplicate issue id: {validated_issue_id}")
    return frozenset(matches[0].related_artifacts)


def _root_scope_for_loaded_record(loaded: LoadedRecord) -> MemoryRootScope:
    if loaded.is_private_scope:
        return MemoryRootScope.PRIVATE
    kb_root_name = (loaded.kb_root_name or "").lower()
    if kb_root_name == "framework":
        return MemoryRootScope.FRAMEWORK
    if loaded.is_public_kb_root or kb_root_name == "public":
        return MemoryRootScope.PUBLIC
    return MemoryRootScope.WORKSPACE


def _card_summary(artifact: BaseArtifact) -> str:
    domain = ", ".join(artifact.domain) if artifact.domain else "unspecified domain"
    return (
        f"{artifact.type.value} card for {artifact.title} "
        f"({artifact.status.value}; {domain})."
    )


def _source_labels(sources: list[SourceMetadata]) -> list[str]:
    labels = []
    for source in sources:
        label = (
            source.doi
            or source.arxiv
            or source.url
            or source.title
            or source.kind
        )
        labels.append(label)
    return labels


def _formalization_state(artifact: BaseArtifact) -> str:
    if not artifact.formalizations:
        return "none"
    statuses = sorted(
        {formalization.status for formalization in artifact.formalizations}
    )
    return ",".join(statuses)


def _recent_failure_directions(
    artifact: BaseArtifact,
    *,
    limit: int = 3,
) -> list[str]:
    entries = sorted(
        artifact.failure_log,
        key=lambda entry: (entry.attempted_at, entry.failure_id),
        reverse=True,
    )
    return [entry.direction for entry in entries[:limit]]


def _risk_flags(
    artifact: BaseArtifact,
    root_scope: MemoryRootScope,
) -> list[str]:
    flags = []
    if root_scope is MemoryRootScope.PRIVATE:
        flags.append("private")
    if artifact.status.value == "draft":
        flags.append("draft")
    if artifact.status.value in {"refuted", "obsolete", "superseded"}:
        flags.append(artifact.status.value)
    if artifact.risk.level != "low":
        flags.append(f"risk:{artifact.risk.level}")
    flags.extend(
        f"failure-log:{status}"
        for status in sorted({entry.status for entry in artifact.failure_log})
    )
    return flags


def _trust_score(status: str) -> float:
    return {
        "accepted": 1.0,
        "human_reviewed": 0.8,
        "machine_checked": 0.7,
        "adversarially_tested": 0.6,
        "locally_tested": 0.5,
        "draft": 0.1,
        "raw": 0.0,
        "refuted": 0.0,
        "obsolete": 0.0,
        "superseded": 0.0,
    }.get(status, 0.0)
