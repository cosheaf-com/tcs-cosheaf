"""Status/path and evidence path gates."""

from __future__ import annotations

from pathlib import Path

from cosheaf.core.artifact import (
    BaseArtifact,
    Evidence,
    evidence_revision_mismatches,
    review_revision_mismatch,
)
from cosheaf.core.status import ArtifactStatus, expected_status_for_path
from cosheaf.gates.schema_gate import ValidationFailure
from cosheaf.storage.loader import LoadedRecord
from cosheaf.storage.repo import RepoContext


def validate_status_paths(records: tuple[LoadedRecord, ...]) -> list[ValidationFailure]:
    """Validate artifact lifecycle status against repository path rules."""
    failures: list[ValidationFailure] = []

    for loaded in records:
        record = loaded.record
        if not isinstance(record, BaseArtifact):
            continue
        allowed = expected_status_for_path(_status_path_for_loaded_record(loaded))
        if record.status not in allowed:
            failures.append(
                ValidationFailure(
                    gate="status/path",
                    source_path=loaded.source_path.as_posix(),
                    artifact_id=record.id,
                    message=(
                        "status/path mismatch: "
                        f"status {record.status.value} is not allowed here; "
                        f"expected one of {_format_statuses(allowed)}"
                    ),
                )
            )
        if record.status is ArtifactStatus.ACCEPTED:
            mismatch = review_revision_mismatch(record, records)
            if mismatch is not None:
                failures.append(
                    ValidationFailure(
                        gate="status/path",
                        source_path=loaded.source_path.as_posix(),
                        artifact_id=record.id,
                        message=mismatch,
                    )
                )

    return failures


def _status_path_for_loaded_record(loaded: LoadedRecord) -> str:
    if loaded.kb_relative_path is None:
        return loaded.source_path.as_posix()
    relative = loaded.kb_relative_path.as_posix()
    return "kb" if not relative else f"kb/{relative}"


def validate_evidence_paths(
    context: RepoContext,
    records: tuple[LoadedRecord, ...],
) -> list[ValidationFailure]:
    """Validate repository-local evidence paths for loaded artifacts."""
    failures: list[ValidationFailure] = []

    for loaded in records:
        record = loaded.record
        if not isinstance(record, BaseArtifact):
            continue
        for _, message in evidence_revision_mismatches(record):
            failures.append(
                ValidationFailure(
                    gate="evidence path",
                    source_path=loaded.source_path.as_posix(),
                    artifact_id=record.id,
                    message=message,
                )
            )
        for evidence in record.evidence:
            if _is_external_evidence(evidence):
                continue
            failures.extend(_validate_one_evidence_path(context, loaded, evidence))

    return failures


def _validate_one_evidence_path(
    context: RepoContext,
    loaded: LoadedRecord,
    evidence: Evidence,
) -> list[ValidationFailure]:
    artifact = loaded.record
    if not isinstance(artifact, BaseArtifact):
        return []

    if not evidence.path.strip():
        return [
            ValidationFailure(
                gate="evidence path",
                source_path=loaded.source_path.as_posix(),
                artifact_id=artifact.id,
                message="missing evidence path: <empty>",
            )
        ]

    raw_path = Path(evidence.path)
    resolved_path = (
        raw_path.resolve() if raw_path.is_absolute() else context.resolve(raw_path)
    )

    try:
        resolved_path.relative_to(context.repo_root)
    except ValueError:
        return [
            ValidationFailure(
                gate="evidence path",
                source_path=loaded.source_path.as_posix(),
                artifact_id=artifact.id,
                message=f"evidence path escapes repository: {evidence.path}",
            )
        ]

    if resolved_path.exists():
        return []

    return [
        ValidationFailure(
            gate="evidence path",
            source_path=loaded.source_path.as_posix(),
            artifact_id=artifact.id,
            message=f"missing evidence path: {evidence.path}",
        )
    ]


def _format_statuses(statuses: frozenset[ArtifactStatus]) -> str:
    return ", ".join(sorted(status.value for status in statuses))


def _is_external_evidence(evidence: Evidence) -> bool:
    return (
        evidence.kind.strip().lower() == "external"
        or evidence.path.strip().lower().startswith("external:")
    )
