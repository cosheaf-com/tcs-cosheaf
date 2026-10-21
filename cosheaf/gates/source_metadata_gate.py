"""Source metadata policy gate for accepted public artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from cosheaf.core.artifact import BaseArtifact, SourceMetadata
from cosheaf.core.status import ArtifactStatus
from cosheaf.gates.schema_gate import ValidationFailure, sort_failures
from cosheaf.storage.loader import LoadedRecord
from cosheaf.storage.repo import RepoContext

CheckStatus = Literal["pass", "fail"]

LEGACY_SOURCE_POLICY_REASON = (
    "Legacy single-root mode has no public KB root; "
    "source metadata policy is not enforced."
)
DISABLED_SOURCE_POLICY_REASON = (
    "accepted_requires_source is false; source metadata policy is not enforced."
)


@dataclass(frozen=True)
class SourceMetadataCheck:
    """One accepted public artifact source metadata check row."""

    artifact_id: str
    source_path: str
    kb_root: str
    status: CheckStatus
    source_count: int
    missing_metadata: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "artifact_id": self.artifact_id,
            "source_path": self.source_path,
            "kb_root": self.kb_root,
            "status": self.status,
            "source_count": self.source_count,
            "missing_metadata": list(self.missing_metadata),
        }


@dataclass(frozen=True)
class SourceMetadataResult:
    """Aggregate result for the source metadata policy gate."""

    checks: tuple[SourceMetadataCheck, ...]
    failures: tuple[ValidationFailure, ...]
    policy_reason: str = ""

    @property
    def applicable_count(self) -> int:
        return len(self.checks)

    @property
    def ok(self) -> bool:
        return not self.failures


def validate_source_metadata_policy(
    context: RepoContext,
    records: tuple[LoadedRecord, ...],
) -> SourceMetadataResult:
    """Validate configured public accepted artifacts have complete source metadata."""
    if not context.workspace_config.configured:
        return SourceMetadataResult(
            checks=(),
            failures=(),
            policy_reason=LEGACY_SOURCE_POLICY_REASON,
        )
    if not context.workspace_config.policy.accepted_requires_source:
        return SourceMetadataResult(
            checks=(),
            failures=(),
            policy_reason=DISABLED_SOURCE_POLICY_REASON,
        )

    checks: list[SourceMetadataCheck] = []
    failures: list[ValidationFailure] = []
    for loaded in sorted(records, key=_record_sort_key):
        artifact = loaded.record
        if not isinstance(artifact, BaseArtifact):
            continue
        if not _requires_source_metadata(loaded, artifact):
            continue

        missing_metadata = missing_required_source_metadata(artifact)
        check = SourceMetadataCheck(
            artifact_id=artifact.id,
            source_path=loaded.source_path.as_posix(),
            kb_root=loaded.kb_root_name or "",
            status="fail" if missing_metadata else "pass",
            source_count=len(artifact.sources),
            missing_metadata=missing_metadata,
        )
        checks.append(check)
        if missing_metadata:
            failures.append(_failure_from_check(check))

    return SourceMetadataResult(
        checks=tuple(sorted(checks, key=_check_sort_key)),
        failures=sort_failures(failures),
    )


def _requires_source_metadata(
    loaded: LoadedRecord,
    artifact: BaseArtifact,
) -> bool:
    return (
        loaded.is_public_kb_root
        and artifact.status is ArtifactStatus.ACCEPTED
    )


def missing_required_source_metadata(artifact: BaseArtifact) -> tuple[str, ...]:
    """Return missing required source metadata fields for an artifact."""
    sources = artifact.sources
    if not sources:
        return ("sources",)

    missing: list[str] = []
    for index, source in enumerate(sources):
        prefix = f"source[{index}]"
        if not source.title:
            missing.append(f"{prefix}.title")
        if not source.authors:
            missing.append(f"{prefix}.authors")
        if source.year is None:
            missing.append(f"{prefix}.year")
        if not _has_citation_locator(source):
            missing.append(f"{prefix}.citation_locator")
    return tuple(missing)


def _has_citation_locator(source: SourceMetadata) -> bool:
    return any(
        (
            source.doi,
            source.arxiv,
            source.url,
            source.theorem_number,
            source.page,
        )
    )


def _failure_from_check(check: SourceMetadataCheck) -> ValidationFailure:
    if check.missing_metadata == ("sources",):
        message = "accepted public artifact requires source metadata"
    else:
        message = "incomplete source metadata: " + ", ".join(
            check.missing_metadata
        )
    return ValidationFailure(
        gate="source metadata",
        source_path=check.source_path,
        artifact_id=check.artifact_id,
        message=message,
    )


def _record_sort_key(record: LoadedRecord) -> tuple[str, str]:
    return (record.source_path.as_posix(), record.id)


def _check_sort_key(check: SourceMetadataCheck) -> tuple[str, str]:
    return (check.source_path, check.artifact_id)
