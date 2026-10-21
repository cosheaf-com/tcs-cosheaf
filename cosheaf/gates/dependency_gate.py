"""ID uniqueness and dependency gates."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass

from cosheaf.core.artifact import BaseArtifact, is_external_dependency_ref
from cosheaf.core.status import ArtifactStatus, is_preaccepted_status
from cosheaf.gates.schema_gate import ValidationFailure
from cosheaf.storage.loader import LoadedRecord


@dataclass(frozen=True)
class TransitionDecision:
    """Shared policy decision for an artifact lifecycle status transition."""

    artifact_id: str
    old_status: ArtifactStatus
    target_status: ArtifactStatus
    allowed: bool
    blocking_reason_codes: tuple[str, ...] = ()
    blocking_messages: tuple[str, ...] = ()
    impacted_accepted_dependents: tuple[str, ...] = ()


def validate_id_uniqueness(
    records: tuple[LoadedRecord, ...],
) -> list[ValidationFailure]:
    """Validate global uniqueness for loaded record IDs."""
    paths_by_id: defaultdict[str, list[str]] = defaultdict(list)
    for loaded in records:
        paths_by_id[loaded.id].append(loaded.source_path.as_posix())

    failures: list[ValidationFailure] = []
    for artifact_id in sorted(paths_by_id):
        paths = sorted(paths_by_id[artifact_id])
        if len(paths) <= 1:
            continue
        failures.append(
            ValidationFailure(
                gate="id uniqueness",
                source_path=paths[0],
                artifact_id=artifact_id,
                message=f"duplicate id {artifact_id}: {', '.join(paths)}",
            )
        )
    return failures


def validate_dependencies(records: tuple[LoadedRecord, ...]) -> list[ValidationFailure]:
    """Validate dependency existence and accepted-to-draft dependency rules."""
    artifact_records = [
        loaded for loaded in records if isinstance(loaded.record, BaseArtifact)
    ]
    artifacts_by_id = {
        loaded.id: loaded for loaded in sorted(artifact_records, key=_record_sort_key)
    }

    failures: list[ValidationFailure] = []
    for loaded in sorted(artifact_records, key=_record_sort_key):
        artifact = loaded.record
        if not isinstance(artifact, BaseArtifact):
            continue
        for dependency_id in artifact.depends_on:
            if is_external_dependency_ref(dependency_id):
                continue
            dependency = artifacts_by_id.get(dependency_id)
            if dependency is None:
                failures.append(
                    ValidationFailure(
                        gate="dependency",
                        source_path=loaded.source_path.as_posix(),
                        artifact_id=artifact.id,
                        message=f"missing dependency: {dependency_id}",
                    )
                )
                continue

            dependency_record = dependency.record
            if not isinstance(dependency_record, BaseArtifact):
                continue
            if _public_depends_on_private(loaded, dependency):
                failures.append(
                    ValidationFailure(
                        gate="dependency",
                        source_path=loaded.source_path.as_posix(),
                        artifact_id=artifact.id,
                        message=(
                            "public artifact depends on private artifact: "
                            f"{dependency_id} at {dependency.source_path.as_posix()}"
                        ),
                    )
                )
            if (
                artifact.status is ArtifactStatus.ACCEPTED
                and is_preaccepted_status(dependency_record.status)
            ):
                failures.append(
                    ValidationFailure(
                        gate="dependency",
                        source_path=loaded.source_path.as_posix(),
                        artifact_id=artifact.id,
                        message=(
                            "accepted artifact depends on draft artifact: "
                            f"{dependency_id} at {dependency.source_path.as_posix()}"
                        ),
                    )
                )
            elif (
                artifact.status is ArtifactStatus.ACCEPTED
                and dependency_record.status is not ArtifactStatus.ACCEPTED
            ):
                failures.append(
                    ValidationFailure(
                        gate="dependency",
                        source_path=loaded.source_path.as_posix(),
                        artifact_id=artifact.id,
                        message=(
                            "accepted artifact depends on "
                            f"{dependency_record.status.value} artifact: "
                            f"{dependency_id} at {dependency.source_path.as_posix()}"
                        ),
                    )
                )

    return failures


def find_blocking_reverse_dependents(
    records: tuple[LoadedRecord, ...],
    artifact_id: str,
    target_status: ArtifactStatus,
) -> tuple[LoadedRecord, ...]:
    """Return accepted artifacts that depend (directly or transitively) on artifact_id
    when transitioning artifact_id to a non-accepted status.
    """
    if target_status is ArtifactStatus.ACCEPTED:
        return ()

    artifact_records = [
        loaded for loaded in records if isinstance(loaded.record, BaseArtifact)
    ]
    dependents_by_id: defaultdict[str, list[str]] = defaultdict(list)
    for loaded in artifact_records:
        record = loaded.record
        if not isinstance(record, BaseArtifact):
            continue
        for dep_id in record.depends_on:
            if not is_external_dependency_ref(dep_id):
                dependents_by_id[dep_id].append(loaded.id)

    visited: set[str] = set()
    queue: deque[str] = deque(dependents_by_id.get(artifact_id, ()))
    while queue:
        current = queue.popleft()
        if current == artifact_id or current in visited:
            continue
        visited.add(current)
        for nxt in dependents_by_id.get(current, ()):
            if nxt not in visited:
                queue.append(nxt)

    blocking = [
        loaded
        for loaded in sorted(artifact_records, key=_record_sort_key)
        if loaded.id in visited
        and isinstance(loaded.record, BaseArtifact)
        and loaded.record.status is ArtifactStatus.ACCEPTED
    ]
    return tuple(blocking)


def evaluate_status_transition(
    records: tuple[LoadedRecord, ...],
    artifact: BaseArtifact,
    target_status: ArtifactStatus,
) -> TransitionDecision:
    """Evaluate reverse-dependency and status eligibility for a status transition."""
    blocking_dependents = find_blocking_reverse_dependents(
        records,
        artifact.id,
        target_status,
    )
    if blocking_dependents:
        dependents_summary = ", ".join(
            f"{dep.id} at {dep.source_path.as_posix()}"
            for dep in blocking_dependents
        )
        message = (
            f"cannot transition {artifact.id} from {artifact.status.value} to "
            f"{target_status.value}: accepted artifact(s) depend on it: "
            f"{dependents_summary}"
        )
        return TransitionDecision(
            artifact_id=artifact.id,
            old_status=artifact.status,
            target_status=target_status,
            allowed=False,
            blocking_reason_codes=("reverse_dependency_violation",),
            blocking_messages=(message,),
            impacted_accepted_dependents=tuple(dep.id for dep in blocking_dependents),
        )
    return TransitionDecision(
        artifact_id=artifact.id,
        old_status=artifact.status,
        target_status=target_status,
        allowed=True,
    )


def _public_depends_on_private(source: LoadedRecord, dependency: LoadedRecord) -> bool:
    return source.kb_root_name == "public" and dependency.kb_root_name == "private"


def _record_sort_key(record: LoadedRecord) -> tuple[str, str]:
    return (record.source_path.as_posix(), record.id)

