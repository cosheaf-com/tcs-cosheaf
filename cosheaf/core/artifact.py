"""Pydantic models for typed research artifacts."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from pathlib import PurePosixPath
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from cosheaf.core.formal_library import validate_library_ref
from cosheaf.core.ids import validate_artifact_id
from cosheaf.core.status import ArtifactStatus, ArtifactType

SHA256_REVISION_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")


def validate_revision_hash(value: str | None) -> str | None:
    """Validate an optional sha256 content revision hash."""
    if value is None:
        return None
    normalized = value.strip().lower()
    if not normalized:
        return None
    if not SHA256_REVISION_PATTERN.fullmatch(normalized):
        raise ValueError("revision hash must use sha256:<64 lowercase hex>")
    return normalized


class ArtifactRevision(BaseModel):
    """Deterministic semantic revision descriptor for an artifact."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    artifact_id: str
    content_hash: str
    depends_on: tuple[str, ...] = Field(default_factory=tuple)

    @field_validator("artifact_id")
    @classmethod
    def _validate_artifact_id(cls, value: str) -> str:
        return validate_artifact_id(value)

    @field_validator("content_hash")
    @classmethod
    def _validate_content_hash(cls, value: str) -> str:
        validated = validate_revision_hash(value)
        if validated is None:
            raise ValueError("content_hash must not be empty")
        return validated


class Evidence(BaseModel):
    """A repository-local evidence reference."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: str
    path: str
    summary: str
    target_revision: str | None = None

    @field_validator("target_revision")
    @classmethod
    def _validate_target_revision(cls, value: str | None) -> str | None:
        return validate_revision_hash(value)


class ReviewRef(BaseModel):
    """Inline review state attached to an artifact."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    state: Literal[
        "none",
        "requested",
        "in_review",
        "approved",
        "changes_requested",
        "human_reviewed",
        "accepted",
    ] = "none"
    notes: str = ""
    reviewed_revision: str | None = None

    @field_validator("reviewed_revision")
    @classmethod
    def _validate_reviewed_revision(cls, value: str | None) -> str | None:
        return validate_revision_hash(value)


class SourceMetadata(BaseModel):
    """Structured citation metadata for source-backed artifacts."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal[
        "paper",
        "book",
        "survey",
        "lecture_note",
        "website",
        "internal_note",
        "other",
    ]
    title: str = ""
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    doi: str = ""
    arxiv: str = ""
    url: str = ""
    theorem_number: str = ""
    page: str = ""
    notes: str = ""

    @field_validator(
        "title",
        "doi",
        "arxiv",
        "url",
        "theorem_number",
        "page",
        "notes",
    )
    @classmethod
    def _strip_strings(cls, value: str) -> str:
        return value.strip()

    @field_validator("authors")
    @classmethod
    def _strip_authors(cls, values: list[str]) -> list[str]:
        return [value.strip() for value in values if value.strip()]

    @field_validator("year")
    @classmethod
    def _validate_year(cls, value: int | None) -> int | None:
        if value is not None and value <= 0:
            raise ValueError("year must be positive")
        return value


class FormalizationRef(BaseModel):
    """Reference to an external formal library declaration."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(min_length=1)
    system: Literal["lean4"]
    library: str = Field(min_length=1)
    library_ref: str = Field(min_length=1)
    import_path: str = Field(min_length=1)
    symbol: str = Field(min_length=1)
    declaration_kind: Literal[
        "definition",
        "theorem",
        "lemma",
        "instance",
        "structure",
        "other",
    ]
    status: Literal["planned", "linked", "checked", "broken", "deprecated"]
    check_mode: Literal["external_library_ref", "local_file"]
    expected_type: str = ""
    notes: str = ""

    @field_validator(
        "id",
        "library",
        "library_ref",
        "import_path",
        "symbol",
        "expected_type",
        "notes",
    )
    @classmethod
    def _strip_strings(cls, value: str) -> str:
        return value.strip()

    @field_validator("id", "library", "library_ref", "import_path", "symbol")
    @classmethod
    def _validate_required_strings(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("formalization reference field must not be empty")
        return stripped

    @field_validator("id")
    @classmethod
    def _validate_id(cls, value: str) -> str:
        return validate_artifact_id(value)

    @field_validator("library_ref")
    @classmethod
    def _validate_library_ref(cls, value: str) -> str:
        return validate_library_ref(value)


class AlignmentReview(BaseModel):
    """Semantic alignment review between informal and formal statements."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: Literal["none", "requested", "human_reviewed", "rejected"] = "none"
    reviewer: str = ""
    reviewed_at: datetime | None = None
    convention_notes: list[str] = Field(default_factory=list)
    limitations: str = ""
    reviewed_revision: str | None = None

    @field_validator("reviewed_revision")
    @classmethod
    def _validate_reviewed_revision(cls, value: str | None) -> str | None:
        return validate_revision_hash(value)

    @field_validator("reviewer", "limitations")
    @classmethod
    def _strip_strings(cls, value: str) -> str:
        return value.strip()

    @field_validator("convention_notes")
    @classmethod
    def _strip_convention_notes(cls, values: list[str]) -> list[str]:
        return [value.strip() for value in values if value.strip()]

    @field_validator("reviewed_at")
    @classmethod
    def _validate_review_timestamp(cls, value: datetime | None) -> datetime | None:
        if value is not None and (value.tzinfo is None or value.utcoffset() is None):
            raise ValueError("reviewed_at must include timezone information")
        return value

    @model_validator(mode="after")
    def _validate_reviewer_for_completed_review(self) -> AlignmentReview:
        if self.status in {"human_reviewed", "rejected"} and not self.reviewer:
            raise ValueError(
                "reviewer must be non-empty when alignment status is "
                "human_reviewed or rejected"
            )
        return self


class VerificationPolicy(BaseModel):
    """Per-artifact verification policy for formal links and alignment review."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    level: Literal[
        "source_reviewed",
        "source_reviewed_with_formal_link",
        "machine_checked",
        "lean_required",
    ] = "source_reviewed"
    require_formal_link: bool = False
    require_lean_check: bool = False
    require_alignment_review: bool = False

    @model_validator(mode="after")
    def _validate_policy_consistency(self) -> VerificationPolicy:
        if (
            self.level == "source_reviewed_with_formal_link"
            and not self.require_formal_link
        ):
            raise ValueError(
                "source_reviewed_with_formal_link policy must require a formal link"
            )
        if self.level == "lean_required" and not self.require_formal_link:
            raise ValueError("lean_required policy must require a formal link")
        if self.level == "lean_required" and not self.require_lean_check:
            raise ValueError(
                "lean_required policy must require a Lean check"
            )
        return self


class Risk(BaseModel):
    """Risk classification for an artifact."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    level: Literal["low", "medium", "high"] = "low"
    notes: str = ""


class FailureLogEntry(BaseModel):
    """Durable failed-attempt memory attached to an artifact."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    failure_id: str
    attempted_at: datetime
    recorded_by: str
    origin: Literal["human", "agent", "provider", "verifier", "imported_bundle"]
    attempt_kind: Literal[
        "proof_attempt",
        "reduction_attempt",
        "construction_attempt",
        "counterexample_search",
        "formalization_attempt",
        "verifier_attempt",
        "retrieval_attempt",
        "other",
    ]
    target: str = ""
    direction: str
    summary: str
    failed_because: str
    evidence_paths: list[str] = Field(default_factory=list)
    related_verifier_results: list[str] = Field(default_factory=list)
    related_counterexample_candidates: list[str] = Field(default_factory=list)
    next_possible_directions: list[str] = Field(default_factory=list)
    status: Literal["open", "superseded", "invalidated", "resolved", "archived"]
    limitations: str

    @field_validator("failure_id")
    @classmethod
    def _validate_failure_id(cls, value: str) -> str:
        return validate_artifact_id(value.strip())

    @field_validator("attempted_at")
    @classmethod
    def _validate_attempt_timestamp(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("attempted_at must include timezone information")
        return value

    @field_validator(
        "recorded_by",
        "direction",
        "summary",
        "failed_because",
        "limitations",
    )
    @classmethod
    def _strip_required_strings(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("failure_log text fields must not be empty")
        return stripped

    @field_validator("target")
    @classmethod
    def _validate_target(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            return ""
        return validate_dependency_ref(stripped)

    @field_validator("evidence_paths")
    @classmethod
    def _validate_evidence_paths(cls, values: list[str]) -> list[str]:
        return [_validate_failure_log_path(value) for value in values]

    @field_validator(
        "related_verifier_results",
        "related_counterexample_candidates",
        "next_possible_directions",
    )
    @classmethod
    def _strip_reference_list(cls, values: list[str]) -> list[str]:
        stripped_values: list[str] = []
        for value in values:
            stripped = value.strip()
            if not stripped:
                raise ValueError("failure_log list entries must not be empty")
            stripped_values.append(stripped)
        return stripped_values


class BaseArtifact(BaseModel):
    """Base typed artifact model shared by current artifact examples."""

    model_config = ConfigDict(extra="forbid", use_enum_values=False)

    id: str
    type: ArtifactType
    title: str
    domain: list[str] = Field(default_factory=list)
    status: ArtifactStatus
    created_at: datetime
    updated_at: datetime
    authors: list[str] = Field(default_factory=list)
    depends_on: list[str] = Field(default_factory=list)
    supersedes: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    statement: str
    evidence: list[Evidence] = Field(default_factory=list)
    sources: list[SourceMetadata] = Field(default_factory=list)
    formalizations: list[FormalizationRef] = Field(default_factory=list)
    alignment: AlignmentReview = Field(default_factory=AlignmentReview)
    verification_policy: VerificationPolicy = Field(
        default_factory=VerificationPolicy
    )
    review: ReviewRef = Field(default_factory=ReviewRef)
    risk: Risk = Field(default_factory=Risk)
    failure_log: list[FailureLogEntry] = Field(default_factory=list)

    @field_validator("id")
    @classmethod
    def _validate_id(cls, value: str) -> str:
        return validate_artifact_id(value)

    @field_validator("depends_on")
    @classmethod
    def _validate_dependency_refs(cls, values: list[str]) -> list[str]:
        return [validate_dependency_ref(value) for value in values]

    @field_validator("supersedes")
    @classmethod
    def _validate_artifact_id_list(cls, values: list[str]) -> list[str]:
        return [validate_artifact_id(value) for value in values]

    @field_validator("created_at", "updated_at")
    @classmethod
    def _validate_timestamp(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("timestamp must include timezone information")
        return value

    @field_validator("updated_at")
    @classmethod
    def _validate_update_order(cls, value: datetime, info: Any) -> datetime:
        created_at = info.data.get("created_at")
        if isinstance(created_at, datetime) and value < created_at:
            raise ValueError("updated_at must be greater than or equal to created_at")
        return value

    @property
    def content_revision(self) -> str:
        """Return the deterministic semantic content hash of this artifact."""
        return compute_artifact_revision(self)

    def revision_info(self) -> ArtifactRevision:
        """Return the structured ArtifactRevision descriptor for this artifact."""
        return build_artifact_revision(self)


def compute_artifact_revision(artifact: BaseArtifact) -> str:
    """Compute a deterministic sha256 content hash over semantic artifact fields."""
    payload = {
        "id": artifact.id,
        "type": artifact.type.value,
        "title": artifact.title,
        "domain": list(artifact.domain),
        "statement": artifact.statement,
        "depends_on": list(artifact.depends_on),
        "supersedes": list(artifact.supersedes),
        "formalizations": [
            item.model_dump(mode="json") for item in artifact.formalizations
        ],
        "verification_policy": artifact.verification_policy.model_dump(mode="json"),
    }
    encoded = json.dumps(payload, ensure_ascii=True, sort_keys=True).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def build_artifact_revision(artifact: BaseArtifact) -> ArtifactRevision:
    """Build an ArtifactRevision descriptor for an artifact."""
    return ArtifactRevision(
        artifact_id=artifact.id,
        content_hash=compute_artifact_revision(artifact),
        depends_on=tuple(artifact.depends_on),
    )


HUMAN_PROMOTION_REVIEW_DECISIONS = frozenset(
    {
        "approve",
        "accept_for_private_use",
        "accept_for_public_candidate",
    }
)


def review_revision_mismatch(
    artifact: BaseArtifact,
    records: Any = (),
) -> str | None:
    """Return an error message if a human review targets a stale revision."""
    current_revision = artifact.content_revision
    if (
        artifact.review.reviewed_revision is not None
        and artifact.review.reviewed_revision != current_revision
    ):
        return (
            "stale_review: review was recorded for revision "
            f"{artifact.review.reviewed_revision}, but current artifact revision "
            f"is {current_revision} for {artifact.id}"
        )

    matching_decisions: list[tuple[datetime, str, str]] = []
    for item in records:
        record = getattr(item, "record", item)
        if getattr(record, "type", None) != "review":
            continue
        if getattr(record, "target", None) != artifact.id:
            continue
        if getattr(record, "status", None) not in {"human_reviewed", "accepted"}:
            continue
        if getattr(record, "decision", None) not in HUMAN_PROMOTION_REVIEW_DECISIONS:
            continue
        rev = getattr(record, "reviewed_revision", None)
        if rev is not None:
            matching_decisions.append(
                (getattr(record, "updated_at"), getattr(record, "id", ""), rev)
            )

    if matching_decisions:
        matching_decisions.sort(key=lambda entry: (entry[0], entry[1]))
        _, latest_id, latest_rev = matching_decisions[-1]
        if latest_rev != current_revision:
            return (
                f"stale_review: review decision {latest_id} was recorded for "
                f"revision {latest_rev}, but current artifact revision is "
                f"{current_revision} for {artifact.id}"
            )
    return None


def evidence_revision_mismatches(
    artifact: BaseArtifact,
) -> tuple[tuple[str, str], ...]:
    """Return (path, message) pairs for evidence items bound to a stale revision."""
    current_revision = artifact.content_revision
    mismatches: list[tuple[str, str]] = []
    for evidence in artifact.evidence:
        if (
            evidence.target_revision is not None
            and evidence.target_revision != current_revision
        ):
            mismatches.append(
                (
                    evidence.path,
                    f"stale_evidence: evidence {evidence.path} targets revision "
                    f"{evidence.target_revision}, but current artifact revision "
                    f"is {current_revision} for {artifact.id}",
                )
            )
    return tuple(mismatches)


def is_external_dependency_ref(value: str) -> bool:
    """Return whether a dependency reference is explicitly external."""
    normalized = value.strip().lower()
    return normalized.startswith("external:") and bool(
        normalized.removeprefix("external:").strip()
    )


def validate_dependency_ref(value: str) -> str:
    """Validate a local artifact ID or explicit external dependency reference."""
    stripped = value.strip()
    if is_external_dependency_ref(stripped):
        return stripped
    return validate_artifact_id(stripped)


def _validate_failure_log_path(value: str) -> str:
    """Validate a repository-local failure-log support path."""
    normalized = value.strip().replace("\\", "/")
    if not normalized:
        raise ValueError("failure_log evidence path must not be empty")
    if normalized.startswith("/") or re.match(r"^[A-Za-z]:", normalized):
        raise ValueError("failure_log evidence path must be repository-local")

    path = PurePosixPath(normalized)
    parts = path.parts
    if ".." in parts:
        raise ValueError("failure_log evidence path must not traverse parents")
    if parts and parts[0] == "kb" and "accepted" in parts[1:]:
        raise ValueError(
            "failure_log evidence path must not target accepted KB paths"
        )
    return path.as_posix()
