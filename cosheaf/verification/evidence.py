"""Serializable verifier evidence records."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from enum import StrEnum
from pathlib import PurePosixPath, PureWindowsPath
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from cosheaf.core.ids import validate_artifact_id
from cosheaf.core.paths import normalize_repo_path
from cosheaf.verification.result import VerificationResult, VerificationStatus

REASON_CODE_PATTERN = re.compile(r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)*$")
SHA256_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
DEFAULT_LIMITATIONS = (
    "Verifier evidence is not human review.",
    "Verifier evidence does not auto-promote accepted knowledge.",
)
SKIPPED_LIMITATION = "Skipped verifier evidence is not a pass."
LEAN_REF_LIMITATION = (
    "Lean #check resolves imports and symbols only; semantic alignment is not checked."
)


class VerifierEvidenceKind(StrEnum):
    """Verifier evidence backend family."""

    PYTHON = "python"
    SAT = "sat"
    SMT = "smt"
    LEAN = "lean"
    EXTERNAL_REFERENCE = "external_reference"
    MANUAL_NOTE = "manual_note"


class VerifierEvidenceRecord(BaseModel):
    """Durable v1 verifier evidence record.

    This record is a serialization boundary for verifier outputs. It is not a
    human review record and does not authorize accepted promotion.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, use_enum_values=False)

    evidence_id: str
    artifact_id: str | None = None
    claim_id: str | None = None
    verifier_kind: VerifierEvidenceKind
    tool_name: str
    tool_version: str | None = None
    command_argv: tuple[str, ...] | None = None
    cwd: str | None = None
    result: VerificationStatus
    reason_code: str
    stdout_path: str | None = None
    stderr_path: str | None = None
    log_path: str | None = None
    created_at: datetime
    checker_input_hash: str | None = None
    checker_output_hash: str | None = None
    target_revision: str | None = None
    limitations: tuple[str, ...]

    @classmethod
    def from_verification_result(
        cls,
        result: VerificationResult,
        *,
        evidence_id: str | None = None,
        claim_id: str | None = None,
        reason_code: str | None = None,
        limitations: tuple[str, ...] | list[str] | None = None,
        checker_input_hash: str | None = None,
        checker_output_hash: str | None = None,
        target_revision: str | None = None,
        log_path: str | None = None,
    ) -> VerifierEvidenceRecord:
        """Create a v1 evidence record from an existing runtime result."""
        verifier_kind = _kind_from_verifier(result.verifier)
        normalized_limitations = _default_limitations(
            status=result.status,
            kind=verifier_kind,
            extra=limitations,
        )
        return cls(
            evidence_id=evidence_id or _evidence_id(result, verifier_kind),
            artifact_id=result.artifact_id,
            claim_id=claim_id,
            verifier_kind=verifier_kind,
            tool_name=result.tool_name or result.verifier,
            tool_version=result.tool_version,
            command_argv=result.command,
            cwd=result.cwd,
            result=result.status,
            reason_code=reason_code or _reason_code(result.status),
            stdout_path=result.stdout_path,
            stderr_path=result.stderr_path,
            log_path=log_path,
            created_at=result.ended_at,
            checker_input_hash=checker_input_hash,
            checker_output_hash=checker_output_hash,
            target_revision=target_revision,
            limitations=normalized_limitations,
        )

    @field_validator("evidence_id")
    @classmethod
    def _evidence_id(cls, value: str) -> str:
        return validate_artifact_id(value.strip())

    @field_validator("artifact_id", "claim_id")
    @classmethod
    def _optional_artifact_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return validate_artifact_id(value.strip())

    @field_validator("tool_name", "reason_code")
    @classmethod
    def _required_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("text field must be non-empty")
        return normalized

    @field_validator("reason_code")
    @classmethod
    def _reason_code(cls, value: str) -> str:
        if not REASON_CODE_PATTERN.fullmatch(value):
            raise ValueError(
                "reason_code must be lowercase snake_case, for example "
                "verifier_skipped"
            )
        return value

    @field_validator("tool_version")
    @classmethod
    def _optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        return normalized or None

    @field_validator("command_argv", mode="before")
    @classmethod
    def _command_argv(cls, value: Any) -> tuple[str, ...] | None:
        if value is None:
            return None
        if isinstance(value, str):
            raise ValueError("command_argv must be a sequence of arguments")
        try:
            command = tuple(str(part) for part in value)
        except TypeError as exc:
            raise ValueError(
                "command_argv must be a sequence of arguments"
            ) from exc
        return command or None

    @field_validator("stdout_path", "stderr_path", "log_path")
    @classmethod
    def _repo_path(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _validate_repo_local_path(value)

    @field_validator("created_at")
    @classmethod
    def _created_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("created_at must include timezone information")
        return value

    @field_validator("checker_input_hash", "checker_output_hash", "target_revision")
    @classmethod
    def _sha256(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip().lower()
        if not SHA256_PATTERN.fullmatch(normalized):
            raise ValueError("checker hash must use sha256:<64 lowercase hex>")
        return normalized

    @field_validator("limitations", mode="before")
    @classmethod
    def _limitations(cls, value: Any) -> tuple[str, ...]:
        if value is None:
            return ()
        values: tuple[str, ...]
        if isinstance(value, str):
            values = (value,)
        else:
            try:
                values = tuple(str(item) for item in value)
            except TypeError as exc:
                raise ValueError("limitations must be a sequence") from exc
        normalized = tuple(
            dict.fromkeys(item.strip() for item in values if item.strip())
        )
        if not normalized:
            raise ValueError("limitations must contain at least one item")
        return normalized

    @model_validator(mode="after")
    def _record_consistency(self) -> Self:
        if self.command_argv and not self.cwd:
            raise ValueError("cwd is required when command_argv is recorded")
        if (
            self.result is VerificationStatus.SKIPPED
            and SKIPPED_LIMITATION not in self.limitations
        ):
            raise ValueError("skipped evidence records must say skipped is not pass")
        return self

    @property
    def is_pass(self) -> bool:
        """Return whether this evidence record has a passing verifier result."""
        return self.result is VerificationStatus.PASS

    @property
    def is_skipped(self) -> bool:
        """Return whether this evidence record is an honest skip."""
        return self.result is VerificationStatus.SKIPPED

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic JSON-serializable mapping."""
        dumped = self.model_dump(mode="json")
        if dumped.get("target_revision") is None:
            dumped.pop("target_revision", None)
        return dumped

    def to_json(self) -> str:
        """Return deterministic JSON for this evidence record."""
        return json.dumps(self.to_dict(), ensure_ascii=True, indent=2) + "\n"


def _kind_from_verifier(verifier: str) -> VerifierEvidenceKind:
    normalized = verifier.strip().lower()
    if normalized == "python_checker" or normalized.startswith("python"):
        return VerifierEvidenceKind.PYTHON
    if normalized == "sat":
        return VerifierEvidenceKind.SAT
    if normalized == "smt":
        return VerifierEvidenceKind.SMT
    if normalized == "lean":
        return VerifierEvidenceKind.LEAN
    if normalized == "lean_library_ref":
        return VerifierEvidenceKind.EXTERNAL_REFERENCE
    return VerifierEvidenceKind.MANUAL_NOTE


def _reason_code(status: VerificationStatus) -> str:
    if status is VerificationStatus.PASS:
        return "verifier_passed"
    if status is VerificationStatus.FAIL:
        return "verifier_failed"
    if status is VerificationStatus.ERROR:
        return "verifier_error"
    return "verifier_skipped"


def _default_limitations(
    *,
    status: VerificationStatus,
    kind: VerifierEvidenceKind,
    extra: tuple[str, ...] | list[str] | None,
) -> tuple[str, ...]:
    values = list(DEFAULT_LIMITATIONS)
    if status is VerificationStatus.SKIPPED:
        values.append(SKIPPED_LIMITATION)
    if kind is VerifierEvidenceKind.EXTERNAL_REFERENCE:
        values.append(LEAN_REF_LIMITATION)
    if extra is not None:
        values.extend(extra)
    return tuple(dict.fromkeys(item.strip() for item in values if item.strip()))


def _evidence_id(
    result: VerificationResult,
    verifier_kind: VerifierEvidenceKind,
) -> str:
    payload = {
        "artifact_id": result.artifact_id,
        "verifier": result.verifier,
        "kind": verifier_kind.value,
        "status": result.status.value,
        "command": list(result.command or ()),
        "cwd": result.cwd,
        "exit_code": result.exit_code,
        "stdout_path": result.stdout_path,
        "stderr_path": result.stderr_path,
        "evidence_paths": list(result.evidence_paths),
        "input_paths": list(result.input_paths),
        "output_paths": list(result.output_paths),
        "tool_name": result.tool_name,
        "tool_version": result.tool_version,
        "environment": result.environment,
        "message": result.message,
    }
    digest = hashlib.sha256(
        json.dumps(payload, ensure_ascii=True, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    kind_segment = verifier_kind.value.replace("_", "-")
    return f"verifier-evidence.{result.artifact_id}.{kind_segment}.h{digest}"


def _validate_repo_local_path(value: str) -> str:
    normalized = normalize_repo_path(value)
    is_absolute = PureWindowsPath(value).is_absolute()
    parts = PurePosixPath(normalized).parts
    if (
        not normalized
        or is_absolute
        or normalized == ".."
        or normalized.startswith("../")
        or ".." in parts
    ):
        raise ValueError("path must be repository-local")
    return normalized
