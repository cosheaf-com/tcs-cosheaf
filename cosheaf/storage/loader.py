"""Filesystem-backed YAML artifact loader."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import yaml  # type: ignore[import-untyped]
from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
)
from yaml import YAMLError

from cosheaf.core.artifact import BaseArtifact, validate_revision_hash
from cosheaf.core.ids import validate_artifact_id
from cosheaf.core.paths import is_yaml_path, repo_relative_posix
from cosheaf.core.status import ArtifactType
from cosheaf.core.task import AgentTask
from cosheaf.storage.repo import RepoContext

ARTIFACT_TYPE_VALUES = frozenset(artifact_type.value for artifact_type in ArtifactType)


class LoadError(ValueError):
    """Raised when a YAML artifact cannot be loaded."""


class UnsupportedArtifactTypeError(LoadError):
    """Raised when a YAML record declares an unsupported artifact type."""


class IssueRecord(BaseModel):
    """Repository-local issue record model used by storage and context flows."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    id: str
    type: Literal["issue"]
    title: str
    status: Literal["open", "blocked", "closed"]
    summary: str = Field(validation_alias=AliasChoices("summary", "description"))
    created_at: datetime
    updated_at: datetime
    authors: list[str] = Field(default_factory=list)
    severity: Literal["low", "medium", "high"] = "medium"
    labels: list[str] = Field(
        default_factory=list,
        validation_alias=AliasChoices("labels", "tags"),
    )
    related_artifacts: list[str] = Field(default_factory=list)
    related_sources: list[str] = Field(default_factory=list)
    parent_issue: str | None = None
    external_links: list[str] = Field(default_factory=list)
    scope: Literal["private", "public"] = "private"
    close_reason: str | None = None

    @field_validator("id")
    @classmethod
    def _validate_id(cls, value: str) -> str:
        return validate_artifact_id(value)

    @field_validator("title", "summary")
    @classmethod
    def _validate_required_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("text field must be non-empty")
        return normalized

    @field_validator("authors", "labels", "external_links")
    @classmethod
    def _normalize_text_list(cls, values: list[str]) -> list[str]:
        result: list[str] = []
        seen: set[str] = set()
        for value in values:
            normalized = value.strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            result.append(normalized)
        return result

    @field_validator("related_artifacts", "related_sources")
    @classmethod
    def _validate_related_ids(cls, values: list[str]) -> list[str]:
        return [validate_artifact_id(value) for value in values]

    @field_validator("parent_issue")
    @classmethod
    def _validate_parent_issue(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return validate_artifact_id(value)

    @field_validator("close_reason")
    @classmethod
    def _validate_close_reason(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("close_reason must be non-empty when set")
        return normalized

    @property
    def description(self) -> str:
        """Legacy accessor for context-pack callers."""
        return self.summary

    @property
    def tags(self) -> list[str]:
        """Legacy accessor for issue-local relevance callers."""
        return self.labels


class ReviewRecord(BaseModel):
    """Minimal review record model used by the storage loader."""

    model_config = ConfigDict(extra="forbid")

    id: str
    type: Literal["review"]
    title: str
    status: Literal["draft", "human_reviewed", "accepted"]
    created_at: datetime
    updated_at: datetime
    authors: list[str] = Field(default_factory=list)
    target: str
    summary: str
    findings: list[str] = Field(default_factory=list)
    decision: Literal[
        "approve",
        "request_changes",
        "reject",
        "informational",
        "accept_for_private_use",
        "accept_for_public_candidate",
        "changes_requested",
        "keep_draft",
        "refute_candidate",
        "mark_obsolete",
    ]
    reviewed_revision: str | None = None

    @field_validator("id", "target")
    @classmethod
    def _validate_id_reference(cls, value: str) -> str:
        return validate_artifact_id(value)

    @field_validator("reviewed_revision")
    @classmethod
    def _validate_reviewed_revision(cls, value: str | None) -> str | None:
        return validate_revision_hash(value)


LoadedModel = BaseArtifact | IssueRecord | ReviewRecord | AgentTask


@dataclass(frozen=True)
class LoadedRecord:
    """A loaded YAML record and its repository-relative source path."""

    source_path: Path
    record: LoadedModel
    kb_root_name: str | None = None
    kb_root_path: Path | None = None
    kb_root_readonly: bool = False
    kb_relative_path: Path | None = None

    @property
    def id(self) -> str:
        if isinstance(self.record, AgentTask):
            return self.record.task_id
        return self.record.id


def discover_yaml_paths(context: RepoContext) -> list[Path]:
    """Discover YAML files under the repository discovery roots."""
    paths: list[Path] = []
    seen: set[str] = set()
    for root_name in context.discovery_roots():
        root = context.resolve(root_name)
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file() or not is_yaml_path(path):
                continue
            relative_path = repo_relative_posix(context.repo_root, path)
            if relative_path in seen:
                continue
            seen.add(relative_path)
            paths.append(path)
    return sorted(paths, key=lambda path: repo_relative_posix(context.repo_root, path))


def load_artifacts(context: RepoContext) -> list[LoadedRecord]:
    """Load all discovered YAML records in deterministic order."""
    records = [load_yaml_file(context, path) for path in discover_yaml_paths(context)]
    return sorted(
        records,
        key=lambda record: (record.source_path.as_posix(), record.id),
    )


def load_yaml_file(context: RepoContext, path: Path) -> LoadedRecord:
    """Load one YAML file as a typed record."""
    relative_path = repo_relative_posix(context.repo_root, path)
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except YAMLError as exc:
        raise LoadError(f"{relative_path}: invalid YAML: {exc}") from exc

    if not isinstance(raw, dict):
        raise LoadError(f"{relative_path}: expected a YAML mapping at document root")

    record = _parse_record(relative_path, raw)
    kb_root = context.kb_root_for_path(relative_path)
    kb_relative_path = context.kb_relative_path(relative_path)
    return LoadedRecord(
        source_path=Path(relative_path),
        record=record,
        kb_root_name=kb_root.name if kb_root is not None else None,
        kb_root_path=Path(kb_root.path) if kb_root is not None else None,
        kb_root_readonly=kb_root.readonly if kb_root is not None else False,
        kb_relative_path=kb_relative_path,
    )


def _parse_record(relative_path: str, raw: dict[str, Any]) -> LoadedModel:
    record_type = raw.get("type")
    if not isinstance(record_type, str):
        if _looks_like_task_record(raw):
            try:
                return AgentTask.model_validate(raw)
            except ValidationError as exc:
                message = _format_validation_error(relative_path, exc)
                raise LoadError(message) from exc
        raise LoadError(f"{relative_path}: missing required field: type")

    try:
        if record_type == "issue":
            return IssueRecord.model_validate(raw)
        if record_type == "review":
            return ReviewRecord.model_validate(raw)
        if record_type in ARTIFACT_TYPE_VALUES:
            return BaseArtifact.model_validate(raw)
    except ValidationError as exc:
        message = _format_validation_error(relative_path, exc)
        raise LoadError(message) from exc

    raise UnsupportedArtifactTypeError(
        f"{relative_path}: unsupported artifact type: {record_type}"
    )


def _looks_like_task_record(raw: dict[str, Any]) -> bool:
    return "task_id" in raw and "worker_type" in raw


def _format_validation_error(relative_path: str, exc: ValidationError) -> str:
    missing = [
        ".".join(str(part) for part in error["loc"])
        for error in exc.errors()
        if error["type"] == "missing"
    ]
    if missing:
        return f"{relative_path}: missing required fields: {', '.join(missing)}"
    details = "; ".join(
        f"{'.'.join(str(part) for part in error['loc'])}: {error['msg']}"
        for error in exc.errors()
    )
    return f"{relative_path}: validation failed: {details}"
