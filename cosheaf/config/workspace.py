"""Workspace configuration loading."""

from __future__ import annotations

import tomllib
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from cosheaf.core.paths import normalize_repo_path

CONFIG_FILENAME = "cosheaf.toml"
KbVisibility = Literal["public", "private"]


def infer_kb_visibility(*values: str) -> KbVisibility:
    """Infer KB visibility from root name or path segments when not explicit."""
    for value in values:
        parts = value.lower().replace("\\", "/").split("/")
        if any("private" in part for part in parts):
            return "private"
    return "public"


class WorkspaceConfigError(ValueError):
    """Raised when `cosheaf.toml` cannot be loaded as workspace config."""


class KbRootConfig(BaseModel):
    """One configured knowledge-base root."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    path: str
    readonly: bool
    priority: int
    visibility: KbVisibility | None = None

    @field_validator("name")
    @classmethod
    def _validate_name(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("must not be empty")
        return normalized

    @field_validator("path")
    @classmethod
    def _validate_path(cls, value: str) -> str:
        normalized = normalize_repo_path(value.strip())
        if not normalized or normalized == ".":
            raise ValueError("must not be empty")
        if PureWindowsPath(value).is_absolute() or PurePosixPath(value).is_absolute():
            raise ValueError("must be repository-relative")
        if ".." in PurePosixPath(normalized).parts:
            raise ValueError("must not contain parent-directory traversal")
        return normalized

    @model_validator(mode="after")
    def _default_visibility(self) -> Self:
        if self.visibility is None:
            object.__setattr__(
                self,
                "visibility",
                infer_kb_visibility(self.name, self.path),
            )
        return self

    @property
    def effective_visibility(self) -> KbVisibility:
        return self.visibility or infer_kb_visibility(self.name, self.path)

    @property
    def is_private(self) -> bool:
        return self.effective_visibility == "private"

    @property
    def is_public(self) -> bool:
        return self.effective_visibility == "public"


class WorkspacePolicy(BaseModel):
    """Workspace-level public/private layering policy."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    private_can_depend_on_public: bool = True
    public_can_depend_on_private: bool = False
    accepted_requires_source: bool = True


class WorkspaceConfig(BaseModel):
    """Workspace-level configuration for one or more KB roots."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    kb: tuple[KbRootConfig, ...] = Field(default_factory=tuple)
    policy: WorkspacePolicy = Field(default_factory=WorkspacePolicy)
    configured: bool = False

    @classmethod
    def legacy(cls, repo_root: Path) -> WorkspaceConfig:
        """Return the no-config single-repository default."""
        return cls(
            name=repo_root.name,
            kb=(KbRootConfig(name="default", path="kb", readonly=False, priority=0),),
            policy=WorkspacePolicy(),
            configured=False,
        )

    @field_validator("name")
    @classmethod
    def _validate_name(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("must not be empty")
        return normalized

    @model_validator(mode="after")
    def _validate_kb_roots(self) -> Self:
        if not self.kb:
            raise ValueError("at least one kb root is required")

        names = [root.name for root in self.kb]
        duplicate_names = sorted({name for name in names if names.count(name) > 1})
        if duplicate_names:
            raise ValueError(
                f"duplicate kb root name(s): {', '.join(duplicate_names)}"
            )

        paths = [root.path for root in self.kb]
        duplicate_paths = sorted({path for path in paths if paths.count(path) > 1})
        if duplicate_paths:
            raise ValueError(
                f"duplicate kb root path(s): {', '.join(duplicate_paths)}"
            )

        return self

    @property
    def ordered_kb(self) -> tuple[KbRootConfig, ...]:
        """Return KB roots in deterministic priority order."""
        return tuple(
            sorted(self.kb, key=lambda root: (root.priority, root.name, root.path))
        )

    def root_by_name(self, name: str) -> KbRootConfig | None:
        """Return a configured KB root by name."""
        for root in self.kb:
            if root.name == name:
                return root
        return None


def load_workspace_config(repo_root: str | Path) -> WorkspaceConfig:
    """Load `cosheaf.toml`, or return legacy single-root config if absent."""
    root = Path(repo_root).resolve()
    config_path = root / CONFIG_FILENAME
    if not config_path.exists():
        return WorkspaceConfig.legacy(root)

    try:
        with config_path.open("rb") as handle:
            raw = tomllib.load(handle)
    except tomllib.TOMLDecodeError as exc:
        raise WorkspaceConfigError(f"{CONFIG_FILENAME}: invalid TOML: {exc}") from exc
    except OSError as exc:
        raise WorkspaceConfigError(
            f"{CONFIG_FILENAME}: cannot read file: {exc}"
        ) from exc

    return _workspace_config_from_toml(raw)


def _workspace_config_from_toml(raw: dict[str, Any]) -> WorkspaceConfig:
    workspace = raw.get("workspace")
    if not isinstance(workspace, dict):
        raise WorkspaceConfigError(f"{CONFIG_FILENAME}: missing [workspace] table")

    kb = raw.get("kb")
    if not isinstance(kb, list):
        raise WorkspaceConfigError(f"{CONFIG_FILENAME}: missing [[kb]] table")

    policy = raw.get("policy", {})
    if not isinstance(policy, dict):
        raise WorkspaceConfigError(f"{CONFIG_FILENAME}: [policy] must be a table")

    try:
        return WorkspaceConfig.model_validate(
            {
                "name": workspace.get("name"),
                "kb": kb,
                "policy": policy,
                "configured": True,
            }
        )
    except ValidationError as exc:
        raise WorkspaceConfigError(_format_config_errors(exc)) from exc


def _format_config_errors(exc: ValidationError) -> str:
    details = "; ".join(
        f"{'.'.join(str(part) for part in error['loc'])}: {error['msg']}"
        for error in exc.errors()
    )
    return f"{CONFIG_FILENAME}: validation failed: {details}"
