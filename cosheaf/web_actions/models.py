"""Stable WebAction request, result, plan, and audit DTOs."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from pathlib import PurePosixPath, PureWindowsPath
from typing import Literal

from pydantic import Field, field_validator, model_validator

from cosheaf.core.ids import validate_artifact_id
from cosheaf.core.paths import normalize_repo_path
from cosheaf.services.models import AgentAccessModel


class WebActionKind(StrEnum):
    """Allowed action identifiers for the B2 Web Workbench contract."""

    READ_WORKSPACE = "read.workspace"
    ISSUE_CREATE = "issue.create"
    ISSUE_UPDATE = "issue.update"
    ISSUE_CLOSE = "issue.close"
    ISSUE_PUBLISH_GITHUB = "issue.publish_github"
    ARTIFACT_CREATE = "artifact.create"
    ARTIFACT_UPDATE = "artifact.update"
    CONTEXT_BUILD = "context.build"
    VALIDATE_RUN = "validate.run"
    GATE_RUN = "gate.run"
    SOURCE_ATTACH = "source.attach"
    EVIDENCE_ATTACH = "evidence.attach"
    REVIEW_PACKET_CREATE = "review.packet_create"
    REVIEW_DECISION_CREATE = "review.decision_create"
    PROMOTION_PREVIEW = "promotion.preview"
    PROMOTION_CONFIRM = "promotion.confirm"
    FORGE_BRANCH_CREATE = "forge.branch_create"
    FORGE_COMMIT_CREATE = "forge.commit_create"
    FORGE_PUSH_CREATE = "forge.push_create"
    FORGE_PR_CREATE = "forge.pr_create"
    AUDIT_READ = "audit.read"


class WebActionMode(StrEnum):
    """Runtime mode for web action DTOs."""

    STATIC = "static"
    LOCAL = "local"
    HOSTED = "hosted"


class WebActionError(AgentAccessModel):
    """Machine-readable web action error."""

    code: str
    message: str
    remediation: str
    blocking: bool
    related_path: str | None = None
    related_artifact: str | None = None
    details: dict[str, str] = Field(default_factory=dict)

    @field_validator("code", "message", "remediation")
    @classmethod
    def _text(cls, value: str) -> str:
        return _non_empty(value)

    @field_validator("related_path")
    @classmethod
    def _path(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _repo_local_path(value)

    @field_validator("related_artifact")
    @classmethod
    def _artifact(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return validate_artifact_id(value.strip())

    @field_validator("details")
    @classmethod
    def _details(cls, values: dict[str, str]) -> dict[str, str]:
        return {_non_empty(key): _non_empty(value) for key, value in values.items()}


class RepoWritePlan(AgentAccessModel):
    """Planned repository file writes."""

    planned_files: list[str] = Field(default_factory=list)
    written_files: list[str] = Field(default_factory=list)
    repo_writes_performed: bool = False

    @field_validator("planned_files", "written_files")
    @classmethod
    def _paths(cls, values: list[str]) -> list[str]:
        return [_repo_local_path(value) for value in values]


class GitWritePlan(AgentAccessModel):
    """Planned local git writes."""

    base: str = ""
    head: str = ""
    branch: str = ""
    planned_git_commands: list[str] = Field(default_factory=list)
    git_writes_performed: bool = False

    @field_validator("base", "head", "branch")
    @classmethod
    def _optional_text(cls, value: str) -> str:
        return value.strip()

    @field_validator("planned_git_commands")
    @classmethod
    def _commands(cls, values: list[str]) -> list[str]:
        return [_non_empty(value) for value in values]


class GitHubWritePlan(AgentAccessModel):
    """Planned GitHub issue or pull-request write."""

    github_action: Literal["issue", "pull_request"]
    title: str
    body: str
    labels: list[str] = Field(default_factory=list)
    result_url: str | None = None
    github_writes_performed: bool = False
    network_calls_performed: bool = False

    @field_validator("title", "body")
    @classmethod
    def _text(cls, value: str) -> str:
        return _non_empty(value)

    @field_validator("labels")
    @classmethod
    def _labels(cls, values: list[str]) -> list[str]:
        return _unique_text(values)

    @field_validator("result_url")
    @classmethod
    def _url(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _non_empty(value)


class ReviewDecisionPlan(AgentAccessModel):
    """Planned human review decision write."""

    target_artifact: str
    reviewer_identity: str
    decision: Literal[
        "accept_for_private_use",
        "accept_for_public_candidate",
        "changes_requested",
        "keep_draft",
        "refute_candidate",
        "mark_obsolete",
    ]
    notes_required: bool = True
    human_confirmation_required: bool = True
    human_review_recorded: bool = False

    @field_validator("target_artifact")
    @classmethod
    def _artifact(cls, value: str) -> str:
        return validate_artifact_id(value.strip())

    @field_validator("reviewer_identity")
    @classmethod
    def _reviewer(cls, value: str) -> str:
        return _non_empty(value)


class PromotionPlan(AgentAccessModel):
    """Planned artifact promotion."""

    artifact_id: str
    target_state: Literal["accepted", "refuted", "obsolete"]
    readiness_summary: str
    required_confirmation: Literal[
        "PROMOTE TO ACCEPTED",
        "MARK REFUTED",
        "MARK OBSOLETE",
    ]
    validation_required: bool = True
    gate_required: bool = True
    human_review_required: bool = True
    promotion_performed: bool = False

    @field_validator("artifact_id")
    @classmethod
    def _artifact_id(cls, value: str) -> str:
        return validate_artifact_id(value.strip())

    @field_validator("readiness_summary")
    @classmethod
    def _summary(cls, value: str) -> str:
        return _non_empty(value)


class WebActionPreviewRequest(AgentAccessModel):
    """Request to preview a web action without side effects."""

    action: WebActionKind
    mode: WebActionMode
    actor: str | None = None
    parameters: dict[str, str] = Field(default_factory=dict)

    @field_validator("actor")
    @classmethod
    def _actor(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _non_empty(value)

    @field_validator("parameters")
    @classmethod
    def _parameters(cls, values: dict[str, str]) -> dict[str, str]:
        return {_non_empty(key): _non_empty(value) for key, value in values.items()}


class WebActionConfirmRequest(AgentAccessModel):
    """Request to confirm a previously previewed web action."""

    action: WebActionKind
    mode: WebActionMode
    preview_plan_hash: str
    confirm: Literal[True] = True
    actor: str | None = None
    parameters: dict[str, str] = Field(default_factory=dict)

    @field_validator("preview_plan_hash")
    @classmethod
    def _hash(cls, value: str) -> str:
        return _non_empty(value)

    @field_validator("actor")
    @classmethod
    def _actor(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _non_empty(value)

    @field_validator("parameters")
    @classmethod
    def _parameters(cls, values: dict[str, str]) -> dict[str, str]:
        return {_non_empty(key): _non_empty(value) for key, value in values.items()}


class WebActionResult(AgentAccessModel):
    """Result for a previewed or confirmed web action."""

    action: WebActionKind
    mode: WebActionMode
    preview_only: bool
    confirm_required: bool
    confirmed: bool
    performed: bool
    repo_writes_performed: bool = False
    git_writes_performed: bool = False
    github_writes_performed: bool = False
    network_calls_performed: bool = False
    planned_files: list[str] = Field(default_factory=list)
    written_files: list[str] = Field(default_factory=list)
    validation_summary: str | None = None
    gate_summary: str | None = None
    authority_warnings: list[str] = Field(default_factory=list)
    audit_path: str | None = None
    errors: list[WebActionError] = Field(default_factory=list)
    repo_write_plan: RepoWritePlan | None = None
    git_write_plan: GitWritePlan | None = None
    github_write_plan: GitHubWritePlan | None = None
    review_decision_plan: ReviewDecisionPlan | None = None
    promotion_plan: PromotionPlan | None = None

    @field_validator("planned_files", "written_files")
    @classmethod
    def _paths(cls, values: list[str]) -> list[str]:
        return [_repo_local_path(value) for value in values]

    @field_validator("validation_summary", "gate_summary")
    @classmethod
    def _optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _non_empty(value)

    @field_validator("authority_warnings")
    @classmethod
    def _warnings(cls, values: list[str]) -> list[str]:
        return _unique_text(values)

    @field_validator("audit_path")
    @classmethod
    def _audit_path(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _repo_local_path(value)

    @model_validator(mode="after")
    def _side_effect_flags_are_coherent(self) -> WebActionResult:
        if self.preview_only:
            if (
                self.confirmed
                or self.performed
                or self.repo_writes_performed
                or self.git_writes_performed
                or self.github_writes_performed
                or self.network_calls_performed
                or self.written_files
            ):
                raise ValueError(
                    "preview-only web action results cannot perform writes"
                )
        if self.performed and self.confirm_required and not self.confirmed:
            raise ValueError("confirmed web actions require confirmed=true")
        if self.written_files and not self.repo_writes_performed:
            raise ValueError("written_files require repo_writes_performed=true")
        return self


class WebActionAuditEntry(AgentAccessModel):
    """One machine-readable audit record for a web action."""

    timestamp: datetime
    actor: str
    action: WebActionKind
    mode: WebActionMode
    repo_root: str
    branch: str | None = None
    base: str | None = None
    head: str | None = None
    commit_hash: str | None = None
    preview_only: bool
    confirm_required: bool = False
    confirmed: bool
    explicit_confirm: bool = False
    performed: bool
    repo_writes_performed: bool = False
    git_writes_performed: bool = False
    github_writes_performed: bool = False
    network_calls_performed: bool = False
    planned_files: list[str] = Field(default_factory=list)
    written_files: list[str] = Field(default_factory=list)
    validation_summary: str | None = None
    gate_summary: str | None = None
    github_urls: list[str] = Field(default_factory=list)
    credential_provider: str | None = None
    result_status: str | None = None
    authority_warnings: list[str] = Field(default_factory=list)
    operator_notes: str | None = None
    error_code: str | None = None
    errors: list[WebActionError] = Field(default_factory=list)

    @field_validator("actor", "repo_root")
    @classmethod
    def _text(cls, value: str) -> str:
        return _non_empty(value)

    @field_validator("branch", "base", "head", "commit_hash")
    @classmethod
    def _optional_ref(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _non_empty(value)

    @field_validator("planned_files", "written_files")
    @classmethod
    def _paths(cls, values: list[str]) -> list[str]:
        return [_repo_local_path(value) for value in values]

    @field_validator(
        "validation_summary",
        "gate_summary",
        "credential_provider",
        "result_status",
        "operator_notes",
        "error_code",
    )
    @classmethod
    def _optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _non_empty(value)

    @field_validator("github_urls")
    @classmethod
    def _github_urls(cls, values: list[str]) -> list[str]:
        return _unique_text(values)

    @field_validator("authority_warnings")
    @classmethod
    def _warnings(cls, values: list[str]) -> list[str]:
        return _unique_text(values)


class WebActionDtoBundle(AgentAccessModel):
    """Schema bundle containing all public WebAction DTOs."""

    preview_request: WebActionPreviewRequest
    confirm_request: WebActionConfirmRequest
    result: WebActionResult
    audit_entry: WebActionAuditEntry


def _non_empty(value: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError("text field must be non-empty")
    return normalized


def _unique_text(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        normalized = _non_empty(value)
        if normalized in seen:
            continue
        seen.add(normalized)
        result.append(normalized)
    return result


def _repo_local_path(value: str) -> str:
    normalized = normalize_repo_path(value)
    is_absolute = PureWindowsPath(value).is_absolute() or PurePosixPath(
        value
    ).is_absolute()
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


__all__ = [
    "GitHubWritePlan",
    "GitWritePlan",
    "PromotionPlan",
    "RepoWritePlan",
    "ReviewDecisionPlan",
    "WebActionAuditEntry",
    "WebActionConfirmRequest",
    "WebActionDtoBundle",
    "WebActionError",
    "WebActionKind",
    "WebActionMode",
    "WebActionPreviewRequest",
    "WebActionResult",
]
