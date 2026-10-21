"""Role-specific hosted worker bridge over the provider gateway.

Hosted worker output is untrusted. This module validates provider output into
WorkerBundle v2 or typed review-only sub-results and never writes accepted
knowledge, marks human review, promotes artifacts, or performs real network
transport on its own.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from datetime import UTC, datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from cosheaf.agent.model_provider import ProviderName
from cosheaf.agent.providers import (
    OpenAICompatibleProvider,
    ProviderConfig,
    ProviderError,
    ProviderGatewayRequest,
    ProviderMode,
)
from cosheaf.agent.roles import RoleName, get_role_contract
from cosheaf.agent.task import WorkerType
from cosheaf.agent.worker_bundle_v2 import WorkerBundleV2
from cosheaf.core.ids import validate_artifact_id
from cosheaf.services.model_calls import ModelCallService
from cosheaf.services.models import (
    ErrorResult,
    ModelCallResult,
    ProviderConsent,
    ProviderRunRecord,
)
from cosheaf.storage.repo import RepoContext


class HostedWorkerStatus(StrEnum):
    """Normalized hosted worker execution status."""

    COMPLETED = "completed"
    REJECTED = "rejected"
    FAILED = "failed"
    SKIPPED = "skipped"


class HostedWorkerInput(BaseModel):
    """Input for one role-specific hosted worker call."""

    model_config = ConfigDict(extra="forbid", frozen=True, use_enum_values=False)

    issue_id: str
    role: RoleName
    prompt: str
    context_artifact_ids: list[str] = Field(default_factory=list)
    root_scopes: list[str] = Field(default_factory=list)
    upstream_bundle_paths: list[str] = Field(default_factory=list)
    upstream_proposed_paths: list[str] = Field(default_factory=list)
    consent: ProviderConsent

    @field_validator("issue_id")
    @classmethod
    def _issue_id(cls, value: str) -> str:
        return validate_artifact_id(value.strip())

    @field_validator("role", mode="before")
    @classmethod
    def _role(cls, value: RoleName | str) -> RoleName:
        return value if isinstance(value, RoleName) else RoleName(value)

    @field_validator("prompt")
    @classmethod
    def _prompt(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("hosted worker prompt must be non-empty")
        return normalized

    @field_validator("context_artifact_ids")
    @classmethod
    def _artifact_ids(cls, values: list[str]) -> list[str]:
        return _dedupe(validate_artifact_id(value.strip()) for value in values)

    @field_validator("root_scopes")
    @classmethod
    def _root_scopes(cls, values: list[str]) -> list[str]:
        return _dedupe(_non_empty(value) for value in values)

    @field_validator("upstream_bundle_paths", "upstream_proposed_paths")
    @classmethod
    def _upstream_paths(cls, values: list[str]) -> list[str]:
        return _dedupe(_non_empty(value) for value in values)


class HostedWorkerTypedResult(BaseModel):
    """Review-only typed sub-result for non-bundle hosted roles."""

    model_config = ConfigDict(extra="forbid", frozen=True, use_enum_values=False)

    issue_id: str
    role: RoleName
    summary: str
    selected_artifacts: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    risk_flags: list[str] = Field(default_factory=list)
    next_steps: list[str] = Field(default_factory=list)

    @field_validator("issue_id")
    @classmethod
    def _issue_id(cls, value: str) -> str:
        return validate_artifact_id(value.strip())

    @field_validator("summary")
    @classmethod
    def _summary(cls, value: str) -> str:
        return _non_empty(value)

    @field_validator("selected_artifacts")
    @classmethod
    def _selected_artifacts(cls, values: list[str]) -> list[str]:
        return _dedupe(validate_artifact_id(value.strip()) for value in values)

    @field_validator("notes", "risk_flags", "next_steps")
    @classmethod
    def _text_list(cls, values: list[str]) -> list[str]:
        return _dedupe(_non_empty(value) for value in values)


class HostedWorkerOutput(BaseModel):
    """Result of one hosted worker run after local validation and policy checks."""

    model_config = ConfigDict(extra="forbid", frozen=True, use_enum_values=False)

    issue_id: str
    role: RoleName
    status: HostedWorkerStatus
    provider_run: ProviderRunRecord | None = None
    provider_log_path: str | None = None
    bundle: WorkerBundleV2 | None = None
    typed_result: HostedWorkerTypedResult | None = None
    error: ErrorResult | None = None
    accepted_write_performed: Literal[False] = False

    @field_validator("issue_id")
    @classmethod
    def _issue_id(cls, value: str) -> str:
        return validate_artifact_id(value.strip())


class HostedWorkerService:
    """Run role-specific hosted workers through the provider gateway boundary."""

    def __init__(self, context: RepoContext) -> None:
        self.context = context
        self.model_calls = ModelCallService(context)

    def run(
        self,
        worker_input: HostedWorkerInput,
        *,
        config: ProviderConfig | None = None,
        provider: OpenAICompatibleProvider | None = None,
    ) -> HostedWorkerOutput:
        """Run one hosted worker and validate the result for review-only use."""
        role = _canonical_hosted_role(worker_input.role)
        contract = get_role_contract(role)
        provider_config = config or ProviderConfig(
            provider=ProviderName.FAKE,
            mode=ProviderMode.FAKE,
            model="fake-deterministic",
            enabled=True,
        )
        output_kind: Literal["worker_bundle", "proposal"] = (
            "worker_bundle"
            if contract.provider_capability_requirements.requires_worker_bundle
            else "proposal"
        )
        try:
            request = ProviderGatewayRequest(
                provider=provider_config.provider,
                model=provider_config.model,
                worker_role=_provider_worker_type(role),
                prompt=_worker_prompt(worker_input, role),
                consent=worker_input.consent,
                context_artifact_ids=worker_input.context_artifact_ids,
                root_scopes=worker_input.root_scopes,
                output_kind=output_kind,
                expected_output_paths=_expected_output_paths(worker_input, role),
                tool_policy=contract.tool_policy.tool_policy,
                network_policy=contract.tool_policy.network_policy,
            )
        except ValueError as exc:
            return _rejected(
                worker_input,
                "provider_request_validation_failed",
                str(exc),
                "Adjust hosted worker context policy, consent, or output paths.",
            )

        result = self.model_calls.call(
            request,
            config=provider_config,
            provider=provider,
        )
        if isinstance(result, ProviderError):
            return _provider_error_output(worker_input, result)
        if provider_config.mode is ProviderMode.FAKE:
            return self._fake_output(worker_input, role, result)
        return self._provider_output(worker_input, role, result)

    def _fake_output(
        self,
        worker_input: HostedWorkerInput,
        role: RoleName,
        result: ModelCallResult,
    ) -> HostedWorkerOutput:
        if _bundle_worker_type(role) is None:
            return HostedWorkerOutput(
                issue_id=worker_input.issue_id,
                role=role,
                status=HostedWorkerStatus.COMPLETED,
                provider_run=result.provider_run,
                provider_log_path=result.provider_run.log_path,
                typed_result=_fake_typed_result(worker_input, role),
                accepted_write_performed=False,
            )
        bundle = _fake_bundle(worker_input, role)
        policy_error = _policy_error_for_bundle(role, bundle)
        if policy_error is not None:
            return _rejected_from_result(worker_input, role, result, policy_error)
        return HostedWorkerOutput(
            issue_id=worker_input.issue_id,
            role=role,
            status=HostedWorkerStatus.COMPLETED,
            provider_run=result.provider_run,
            provider_log_path=result.provider_run.log_path,
            bundle=bundle,
            accepted_write_performed=False,
        )

    def _provider_output(
        self,
        worker_input: HostedWorkerInput,
        role: RoleName,
        result: ModelCallResult,
    ) -> HostedWorkerOutput:
        if _bundle_worker_type(role) is None:
            typed = _parse_typed_result(worker_input, role, result.content)
            if isinstance(typed, ErrorResult):
                return _rejected_from_result(worker_input, role, result, typed)
            return HostedWorkerOutput(
                issue_id=worker_input.issue_id,
                role=role,
                status=HostedWorkerStatus.COMPLETED,
                provider_run=result.provider_run,
                provider_log_path=result.provider_run.log_path,
                typed_result=typed,
                accepted_write_performed=False,
            )
        bundle = _parse_bundle(result.content)
        if isinstance(bundle, ErrorResult):
            return _rejected_from_result(worker_input, role, result, bundle)
        policy_error = _policy_error_for_bundle(role, bundle)
        if policy_error is not None:
            return _rejected_from_result(worker_input, role, result, policy_error)
        return HostedWorkerOutput(
            issue_id=worker_input.issue_id,
            role=role,
            status=HostedWorkerStatus.COMPLETED,
            provider_run=result.provider_run,
            provider_log_path=result.provider_run.log_path,
            bundle=bundle,
            accepted_write_performed=False,
        )


def _canonical_hosted_role(role: RoleName) -> RoleName:
    if role is RoleName.LIBRARIAN or role is RoleName.COLLECTOR:
        return RoleName.LIBRARIAN_SUMMARIZER
    return role


def _provider_worker_type(role: RoleName) -> WorkerType:
    bundle_worker = _bundle_worker_type(role)
    if bundle_worker is not None:
        return bundle_worker
    if role is RoleName.EXPLORER:
        return WorkerType.CONSTRUCTION_SEARCHER
    return WorkerType.LITERATURE_SCOUT


def _bundle_worker_type(role: RoleName) -> WorkerType | None:
    return {
        RoleName.REASONER: WorkerType.REASONER,
        RoleName.VERIFIER: WorkerType.VERIFIER,
        RoleName.COUNTEREXAMPLER: WorkerType.COUNTEREXAMPLER,
        RoleName.FORMALIZER: WorkerType.FORMALIZER,
    }.get(role)


def _worker_prompt(worker_input: HostedWorkerInput, role: RoleName) -> str:
    contract = get_role_contract(role)
    upstream_lines: list[str] = []
    if worker_input.upstream_bundle_paths:
        upstream_lines.append(
            "Upstream worker bundles: "
            + ", ".join(worker_input.upstream_bundle_paths)
        )
    if worker_input.upstream_proposed_paths:
        upstream_lines.append(
            "Upstream proposed draft paths: "
            + ", ".join(worker_input.upstream_proposed_paths)
        )
    lines = [
        contract.system_prompt,
        "",
        "Required output fields: "
        + ", ".join(contract.required_output_schema.required_fields),
        "Optional output fields: "
        + ", ".join(contract.required_output_schema.optional_fields),
        "Forbidden authority: " + ", ".join(contract.forbidden_actions),
        "",
        "Return review-only output. Do not write accepted knowledge, mark "
        "human review, promote artifacts, or claim unchecked verification.",
    ]
    if upstream_lines:
        lines.extend(["", *upstream_lines])
    lines.extend(["", worker_input.prompt])
    return "\n".join(lines)


def _expected_output_paths(
    worker_input: HostedWorkerInput,
    role: RoleName,
) -> list[str]:
    if _bundle_worker_type(role) is None:
        return []
    return [
        (
            "kb/draft/hosted-workers/"
            f"{worker_input.issue_id}.{role.value}.proposal.yaml"
        )
    ]


def _fake_bundle(worker_input: HostedWorkerInput, role: RoleName) -> WorkerBundleV2:
    worker_type = _bundle_worker_type(role)
    if worker_type is None:
        raise ValueError(f"role {role.value!r} does not use WorkerBundle v2")
    raw = {
        "bundle_id": f"bundle.{worker_input.issue_id}.{role.value}.0001",
        "task_id": f"task.{worker_input.issue_id}.{role.value}",
        "worker_role": worker_type.value,
        "created_at": _fixed_timestamp(),
        "summary": f"{role.value} hosted worker produced review-only output.",
        "used_artifacts": worker_input.context_artifact_ids,
        "used_sources": [],
        "claims": ["This is review context, not accepted knowledge."],
        "proposed_artifacts": [
            {
                "path": _expected_output_paths(worker_input, role)[0],
                "summary": "Draft-only hosted worker proposal.",
            }
        ],
        "assumptions": ["Hosted worker output is review-only."],
        "uncertainty": ["No human review or semantic alignment review was run."],
        "verification_requests": ["Run validate and gate before review."],
        "failed_attempts": [],
        "counterexamples": [],
        "failures_or_counterexamples": ["No external verifier was run."],
        "dependency_questions": [],
        "risk_flags": ["needs_human_review"],
        "next_steps": ["Request explicit human review."],
        "confidence": "low",
    }
    return WorkerBundleV2.model_validate(raw)


def _fake_typed_result(
    worker_input: HostedWorkerInput,
    role: RoleName,
) -> HostedWorkerTypedResult:
    return HostedWorkerTypedResult(
        issue_id=worker_input.issue_id,
        role=role,
        summary=f"{role.value} hosted worker produced typed review-only output.",
        selected_artifacts=worker_input.context_artifact_ids,
        notes=["No accepted knowledge was written."],
        risk_flags=["needs_human_review"],
        next_steps=["Use ordinary validate, gate, and review workflow."],
    )


def _parse_bundle(content: str) -> WorkerBundleV2 | ErrorResult:
    try:
        raw = json.loads(content)
    except json.JSONDecodeError as exc:
        return _error(
            "provider_output_validation_failed",
            f"provider output is not valid JSON: {exc.msg}",
            "Ask the provider for a valid WorkerBundle JSON object.",
        )
    try:
        return WorkerBundleV2.model_validate(raw)
    except (TypeError, ValueError, ValidationError) as exc:
        return _error(
            "provider_output_validation_failed",
            f"provider WorkerBundle output is invalid: {exc}",
            "Ask the provider for a valid WorkerBundle v2 payload.",
        )


def _parse_typed_result(
    worker_input: HostedWorkerInput,
    role: RoleName,
    content: str,
) -> HostedWorkerTypedResult | ErrorResult:
    try:
        raw = json.loads(content)
    except json.JSONDecodeError as exc:
        return _error(
            "provider_output_validation_failed",
            f"provider typed sub-result is not valid JSON: {exc.msg}",
            "Ask the provider for a valid hosted worker typed sub-result.",
        )
    if not isinstance(raw, dict):
        return _error(
            "provider_output_validation_failed",
            "provider typed sub-result must be a JSON object",
            "Ask the provider for a valid hosted worker typed sub-result.",
        )
    raw.setdefault("issue_id", worker_input.issue_id)
    raw.setdefault("role", role.value)
    try:
        return HostedWorkerTypedResult.model_validate(raw)
    except (TypeError, ValueError, ValidationError) as exc:
        return _error(
            "provider_output_validation_failed",
            f"provider typed sub-result is invalid: {exc}",
            "Ask the provider for a valid hosted worker typed sub-result.",
        )


def _policy_error_for_bundle(
    role: RoleName,
    bundle: WorkerBundleV2,
) -> ErrorResult | None:
    text = bundle.to_json().lower()
    if role is RoleName.VERIFIER and (
        "human_reviewed" in text
        or "review.state accepted" in text
        or "review_state accepted" in text
        or "status accepted" in text
        or "mark accepted" in text
        or "marked accepted" in text
    ):
        return _error(
            "hosted_worker_policy_violation",
            "verifier output must not mark human_reviewed or accepted state",
            "Keep verifier output as evidence requests or review-only notes.",
        )
    if role is RoleName.REASONER and (
        "conjecture is now a theorem" in text
        or ("conjecture" in text and "theorem" in text)
    ):
        return _error(
            "hosted_worker_policy_violation",
            "reasoner output must not claim a conjecture is now a theorem",
            "Keep reasoner output as draft reasoning for human review.",
        )
    if role is RoleName.FORMALIZER and (
        "semantic alignment" in text or "informal/formal equivalence" in text
    ):
        return _error(
            "hosted_worker_policy_violation",
            "formalizer output must not claim semantic alignment was checked",
            "Record formal links as metadata unless a checker and alignment "
            "review exist.",
        )
    return None


def _provider_error_output(
    worker_input: HostedWorkerInput,
    provider_error: ProviderError,
) -> HostedWorkerOutput:
    return HostedWorkerOutput(
        issue_id=worker_input.issue_id,
        role=_canonical_hosted_role(worker_input.role),
        status=HostedWorkerStatus.REJECTED,
        provider_log_path=provider_error.details.get("log_path"),
        error=_error(
            provider_error.code,
            provider_error.message,
            provider_error.remediation,
            details=provider_error.details,
        ),
        accepted_write_performed=False,
    )


def _rejected_from_result(
    worker_input: HostedWorkerInput,
    role: RoleName,
    result: ModelCallResult,
    error: ErrorResult,
) -> HostedWorkerOutput:
    return HostedWorkerOutput(
        issue_id=worker_input.issue_id,
        role=role,
        status=HostedWorkerStatus.REJECTED,
        provider_run=result.provider_run,
        provider_log_path=result.provider_run.log_path,
        error=error,
        accepted_write_performed=False,
    )


def _rejected(
    worker_input: HostedWorkerInput,
    code: str,
    message: str,
    remediation: str,
) -> HostedWorkerOutput:
    return HostedWorkerOutput(
        issue_id=worker_input.issue_id,
        role=_canonical_hosted_role(worker_input.role),
        status=HostedWorkerStatus.REJECTED,
        error=_error(code, message, remediation),
        accepted_write_performed=False,
    )


def _error(
    code: str,
    message: str,
    remediation: str,
    *,
    details: dict[str, str] | None = None,
) -> ErrorResult:
    return ErrorResult(
        code=code,
        message=message,
        remediation=remediation,
        blocking=True,
        details=details or {},
    )


def _fixed_timestamp() -> str:
    return datetime(2026, 6, 10, tzinfo=UTC).isoformat().replace("+00:00", "Z")


def _non_empty(value: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError("text field must be non-empty")
    return normalized


def _dedupe(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


__all__ = [
    "HostedWorkerInput",
    "HostedWorkerOutput",
    "HostedWorkerService",
    "HostedWorkerStatus",
    "HostedWorkerTypedResult",
]
