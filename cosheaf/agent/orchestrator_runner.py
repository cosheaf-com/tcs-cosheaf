"""Local and hosted-worker orchestrator runners.

This module wires the deterministic planner to the existing local command
runner and, when explicitly configured, to the hosted-worker service boundary.
It does not perform built-in real network transport, merge worker output,
request human review, or promote accepted knowledge.
"""

from __future__ import annotations

import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from cosheaf.agent.dry_run_workers import dry_run_worker_command
from cosheaf.agent.hosted_workers import (
    HostedWorkerInput,
    HostedWorkerOutput,
    HostedWorkerService,
    HostedWorkerStatus,
)
from cosheaf.agent.local_runner import (
    LocalWorkerRunConfig,
    LocalWorkerRunError,
    LocalWorkerRunner,
)
from cosheaf.agent.model_provider import ProviderName
from cosheaf.agent.orchestrator_planner import (
    OrchestratorPlannerError,
    plan_for_issue,
)
from cosheaf.agent.orchestrator_state import (
    OrchestratorRun,
    OrchestratorState,
    StopCondition,
    TaskNode,
    WorkerCall,
)
from cosheaf.agent.orchestrator_stub import OrchestratorStub, TaskHarnessError
from cosheaf.agent.providers import (
    OpenAICompatibleProvider,
    ProviderConfig,
    ProviderMode,
)
from cosheaf.agent.roles import RoleName
from cosheaf.agent.run_logging import write_orchestrator_run_log
from cosheaf.agent.task import AgentTask, WorkerType
from cosheaf.agent.worker_bundle_v2 import (
    WorkerBundleV2Error,
    reduce_worker_bundle_v2,
)
from cosheaf.core.ids import validate_artifact_id
from cosheaf.core.paths import repo_relative_posix
from cosheaf.services.context_policy import ContextSendPolicyService
from cosheaf.services.models import (
    ContextBuildRequest,
    ContextPolicyMode,
    ErrorResult,
    ProviderConsent,
    ProviderContextPreview,
)
from cosheaf.storage.repo import RepoContext
from cosheaf.storage.writer import write_yaml_deterministic


class OrchestratorLocalRunError(ValueError):
    """Raised for expected local orchestrator run failures."""


class OrchestratorHostedRunError(ValueError):
    """Raised for expected hosted-worker orchestrator failures."""

    def __init__(self, error: ErrorResult) -> None:
        super().__init__(error.message)
        self.error = error


@dataclass(frozen=True)
class OrchestratorLocalRunConfig:
    """Configuration for one deterministic local-only orchestrator run."""

    issue_id: str
    timeout_seconds: int = 60
    worker_command: Sequence[str] | None = None
    proposal_path: str | Path | None = None
    run_id: str | None = None
    now: datetime | None = None

    def __post_init__(self) -> None:
        validate_artifact_id(self.issue_id)
        if self.timeout_seconds <= 0:
            raise OrchestratorLocalRunError("timeout_seconds must be positive")
        if self.run_id is not None:
            validate_artifact_id(self.run_id)
        if self.worker_command is not None:
            if isinstance(self.worker_command, str) or not self.worker_command:
                raise OrchestratorLocalRunError(
                    "worker_command must be an explicit argv list"
                )
            for argument in self.worker_command:
                if not isinstance(argument, str) or argument == "":
                    raise OrchestratorLocalRunError(
                        "worker_command arguments must be non-empty strings"
                    )


@dataclass(frozen=True)
class OrchestratorLocalRunResult:
    """Filesystem outputs and final state for one orchestrator run."""

    run: OrchestratorRun
    run_root: Path
    record_path: Path
    log_path: Path


@dataclass(frozen=True)
class OrchestratorHostedRunConfig:
    """Configuration for an explicit hosted-worker orchestrator run."""

    issue_id: str
    provider: ProviderName | str = ProviderName.FAKE
    confirm_send: bool = False
    include_private: bool = False
    policy_mode: ContextPolicyMode = ContextPolicyMode.PUBLIC
    allow_private_context: bool = False
    max_cards: int = 20
    run_id: str | None = None
    now: datetime | None = None

    def __post_init__(self) -> None:
        validate_artifact_id(self.issue_id)
        _resolve_hosted_provider(self.provider)
        if self.run_id is not None:
            validate_artifact_id(self.run_id)
        if self.max_cards <= 0:
            raise OrchestratorHostedRunError(
                _error(
                    "provider_request_validation_failed",
                    "max_cards must be positive",
                    "Use a positive --max-cards value.",
                )
            )


@dataclass(frozen=True)
class OrchestratorHostedRunResult:
    """Filesystem outputs and provider metadata for one hosted-worker run."""

    run: OrchestratorRun
    run_root: Path
    record_path: Path
    log_path: Path
    context_preview: ProviderContextPreview
    provider: ProviderName
    provider_mode: ProviderMode
    provider_run_record_paths: tuple[Path, ...]
    accepted_write_performed: bool = False


class OrchestratorLocalRunner:
    """Execute deterministic local worker tasks for a planned issue."""

    def __init__(self, context: RepoContext) -> None:
        self.context = context
        self.task_harness = OrchestratorStub(context)
        self.local_runner = LocalWorkerRunner(context)

    def run_issue(
        self,
        config: OrchestratorLocalRunConfig,
    ) -> OrchestratorLocalRunResult:
        """Plan and run local worker commands for one issue."""
        now = _normalize_timestamp(config.now or _utc_now())
        run_id = config.run_id or validate_artifact_id(f"run.{config.issue_id}")
        try:
            plan = plan_for_issue(self.context, config.issue_id)
        except OrchestratorPlannerError as exc:
            raise OrchestratorLocalRunError(str(exc)) from exc

        run_root = self.context.resolve(
            Path(".cosheaf")
            / "orchestrator"
            / config.issue_id
            / "runs"
            / run_id
        )
        record_path = run_root / "run.yaml"
        log_path = run_root / "run_log.json"
        if run_root.exists():
            raise OrchestratorLocalRunError(
                f"orchestrator run already exists: {run_id}"
            )
        run_root.mkdir(parents=True, exist_ok=False)

        run = (
            OrchestratorRun.create(
                run_id=run_id,
                issue_id=config.issue_id,
                now=now,
            )
            .transition(OrchestratorState.PLANNED, now=now, plan=plan)
            .transition(OrchestratorState.RUNNING, now=now)
        )

        worker_calls: list[WorkerCall] = []
        reducer_results = []
        stop_conditions: list[StopCondition] = []

        for index, node in enumerate(plan.task_dag.nodes, start=1):
            task = self._ensure_task_for_node(config.issue_id, node, now=now)
            bundle_path = self._bundle_path(config, run_id, node)
            command = (
                list(config.worker_command)
                if config.worker_command is not None
                else _default_worker_command(
                    bundle_path=bundle_path,
                    task=task,
                    node=node,
                    proposal_path=self._proposal_path(config, run_id, node),
                    created_at=now,
                )
            )
            try:
                local_result = self.local_runner.run_task(
                    task.task_id,
                    LocalWorkerRunConfig(
                        command=command,
                        timeout_seconds=config.timeout_seconds,
                        run_id=validate_artifact_id(f"{run_id}.{node.node_id}"),
                        started_at=now,
                    ),
                )
            except LocalWorkerRunError as exc:
                raise OrchestratorLocalRunError(str(exc)) from exc
            worker_calls.append(
                WorkerCall(
                    call_id=validate_artifact_id(f"call.{node.node_id}"),
                    task_node_id=node.node_id,
                    worker_type=node.worker_type,
                    status=local_result.status,
                    command=command,
                    cwd=".",
                    started_at=now,
                    completed_at=now,
                    exit_code=local_result.returncode,
                    stdout_path=repo_relative_posix(
                        self.context.repo_root,
                        local_result.stdout_path,
                    ),
                    stderr_path=repo_relative_posix(
                        self.context.repo_root,
                        local_result.stderr_path,
                    ),
                    bundle_path=repo_relative_posix(
                        self.context.repo_root,
                        self.context.resolve(bundle_path),
                    ),
                    notes=f"local-only worker call {index}",
                )
            )
            if local_result.status != "completed":
                stop_conditions.append(
                    StopCondition(
                        reason="worker_failed",
                        description=(
                            f"{node.node_id} finished with status "
                            f"{local_result.status}"
                        ),
                    )
                )
                break

            try:
                reducer_results.append(
                    reduce_worker_bundle_v2(
                        self.context,
                        bundle_path,
                        reducer_id=validate_artifact_id(f"reducer.{node.node_id}"),
                    )
                )
            except WorkerBundleV2Error as exc:
                stop_conditions.append(
                    StopCondition(
                        reason="worker_bundle_invalid",
                        description=str(exc),
                    )
                )
                break

        terminal_state = (
            OrchestratorState.FAILED if stop_conditions else OrchestratorState.COMPLETED
        )
        final_run = run.transition(
            terminal_state,
            now=_utc_now(),
            worker_calls=worker_calls,
            reducer_results=reducer_results,
            stop_conditions=stop_conditions,
        )
        write_yaml_deterministic(record_path, final_run)
        write_orchestrator_run_log(log_path, final_run)
        return OrchestratorLocalRunResult(
            run=final_run,
            run_root=run_root,
            record_path=record_path,
            log_path=log_path,
        )

    def _ensure_task_for_node(
        self,
        issue_id: str,
        node: TaskNode,
        *,
        now: datetime,
    ) -> AgentTask:
        task_id = validate_artifact_id(f"task.{node.node_id}")
        try:
            return self.task_harness.load_task(task_id)
        except TaskHarnessError as exc:
            if "task not found" not in str(exc):
                raise OrchestratorLocalRunError(str(exc)) from exc
            return self.task_harness.create_task(
                issue_id=issue_id,
                worker_type=node.worker_type,
                now=now,
                task_id=task_id,
            )

    def _bundle_path(
        self,
        config: OrchestratorLocalRunConfig,
        run_id: str,
        node: TaskNode,
    ) -> Path:
        return (
            Path(".cosheaf")
            / "orchestrator"
            / config.issue_id
            / "runs"
            / run_id
            / "bundles"
            / f"{node.node_id}.yaml"
        )

    def _proposal_path(
        self,
        config: OrchestratorLocalRunConfig,
        run_id: str,
        node: TaskNode,
    ) -> str:
        if config.proposal_path is not None:
            return str(config.proposal_path)
        return (
            Path(".cosheaf")
            / "orchestrator"
            / config.issue_id
            / "runs"
            / run_id
            / "proposals"
            / f"{node.node_id}.yaml"
        ).as_posix()


class OrchestratorHostedRunner:
    """Dispatch planned task nodes to hosted-worker service calls."""

    def __init__(self, context: RepoContext) -> None:
        self.context = context
        self.hosted_workers = HostedWorkerService(context)

    def run_issue(
        self,
        config: OrchestratorHostedRunConfig,
        *,
        provider_config: ProviderConfig | None = None,
        provider: OpenAICompatibleProvider | None = None,
    ) -> OrchestratorHostedRunResult:
        """Run one issue through hosted-worker dispatch boundaries."""
        resolved_provider = _resolve_hosted_provider(config.provider)
        provider_mode = _provider_mode_for_name(resolved_provider)
        if resolved_provider is not ProviderName.FAKE and not config.confirm_send:
            raise OrchestratorHostedRunError(
                _error(
                    "provider_confirm_send_required",
                    "hosted provider dispatch requires --confirm-send",
                    "Preview the context, then rerun with --confirm-send only "
                    "when the operator approves the provider call.",
                )
            )

        now = _normalize_timestamp(config.now or _utc_now())
        run_id = config.run_id or validate_artifact_id(f"run.{config.issue_id}")
        try:
            plan = plan_for_issue(self.context, config.issue_id)
        except OrchestratorPlannerError as exc:
            raise OrchestratorHostedRunError(
                _error(
                    "orchestrator_plan_failed",
                    str(exc),
                    "Check the issue ID and repository records.",
                )
            ) from exc

        preview = self._context_preview(config)
        consent = _provider_consent(config, resolved_provider)
        run_root = self.context.resolve(
            Path(".cosheaf")
            / "orchestrator"
            / config.issue_id
            / "runs"
            / run_id
        )
        record_path = run_root / "run.yaml"
        log_path = run_root / "run_log.json"
        if run_root.exists():
            raise OrchestratorHostedRunError(
                _error(
                    "provider_request_validation_failed",
                    f"orchestrator run already exists: {run_id}",
                    "Choose a distinct run ID.",
                )
            )
        run_root.mkdir(parents=True, exist_ok=False)

        run = (
            OrchestratorRun.create(
                run_id=run_id,
                issue_id=config.issue_id,
                now=now,
            )
            .transition(OrchestratorState.PLANNED, now=now, plan=plan)
            .transition(OrchestratorState.RUNNING, now=now)
        )

        worker_calls: list[WorkerCall] = []
        reducer_results = []
        stop_conditions: list[StopCondition] = []
        provider_run_record_paths: list[Path] = []

        active_config = provider_config or _default_provider_config(resolved_provider)
        completed_nodes: dict[str, _HostedNodeExecution] = {}
        for index, node in enumerate(plan.task_dag.nodes, start=1):
            missing_deps = [
                dep_id
                for dep_id in node.depends_on
                if dep_id not in completed_nodes
            ]
            if missing_deps:
                stop_conditions.append(
                    StopCondition(
                        reason="unsatisfied_node_dependency",
                        description=(
                            f"{node.node_id} missing upstream dependencies: "
                            + ", ".join(missing_deps)
                        ),
                    )
                )
                break
            upstream_executions = [
                completed_nodes[dep_id] for dep_id in node.depends_on
            ]
            node_artifact_ids = _merge_upstream_artifact_ids(
                preview.artifact_ids,
                node.input_artifacts,
                upstream_executions,
            )
            upstream_bundle_paths = _collect_upstream_bundle_paths(
                upstream_executions
            )
            upstream_proposed_paths = _collect_upstream_proposed_paths(
                upstream_executions
            )
            role = _role_for_node(node)
            worker_output = self.hosted_workers.run(
                HostedWorkerInput(
                    issue_id=config.issue_id,
                    role=role,
                    prompt=_hosted_worker_prompt(
                        plan.objective,
                        node,
                        upstream_executions=upstream_executions,
                    ),
                    context_artifact_ids=node_artifact_ids,
                    root_scopes=[scope.value for scope in preview.root_scopes],
                    upstream_bundle_paths=upstream_bundle_paths,
                    upstream_proposed_paths=upstream_proposed_paths,
                    consent=consent,
                ),
                config=active_config,
                provider=provider,
            )
            provider_log_copy = _copy_provider_log(
                self.context,
                run_root,
                index=index,
                role=role,
                output=worker_output,
            )
            if provider_log_copy is not None:
                provider_run_record_paths.append(provider_log_copy)

            bundle_path = None
            typed_result_path = None
            if worker_output.bundle is not None:
                bundle_path = _hosted_bundle_path(run_root, node)
                write_yaml_deterministic(bundle_path, worker_output.bundle)
            elif worker_output.typed_result is not None:
                typed_result_path = _hosted_typed_result_path(run_root, node)
                write_yaml_deterministic(
                    typed_result_path,
                    worker_output.typed_result,
                )

            rel_bundle_path = (
                repo_relative_posix(self.context.repo_root, bundle_path)
                if bundle_path is not None
                else None
            )
            rel_typed_result_path = (
                repo_relative_posix(self.context.repo_root, typed_result_path)
                if typed_result_path is not None
                else None
            )

            worker_calls.append(
                WorkerCall(
                    call_id=validate_artifact_id(f"call.{node.node_id}"),
                    task_node_id=node.node_id,
                    worker_type=node.worker_type,
                    status=worker_output.status.value,
                    command=[
                        "hosted-worker",
                        role.value,
                        "--provider",
                        _provider_cli_name(resolved_provider),
                    ],
                    cwd=".",
                    started_at=now,
                    completed_at=now,
                    exit_code=0
                    if worker_output.status is HostedWorkerStatus.COMPLETED
                    else 1,
                    bundle_path=rel_bundle_path,
                    notes=_hosted_worker_notes(worker_output),
                )
            )
            if worker_output.status is not HostedWorkerStatus.COMPLETED:
                stop_conditions.append(
                    StopCondition(
                        reason="hosted_worker_failed",
                        description=_hosted_worker_notes(worker_output),
                    )
                )
                break

            if worker_output.bundle is not None and bundle_path is not None:
                try:
                    reducer_results.append(
                        reduce_worker_bundle_v2(
                            self.context,
                            bundle_path,
                            reducer_id=validate_artifact_id(
                                f"reducer.{node.node_id}"
                            ),
                        )
                    )
                except WorkerBundleV2Error as exc:
                    stop_conditions.append(
                        StopCondition(
                            reason="worker_bundle_invalid",
                            description=str(exc),
                        )
                    )
                    break

            completed_nodes[node.node_id] = _HostedNodeExecution(
                node=node,
                role=role,
                output=worker_output,
                bundle_path=rel_bundle_path,
                typed_result_path=rel_typed_result_path,
            )

        terminal_state = (
            OrchestratorState.FAILED if stop_conditions else OrchestratorState.COMPLETED
        )
        final_run = run.transition(
            terminal_state,
            now=_utc_now(),
            worker_calls=worker_calls,
            reducer_results=reducer_results,
            stop_conditions=stop_conditions,
        )
        write_yaml_deterministic(record_path, final_run)
        write_orchestrator_run_log(log_path, final_run)
        return OrchestratorHostedRunResult(
            run=final_run,
            run_root=run_root,
            record_path=record_path,
            log_path=log_path,
            context_preview=preview,
            provider=resolved_provider,
            provider_mode=provider_mode,
            provider_run_record_paths=tuple(provider_run_record_paths),
            accepted_write_performed=False,
        )

    def _context_preview(
        self,
        config: OrchestratorHostedRunConfig,
    ) -> ProviderContextPreview:
        request = ContextBuildRequest(
            issue_id=config.issue_id,
            max_cards=config.max_cards,
            max_full_artifacts=0,
            policy_mode=config.policy_mode,
            public_only=not config.include_private,
            allow_private_context=config.allow_private_context,
        )
        preview = ContextSendPolicyService(self.context).provider_preview(request)
        if isinstance(preview, ErrorResult):
            raise OrchestratorHostedRunError(preview)
        return preview


def _default_worker_command(
    *,
    bundle_path: Path,
    task: AgentTask,
    node: TaskNode,
    proposal_path: str,
    created_at: datetime,
) -> list[str]:
    return dry_run_worker_command(
        bundle_path=bundle_path,
        task=task,
        node=node,
        proposal_path=proposal_path,
        created_at=created_at,
    )


def _default_provider_config(provider: ProviderName) -> ProviderConfig:
    if provider is ProviderName.FAKE:
        return ProviderConfig(
            provider=ProviderName.FAKE,
            mode=ProviderMode.FAKE,
            model="fake-deterministic",
            enabled=True,
        )
    return ProviderConfig(
        provider=ProviderName.OPENAI,
        mode=ProviderMode.OPENAI_COMPATIBLE,
        model="openai-compatible",
        enabled=True,
        api_key_env=None,
    )


def _provider_consent(
    config: OrchestratorHostedRunConfig,
    provider: ProviderName,
) -> ProviderConsent:
    consent_required = provider is not ProviderName.FAKE or config.include_private
    consent_granted = config.confirm_send or (
        provider is ProviderName.FAKE and config.allow_private_context
    )
    return ProviderConsent(
        consent_required=consent_required,
        consent_granted=consent_granted,
        allow_private_context=config.allow_private_context,
        policy_scope=config.policy_mode,
        operator_note=(
            "Explicit hosted orchestrator dispatch consent."
            if consent_granted
            else ""
        ),
    )


def _resolve_hosted_provider(provider: ProviderName | str) -> ProviderName:
    if isinstance(provider, ProviderName):
        resolved = provider
    else:
        normalized = provider.strip().replace("_", "-").lower()
        if normalized in {"openai-compatible", "openai"}:
            resolved = ProviderName.OPENAI
        elif normalized == "fake":
            resolved = ProviderName.FAKE
        else:
            raise OrchestratorHostedRunError(
                _error(
                    "provider_unsupported",
                    f"orchestrator provider is not supported: {provider}",
                    "Use --provider fake or --provider openai-compatible.",
                    details={"supported_providers": "fake,openai-compatible"},
                )
            )
    if resolved in {ProviderName.FAKE, ProviderName.OPENAI}:
        return resolved
    raise OrchestratorHostedRunError(
        _error(
            "provider_unsupported",
            f"orchestrator provider is not supported: {resolved.value}",
            "Use --provider fake or --provider openai-compatible.",
            details={"supported_providers": "fake,openai-compatible"},
        )
    )


def _provider_mode_for_name(provider: ProviderName) -> ProviderMode:
    if provider is ProviderName.FAKE:
        return ProviderMode.FAKE
    return ProviderMode.OPENAI_COMPATIBLE


def _provider_cli_name(provider: ProviderName) -> str:
    if provider is ProviderName.OPENAI:
        return "openai-compatible"
    return provider.value


def _role_for_node(node: TaskNode) -> RoleName:
    if node.worker_type is WorkerType.REASONER:
        return RoleName.REASONER
    if node.worker_type is WorkerType.VERIFIER:
        return RoleName.VERIFIER
    if node.worker_type is WorkerType.COUNTEREXAMPLER:
        return RoleName.COUNTEREXAMPLER
    if node.worker_type is WorkerType.FORMALIZER:
        return RoleName.FORMALIZER
    if "librarian" in node.node_id:
        return RoleName.LIBRARIAN_SUMMARIZER
    return RoleName.EXPLORER


@dataclass(frozen=True)
class _HostedNodeExecution:
    node: TaskNode
    role: RoleName
    output: HostedWorkerOutput
    bundle_path: str | None = None
    typed_result_path: str | None = None


def _merge_upstream_artifact_ids(
    preview_artifact_ids: Sequence[str],
    node_input_artifacts: Sequence[str],
    upstream_executions: Sequence[_HostedNodeExecution],
) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()
    candidates: list[str] = [
        *preview_artifact_ids,
        *node_input_artifacts,
    ]
    for up in upstream_executions:
        if up.output.bundle is not None:
            candidates.extend(up.output.bundle.used_artifacts)
        if up.output.typed_result is not None:
            candidates.extend(up.output.typed_result.selected_artifacts)
    for item in candidates:
        if item in seen:
            continue
        seen.add(item)
        merged.append(item)
    return merged


def _collect_upstream_bundle_paths(
    upstream_executions: Sequence[_HostedNodeExecution],
) -> list[str]:
    paths: list[str] = []
    for up in upstream_executions:
        if up.bundle_path is not None and up.bundle_path not in paths:
            paths.append(up.bundle_path)
        if up.typed_result_path is not None and up.typed_result_path not in paths:
            paths.append(up.typed_result_path)
    return paths


def _collect_upstream_proposed_paths(
    upstream_executions: Sequence[_HostedNodeExecution],
) -> list[str]:
    paths: list[str] = []
    for up in upstream_executions:
        if up.output.bundle is None:
            continue
        for proposed in up.output.bundle.proposed_artifacts:
            if proposed.path not in paths:
                paths.append(proposed.path)
    return paths


def _hosted_worker_prompt(
    objective: str,
    node: TaskNode,
    *,
    upstream_executions: Sequence[_HostedNodeExecution] = (),
) -> str:
    lines = [
        f"Objective: {objective}",
        f"Task node: {node.node_id}",
        f"Description: {node.description}",
    ]
    if upstream_executions:
        lines.append(
            "Upstream dependencies: "
            + ", ".join(up.node.node_id for up in upstream_executions)
        )
        for up in upstream_executions:
            if up.bundle_path is not None:
                lines.append(
                    f"Upstream bundle ({up.node.node_id}, {up.role.value}): "
                    f"{up.bundle_path}"
                )
            elif up.typed_result_path is not None:
                lines.append(
                    f"Upstream typed result ({up.node.node_id}, {up.role.value}): "
                    f"{up.typed_result_path}"
                )
            if up.output.bundle is not None:
                lines.append(
                    f"Upstream summary ({up.node.node_id}): "
                    f"{up.output.bundle.summary}"
                )
                if up.output.bundle.claims:
                    lines.append(
                        f"Upstream claims ({up.node.node_id}): "
                        + "; ".join(up.output.bundle.claims)
                    )
                if up.output.bundle.proposed_artifacts:
                    lines.append(
                        f"Upstream proposed drafts ({up.node.node_id}): "
                        + ", ".join(
                            prop.path
                            for prop in up.output.bundle.proposed_artifacts
                        )
                    )
                if up.output.bundle.verification_requests:
                    lines.append(
                        f"Upstream verification requests ({up.node.node_id}): "
                        + "; ".join(up.output.bundle.verification_requests)
                    )
            elif up.output.typed_result is not None:
                lines.append(
                    f"Upstream summary ({up.node.node_id}): "
                    f"{up.output.typed_result.summary}"
                )
    lines.append("Return review-only output for this orchestrator run.")
    return "\n".join(lines)


def _hosted_bundle_path(run_root: Path, node: TaskNode) -> Path:
    return run_root / "bundles" / f"{node.node_id}.yaml"


def _hosted_typed_result_path(run_root: Path, node: TaskNode) -> Path:
    return run_root / "typed-results" / f"{node.node_id}.yaml"


def _copy_provider_log(
    context: RepoContext,
    run_root: Path,
    *,
    index: int,
    role: RoleName,
    output: HostedWorkerOutput,
) -> Path | None:
    if output.provider_log_path is None:
        return None
    source = context.resolve(Path(output.provider_log_path))
    if not source.is_file():
        return None
    target = run_root / "providers" / f"{index:02d}-{role.value}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    return target


def _hosted_worker_notes(output: HostedWorkerOutput) -> str:
    if output.error is None:
        return "hosted worker review-only output"
    return f"{output.error.code}: {output.error.message}"


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


def _utc_now() -> datetime:
    return datetime.now(UTC).replace(microsecond=0)


def _normalize_timestamp(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise OrchestratorLocalRunError("timestamp must include timezone information")
    return value.astimezone(UTC).replace(microsecond=0)


def _format_timestamp(value: datetime) -> str:
    return value.astimezone(UTC).replace(microsecond=0).isoformat().replace(
        "+00:00",
        "Z",
    )
