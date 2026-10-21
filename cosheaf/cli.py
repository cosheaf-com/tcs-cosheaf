from __future__ import annotations

import json
import os
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, NoReturn, cast

import typer
from pydantic import ValidationError
from rich.console import Console

from cosheaf import __version__
from cosheaf.actions.cli import action_app
from cosheaf.agent.local_runner import (
    LocalWorkerRunError,
)
from cosheaf.agent.model_provider import NetworkPolicy, ProviderName
from cosheaf.agent.orchestrator_planner import (
    OrchestratorPlannerError,
    plan_for_issue,
)
from cosheaf.agent.orchestrator_runner import (
    OrchestratorHostedRunConfig,
    OrchestratorHostedRunError,
    OrchestratorHostedRunner,
    OrchestratorHostedRunResult,
    OrchestratorLocalRunConfig,
    OrchestratorLocalRunError,
    OrchestratorLocalRunner,
    OrchestratorLocalRunResult,
)
from cosheaf.agent.orchestrator_stub import TaskHarnessError
from cosheaf.agent.providers import (
    OpenAICompatibleHttpTransport,
    OpenAICompatibleProvider,
    ProviderConfig,
    ProviderError,
    ProviderGatewayRequest,
    ProviderMode,
)
from cosheaf.agent.task import WorkerType
from cosheaf.benchmark import (
    BenchmarkError,
    BenchmarkSuiteName,
    list_benchmark_suites,
    run_benchmark_suite,
    write_benchmark_report,
)
from cosheaf.checkers.cli import checker_app
from cosheaf.compare import (
    COMPARE_AUTHORITY_NOTICE,
    compare_benchmarks,
    compare_campaigns,
    compare_workflows,
)
from cosheaf.config.workspace import KbRootConfig, WorkspaceConfigError
from cosheaf.context_cli import context_app
from cosheaf.core.artifact import (
    BaseArtifact,
    evidence_revision_mismatches,
    is_external_dependency_ref,
    review_revision_mismatch,
)
from cosheaf.core.ids import validate_artifact_id
from cosheaf.core.paths import (
    lifecycle_artifact_path,
    normalize_repo_path,
    repo_relative_posix,
)
from cosheaf.core.status import (
    ArtifactStatus,
    ArtifactType,
    expected_status_for_path,
    is_preaccepted_status,
)
from cosheaf.evals import (
    DEFAULT_CAMPAIGN_EVAL_CASES,
    DEFAULT_CHECKED_EVIDENCE_RUN_LOOP_EVAL_CASES,
    DEFAULT_CHECKER_CROSSCHECK_EVAL_CASES,
    DEFAULT_CONTEXT_EVAL_CASES,
    DEFAULT_RESEARCH_LOOP_EVAL_CASES,
    DEFAULT_RESEARCH_RUN_LOOP_EVAL_CASES,
    DEFAULT_RETRIEVAL_EVAL_CASES,
    DEFAULT_REVIEWABLE_WORKFLOW_EVAL_CASES,
    DEFAULT_STRATEGY_PLANNER_EVAL_CASES,
    CampaignEvalError,
    CheckedEvidenceRunLoopEvalError,
    CheckerCrossCheckEvalError,
    ContextEvalError,
    ResearchLoopEvalError,
    ResearchRunLoopEvalError,
    RetrievalEvalError,
    ReviewableWorkflowEvalError,
    StrategyPlannerEvalError,
    load_campaign_eval_suite,
    load_checked_evidence_run_loop_eval_suite,
    load_checker_crosscheck_eval_suite,
    load_context_eval_suite,
    load_research_loop_eval_suite,
    load_research_run_loop_eval_suite,
    load_retrieval_eval_suite,
    load_reviewable_workflow_eval_suite,
    load_strategy_planner_eval_suite,
    resolve_campaign_eval_case_path,
    resolve_checked_evidence_run_loop_eval_case_path,
    resolve_checker_crosscheck_eval_case_path,
    resolve_context_eval_case_path,
    resolve_research_loop_eval_case_path,
    resolve_research_run_loop_eval_case_path,
    resolve_retrieval_eval_case_path,
    resolve_reviewable_workflow_eval_case_path,
    resolve_strategy_planner_eval_case_path,
    run_campaign_eval_suite,
    run_checked_evidence_run_loop_eval_suite,
    run_checker_crosscheck_eval_suite,
    run_context_eval_suite,
    run_research_loop_eval_suite,
    run_research_run_loop_eval_suite,
    run_retrieval_eval_suite,
    run_reviewable_workflow_eval_suite,
    run_strategy_planner_eval_suite,
)
from cosheaf.forge.cli import forge_app
from cosheaf.gap_cli import gap_app
from cosheaf.gates.gatekeeper import (
    GatekeeperRunResult,
    ValidationReport,
    run_gatekeeper,
    validate_artifact_file,
    validate_repository,
)
from cosheaf.gates.promotion_readiness import build_promotion_readiness_report
from cosheaf.gates.source_metadata_gate import missing_required_source_metadata
from cosheaf.graph.claim_graph import DependencyGraph, build_dependency_graph
from cosheaf.ingest import IngestError, MarkItDownIngestAdapter
from cosheaf.issues.cli import issue_app
from cosheaf.librarian.cli import librarian_app
from cosheaf.mcp import READ_ONLY_TOOL_NAMES, serve_stdio
from cosheaf.memory import (
    MEMORY_GRAPH_SIDECAR,
    ArtifactCardStatus,
    MemoryCardError,
    MemoryGraphError,
    MemoryRootScope,
    MemorySearchError,
    MemoryUpdateError,
    build_memory_graph,
    compute_global_pagerank,
    explain_memory_weight,
    load_memory_graph_snapshot,
    rebuild_memory_weights,
    update_memory_from_campaign,
    update_memory_from_workflow,
)
from cosheaf.operator_session import (
    OPERATOR_SESSION_AUTHORITY_NOTICE,
    SKIPPED_OPERATOR_SESSION_LIMITATION,
    OperatorArtifactRef,
    OperatorArtifactRefKind,
    OperatorCheckKind,
    OperatorCheckResult,
    OperatorCheckStatus,
    OperatorPolicyMode,
    OperatorSession,
    OperatorSessionError,
    OperatorSessionStatus,
    append_operator_session_event,
    build_operator_handoff,
    export_operator_handoff,
    load_operator_handoff,
    load_operator_session,
    scan_operator_session,
    start_operator_session,
    write_operator_session,
)
from cosheaf.orchestrator_fsm.cli import fsm_app
from cosheaf.reports import (
    write_benchmark_static_report,
    write_campaign_report,
    write_workflow_report,
)
from cosheaf.research.run import (
    RESEARCH_RUN_AUTHORITY_NOTICE,
    ResearchRunError,
    append_artifact_to_research_run,
    append_command_to_research_run,
    append_output_to_research_run,
    build_replay_plan,
    build_research_run_evidence_report,
    export_research_run_review,
    finalize_research_run,
    load_research_run,
    start_research_run,
)
from cosheaf.server.cli import server_app
from cosheaf.services import (
    BundleValidationService,
    ContextSendPolicyService,
    ControlledWriteResult,
    DraftWriteService,
    DraftWriteServiceError,
    FailureLogFromBundlePlanResult,
    FailureLogFromBundleWriteResult,
    MemorySearchService,
    ServiceError,
    TaskService,
    ValidationService,
    WorkspaceService,
)
from cosheaf.services.model_calls import ModelCallService
from cosheaf.services.models import (
    AgentAccessModel,
    AgentKbRoot,
    ContextBuildRequest,
    ContextPolicyMode,
    DraftArtifactWriteRequest,
    ErrorResult,
    KbRootPolicy,
    ModelCallResult,
    ProviderConsent,
    ProviderContextPreview,
    WorkerBundleSubmitRequest,
)
from cosheaf.services.models import (
    WorkspaceInfoResult as AgentWorkspaceInfoResult,
)
from cosheaf.site.cli import site_app
from cosheaf.storage.index import rebuild_index
from cosheaf.storage.loader import LoadedRecord, LoadError, load_artifacts
from cosheaf.storage.repo import RepoContext
from cosheaf.storage.writer import write_yaml_deterministic
from cosheaf.strategy.models import STRATEGY_AUTHORITY_NOTICE, StrategyError
from cosheaf.strategy.planner import build_strategy_plan
from cosheaf.strategy.storage import (
    attach_context_reference,
    export_strategy_review,
    load_strategy_plan,
    update_strategy_plan_from_run,
    write_strategy_plan,
)
from cosheaf.validation_cli import (
    gate_app,
    register_validation_commands,
    run_validation,
)
from cosheaf.verification.counterexample_evidence import (
    CHECKED_COUNTEREXAMPLE_AUTHORITY_NOTICE,
    CheckedCounterexampleEvidenceError,
    show_checked_counterexample_evidence,
    stage_checked_counterexample_evidence,
    validate_checked_counterexample_evidence_payload,
)
from cosheaf.workflow.cli import workflow_app

app = typer.Typer(
    add_completion=False,
    help="TCS-Cosheaf research knowledge base harness.",
    no_args_is_help=True,
)
artifact_app = typer.Typer(
    add_completion=False,
    help="Artifact commands.",
    no_args_is_help=True,
)
artifact_failure_app = typer.Typer(
    add_completion=False,
    help="Artifact failure-log write commands.",
    no_args_is_help=True,
)
counterexample_app = typer.Typer(
    add_completion=False,
    help="Counterexample commands.",
    no_args_is_help=True,
)
counterexample_evidence_app = typer.Typer(
    add_completion=False,
    help="Checked counterexample evidence commands.",
    no_args_is_help=True,
)
index_app = typer.Typer(
    add_completion=False,
    help="Index commands.",
    no_args_is_help=True,
)
graph_app = typer.Typer(
    add_completion=False,
    help="Graph commands.",
    no_args_is_help=True,
)
interface_app = typer.Typer(
    add_completion=False,
    help="Public interface discovery commands.",
    no_args_is_help=True,
)
task_app = typer.Typer(
    add_completion=False,
    help="Agent task commands.",
    no_args_is_help=True,
)
draft_app = typer.Typer(
    add_completion=False,
    help="Controlled draft/staging write commands.",
    no_args_is_help=True,
)
bundle_app = typer.Typer(
    add_completion=False,
    help="Worker bundle review-submission commands.",
    no_args_is_help=True,
)
review_app = typer.Typer(
    add_completion=False,
    help="Controlled review request commands.",
    no_args_is_help=True,
)
run_app = typer.Typer(
    add_completion=False,
    help="Research-run provenance commands.",
    no_args_is_help=True,
)
research_loop_app = typer.Typer(
    add_completion=False,
    help="Bounded research-loop commands.",
    no_args_is_help=True,
)
campaign_app = typer.Typer(
    add_completion=False,
    help="Bounded research campaign commands.",
    no_args_is_help=True,
)
orchestrator_app = typer.Typer(
    add_completion=False,
    help="Deterministic local orchestrator commands.",
    no_args_is_help=True,
)
strategy_app = typer.Typer(
    add_completion=False,
    help="Strategy planner and research task graph commands.",
    no_args_is_help=True,
)
workspace_app = typer.Typer(
    add_completion=False,
    help="Workspace configuration commands.",
    no_args_is_help=True,
)
ingest_app = typer.Typer(
    add_completion=False,
    help="Source ingestion commands.",
    no_args_is_help=True,
)
eval_app = typer.Typer(
    add_completion=False,
    help="Deterministic local evaluation commands.",
    no_args_is_help=True,
)
benchmark_app = typer.Typer(
    add_completion=False,
    help="Deterministic benchmark suite commands.",
    no_args_is_help=True,
)
compare_app = typer.Typer(
    add_completion=False,
    help="Analytical comparison commands.",
    no_args_is_help=True,
)
report_app = typer.Typer(
    add_completion=False,
    help="Static review report commands.",
    no_args_is_help=True,
)
memory_app = typer.Typer(
    add_completion=False,
    help="Deterministic memory/card commands.",
    no_args_is_help=True,
)
memory_graph_app = typer.Typer(
    add_completion=False,
    help="Deterministic memory graph commands.",
    no_args_is_help=True,
)
mcp_app = typer.Typer(
    add_completion=False,
    help="Optional MCP adapter commands.",
    no_args_is_help=True,
)
provider_app = typer.Typer(
    add_completion=False,
    help="Provider gateway preview and fake-run commands.",
    no_args_is_help=True,
)
operator_app = typer.Typer(
    add_completion=False,
    help="Operator audit and handoff commands.",
    no_args_is_help=True,
)
operator_session_app = typer.Typer(
    add_completion=False,
    help="Operator session metadata commands.",
    no_args_is_help=True,
)
operator_handoff_app = typer.Typer(
    add_completion=False,
    help="Operator handoff bundle commands.",
    no_args_is_help=True,
)
promotion_app = typer.Typer(
    add_completion=False,
    help="Read-only promotion readiness reports.",
    no_args_is_help=True,
)
app.add_typer(artifact_app, name="artifact")
artifact_app.add_typer(artifact_failure_app, name="failure")
app.add_typer(counterexample_app, name="counterexample")
counterexample_app.add_typer(counterexample_evidence_app, name="evidence")
app.add_typer(index_app, name="index")
app.add_typer(graph_app, name="graph")
app.add_typer(interface_app, name="interface")
app.add_typer(gate_app, name="gate")
app.add_typer(checker_app, name="checker")
app.add_typer(gap_app, name="gap")
app.add_typer(action_app, name="action")
app.add_typer(librarian_app, name="librarian")
app.add_typer(fsm_app, name="orchestrator-fsm")
app.add_typer(workflow_app, name="workflow")
app.add_typer(context_app, name="context")
app.add_typer(issue_app, name="issue")
app.add_typer(forge_app, name="forge")
app.add_typer(site_app, name="site")
app.add_typer(server_app, name="server")
app.add_typer(task_app, name="task")
app.add_typer(draft_app, name="draft")
app.add_typer(bundle_app, name="bundle")
app.add_typer(review_app, name="review")
app.add_typer(run_app, name="research-run")
app.add_typer(run_app, name="run")
app.add_typer(research_loop_app, name="research-loop")
app.add_typer(campaign_app, name="campaign")
app.add_typer(orchestrator_app, name="orchestrator")
app.add_typer(strategy_app, name="strategy")
app.add_typer(workspace_app, name="workspace")
app.add_typer(ingest_app, name="ingest")
app.add_typer(eval_app, name="eval")
app.add_typer(benchmark_app, name="benchmark")
app.add_typer(compare_app, name="compare")
app.add_typer(report_app, name="report")
app.add_typer(memory_app, name="memory")
app.add_typer(mcp_app, name="mcp")
app.add_typer(provider_app, name="provider")
app.add_typer(operator_app, name="operator")
operator_app.add_typer(operator_session_app, name="session")
operator_app.add_typer(operator_handoff_app, name="handoff")
app.add_typer(promotion_app, name="promotion")
memory_app.add_typer(memory_graph_app, name="graph")

_SUPPORTED_PROVIDER_CLI_NAMES = (ProviderName.FAKE, ProviderName.OPENAI)
_OPERATOR_SESSION_CLI_CHECK_KINDS = frozenset(
    {
        OperatorCheckKind.VALIDATE.value,
        OperatorCheckKind.GATE.value,
        OperatorCheckKind.TEST.value,
        OperatorCheckKind.EVAL.value,
    }
)
_OPERATOR_SESSION_CLI_REF_KINDS = frozenset(
    {
        OperatorArtifactRefKind.DRAFT.value,
        OperatorArtifactRefKind.REVIEW_CONTEXT.value,
        OperatorArtifactRefKind.RUNTIME.value,
        OperatorArtifactRefKind.REPORT.value,
    }
)
OperatorArtifactRefScope = Literal[
    "public",
    "private",
    "workspace",
    "framework",
    "unknown",
]
_OPERATOR_SESSION_CLI_REF_SCOPES = frozenset(
    {
        "public",
        "private",
        "workspace",
        "framework",
        "unknown",
    }
)
_FAILURE_LOG_AUTHORITY_NOTICE = (
    "failure_log is research memory only; it is not proof, verifier success, "
    "checked counterexample evidence, human review, gate success, accepted "
    "status, or promotion evidence"
)
_INTERFACE_AUTHORITY_NOTICE = (
    "Interface discovery is documentation metadata only; it does not grant "
    "proof, source metadata, human review, verifier pass, gate pass, accepted "
    "status, accepted theorem/refutation status, or promotion authority."
)
_STABLE_V1_CLI_SURFACE: tuple[dict[str, str], ...] = (
    {
        "command": "workspace",
        "preferred_invocation": "cosheaf workspace info",
        "status": "stable",
        "notes": "Inspect configured KB roots and workspace policy.",
    },
    {
        "command": "validate",
        "preferred_invocation": "cosheaf validate",
        "status": "stable",
        "notes": "Validate YAML records and repository invariants.",
    },
    {
        "command": "gate",
        "preferred_invocation": "cosheaf gate run",
        "status": "stable",
        "notes": "Run gatekeeper checks; skipped rows are not passes.",
    },
    {
        "command": "index",
        "preferred_invocation": "cosheaf index rebuild",
        "status": "stable",
        "notes": "Rebuild deterministic local search/query index.",
    },
    {
        "command": "context",
        "preferred_invocation": "cosheaf context build <issue-id>",
        "status": "stable",
        "notes": "Build review context packs for issues.",
    },
    {
        "command": "issue",
        "preferred_invocation": "cosheaf issue ...",
        "status": "stable",
        "notes": "Manage repository-local issue YAML without GitHub calls.",
    },
    {
        "command": "forge",
        "preferred_invocation": "cosheaf forge ...",
        "status": "stable_confirmed",
        "notes": (
            "Preview GitHub plans, run confirmed local git actions, "
            "and create confirmed GitHub issues/PRs."
        ),
    },
    {
        "command": "site",
        "preferred_invocation": "cosheaf site export --out .cosheaf/site-data",
        "status": "stable",
        "notes": (
            "Export deterministic sanitized JSON sidecars for the read-only "
            "website."
        ),
    },
    {
        "command": "server",
        "preferred_invocation": (
            "cosheaf server serve --readonly --port 8765 --local-actor <name>"
        ),
        "status": "stable_readonly_local",
        "notes": (
            "Serve localhost website JSON APIs through the app facade. "
            "--local-actor is required before confirmed human review or "
            "promotion actions in local mode."
        ),
    },
    {
        "command": "artifact",
        "preferred_invocation": "cosheaf artifact ...",
        "status": "stable",
        "notes": "Inspect and manage artifact lifecycle through governed paths.",
    },
    {
        "command": "promotion",
        "preferred_invocation": "cosheaf promotion readiness <artifact-id>",
        "status": "stable",
        "notes": "Read-only promotion readiness reporting.",
    },
    {
        "command": "workflow",
        "preferred_invocation": "cosheaf workflow ...",
        "status": "stable",
        "notes": "Reviewable workflow records, draft proposals, and handoffs.",
    },
    {
        "command": "checker",
        "preferred_invocation": "cosheaf checker ...",
        "status": "stable",
        "notes": "Checker registry and local checker runs.",
    },
    {
        "command": "gap",
        "preferred_invocation": "cosheaf gap ...",
        "status": "stable",
        "notes": "Workflow gap reports and exports.",
    },
    {
        "command": "research-loop",
        "preferred_invocation": "cosheaf research-loop ...",
        "status": "stable",
        "notes": "Bounded local research-loop metadata.",
    },
    {
        "command": "research-run",
        "preferred_invocation": "cosheaf research-run ...",
        "status": "stable",
        "notes": "Research-run provenance and review context.",
    },
    {
        "command": "campaign",
        "preferred_invocation": "cosheaf campaign ...",
        "status": "stable",
        "notes": "Bounded research campaign metadata.",
    },
    {
        "command": "memory",
        "preferred_invocation": "cosheaf memory ...",
        "status": "stable",
        "notes": "Deterministic memory cards, search, and sidecar updates.",
    },
    {
        "command": "benchmark",
        "preferred_invocation": "cosheaf benchmark ...",
        "status": "stable",
        "notes": "Deterministic local benchmark suite commands.",
    },
    {
        "command": "compare",
        "preferred_invocation": "cosheaf compare ...",
        "status": "stable",
        "notes": "Analytical comparison reports.",
    },
    {
        "command": "report",
        "preferred_invocation": "cosheaf report ...",
        "status": "stable",
        "notes": "Static review report generation.",
    },
    {
        "command": "operator session",
        "preferred_invocation": "cosheaf operator session ...",
        "status": "stable",
        "notes": "Operator audit-session metadata.",
    },
    {
        "command": "provider",
        "preferred_invocation": "cosheaf provider fake-run ...",
        "status": "stable_default_off",
        "notes": "Fake/preview/config paths; real hosted send remains explicit.",
    },
    {
        "command": "mcp",
        "preferred_invocation": "cosheaf mcp ...",
        "status": "optional_adapter",
        "notes": "Optional adapter surface; CLI remains the oracle.",
    },
)
_COMPATIBILITY_ALIASES: tuple[dict[str, str], ...] = (
    {
        "alias": "cosheaf gate",
        "preferred": "cosheaf gate run",
        "status": "compatibility_supported",
        "notes": "Kept for the v1.0 line; new docs should prefer gate run.",
    },
    {
        "alias": "cosheaf run",
        "preferred": "cosheaf research-run",
        "status": "compatibility_supported",
        "notes": "Kept for existing research-run provenance scripts.",
    },
)
_OPTIONAL_ADAPTER_SURFACES: tuple[dict[str, str], ...] = (
    {
        "surface": "provider real-run",
        "status": "experimental_default_off",
        "notes": "Requires explicit consent, network allowance, and configuration.",
    },
    {
        "surface": "mcp",
        "status": "optional_adapter",
        "notes": "Does not replace CLI authority.",
    },
    {
        "surface": "external verifier execution",
        "status": "optional",
        "notes": "Unavailable tools produce skipped results, not passes.",
    },
)


def _interface_list_payload() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "package": "tcs-cosheaf",
        "version": __version__,
        "target_release": "v1.0.0",
        "authority_notice": _INTERFACE_AUTHORITY_NOTICE,
        "stable_cli_surface": list(_STABLE_V1_CLI_SURFACE),
        "compatibility_aliases": list(_COMPATIBILITY_ALIASES),
        "optional_adapter_surfaces": list(_OPTIONAL_ADAPTER_SURFACES),
    }


@app.command()
def version(
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Emit deterministic JSON instead of text output.",
    ),
) -> None:
    """Print the TCS-Cosheaf version."""
    if json_output:
        _emit_json(
            {
                "schema_version": 1,
                "package": "tcs-cosheaf",
                "version": __version__,
            }
        )
        return
    Console().print(f"tcs-cosheaf {__version__}")


@interface_app.command("list")
def interface_list(
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Emit deterministic JSON instead of text output.",
    ),
) -> None:
    """List the stable v1.0 CLI surface and compatibility aliases."""
    payload = _interface_list_payload()
    if json_output:
        _emit_json(payload)
        return
    console = Console(markup=False)
    console.print("Stable v1.0 CLI surface:")
    for item in payload["stable_cli_surface"]:
        console.print(f"- {item['preferred_invocation']} [{item['status']}]")
    console.print("Compatibility aliases:")
    for item in payload["compatibility_aliases"]:
        console.print(f"- {item['alias']} -> {item['preferred']}")
    console.print(f"Authority: {payload['authority_notice']}")


@workspace_app.command("info")
def workspace_info(
    repo_root: Path = typer.Option(
        Path("."),
        "--repo-root",
        help="Repository root to inspect.",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Emit deterministic JSON instead of text output.",
    ),
) -> None:
    """Show workspace configuration and KB roots."""
    console = Console(width=120, markup=False)
    try:
        context = RepoContext(repo_root)
    except WorkspaceConfigError as exc:
        if json_output:
            _emit_error(
                ErrorResult(
                    code="workspace_config_failed",
                    message=str(exc),
                    remediation="Fix cosheaf.toml and rerun workspace info.",
                    blocking=True,
                    related_path="cosheaf.toml",
                )
            )
            raise typer.Exit(code=1) from None
        console.print(f"Workspace config failed: {exc}")
        raise typer.Exit(code=1) from None

    info = WorkspaceService(context).info()
    if json_output:
        _emit_model(_workspace_info_to_agent_result(context, info))
        return
    console.print(f"Workspace: {info.name}")
    console.print(f"- repo_root: {info.repo_root}")
    console.print(f"- mode: {info.mode}")
    console.print("KB roots:")
    for root in info.kb_roots:
        readonly = str(root.readonly).lower()
        console.print(
            f"- {root.name} | {root.path} | "
            f"readonly={readonly} | priority={root.priority}"
        )


register_validation_commands(app)


@artifact_app.command("validate")
def artifact_validate(
    path: Path = typer.Argument(..., help="Repository-local artifact YAML path."),
    repo_root: Path = typer.Option(
        Path("."),
        "--repo-root",
        help="Repository root used to resolve the artifact path.",
    ),
    debug: bool = typer.Option(
        False,
        "--debug",
        help="Show tracebacks for unexpected validation errors.",
    ),
) -> None:
    """Validate one artifact YAML file with file-local checks."""
    context = RepoContext(repo_root)
    run_validation(
        report_factory=lambda: ValidationService(context).validate_artifact_file(path),
        success_message="Artifact validation passed",
        failure_message="Artifact validation failed",
        debug=debug,
    )


@artifact_app.command("failures")
def artifact_failures(
    artifact_id: str = typer.Argument(..., help="Artifact ID to inspect."),
    repo_root: Path = typer.Option(
        Path("."),
        "--repo-root",
        help="Repository root to inspect.",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Emit deterministic JSON instead of text output.",
    ),
) -> None:
    """Inspect artifact failure-log entries without writing anything."""
    console = Console(width=120, markup=False)
    try:
        context = RepoContext(repo_root)
    except WorkspaceConfigError as exc:
        _exit_with_error(
            ErrorResult(
                code="workspace_config_failed",
                message=str(exc),
                remediation="Fix cosheaf.toml and rerun artifact failures.",
                blocking=True,
                related_path="cosheaf.toml",
            ),
            json_output=json_output,
            console=console,
        )

    try:
        validate_artifact_id(artifact_id)
    except ValueError as exc:
        _exit_with_error(
            ErrorResult(
                code="invalid_artifact_id",
                message=str(exc),
                remediation="Use a dot-separated lowercase artifact ID.",
                blocking=True,
            ),
            json_output=json_output,
            console=console,
        )

    try:
        loaded = _find_failure_log_artifact(context, artifact_id)
    except LoadError as exc:
        _exit_with_error(
            ErrorResult(
                code="repository_load_failed",
                message=f"cannot load repository records: {exc}",
                remediation="Fix repository YAML load errors and rerun the command.",
                blocking=True,
            ),
            json_output=json_output,
            console=console,
        )
    except ArtifactLifecycleError as exc:
        code = "artifact_not_found"
        remediation = "Check the artifact ID and rerun the command."
        if not str(exc).startswith("artifact not found:"):
            code = "repository_load_failed"
            remediation = "Fix duplicate or non-artifact record state and retry."
        _exit_with_error(
            ErrorResult(
                code=code,
                message=str(exc),
                remediation=remediation,
                blocking=True,
                related_artifact=artifact_id,
            ),
            json_output=json_output,
            console=console,
        )

    payload = _artifact_failure_log_payload(context, loaded)
    if json_output:
        _emit_json(payload)
        return

    console.print(f"Artifact failure log: {payload['artifact_id']}")
    console.print(f"- path: {payload['artifact_path']}")
    console.print(f"- root_scope: {payload['root_scope']}")
    console.print(f"- failure_count: {payload['failure_count']}")
    console.print(f"- authority: {payload['authority_notice']}")
    for entry in payload["failure_log"]:
        console.print(
            f"- {entry['failure_id']} | {entry['attempt_kind']} | "
            f"{entry['status']} | {entry['direction']}"
        )


@artifact_failure_app.command("add")
def artifact_failure_add(
    artifact_id: str = typer.Option(
        ...,
        "--artifact",
        help="Artifact ID whose failure_log should receive the entry.",
    ),
    input_json: Path = typer.Option(
        ...,
        "--input-json",
        help="JSON failure-log entry to append.",
    ),
    repo_root: Path = typer.Option(
        Path("."),
        "--repo-root",
        help="Repository root used for the controlled write.",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Emit deterministic JSON instead of text output.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Validate and report target path without writing files.",
    ),
) -> None:
    """Append or preview one failure-log entry on a writable artifact."""
    console = Console(width=120, markup=False)
    raw = _read_input_json_or_exit(input_json, json_output=json_output)
    try:
        result = DraftWriteService(RepoContext(repo_root)).append_failure_log_entry(
            artifact_id,
            raw,
            dry_run=dry_run,
        )
    except DraftWriteServiceError as exc:
        _exit_with_error(
            exc.to_error_result(),
            json_output=json_output,
            console=console,
        )

    _emit_controlled_write(result, json_output=json_output, console=console)


@artifact_failure_app.command("plan-from-bundle")
def artifact_failure_plan_from_bundle(
    bundle_path: Path = typer.Option(
        ...,
        "--bundle",
        help="Repository-local WorkerBundle v2 YAML path.",
    ),
    target_artifact_id: str = typer.Option(
        ...,
        "--target-artifact",
        help="Artifact ID that would receive the derived failure-log entries.",
    ),
    repo_root: Path = typer.Option(
        Path("."),
        "--repo-root",
        help="Repository root used for planning.",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Emit deterministic JSON instead of text output.",
    ),
) -> None:
    """Plan failure-log entries from a WorkerBundle without writing."""
    console = Console(width=120, markup=False)
    try:
        result = DraftWriteService(
            RepoContext(repo_root)
        ).plan_failure_log_entries_from_bundle(
            bundle_path,
            target_artifact_id=target_artifact_id,
        )
    except DraftWriteServiceError as exc:
        _exit_with_error(
            exc.to_error_result(),
            json_output=json_output,
            console=console,
        )

    if json_output:
        _emit_json(_failure_log_bundle_plan_payload(result))
        return

    console.print(f"Failure-log bundle plan: {result.artifact_id}")
    console.print(f"- bundle: {result.bundle.bundle_id}")
    console.print(f"- target: {result.relative_path.as_posix()}")
    console.print(f"- entries: {len(result.entries)}")
    console.print("- accepted knowledge merge: not performed")


@artifact_failure_app.command("add-from-bundle")
def artifact_failure_add_from_bundle(
    bundle_path: Path = typer.Option(
        ...,
        "--bundle",
        help="Repository-local WorkerBundle v2 YAML path.",
    ),
    target_artifact_id: str = typer.Option(
        ...,
        "--target-artifact",
        help="Artifact ID that should receive the derived failure-log entries.",
    ),
    repo_root: Path = typer.Option(
        Path("."),
        "--repo-root",
        help="Repository root used for the controlled write.",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Emit deterministic JSON instead of text output.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Validate and report target path without writing files.",
    ),
) -> None:
    """Append or preview WorkerBundle-derived failure-log entries."""
    console = Console(width=120, markup=False)
    try:
        result = DraftWriteService(
            RepoContext(repo_root)
        ).append_failure_log_entries_from_bundle(
            bundle_path,
            target_artifact_id=target_artifact_id,
            dry_run=dry_run,
        )
    except DraftWriteServiceError as exc:
        _exit_with_error(
            exc.to_error_result(),
            json_output=json_output,
            console=console,
        )

    if json_output:
        _emit_json(_failure_log_bundle_write_payload(result))
        return

    action = "would write" if result.write_result.dry_run else "wrote"
    console.print(
        f"{result.write_result.kind}: {action} "
        f"{result.write_result.relative_path.as_posix()}"
    )
    console.print(f"- entries: {len(result.plan.entries)}")
    console.print("- accepted knowledge merge: not performed")


@counterexample_evidence_app.command("validate")
def counterexample_evidence_validate(
    input_json: Path = typer.Option(
        ...,
        "--input-json",
        help="JSON checked counterexample evidence record to validate.",
    ),
    repo_root: Path = typer.Option(
        Path("."),
        "--repo-root",
        help="Repository root used for path policy context.",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Emit deterministic JSON instead of text output.",
    ),
) -> None:
    """Validate a checked counterexample evidence record without writing."""
    console = Console(width=120, markup=False)
    raw = _read_input_json_or_exit(input_json, json_output=json_output)
    try:
        RepoContext(repo_root)
        record = validate_checked_counterexample_evidence_payload(raw)
    except (CheckedCounterexampleEvidenceError, ValidationError, ValueError) as exc:
        _exit_with_error(
            _checked_evidence_error_result(exc),
            json_output=json_output,
            console=console,
        )

    payload = {
        "schema_version": 1,
        "valid": True,
        "accepted_write_performed": False,
        "authority_notice": CHECKED_COUNTEREXAMPLE_AUTHORITY_NOTICE,
        "evidence": record.to_dict(),
    }
    if json_output:
        _emit_json(payload)
        return
    console.print(f"Checked counterexample evidence valid: {record.evidence_id}")
    console.print("- accepted write: not performed")
    console.print(f"- authority: {CHECKED_COUNTEREXAMPLE_AUTHORITY_NOTICE}")


@counterexample_evidence_app.command("stage")
def counterexample_evidence_stage(
    input_json: Path = typer.Option(
        ...,
        "--input-json",
        help="JSON checked counterexample evidence record to stage.",
    ),
    repo_root: Path = typer.Option(
        Path("."),
        "--repo-root",
        help="Repository root used for controlled checked evidence staging.",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Emit deterministic JSON instead of text output.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Validate and report target path without writing files.",
    ),
) -> None:
    """Write or preview a checked counterexample evidence record."""
    console = Console(width=120, markup=False)
    raw = _read_input_json_or_exit(input_json, json_output=json_output)
    try:
        result = stage_checked_counterexample_evidence(
            RepoContext(repo_root),
            raw,
            dry_run=dry_run,
        )
    except (CheckedCounterexampleEvidenceError, ValidationError, ValueError) as exc:
        _exit_with_error(
            _checked_evidence_error_result(exc),
            json_output=json_output,
            console=console,
        )

    if json_output:
        _emit_json(result.to_dict())
        return
    action = "would write" if result.dry_run else "wrote"
    console.print(
        f"{result.to_dict()['kind']}: {action} {result.relative_path.as_posix()}"
    )
    console.print("- accepted knowledge merge: not performed")
    console.print(f"- authority: {CHECKED_COUNTEREXAMPLE_AUTHORITY_NOTICE}")


@counterexample_evidence_app.command("show")
def counterexample_evidence_show(
    evidence: str = typer.Option(
        ...,
        "--evidence",
        help="Checked evidence ID or repository-local YAML path.",
    ),
    repo_root: Path = typer.Option(
        Path("."),
        "--repo-root",
        help="Repository root to inspect.",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Emit deterministic JSON instead of text output.",
    ),
) -> None:
    """Show staged checked counterexample evidence by ID or path."""
    console = Console(width=120, markup=False)
    try:
        result = show_checked_counterexample_evidence(RepoContext(repo_root), evidence)
    except (CheckedCounterexampleEvidenceError, ValidationError, ValueError) as exc:
        _exit_with_error(
            _checked_evidence_error_result(exc),
            json_output=json_output,
            console=console,
        )

    if json_output:
        _emit_json(result.to_dict())
        return
    console.print(f"Checked counterexample evidence: {result.record.evidence_id}")
    console.print(f"- path: {result.relative_path.as_posix()}")
    console.print(f"- result: {result.record.checked_result.value}")
    console.print(f"- authority: {CHECKED_COUNTEREXAMPLE_AUTHORITY_NOTICE}")


@operator_session_app.command("start")
def operator_session_start(
    issue_id: str = typer.Option(
        ...,
        "--issue",
        help="Issue ID this operator session addresses.",
    ),
    policy_mode: str = typer.Option(
        OperatorPolicyMode.PUBLIC_ONLY.value,
        "--policy",
        help="Session policy: public_only or private_research.",
    ),
    operator_label: str = typer.Option(
        "external operator",
        "--operator-label",
        help="Human-readable operator label.",
    ),
    session_id: str | None = typer.Option(
        None,
        "--session-id",
        help="Optional deterministic operator session ID.",
    ),
    repo_root: Path = typer.Option(
        Path("."),
        "--repo-root",
        help="Repository root used for operator-session storage.",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Emit deterministic JSON instead of text output.",
    ),
) -> None:
    """Start a repository-local operator session metadata record."""
    console = Console(width=120, markup=False)
    try:
        result = start_operator_session(
            RepoContext(repo_root),
            issue_id=issue_id,
            policy_mode=OperatorPolicyMode(policy_mode),
            operator_label=operator_label,
            session_id=session_id,
        )
    except (OperatorSessionError, ValidationError, ValueError) as exc:
        _exit_with_error(
            _operator_session_error_result(exc),
            json_output=json_output,
            console=console,
        )
    if json_output:
        _emit_json(result.to_dict())
        return
    console.print(f"Operator session started: {result.session.session_id}")
    console.print(f"- path: {result.relative_path.as_posix()}")
    console.print(f"- authority: {OPERATOR_SESSION_AUTHORITY_NOTICE}")


@operator_session_app.command("show")
def operator_session_show(
    session_id: str = typer.Argument(..., help="Operator session ID."),
    repo_root: Path = typer.Option(
        Path("."),
        "--repo-root",
        help="Repository root used for operator-session storage.",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Emit deterministic JSON instead of text output.",
    ),
) -> None:
    """Show one runtime operator session record."""
    console = Console(width=120, markup=False)
    try:
        result = load_operator_session(RepoContext(repo_root), session_id)
    except (OperatorSessionError, ValidationError, ValueError) as exc:
        _exit_with_error(
            _operator_session_error_result(exc),
            json_output=json_output,
            console=console,
        )
    if json_output:
        _emit_json(result.to_dict())
        return
    console.print(f"Operator session: {result.session.session_id}")
    console.print(f"- status: {result.session.status.value}")
    console.print(f"- authority: {OPERATOR_SESSION_AUTHORITY_NOTICE}")


@operator_session_app.command("append-check")
def operator_session_append_check(
    session_id: str = typer.Argument(..., help="Operator session ID."),
    kind: str = typer.Option(
        ...,
        "--kind",
        help="Check kind: validate, gate, test, or eval.",
    ),
    status: str = typer.Option(
        ...,
        "--status",
        help="Check status: pass, fail, error, or skipped.",
    ),
    summary: str | None = typer.Option(
        None,
        "--summary",
        help="Bounded check summary. Required for non-skipped checks.",
    ),
    report_path: str | None = typer.Option(
        None,
        "--report-path",
        help="Optional repository-local report path.",
    ),
    repo_root: Path = typer.Option(
        Path("."),
        "--repo-root",
        help="Repository root used for operator-session storage.",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Emit deterministic JSON instead of text output.",
    ),
) -> None:
    """Append one external check-status summary to an operator session."""
    console = Console(width=120, markup=False)
    context = RepoContext(repo_root)
    try:
        loaded = load_operator_session(context, session_id)
        check_kind = _parse_operator_session_check_kind(kind)
        check_status = OperatorCheckStatus(status)
        check_summary = _operator_session_check_summary(
            kind=check_kind,
            status=check_status,
            summary=summary,
        )
        result_record = OperatorCheckResult(
            kind=check_kind,
            status=check_status,
            summary=check_summary,
            report_path=report_path,
            recorded_at=datetime.now(UTC),
        )
        updated = loaded.session.with_check_result(result_record)
        result = write_operator_session(context, updated)
        append_operator_session_event(
            context,
            session_id=updated.session_id,
            event=result_record,
        )
    except (OperatorSessionError, ValidationError, ValueError) as exc:
        _exit_with_error(
            _operator_session_error_result(exc),
            json_output=json_output,
            console=console,
        )
    if json_output:
        _emit_json(result.to_dict())
        return
    console.print(f"Operator session check appended: {result.session.session_id}")
    console.print(f"- checks: {len(result.session.check_results)}")


@operator_session_app.command("append-ref")
def operator_session_append_ref(
    session_id: str = typer.Argument(..., help="Operator session ID."),
    path: str = typer.Option(
        ...,
        "--path",
        help="Repository-local path to reference.",
    ),
    kind: str = typer.Option(
        ...,
        "--kind",
        help="Reference kind: draft, review_context, runtime, or report.",
    ),
    artifact_id: str | None = typer.Option(
        None,
        "--artifact",
        help="Optional artifact ID associated with the reference.",
    ),
    scope: str = typer.Option(
        "unknown",
        "--scope",
        help="Reference scope: public, private, workspace, framework, or unknown.",
    ),
    summary: str | None = typer.Option(
        None,
        "--summary",
        help="Bounded reference summary.",
    ),
    repo_root: Path = typer.Option(
        Path("."),
        "--repo-root",
        help="Repository root used for operator-session storage.",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Emit deterministic JSON instead of text output.",
    ),
) -> None:
    """Append one safe file/artifact reference to an operator session."""
    console = Console(width=120, markup=False)
    context = RepoContext(repo_root)
    try:
        loaded = load_operator_session(context, session_id)
        ref_scope = _parse_operator_session_ref_scope(scope)
        _ensure_operator_session_ref_allowed(
            session=loaded.session,
            path=path,
            scope=ref_scope,
        )
        ref = OperatorArtifactRef(
            kind=_parse_operator_session_ref_kind(kind),
            path=path,
            artifact_id=artifact_id,
            summary=summary,
            scope=ref_scope,
        )
        updated = loaded.session.with_artifact_ref(ref)
        result = write_operator_session(context, updated)
        append_operator_session_event(
            context,
            session_id=updated.session_id,
            event=ref,
        )
    except (OperatorSessionError, ValidationError, ValueError) as exc:
        _exit_with_error(
            _operator_session_error_result(exc),
            json_output=json_output,
            console=console,
        )
    if json_output:
        _emit_json(result.to_dict())
        return
    console.print(f"Operator session reference appended: {result.session.session_id}")
    console.print(f"- references: {len(result.session.artifact_refs)}")


@operator_session_app.command("finalize")
def operator_session_finalize(
    session_id: str = typer.Argument(..., help="Operator session ID."),
    repo_root: Path = typer.Option(
        Path("."),
        "--repo-root",
        help="Repository root used for operator-session storage.",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Emit deterministic JSON instead of text output.",
    ),
) -> None:
    """Finalize an operator session metadata record."""
    console = Console(width=120, markup=False)
    context = RepoContext(repo_root)
    try:
        loaded = load_operator_session(context, session_id)
        updated = loaded.session.finalize(
            now=datetime.now(UTC),
            status=OperatorSessionStatus.FINALIZED,
        )
        result = write_operator_session(context, updated)
    except (OperatorSessionError, ValidationError, ValueError) as exc:
        _exit_with_error(
            _operator_session_error_result(exc),
            json_output=json_output,
            console=console,
        )
    if json_output:
        _emit_json(result.to_dict())
        return
    console.print(f"Operator session finalized: {result.session.session_id}")
    console.print(f"- authority: {OPERATOR_SESSION_AUTHORITY_NOTICE}")


@operator_session_app.command("scan")
def operator_session_scan(
    session_id: str = typer.Argument(..., help="Operator session ID."),
    repo_root: Path = typer.Option(
        Path("."),
        "--repo-root",
        help="Repository root used for operator-session storage.",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Emit deterministic JSON instead of text output.",
    ),
) -> None:
    """Scan one operator session for leaks before handoff."""
    console = Console(width=120, markup=False)
    try:
        result = scan_operator_session(RepoContext(repo_root), session_id)
    except (OperatorSessionError, ValidationError, ValueError) as exc:
        _exit_with_error(
            _operator_session_error_result(exc),
            json_output=json_output,
            console=console,
        )
    if json_output:
        _emit_json(result.to_dict())
        if result.handoff_blocked:
            raise typer.Exit(1)
        return
    console.print(f"Operator session scan: {result.session_id}")
    console.print(f"- findings: {result.finding_count}")
    console.print(f"- blockers: {result.blocking_finding_count}")
    console.print(f"- report: {result.report_path.as_posix()}")
    console.print(f"- authority: {OPERATOR_SESSION_AUTHORITY_NOTICE}")
    if result.handoff_blocked:
        raise typer.Exit(1)


@operator_handoff_app.command("build")
def operator_handoff_build(
    session_id: str = typer.Option(
        ...,
        "--session",
        help="Finalized operator session ID to summarize.",
    ),
    repo_root: Path = typer.Option(
        Path("."),
        "--repo-root",
        help="Repository root used for operator handoff storage.",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Emit deterministic JSON instead of text output.",
    ),
) -> None:
    """Build a runtime review handoff bundle from one finalized session."""
    console = Console(width=120, markup=False)
    try:
        result = build_operator_handoff(
            RepoContext(repo_root),
            session_id=session_id,
        )
    except (OperatorSessionError, ValidationError, ValueError) as exc:
        _exit_with_error(
            _operator_session_error_result(exc),
            json_output=json_output,
            console=console,
        )
    if json_output:
        _emit_json(result.to_dict())
        return
    console.print(f"Operator handoff built: {result.handoff.handoff_id}")
    console.print(f"- session: {result.handoff.session_id}")
    console.print(f"- path: {result.relative_path.as_posix()}")
    console.print(f"- authority: {OPERATOR_SESSION_AUTHORITY_NOTICE}")


@operator_handoff_app.command("show")
def operator_handoff_show(
    handoff_id: str = typer.Argument(..., help="Operator handoff ID."),
    repo_root: Path = typer.Option(
        Path("."),
        "--repo-root",
        help="Repository root used for operator handoff storage.",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Emit deterministic JSON instead of text output.",
    ),
) -> None:
    """Show one runtime operator handoff bundle."""
    console = Console(width=120, markup=False)
    try:
        result = load_operator_handoff(RepoContext(repo_root), handoff_id)
    except (OperatorSessionError, ValidationError, ValueError) as exc:
        _exit_with_error(
            _operator_session_error_result(exc),
            json_output=json_output,
            console=console,
        )
    if json_output:
        _emit_json(result.to_dict())
        return
    console.print(f"Operator handoff: {result.handoff.handoff_id}")
    console.print(f"- session: {result.handoff.session_id}")
    console.print(f"- path: {result.relative_path.as_posix()}")
    console.print(f"- authority: {OPERATOR_SESSION_AUTHORITY_NOTICE}")


@operator_handoff_app.command("export")
def operator_handoff_export(
    handoff_id: str = typer.Option(
        ...,
        "--handoff",
        help="Operator handoff ID to export as review context.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Report the export target without writing.",
    ),
    repo_root: Path = typer.Option(
        Path("."),
        "--repo-root",
        help="Repository root used for operator handoff export.",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Emit deterministic JSON instead of text output.",
    ),
) -> None:
    """Export one handoff bundle as explicit review-context YAML."""
    console = Console(width=120, markup=False)
    try:
        result = export_operator_handoff(
            RepoContext(repo_root),
            handoff_id=handoff_id,
            dry_run=dry_run,
        )
    except (OperatorSessionError, ValidationError, ValueError) as exc:
        _exit_with_error(
            _operator_session_error_result(exc),
            json_output=json_output,
            console=console,
        )
    if json_output:
        _emit_json(result.to_dict())
        return
    action = "dry-run" if result.dry_run else "written"
    console.print(f"Operator handoff export {action}: {result.handoff_id}")
    console.print(f"- target: {result.target_path}")
    console.print(f"- authority: {OPERATOR_SESSION_AUTHORITY_NOTICE}")


@run_app.command("start")
def research_run_start(
    issue_id: str = typer.Option(
        ...,
        "--issue",
        help="Issue ID this research run addresses.",
    ),
    operator: str = typer.Option(
        ...,
        "--operator",
        help="Operator kind, usually external.",
    ),
    operator_label: str = typer.Option(
        "external operator",
        "--operator-label",
        help="Human-readable operator label.",
    ),
    run_id: str | None = typer.Option(
        None,
        "--run-id",
        help="Optional deterministic run ID.",
    ),
    repo_root: Path = typer.Option(
        Path("."),
        "--repo-root",
        help="Repository root used for run storage.",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Emit deterministic JSON instead of text output.",
    ),
) -> None:
    """Start a repository-local research run record."""
    console = Console(width=120, markup=False)
    try:
        result = start_research_run(
            RepoContext(repo_root),
            issue_id=issue_id,
            operator_kind=operator,
            operator_label=operator_label,
            run_id=run_id,
        )
    except (ResearchRunError, ValidationError, ValueError) as exc:
        _exit_with_error(
            _research_run_error_result(exc),
            json_output=json_output,
            console=console,
        )
    if json_output:
        _emit_json(result.to_dict())
        return
    console.print(f"Research run started: {result.record.run_id}")
    console.print(f"- path: {result.relative_path.as_posix()}")
    console.print(f"- authority: {RESEARCH_RUN_AUTHORITY_NOTICE}")


@run_app.command("append-command")
def research_run_append_command(
    run_id: str = typer.Option(..., "--run", help="Research run ID."),
    input_json: Path = typer.Option(
        ...,
        "--input-json",
        help="JSON command record to append.",
    ),
    repo_root: Path = typer.Option(
        Path("."),
        "--repo-root",
        help="Repository root used for run storage.",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Emit deterministic JSON instead of text output.",
    ),
) -> None:
    """Append a command record to an in-progress research run."""
    console = Console(width=120, markup=False)
    raw = _read_input_json_or_exit(input_json, json_output=json_output)
    try:
        result = append_command_to_research_run(
            RepoContext(repo_root),
            run_id=run_id,
            payload=raw,
        )
    except (ResearchRunError, ValidationError, ValueError) as exc:
        _exit_with_error(
            _research_run_error_result(exc),
            json_output=json_output,
            console=console,
        )
    if json_output:
        _emit_json(result.to_dict())
        return
    console.print(f"Research run command appended: {result.record.run_id}")
    console.print(f"- commands: {len(result.record.commands)}")


@run_app.command("append-artifact")
def research_run_append_artifact(
    run_id: str = typer.Option(..., "--run", help="Research run ID."),
    artifact_id: str = typer.Option(
        ...,
        "--artifact",
        help="Artifact ID read or touched during the run.",
    ),
    mode: str = typer.Option(
        "read",
        "--mode",
        help="Artifact relation: read or touched.",
    ),
    repo_root: Path = typer.Option(
        Path("."),
        "--repo-root",
        help="Repository root used for run storage.",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Emit deterministic JSON instead of text output.",
    ),
) -> None:
    """Append an artifact read/touched marker to a research run."""
    console = Console(width=120, markup=False)
    try:
        result = append_artifact_to_research_run(
            RepoContext(repo_root),
            run_id=run_id,
            artifact_id=artifact_id,
            mode=mode,
        )
    except (ResearchRunError, ValidationError, ValueError) as exc:
        _exit_with_error(
            _research_run_error_result(exc),
            json_output=json_output,
            console=console,
        )
    if json_output:
        _emit_json(result.to_dict())
        return
    console.print(f"Research run artifact appended: {result.record.run_id}")


@run_app.command("append-output")
def research_run_append_output(
    run_id: str = typer.Option(..., "--run", help="Research run ID."),
    input_json: Path = typer.Option(
        ...,
        "--input-json",
        help="JSON output/reference record to append.",
    ),
    repo_root: Path = typer.Option(
        Path("."),
        "--repo-root",
        help="Repository root used for run storage.",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Emit deterministic JSON instead of text output.",
    ),
) -> None:
    """Append an output/reference record to an in-progress research run."""
    console = Console(width=120, markup=False)
    raw = _read_input_json_or_exit(input_json, json_output=json_output)
    try:
        result = append_output_to_research_run(
            RepoContext(repo_root),
            run_id=run_id,
            payload=raw,
        )
    except (ResearchRunError, ValidationError, ValueError) as exc:
        _exit_with_error(
            _research_run_error_result(exc),
            json_output=json_output,
            console=console,
        )
    if json_output:
        _emit_json(result.to_dict())
        return
    console.print(f"Research run output appended: {result.record.run_id}")


@run_app.command("finalize")
def research_run_finalize(
    run_id: str = typer.Option(..., "--run", help="Research run ID."),
    status: str = typer.Option(..., "--status", help="Terminal run status."),
    stop_reason: str = typer.Option(
        ...,
        "--stop-reason",
        help="Why the run stopped.",
    ),
    repo_root: Path = typer.Option(
        Path("."),
        "--repo-root",
        help="Repository root used for run storage.",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Emit deterministic JSON instead of text output.",
    ),
) -> None:
    """Finalize a research run with a terminal status."""
    console = Console(width=120, markup=False)
    try:
        result = finalize_research_run(
            RepoContext(repo_root),
            run_id=run_id,
            status=status,
            stop_reason=stop_reason,
        )
    except (ResearchRunError, ValidationError, ValueError) as exc:
        _exit_with_error(
            _research_run_error_result(exc),
            json_output=json_output,
            console=console,
        )
    if json_output:
        _emit_json(result.to_dict())
        return
    console.print(f"Research run finalized: {result.record.run_id}")
    console.print(f"- status: {result.record.status.value}")


@run_app.command("show")
def research_run_show(
    run_id: str = typer.Argument(..., help="Research run ID."),
    repo_root: Path = typer.Option(
        Path("."),
        "--repo-root",
        help="Repository root used for run storage.",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Emit deterministic JSON instead of text output.",
    ),
) -> None:
    """Show one runtime research run record."""
    console = Console(width=120, markup=False)
    try:
        result = load_research_run(RepoContext(repo_root), run_id)
    except (ResearchRunError, ValidationError, ValueError) as exc:
        _exit_with_error(
            _research_run_error_result(exc),
            json_output=json_output,
            console=console,
        )
    if json_output:
        _emit_json(result.to_dict())
        return
    console.print(f"Research run: {result.record.run_id}")
    console.print(f"- status: {result.record.status.value}")
    console.print(f"- authority: {RESEARCH_RUN_AUTHORITY_NOTICE}")


@run_app.command("evidence-report")
def research_run_evidence_report(
    run_id: str = typer.Option(..., "--run", help="Research run ID."),
    repo_root: Path = typer.Option(
        Path("."),
        "--repo-root",
        help="Repository root used for run storage.",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Emit deterministic JSON instead of text output.",
    ),
) -> None:
    """Show read-only evidence counts for a research run."""
    console = Console(width=120, markup=False)
    try:
        result = load_research_run(RepoContext(repo_root), run_id)
    except (ResearchRunError, ValidationError, ValueError) as exc:
        _exit_with_error(
            _research_run_error_result(exc),
            json_output=json_output,
            console=console,
        )
    payload = build_research_run_evidence_report(result.record)
    if json_output:
        _emit_json(payload)
        return
    console.print(f"Research run evidence report: {result.record.run_id}")
    console.print(f"- commands: {payload['command_count']}")
    console.print(f"- authority: {RESEARCH_RUN_AUTHORITY_NOTICE}")


@run_app.command("export-review")
def research_run_export_review(
    run_id: str = typer.Option(..., "--run", help="Research run ID."),
    repo_root: Path = typer.Option(
        Path("."),
        "--repo-root",
        help="Repository root used for run storage.",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Emit deterministic JSON instead of text output.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Preview review export path without writing.",
    ),
) -> None:
    """Export a runtime research run into review-controlled YAML."""
    console = Console(width=120, markup=False)
    try:
        result = export_research_run_review(
            RepoContext(repo_root),
            run_id=run_id,
            dry_run=dry_run,
        )
    except (ResearchRunError, ValidationError, ValueError) as exc:
        _exit_with_error(
            _research_run_error_result(exc),
            json_output=json_output,
            console=console,
        )
    if json_output:
        _emit_json(result.to_dict())
        return
    action = "would write" if result.dry_run else "wrote"
    console.print(
        f"Research run review export: {action} {result.relative_path.as_posix()}"
    )


@run_app.command("replay-plan")
def research_run_replay_plan(
    run_id: str = typer.Option(..., "--run", help="Research run ID."),
    repo_root: Path = typer.Option(
        Path("."),
        "--repo-root",
        help="Repository root used for run storage.",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Emit deterministic JSON instead of text output.",
    ),
) -> None:
    """Show a read-only replay plan for recorded commands."""
    console = Console(width=120, markup=False)
    try:
        result = load_research_run(RepoContext(repo_root), run_id)
    except (ResearchRunError, ValidationError, ValueError) as exc:
        _exit_with_error(
            _research_run_error_result(exc),
            json_output=json_output,
            console=console,
        )
    payload = build_replay_plan(result.record)
    if json_output:
        _emit_json(payload)
        return
    console.print(f"Research run replay plan: {result.record.run_id}")
    console.print("- read-only: true")


@strategy_app.command("plan")
def strategy_plan(
    issue_id: str = typer.Option(
        ...,
        "--issue",
        help="Issue ID to plan for.",
    ),
    from_context: Path | None = typer.Option(
        None,
        "--from-context",
        help="Repository-local context-pack directory used as planner input.",
    ),
    repo_root: Path = typer.Option(
        Path("."),
        "--repo-root",
        help="Repository root used for strategy storage.",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Emit deterministic JSON instead of text output.",
    ),
) -> None:
    """Build and persist a deterministic strategy plan for one issue."""
    console = Console(width=120, markup=False)
    try:
        context = RepoContext(repo_root)
        built = build_strategy_plan(context, issue_id)
        plan = (
            attach_context_reference(context, built.plan, from_context)
            if from_context is not None
            else built.plan
        )
        result = write_strategy_plan(context, plan)
    except (StrategyError, ValidationError, ValueError, LoadError) as exc:
        _exit_with_error(
            _strategy_error_result(exc),
            json_output=json_output,
            console=console,
        )
    if json_output:
        _emit_json(result.to_dict())
        return
    console.print(f"Strategy plan: {result.plan.plan_id}")
    console.print(f"- path: {result.relative_path.as_posix()}")
    console.print("- accepted write: not performed")
    console.print(f"- authority: {STRATEGY_AUTHORITY_NOTICE}")


@strategy_app.command("update-from-run")
def strategy_update_from_run(
    plan_id: str = typer.Option(..., "--plan", help="Strategy plan ID."),
    run_id: str = typer.Option(..., "--run", help="Research run ID."),
    repo_root: Path = typer.Option(
        Path("."),
        "--repo-root",
        help="Repository root used for strategy storage.",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Emit deterministic JSON instead of text output.",
    ),
) -> None:
    """Update a strategy plan from research-run provenance."""
    console = Console(width=120, markup=False)
    try:
        result = update_strategy_plan_from_run(
            RepoContext(repo_root),
            plan_id=plan_id,
            run_id=run_id,
        )
    except (StrategyError, ValidationError, ValueError) as exc:
        _exit_with_error(
            _strategy_error_result(exc),
            json_output=json_output,
            console=console,
        )
    if json_output:
        _emit_json(result.to_dict())
        return
    console.print(f"Strategy plan updated: {result.plan.plan_id}")
    console.print(f"- run: {result.run_id}")
    console.print("- accepted write: not performed")


@strategy_app.command("export-review")
def strategy_export_review(
    plan_id: str = typer.Option(..., "--plan", help="Strategy plan ID."),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Show target review export without writing it.",
    ),
    repo_root: Path = typer.Option(
        Path("."),
        "--repo-root",
        help="Repository root used for strategy storage.",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Emit deterministic JSON instead of text output.",
    ),
) -> None:
    """Export strategy guidance to non-authoritative review context."""
    console = Console(width=120, markup=False)
    try:
        result = export_strategy_review(
            RepoContext(repo_root),
            plan_id=plan_id,
            dry_run=dry_run,
        )
    except (StrategyError, ValidationError, ValueError) as exc:
        _exit_with_error(
            _strategy_error_result(exc),
            json_output=json_output,
            console=console,
        )
    if json_output:
        _emit_json(result.to_dict())
        return
    action = "would export" if result.dry_run else "exported"
    console.print(f"Strategy review {action}: {result.relative_path.as_posix()}")
    console.print("- accepted write: not performed")


@strategy_app.command("show")
def strategy_show(
    plan_id: str = typer.Argument(..., help="Strategy plan ID."),
    repo_root: Path = typer.Option(
        Path("."),
        "--repo-root",
        help="Repository root used for strategy storage.",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Emit deterministic JSON instead of text output.",
    ),
) -> None:
    """Show one runtime strategy plan."""
    console = Console(width=120, markup=False)
    try:
        result = load_strategy_plan(RepoContext(repo_root), plan_id)
    except (StrategyError, ValidationError, ValueError) as exc:
        _exit_with_error(
            _strategy_error_result(exc),
            json_output=json_output,
            console=console,
        )
    if json_output:
        _emit_json(result.to_dict())
        return
    console.print(f"Strategy plan: {result.plan.plan_id}")
    console.print(f"- issue: {result.plan.issue_id}")
    console.print(f"- authority: {STRATEGY_AUTHORITY_NOTICE}")


@strategy_app.command("graph")
def strategy_graph(
    plan_id: str = typer.Argument(..., help="Strategy plan ID."),
    repo_root: Path = typer.Option(
        Path("."),
        "--repo-root",
        help="Repository root used for strategy storage.",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Emit deterministic JSON instead of text output.",
    ),
) -> None:
    """Show the task graph for one runtime strategy plan."""
    console = Console(width=120, markup=False)
    try:
        result = load_strategy_plan(RepoContext(repo_root), plan_id)
    except (StrategyError, ValidationError, ValueError) as exc:
        _exit_with_error(
            _strategy_error_result(exc),
            json_output=json_output,
            console=console,
        )
    payload = {
        "schema_version": 1,
        "kind": "strategy_task_graph",
        "plan_id": result.plan.plan_id,
        "accepted_write_performed": False,
        "authority_notice": STRATEGY_AUTHORITY_NOTICE,
        "graph": result.plan.graph.to_dict(),
    }
    if json_output:
        _emit_json(payload)
        return
    console.print(f"Strategy graph: {result.plan.plan_id}")
    console.print(f"- nodes: {len(result.plan.graph.nodes)}")
    console.print(f"- edges: {len(result.plan.graph.edges)}")


@strategy_app.command("next")
def strategy_next(
    plan_id: str = typer.Argument(..., help="Strategy plan ID."),
    repo_root: Path = typer.Option(
        Path("."),
        "--repo-root",
        help="Repository root used for strategy storage.",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Emit deterministic JSON instead of text output.",
    ),
) -> None:
    """Show ranked next steps for one runtime strategy plan."""
    console = Console(width=120, markup=False)
    try:
        result = load_strategy_plan(RepoContext(repo_root), plan_id)
    except (StrategyError, ValidationError, ValueError) as exc:
        _exit_with_error(
            _strategy_error_result(exc),
            json_output=json_output,
            console=console,
        )
    payload = {
        "schema_version": 1,
        "kind": "strategy_next_steps",
        "plan_id": result.plan.plan_id,
        "accepted_write_performed": False,
        "authority_notice": STRATEGY_AUTHORITY_NOTICE,
        "next_steps": [step.to_dict() for step in result.plan.next_steps],
    }
    if json_output:
        _emit_json(payload)
        return
    console.print(f"Strategy next steps: {result.plan.plan_id}")
    for step in result.plan.next_steps:
        command = " ".join(step.command) if step.command else "-"
        console.print(f"- {step.rank}. {step.node_id} | score={step.score} | {command}")


@artifact_app.command("create")
def artifact_create(
    artifact_id: str = typer.Option(..., "--id", help="Globally unique artifact ID."),
    artifact_type: ArtifactType = typer.Option(..., "--type", help="Artifact type."),
    title: str = typer.Option(..., "--title", help="Artifact title."),
    domain: list[str] = typer.Option(
        ...,
        "--domain",
        help="Artifact domain. Repeat for multiple domains.",
    ),
    status: ArtifactStatus = typer.Option(
        ...,
        "--status",
        help="Initial artifact lifecycle status.",
    ),
    statement: str = typer.Option(..., "--statement", help="Artifact statement."),
    author: list[str] | None = typer.Option(
        None,
        "--author",
        help="Artifact author. Repeat for multiple authors.",
    ),
    tag: list[str] | None = typer.Option(
        None,
        "--tag",
        help="Artifact tag. Repeat for multiple tags.",
    ),
    depends_on: list[str] | None = typer.Option(
        None,
        "--depends-on",
        help="Artifact dependency ID. Repeat for multiple dependencies.",
    ),
    supersedes: list[str] | None = typer.Option(
        None,
        "--supersedes",
        help="Superseded artifact ID. Repeat for multiple IDs.",
    ),
    created_at: str | None = typer.Option(
        None,
        "--created-at",
        help="UTC timestamp for created_at/updated_at; defaults to current UTC.",
    ),
    repo_root: Path = typer.Option(
        Path("."),
        "--repo-root",
        help="Repository root used for artifact creation.",
    ),
) -> None:
    """Create a deterministic artifact YAML record in the lifecycle tree."""
    console = Console(width=120, markup=False)
    try:
        result = DraftWriteService(RepoContext(repo_root)).create_artifact(
            artifact_id=artifact_id,
            artifact_type=artifact_type,
            title=title,
            domain=domain,
            status=status,
            statement=statement,
            authors=author or [],
            tags=tag or [],
            depends_on=depends_on or [],
            supersedes=supersedes or [],
            created_at=created_at,
        )
    except DraftWriteServiceError as exc:
        console.print(f"Artifact create failed: {exc}")
        raise typer.Exit(code=1) from None

    artifact = result.artifact
    relative_path = result.relative_path
    console.print(f"Artifact created: {relative_path.as_posix()}")
    console.print(f"- id: {artifact.id}")
    console.print(f"- status: {artifact.status.value}")


@draft_app.command("write-artifact")
def draft_write_artifact(
    input_json: Path = typer.Option(
        ...,
        "--input-json",
        help="JSON request for a draft artifact write.",
    ),
    repo_root: Path = typer.Option(
        Path("."),
        "--repo-root",
        help="Repository root used for controlled draft writes.",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Emit deterministic JSON instead of text output.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Validate and report target paths without writing files.",
    ),
) -> None:
    """Write or preview a controlled draft artifact request."""
    console = Console(width=120, markup=False)
    raw = _read_input_json_or_exit(input_json, json_output=json_output)
    if str(raw.get("status", "")).strip() == ArtifactStatus.ACCEPTED.value:
        _exit_with_error(
            ErrorResult(
                code="accepted_write_forbidden",
                message="draft write-artifact cannot target accepted knowledge",
                remediation=(
                    "Use draft status and the explicit promotion workflow after "
                    "review and gates."
                ),
                blocking=True,
            ),
            json_output=json_output,
            console=console,
        )

    try:
        request = DraftArtifactWriteRequest.model_validate(raw)
        result = DraftWriteService(RepoContext(repo_root)).write_artifact_request(
            request,
            dry_run=dry_run,
        )
    except (DraftWriteServiceError, ValidationError) as exc:
        _exit_with_error(
            _exception_to_error_result(exc, default_code="draft_write_failed"),
            json_output=json_output,
            console=console,
        )

    _emit_controlled_write(result, json_output=json_output, console=console)


@draft_app.command("write-source-note")
def draft_write_source_note(
    input_json: Path = typer.Option(
        ...,
        "--input-json",
        help="JSON request for a staged source note.",
    ),
    repo_root: Path = typer.Option(
        Path("."),
        "--repo-root",
        help="Repository root used for controlled source-note writes.",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Emit deterministic JSON instead of text output.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Validate and report target paths without writing files.",
    ),
) -> None:
    """Write or preview a staged draft source note."""
    console = Console(width=120, markup=False)
    raw = _read_input_json_or_exit(input_json, json_output=json_output)
    try:
        result = DraftWriteService(RepoContext(repo_root)).write_source_note(
            raw,
            dry_run=dry_run,
        )
    except DraftWriteServiceError as exc:
        _exit_with_error(
            exc.to_error_result(),
            json_output=json_output,
            console=console,
        )

    _emit_controlled_write(result, json_output=json_output, console=console)


@review_app.command("request")
def review_request(
    input_json: Path = typer.Option(
        ...,
        "--input-json",
        help="JSON request for a draft review request record.",
    ),
    repo_root: Path = typer.Option(
        Path("."),
        "--repo-root",
        help="Repository root used for controlled review requests.",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Emit deterministic JSON instead of text output.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Validate and report target paths without writing files.",
    ),
) -> None:
    """Write or preview a draft informational review request."""
    console = Console(width=120, markup=False)
    raw = _read_input_json_or_exit(input_json, json_output=json_output)
    status = str(raw.get("status", "")).strip()
    if status in {"human_reviewed", "accepted"}:
        _exit_with_error(
            ErrorResult(
                code="human_review_forbidden",
                message="review request cannot mark human review complete",
                remediation=(
                    "Use status=draft and decision=informational. Human review "
                    "must be recorded explicitly by a reviewer."
                ),
                blocking=True,
            ),
            json_output=json_output,
            console=console,
        )

    try:
        result = DraftWriteService(RepoContext(repo_root)).write_review_request(
            raw,
            dry_run=dry_run,
        )
    except DraftWriteServiceError as exc:
        _exit_with_error(
            exc.to_error_result(),
            json_output=json_output,
            console=console,
        )

    _emit_controlled_write(result, json_output=json_output, console=console)


@review_app.command("request-from-bundle")
def review_request_from_bundle(
    bundle_path: Path = typer.Option(
        ...,
        "--bundle",
        help="Repository-local WorkerBundle v2 YAML path.",
    ),
    repo_root: Path = typer.Option(
        Path("."),
        "--repo-root",
        help="Repository root used for controlled review requests.",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Emit deterministic JSON instead of text output.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Validate and report target paths without writing files.",
    ),
) -> None:
    """Generate a draft informational review request from a WorkerBundle."""
    console = Console(width=120, markup=False)
    try:
        result = DraftWriteService(
            RepoContext(repo_root)
        ).write_review_request_from_bundle(
            bundle_path,
            dry_run=dry_run,
        )
    except DraftWriteServiceError as exc:
        _exit_with_error(
            exc.to_error_result(),
            json_output=json_output,
            console=console,
        )

    write_result = result.write_result
    if json_output:
        _emit_json(
            {
                "schema_version": 1,
                "kind": write_result.kind,
                "bundle_id": result.bundle.bundle_id,
                "task_id": result.bundle.task_id,
                "review_id": write_result.record_id,
                "path": write_result.relative_path.as_posix(),
                "written_paths": [
                    path.as_posix() for path in write_result.written_paths
                ],
                "dry_run": write_result.dry_run,
                "accepted_write_performed": False,
                "generated_request": dict(result.request),
            }
        )
        return

    console.print(f"Review request staged: {write_result.relative_path.as_posix()}")
    console.print("- accepted knowledge merge: not performed")
    console.print("- human review decision: not performed")


@bundle_app.command("submit")
def bundle_submit(
    input_json: Path = typer.Option(
        ...,
        "--input-json",
        help="JSON request for worker bundle review submission.",
    ),
    repo_root: Path = typer.Option(
        Path("."),
        "--repo-root",
        help="Repository root used for bundle submission.",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Emit deterministic JSON instead of text output.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Validate and report without changing task state.",
    ),
) -> None:
    """Validate a worker bundle for review without promotion."""
    console = Console(width=120, markup=False)
    raw = _read_input_json_or_exit(input_json, json_output=json_output)
    try:
        request = WorkerBundleSubmitRequest.model_validate(raw)
        result = BundleValidationService(RepoContext(repo_root)).submit(
            request,
            dry_run=dry_run,
        )
    except (ServiceError, ValidationError) as exc:
        _exit_with_error(
            _exception_to_error_result(exc, default_code="bundle_submit_failed"),
            json_output=json_output,
            console=console,
        )

    if json_output:
        _emit_model(result)
        return

    console.print(f"Bundle accepted for review: {result.bundle_id}")
    console.print("- accepted knowledge merge: not performed")


@artifact_app.command("move-status")
def artifact_move_status(
    artifact_id: str = typer.Argument(..., help="Artifact ID to move."),
    new_status: ArtifactStatus = typer.Argument(..., help="New lifecycle status."),
    repo_root: Path = typer.Option(
        Path("."),
        "--repo-root",
        help="Repository root used for status movement.",
    ),
) -> None:
    """Move an artifact through a safe lifecycle status transition."""
    console = Console(width=120, markup=False)
    try:
        old_status, old_path, new_path = _move_artifact_status(
            context=RepoContext(repo_root),
            artifact_id=artifact_id,
            new_status=new_status,
        )
    except ArtifactLifecycleError as exc:
        console.print(f"Artifact move-status failed: {exc}")
        raise typer.Exit(code=1) from None

    console.print(
        f"Artifact moved: {artifact_id} | {old_status.value} -> {new_status.value}"
    )
    console.print(f"- from: {old_path.as_posix()}")
    console.print(f"- to: {new_path.as_posix()}")


@artifact_app.command("promote")
def artifact_promote(
    artifact_id: str = typer.Argument(..., help="Artifact ID to promote."),
    repo_root: Path = typer.Option(
        Path("."),
        "--repo-root",
        help="Repository root used for accepted promotion.",
    ),
) -> None:
    """Promote an eligible lifecycle artifact into accepted knowledge."""
    console = Console(width=120, markup=False)
    try:
        old_status, old_path, new_path = _promote_artifact(
            context=RepoContext(repo_root),
            artifact_id=artifact_id,
        )
    except ArtifactLifecycleError as exc:
        console.print(f"Artifact promote failed: {exc}")
        raise typer.Exit(code=1) from None

    console.print(f"Artifact promoted: {artifact_id} | {old_status.value} -> accepted")
    console.print(f"- from: {old_path.as_posix()}")
    console.print(f"- to: {new_path.as_posix()}")


@promotion_app.command("readiness")
def promotion_readiness(
    artifact_id: str | None = typer.Option(
        None,
        "--artifact",
        help="Artifact ID to evaluate for promotion readiness.",
    ),
    issue_id: str | None = typer.Option(
        None,
        "--issue",
        help="Issue ID whose related_artifacts should be evaluated.",
    ),
    repo_root: Path = typer.Option(
        Path("."),
        "--repo-root",
        help="Repository root to inspect.",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Emit deterministic JSON instead of text output.",
    ),
) -> None:
    """Report promotion readiness without promoting or writing accepted artifacts."""
    console = Console(width=120, markup=False)
    if (artifact_id is None) == (issue_id is None):
        payload = {
            "schema_version": 1,
            "code": "invalid_promotion_readiness_target",
            "message": "provide exactly one of --artifact or --issue",
            "blocking": True,
        }
        if json_output:
            _emit_json(payload)
        else:
            console.print(
                "Promotion readiness failed: provide exactly one of "
                "--artifact or --issue"
            )
        raise typer.Exit(code=1) from None

    try:
        report = build_promotion_readiness_report(
            RepoContext(repo_root),
            artifact_id=artifact_id,
            issue_id=issue_id,
        )
    except (ValueError, WorkspaceConfigError) as exc:
        payload = {
            "schema_version": 1,
            "code": "promotion_readiness_failed",
            "message": str(exc),
            "blocking": True,
        }
        if json_output:
            _emit_json(payload)
        else:
            console.print(f"Promotion readiness failed: {exc}")
        raise typer.Exit(code=1) from None

    if json_output:
        _emit_json(report.to_dict())
    else:
        _print_promotion_readiness_report(console, report.to_dict())

    if not report.ready:
        raise typer.Exit(code=1)


@index_app.command("rebuild")
def index_rebuild(
    repo_root: Path = typer.Option(
        Path("."),
        "--repo-root",
        help="Repository root to index.",
    ),
) -> None:
    """Rebuild deterministic SQLite and manifest index outputs."""
    console = Console(width=120)
    try:
        result = rebuild_index(RepoContext(repo_root))
    except Exception as exc:
        console.print(f"[bold red]Index rebuild failed[/bold red]: {exc}")
        raise typer.Exit(code=1) from None

    console.print(
        "[bold green]Index rebuilt[/bold green]: "
        f"{result.sqlite_path} and {result.manifest_path} "
        f"({result.artifact_count} artifact(s), {result.edge_count} edge(s))."
    )


@ingest_app.command("convert")
def ingest_convert(
    path: Path = typer.Argument(..., help="Repository-local source file to convert."),
    out_dir: Path = typer.Option(
        Path(".cosheaf/ingest"),
        "--out",
        help="Repository-local staging directory for Markdown and metadata output.",
    ),
    metadata_json: bool = typer.Option(
        False,
        "--metadata-json",
        help="Emit deterministic provenance metadata JSON.",
    ),
    repo_root: Path = typer.Option(
        Path("."),
        "--repo-root",
        help="Repository root used to resolve source and output paths.",
    ),
) -> None:
    """Convert a local source file into staged Markdown with provenance."""
    console = Console(width=120, markup=False)
    try:
        result = MarkItDownIngestAdapter().convert(
            RepoContext(repo_root),
            path,
            out_dir=out_dir,
        )
    except IngestError as exc:
        console.print(f"Ingest failed: {exc}")
        raise typer.Exit(code=1) from None

    if metadata_json:
        typer.echo(result.to_json(), nl=False)
    else:
        console.print(f"Ingest status: {result.status}")
        if result.output_path is not None:
            console.print(f"- output: {result.output_path}")
        if result.metadata_path is not None:
            console.print(f"- metadata: {result.metadata_path}")
        if result.message:
            console.print(f"- note: {result.message}")

    if result.status == "unavailable":
        raise typer.Exit(code=1)


@graph_app.command("show")
def graph_show(
    repo_root: Path = typer.Option(
        Path("."),
        "--repo-root",
        help="Repository root to inspect.",
    ),
) -> None:
    """Show the directed artifact dependency graph."""
    console = Console(width=120)
    try:
        records = tuple(load_artifacts(RepoContext(repo_root)))
        graph = build_dependency_graph(records)
    except Exception as exc:
        console.print(f"[bold red]Graph load failed[/bold red]: {exc}")
        raise typer.Exit(code=1) from None

    _print_dependency_graph(console, graph)


@memory_app.command("cards")
def memory_cards(
    repo_root: Path = typer.Option(
        Path("."),
        "--repo-root",
        help="Repository root to inspect.",
    ),
    issue: str | None = typer.Option(
        None,
        "--issue",
        help="Optional issue ID whose direct related artifacts should be shown.",
    ),
    status: ArtifactCardStatus | None = typer.Option(
        None,
        "--status",
        help="Optional artifact-card status filter.",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Emit deterministic JSON instead of text lines.",
    ),
) -> None:
    """Build compact artifact cards from existing repository metadata."""
    console = Console(width=120, markup=False)
    try:
        cards = MemorySearchService(RepoContext(repo_root)).cards(
            issue_id=issue,
            status=status,
        )
    except MemoryCardError as exc:
        if json_output:
            _emit_error(
                ErrorResult(
                    code="memory_cards_failed",
                    message=str(exc),
                    remediation=(
                        "Check the issue ID, status filter, and repository records."
                    ),
                    blocking=True,
                    related_artifact=_valid_related_artifact(issue),
                )
            )
            raise typer.Exit(code=1) from None
        console.print(f"Memory cards failed: {exc}")
        raise typer.Exit(code=1) from None

    if json_output:
        typer.echo(
            json.dumps(
                [card.to_dict() for card in cards],
                ensure_ascii=True,
                indent=2,
            )
        )
        return

    if not cards:
        console.print("No memory cards.")
        return

    for card in cards:
        console.print(
            f"{card.id} | {card.title} | {card.status.value} | "
            f"{card.root_scope.value} | {card.path}"
        )


@memory_app.command("search")
def memory_search(
    query: str = typer.Argument(..., help="Search query."),
    repo_root: Path = typer.Option(
        Path("."),
        "--repo-root",
        help="Repository root to inspect.",
    ),
    issue: str | None = typer.Option(
        None,
        "--issue",
        help="Optional issue ID used as a Personalized PageRank seed.",
    ),
    seed_artifact: list[str] | None = typer.Option(
        None,
        "--seed-artifact",
        help="Explicit artifact ID to seed personalized ranking. Repeatable.",
    ),
    pin_artifact: list[str] | None = typer.Option(
        None,
        "--pin-artifact",
        help="Artifact ID to strongly pin into personalized ranking. Repeatable.",
    ),
    status: ArtifactCardStatus | None = typer.Option(
        None,
        "--status",
        help="Optional artifact-card status filter.",
    ),
    include_refuted: bool = typer.Option(
        False,
        "--include-refuted",
        help="Include refuted cards with an explicit score penalty.",
    ),
    include_obsolete: bool = typer.Option(
        False,
        "--include-obsolete",
        help="Include obsolete or superseded cards with an explicit score penalty.",
    ),
    explain: bool = typer.Option(
        False,
        "--explain",
        help="Print score component breakdowns in text output.",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Emit deterministic JSON retrieval result instead of text lines.",
    ),
) -> None:
    """Search compact artifact cards with deterministic local scoring."""
    console = Console(width=120, markup=False)
    try:
        result = MemorySearchService(RepoContext(repo_root)).search(
            query,
            issue_id=issue,
            status=status,
            seed_artifacts=tuple(seed_artifact or ()),
            pinned_artifacts=tuple(pin_artifact or ()),
            include_refuted=include_refuted,
            include_obsolete=include_obsolete,
        )
    except MemorySearchError as exc:
        if json_output:
            _emit_error(
                ErrorResult(
                    code="memory_search_failed",
                    message=str(exc),
                    remediation=(
                        "Check the query, issue ID, filters, and repository records."
                    ),
                    blocking=True,
                    related_artifact=_valid_related_artifact(issue),
                )
            )
            raise typer.Exit(code=1) from None
        console.print(f"Memory search failed: {exc}")
        raise typer.Exit(code=1) from None

    if json_output:
        typer.echo(result.to_json(), nl=False)
        return

    if not result.cards:
        console.print("No memory search results.")
        return

    for hit in result.cards:
        card = hit.card
        console.print(
            f"{card.id} | score={hit.score_breakdown.total:.6f} | "
            f"{card.title} | {card.status.value} | "
            f"{card.root_scope.value} | {card.path}"
        )
        if explain:
            breakdown = hit.score_breakdown
            console.print(
                "  breakdown: "
                f"retrieval_hybrid={breakdown.retrieval_hybrid:.6f} "
                f"personalized_pagerank={breakdown.personalized_pagerank:.6f} "
                f"global_pagerank={breakdown.global_pagerank:.6f} "
                f"quality_prior={breakdown.quality_prior:.6f} "
                f"freshness={breakdown.freshness:.6f} "
                f"penalty={breakdown.penalty:.6f}"
            )
            for reason in hit.why_relevant:
                console.print(f"  why: {reason}")


@benchmark_app.command("list")
def benchmark_list(
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Emit deterministic JSON instead of text lines.",
    ),
) -> None:
    """List deterministic benchmark suites."""
    result = list_benchmark_suites()
    if json_output:
        _emit_json(result.to_dict())
        return
    console = Console(width=120, markup=False)
    for suite in result.suites:
        components = ", ".join(component.value for component in suite.components)
        console.print(f"{suite.name.value}: {components}")


@benchmark_app.command("run")
def benchmark_run(
    suite: BenchmarkSuiteName = typer.Option(
        ...,
        "--suite",
        help="Benchmark suite to run.",
    ),
    repo_root: Path = typer.Option(
        Path("."),
        "--repo-root",
        help="Repository root to inspect.",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Emit deterministic JSON instead of text lines.",
    ),
) -> None:
    """Run a deterministic benchmark suite and persist its sidecar."""
    console = Console(width=120, markup=False)
    try:
        result = run_benchmark_suite(RepoContext(repo_root), suite)
    except BenchmarkError as exc:
        if json_output:
            _emit_error(
                ErrorResult(
                    code="benchmark_run_failed",
                    message=str(exc),
                    remediation="Inspect benchmark suite names and eval case files.",
                    blocking=True,
                )
            )
            raise typer.Exit(code=1) from None
        console.print(f"Benchmark run failed: {exc}")
        raise typer.Exit(code=1) from None

    if json_output:
        _emit_json(result.to_dict())
        if not result.passed:
            raise typer.Exit(code=1)
        return
    verdict = "pass" if result.passed else "fail"
    console.print(f"Benchmark verdict: {verdict}")
    console.print(f"- run_id: {result.run_id}")
    console.print(f"- suite: {result.suite.value}")
    console.print(f"- pass_count: {result.metrics.pass_count}")
    console.print(f"- fail_count: {result.metrics.fail_count}")
    console.print(f"- skipped_count: {result.metrics.skipped_count}")
    console.print(
        f"- authority_violation_count: {result.metrics.authority_violation_count}"
    )
    console.print(f"- private_leak_count: {result.metrics.private_leak_count}")
    console.print(f"- authority: {result.authority_notice}")
    if not result.passed:
        raise typer.Exit(code=1)


@benchmark_app.command("report")
def benchmark_report(
    run_id: str = typer.Argument(..., help="Benchmark run ID."),
    out: Path = typer.Option(
        ...,
        "--out",
        help="Repository-local .md or .json output path.",
    ),
    repo_root: Path = typer.Option(
        Path("."),
        "--repo-root",
        help="Repository root to inspect.",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Emit deterministic JSON summary instead of text lines.",
    ),
) -> None:
    """Write a static benchmark report from an existing run sidecar."""
    console = Console(width=120, markup=False)
    try:
        result = write_benchmark_report(RepoContext(repo_root), run_id, out)
    except BenchmarkError as exc:
        if json_output:
            _emit_error(
                ErrorResult(
                    code="benchmark_report_failed",
                    message=str(exc),
                    remediation="Run the benchmark first and use a safe output path.",
                    blocking=True,
                    related_artifact=_valid_related_artifact(run_id),
                )
            )
            raise typer.Exit(code=1) from None
        console.print(f"Benchmark report failed: {exc}")
        raise typer.Exit(code=1) from None

    if json_output:
        _emit_json(result.to_dict())
        return
    console.print(f"Benchmark report written: {result.out_path}")
    console.print(f"- format: {result.report_format}")
    console.print(f"- authority: {result.authority_notice}")


@compare_app.command("workflows")
def compare_workflows_cli(
    before_id: str = typer.Argument(..., help="Earlier workflow ID."),
    after_id: str = typer.Argument(..., help="Later workflow ID."),
    repo_root: Path = typer.Option(
        Path("."),
        "--repo-root",
        help="Repository root to inspect.",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Emit deterministic JSON instead of text lines.",
    ),
) -> None:
    """Compare two workflow records."""
    _emit_compare_or_exit(
        lambda: compare_workflows(RepoContext(repo_root), before_id, after_id),
        json_output=json_output,
        code="workflow_compare_failed",
    )


@compare_app.command("campaigns")
def compare_campaigns_cli(
    before_id: str = typer.Argument(..., help="Earlier campaign ID."),
    after_id: str = typer.Argument(..., help="Later campaign ID."),
    repo_root: Path = typer.Option(
        Path("."),
        "--repo-root",
        help="Repository root to inspect.",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Emit deterministic JSON instead of text lines.",
    ),
) -> None:
    """Compare two campaign records."""
    _emit_compare_or_exit(
        lambda: compare_campaigns(RepoContext(repo_root), before_id, after_id),
        json_output=json_output,
        code="campaign_compare_failed",
    )


@compare_app.command("benchmarks")
def compare_benchmarks_cli(
    before_id: str = typer.Argument(..., help="Earlier benchmark run ID."),
    after_id: str = typer.Argument(..., help="Later benchmark run ID."),
    repo_root: Path = typer.Option(
        Path("."),
        "--repo-root",
        help="Repository root to inspect.",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Emit deterministic JSON instead of text lines.",
    ),
) -> None:
    """Compare two benchmark runs."""
    _emit_compare_or_exit(
        lambda: compare_benchmarks(RepoContext(repo_root), before_id, after_id),
        json_output=json_output,
        code="benchmark_compare_failed",
    )


@report_app.command("workflow")
def report_workflow(
    workflow_id: str = typer.Argument(..., help="Workflow ID."),
    out: Path = typer.Option(
        ...,
        "--out",
        help="Repository-local output directory.",
    ),
    repo_root: Path = typer.Option(
        Path("."),
        "--repo-root",
        help="Repository root to inspect.",
    ),
    public_only: bool = typer.Option(
        False,
        "--public-only",
        help="Reject reports that include private references.",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Emit deterministic JSON instead of text lines.",
    ),
) -> None:
    """Write static Markdown/JSON report files for a workflow."""
    _emit_static_report_or_exit(
        lambda: write_workflow_report(
            RepoContext(repo_root),
            workflow_id,
            out,
            public_only=public_only,
        ),
        json_output=json_output,
        code="workflow_report_failed",
    )


@report_app.command("campaign")
def report_campaign(
    campaign_id: str = typer.Argument(..., help="Campaign ID."),
    out: Path = typer.Option(
        ...,
        "--out",
        help="Repository-local output directory.",
    ),
    repo_root: Path = typer.Option(
        Path("."),
        "--repo-root",
        help="Repository root to inspect.",
    ),
    public_only: bool = typer.Option(
        False,
        "--public-only",
        help="Reject reports that include private references.",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Emit deterministic JSON instead of text lines.",
    ),
) -> None:
    """Write static Markdown/JSON report files for a campaign."""
    _emit_static_report_or_exit(
        lambda: write_campaign_report(
            RepoContext(repo_root),
            campaign_id,
            out,
            public_only=public_only,
        ),
        json_output=json_output,
        code="campaign_report_failed",
    )


@report_app.command("benchmark")
def report_benchmark(
    run_id: str = typer.Argument(..., help="Benchmark run ID."),
    out: Path = typer.Option(
        ...,
        "--out",
        help="Repository-local output directory.",
    ),
    repo_root: Path = typer.Option(
        Path("."),
        "--repo-root",
        help="Repository root to inspect.",
    ),
    public_only: bool = typer.Option(
        False,
        "--public-only",
        help="Keep the static report on aggregate benchmark fields only.",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Emit deterministic JSON instead of text lines.",
    ),
) -> None:
    """Write static Markdown/JSON report files for a benchmark run."""
    _emit_static_report_or_exit(
        lambda: write_benchmark_static_report(
            RepoContext(repo_root),
            run_id,
            out,
            public_only=public_only,
        ),
        json_output=json_output,
        code="benchmark_static_report_failed",
    )


@eval_app.command("campaign")
def eval_campaign(
    repo_root: Path = typer.Option(
        Path("."),
        "--repo-root",
        help="Repository root to inspect.",
    ),
    cases: Path = typer.Option(
        DEFAULT_CAMPAIGN_EVAL_CASES,
        "--cases",
        help="Repository-local YAML campaign eval case file.",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Emit deterministic JSON instead of text summary.",
    ),
) -> None:
    """Run deterministic campaign handoff/boundary regression cases."""
    console = Console(width=120, markup=False)
    try:
        context = RepoContext(repo_root)
        case_path = resolve_campaign_eval_case_path(context, cases)
        suite = load_campaign_eval_suite(case_path)
        report = run_campaign_eval_suite(context, suite)
    except CampaignEvalError as exc:
        console.print(f"Campaign eval failed: {exc}")
        raise typer.Exit(code=1) from None

    if json_output:
        typer.echo(report.to_json(), nl=False)
        return

    verdict = "pass" if report.passed else "fail"
    console.print(f"Campaign eval verdict: {verdict}")
    console.print(f"- cases: {report.case_count}")
    console.print(f"- attempt_count: {report.metrics.attempt_count}")
    console.print(
        f"- unique_direction_count: {report.metrics.unique_direction_count}"
    )
    console.print(f"- repeat_failure_count: {report.metrics.repeat_failure_count}")
    console.print(
        f"- reviewable_draft_count: {report.metrics.reviewable_draft_count}"
    )
    console.print(
        f"- checked_evidence_count: {report.metrics.checked_evidence_count}"
    )
    console.print(f"- gap_count: {report.metrics.gap_count}")
    console.print(f"- unsafe_output_count: {report.metrics.unsafe_output_count}")
    console.print(
        f"- budget_stop_accuracy: {report.metrics.budget_stop_accuracy:.6f}"
    )
    console.print(
        "- operator_contract_validity: "
        f"{report.metrics.operator_contract_validity:.6f}"
    )
    for case in report.cases:
        failures = ",".join(case.failures) if case.failures else "-"
        console.print(f"- {case.id}: passed={str(case.passed).lower()} {failures}")

    if not report.passed:
        raise typer.Exit(code=1)


@eval_app.command("retrieval")
def eval_retrieval(
    repo_root: Path = typer.Option(
        Path("."),
        "--repo-root",
        help="Repository root to inspect.",
    ),
    cases: Path = typer.Option(
        DEFAULT_RETRIEVAL_EVAL_CASES,
        "--cases",
        help="Repository-local YAML retrieval eval case file.",
    ),
    k: int = typer.Option(
        5,
        "--k",
        min=1,
        help="Top-k cutoff used for hit@k.",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Emit deterministic JSON instead of text summary.",
    ),
) -> None:
    """Run deterministic retrieval regression cases."""
    console = Console(width=120, markup=False)
    try:
        context = RepoContext(repo_root)
        case_path = resolve_retrieval_eval_case_path(context, cases)
        suite = load_retrieval_eval_suite(case_path)
        report = run_retrieval_eval_suite(context, suite, k=k)
    except RetrievalEvalError as exc:
        console.print(f"Retrieval eval failed: {exc}")
        raise typer.Exit(code=1) from None

    if json_output:
        typer.echo(report.to_json(), nl=False)
        return

    verdict = "pass" if report.passed else "fail"
    console.print(f"Retrieval eval verdict: {verdict}")
    console.print(f"- cases: {report.case_count}")
    console.print(f"- hit@{k}: {report.metrics.hit_at_k:.6f}")
    console.print(f"- forbidden_hit_count: {report.metrics.forbidden_hit_count}")
    console.print(
        f"- accepted_priority_score: {report.metrics.accepted_priority_score:.6f}"
    )
    console.print(f"- private_leakage_count: {report.metrics.private_leakage_count}")
    for case in report.cases:
        console.print(
            f"- {case.id}: hit@{k}={case.hit_at_k:.6f} "
            f"forbidden={case.forbidden_hit_count} "
            f"private_leakage={case.private_leakage_count} "
            f"returned={','.join(case.returned_artifacts) or '-'}"
        )

    if not report.passed:
        raise typer.Exit(code=1)


@eval_app.command("context")
def eval_context(
    repo_root: Path = typer.Option(
        Path("."),
        "--repo-root",
        help="Repository root to inspect.",
    ),
    cases: Path = typer.Option(
        DEFAULT_CONTEXT_EVAL_CASES,
        "--cases",
        help="Repository-local YAML context eval case file.",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Emit deterministic JSON instead of text summary.",
    ),
) -> None:
    """Run deterministic context-pack regression cases."""
    console = Console(width=120, markup=False)
    try:
        context = RepoContext(repo_root)
        case_path = resolve_context_eval_case_path(context, cases)
        suite = load_context_eval_suite(case_path)
        report = run_context_eval_suite(context, suite)
    except ContextEvalError as exc:
        console.print(f"Context eval failed: {exc}")
        raise typer.Exit(code=1) from None

    if json_output:
        typer.echo(report.to_json(), nl=False)
        return

    verdict = "pass" if report.passed else "fail"
    console.print(f"Context eval verdict: {verdict}")
    console.print(f"- cases: {report.case_count}")
    console.print(f"- max_cards: {report.metrics.max_cards}")
    console.print(f"- max_full_artifacts: {report.metrics.max_full_artifacts}")
    console.print(f"- token_estimate: {report.metrics.token_estimate}")
    console.print(f"- accepted_ratio: {report.metrics.accepted_ratio:.6f}")
    console.print(f"- draft_ratio: {report.metrics.draft_ratio:.6f}")
    console.print(f"- private_leakage_count: {report.metrics.private_leakage_count}")
    console.print(
        f"- required_artifact_hit: {report.metrics.required_artifact_hit:.6f}"
    )
    for case in report.cases:
        failures = ",".join(case.failures) if case.failures else "-"
        console.print(
            f"- {case.id}: cards={case.metrics.max_cards} "
            f"full={case.metrics.max_full_artifacts} "
            f"private_leakage={case.metrics.private_leakage_count} "
            f"required_hit={case.metrics.required_artifact_hit:.6f} "
            f"failures={failures}"
        )

    if not report.passed:
        raise typer.Exit(code=1)


@eval_app.command("checker-crosscheck")
def eval_checker_crosscheck(
    repo_root: Path = typer.Option(
        Path("."),
        "--repo-root",
        help="Repository root to inspect.",
    ),
    cases: Path = typer.Option(
        DEFAULT_CHECKER_CROSSCHECK_EVAL_CASES,
        "--cases",
        help="Repository-local YAML checker/cross-check eval case file.",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Emit deterministic JSON instead of text summary.",
    ),
) -> None:
    """Run deterministic checker/cross-check regression cases."""
    console = Console(width=120, markup=False)
    try:
        context = RepoContext(repo_root)
        case_path = resolve_checker_crosscheck_eval_case_path(context, cases)
        suite = load_checker_crosscheck_eval_suite(case_path)
        report = run_checker_crosscheck_eval_suite(context, suite)
    except CheckerCrossCheckEvalError as exc:
        console.print(f"Checker/cross-check eval failed: {exc}")
        raise typer.Exit(code=1) from None

    if json_output:
        typer.echo(report.to_json(), nl=False)
        return

    verdict = "pass" if report.passed else "fail"
    console.print(f"Checker/cross-check eval verdict: {verdict}")
    console.print(f"- cases: {report.case_count}")
    console.print(
        "- checked_pass_boundary_rate: "
        f"{report.metrics.checked_pass_boundary_rate:.6f}"
    )
    console.print(
        "- failed_checker_detection_rate: "
        f"{report.metrics.failed_checker_detection_rate:.6f}"
    )
    console.print(
        "- authority_overclaim_rejection_rate: "
        f"{report.metrics.authority_overclaim_rejection_rate:.6f}"
    )
    console.print(
        "- private_leak_rejection_rate: "
        f"{report.metrics.private_leak_rejection_rate:.6f}"
    )
    console.print(f"- skipped_not_pass_count: {report.metrics.skipped_not_pass_count}")
    console.print(
        "- inconclusive_not_pass_count: "
        f"{report.metrics.inconclusive_not_pass_count}"
    )
    console.print(
        "- accepted_write_violation_count: "
        f"{report.metrics.accepted_write_violation_count}"
    )
    for case in report.cases:
        failures = ",".join(case.failures) if case.failures else "-"
        console.print(f"- {case.id}: passed={str(case.passed).lower()} {failures}")

    if not report.passed:
        raise typer.Exit(code=1)


@eval_app.command("checked-evidence-run-loop")
def eval_checked_evidence_run_loop(
    repo_root: Path = typer.Option(
        Path("."),
        "--repo-root",
        help="Repository root to inspect.",
    ),
    cases: Path = typer.Option(
        DEFAULT_CHECKED_EVIDENCE_RUN_LOOP_EVAL_CASES,
        "--cases",
        help="Repository-local YAML checked-evidence eval case file.",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Emit deterministic JSON instead of text summary.",
    ),
) -> None:
    """Run deterministic checked-evidence run-loop regression cases."""
    console = Console(width=120, markup=False)
    try:
        context = RepoContext(repo_root)
        case_path = resolve_checked_evidence_run_loop_eval_case_path(
            context,
            cases,
        )
        suite = load_checked_evidence_run_loop_eval_suite(case_path)
        report = run_checked_evidence_run_loop_eval_suite(context, suite)
    except CheckedEvidenceRunLoopEvalError as exc:
        console.print(f"Checked-evidence run-loop eval failed: {exc}")
        raise typer.Exit(code=1) from None

    if json_output:
        typer.echo(report.to_json(), nl=False)
        return

    verdict = "pass" if report.passed else "fail"
    console.print(f"Checked-evidence run-loop eval verdict: {verdict}")
    console.print(f"- cases: {report.case_count}")
    console.print(
        "- candidate_checked_separation_accuracy: "
        f"{report.metrics.candidate_checked_separation_accuracy:.6f}"
    )
    console.print(
        "- checked_refutes_support_count: "
        f"{report.metrics.checked_refutes_support_count}"
    )
    console.print(f"- skipped_not_pass_count: {report.metrics.skipped_not_pass_count}")
    console.print(
        "- accepted_write_violation_count: "
        f"{report.metrics.accepted_write_violation_count}"
    )
    for case in report.cases:
        failures = ",".join(case.failures) if case.failures else "-"
        console.print(
            f"- {case.id}: result={case.checked_result or '-'} "
            f"candidate_only={str(case.candidate_review_only).lower()} "
            f"checked_refutation={str(case.checked_refutation).lower()} "
            f"support={str(case.support_present).lower()} failures={failures}"
        )

    if not report.passed:
        raise typer.Exit(code=1)


@eval_app.command("research-run-loop")
def eval_research_run_loop(
    repo_root: Path = typer.Option(
        Path("."),
        "--repo-root",
        help="Repository root to inspect.",
    ),
    cases: Path = typer.Option(
        DEFAULT_RESEARCH_RUN_LOOP_EVAL_CASES,
        "--cases",
        help="Repository-local YAML research-run eval case file.",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Emit deterministic JSON instead of text summary.",
    ),
) -> None:
    """Run deterministic research-run loop regression cases."""
    console = Console(width=120, markup=False)
    try:
        context = RepoContext(repo_root)
        case_path = resolve_research_run_loop_eval_case_path(context, cases)
        suite = load_research_run_loop_eval_suite(case_path)
        report = run_research_run_loop_eval_suite(context, suite)
    except ResearchRunLoopEvalError as exc:
        console.print(f"Research-run loop eval failed: {exc}")
        raise typer.Exit(code=1) from None

    if json_output:
        typer.echo(report.to_json(), nl=False)
        return

    verdict = "pass" if report.passed else "fail"
    console.print(f"Research-run loop eval verdict: {verdict}")
    console.print(f"- cases: {report.case_count}")
    console.print(
        f"- command_coverage_accuracy: {report.metrics.command_coverage_accuracy:.6f}"
    )
    console.print(f"- skipped_not_pass_count: {report.metrics.skipped_not_pass_count}")
    console.print(
        f"- authority_escalation_count: {report.metrics.authority_escalation_count}"
    )
    for case in report.cases:
        failures = ",".join(case.failures) if case.failures else "-"
        console.print(f"- {case.id}: passed={str(case.passed).lower()} {failures}")

    if not report.passed:
        raise typer.Exit(code=1)


@eval_app.command("research-loop")
def eval_research_loop(
    repo_root: Path = typer.Option(
        Path("."),
        "--repo-root",
        help="Repository root to inspect.",
    ),
    cases: Path = typer.Option(
        DEFAULT_RESEARCH_LOOP_EVAL_CASES,
        "--cases",
        help="Repository-local YAML research-loop eval case file.",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Emit deterministic JSON instead of text summary.",
    ),
) -> None:
    """Run deterministic bounded research-loop regression cases."""
    console = Console(width=120, markup=False)
    try:
        context = RepoContext(repo_root)
        case_path = resolve_research_loop_eval_case_path(context, cases)
        suite = load_research_loop_eval_suite(case_path)
        report = run_research_loop_eval_suite(context, suite)
    except ResearchLoopEvalError as exc:
        console.print(f"Research-loop eval failed: {exc}")
        raise typer.Exit(code=1) from None

    if json_output:
        typer.echo(report.to_json(), nl=False)
        return

    verdict = "pass" if report.passed else "fail"
    console.print(f"Research-loop eval verdict: {verdict}")
    console.print(f"- cases: {report.case_count}")
    console.print(f"- loop_validity_rate: {report.metrics.loop_validity_rate:.6f}")
    console.print(
        "- repeat_failure_detection_rate: "
        f"{report.metrics.repeat_failure_detection_rate:.6f}"
    )
    console.print(
        f"- scanner_blocker_accuracy: {report.metrics.scanner_blocker_accuracy:.6f}"
    )
    console.print(f"- skipped_not_pass_count: {report.metrics.skipped_not_pass_count}")
    for case in report.cases:
        failures = ",".join(case.failures) if case.failures else "-"
        console.print(f"- {case.id}: passed={str(case.passed).lower()} {failures}")

    if not report.passed:
        raise typer.Exit(code=1)


@eval_app.command("reviewable-workflow")
def eval_reviewable_workflow(
    repo_root: Path = typer.Option(
        Path("."),
        "--repo-root",
        help="Repository root to inspect.",
    ),
    cases: Path = typer.Option(
        DEFAULT_REVIEWABLE_WORKFLOW_EVAL_CASES,
        "--cases",
        help="Repository-local YAML reviewable-workflow eval case file.",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Emit deterministic JSON instead of text summary.",
    ),
) -> None:
    """Run deterministic issue-to-reviewable-packet regression cases."""
    console = Console(width=120, markup=False)
    try:
        context = RepoContext(repo_root)
        case_path = resolve_reviewable_workflow_eval_case_path(context, cases)
        suite = load_reviewable_workflow_eval_suite(case_path)
        report = run_reviewable_workflow_eval_suite(context, suite)
    except ReviewableWorkflowEvalError as exc:
        console.print(f"Reviewable-workflow eval failed: {exc}")
        raise typer.Exit(code=1) from None

    if json_output:
        typer.echo(report.to_json(), nl=False)
        return

    verdict = "pass" if report.passed else "fail"
    console.print(f"Reviewable-workflow eval verdict: {verdict}")
    console.print(f"- cases: {report.case_count}")
    console.print(
        "- workflow_validity_rate: "
        f"{report.metrics.workflow_validity_rate:.6f}"
    )
    console.print(
        "- draft_proposal_validity_rate: "
        f"{report.metrics.draft_proposal_validity_rate:.6f}"
    )
    console.print(
        "- handoff_scanner_block_rate: "
        f"{report.metrics.handoff_scanner_block_rate:.6f}"
    )
    console.print(
        "- review_readiness_classification_rate: "
        f"{report.metrics.review_readiness_classification_rate:.6f}"
    )
    console.print(f"- skipped_not_pass_count: {report.metrics.skipped_not_pass_count}")
    for case in report.cases:
        failures = ",".join(case.failures) if case.failures else "-"
        console.print(f"- {case.id}: passed={str(case.passed).lower()} {failures}")

    if not report.passed:
        raise typer.Exit(code=1)


@eval_app.command("strategy-planner")
def eval_strategy_planner(
    repo_root: Path = typer.Option(
        Path("."),
        "--repo-root",
        help="Repository root to inspect.",
    ),
    cases: Path = typer.Option(
        DEFAULT_STRATEGY_PLANNER_EVAL_CASES,
        "--cases",
        help="Repository-local YAML strategy-planner eval case file.",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Emit deterministic JSON instead of text summary.",
    ),
) -> None:
    """Run deterministic strategy-planner boundary regression cases."""
    console = Console(width=120, markup=False)
    try:
        context = RepoContext(repo_root)
        case_path = resolve_strategy_planner_eval_case_path(context, cases)
        suite = load_strategy_planner_eval_suite(case_path)
        report = run_strategy_planner_eval_suite(context, suite)
    except StrategyPlannerEvalError as exc:
        console.print(f"Strategy-planner eval failed: {exc}")
        raise typer.Exit(code=1) from None

    if json_output:
        typer.echo(report.to_json(), nl=False)
        return

    verdict = "pass" if report.passed else "fail"
    console.print(f"Strategy-planner eval verdict: {verdict}")
    console.print(f"- cases: {report.case_count}")
    console.print(
        "- failed_direction_repeat_count: "
        f"{report.metrics.failed_direction_repeat_count}"
    )
    console.print(f"- skipped_not_pass_count: {report.metrics.skipped_not_pass_count}")
    console.print(
        f"- authority_escalation_count: {report.metrics.authority_escalation_count}"
    )
    for case in report.cases:
        failures = ",".join(case.failures) if case.failures else "-"
        console.print(f"- {case.id}: passed={str(case.passed).lower()} {failures}")

    if not report.passed:
        raise typer.Exit(code=1)


@memory_app.command("update-from-workflow")
def memory_update_from_workflow(
    workflow_id: str = typer.Argument(..., help="Workflow ID to learn from."),
    repo_root: Path = typer.Option(
        Path("."),
        "--repo-root",
        help="Repository root to inspect.",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Emit deterministic JSON instead of text lines.",
    ),
) -> None:
    """Update sidecar memory weights from one persisted workflow."""
    console = Console(width=120, markup=False)
    try:
        result = update_memory_from_workflow(RepoContext(repo_root), workflow_id)
    except (MemoryUpdateError, ValidationError, ValueError) as exc:
        if json_output:
            _emit_error(
                ErrorResult(
                    code="memory_update_failed",
                    message=str(exc),
                    remediation="Check the workflow ID and runtime sidecar files.",
                    blocking=True,
                    related_artifact=_valid_related_artifact(workflow_id),
                )
            )
            raise typer.Exit(code=1) from None
        console.print(f"Memory update failed: {exc}")
        raise typer.Exit(code=1) from None

    if json_output:
        _emit_json(result.to_dict())
        return
    console.print(f"Memory update run written: {result.run_path}")
    console.print(f"- updates: {len(result.run.updates)}")
    console.print(f"- weights: {result.weights_path}")
    console.print(f"- authority: {result.authority_notice}")


@memory_app.command("update-from-campaign")
def memory_update_from_campaign(
    campaign_id: str = typer.Argument(..., help="Campaign ID to learn from."),
    repo_root: Path = typer.Option(
        Path("."),
        "--repo-root",
        help="Repository root to inspect.",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Emit deterministic JSON instead of text lines.",
    ),
) -> None:
    """Update sidecar memory weights from one persisted campaign."""
    console = Console(width=120, markup=False)
    try:
        result = update_memory_from_campaign(RepoContext(repo_root), campaign_id)
    except (MemoryUpdateError, ValidationError, ValueError) as exc:
        if json_output:
            _emit_error(
                ErrorResult(
                    code="memory_update_failed",
                    message=str(exc),
                    remediation="Check the campaign ID and runtime sidecar files.",
                    blocking=True,
                    related_artifact=_valid_related_artifact(campaign_id),
                )
            )
            raise typer.Exit(code=1) from None
        console.print(f"Memory update failed: {exc}")
        raise typer.Exit(code=1) from None

    if json_output:
        _emit_json(result.to_dict())
        return
    console.print(f"Memory update run written: {result.run_path}")
    console.print(f"- updates: {len(result.run.updates)}")
    console.print(f"- weights: {result.weights_path}")
    console.print(f"- authority: {result.authority_notice}")


@memory_app.command("explain")
def memory_explain(
    artifact_id: str = typer.Argument(..., help="Artifact ID to explain."),
    repo_root: Path = typer.Option(
        Path("."),
        "--repo-root",
        help="Repository root to inspect.",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Emit deterministic JSON instead of text lines.",
    ),
) -> None:
    """Explain sidecar memory weights touching one artifact."""
    console = Console(width=120, markup=False)
    try:
        result = explain_memory_weight(RepoContext(repo_root), artifact_id)
    except (MemoryUpdateError, ValidationError, ValueError) as exc:
        if json_output:
            _emit_error(
                ErrorResult(
                    code="memory_explain_failed",
                    message=str(exc),
                    remediation="Run `cosheaf memory rebuild` first.",
                    blocking=True,
                    related_artifact=_valid_related_artifact(artifact_id),
                )
            )
            raise typer.Exit(code=1) from None
        console.print(f"Memory explain failed: {exc}")
        raise typer.Exit(code=1) from None

    if json_output:
        _emit_json(result.to_dict())
        return
    if not result.edges:
        console.print(f"No memory weights for {artifact_id}.")
    for edge in result.edges:
        console.print(
            f"{edge.source_id} -> {edge.target_id} | weight={edge.weight:.6f} | "
            f"signals={','.join(edge.signals)}"
        )
    console.print(f"- authority: {result.authority_notice}")


@memory_app.command("rebuild")
def memory_rebuild(
    repo_root: Path = typer.Option(
        Path("."),
        "--repo-root",
        help="Repository root to inspect.",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Emit deterministic JSON instead of text lines.",
    ),
) -> None:
    """Rebuild sidecar memory weights from update-run sidecars."""
    console = Console(width=120, markup=False)
    try:
        result = rebuild_memory_weights(RepoContext(repo_root))
    except (MemoryUpdateError, ValidationError, ValueError) as exc:
        if json_output:
            _emit_error(
                ErrorResult(
                    code="memory_rebuild_failed",
                    message=str(exc),
                    remediation="Inspect .cosheaf/memory/update-runs/*.json.",
                    blocking=True,
                )
            )
            raise typer.Exit(code=1) from None
        console.print(f"Memory rebuild failed: {exc}")
        raise typer.Exit(code=1) from None

    if json_output:
        _emit_json(result.to_dict())
        return
    console.print(f"Memory weights rebuilt: {result.weight_count} edge(s).")
    console.print("- sidecar: .cosheaf/memory/weights.json")
    console.print(f"- authority: {result.authority_notice}")


@memory_graph_app.command("build")
def memory_graph_build(
    repo_root: Path = typer.Option(
        Path("."),
        "--repo-root",
        help="Repository root to inspect.",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Emit deterministic JSON summary instead of text lines.",
    ),
) -> None:
    """Build the rebuildable memory graph sidecar."""
    console = Console(width=120, markup=False)
    try:
        context = RepoContext(repo_root)
        snapshot = build_memory_graph(context, persist=True)
    except MemoryGraphError as exc:
        console.print(f"Memory graph build failed: {exc}")
        raise typer.Exit(code=1) from None

    sidecar = MEMORY_GRAPH_SIDECAR.as_posix()
    if json_output:
        typer.echo(
            json.dumps(
                {
                    "schema_version": snapshot.schema_version,
                    "graph_fingerprint": snapshot.graph_fingerprint,
                    "node_count": snapshot.node_count,
                    "edge_count": snapshot.edge_count,
                    "sidecar_path": sidecar,
                    "warnings": snapshot.warnings,
                },
                ensure_ascii=True,
                indent=2,
            )
        )
        return

    console.print(
        "Memory graph built: "
        f"{sidecar} ({snapshot.node_count} node(s), "
        f"{snapshot.edge_count} edge(s))."
    )
    console.print(f"- fingerprint: {snapshot.graph_fingerprint}")
    for warning in snapshot.warnings:
        console.print(f"- warning: {warning}")


@memory_graph_app.command("pagerank")
def memory_graph_pagerank(
    repo_root: Path = typer.Option(
        Path("."),
        "--repo-root",
        help="Repository root to inspect.",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Emit deterministic JSON PageRank output instead of text lines.",
    ),
) -> None:
    """Compute global PageRank from an existing memory graph sidecar."""
    console = Console(width=120, markup=False)
    try:
        snapshot = load_memory_graph_snapshot(RepoContext(repo_root))
        result = compute_global_pagerank(snapshot)
    except MemoryGraphError as exc:
        console.print(f"Memory graph PageRank failed: {exc}")
        raise typer.Exit(code=1) from None

    if json_output:
        typer.echo(result.to_json(), nl=False)
        return

    if not result.rows:
        console.print("No memory graph PageRank rows.")
        return

    for row in result.rows:
        console.print(
            f"{row.rank}. {row.node_id} | score={row.score:.12f} | "
            f"{row.kind} | {row.record_id}"
        )


@provider_app.command("list")
def provider_list(
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Emit deterministic JSON instead of text output.",
    ),
) -> None:
    """List provider gateway modes available to agent callers."""
    providers = [
        {
            "provider": ProviderName.FAKE.value,
            "mode": ProviderMode.FAKE.value,
            "enabled_by_default": True,
            "network": "not_used",
            "api_key_required": False,
            "fake_run_cli": True,
            "real_run_cli": False,
        },
        {
            "provider": ProviderName.OPENAI.value,
            "mode": ProviderMode.OPENAI_COMPATIBLE.value,
            "enabled_by_default": False,
            "network": "explicit_config_only",
            "api_key_required": True,
            "fake_run_cli": False,
            "real_run_cli": True,
        },
    ]
    if json_output:
        _emit_json({"schema_version": 1, "providers": providers})
        return

    console = Console(width=120, markup=False)
    for provider in providers:
        console.print(
            f"{provider['provider']} | mode={provider['mode']} | "
            f"network={provider['network']} | "
            f"real_run_cli={str(provider['real_run_cli']).lower()}"
        )


@provider_app.command("config-check")
def provider_config_check(
    repo_root: Path = typer.Option(
        Path("."),
        "--repo-root",
        help="Repository root to inspect.",
    ),
    provider: ProviderName = typer.Option(
        ProviderName.FAKE,
        "--provider",
        help="Provider identifier to check.",
    ),
    api_key_env: str | None = typer.Option(
        None,
        "--api-key-env",
        help="Environment variable name for provider API key presence checks.",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Emit deterministic JSON instead of text output.",
    ),
) -> None:
    """Check provider configuration without revealing secret values."""
    console = Console(width=120, markup=False)
    _ensure_supported_provider_cli(provider, json_output=json_output, console=console)
    try:
        RepoContext(repo_root)
    except WorkspaceConfigError as exc:
        _exit_with_error(
            ErrorResult(
                code="workspace_config_failed",
                message=str(exc),
                remediation="Fix cosheaf.toml and rerun provider config-check.",
                blocking=True,
                related_path="cosheaf.toml",
            ),
            json_output=json_output,
            console=console,
        )

    env_name = api_key_env or _default_provider_api_key_env(provider)
    api_key_present = bool(env_name and os.environ.get(env_name))
    payload = {
        "schema_version": 1,
        "provider": provider.value,
        "mode": _provider_mode(provider).value,
        "enabled": provider is ProviderName.FAKE,
        "api_key_env": env_name,
        "api_key_present": api_key_present,
        "api_key_value": "<redacted>" if api_key_present else "missing",
        "real_run_cli": provider is ProviderName.OPENAI,
        "network": "not_used"
        if provider is ProviderName.FAKE
        else "explicit_config_only",
    }
    if json_output:
        _emit_json(payload)
        return

    console.print(f"provider: {payload['provider']}")
    console.print(f"mode: {payload['mode']}")
    console.print(f"api_key_present: {str(api_key_present).lower()}")
    api_key_value = "<redacted>" if api_key_present else "missing"
    console.print(f"api_key_value: {api_key_value}")
    console.print(f"real_run_cli: {str(payload['real_run_cli']).lower()}")


@provider_app.command("preview-send")
def provider_preview_send(
    issue: str = typer.Option(..., "--issue", help="Issue ID to preview."),
    provider: ProviderName = typer.Option(
        ProviderName.FAKE,
        "--provider",
        help="Provider identifier for preview metadata.",
    ),
    repo_root: Path = typer.Option(
        Path("."),
        "--repo-root",
        help="Repository root to inspect.",
    ),
    include_private: bool = typer.Option(
        False,
        "--include-private",
        help="Preview private context. Requires private policy and consent.",
    ),
    policy_mode: ContextPolicyMode = typer.Option(
        ContextPolicyMode.PUBLIC,
        "--policy-mode",
        help="Context policy mode.",
    ),
    allow_private_context: bool = typer.Option(
        False,
        "--allow-private-context",
        help="Confirm private-context preview for private research mode.",
    ),
    max_cards: int = typer.Option(
        20,
        "--max-cards",
        help="Maximum preview cards.",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Emit deterministic JSON instead of text output.",
    ),
) -> None:
    """Preview provider-send context shape without sending artifact text."""
    console = Console(width=120, markup=False)
    _ensure_supported_provider_cli(provider, json_output=json_output, console=console)
    try:
        request = ContextBuildRequest(
            issue_id=issue,
            max_cards=max_cards,
            max_full_artifacts=0,
            policy_mode=policy_mode,
            public_only=not include_private,
            allow_private_context=allow_private_context,
        )
        preview = ContextSendPolicyService(RepoContext(repo_root)).provider_preview(
            request
        )
    except (ValidationError, WorkspaceConfigError) as exc:
        _exit_with_error(
            _exception_to_error_result(
                exc,
                default_code="provider_context_preview_failed",
            ),
            json_output=json_output,
            console=console,
        )

    if isinstance(preview, ErrorResult):
        _exit_with_error(preview, json_output=json_output, console=console)

    preview_payload = preview.to_dict()
    payload = {
        "schema_version": 1,
        "provider": provider.value,
        "mode": _provider_mode(provider).value,
        "real_run_performed": False,
        "preview": preview_payload,
        "payload_shape": {
            "artifact_count": len(preview.artifact_ids),
            "card_count": preview.card_count,
            "full_artifact_count": preview.full_artifact_count,
            "content_mode": preview.content_mode,
            "root_scopes": preview_payload["root_scopes"],
            "estimated_tokens": preview.estimated_tokens,
            "private_context_included": preview.private_context_included,
            "risk_flags": preview.risk_flags,
        },
    }
    if json_output:
        _emit_json(payload)
        return

    console.print(f"provider: {provider.value}")
    console.print(f"artifact_count: {len(preview.artifact_ids)}")
    console.print(f"full_artifact_count: {preview.full_artifact_count}")
    console.print(f"root_scopes: {', '.join(preview_payload['root_scopes'])}")
    console.print(f"estimated_tokens: {preview.estimated_tokens}")


@provider_app.command("fake-run")
def provider_fake_run(
    input_json: Path = typer.Option(
        ...,
        "--input-json",
        help="JSON request for a fake provider run.",
    ),
    repo_root: Path = typer.Option(
        Path("."),
        "--repo-root",
        help="Repository root for provider run logs.",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Emit deterministic JSON instead of text output.",
    ),
) -> None:
    """Run the deterministic fake provider; no hosted API call is performed."""
    console = Console(width=120, markup=False)
    context = RepoContext(repo_root)
    raw = _read_input_json_or_exit(input_json, json_output=json_output)
    raw["provider"] = ProviderName.FAKE.value
    raw.setdefault("model", "fake-deterministic")
    raw.setdefault(
        "consent",
        ProviderConsent(
            consent_required=False,
            consent_granted=False,
            allow_private_context=False,
            policy_scope=ContextPolicyMode.PUBLIC,
        ).to_dict(),
    )
    try:
        request = ProviderGatewayRequest.model_validate(raw)
        result = ModelCallService(context).call(
            request,
            config=ProviderConfig(
                provider=ProviderName.FAKE,
                mode=ProviderMode.FAKE,
                model=request.model,
                enabled=True,
            ),
        )
    except (ValidationError, WorkspaceConfigError) as exc:
        _exit_with_error(
            _exception_to_error_result(exc, default_code="provider_fake_run_failed"),
            json_output=json_output,
            console=console,
        )

    if isinstance(result, ProviderError):
        _exit_with_error(
            ErrorResult(
                code=result.code,
                message=result.message,
                remediation=result.remediation,
                blocking=result.blocking,
                details=result.details,
            ),
            json_output=json_output,
            console=console,
        )

    provider_log = _provider_log_payload(context, result)
    payload = result.to_dict()
    payload["provider_log"] = provider_log
    if json_output:
        _emit_json(payload)
        return

    console.print(f"provider: {result.provider.value}")
    console.print(f"status: {result.status.value}")
    if result.provider_run.log_path:
        console.print(f"log_path: {result.provider_run.log_path}")


@provider_app.command("real-run")
def provider_real_run(
    input_json: Path = typer.Option(
        ...,
        "--input-json",
        help="JSON request envelope for an explicit real provider run.",
    ),
    provider: str = typer.Option(
        "openai-compatible",
        "--provider",
        help="Real provider identifier. Currently supports openai-compatible.",
    ),
    repo_root: Path = typer.Option(
        Path("."),
        "--repo-root",
        help="Repository root for provider run logs.",
    ),
    confirm_send: bool = typer.Option(
        False,
        "--confirm-send",
        help="Confirm that the operator approves sending this request.",
    ),
    allow_network: bool = typer.Option(
        False,
        "--allow-network",
        help="Allow the real provider command to use network transport.",
    ),
    allow_private_context: bool = typer.Option(
        False,
        "--allow-private-context",
        help="Confirm private-context send after private-research preview.",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Emit deterministic JSON instead of text output.",
    ),
) -> None:
    """Run one explicitly consented OpenAI-compatible provider call."""
    console = Console(width=120, markup=False)
    resolved_provider = _resolve_real_run_provider(
        provider,
        json_output=json_output,
        console=console,
    )
    if not confirm_send:
        _exit_with_error(
            ErrorResult(
                code="provider_confirm_send_required",
                message="provider real-run requires --confirm-send",
                remediation=(
                    "Preview the context, then rerun with --confirm-send only "
                    "when the operator approves sending it."
                ),
                blocking=True,
            ),
            json_output=json_output,
            console=console,
        )
    if not allow_network:
        _exit_with_error(
            ErrorResult(
                code="provider_network_not_allowed",
                message="provider real-run requires --allow-network",
                remediation=(
                    "Rerun with --allow-network only when this real provider "
                    "call is intentional."
                ),
                blocking=True,
            ),
            json_output=json_output,
            console=console,
        )

    context = RepoContext(repo_root)
    raw = _read_input_json_or_exit(input_json, json_output=json_output)
    preview = _real_run_context_preview_or_exit(
        raw,
        allow_private_context=allow_private_context,
        json_output=json_output,
        console=console,
    )
    config = _real_run_provider_config_or_exit(
        raw,
        provider=resolved_provider,
        json_output=json_output,
        console=console,
    )
    request = _real_run_gateway_request_or_exit(
        raw,
        provider=resolved_provider,
        preview=preview,
        confirm_send=confirm_send,
        allow_network=allow_network,
        allow_private_context=allow_private_context,
        json_output=json_output,
        console=console,
    )

    result = ModelCallService(context).call(
        request,
        config=config,
        provider=OpenAICompatibleProvider(transport=OpenAICompatibleHttpTransport()),
    )
    if isinstance(result, ProviderError):
        _exit_with_error(
            ErrorResult(
                code=result.code,
                message=result.message,
                remediation=result.remediation,
                blocking=result.blocking,
                details=result.details,
            ),
            json_output=json_output,
            console=console,
        )

    payload = result.to_dict()
    payload["real_run_performed"] = True
    payload["context_preview"] = preview.to_dict()
    payload["provider_log"] = _provider_log_payload(context, result)
    if json_output:
        _emit_json(payload)
        return

    console.print(f"provider: {result.provider.value}")
    console.print(f"status: {result.status.value}")
    if result.provider_run.log_path:
        console.print(f"log_path: {result.provider_run.log_path}")


@mcp_app.command("list-tools")
def mcp_list_tools(
    repo_root: Path = typer.Option(
        Path("."),
        "--repo-root",
        help="Repository root used for command consistency.",
    ),
) -> None:
    """List whitelisted MCP tool names."""
    RepoContext(repo_root)
    for tool_name in READ_ONLY_TOOL_NAMES:
        typer.echo(tool_name)


@mcp_app.command("serve")
def mcp_serve(
    stdio: bool = typer.Option(
        False,
        "--stdio",
        help="Serve line-delimited JSON-RPC over stdio.",
    ),
    repo_root: Path = typer.Option(
        Path("."),
        "--repo-root",
        help="Repository root to expose through whitelisted MCP tools.",
    ),
) -> None:
    """Serve the optional MCP JSON-RPC surface."""
    if not stdio:
        Console(width=120, markup=False).print("MCP serve failed: --stdio is required")
        raise typer.Exit(code=1)
    serve_stdio(RepoContext(repo_root))


@orchestrator_app.command("plan")
def orchestrator_plan(
    issue: str = typer.Option(..., "--issue", help="Issue ID to plan for."),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Emit deterministic JSON for the plan.",
    ),
    repo_root: Path = typer.Option(
        Path("."),
        "--repo-root",
        help="Repository root to inspect.",
    ),
) -> None:
    """Create a deterministic task-DAG plan without executing workers."""
    console = Console(width=120, markup=False)
    try:
        plan = plan_for_issue(RepoContext(repo_root), issue)
    except OrchestratorPlannerError as exc:
        if json_output:
            _emit_error(
                ErrorResult(
                    code="orchestrator_plan_failed",
                    message=str(exc),
                    remediation="Check the issue ID and repository records.",
                    blocking=True,
                    related_artifact=_valid_related_artifact(issue),
                )
            )
            raise typer.Exit(code=1) from None
        console.print(f"Orchestrator plan failed: {exc}")
        raise typer.Exit(code=1) from None

    if json_output:
        _emit_json({"schema_version": 1, **plan.to_dict()})
        return

    console.print(f"Plan: {plan.plan_id}")
    console.print(f"- issue: {plan.issue_id}")
    console.print(f"- objective: {plan.objective}")
    console.print("- execution: not performed")
    console.print("- accepted knowledge writes: not performed")
    console.print("Task DAG:")
    for node in plan.task_dag.nodes:
        depends_on = ", ".join(node.depends_on) if node.depends_on else "-"
        console.print(
            f"- {node.node_id} | {node.worker_type.value} | depends_on={depends_on}"
        )


@orchestrator_app.command("run")
def orchestrator_run(
    issue: str = typer.Option(..., "--issue", help="Issue ID to run locally."),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Run deterministic dry-run workers only.",
    ),
    local_only: bool = typer.Option(
        False,
        "--local-only",
        help="Require local-only execution with no hosted LLM or network.",
    ),
    timeout_seconds: int = typer.Option(
        60,
        "--timeout-seconds",
        help="Maximum runtime for each local worker command in seconds.",
    ),
    provider: str | None = typer.Option(
        None,
        "--provider",
        help=(
            "Explicit hosted-worker provider path: fake or openai-compatible. "
            "Omit to use the local-only dry-run path."
        ),
    ),
    confirm_send: bool = typer.Option(
        False,
        "--confirm-send",
        help="Confirm provider dispatch after reviewing context-send policy.",
    ),
    include_private: bool = typer.Option(
        False,
        "--include-private",
        help="Include private context. Requires private policy and consent.",
    ),
    policy_mode: ContextPolicyMode = typer.Option(
        ContextPolicyMode.PUBLIC,
        "--policy-mode",
        help="Context policy mode for provider dispatch.",
    ),
    allow_private_context: bool = typer.Option(
        False,
        "--allow-private-context",
        help="Confirm private-context provider dispatch for private research mode.",
    ),
    max_cards: int = typer.Option(
        20,
        "--max-cards",
        help="Maximum context-preview cards for provider dispatch.",
    ),
    repo_root: Path = typer.Option(
        Path("."),
        "--repo-root",
        help="Repository root to inspect.",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Emit deterministic JSON instead of text output.",
    ),
) -> None:
    """Run a local-only or explicit hosted-worker orchestrator path."""
    console = Console(width=120, markup=False)
    if provider is not None:
        try:
            hosted_result = OrchestratorHostedRunner(RepoContext(repo_root)).run_issue(
                OrchestratorHostedRunConfig(
                    issue_id=issue,
                    provider=provider,
                    confirm_send=confirm_send,
                    include_private=include_private,
                    policy_mode=policy_mode,
                    allow_private_context=allow_private_context,
                    max_cards=max_cards,
                )
            )
        except OrchestratorHostedRunError as exc:
            _exit_with_error(exc.error, json_output=json_output, console=console)
        except WorkspaceConfigError as exc:
            _exit_with_error(
                _exception_to_error_result(
                    exc,
                    default_code="workspace_config_failed",
                ),
                json_output=json_output,
                console=console,
            )

        if json_output:
            _emit_json(_orchestrator_hosted_result_payload(hosted_result))
        else:
            console.print(f"Orchestrator run: {hosted_result.run.run_id}")
            console.print(f"- issue: {hosted_result.run.issue_id}")
            console.print(f"- state: {hosted_result.run.state.value}")
            console.print(f"- provider: {_orchestrator_provider_label(hosted_result)}")
            console.print(f"- mode: {hosted_result.provider_mode.value}")
            console.print("- accepted_writes: not performed")
            console.print(f"- run_record: {hosted_result.record_path}")
            console.print(f"- worker_calls: {len(hosted_result.run.worker_calls)}")
            console.print(
                f"- reducer_results: {len(hosted_result.run.reducer_results)}"
            )
            for stop in hosted_result.run.stop_conditions:
                console.print(f"- stop: {stop.reason} | {stop.description}")
        if hosted_result.run.state.value != "completed":
            raise typer.Exit(code=1)
        return

    if not dry_run:
        _exit_with_error(
            ErrorResult(
                code="orchestrator_run_failed",
                message="--dry-run is required when --provider is omitted",
                remediation=(
                    "Use --dry-run --local-only, or set --provider fake for "
                    "the explicit hosted-worker path."
                ),
                blocking=True,
            ),
            json_output=json_output,
            console=console,
        )
    if not local_only:
        _exit_with_error(
            ErrorResult(
                code="orchestrator_run_failed",
                message="--local-only is required when --provider is omitted",
                remediation=(
                    "Use --dry-run --local-only, or set --provider fake for "
                    "the explicit hosted-worker path."
                ),
                blocking=True,
            ),
            json_output=json_output,
            console=console,
        )

    try:
        result = OrchestratorLocalRunner(RepoContext(repo_root)).run_issue(
            OrchestratorLocalRunConfig(
                issue_id=issue,
                timeout_seconds=timeout_seconds,
            )
        )
    except OrchestratorLocalRunError as exc:
        _exit_with_error(
            ErrorResult(
                code="orchestrator_run_failed",
                message=str(exc),
                remediation="Check the issue ID, run ID, and local runner inputs.",
                blocking=True,
            ),
            json_output=json_output,
            console=console,
        )

    if json_output:
        _emit_json(_orchestrator_local_result_payload(result))
        if result.run.state.value != "completed":
            raise typer.Exit(code=1)
        return

    console.print(f"Orchestrator run: {result.run.run_id}")
    console.print(f"- issue: {result.run.issue_id}")
    console.print(f"- state: {result.run.state.value}")
    console.print("- local_only: true")
    console.print("- hosted_llm: not used")
    console.print("- network: not used")
    console.print("- accepted_writes: not performed")
    console.print(f"- run_record: {result.record_path}")
    console.print(f"- worker_calls: {len(result.run.worker_calls)}")
    console.print(f"- reducer_results: {len(result.run.reducer_results)}")
    for stop in result.run.stop_conditions:
        console.print(f"- stop: {stop.reason} | {stop.description}")

    if result.run.state.value != "completed":
        raise typer.Exit(code=1)


@task_app.command("create")
def task_create(
    issue: str = typer.Option(..., "--issue", help="Issue ID for the task."),
    worker: WorkerType = typer.Option(
        ...,
        "--worker",
        help="Protocol worker type for the task.",
    ),
    repo_root: Path = typer.Option(
        Path("."),
        "--repo-root",
        help="Repository root to inspect.",
    ),
) -> None:
    """Create an open local agent task without invoking a worker."""
    console = Console(width=120, markup=False)
    try:
        task = TaskService(RepoContext(repo_root)).create_task(
            issue_id=issue,
            worker_type=worker,
        )
    except TaskHarnessError as exc:
        console.print(f"Task create failed: {exc}")
        raise typer.Exit(code=1) from None

    console.print(f"Task created: {task.task_id}")
    console.print(f"- issue: {task.issue_id}")
    console.print(f"- worker: {task.worker_type.value}")
    console.print(f"- status: {task.status.value}")


@task_app.command("list")
def task_list(
    repo_root: Path = typer.Option(
        Path("."),
        "--repo-root",
        help="Repository root to inspect.",
    ),
) -> None:
    """List local agent tasks."""
    console = Console(width=120, markup=False)
    try:
        tasks = TaskService(RepoContext(repo_root)).list_tasks()
    except TaskHarnessError as exc:
        console.print(f"Task list failed: {exc}")
        raise typer.Exit(code=1) from None

    if not tasks:
        console.print("No tasks.")
        return

    for task in tasks:
        console.print(
            f"{task.task_id} | {task.issue_id} | "
            f"{task.worker_type.value} | {task.status.value}"
        )


@task_app.command("complete")
def task_complete(
    task_id: str = typer.Argument(..., help="Task ID to mark complete."),
    bundle: Path = typer.Option(
        ...,
        "--bundle",
        help="Worker output bundle manifest path or containing directory.",
    ),
    repo_root: Path = typer.Option(
        Path("."),
        "--repo-root",
        help="Repository root to inspect.",
    ),
) -> None:
    """Validate a worker output bundle and mark the task complete."""
    console = Console(width=120, markup=False)
    try:
        result = TaskService(RepoContext(repo_root)).complete_task(
            task_id=task_id,
            bundle_path=bundle,
        )
    except TaskHarnessError as exc:
        console.print(f"Task complete failed: {exc}")
        raise typer.Exit(code=1) from None

    console.print(f"Task completed: {result.task.task_id}")
    console.print(f"- bundle outputs: {len(result.bundle.outputs)}")
    console.print("- accepted knowledge merge: not performed")


@task_app.command("run")
def task_run(
    task_id: str = typer.Argument(..., help="Task ID to run a local command for."),
    command: list[str] = typer.Argument(
        ...,
        help="Explicit command argv. Separate it from options with '--'.",
    ),
    timeout_seconds: int = typer.Option(
        60,
        "--timeout-seconds",
        help="Maximum command runtime in seconds.",
    ),
    cwd: Path | None = typer.Option(
        None,
        "--cwd",
        help="Optional repository-local working directory.",
    ),
    bundle: Path | None = typer.Option(
        None,
        "--bundle",
        help="Validate a worker output bundle after a successful command.",
    ),
    complete_with_bundle: Path | None = typer.Option(
        None,
        "--complete-with-bundle",
        help=(
            "Validate a worker output bundle after a successful command and "
            "delegate task completion to the orchestrator stub."
        ),
    ),
    repo_root: Path = typer.Option(
        Path("."),
        "--repo-root",
        help="Repository root to inspect.",
    ),
) -> None:
    """Run an explicit local command for an existing task."""
    console = Console(width=120, markup=False)
    if bundle is not None and complete_with_bundle is not None:
        console.print("Task run failed: use either --bundle or --complete-with-bundle")
        raise typer.Exit(code=1)

    bundle_path = complete_with_bundle or bundle
    try:
        context = RepoContext(repo_root)
        service = TaskService(context)
        result = service.run_task(
            task_id,
            command=command,
            timeout_seconds=timeout_seconds,
            cwd=cwd,
            bundle_path=bundle_path,
        )
    except LocalWorkerRunError as exc:
        console.print(f"Task run failed: {exc}")
        raise typer.Exit(code=1) from None

    task_completed = False
    if result.status == "completed" and complete_with_bundle is not None:
        try:
            service.complete_task(
                task_id=task_id,
                bundle_path=complete_with_bundle,
            )
        except TaskHarnessError as exc:
            console.print(f"Task complete failed: {exc}")
            raise typer.Exit(code=1) from None
        task_completed = True

    console.print(f"status: {result.status}")
    console.print(f"returncode: {result.returncode}")
    console.print(f"run_directory: {result.run_dir}")
    if result.bundle_valid is not None:
        console.print(f"bundle_valid: {str(result.bundle_valid).lower()}")
    if complete_with_bundle is not None:
        console.print(f"task_completed: {str(task_completed).lower()}")

    if result.status != "completed":
        raise typer.Exit(code=1)


def _emit_json(payload: dict[str, Any] | list[Any]) -> None:
    typer.echo(json.dumps(payload, ensure_ascii=True, indent=2))


def _emit_compare_or_exit(
    run: Callable[[], Any],
    *,
    json_output: bool,
    code: str,
) -> None:
    console = Console(width=120, markup=False)
    try:
        result = run()
    except ValueError as exc:
        if json_output:
            _emit_error(
                ErrorResult(
                    code=code,
                    message=str(exc),
                    remediation="Pass existing workflow, campaign, or benchmark IDs.",
                    blocking=True,
                )
            )
            raise typer.Exit(code=1) from None
        console.print(f"Comparison failed: {exc}")
        raise typer.Exit(code=1) from None

    payload = result.to_dict()
    if json_output:
        _emit_json(payload)
        return
    console.print(
        f"Comparison: {payload['subject']} {payload['before_id']} -> "
        f"{payload['after_id']}"
    )
    console.print(
        f"- safety_regressions: {len(payload.get('safety_regressions', []))}"
    )
    console.print(f"- authority: {COMPARE_AUTHORITY_NOTICE}")


def _emit_static_report_or_exit(
    run: Callable[[], Any],
    *,
    json_output: bool,
    code: str,
) -> None:
    console = Console(width=120, markup=False)
    try:
        result = run()
    except ValueError as exc:
        if json_output:
            _emit_error(
                ErrorResult(
                    code=code,
                    message=str(exc),
                    remediation=(
                        "Use an existing runtime record ID and a repository-local "
                        "non-accepted output directory."
                    ),
                    blocking=True,
                )
            )
            raise typer.Exit(code=1) from None
        console.print(f"Static report failed: {exc}")
        raise typer.Exit(code=1) from None

    payload = result.to_dict()
    if json_output:
        _emit_json(payload)
        return
    console.print(
        f"Static report written: {payload['subject']} {payload['record_id']} "
        f"-> {payload['out_dir']}"
    )
    console.print(f"- files: {len(payload.get('files', {}))}")
    console.print(f"- authority: {payload['authority_notice']}")


def _emit_model(model: AgentAccessModel) -> None:
    typer.echo(model.to_json(), nl=False)


def _print_promotion_readiness_report(
    console: Console,
    payload: dict[str, Any],
) -> None:
    ready = str(payload.get("ready", False)).lower()
    target = payload.get("target", {})
    if not isinstance(target, dict):
        target = {}
    mode = str(target.get("mode", ""))
    target_id = str(target.get("artifact_id") or target.get("issue_id") or "")
    console.print(f"Promotion readiness: ready={ready} | {mode}={target_id}")
    console.print("- accepted write: not performed")
    artifacts = payload.get("artifacts", [])
    if not isinstance(artifacts, list):
        return
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            continue
        artifact_ready = str(artifact.get("ready", False)).lower()
        artifact_id = str(artifact.get("artifact_id", ""))
        status = str(artifact.get("status", ""))
        console.print(f"- {artifact_id}: ready={artifact_ready} | status={status}")
        reasons = artifact.get("reasons", [])
        if not isinstance(reasons, list):
            continue
        for reason in reasons:
            if not isinstance(reason, dict):
                continue
            code = str(reason.get("code", "reason"))
            severity = str(reason.get("severity", ""))
            message = str(reason.get("message", ""))
            console.print(f"  - {severity} {code}: {message}")


def _emit_error(error: ErrorResult) -> None:
    _emit_model(error)


def _read_input_json_or_exit(
    input_json: Path,
    *,
    json_output: bool,
) -> dict[str, Any]:
    try:
        raw = json.loads(input_json.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        _exit_with_error(
            ErrorResult(
                code="invalid_input_json",
                message=f"invalid input JSON: {exc}",
                remediation="Provide a readable JSON object via --input-json.",
                blocking=True,
                related_path=_valid_related_path(str(input_json)),
            ),
            json_output=json_output,
            console=Console(width=120, markup=False),
        )
    if not isinstance(raw, dict):
        _exit_with_error(
            ErrorResult(
                code="invalid_input_json",
                message="input JSON must be an object",
                remediation="Provide a JSON object at the input document root.",
                blocking=True,
                related_path=_valid_related_path(str(input_json)),
            ),
            json_output=json_output,
            console=Console(width=120, markup=False),
        )
    return dict(raw)


def _exit_with_error(
    error: ErrorResult,
    *,
    json_output: bool,
    console: Console,
) -> NoReturn:
    if json_output:
        _emit_error(error)
    else:
        console.print(f"{error.code}: {error.message}")
        console.print(f"remediation: {error.remediation}")
    raise typer.Exit(code=1)


def _exception_to_error_result(
    exc: Exception,
    *,
    default_code: str,
) -> ErrorResult:
    if isinstance(exc, ServiceError):
        return exc.to_error_result()
    if isinstance(exc, ValidationError):
        return ErrorResult(
            code=default_code,
            message=_format_pydantic_errors(exc),
            remediation="Fix the input JSON fields and retry.",
            blocking=True,
        )
    return ErrorResult(
        code=default_code,
        message=str(exc),
        remediation="Fix the request and retry.",
        blocking=True,
    )


def _operator_session_error_result(exc: Exception) -> ErrorResult:
    if isinstance(exc, OperatorSessionError):
        return ErrorResult(
            code=exc.code,
            message=str(exc),
            remediation=exc.remediation,
            blocking=True,
            details=exc.details,
        )
    message = (
        _format_pydantic_errors(exc) if isinstance(exc, ValidationError) else str(exc)
    )
    if "accepted KB paths" in message or "accepted_write_forbidden" in message:
        return ErrorResult(
            code="accepted_write_forbidden",
            message=message,
            remediation=(
                "Operator sessions may reference runtime, draft, and "
                "review-context files only; do not reference accepted KB paths."
            ),
            blocking=True,
        )
    return ErrorResult(
        code="operator_session_validation_failed",
        message=message,
        remediation=(
            "Fix the operator-session metadata request and retry. Operator "
            "sessions are review metadata only."
        ),
        blocking=True,
    )


def _parse_operator_session_check_kind(value: str) -> OperatorCheckKind:
    normalized = value.strip()
    if normalized not in _OPERATOR_SESSION_CLI_CHECK_KINDS:
        raise OperatorSessionError(
            f"unsupported operator-session check kind: {normalized}",
            code="operator_session_validation_failed",
            remediation="Use one of validate, gate, test, or eval.",
        )
    return OperatorCheckKind(normalized)


def _parse_operator_session_ref_kind(value: str) -> OperatorArtifactRefKind:
    normalized = value.strip()
    if normalized not in _OPERATOR_SESSION_CLI_REF_KINDS:
        raise OperatorSessionError(
            f"unsupported operator-session reference kind: {normalized}",
            code="operator_session_validation_failed",
            remediation="Use one of draft, review_context, runtime, or report.",
        )
    return OperatorArtifactRefKind(normalized)


def _parse_operator_session_ref_scope(value: str) -> OperatorArtifactRefScope:
    normalized = value.strip()
    if normalized not in _OPERATOR_SESSION_CLI_REF_SCOPES:
        raise OperatorSessionError(
            f"unsupported operator-session reference scope: {normalized}",
            code="operator_session_validation_failed",
            remediation=(
                "Use one of public, private, workspace, framework, or unknown."
            ),
        )
    return cast(OperatorArtifactRefScope, normalized)


def _operator_session_check_summary(
    *,
    kind: OperatorCheckKind,
    status: OperatorCheckStatus,
    summary: str | None,
) -> str:
    if status is OperatorCheckStatus.SKIPPED and summary is None:
        return SKIPPED_OPERATOR_SESSION_LIMITATION
    if summary is None:
        return (
            f"{kind.value} was recorded as {status.value} by the operator-session "
            "metadata CLI. This record does not create review, verifier, gate, "
            "accepted-status, or promotion authority."
        )
    return summary


def _ensure_operator_session_ref_allowed(
    *,
    session: OperatorSession,
    path: str,
    scope: OperatorArtifactRefScope,
) -> None:
    if _operator_session_path_is_accepted(path):
        raise OperatorSessionError(
            "operator sessions cannot reference accepted KB paths",
            code="accepted_write_forbidden",
            remediation=(
                "Use draft, runtime, report, or review-context references; "
                "accepted KB content remains outside session write authority."
            ),
            details={"path": normalize_repo_path(path)},
        )
    if (
        session.policy_mode is OperatorPolicyMode.PUBLIC_ONLY
        and _operator_session_ref_is_private(path=path, scope=scope)
    ):
        raise OperatorSessionError(
            "public-only operator sessions cannot reference private paths or scope",
            code="private_context_requires_policy",
            remediation=(
                "Start the session with --policy private_research before "
                "recording private references."
            ),
            details={"path": normalize_repo_path(path), "scope": scope},
        )


def _operator_session_path_is_accepted(path: str) -> bool:
    parts = _operator_session_path_parts(path)
    return bool(parts) and parts[0] == "kb" and "accepted" in parts


def _operator_session_ref_is_private(*, path: str, scope: str) -> bool:
    parts = _operator_session_path_parts(path)
    return scope == "private" or (
        len(parts) >= 2 and parts[0] == "kb" and parts[1] == "private"
    )


def _operator_session_path_parts(path: str) -> tuple[str, ...]:
    return tuple(part for part in normalize_repo_path(path).split("/") if part)


def _checked_evidence_error_result(exc: Exception) -> ErrorResult:
    if isinstance(exc, CheckedCounterexampleEvidenceError):
        return ErrorResult(
            code=exc.code,
            message=str(exc),
            remediation=exc.remediation,
            blocking=True,
            details=exc.details,
        )
    if isinstance(exc, ValidationError):
        return ErrorResult(
            code="checked_evidence_validation_failed",
            message=_format_pydantic_errors(exc),
            remediation=(
                "Fix the checked counterexample evidence fields and retry. "
                "Checked evidence is review evidence only."
            ),
            blocking=True,
        )
    return ErrorResult(
        code="checked_evidence_validation_failed",
        message=str(exc),
        remediation=(
            "Fix the checked counterexample evidence fields and retry. "
            "Checked evidence is review evidence only."
        ),
        blocking=True,
    )


def _research_run_error_result(exc: Exception) -> ErrorResult:
    if isinstance(exc, ResearchRunError):
        return ErrorResult(
            code=exc.code,
            message=str(exc),
            remediation=exc.remediation,
            blocking=True,
            details=exc.details,
        )
    if isinstance(exc, ValidationError):
        return ErrorResult(
            code="research_run_validation_failed",
            message=_format_pydantic_errors(exc),
            remediation=(
                "Fix the research-run payload and retry. "
                "Research runs are provenance only."
            ),
            blocking=True,
        )
    return ErrorResult(
        code="research_run_validation_failed",
        message=str(exc),
        remediation=(
            "Fix the research-run payload and retry. Research runs are provenance only."
        ),
        blocking=True,
    )


def _campaign_error_result(exc: Exception) -> ErrorResult:
    from cosheaf.campaigns import CampaignError

    if isinstance(exc, CampaignError):
        details = exc.details
        return ErrorResult(
            code=exc.code,
            message=str(exc),
            remediation=exc.remediation,
            blocking=True,
            details=details,
        )
    if isinstance(exc, ValidationError):
        return ErrorResult(
            code="campaign_validation_failed",
            message=_format_pydantic_errors(exc),
            remediation=(
                "Fix the campaign payload and retry. Campaign records are "
                "review context only."
            ),
            blocking=True,
        )
    return ErrorResult(
        code="campaign_validation_failed",
        message=str(exc),
        remediation=(
            "Fix the campaign request and retry. Campaign records are review "
            "context only."
        ),
        blocking=True,
    )


def _strategy_error_result(exc: Exception) -> ErrorResult:
    if isinstance(exc, StrategyError):
        return ErrorResult(
            code=exc.code,
            message=str(exc),
            remediation=exc.remediation,
            blocking=True,
            details=exc.details,
        )
    if isinstance(exc, ValidationError):
        return ErrorResult(
            code="strategy_validation_failed",
            message=_format_pydantic_errors(exc),
            remediation=(
                "Regenerate or repair the strategy plan. Strategy plans are "
                "guidance only."
            ),
            blocking=True,
        )
    return ErrorResult(
        code="strategy_failed",
        message=str(exc),
        remediation="Inspect the issue, plan ID, or runtime strategy record.",
        blocking=True,
    )


def _emit_controlled_write(
    result: ControlledWriteResult,
    *,
    json_output: bool,
    console: Console,
) -> None:
    if json_output:
        _emit_json(
            {
                "schema_version": 1,
                "kind": result.kind,
                "path": result.relative_path.as_posix(),
                "written_paths": [path.as_posix() for path in result.written_paths],
                "dry_run": result.dry_run,
                "accepted_write_performed": result.accepted_write_performed,
                "record_id": result.record_id,
            }
        )
        return

    action = "would write" if result.dry_run else "wrote"
    console.print(f"{result.kind}: {action} {result.relative_path.as_posix()}")
    console.print("- accepted knowledge merge: not performed")


def _resolve_real_run_provider(
    provider: str,
    *,
    json_output: bool,
    console: Console,
) -> ProviderName:
    normalized = provider.strip().replace("_", "-").lower()
    if normalized in {"openai", "openai-compatible"}:
        return ProviderName.OPENAI
    _exit_with_error(
        ErrorResult(
            code="provider_unsupported",
            message=f"Provider real-run does not support {provider!r}.",
            remediation="Use --provider openai-compatible.",
            blocking=True,
            details={"supported_providers": "openai-compatible"},
        ),
        json_output=json_output,
        console=console,
    )


def _real_run_context_preview_or_exit(
    raw: dict[str, Any],
    *,
    allow_private_context: bool,
    json_output: bool,
    console: Console,
) -> ProviderContextPreview:
    preview_raw = raw.get("context_preview")
    if not isinstance(preview_raw, dict):
        _exit_with_error(
            ErrorResult(
                code="provider_context_preview_failed",
                message="provider real-run requires inline context_preview",
                remediation=(
                    "Run provider preview-send first and include the resulting "
                    "preview object in the real-run input JSON."
                ),
                blocking=True,
            ),
            json_output=json_output,
            console=console,
        )
    try:
        preview = ProviderContextPreview.model_validate(preview_raw)
    except ValidationError as exc:
        _exit_with_error(
            ErrorResult(
                code="provider_context_preview_failed",
                message=_format_pydantic_errors(exc),
                remediation="Fix the inline context_preview object and retry.",
                blocking=True,
            ),
            json_output=json_output,
            console=console,
        )
    if (
        preview.private_context_included
        and preview.policy_mode is not ContextPolicyMode.PRIVATE_RESEARCH
    ):
        _exit_with_error(
            ErrorResult(
                code="private_context_requires_policy",
                message="private context requires private_research policy",
                remediation=(
                    "Rebuild the preview with private_research policy before "
                    "sending private KB context."
                ),
                blocking=True,
            ),
            json_output=json_output,
            console=console,
        )
    if preview.private_context_included and not allow_private_context:
        _exit_with_error(
            ErrorResult(
                code="private_context_requires_consent",
                message="private context real-run requires --allow-private-context",
                remediation=(
                    "Rerun with --allow-private-context only when private KB "
                    "context send is explicitly approved."
                ),
                blocking=True,
            ),
            json_output=json_output,
            console=console,
        )
    return preview


def _real_run_provider_config_or_exit(
    raw: dict[str, Any],
    *,
    provider: ProviderName,
    json_output: bool,
    console: Console,
) -> ProviderConfig:
    config_raw = raw.get("provider_config")
    if not isinstance(config_raw, dict):
        _exit_with_error(
            ErrorResult(
                code="provider_config_missing",
                message="provider real-run requires provider_config",
                remediation=(
                    "Add provider_config with model, base_url, api_key_env, "
                    "timeout_seconds, and retry policy."
                ),
                blocking=True,
            ),
            json_output=json_output,
            console=console,
        )
    config_payload = dict(config_raw)
    config_payload["provider"] = provider
    config_payload["mode"] = ProviderMode.OPENAI_COMPATIBLE
    config_payload["enabled"] = True
    config_payload.setdefault("model", raw.get("model", "openai-compatible"))
    try:
        config = ProviderConfig.model_validate(config_payload)
    except ValidationError as exc:
        _exit_with_error(
            ErrorResult(
                code="provider_config_missing",
                message=_format_pydantic_errors(exc),
                remediation="Fix provider_config and retry.",
                blocking=True,
            ),
            json_output=json_output,
            console=console,
        )
    if config.base_url is None or config.api_key_env is None:
        _exit_with_error(
            ErrorResult(
                code="provider_config_missing",
                message="provider_config requires base_url and api_key_env",
                remediation=(
                    "Set provider_config.base_url and provider_config.api_key_env. "
                    "The key value itself must remain in the environment."
                ),
                blocking=True,
            ),
            json_output=json_output,
            console=console,
        )
    return config


def _real_run_gateway_request_or_exit(
    raw: dict[str, Any],
    *,
    provider: ProviderName,
    preview: ProviderContextPreview,
    confirm_send: bool,
    allow_network: bool,
    allow_private_context: bool,
    json_output: bool,
    console: Console,
) -> ProviderGatewayRequest:
    request_payload: dict[str, Any] = {
        "provider": provider,
        "model": raw.get("model"),
        "worker_role": raw.get("worker_role"),
        "prompt": raw.get("prompt"),
        "consent": ProviderConsent(
            consent_required=True,
            consent_granted=confirm_send,
            allow_private_context=allow_private_context,
            policy_scope=preview.policy_mode,
            operator_note="Explicit provider real-run send consent.",
        ).to_dict(),
        "context_artifact_ids": list(preview.artifact_ids),
        "root_scopes": [scope.value for scope in preview.root_scopes],
        "output_kind": raw.get("output_kind", "text"),
        "expected_output_paths": raw.get("expected_output_paths", []),
        "network_policy": NetworkPolicy.EXPLICIT_ALLOW
        if allow_network
        else NetworkPolicy.DISABLED,
    }
    for key in (
        "temperature",
        "top_p",
        "reasoning_effort",
        "max_output_tokens",
        "tool_policy",
    ):
        if key in raw:
            request_payload[key] = raw[key]
    try:
        return ProviderGatewayRequest.model_validate(request_payload)
    except ValidationError as exc:
        _exit_with_error(
            ErrorResult(
                code="provider_request_validation_failed",
                message=_format_pydantic_errors(exc),
                remediation="Fix the real-run request fields and retry.",
                blocking=True,
            ),
            json_output=json_output,
            console=console,
        )


def _provider_mode(provider: ProviderName) -> ProviderMode:
    if provider is ProviderName.OPENAI:
        return ProviderMode.OPENAI_COMPATIBLE
    if provider is ProviderName.FAKE:
        return ProviderMode.FAKE
    msg = f"provider CLI does not support {provider.value!r}"
    raise ValueError(msg)


def _ensure_supported_provider_cli(
    provider: ProviderName,
    *,
    json_output: bool,
    console: Console,
) -> None:
    if provider in _SUPPORTED_PROVIDER_CLI_NAMES:
        return
    supported = ",".join(provider.value for provider in _SUPPORTED_PROVIDER_CLI_NAMES)
    _exit_with_error(
        ErrorResult(
            code="provider_unsupported",
            message=f"Provider CLI does not support {provider.value!r}.",
            remediation=(
                "Use `cosheaf provider list --json` to inspect currently "
                "supported provider CLI modes."
            ),
            blocking=True,
            details={"supported_providers": supported},
        ),
        json_output=json_output,
        console=console,
    )


def _default_provider_api_key_env(provider: ProviderName) -> str | None:
    if provider is ProviderName.OPENAI:
        return "OPENAI_API_KEY"
    return None


def _provider_log_payload(
    context: RepoContext,
    result: ModelCallResult,
) -> dict[str, Any]:
    log_path = result.provider_run.log_path
    if log_path is None:
        return {}
    raw = json.loads(context.resolve(Path(log_path)).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        return {}
    return dict(raw)


def _orchestrator_local_result_payload(
    result: OrchestratorLocalRunResult,
) -> dict[str, Any]:
    repo_root = _orchestrator_run_repo_root(result.run_root)
    return {
        "schema_version": 1,
        "run_id": result.run.run_id,
        "issue_id": result.run.issue_id,
        "state": result.run.state.value,
        "local_only": True,
        "provider": None,
        "hosted_network": "not_used",
        "accepted_write_performed": False,
        "run_record": repo_relative_posix(repo_root, result.record_path),
        "worker_calls": [call.to_dict() for call in result.run.worker_calls],
        "reducer_results": [
            reducer.to_dict() for reducer in result.run.reducer_results
        ],
        "stop_conditions": [stop.to_dict() for stop in result.run.stop_conditions],
    }


def _orchestrator_hosted_result_payload(
    result: OrchestratorHostedRunResult,
) -> dict[str, Any]:
    repo_root = _orchestrator_run_repo_root(result.run_root)
    return {
        "schema_version": 1,
        "run_id": result.run.run_id,
        "issue_id": result.run.issue_id,
        "state": result.run.state.value,
        "provider": _orchestrator_provider_label(result),
        "mode": result.provider_mode.value,
        "context_preview": result.context_preview.to_dict(),
        "hosted_network": "not_used"
        if result.provider is ProviderName.FAKE
        else "explicit_config_only",
        "accepted_write_performed": result.accepted_write_performed,
        "run_record": repo_relative_posix(repo_root, result.record_path),
        "provider_run_record_paths": [
            repo_relative_posix(repo_root, path)
            for path in result.provider_run_record_paths
        ],
        "worker_calls": [call.to_dict() for call in result.run.worker_calls],
        "reducer_results": [
            reducer.to_dict() for reducer in result.run.reducer_results
        ],
        "stop_conditions": [stop.to_dict() for stop in result.run.stop_conditions],
    }


def _orchestrator_provider_label(result: OrchestratorHostedRunResult) -> str:
    if result.provider is ProviderName.OPENAI:
        return "openai-compatible"
    return result.provider.value


def _orchestrator_run_repo_root(run_root: Path) -> Path:
    return run_root.parents[4]


def _workspace_info_to_agent_result(
    context: RepoContext,
    info: Any,
) -> AgentWorkspaceInfoResult:
    return AgentWorkspaceInfoResult(
        workspace_name=info.name,
        repo_root=str(context.repo_root),
        mode=info.mode,
        kb_roots=[
            AgentKbRoot(
                name=root.name,
                path=root.path,
                scope=_kb_root_scope(root),
                readonly=root.readonly,
                priority=root.priority,
            )
            for root in info.kb_roots
        ],
        policy=KbRootPolicy(
            private_can_depend_on_public=(
                context.workspace_config.policy.private_can_depend_on_public
            ),
            public_can_depend_on_private=(
                context.workspace_config.policy.public_can_depend_on_private
            ),
            accepted_requires_source=(
                context.workspace_config.policy.accepted_requires_source
            ),
        ),
    )


def _kb_root_scope(root: KbRootConfig) -> MemoryRootScope:
    name = root.name.lower()
    if name == "public":
        return MemoryRootScope.PUBLIC
    if name == "private":
        return MemoryRootScope.PRIVATE
    if name == "framework":
        return MemoryRootScope.FRAMEWORK
    return MemoryRootScope.WORKSPACE


def _find_failure_log_artifact(
    context: RepoContext,
    artifact_id: str,
) -> LoadedRecord:
    records = tuple(load_artifacts(context))
    return _find_unique_base_artifact(records, artifact_id)


def _artifact_failure_log_payload(
    context: RepoContext,
    loaded: LoadedRecord,
) -> dict[str, Any]:
    artifact = loaded.record
    if not isinstance(artifact, BaseArtifact):
        raise AssertionError("unreachable non-artifact failure-log record")
    root = (
        context.workspace_config.root_by_name(loaded.kb_root_name)
        if loaded.kb_root_name is not None
        else None
    )
    root_scope = _kb_root_scope(root) if root is not None else MemoryRootScope.WORKSPACE
    failure_log = [entry.model_dump(mode="json") for entry in artifact.failure_log]
    return {
        "schema_version": 1,
        "kind": "artifact_failure_log",
        "artifact_id": artifact.id,
        "artifact_path": loaded.source_path.as_posix(),
        "root_name": loaded.kb_root_name,
        "root_scope": root_scope.value,
        "root_readonly": loaded.kb_root_readonly,
        "failure_count": len(failure_log),
        "failure_log": failure_log,
        "authority_notice": _FAILURE_LOG_AUTHORITY_NOTICE,
    }


def _failure_log_bundle_plan_payload(
    result: FailureLogFromBundlePlanResult,
) -> dict[str, Any]:
    entries = [entry.model_dump(mode="json") for entry in result.entries]
    return {
        "schema_version": 1,
        "kind": "artifact_failure_log_bundle_plan",
        "artifact_id": result.artifact_id,
        "artifact_path": result.relative_path.as_posix(),
        "bundle_id": result.bundle.bundle_id,
        "entry_count": len(entries),
        "entries": entries,
        "accepted_write_performed": False,
        "authority_notice": _FAILURE_LOG_AUTHORITY_NOTICE,
    }


def _failure_log_bundle_write_payload(
    result: FailureLogFromBundleWriteResult,
) -> dict[str, Any]:
    write = result.write_result
    entries = [entry.model_dump(mode="json") for entry in result.plan.entries]
    return {
        "schema_version": 1,
        "kind": write.kind,
        "path": write.relative_path.as_posix(),
        "written_paths": [path.as_posix() for path in write.written_paths],
        "dry_run": write.dry_run,
        "accepted_write_performed": write.accepted_write_performed,
        "record_id": write.record_id,
        "artifact_id": result.plan.artifact_id,
        "bundle_id": result.plan.bundle.bundle_id,
        "entry_count": len(entries),
        "planned_entries": entries,
        "authority_notice": _FAILURE_LOG_AUTHORITY_NOTICE,
    }


def _valid_related_path(value: str | None) -> str | None:
    if not value:
        return None
    try:
        ErrorResult(
            code="path_probe",
            message="Path probe.",
            remediation="Path probe.",
            blocking=False,
            related_path=value,
        )
    except ValidationError:
        return None
    return value


def _valid_related_artifact(value: str | None) -> str | None:
    if not value:
        return None
    try:
        return validate_artifact_id(value.strip())
    except ValueError:
        return None


def _print_dependency_graph(console: Console, graph: DependencyGraph) -> None:
    console.print(
        f"[bold]Dependency graph[/bold]: "
        f"{len(graph.nodes)} node(s), {len(graph.edges)} edge(s)."
    )
    if graph.edges:
        for edge in graph.edges:
            console.print(f"- {edge.source_id} -> {edge.target_id}")
    else:
        console.print("- no dependency edges")

    for issue in graph.missing_dependencies:
        console.print(
            f"[red]- missing dependency[/red] | "
            f"{issue.source_id} -> {issue.target_id} | {issue.source_path}"
        )
    for cycle in graph.cycles:
        console.print(f"[red]- cycle[/red] | {' -> '.join(cycle)}")
    for issue in graph.accepted_draft_violations:
        console.print(
            f"[red]- accepted->draft[/red] | "
            f"{issue.source_id} -> {issue.target_id} | {issue.source_path}"
        )


class ArtifactLifecycleError(ValueError):
    """Raised for expected artifact lifecycle CLI failures."""


PROMOTION_REVIEW_STATES = frozenset({"human_reviewed", "accepted"})
BLOCKING_VERIFIER_STATUSES = frozenset({"fail", "error"})


def _create_artifact_record(
    *,
    context: RepoContext,
    artifact_id: str,
    artifact_type: ArtifactType,
    title: str,
    domain: list[str],
    status: ArtifactStatus,
    statement: str,
    authors: list[str],
    tags: list[str],
    depends_on: list[str],
    supersedes: list[str],
    created_at: str | None,
) -> tuple[BaseArtifact, Path]:
    if not domain:
        raise ArtifactLifecycleError("at least one --domain value is required")
    if status is ArtifactStatus.ACCEPTED:
        raise ArtifactLifecycleError(
            "accepted artifacts must be promoted through a dedicated gate/review "
            "workflow"
        )

    try:
        validate_artifact_id(artifact_id)
    except ValueError as exc:
        raise ArtifactLifecycleError(str(exc)) from exc

    timestamp = _parse_artifact_timestamp(created_at)
    try:
        relative_path = _workspace_lifecycle_artifact_path(
            context=context,
            artifact_type=artifact_type,
            status=status,
            artifact_id=artifact_id,
        )
    except ValueError as exc:
        raise ArtifactLifecycleError(str(exc)) from exc

    _ensure_artifact_id_is_available(context, artifact_id)
    target_path = context.resolve(relative_path)
    if target_path.exists():
        raise ArtifactLifecycleError(
            f"artifact path already exists: {relative_path.as_posix()}"
        )

    try:
        artifact = BaseArtifact.model_validate(
            {
                "id": artifact_id,
                "type": artifact_type,
                "title": title,
                "domain": domain,
                "status": status,
                "created_at": timestamp,
                "updated_at": timestamp,
                "authors": authors,
                "depends_on": depends_on,
                "supersedes": supersedes,
                "tags": tags,
                "statement": statement,
                "evidence": [],
                "review": {"state": "requested", "notes": "Created by CLI."},
                "risk": {"level": "low", "notes": ""},
            }
        )
    except ValidationError as exc:
        raise ArtifactLifecycleError(_format_pydantic_errors(exc)) from exc

    write_yaml_deterministic(target_path, artifact)
    report = validate_artifact_file(context, relative_path)
    if report.ok:
        return artifact, relative_path

    target_path.unlink(missing_ok=True)
    raise ArtifactLifecycleError(_format_report_failures(report))


def _move_artifact_status(
    *,
    context: RepoContext,
    artifact_id: str,
    new_status: ArtifactStatus,
) -> tuple[ArtifactStatus, Path, Path]:
    records = _load_records_for_lifecycle(context)
    loaded = _find_unique_base_artifact(records, artifact_id)
    artifact = loaded.record
    if not isinstance(artifact, BaseArtifact):
        raise AssertionError("unreachable non-artifact lifecycle record")

    _ensure_kb_lifecycle_record(loaded)
    _ensure_current_status_path_consistent(loaded)
    _ensure_loaded_record_is_writable(loaded)

    if new_status is ArtifactStatus.ACCEPTED:
        raise ArtifactLifecycleError(
            "accepted promotion requires a dedicated gate/review workflow; "
            "move-status refuses direct moves into kb/accepted"
        )

    old_status = artifact.status
    old_relative_path = loaded.source_path
    new_relative_path = _workspace_status_move_path(loaded, artifact, new_status)

    if old_status is new_status and old_relative_path == new_relative_path:
        return old_status, old_relative_path, new_relative_path

    _ensure_repository_valid_for_lifecycle_move(context)
    _write_status_move(
        context=context,
        artifact=artifact,
        source_relative_path=old_relative_path,
        target_relative_path=new_relative_path,
        new_status=new_status,
    )
    report = validate_artifact_file(context, new_relative_path)
    if not report.ok:
        raise ArtifactLifecycleError(_format_report_failures(report))
    return old_status, old_relative_path, new_relative_path


def _promote_artifact(
    *,
    context: RepoContext,
    artifact_id: str,
) -> tuple[ArtifactStatus, Path, Path]:
    validation_report = validate_repository(context)
    if not validation_report.ok:
        raise ArtifactLifecycleError(
            "repository validation failed before promotion: "
            f"{_format_report_failures(validation_report)}"
        )

    records = validation_report.records
    loaded = _find_unique_promotable_artifact(records, artifact_id)
    artifact = loaded.record
    if not isinstance(artifact, BaseArtifact):
        raise AssertionError("unreachable non-artifact promotion record")

    _ensure_kb_lifecycle_record(loaded)
    _ensure_current_status_path_consistent(loaded)
    _ensure_loaded_record_is_writable(loaded)
    _ensure_preaccepted_for_promotion(artifact)

    gatekeeper_result = run_gatekeeper(context)
    _ensure_gatekeeper_allows_promotion(gatekeeper_result, artifact_id)
    _ensure_artifact_reviewed_for_promotion(artifact, records=records)
    _ensure_promotion_dependencies_accepted(records, artifact)
    _ensure_source_metadata_for_public_promotion(context, loaded, artifact)

    old_status = artifact.status
    old_relative_path = loaded.source_path
    new_relative_path = _workspace_status_move_path(
        loaded,
        artifact,
        ArtifactStatus.ACCEPTED,
    )

    _write_accepted_promotion(
        context=context,
        artifact=artifact,
        source_relative_path=old_relative_path,
        target_relative_path=new_relative_path,
    )
    return old_status, old_relative_path, new_relative_path


def _find_unique_promotable_artifact(
    records: tuple[LoadedRecord, ...],
    artifact_id: str,
) -> LoadedRecord:
    matches = [record for record in records if record.id == artifact_id]
    if not matches:
        raise ArtifactLifecycleError(f"artifact not found: {artifact_id}")
    if len(matches) > 1:
        paths = ", ".join(sorted(record.source_path.as_posix() for record in matches))
        raise ArtifactLifecycleError(f"duplicate artifact id {artifact_id}: {paths}")

    loaded = matches[0]
    if not isinstance(loaded.record, BaseArtifact):
        raise ArtifactLifecycleError(
            f"record is not a promotable lifecycle artifact: {artifact_id}"
        )
    try:
        lifecycle_artifact_path(
            loaded.record.type,
            ArtifactStatus.ACCEPTED,
            loaded.record.id,
        )
    except ValueError as exc:
        raise ArtifactLifecycleError(
            f"record is not a promotable lifecycle artifact: {artifact_id}"
        ) from exc
    return loaded


def _ensure_preaccepted_for_promotion(artifact: BaseArtifact) -> None:
    if artifact.status is ArtifactStatus.ACCEPTED:
        raise ArtifactLifecycleError(f"artifact is already accepted: {artifact.id}")
    if not is_preaccepted_status(artifact.status):
        raise ArtifactLifecycleError(
            "only pre-accepted lifecycle artifacts may be promoted: "
            f"{artifact.id} has status {artifact.status.value}"
        )


def _ensure_gatekeeper_allows_promotion(
    result: GatekeeperRunResult,
    artifact_id: str,
) -> None:
    target_blockers = _target_verifier_blockers(result, artifact_id)
    if target_blockers:
        raise ArtifactLifecycleError(
            f"target verifier result blocks promotion: {'; '.join(target_blockers)}"
        )
    if result.report.blocking_issues:
        raise ArtifactLifecycleError(
            "gatekeeper blocking issues prevent promotion: "
            f"{_format_gatekeeper_blocking_issues(result)}"
        )


def _target_verifier_blockers(
    result: GatekeeperRunResult,
    artifact_id: str,
) -> list[str]:
    blockers: list[str] = []
    for gate in result.report.gates:
        if gate.gate_id != "G6":
            continue
        for detail in gate.details:
            if detail.get("artifact_id") != artifact_id:
                continue
            status = detail.get("status")
            if status not in BLOCKING_VERIFIER_STATUSES:
                continue
            verifier = str(detail.get("verifier", "verifier"))
            message = str(detail.get("message", "")).strip()
            rendered = f"{verifier} {status}"
            if message:
                rendered = f"{rendered}: {message}"
            blockers.append(rendered)
    if blockers:
        return blockers

    return [
        issue.message
        for issue in result.report.blocking_issues
        if issue.gate_id == "G6" and issue.artifact_id == artifact_id
    ]


def _format_gatekeeper_blocking_issues(result: GatekeeperRunResult) -> str:
    return "; ".join(
        f"{issue.gate_id} | {issue.source_path or '-'} | "
        f"{issue.artifact_id or '-'} | {issue.message}"
        for issue in result.report.blocking_issues
    )


def _ensure_artifact_reviewed_for_promotion(
    artifact: BaseArtifact,
    *,
    records: tuple[LoadedRecord, ...] = (),
) -> None:
    if artifact.review.state not in PROMOTION_REVIEW_STATES:
        raise ArtifactLifecycleError(
            "review.state must be human_reviewed or accepted before promotion: "
            f"{artifact.id} has {artifact.review.state}"
        )
    mismatch = review_revision_mismatch(artifact, records)
    if mismatch is not None:
        raise ArtifactLifecycleError(f"stale_review: {mismatch}")
    evidence_mismatches = evidence_revision_mismatches(artifact)
    if evidence_mismatches:
        raise ArtifactLifecycleError(
            f"stale_evidence: {'; '.join(msg for _, msg in evidence_mismatches)}"
        )


def _ensure_promotion_dependencies_accepted(
    records: tuple[LoadedRecord, ...],
    artifact: BaseArtifact,
) -> None:
    artifacts_by_id = {
        record.id: record
        for record in records
        if isinstance(record.record, BaseArtifact)
    }
    for dependency_id in artifact.depends_on:
        if is_external_dependency_ref(dependency_id):
            continue
        dependency = artifacts_by_id.get(dependency_id)
        if dependency is None or not isinstance(dependency.record, BaseArtifact):
            raise ArtifactLifecycleError(f"dependency is missing: {dependency_id}")
        if dependency.record.status is ArtifactStatus.ACCEPTED:
            continue
        raise ArtifactLifecycleError(
            "dependency is not accepted: "
            f"{dependency_id} has status {dependency.record.status.value} "
            f"at {dependency.source_path.as_posix()}"
        )


def _ensure_source_metadata_for_public_promotion(
    context: RepoContext,
    loaded: LoadedRecord,
    artifact: BaseArtifact,
) -> None:
    if not context.workspace_config.policy.accepted_requires_source:
        return
    if loaded.kb_root_name != "public":
        return
    missing = missing_required_source_metadata(artifact)
    if not missing:
        return
    if missing == ("sources",):
        raise ArtifactLifecycleError(
            "accepted public artifact requires source metadata before promotion: "
            f"{artifact.id}"
        )
    raise ArtifactLifecycleError(
        "accepted public artifact has incomplete source metadata before promotion: "
        + ", ".join(missing)
    )


def _write_accepted_promotion(
    *,
    context: RepoContext,
    artifact: BaseArtifact,
    source_relative_path: Path,
    target_relative_path: Path,
) -> None:
    source_path = context.resolve(source_relative_path)
    target_path = context.resolve(target_relative_path)
    if target_path.exists() and source_path != target_path:
        raise ArtifactLifecycleError(
            f"target artifact path already exists: {target_relative_path.as_posix()}"
        )

    original_text = source_path.read_text(encoding="utf-8")
    updated = artifact.model_copy(
        update={
            "status": ArtifactStatus.ACCEPTED,
            "updated_at": datetime.now(UTC).replace(microsecond=0),
        }
    )
    write_yaml_deterministic(source_path, updated)
    if source_path != target_path:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            source_path.rename(target_path)
        except OSError:
            source_path.write_text(original_text, encoding="utf-8", newline="\n")
            raise

    report = validate_artifact_file(context, target_relative_path)
    if report.ok:
        return

    _restore_failed_promotion(
        source_path=source_path,
        target_path=target_path,
        original_text=original_text,
    )
    raise ArtifactLifecycleError(_format_report_failures(report))


def _restore_failed_promotion(
    *,
    source_path: Path,
    target_path: Path,
    original_text: str,
) -> None:
    if source_path != target_path and target_path.exists():
        target_path.rename(source_path)
    source_path.write_text(original_text, encoding="utf-8", newline="\n")


def _write_status_move(
    *,
    context: RepoContext,
    artifact: BaseArtifact,
    source_relative_path: Path,
    target_relative_path: Path,
    new_status: ArtifactStatus,
) -> None:
    source_path = context.resolve(source_relative_path)
    target_path = context.resolve(target_relative_path)
    if target_path.exists() and source_path != target_path:
        raise ArtifactLifecycleError(
            f"target artifact path already exists: {target_relative_path.as_posix()}"
        )

    updated = artifact.model_copy(update={"status": new_status})
    original_text = source_path.read_text(encoding="utf-8")
    write_yaml_deterministic(source_path, updated)
    if source_path == target_path:
        return

    target_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        source_path.rename(target_path)
    except OSError:
        source_path.write_text(original_text, encoding="utf-8", newline="\n")
        raise


def _parse_artifact_timestamp(value: str | None) -> datetime:
    if value is None:
        return datetime.now(UTC).replace(microsecond=0)
    try:
        normalized = value.replace("Z", "+00:00") if value.endswith("Z") else value
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise ArtifactLifecycleError(
            f"invalid --created-at timestamp: {value}"
        ) from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ArtifactLifecycleError("--created-at must include timezone information")
    return parsed.astimezone(UTC).replace(microsecond=0)


def _ensure_artifact_id_is_available(context: RepoContext, artifact_id: str) -> None:
    records = _load_records_for_lifecycle(context)
    if any(record.id == artifact_id for record in records):
        raise ArtifactLifecycleError(f"artifact already exists: {artifact_id}")


def _load_records_for_lifecycle(context: RepoContext) -> tuple[LoadedRecord, ...]:
    try:
        return tuple(load_artifacts(context))
    except LoadError as exc:
        raise ArtifactLifecycleError(f"cannot load repository records: {exc}") from exc


def _find_unique_base_artifact(
    records: tuple[LoadedRecord, ...],
    artifact_id: str,
) -> LoadedRecord:
    matches = [record for record in records if record.id == artifact_id]
    if not matches:
        raise ArtifactLifecycleError(f"artifact not found: {artifact_id}")
    if len(matches) > 1:
        paths = ", ".join(sorted(record.source_path.as_posix() for record in matches))
        raise ArtifactLifecycleError(f"duplicate artifact id {artifact_id}: {paths}")

    loaded = matches[0]
    if not isinstance(loaded.record, BaseArtifact):
        raise ArtifactLifecycleError(f"record is not an artifact: {artifact_id}")
    return loaded


def _ensure_kb_lifecycle_record(loaded: LoadedRecord) -> None:
    if loaded.kb_relative_path is None:
        raise ArtifactLifecycleError(
            "lifecycle status moves are only supported for records under a KB root: "
            f"{loaded.source_path.as_posix()}"
        )


def _ensure_current_status_path_consistent(loaded: LoadedRecord) -> None:
    artifact = loaded.record
    if not isinstance(artifact, BaseArtifact):
        return
    allowed = expected_status_for_path(_status_path_for_loaded_record(loaded))
    if artifact.status in allowed:
        return
    expected = ", ".join(sorted(status.value for status in allowed))
    raise ArtifactLifecycleError(
        "status/path mismatch: "
        f"{loaded.source_path.as_posix()} has status {artifact.status.value}; "
        f"expected one of {expected}"
    )


def _ensure_loaded_record_is_writable(loaded: LoadedRecord) -> None:
    if not loaded.kb_root_readonly:
        return
    root_name = loaded.kb_root_name or "<unknown>"
    raise ArtifactLifecycleError(f"readonly KB root cannot be modified: {root_name}")


def _workspace_lifecycle_artifact_path(
    *,
    context: RepoContext,
    artifact_type: ArtifactType,
    status: ArtifactStatus,
    artifact_id: str,
) -> Path:
    legacy_path = lifecycle_artifact_path(artifact_type, status, artifact_id)
    if not context.workspace_config.configured:
        return legacy_path
    root = _default_writable_kb_root(context)
    return Path(root.path) / _strip_legacy_kb_prefix(legacy_path)


def _workspace_status_move_path(
    loaded: LoadedRecord,
    artifact: BaseArtifact,
    new_status: ArtifactStatus,
) -> Path:
    legacy_path = lifecycle_artifact_path(artifact.type, new_status, artifact.id)
    if loaded.kb_root_path is None:
        return legacy_path
    return loaded.kb_root_path / _strip_legacy_kb_prefix(legacy_path)


def _default_writable_kb_root(context: RepoContext) -> KbRootConfig:
    writable_roots = [root for root in context.workspace_config.kb if not root.readonly]
    if not writable_roots:
        raise ArtifactLifecycleError("no writable KB root is configured")

    private_roots = [root for root in writable_roots if root.name == "private"]
    candidates = private_roots or writable_roots
    return sorted(
        candidates,
        key=lambda root: (-root.priority, root.name, root.path),
    )[0]


def _strip_legacy_kb_prefix(path: Path) -> Path:
    parts = path.parts
    if parts and parts[0] == "kb":
        return Path(*parts[1:])
    return path


def _status_path_for_loaded_record(loaded: LoadedRecord) -> str:
    if loaded.kb_relative_path is None:
        return loaded.source_path.as_posix()
    relative = loaded.kb_relative_path.as_posix()
    return "kb" if not relative else f"kb/{relative}"


def _ensure_repository_valid_for_lifecycle_move(context: RepoContext) -> None:
    report = validate_repository(context)
    if not report.ok:
        raise ArtifactLifecycleError(
            "repository validation failed before status move: "
            f"{_format_report_failures(report)}"
        )


def _format_pydantic_errors(exc: ValidationError) -> str:
    return "; ".join(
        f"{'.'.join(str(part) for part in error['loc'])}: {error['msg']}"
        for error in exc.errors()
    )


def _format_report_failures(report: ValidationReport) -> str:
    return "; ".join(
        f"{failure.gate} | {failure.source_path} | "
        f"{failure.artifact_id} | {failure.message}"
        for failure in report.failures
    )


# Campaign commands


@campaign_app.command("start")
def campaign_start(
    issue_id: str = typer.Option(
        ...,
        "--issue",
        help="Issue ID this campaign targets.",
    ),
    max_attempts: int = typer.Option(
        10,
        "--max-attempts",
        help="Maximum attempts for this campaign.",
    ),
    campaign_id: str | None = typer.Option(
        None,
        "--campaign-id",
        help="Optional deterministic campaign ID.",
    ),
    repo_root: Path = typer.Option(
        Path("."),
        "--repo-root",
        help="Repository root.",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Emit deterministic JSON instead of text output.",
    ),
) -> None:
    """Start a bounded multi-run research campaign."""
    from cosheaf.campaigns import CampaignBudget, CampaignError, start_campaign

    console = Console(width=120, markup=False)
    try:
        result = start_campaign(
            RepoContext(repo_root),
            issue_id=issue_id,
            campaign_id=campaign_id,
            budget=CampaignBudget(max_attempts=max_attempts),
        )
    except (CampaignError, ValidationError, ValueError) as exc:
        _exit_with_error(
            _campaign_error_result(exc),
            json_output=json_output,
            console=console,
        )
    if json_output:
        _emit_json(result.to_dict())
        return
    console.print(f"Campaign started: {result.campaign.campaign_id}")
    console.print(f"- path: {result.relative_path.as_posix()}")
    console.print(f"- scorecard: {result.scorecard_path.as_posix()}")
    console.print(f"- events: {result.events_path.as_posix()}")
    console.print(f"- authority: {result.campaign.authority_notice}")


@campaign_app.command("show")
def campaign_show(
    campaign_id: str = typer.Argument(..., help="Campaign ID to show."),
    repo_root: Path = typer.Option(
        Path("."),
        "--repo-root",
        help="Repository root.",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Emit deterministic JSON instead of text output.",
    ),
) -> None:
    """Show one campaign runtime record."""
    from cosheaf.campaigns import CampaignError, load_campaign

    console = Console(width=120, markup=False)
    try:
        result = load_campaign(RepoContext(repo_root), campaign_id)
    except (CampaignError, ValidationError, ValueError) as exc:
        _exit_with_error(
            _campaign_error_result(exc),
            json_output=json_output,
            console=console,
        )
    if json_output:
        _emit_json(result.to_dict())
        return
    console.print(f"Campaign: {result.campaign.campaign_id}")
    console.print(f"- issue: {result.campaign.issue_id}")
    console.print(f"- status: {result.campaign.status.value}")
    console.print(f"- attempts: {len(result.campaign.attempts)}")
    console.print(f"- authority: {result.campaign.authority_notice}")


@campaign_app.command("append-attempt")
def campaign_append_attempt(
    campaign_id: str = typer.Argument(..., help="Campaign ID."),
    input_json: Path = typer.Option(
        ...,
        "--input-json",
        help="CampaignAttempt JSON payload.",
    ),
    repo_root: Path = typer.Option(
        Path("."),
        "--repo-root",
        help="Repository root.",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Emit deterministic JSON instead of text output.",
    ),
) -> None:
    """Append a validated attempt record to a campaign."""
    from cosheaf.campaigns import (
        CampaignAttempt,
        CampaignError,
        append_campaign_attempt,
        load_campaign,
    )

    console = Console(width=120, markup=False)
    context = RepoContext(repo_root)
    try:
        loaded = load_campaign(context, campaign_id)
        raw = _read_input_json_or_exit(input_json, json_output=json_output)
        raw.setdefault("campaign_id", loaded.campaign.campaign_id)
        raw.setdefault("attempt_number", len(loaded.campaign.attempts) + 1)
        raw.setdefault(
            "attempt_id",
            f"{loaded.campaign.campaign_id}.attempt."
            f"{len(loaded.campaign.attempts) + 1}",
        )
        attempt = CampaignAttempt.model_validate(raw)
        result = append_campaign_attempt(
            context,
            loaded.campaign.campaign_id,
            attempt,
        )
    except (CampaignError, ValidationError, ValueError) as exc:
        _exit_with_error(
            _campaign_error_result(exc),
            json_output=json_output,
            console=console,
        )
    if json_output:
        _emit_json(result.to_dict())
        return
    console.print(f"Campaign attempt appended: {result.attempt.attempt_id}")
    console.print(f"- outcome: {result.attempt.outcome.value}")
    console.print(f"- path: {result.attempt_path.as_posix()}")
    console.print(f"- authority: {result.attempt.authority_notice}")


@campaign_app.command("scorecard")
def campaign_scorecard(
    campaign_id: str = typer.Argument(..., help="Campaign ID."),
    repo_root: Path = typer.Option(
        Path("."),
        "--repo-root",
        help="Repository root.",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Emit deterministic JSON instead of text output.",
    ),
) -> None:
    """Show and refresh a deterministic campaign scorecard."""
    from cosheaf.campaigns import CampaignError, show_campaign_scorecard

    console = Console(width=120, markup=False)
    try:
        result = show_campaign_scorecard(RepoContext(repo_root), campaign_id)
    except (CampaignError, ValidationError, ValueError) as exc:
        _exit_with_error(
            _campaign_error_result(exc),
            json_output=json_output,
            console=console,
        )
    if json_output:
        _emit_json(result.to_dict())
        return
    console.print(f"Campaign scorecard: {result.campaign.campaign_id}")
    console.print(f"- attempts: {result.scorecard.attempt_count}")
    console.print(f"- results: {result.scorecard.result_count}")
    console.print(f"- failures: {result.scorecard.failure_count}")
    console.print(f"- authority: {result.scorecard.authority_notice}")


@campaign_app.command("finalize")
def campaign_finalize(
    campaign_id: str = typer.Argument(..., help="Campaign ID."),
    status: str = typer.Option(
        "finalized",
        "--status",
        help="Terminal status: finalized, abandoned, or failed.",
    ),
    reason: str | None = typer.Option(
        None,
        "--reason",
        help="Optional finalization reason.",
    ),
    repo_root: Path = typer.Option(
        Path("."),
        "--repo-root",
        help="Repository root.",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Emit deterministic JSON instead of text output.",
    ),
) -> None:
    """Finalize a campaign without granting accepted authority."""
    from cosheaf.campaigns import CampaignError, CampaignStatus, finalize_campaign

    console = Console(width=120, markup=False)
    try:
        result = finalize_campaign(
            RepoContext(repo_root),
            campaign_id,
            status=CampaignStatus(status),
            reason=reason,
        )
    except (CampaignError, ValidationError, ValueError) as exc:
        _exit_with_error(
            _campaign_error_result(exc),
            json_output=json_output,
            console=console,
        )
    if json_output:
        _emit_json(result.to_dict())
        return
    console.print(f"Campaign finalized: {result.campaign.campaign_id}")
    console.print(f"- status: {result.campaign.status.value}")
    console.print(f"- authority: {result.campaign.authority_notice}")


@campaign_app.command("next")
def campaign_next(
    campaign_id: str = typer.Argument(..., help="Campaign ID."),
    repo_root: Path = typer.Option(
        Path("."),
        "--repo-root",
        help="Repository root.",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Emit deterministic JSON instead of text output.",
    ),
) -> None:
    """Preview the next bounded external-operator task for a campaign."""
    from cosheaf.campaigns import CampaignError, next_campaign_operator_task

    console = Console(width=120, markup=False)
    try:
        result = next_campaign_operator_task(RepoContext(repo_root), campaign_id)
    except (CampaignError, ValidationError, ValueError) as exc:
        _exit_with_error(
            _campaign_error_result(exc),
            json_output=json_output,
            console=console,
        )
    if json_output:
        _emit_json(result.to_dict())
        return
    console.print(f"Campaign next: {result.campaign_id}")
    console.print(f"- action: {result.next_action}")
    console.print(f"- attempt: {result.attempt_id or 'none'}")
    console.print(f"- previous failures: {len(result.previous_failures_to_avoid)}")
    console.print(f"- authority: {result.authority_notice}")


@campaign_app.command("export-task")
def campaign_export_task(
    campaign_id: str = typer.Argument(..., help="Campaign ID."),
    out: Path = typer.Option(
        ...,
        "--out",
        help="Repository-local operator_task_v2 JSON path.",
    ),
    repo_root: Path = typer.Option(
        Path("."),
        "--repo-root",
        help="Repository root.",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Emit deterministic JSON instead of text output.",
    ),
) -> None:
    """Export a bounded external-operator task packet for a campaign."""
    from cosheaf.campaigns import (
        CAMPAIGN_AUTHORITY_NOTICE,
        CampaignError,
        export_campaign_operator_task,
        next_campaign_operator_task,
    )

    console = Console(width=120, markup=False)
    try:
        context = RepoContext(repo_root)
        relative_path = export_campaign_operator_task(context, campaign_id, out)
        next_result = next_campaign_operator_task(context, campaign_id)
    except (CampaignError, ValidationError, ValueError) as exc:
        _exit_with_error(
            _campaign_error_result(exc),
            json_output=json_output,
            console=console,
        )
    payload = {
        "schema_version": 1,
        "kind": "campaign_operator_task_export",
        "campaign_id": campaign_id,
        "attempt_id": next_result.attempt_id,
        "path": relative_path.as_posix(),
        "writes_performed": True,
        "accepted_write_performed": False,
        "authority_notice": CAMPAIGN_AUTHORITY_NOTICE,
    }
    if json_output:
        _emit_json(payload)
        return
    console.print(f"Campaign operator task exported: {relative_path.as_posix()}")
    console.print("- accepted knowledge merge: not performed")


@campaign_app.command("import-result")
def campaign_import_result(
    campaign_id: str = typer.Argument(..., help="Campaign ID."),
    input_json: Path = typer.Option(
        ...,
        "--input-json",
        help="Structured operator_result_v2 JSON payload.",
    ),
    repo_root: Path = typer.Option(
        Path("."),
        "--repo-root",
        help="Repository root.",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Emit deterministic JSON instead of text output.",
    ),
) -> None:
    """Import a structured external-operator result as a campaign attempt."""
    from cosheaf.campaigns import (
        CampaignError,
        CampaignOperatorResult,
        import_campaign_operator_result,
    )

    console = Console(width=120, markup=False)
    try:
        raw = _read_input_json_or_exit(input_json, json_output=json_output)
        payload = CampaignOperatorResult.model_validate(raw)
        result = import_campaign_operator_result(
            RepoContext(repo_root),
            campaign_id,
            payload,
        )
    except (CampaignError, ValidationError, ValueError) as exc:
        _exit_with_error(
            _campaign_error_result(exc),
            json_output=json_output,
            console=console,
        )
    if json_output:
        _emit_json(result.to_dict())
        return
    console.print(f"Campaign operator result imported: {result.attempt_id}")
    console.print(f"- outcome: {result.attempt.outcome.value}")
    console.print(f"- path: {result.attempt_path}")
    console.print("- accepted knowledge merge: not performed")


@campaign_app.command("pause")
def campaign_pause(
    campaign_id: str = typer.Argument(..., help="Campaign ID."),
    reason: str | None = typer.Option(
        None,
        "--reason",
        help="Optional pause reason.",
    ),
    repo_root: Path = typer.Option(
        Path("."),
        "--repo-root",
        help="Repository root.",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Emit deterministic JSON instead of text output.",
    ),
) -> None:
    """Pause a campaign without executing shell or provider work."""
    from cosheaf.campaigns import CampaignError, pause_campaign

    console = Console(width=120, markup=False)
    try:
        result = pause_campaign(RepoContext(repo_root), campaign_id, reason=reason)
    except (CampaignError, ValidationError, ValueError) as exc:
        _exit_with_error(
            _campaign_error_result(exc),
            json_output=json_output,
            console=console,
        )
    if json_output:
        _emit_json(result.to_dict())
        return
    console.print(f"Campaign paused: {result.campaign.campaign_id}")
    console.print(f"- status: {result.campaign.status.value}")
    console.print("- shell/provider execution: not performed")


@campaign_app.command("resume")
def campaign_resume(
    campaign_id: str = typer.Argument(..., help="Campaign ID."),
    repo_root: Path = typer.Option(
        Path("."),
        "--repo-root",
        help="Repository root.",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Emit deterministic JSON instead of text output.",
    ),
) -> None:
    """Resume a paused campaign without bypassing blockers or budgets."""
    from cosheaf.campaigns import CampaignError, resume_campaign

    console = Console(width=120, markup=False)
    try:
        result = resume_campaign(RepoContext(repo_root), campaign_id)
    except (CampaignError, ValidationError, ValueError) as exc:
        _exit_with_error(
            _campaign_error_result(exc),
            json_output=json_output,
            console=console,
        )
    if json_output:
        _emit_json(result.to_dict())
        return
    console.print(f"Campaign resumed: {result.campaign.campaign_id}")
    console.print(f"- status: {result.campaign.status.value}")
    console.print("- shell/provider execution: not performed")


@campaign_app.command("scan")
def campaign_scan(
    campaign_id: str = typer.Argument(..., help="Campaign ID."),
    repo_root: Path = typer.Option(
        Path("."),
        "--repo-root",
        help="Repository root.",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Emit deterministic JSON instead of text output.",
    ),
) -> None:
    """Scan campaign runtime outputs for unsafe authority claims."""
    from cosheaf.campaigns import CampaignError, scan_campaign

    console = Console(width=120, markup=False)
    try:
        result = scan_campaign(RepoContext(repo_root), campaign_id)
    except (CampaignError, ValidationError, ValueError) as exc:
        _exit_with_error(
            _campaign_error_result(exc),
            json_output=json_output,
            console=console,
        )
    if json_output:
        _emit_json(result.to_dict())
        if result.run_blocked:
            raise typer.Exit(1)
        return
    console.print(f"Campaign scan: {result.campaign_id}")
    console.print(f"- findings: {result.finding_count}")
    console.print(f"- blockers: {result.blocking_finding_count}")
    console.print(f"- report: {result.report_path.as_posix()}")
    console.print(f"- authority: {result.authority_notice}")
    if result.run_blocked:
        raise typer.Exit(1)


@campaign_app.command("handoff")
def campaign_handoff(
    campaign_id: str = typer.Argument(..., help="Campaign ID."),
    out: Path = typer.Option(
        ...,
        "--out",
        help="Repository-local directory for campaign_handoff.json.",
    ),
    repo_root: Path = typer.Option(
        Path("."),
        "--repo-root",
        help="Repository root.",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Emit deterministic JSON instead of text output.",
    ),
) -> None:
    """Export a review handoff summary for one campaign."""
    from cosheaf.campaigns import CampaignError, build_campaign_handoff

    console = Console(width=120, markup=False)
    try:
        result = build_campaign_handoff(RepoContext(repo_root), campaign_id, out)
    except (CampaignError, ValidationError, ValueError) as exc:
        _exit_with_error(
            _campaign_error_result(exc),
            json_output=json_output,
            console=console,
        )
    if json_output:
        _emit_json(result.to_dict())
        return
    console.print(f"Campaign handoff exported: {result.campaign.campaign_id}")
    console.print(f"- path: {result.handoff_path.as_posix()}")
    console.print(f"- attempts: {result.metrics.attempt_count}")
    console.print(f"- unsafe outputs: {result.metrics.unsafe_output_count}")
    console.print(f"- authority: {result.authority_notice}")


@campaign_app.command("run")
def campaign_run(
    campaign_id: str = typer.Argument(..., help="Campaign ID."),
    max_attempts: int | None = typer.Option(
        None,
        "--max-attempts",
        help="Optional attempt ceiling for this controller invocation.",
    ),
    repo_root: Path = typer.Option(
        Path("."),
        "--repo-root",
        help="Repository root.",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Emit deterministic JSON instead of text output.",
    ),
) -> None:
    """Run deterministic campaign budget checks without model or shell calls."""
    from cosheaf.campaigns import CampaignError, run_campaign

    console = Console(width=120, markup=False)
    try:
        result = run_campaign(
            RepoContext(repo_root),
            campaign_id,
            max_attempts=max_attempts,
        )
    except (CampaignError, ValidationError, ValueError) as exc:
        _exit_with_error(
            _campaign_error_result(exc),
            json_output=json_output,
            console=console,
        )
    if json_output:
        _emit_json(result.to_dict())
        return
    console.print(f"Campaign run controller: {result.campaign.campaign_id}")
    console.print(f"- status: {result.campaign.status.value}")
    console.print(f"- shell commands: {result.shell_commands_performed}")
    console.print(f"- provider calls: {result.provider_calls_performed}")
    console.print(f"- authority: {result.authority_notice}")


# Research loop commands


@research_loop_app.command("start")
def research_loop_start(
    issue_id: str = typer.Option(
        ...,
        "--issue",
        help="Issue ID this loop targets.",
    ),
    max_attempts: int = typer.Option(
        10,
        "--max-attempts",
        help="Maximum attempts for this bounded loop.",
    ),
    loop_id: str | None = typer.Option(
        None,
        "--loop-id",
        help="Optional deterministic loop ID.",
    ),
    repo_root: Path = typer.Option(
        Path("."),
        "--repo-root",
        help="Repository root.",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Emit deterministic JSON instead of text output.",
    ),
) -> None:
    """Start a bounded research loop for one issue."""
    from cosheaf.research.loop import ResearchLoopBudget, ResearchLoopError, start_loop

    console = Console(width=120, markup=False)
    try:
        result = start_loop(
            RepoContext(repo_root),
            issue_id=issue_id,
            loop_id=loop_id,
            budget=ResearchLoopBudget(max_attempts=max_attempts),
        )
    except (ResearchLoopError, ValidationError, ValueError) as exc:
        _exit_with_error(
            _research_loop_error_result(exc),
            json_output=json_output,
            console=console,
        )
    if json_output:
        _emit_json(result.to_dict())
        return
    console.print(f"Research loop started: {result.loop.loop_id}")
    console.print(f"- path: {result.relative_path.as_posix()}")
    console.print(f"- events: {result.events_path.as_posix()}")
    console.print(f"- authority: {result.loop.authority_notice}")


@research_loop_app.command("show")
def research_loop_show(
    loop_id: str = typer.Argument(..., help="Loop ID to show."),
    repo_root: Path = typer.Option(
        Path("."),
        "--repo-root",
        help="Repository root.",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Emit deterministic JSON instead of text output.",
    ),
) -> None:
    """Show one research-loop runtime record."""
    from cosheaf.research.loop import ResearchLoopError, load_loop

    console = Console(width=120, markup=False)
    try:
        loop = load_loop(RepoContext(repo_root), loop_id)
    except (ResearchLoopError, ValidationError, ValueError) as exc:
        _exit_with_error(
            _research_loop_error_result(exc),
            json_output=json_output,
            console=console,
        )
    if json_output:
        _emit_json(loop.to_dict())
        return
    console.print(f"Research loop: {loop.loop_id}")
    console.print(f"- issue: {loop.issue_id}")
    console.print(f"- status: {loop.status.value}")
    console.print(f"- attempts: {len(loop.attempts)}")
    console.print(f"- authority: {loop.authority_notice}")


@research_loop_app.command("list")
def research_loop_list(
    repo_root: Path = typer.Option(
        Path("."),
        "--repo-root",
        help="Repository root.",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Emit deterministic JSON instead of text output.",
    ),
) -> None:
    """List research loops under the runtime directory."""
    from cosheaf.research.loop import list_loops

    loops = list_loops(RepoContext(repo_root))
    if json_output:
        _emit_json({"loops": loops})
        return
    if not loops:
        typer.echo("No research loops found.")
        return
    for item in loops:
        typer.echo(item)


@research_loop_app.command("append-attempt")
def research_loop_append_attempt(
    loop_id: str = typer.Argument(..., help="Loop ID."),
    input_json: Path = typer.Option(
        ...,
        "--input-json",
        help="ResearchLoopAttempt JSON payload.",
    ),
    repo_root: Path = typer.Option(
        Path("."),
        "--repo-root",
        help="Repository root.",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Emit deterministic JSON instead of text output.",
    ),
) -> None:
    """Append a validated terminal attempt record to a research loop."""
    from cosheaf.research.loop import (
        ResearchLoopAttempt,
        ResearchLoopError,
        append_attempt,
        load_loop,
    )

    console = Console(width=120, markup=False)
    context = RepoContext(repo_root)
    try:
        loop = load_loop(context, loop_id)
        raw = _read_input_json_or_exit(input_json, json_output=json_output)
        raw.setdefault("loop_id", loop.loop_id)
        raw.setdefault("attempt_number", len(loop.attempts) + 1)
        raw.setdefault(
            "attempt_id",
            f"{loop.loop_id}.attempt.{len(loop.attempts) + 1}",
        )
        attempt = ResearchLoopAttempt.model_validate(raw)
        result = append_attempt(context, loop.loop_id, attempt)
    except (ResearchLoopError, ValidationError, ValueError) as exc:
        _exit_with_error(
            _research_loop_error_result(exc),
            json_output=json_output,
            console=console,
        )
    if json_output:
        _emit_json(result.to_dict())
        return
    console.print(f"Research loop attempt appended: {result.attempt.attempt_id}")
    console.print(f"- status: {result.attempt.status.value}")
    console.print(f"- path: {result.relative_path.as_posix()}")
    console.print(f"- authority: {result.attempt.authority_notice}")


@research_loop_app.command("finalize")
def research_loop_finalize(
    loop_id: str = typer.Argument(..., help="Loop ID."),
    status: str = typer.Option(
        "finalized",
        "--status",
        help="Terminal status: finalized, abandoned, or failed.",
    ),
    reason: str | None = typer.Option(
        None,
        "--reason",
        help="Optional finalization reason.",
    ),
    repo_root: Path = typer.Option(
        Path("."),
        "--repo-root",
        help="Repository root.",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Emit deterministic JSON instead of text output.",
    ),
) -> None:
    """Finalize a research loop without granting accepted authority."""
    from cosheaf.research.loop import (
        ResearchLoopError,
        ResearchLoopStatus,
        load_loop,
        write_loop,
    )

    console = Console(width=120, markup=False)
    context = RepoContext(repo_root)
    try:
        loop = load_loop(context, loop_id)
        updated = loop.finalize(
            status=ResearchLoopStatus(status),
            reason=reason,
        )
        result = write_loop(context, updated)
    except (ResearchLoopError, ValidationError, ValueError) as exc:
        _exit_with_error(
            _research_loop_error_result(exc),
            json_output=json_output,
            console=console,
        )
    if json_output:
        _emit_json(result.to_dict())
        return
    console.print(f"Research loop finalized: {result.loop.loop_id}")
    console.print(f"- status: {result.loop.status.value}")
    console.print(f"- authority: {result.loop.authority_notice}")


@research_loop_app.command("next")
def research_loop_next(
    loop_id: str = typer.Argument(..., help="Loop ID."),
    repo_root: Path = typer.Option(
        Path("."),
        "--repo-root",
        help="Repository root.",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Emit deterministic JSON instead of text output.",
    ),
) -> None:
    """Preview the deterministic next action for a research loop."""
    from cosheaf.research.loop import ResearchLoopError, next_loop_action

    console = Console(width=120, markup=False)
    try:
        result = next_loop_action(RepoContext(repo_root), loop_id)
    except (ResearchLoopError, ValidationError, ValueError) as exc:
        _exit_with_error(
            _research_loop_error_result(exc),
            json_output=json_output,
            console=console,
        )
    if json_output:
        _emit_json(result.to_dict())
        return
    console.print(f"Research loop next: {result.loop_id}")
    console.print(f"- attempt: {result.attempt_id}")
    console.print(f"- action: {result.next_action.kind}")
    console.print(f"- previous failures: {len(result.previous_failures_to_avoid)}")


@research_loop_app.command("step")
def research_loop_step(
    loop_id: str = typer.Argument(..., help="Loop ID."),
    repo_root: Path = typer.Option(
        Path("."),
        "--repo-root",
        help="Repository root.",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Emit deterministic JSON instead of text output.",
    ),
) -> None:
    """Record one deterministic planning step for a research loop."""
    from cosheaf.research.loop import ResearchLoopError, step_loop

    console = Console(width=120, markup=False)
    try:
        result = step_loop(RepoContext(repo_root), loop_id)
    except (ResearchLoopError, ValidationError, ValueError) as exc:
        _exit_with_error(
            _research_loop_error_result(exc),
            json_output=json_output,
            console=console,
        )
    if json_output:
        _emit_json(result.to_dict())
        return
    console.print(f"Research loop step: {result.loop_id}")
    console.print(f"- action: {result.next_result.next_action.kind}")
    console.print(f"- event_written: {str(result.event_written).lower()}")


@research_loop_app.command("run")
def research_loop_run(
    loop_id: str = typer.Argument(..., help="Loop ID."),
    max_attempts: int = typer.Option(
        ...,
        "--max-attempts",
        help="Maximum attempts to plan in this run.",
    ),
    wallclock_minutes: int = typer.Option(
        ...,
        "--wallclock-minutes",
        help="Wallclock budget for this bounded run.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Plan next actions without writing runtime state.",
    ),
    execute_local_actions: bool = typer.Option(
        False,
        "--execute-local-actions",
        help=(
            "Execute whitelisted local actions (non-dry-run). "
            "Only safe actions allowed."
        ),
    ),
    repo_root: Path = typer.Option(
        Path("."),
        "--repo-root",
        help="Repository root.",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Emit deterministic JSON instead of text output.",
    ),
) -> None:
    """Run or dry-run a bounded deterministic research loop."""
    from cosheaf.research.loop import ResearchLoopError, run_loop

    console = Console(width=120, markup=False)
    try:
        ctx = RepoContext(repo_root)
        if execute_local_actions and not dry_run:
            from cosheaf.research.loop import load_loop
            from cosheaf.research.loop_executor import run_local_actions_step

            loop = load_loop(ctx, loop_id)
            result = run_local_actions_step(ctx, loop, dry_run=False)
        else:
            result = run_loop(
                ctx,
                loop_id,
                max_attempts=max_attempts,
                wallclock_minutes=wallclock_minutes,
                dry_run=dry_run,
            )
    except (ResearchLoopError, ValidationError, ValueError) as exc:
        _exit_with_error(
            _research_loop_error_result(exc),
            json_output=json_output,
            console=console,
        )
    if json_output:
        payload = (
            result.to_dict()
            if hasattr(result, "to_dict")
            else result.model_dump(mode="json")
        )
        _emit_json(payload)
        return
    if execute_local_actions and not dry_run:
        console.print(f"Research loop local run: {result.loop_id}")
    else:
        console.print(f"Research loop run: {result.loop_id}")
    console.print(f"- dry_run: {str(result.dry_run).lower()}")
    console.print(f"- planned_actions: {len(result.planned_actions)}")
    console.print(f"- writes_performed: {str(result.writes_performed).lower()}")


@research_loop_app.command("export-task")
def research_loop_export_task(
    loop_id: str = typer.Argument(..., help="Loop ID."),
    out: Path = typer.Option(
        ...,
        "--out",
        help="Repository-local output JSON path.",
    ),
    repo_root: Path = typer.Option(
        Path("."),
        "--repo-root",
        help="Repository root.",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Emit deterministic JSON instead of text output.",
    ),
) -> None:
    """Export a bounded external-operator task packet."""
    from cosheaf.research.loop import ResearchLoopError, export_operator_task

    console = Console(width=120, markup=False)
    try:
        relative_path = export_operator_task(RepoContext(repo_root), loop_id, out)
    except (ResearchLoopError, ValidationError, ValueError) as exc:
        _exit_with_error(
            _research_loop_error_result(exc),
            json_output=json_output,
            console=console,
        )
    payload = {
        "loop_id": loop_id,
        "path": relative_path.as_posix(),
        "writes_performed": True,
        "authority_notice": (
            "Research loop task packets are review context only; they are not "
            "accepted knowledge, human review, verifier pass, gate pass, or "
            "promotion authority."
        ),
    }
    if json_output:
        _emit_json(payload)
        return
    console.print(f"Research loop task exported: {relative_path.as_posix()}")


@research_loop_app.command("import-result")
def research_loop_import_result(
    loop_id: str = typer.Argument(..., help="Loop ID."),
    input_json: Path = typer.Option(
        ...,
        "--input-json",
        help="Structured operator_result.json payload.",
    ),
    repo_root: Path = typer.Option(
        Path("."),
        "--repo-root",
        help="Repository root.",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Emit deterministic JSON instead of text output.",
    ),
) -> None:
    """Import a structured external-operator result as a loop attempt."""
    from cosheaf.research.loop import (
        ResearchLoopError,
        ResearchLoopOperatorResult,
        import_operator_result,
    )

    console = Console(width=120, markup=False)
    try:
        raw = _read_input_json_or_exit(input_json, json_output=json_output)
        result_payload = ResearchLoopOperatorResult.model_validate(raw)
        result = import_operator_result(
            RepoContext(repo_root),
            loop_id,
            result_payload,
        )
    except (ResearchLoopError, ValidationError, ValueError) as exc:
        _exit_with_error(
            _research_loop_error_result(exc),
            json_output=json_output,
            console=console,
        )
    if json_output:
        _emit_json(result.to_dict())
        return
    console.print(f"Research loop result imported: {result.attempt_id}")
    console.print(f"- status: {result.attempt.status.value}")
    console.print(f"- path: {result.relative_path}")


@research_loop_app.command("scan")
def research_loop_scan(
    loop_id: str = typer.Argument(..., help="Loop ID."),
    repo_root: Path = typer.Option(
        Path("."),
        "--repo-root",
        help="Repository root.",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Emit deterministic JSON instead of text output.",
    ),
) -> None:
    """Scan one research loop for unsafe runtime material."""
    from cosheaf.research.loop import ResearchLoopError, scan_research_loop

    console = Console(width=120, markup=False)
    try:
        result = scan_research_loop(RepoContext(repo_root), loop_id)
    except (ResearchLoopError, ValidationError, ValueError) as exc:
        _exit_with_error(
            _research_loop_error_result(exc),
            json_output=json_output,
            console=console,
        )
    if json_output:
        _emit_json(result.to_dict())
        if result.handoff_blocked:
            raise typer.Exit(1)
        return
    console.print(f"Research loop scan: {result.loop_id}")
    console.print(f"- findings: {result.finding_count}")
    console.print(f"- blockers: {result.blocking_finding_count}")
    console.print(f"- report: {result.report_path}")
    console.print(f"- authority: {result.authority_notice}")
    if result.handoff_blocked:
        raise typer.Exit(1)


def _research_loop_error_result(exc: Exception) -> ErrorResult:
    if hasattr(exc, "code") and hasattr(exc, "remediation"):
        details = getattr(exc, "details", {})
        related_path = None
        if isinstance(details, dict):
            related_path = _valid_related_path(details.get("path"))
        return ErrorResult(
            code=str(getattr(exc, "code")),
            message=str(exc),
            remediation=str(getattr(exc, "remediation")),
            blocking=True,
            related_path=related_path,
        )
    return _exception_to_error_result(
        exc,
        default_code="research_loop_validation_failed",
    )


if __name__ == "__main__":
    app()
