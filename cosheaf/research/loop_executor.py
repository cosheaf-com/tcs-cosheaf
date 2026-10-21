"""Research-loop local execution engine using the deterministic action registry.

Integrates the v0.8.0 LocalActionRegistry with the v0.7.0 bounded research loop.
Allows bounded non-dry-run execution of whitelisted local actions only.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime

from cosheaf.actions.builtins import build_default_registry
from cosheaf.actions.registry import (
    LocalActionPolicy,
    LocalActionRunRequest,
    LocalActionStatus,
)
from cosheaf.research.loop import (
    ResearchLoop,
    ResearchLoopError,
    ResearchLoopRunResult,
    ResearchLoopStopCondition,
    next_loop_action,
)
from cosheaf.storage.repo import RepoContext

_EXECUTABLE_ACTION_IDS = frozenset(
    {
        "memory.search",
        "context.build",
        "strategy.next",
        "validate.run",
        "gate.run",
        "research_loop.scan",
        "eval.research_loop",
        "operator_handoff.preview",
    }
)


def run_local_actions_step(
    context: RepoContext,
    loop: ResearchLoop,
    *,
    max_attempts: int | None = None,
    wallclock_minutes: int | None = None,
    dry_run: bool = False,
) -> ResearchLoopRunResult:
    """Execute one step of allowed local actions for a research loop.

    Runs a sequence of whitelisted local actions as one execution step.
    Scanner blockers, budget limits, or action failures stop the step.
    """
    if max_attempts is not None and max_attempts < 1:
        raise ResearchLoopError(
            "max_attempts must be at least 1",
            code="invalid_budget",
            remediation="Pass --max-attempts with a positive integer.",
        )
    if wallclock_minutes is not None and wallclock_minutes < 1:
        raise ResearchLoopError(
            "wallclock_minutes must be at least 1",
            code="invalid_budget",
            remediation="Pass --wallclock-minutes with a positive integer.",
        )

    effective_max_attempts = (
        min(max_attempts, loop.budget.max_attempts)
        if max_attempts is not None
        else loop.budget.max_attempts
    )
    if (
        wallclock_minutes is not None
        and loop.budget.max_wallclock_minutes is not None
    ):
        effective_wallclock_minutes: int | None = min(
            wallclock_minutes, loop.budget.max_wallclock_minutes
        )
    elif wallclock_minutes is not None:
        effective_wallclock_minutes = wallclock_minutes
    else:
        effective_wallclock_minutes = loop.budget.max_wallclock_minutes

    now = datetime.now(UTC).replace(microsecond=0)
    if len(loop.attempts) >= effective_max_attempts:
        planned = next_loop_action(context, loop.loop_id)
        return ResearchLoopRunResult(
            loop_id=loop.loop_id,
            mode="local",
            dry_run=dry_run,
            planned_actions=(planned,),
            writes_performed=False,
            stop_conditions=(
                ResearchLoopStopCondition(
                    condition_id=f"stop.{loop.loop_id}.max-attempts",
                    kind="max_attempts",
                    description=(
                        f"Stop when {effective_max_attempts} attempts have been reached"
                    ),
                    triggered=True,
                    triggered_at=now,
                ),
            ),
        )

    registry = build_default_registry()
    policy = LocalActionPolicy(mode="private_research")
    repo_root = context.repo_root
    issue_id = loop.issue_id or ""

    actions_to_run = [
        ("memory.search", {"issue_id": issue_id, "query": issue_id}),
        ("context.build", {"issue_id": issue_id}),
        ("strategy.next", {"issue_id": issue_id}),
    ]

    started_monotonic = time.monotonic()
    for action_id, input_refs in actions_to_run:
        if effective_wallclock_minutes is not None:
            elapsed_seconds = time.monotonic() - started_monotonic
            if elapsed_seconds > effective_wallclock_minutes * 60:
                return ResearchLoopRunResult(
                    loop_id=loop.loop_id,
                    mode="local",
                    dry_run=dry_run,
                    planned_actions=(),
                    writes_performed=False,
                    stop_conditions=(
                        ResearchLoopStopCondition(
                            condition_id=f"stop.{loop.loop_id}.wallclock",
                            kind="wallclock_exceeded",
                            description=(
                                f"Wallclock budget of {effective_wallclock_minutes} "
                                "minutes exceeded before local action execution."
                            ),
                            triggered=True,
                            triggered_at=datetime.now(UTC).replace(microsecond=0),
                        ),
                    ),
                )
        req = LocalActionRunRequest(
            action_id=action_id,
            input_refs=input_refs,
            dry_run=dry_run,
        )
        result = registry.run(req, policy, repo_root)
        if result.status in (
            LocalActionStatus.FAILED,
            LocalActionStatus.ERROR,
            LocalActionStatus.BLOCKED,
        ):
            error_code = (
                result.error.error_code
                if result.error is not None
                else result.status.value.upper()
            )
            detail_msg = (
                result.error.message
                if result.error is not None
                else str(result.status.value)
            )
            message = f"{action_id} [{error_code}]: {detail_msg}"
            return ResearchLoopRunResult(
                loop_id=loop.loop_id,
                mode="local",
                dry_run=dry_run,
                planned_actions=(),
                writes_performed=False,
                stop_conditions=(
                    ResearchLoopStopCondition(
                        condition_id=f"stop.{loop.loop_id}.local-action-failure",
                        kind="action_failure",
                        description=message,
                        triggered=True,
                        triggered_at=datetime.now(UTC).replace(microsecond=0),
                    ),
                ),
            )

    planned = next_loop_action(context, loop.loop_id)
    return ResearchLoopRunResult(
        loop_id=loop.loop_id,
        mode="local",
        dry_run=dry_run,
        planned_actions=(planned,),
        writes_performed=False,
        stop_conditions=(
            ResearchLoopStopCondition(
                condition_id=f"stop.{loop.loop_id}.max-attempts",
                kind="max_attempts",
                description=(
                    f"Stop when {effective_max_attempts} attempts have been reached"
                ),
                triggered=len(loop.attempts) >= effective_max_attempts,
                triggered_at=(
                    datetime.now(UTC).replace(microsecond=0)
                    if len(loop.attempts) >= effective_max_attempts
                    else None
                ),
            ),
        ),
    )
