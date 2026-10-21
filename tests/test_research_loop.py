"""Tests for bounded research-loop models, storage, and CLI."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from typer.testing import CliRunner

from cosheaf.cli import app
from cosheaf.research.loop import (
    AttemptEvidenceSummary,
    AttemptFailureRecord,
    AttemptNextAction,
    AttemptPolicyFinding,
    LoopReviewSummary,
    ResearchLoop,
    ResearchLoopAttempt,
    ResearchLoopAttemptStatus,
    ResearchLoopBudget,
    ResearchLoopDecision,
    ResearchLoopError,
    ResearchLoopFailureTag,
    ResearchLoopOperatorResult,
    ResearchLoopStatus,
    ResearchLoopStopCondition,
    append_attempt,
    export_operator_task,
    import_operator_result,
    list_loops,
    load_loop,
    next_loop_action,
    research_loop_attempt_memory_path,
    research_loop_attempt_path,
    research_loop_events_path,
    research_loop_path,
    run_loop,
    scan_research_loop,
    start_loop,
    step_loop,
    write_loop,
)
from cosheaf.storage.repo import RepoContext

runner = CliRunner()


def _completed_at() -> datetime:
    return datetime(2026, 6, 17, 12, 0, tzinfo=UTC)


def _failure(attempt_id: str = "loop.test.attempt.1") -> AttemptFailureRecord:
    return AttemptFailureRecord(
        failure_id=f"failure.{attempt_id}",
        attempt_id=attempt_id,
        attempted_direction="Try direct induction",
        why_it_failed="The induction hypothesis is too weak",
        evidence_for_failure=("reviews/runs/failure.json",),
        related_artifacts=("claim.example",),
        related_previous_attempts=(),
        counterexample_candidate_ids=(),
        checked_counterexample_ids=(),
        verifier_or_gate_errors=("gate G6 skipped is not pass",),
        should_retry=True,
        retry_conditions="Strengthen the invariant first",
        avoid_in_future="Do not retry direct induction without a stronger invariant",
        tags=(ResearchLoopFailureTag.INSUFFICIENT_EVIDENCE,),
        signature="direct-induction:weak-hypothesis",
        occurred_at=_completed_at(),
    )


def _failed_attempt() -> ResearchLoopAttempt:
    attempt_id = "loop.test.attempt.1"
    return ResearchLoopAttempt(
        attempt_id=attempt_id,
        loop_id="loop.test",
        attempt_number=1,
        status=ResearchLoopAttemptStatus.FAILED,
        planned_direction="Try direct induction",
        completed_at=_completed_at(),
        actions_taken=("cosheaf validate", "cosheaf gate run"),
        failures=(_failure(attempt_id),),
        evidence=AttemptEvidenceSummary(
            evidence_refs=("reviews/runs/failure.json",),
            related_artifacts=("claim.example",),
        ),
    )


def test_required_models_serialize() -> None:
    """All B.1 DTOs should serialize deterministically."""
    budget = ResearchLoopBudget(max_attempts=100, max_wallclock_minutes=60)
    stop = ResearchLoopStopCondition(
        condition_id="stop.max-attempts",
        kind="max_attempts",
        description="Stop when attempts are exhausted",
    )
    decision = ResearchLoopDecision(
        decision_id="decision.loop.test.1",
        loop_id="loop.test",
        decision="append failed attempt",
        rationale="Failure memory must be preserved",
        created_at=_completed_at(),
    )
    finding = AttemptPolicyFinding(
        finding_id="finding.loop.test.1",
        severity="blocking",
        finding_type="accepted_write",
        summary="accepted KB path was referenced",
        blocking=True,
    )
    next_action = AttemptNextAction(
        action_id="action.loop.test.1",
        kind="retry_with_justification",
        summary="Retry only with strengthened invariant",
        rationale="Previous direct induction failed",
        retry_requires_justification=True,
    )
    loop = ResearchLoop(
        loop_id="loop.test",
        issue_id="issue.test",
        budget=budget,
        decisions=(decision,),
        stop_conditions=(stop,),
    )
    summary = LoopReviewSummary(
        loop_id=loop.loop_id,
        issue_id=loop.issue_id,
        status=loop.status,
        attempt_count=0,
        failed_attempt_count=0,
        succeeded_attempt_count=0,
        blocking_policy_findings=0,
    )

    payloads = [
        budget.to_dict(),
        stop.to_dict(),
        decision.to_dict(),
        finding.to_dict(),
        next_action.to_dict(),
        loop.to_dict(),
        summary.to_dict(),
    ]
    assert all(isinstance(payload, dict) for payload in payloads)
    assert loop.status == ResearchLoopStatus.CREATED
    assert "accepted status" in loop.authority_notice


def test_failed_attempt_requires_failure_record() -> None:
    """Failed terminal attempts require structured failures."""
    with pytest.raises(ValueError, match="failed attempts require failures"):
        ResearchLoopAttempt(
            attempt_id="loop.test.attempt.1",
            loop_id="loop.test",
            attempt_number=1,
            status=ResearchLoopAttemptStatus.FAILED,
            planned_direction="Try direct induction",
            completed_at=_completed_at(),
        )


def test_succeeded_attempt_requires_result_or_evidence() -> None:
    """Succeeded terminal attempts require result summary or evidence refs."""
    with pytest.raises(ValueError, match="succeeded attempts require"):
        ResearchLoopAttempt(
            attempt_id="loop.test.attempt.1",
            loop_id="loop.test",
            attempt_number=1,
            status=ResearchLoopAttemptStatus.SUCCEEDED,
            planned_direction="Try construction",
            completed_at=_completed_at(),
        )


def test_terminal_attempt_requires_completed_at() -> None:
    """Terminal attempts must record completion time."""
    with pytest.raises(ValueError, match="terminal attempts require completed_at"):
        ResearchLoopAttempt(
            attempt_id="loop.test.attempt.1",
            loop_id="loop.test",
            attempt_number=1,
            status=ResearchLoopAttemptStatus.FAILED,
            planned_direction="Try direct induction",
            failures=(_failure(),),
        )


def test_accepted_path_rejected() -> None:
    """Loop attempts cannot reference accepted KB paths."""
    with pytest.raises(ValueError, match="accepted KB paths"):
        AttemptEvidenceSummary(evidence_refs=("kb/accepted/claims/claim.x.yaml",))


def test_public_mode_rejects_private_refs() -> None:
    """Public-mode attempts must not leak private references."""
    with pytest.raises(ValueError, match="public_only"):
        ResearchLoopAttempt(
            attempt_id="loop.test.attempt.1",
            loop_id="loop.test",
            attempt_number=1,
            status=ResearchLoopAttemptStatus.SUCCEEDED,
            planned_direction="Summarize public context",
            policy_mode="public_only",
            completed_at=_completed_at(),
            result_summary="Uses a private draft",
            evidence=AttemptEvidenceSummary(evidence_refs=("kb/private/draft.yaml",)),
        )


def test_authority_overclaim_rejected() -> None:
    """Attempts cannot claim human-review, verifier, gate, or promotion authority."""
    with pytest.raises(ValueError, match="cannot claim"):
        ResearchLoopAttempt.model_validate(
            {
                "attempt_id": "loop.test.attempt.1",
                "loop_id": "loop.test",
                "attempt_number": 1,
                "status": "succeeded",
                "planned_direction": "Try construction",
                "completed_at": _completed_at().isoformat(),
                "result_summary": "Looks plausible",
                "human_reviewed": True,
            }
        )


def test_storage_paths_are_deterministic(tmp_path: Path) -> None:
    """Loop, attempt, and event paths are deterministic under .cosheaf."""
    context = RepoContext(tmp_path)
    loop = ResearchLoop(
        loop_id="loop.test",
        issue_id="issue.test",
        budget=ResearchLoopBudget(max_attempts=2),
    )
    result = write_loop(context, loop)

    assert result.relative_path == research_loop_path("loop.test")
    assert result.events_path == research_loop_events_path("loop.test")
    assert research_loop_attempt_path(
        "loop.test", "loop.test.attempt.1"
    ) == Path(".cosheaf/research-loops/loop.test/attempts/loop.test.attempt.1.json")
    assert (tmp_path / result.relative_path).exists()
    assert (tmp_path / result.events_path).exists()


def test_start_append_load_and_finalize_loop(tmp_path: Path) -> None:
    """A loop can be created, appended, shown, and finalized."""
    context = RepoContext(tmp_path)
    started = start_loop(
        context,
        issue_id="issue.test",
        loop_id="loop.test",
        budget=ResearchLoopBudget(max_attempts=2),
    )
    attempt = _failed_attempt()
    appended = append_attempt(context, "loop.test", attempt)
    loaded = load_loop(context, "loop.test")
    finalized = loaded.finalize(reason="Operator stopped after first failure")
    written = write_loop(context, finalized)

    assert started.loop.status == ResearchLoopStatus.CREATED
    assert appended.loop.status == ResearchLoopStatus.RUNNING
    assert len(loaded.attempts) == 1
    assert loaded.attempts[0].failures[0].avoid_in_future.startswith("Do not retry")
    assert written.loop.status == ResearchLoopStatus.FINALIZED
    assert "review context only" in written.loop.authority_notice
    assert (tmp_path / ".cosheaf/research-loops/loop.test/events.jsonl").exists()


def test_invalid_status_transition_after_finalize() -> None:
    """Terminal loops cannot accept more attempts."""
    loop = ResearchLoop(loop_id="loop.test", issue_id="issue.test").finalize()

    with pytest.raises(ResearchLoopError, match="status=finalized") as exc_info:
        loop.add_attempt(_failed_attempt())
    assert exc_info.value.code == "loop_terminal"


def test_attempt_number_must_be_ordered() -> None:
    """Append attempts in deterministic sequence order."""
    loop = ResearchLoop(loop_id="loop.test", issue_id="issue.test")
    attempt = _failed_attempt().model_copy(update={"attempt_number": 2})

    with pytest.raises(ResearchLoopError, match="attempt_number"):
        loop.add_attempt(attempt)


def test_list_loops(tmp_path: Path) -> None:
    """List runtime loops in deterministic order."""
    context = RepoContext(tmp_path)
    start_loop(context, issue_id="issue.b", loop_id="loop.b")
    start_loop(context, issue_id="issue.a", loop_id="loop.a")

    assert list_loops(context) == ["loop.a", "loop.b"]


def test_json_schema_files_cover_research_loop_models() -> None:
    """Schema files for research-loop DTOs should exist and be valid JSON."""
    schema_names = [
        "research_loop.schema.json",
        "research_loop_attempt.schema.json",
        "research_loop_attempt_memory.schema.json",
        "research_loop_attempt_memory_entry.schema.json",
        "attempt_failure_record.schema.json",
        "research_loop_budget.schema.json",
        "research_loop_stop_condition.schema.json",
        "research_loop_decision.schema.json",
        "research_loop_failure_cluster.schema.json",
        "research_loop_metrics.schema.json",
        "attempt_evidence_summary.schema.json",
        "attempt_policy_finding.schema.json",
        "attempt_next_action.schema.json",
        "loop_review_summary.schema.json",
        "previous_failure_summary.schema.json",
        "research_loop_next_result.schema.json",
        "research_loop_step_result.schema.json",
        "research_loop_run_result.schema.json",
        "research_loop_operator_task.schema.json",
        "operator_result_failure.schema.json",
        "research_loop_operator_result.schema.json",
        "research_loop_import_result.schema.json",
        "research_loop_scan.schema.json",
        "research_loop_scan_finding.schema.json",
    ]
    root = Path(__file__).resolve().parents[1]

    for name in schema_names:
        payload = json.loads((root / "schemas" / name).read_text(encoding="utf-8"))
        assert payload["$schema"] == "https://json-schema.org/draft/2020-12/schema"
        assert payload["title"]


def test_next_action_is_deterministic_and_surfaces_previous_failures(
    tmp_path: Path,
) -> None:
    """Next-action planning should be deterministic and failure-aware."""
    context = RepoContext(tmp_path)
    start_loop(
        context,
        issue_id="issue.test",
        loop_id="loop.test",
        budget=ResearchLoopBudget(max_attempts=3),
    )
    append_attempt(context, "loop.test", _failed_attempt())

    first = next_loop_action(context, "loop.test")
    second = next_loop_action(context, "loop.test")

    assert first.to_dict() == second.to_dict()
    assert first.attempt_id == "loop.test.attempt.2"
    assert first.next_action.kind == "retry_with_justification"
    assert first.next_action.retry_requires_justification is True
    assert len(first.previous_failures_to_avoid) == 1
    assert first.previous_failures_to_avoid[0].avoid_in_future.startswith(
        "Do not retry"
    )


def test_run_dry_run_writes_no_source_of_truth_files(tmp_path: Path) -> None:
    """Dry-run planning should not mutate loop runtime files."""
    context = RepoContext(tmp_path)
    start_loop(
        context,
        issue_id="issue.test",
        loop_id="loop.test",
        budget=ResearchLoopBudget(max_attempts=3),
    )
    root = tmp_path / ".cosheaf" / "research-loops"
    before = {
        path.relative_to(root).as_posix(): path.read_text(encoding="utf-8")
        for path in root.rglob("*")
        if path.is_file()
    }

    result = run_loop(
        context,
        "loop.test",
        max_attempts=2,
        wallclock_minutes=5,
        dry_run=True,
    )

    after = {
        path.relative_to(root).as_posix(): path.read_text(encoding="utf-8")
        for path in root.rglob("*")
        if path.is_file()
    }
    assert result.dry_run is True
    assert result.writes_performed is False
    assert len(result.planned_actions) == 2
    assert before == after


def test_step_loop_records_one_planning_event(tmp_path: Path) -> None:
    """A single planning step writes only the bounded event log entry."""
    context = RepoContext(tmp_path)
    start_loop(context, issue_id="issue.test", loop_id="loop.test")

    result = step_loop(context, "loop.test")
    assert result.events_path is not None
    events = (tmp_path / result.events_path).read_text(encoding="utf-8")

    assert result.event_written is True
    assert result.next_result.next_action.kind == "start_attempt"
    assert "next_action_planned" in events


def test_export_operator_task_writes_bounded_packet(tmp_path: Path) -> None:
    """Operator task export should write a bounded review-context packet."""
    context = RepoContext(tmp_path)
    start_loop(context, issue_id="issue.test", loop_id="loop.test")
    append_attempt(context, "loop.test", _failed_attempt())

    relative_path = export_operator_task(
        context,
        "loop.test",
        ".cosheaf/research-loops/loop.test/operator_task.json",
    )
    payload = json.loads((tmp_path / relative_path).read_text(encoding="utf-8"))
    events = (tmp_path / research_loop_events_path("loop.test")).read_text(
        encoding="utf-8"
    )

    assert payload["loop_id"] == "loop.test"
    assert payload["attempt_id"] == "loop.test.attempt.2"
    assert payload["previous_failures_to_avoid"][0]["attempt_id"] == (
        "loop.test.attempt.1"
    )
    assert "write kb/accepted" in payload["forbidden_actions"]
    assert "operator_task_exported" in events


def test_import_operator_result_writes_attempt_and_updates_failure_memory(
    tmp_path: Path,
) -> None:
    """Imported operator results should become runtime attempt memory."""
    context = RepoContext(tmp_path)
    start_loop(context, issue_id="issue.test", loop_id="loop.test")
    payload = ResearchLoopOperatorResult.model_validate(
        {
            "attempted_direction": "Try direct induction",
            "actions_taken": ["cosheaf validate"],
            "artifacts_referenced": ["claim.example"],
            "checks_run": ["cosheaf validate"],
            "failures": [
                {
                    "attempted_direction": "Try direct induction",
                    "why_it_failed": "The induction hypothesis is too weak",
                    "evidence_for_failure": ["reviews/runs/failure.json"],
                    "avoid_in_future": (
                        "Do not retry direct induction without a stronger invariant"
                    ),
                    "tags": ["insufficient_evidence"],
                }
            ],
            "evidence_refs": ["reviews/runs/failure.json"],
            "claimed_authority_flags": {"human_reviewed": False},
        }
    )

    imported = import_operator_result(context, "loop.test", payload)
    next_result = next_loop_action(context, "loop.test")

    assert imported.attempt.status == ResearchLoopAttemptStatus.FAILED
    assert imported.attempt.failures[0].failure_id == (
        "failure.loop.test.attempt.1.1"
    )
    assert next_result.previous_failures_to_avoid[0].attempt_id == (
        "loop.test.attempt.1"
    )


def test_operator_result_rejects_accepted_write_references() -> None:
    """Operator result imports must reject accepted KB references."""
    with pytest.raises(ValueError, match="accepted KB paths"):
        ResearchLoopOperatorResult.model_validate(
            {
                "attempted_direction": "Try accepted write",
                "result_summary": "Unsafe",
                "artifacts_referenced": ["kb/accepted/claims/claim.x.yaml"],
            }
        )


def test_operator_result_rejects_authority_overclaims() -> None:
    """Operator results cannot claim review, verifier, gate, or promotion."""
    with pytest.raises(ValueError, match="cannot claim"):
        ResearchLoopOperatorResult.model_validate(
            {
                "attempted_direction": "Try review spoof",
                "result_summary": "Unsafe",
                "human_reviewed": True,
            }
        )
    with pytest.raises(ValueError, match="must all be false"):
        ResearchLoopOperatorResult.model_validate(
            {
                "attempted_direction": "Try verifier spoof",
                "result_summary": "Unsafe",
                "claimed_authority_flags": {"verifier_pass": True},
            }
        )


def test_operator_result_requires_failure_or_result_summary() -> None:
    """Import payloads need either structured failure or result summary."""
    with pytest.raises(ValueError, match="failures or result_summary"):
        ResearchLoopOperatorResult.model_validate(
            {
                "attempted_direction": "Try empty result",
                "actions_taken": ["cosheaf validate"],
            }
        )


def test_budget_exhaustion_stops_next_action(tmp_path: Path) -> None:
    """Attempt budget exhaustion should produce an explicit stop condition."""
    context = RepoContext(tmp_path)
    start_loop(
        context,
        issue_id="issue.test",
        loop_id="loop.test",
        budget=ResearchLoopBudget(max_attempts=1),
    )
    append_attempt(context, "loop.test", _failed_attempt())

    result = next_loop_action(context, "loop.test")
    dry_run = run_loop(
        context,
        "loop.test",
        max_attempts=1,
        wallclock_minutes=5,
        dry_run=True,
    )

    assert result.next_action.kind == "finalize_loop"
    assert result.stop_conditions[0].triggered is True
    assert dry_run.planned_actions[0].next_action.kind == "finalize_loop"
    assert dry_run.stop_conditions[0].triggered is True


def test_cli_c1_json_smoke(tmp_path: Path) -> None:
    """CLI supports C.1 next/step/run/export/import JSON commands."""
    runner.invoke(
        app,
        [
            "research-loop",
            "start",
            "--issue",
            "issue.test",
            "--loop-id",
            "loop.test",
            "--max-attempts",
            "3",
            "--repo-root",
            str(tmp_path),
            "--json",
        ],
    )

    next_result = runner.invoke(
        app,
        [
            "research-loop",
            "next",
            "loop.test",
            "--repo-root",
            str(tmp_path),
            "--json",
        ],
    )
    assert next_result.exit_code == 0, next_result.output
    assert json.loads(next_result.output)["next_action"]["kind"] == "start_attempt"

    step = runner.invoke(
        app,
        [
            "research-loop",
            "step",
            "loop.test",
            "--repo-root",
            str(tmp_path),
            "--json",
        ],
    )
    assert step.exit_code == 0, step.output
    assert json.loads(step.output)["event_written"] is True

    run = runner.invoke(
        app,
        [
            "research-loop",
            "run",
            "loop.test",
            "--max-attempts",
            "2",
            "--wallclock-minutes",
            "5",
            "--dry-run",
            "--repo-root",
            str(tmp_path),
            "--json",
        ],
    )
    assert run.exit_code == 0, run.output
    assert json.loads(run.output)["writes_performed"] is False

    export = runner.invoke(
        app,
        [
            "research-loop",
            "export-task",
            "loop.test",
            "--out",
            ".cosheaf/research-loops/loop.test/operator_task.json",
            "--repo-root",
            str(tmp_path),
            "--json",
        ],
    )
    assert export.exit_code == 0, export.output
    assert json.loads(export.output)["writes_performed"] is True

    input_path = tmp_path / "operator_result.json"
    input_path.write_text(
        json.dumps(
            {
                "attempted_direction": "Try construction",
                "actions_taken": ["cosheaf validate"],
                "checks_run": ["cosheaf validate"],
                "result_summary": "No contradiction found",
                "evidence_refs": ["reviews/runs/success.json"],
                "claimed_authority_flags": {"accepted": False},
            }
        ),
        encoding="utf-8",
    )
    imported = runner.invoke(
        app,
        [
            "research-loop",
            "import-result",
            "loop.test",
            "--input-json",
            str(input_path),
            "--repo-root",
            str(tmp_path),
            "--json",
        ],
    )
    assert imported.exit_code == 0, imported.output
    assert json.loads(imported.output)["attempt"]["status"] == "succeeded"


def test_attempt_memory_index_clusters_repeat_failures(tmp_path: Path) -> None:
    """Appending attempts should persist deterministic clustered failure memory."""
    context = RepoContext(tmp_path)
    start_loop(context, issue_id="issue.test", loop_id="loop.test")
    first = _failed_attempt()
    second = _failed_attempt().model_copy(
        update={
            "attempt_id": "loop.test.attempt.2",
            "attempt_number": 2,
            "failures": (
                _failure("loop.test.attempt.2").model_copy(
                    update={"failure_id": "failure.loop.test.attempt.2"}
                ),
            ),
        }
    )

    append_attempt(context, "loop.test", first)
    append_attempt(context, "loop.test", second)

    payload = json.loads(
        (tmp_path / research_loop_attempt_memory_path()).read_text(encoding="utf-8")
    )
    assert payload["kind"] == "research_loop_attempt_memory"
    assert payload["entry_count"] == 2
    assert payload["cluster_count"] == 1
    assert payload["clusters"][0]["failure_count"] == 2
    assert payload["metrics"]["attempt_count"] == 2
    assert payload["metrics"]["unique_direction_count"] == 1
    assert payload["metrics"]["repeat_failure_count"] == 1


def test_next_action_uses_cross_loop_attempt_memory(tmp_path: Path) -> None:
    """Next planning should surface similar failures from earlier loops."""
    context = RepoContext(tmp_path)
    start_loop(context, issue_id="issue.test", loop_id="loop.previous")
    previous = _failed_attempt().model_copy(
        update={
            "loop_id": "loop.previous",
            "attempt_id": "loop.previous.attempt.1",
            "failures": (
                _failure("loop.previous.attempt.1").model_copy(
                    update={"failure_id": "failure.loop.previous.attempt.1"}
                ),
            ),
        }
    )
    append_attempt(context, "loop.previous", previous)
    start_loop(context, issue_id="issue.test", loop_id="loop.current")

    result = next_loop_action(context, "loop.current")

    assert result.next_action.kind == "retry_with_justification"
    assert result.next_action.retry_requires_justification is True
    assert result.previous_failures_to_avoid[0].attempt_id == (
        "loop.previous.attempt.1"
    )


def test_repeat_retry_requires_explicit_justification(tmp_path: Path) -> None:
    """Repeated failed directions require an explicit retry justification."""
    context = RepoContext(tmp_path)
    start_loop(context, issue_id="issue.test", loop_id="loop.test")
    append_attempt(context, "loop.test", _failed_attempt())
    repeated = ResearchLoopOperatorResult.model_validate(
        {
            "attempted_direction": "Try direct induction",
            "actions_taken": ["cosheaf validate"],
            "checks_run": ["cosheaf validate"],
            "result_summary": "Retrying the same direction",
            "evidence_refs": ["reviews/runs/retry.json"],
            "claimed_authority_flags": {"accepted": False},
        }
    )

    with pytest.raises(ResearchLoopError, match="repeat failure"):
        import_operator_result(context, "loop.test", repeated)

    justified = ResearchLoopOperatorResult.model_validate(
        {
            "attempted_direction": "Try direct induction",
            "actions_taken": ["cosheaf validate"],
            "checks_run": ["cosheaf validate"],
            "result_summary": "Retrying with a strengthened invariant",
            "retry_justification": "The invariant was strengthened first.",
            "evidence_refs": ["reviews/runs/retry.json"],
            "claimed_authority_flags": {"accepted": False},
        }
    )
    imported = import_operator_result(context, "loop.test", justified)
    assert imported.attempt.policy_findings[0].finding_type == "repeat_retry"
    assert imported.attempt.policy_findings[0].blocking is False


def test_research_loop_scan_blocks_leaks_and_reports_metrics(tmp_path: Path) -> None:
    """Loop scanner should fail closed on unsafe runtime material."""
    context = RepoContext(tmp_path)
    start_loop(context, issue_id="issue.test", loop_id="loop.test")
    append_attempt(
        context,
        "loop.test",
        _failed_attempt().model_copy(update={"policy_mode": "public_only"}),
    )
    events_path = tmp_path / research_loop_events_path("loop.test")
    events_path.write_text(
        json.dumps(
            {
                "sequence": 999,
                "event_kind": "unsafe_fixture",
                "payload": {
                    "provider_payload": {"messages": [{"content": "raw"}]},
                    "hidden_reasoning": "BEGIN HIDDEN REASONING",
                    "path": "kb/private/draft/claims/private.yaml",
                    "accepted_target": "kb/accepted/claims/claim.bad.yaml",
                    "env_dump": "OPENAI_API_KEY=sk-loopscansecret123456",
                    "absolute_path": "C:\\Users\\ywjhn\\private\\claim.yaml",
                    "private_context_sent": True,
                    "policy_scope": "public_only",
                    "consent_granted": False,
                    "human_reviewed": True,
                },
            },
            ensure_ascii=True,
        )
        + "\n",
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        [
            "research-loop",
            "scan",
            "loop.test",
            "--repo-root",
            str(tmp_path),
            "--json",
        ],
    )

    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["kind"] == "research_loop_scan"
    assert payload["handoff_blocked"] is True
    assert payload["metrics"]["attempt_count"] == 1
    assert payload["metrics"]["scanner_blocker_count"] >= 1
    codes = {finding["code"] for finding in payload["findings"]}
    assert codes >= {
        "secret_env_value",
        "hidden_reasoning",
        "private_path_reference",
        "accepted_write_attempt",
        "provider_payload",
        "unapproved_private_context",
        "authority_claim",
    }
    report_path = tmp_path / payload["report_path"]
    assert report_path.is_file()
    assert json.loads(report_path.read_text(encoding="utf-8")) == payload


def test_research_loop_scan_service_clean_loop(tmp_path: Path) -> None:
    """A clean loop scan should write a non-blocking report."""
    context = RepoContext(tmp_path)
    start_loop(context, issue_id="issue.test", loop_id="loop.test")
    append_attempt(context, "loop.test", _failed_attempt())

    result = scan_research_loop(context, "loop.test")

    assert result.finding_count == 0
    assert result.handoff_blocked is False
    assert result.metrics.attempt_count == 1
    assert (tmp_path / result.report_path).is_file()


def test_cli_json_smoke(tmp_path: Path) -> None:
    """CLI supports start/show/append-attempt/finalize with JSON output."""
    start = runner.invoke(
        app,
        [
            "research-loop",
            "start",
            "--issue",
            "issue.test",
            "--loop-id",
            "loop.test",
            "--max-attempts",
            "3",
            "--repo-root",
            str(tmp_path),
            "--json",
        ],
    )
    assert start.exit_code == 0, start.output
    start_payload = json.loads(start.output)
    assert start_payload["loop"]["loop_id"] == "loop.test"
    assert start_payload["relative_path"].endswith("loop.json")

    input_path = tmp_path / "attempt.json"
    input_path.write_text(json.dumps(_failed_attempt().to_dict()), encoding="utf-8")
    append = runner.invoke(
        app,
        [
            "research-loop",
            "append-attempt",
            "loop.test",
            "--input-json",
            str(input_path),
            "--repo-root",
            str(tmp_path),
            "--json",
        ],
    )
    assert append.exit_code == 0, append.output
    append_payload = json.loads(append.output)
    assert append_payload["attempt"]["status"] == "failed"

    show = runner.invoke(
        app,
        [
            "research-loop",
            "show",
            "loop.test",
            "--repo-root",
            str(tmp_path),
            "--json",
        ],
    )
    assert show.exit_code == 0, show.output
    assert json.loads(show.output)["attempts"][0]["attempt_id"] == "loop.test.attempt.1"

    finalize = runner.invoke(
        app,
        [
            "research-loop",
            "finalize",
            "loop.test",
            "--reason",
            "done",
            "--repo-root",
            str(tmp_path),
            "--json",
        ],
    )
    assert finalize.exit_code == 0, finalize.output
    assert json.loads(finalize.output)["loop"]["status"] == "finalized"


def test_research_loop_run_execute_local_actions_succeeds_in_process(
    tmp_path: Path,
) -> None:
    """--execute-local-actions runs local actions in-process without subprocesses."""
    import yaml  # type: ignore[import-untyped]

    issue_dir = tmp_path / "issues" / "open"
    issue_dir.mkdir(parents=True, exist_ok=True)
    (issue_dir / "issue.test.yaml").write_text(
        yaml.safe_dump(
            {
                "id": "issue.test",
                "type": "issue",
                "title": "Test issue for local actions",
                "status": "open",
                "created_at": "2026-04-12T00:00:00Z",
                "updated_at": "2026-04-12T00:00:00Z",
                "authors": ["tester"],
                "severity": "medium",
                "description": "Verify in-process local action execution.",
                "related_artifacts": [],
                "tags": ["test"],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    context = RepoContext(tmp_path)
    start_loop(
        context,
        issue_id="issue.test",
        loop_id="loop.test",
        budget=ResearchLoopBudget(max_attempts=3),
    )

    run_res = runner.invoke(
        app,
        [
            "research-loop",
            "run",
            "loop.test",
            "--execute-local-actions",
            "--max-attempts",
            "2",
            "--wallclock-minutes",
            "5",
            "--repo-root",
            str(tmp_path),
            "--json",
        ],
    )
    assert run_res.exit_code == 0, run_res.output
    payload = json.loads(run_res.output)
    assert payload["mode"] == "local"
    assert len(payload["planned_actions"]) == 1
    assert all(
        cond["kind"] != "action_failure" for cond in payload["stop_conditions"]
    )


def test_research_loop_run_execute_local_actions_exits_nonzero_on_failure(
    tmp_path: Path,
) -> None:
    """When a local action fails, CLI exits non-zero and reports structured error."""
    context = RepoContext(tmp_path)
    start_loop(
        context,
        issue_id="issue.missing",
        loop_id="loop.fail",
        budget=ResearchLoopBudget(max_attempts=3),
    )

    run_res = runner.invoke(
        app,
        [
            "research-loop",
            "run",
            "loop.fail",
            "--execute-local-actions",
            "--max-attempts",
            "2",
            "--wallclock-minutes",
            "5",
            "--repo-root",
            str(tmp_path),
            "--json",
        ],
    )
    assert run_res.exit_code == 1
    payload = json.loads(run_res.output)
    assert payload["stop_conditions"][0]["kind"] == "action_failure"
    assert payload["stop_conditions"][0]["triggered"] is True
    description = payload["stop_conditions"][0]["description"]
    assert "memory.search" in description
    assert "issue not found: issue.missing" in description


def test_research_loop_run_execute_local_actions_respects_attempt_budget(
    tmp_path: Path,
) -> None:
    """Local action execution stops when max_attempts budget is already exhausted."""
    context = RepoContext(tmp_path)
    start_loop(
        context,
        issue_id="issue.missing",
        loop_id="loop.exhausted",
        budget=ResearchLoopBudget(max_attempts=1),
    )
    append_attempt(
        context,
        "loop.exhausted",
        _failed_attempt().model_copy(
            update={
                "loop_id": "loop.exhausted",
                "attempt_id": "loop.exhausted.attempt.1",
            }
        ),
    )

    run_res = runner.invoke(
        app,
        [
            "research-loop",
            "run",
            "loop.exhausted",
            "--execute-local-actions",
            "--max-attempts",
            "1",
            "--wallclock-minutes",
            "5",
            "--repo-root",
            str(tmp_path),
            "--json",
        ],
    )
    assert run_res.exit_code == 0, run_res.output
    payload = json.loads(run_res.output)
    assert payload["stop_conditions"][0]["kind"] == "max_attempts"
    assert payload["stop_conditions"][0]["triggered"] is True
