"""Tests for persistent reviewable workflow runtime storage and CLI."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

from typer.testing import CliRunner

from cosheaf.actions.registry import (
    LocalActionPolicy,
    LocalActionRegistry,
    LocalActionResult,
    LocalActionRunRequest,
    LocalActionSpec,
    LocalActionStatus,
)
from cosheaf.cli import app
from cosheaf.storage.repo import RepoContext
from cosheaf.workflow.engine import (
    WORKFLOW_AUTHORITY_NOTICE,
    ReadinessClass,
    WorkflowError,
    WorkflowStep,
    assess_readiness,
    load_workflow,
    start_workflow,
    step_workflow,
    update_workflow,
    workflow_events_path,
    workflow_librarian_path,
    workflow_path,
)
from cosheaf.workflow.proposal import (
    WORKFLOW_DRAFT_PROPOSAL_AUTHORITY_NOTICE,
    build_draft_proposal,
    write_draft_proposal,
)

runner = CliRunner()


def _json(output: str) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(output))


def test_start_persists_workflow_record_and_events(tmp_path: Path) -> None:
    context = RepoContext(tmp_path)

    result = start_workflow(
        context,
        issue_id="issue.workflow.fixture",
        query="find a small graph lemma",
        workflow_id="workflow.fixture",
    )

    assert result.workflow.workflow_id == "workflow.fixture"
    assert result.relative_path == Path(
        ".cosheaf/workflows/workflow.fixture/workflow.json"
    )
    assert (tmp_path / workflow_path("workflow.fixture")).exists()
    assert (tmp_path / workflow_events_path("workflow.fixture")).exists()

    loaded = load_workflow(context, "workflow.fixture")
    assert loaded.issue_id == "issue.workflow.fixture"
    assert loaded.query == "find a small graph lemma"
    assert loaded.authority_notice == WORKFLOW_AUTHORITY_NOTICE

    events = (tmp_path / workflow_events_path("workflow.fixture")).read_text(
        encoding="utf-8"
    )
    assert "workflow_started" in events


def test_step_persists_workflow_state_and_event(tmp_path: Path) -> None:
    context = RepoContext(tmp_path)
    start_workflow(
        context,
        issue_id="issue.workflow.fixture",
        workflow_id="workflow.fixture",
    )

    result = step_workflow(
        context,
        "workflow.fixture",
        action_id="workspace.info",
        execute_local_action=False,
    )

    assert result.workflow.steps[0].action == "workspace.info"
    assert result.workflow.steps[0].status == "planned"
    loaded = load_workflow(context, "workflow.fixture")
    assert len(loaded.steps) == 1

    events = (tmp_path / workflow_events_path("workflow.fixture")).read_text(
        encoding="utf-8"
    )
    assert "workflow_step" in events


def test_planned_only_workflow_readiness_is_inconclusive(tmp_path: Path) -> None:
    context = RepoContext(tmp_path)
    start_workflow(
        context,
        issue_id="issue.workflow.fixture",
        workflow_id="workflow.fixture",
    )
    result = step_workflow(
        context,
        "workflow.fixture",
        action_id="workspace.info",
        execute_local_action=False,
    )

    readiness = assess_readiness(result.workflow)

    assert readiness.classification == ReadinessClass.INCONCLUSIVE
    assert "planned" in " ".join(readiness.blocker_details)


def test_local_action_step_rejects_non_whitelisted_action(tmp_path: Path) -> None:
    context = RepoContext(tmp_path)
    start_workflow(
        context,
        issue_id="issue.workflow.fixture",
        workflow_id="workflow.fixture",
    )

    result = step_workflow(
        context,
        "workflow.fixture",
        action_id="not.registered",
        execute_local_action=True,
    )

    step = result.workflow.steps[-1]
    assert step.status == "blocked"
    assert step.output_refs["error_code"] == "UNKNOWN_ACTION"
    assert result.workflow.readiness == ReadinessClass.BLOCKED_EVIDENCE


def test_local_action_step_blocks_accepted_write_action(tmp_path: Path) -> None:
    context = RepoContext(tmp_path)
    start_workflow(
        context,
        issue_id="issue.workflow.fixture",
        workflow_id="workflow.fixture",
    )
    registry = LocalActionRegistry()

    def _blocked_action(
        request: LocalActionRunRequest,
        policy: LocalActionPolicy,
        repo_root: Path,
    ) -> LocalActionResult:
        raise AssertionError("accepted-write action must be blocked before execution")

    registry.register(
        LocalActionSpec(
            action_id="accepted.write",
            description="unsafe accepted write fixture",
            writes_accepted=True,
        ),
        _blocked_action,
    )

    result = step_workflow(
        context,
        "workflow.fixture",
        action_id="accepted.write",
        execute_local_action=True,
        registry=registry,
    )

    step = result.workflow.steps[-1]
    assert step.status == "blocked"
    assert step.output_refs["action_status"] == LocalActionStatus.BLOCKED.value
    assert step.output_refs["error_code"] == "ACCEPTED_WRITE_BLOCKED"
    assert result.workflow.readiness == ReadinessClass.BLOCKED_SCANNER


def test_workflow_cli_start_show_step_run_and_readiness(tmp_path: Path) -> None:
    start = runner.invoke(
        app,
        [
            "workflow",
            "start",
            "--issue",
            "issue.workflow.fixture",
            "--query",
            "demo query",
            "--workflow-id",
            "workflow.fixture",
            "--repo-root",
            str(tmp_path),
            "--json",
        ],
    )
    assert start.exit_code == 0, start.output
    start_payload = _json(start.output)
    assert start_payload["workflow"]["workflow_id"] == "workflow.fixture"

    show = runner.invoke(
        app,
        [
            "workflow",
            "show",
            "workflow.fixture",
            "--repo-root",
            str(tmp_path),
            "--json",
        ],
    )
    assert show.exit_code == 0, show.output
    assert _json(show.output)["workflow_id"] == "workflow.fixture"

    step = runner.invoke(
        app,
        [
            "workflow",
            "step",
            "workflow.fixture",
            "--action",
            "workspace.info",
            "--repo-root",
            str(tmp_path),
            "--json",
        ],
    )
    assert step.exit_code == 0, step.output
    assert _json(step.output)["workflow"]["steps"][0]["status"] == "planned"

    run = runner.invoke(
        app,
        [
            "workflow",
            "run",
            "workflow.fixture",
            "--max-steps",
            "1",
            "--execute-local-actions",
            "--repo-root",
            str(tmp_path),
            "--json",
        ],
    )
    assert run.exit_code == 0, run.output
    run_payload = _json(run.output)
    assert run_payload["steps_requested"] == 1
    assert run_payload["steps_executed"] == 1
    assert len(run_payload["workflow"]["steps"]) == 2
    assert run_payload["workflow"]["steps"][-1]["status"] in {
        "success",
        "failed",
        "error",
        "blocked",
    }

    readiness = runner.invoke(
        app,
        [
            "workflow",
            "readiness",
            "workflow.fixture",
            "--repo-root",
            str(tmp_path),
            "--json",
        ],
    )
    assert readiness.exit_code == 0, readiness.output
    readiness_payload = _json(readiness.output)
    assert readiness_payload["workflow_id"] == "workflow.fixture"
    assert "classification" in readiness_payload


def test_workflow_step_rejects_accepted_output_reference() -> None:
    try:
        WorkflowStep(
            step_number=1,
            action="unsafe",
            status="success",
            output_refs={"target": "kb/accepted/claims/claim.bad.yaml"},
        )
    except ValueError as exc:
        assert "accepted KB paths" in str(exc)
    else:
        raise AssertionError("accepted output references must be rejected")


def test_workflow_cli_draft_proposal_dry_run_and_private_write(
    tmp_path: Path,
) -> None:
    context = RepoContext(tmp_path)
    start_workflow(
        context,
        issue_id="issue.workflow.fixture",
        workflow_id="workflow.fixture",
    )

    dry_run = runner.invoke(
        app,
        [
            "workflow",
            "draft-proposal",
            "workflow.fixture",
            "--repo-root",
            str(tmp_path),
            "--dry-run",
            "--json",
        ],
    )
    assert dry_run.exit_code == 0, dry_run.output
    dry_run_payload = _json(dry_run.output)
    assert dry_run_payload["written"] is False
    assert dry_run_payload["dry_run"] is True
    assert dry_run_payload["proposal"]["workflow_id"] == "workflow.fixture"
    assert dry_run_payload["proposal"]["claim_candidates"][0]["status"] == "draft"

    write = runner.invoke(
        app,
        [
            "workflow",
            "draft-proposal",
            "workflow.fixture",
            "--repo-root",
            str(tmp_path),
            "--private-root",
            "kb/private",
            "--artifact-id",
            "claim.workflow.fixture.proposal",
            "--json",
        ],
    )
    assert write.exit_code == 0, write.output
    write_payload = _json(write.output)
    assert write_payload["written"] is True
    assert write_payload["artifact_written"] is True
    assert write_payload["target_path"] == (
        "kb/private/draft/claims/claim.workflow.fixture.proposal.yaml"
    )
    written = (tmp_path / write_payload["target_path"]).read_text(encoding="utf-8")
    assert "status: draft" in written
    assert "human_reviewed" not in written


def test_workflow_cli_draft_proposal_rejects_accepted_output(
    tmp_path: Path,
) -> None:
    context = RepoContext(tmp_path)
    start_workflow(
        context,
        issue_id="issue.workflow.fixture",
        workflow_id="workflow.fixture",
    )

    result = runner.invoke(
        app,
        [
            "workflow",
            "draft-proposal",
            "workflow.fixture",
            "--repo-root",
            str(tmp_path),
            "--out",
            "kb/accepted/claims/proposal.json",
            "--json",
        ],
    )

    assert result.exit_code == 1
    payload = _json(result.output)
    assert payload["ok"] is False
    assert "accepted" in payload["error"]["message"].lower()


def test_build_draft_proposal_preserves_workflow_provenance(tmp_path: Path) -> None:
    context = RepoContext(tmp_path)
    start_workflow(
        context,
        issue_id="issue.workflow.fixture",
        workflow_id="workflow.fixture",
    )
    step_workflow(
        context,
        "workflow.fixture",
        action_id="workspace.info",
        execute_local_action=False,
    )

    proposal = build_draft_proposal(context, "workflow.fixture")

    assert proposal.workflow_id == "workflow.fixture"
    assert proposal.claim_candidates[0].status == "draft"
    assert proposal.claim_candidates[0].candidate_kind == "candidate_claim"
    assert proposal.provenance.workflow_path == (
        ".cosheaf/workflows/workflow.fixture/workflow.json"
    )
    assert proposal.provenance.librarian_path == (
        ".cosheaf/workflows/workflow.fixture/librarian.json"
    )
    assert proposal.provenance.fsm_path == (
        ".cosheaf/workflows/workflow.fixture/fsm.json"
    )
    assert proposal.provenance.loop_path == (
        ".cosheaf/workflows/workflow.fixture/loop.json"
    )
    assert proposal.provenance.action_ids == ["workspace.info"]
    assert proposal.evidence_summary.workflow_steps == 1
    assert proposal.review_checklist.human_review_required is True
    assert proposal.authority_notice == WORKFLOW_DRAFT_PROPOSAL_AUTHORITY_NOTICE


def test_draft_proposal_dry_run_writes_nothing(tmp_path: Path) -> None:
    context = RepoContext(tmp_path)
    start_workflow(
        context,
        issue_id="issue.workflow.fixture",
        workflow_id="workflow.fixture",
    )

    result = write_draft_proposal(
        context,
        "workflow.fixture",
        out=Path(".cosheaf/workflows/workflow.fixture/proposal.json"),
        dry_run=True,
    )

    assert result.written is False
    assert result.target_path == ".cosheaf/workflows/workflow.fixture/proposal.json"
    assert not (tmp_path / ".cosheaf/workflows/workflow.fixture/proposal.json").exists()


def test_draft_proposal_rejects_accepted_output_path(tmp_path: Path) -> None:
    context = RepoContext(tmp_path)
    start_workflow(
        context,
        issue_id="issue.workflow.fixture",
        workflow_id="workflow.fixture",
    )

    try:
        write_draft_proposal(
            context,
            "workflow.fixture",
            out=Path("kb/accepted/claims/proposal.json"),
        )
    except ValueError as exc:
        assert "accepted" in str(exc).lower()
    else:
        raise AssertionError("accepted output paths must be rejected")


def test_draft_proposal_rejects_public_kb_output_path(tmp_path: Path) -> None:
    (tmp_path / "cosheaf.toml").write_text(
        "\n".join(
            [
                "[workspace]",
                "name = \"workflow-proposal-fixture\"",
                "",
                "[[kb]]",
                "name = \"public\"",
                "path = \"kb/public\"",
                "readonly = true",
                "priority = 10",
                "",
                "[[kb]]",
                "name = \"private\"",
                "path = \"kb/private\"",
                "readonly = false",
                "priority = 20",
            ]
        ),
        encoding="utf-8",
    )
    context = RepoContext(tmp_path)
    start_workflow(
        context,
        issue_id="issue.workflow.fixture",
        workflow_id="workflow.fixture",
    )

    try:
        write_draft_proposal(
            context,
            "workflow.fixture",
            out=Path("kb/public/proposals/proposal.json"),
        )
    except ValueError as exc:
        assert "public" in str(exc).lower()
    else:
        raise AssertionError("public KB proposal output must be rejected")


def test_private_root_artifact_write_stays_draft(tmp_path: Path) -> None:
    context = RepoContext(tmp_path)
    start_workflow(
        context,
        issue_id="issue.workflow.fixture",
        workflow_id="workflow.fixture",
    )

    result = write_draft_proposal(
        context,
        "workflow.fixture",
        private_root=Path("kb/private"),
        artifact_id="claim.workflow.fixture.proposal",
    )

    assert result.written is True
    assert result.target_path == (
        "kb/private/draft/claims/claim.workflow.fixture.proposal.yaml"
    )
    written = (tmp_path / result.target_path).read_text(encoding="utf-8")
    assert "status: draft" in written
    assert "human_reviewed" not in written
    assert "accepted" not in written


def test_draft_proposal_rejects_candidate_authority_overclaim() -> None:
    try:
        build_draft_proposal(
            RepoContext(Path(".")),
            "workflow.missing",
            workflow_payload={
                "workflow_id": "workflow.missing",
                "issue_id": "issue.workflow.fixture",
                "query": "",
                "status": "created",
                "steps": [],
                "readiness": None,
                "created_at": "2026-06-18T00:00:00Z",
                "updated_at": "2026-06-18T00:00:00Z",
            },
            candidate_status="accepted",
        )
    except ValueError as exc:
        assert "draft" in str(exc).lower()
    else:
        raise AssertionError("accepted candidate status must be rejected")


def test_start_workflow_rejects_duplicate_id_and_preserves_existing_records(
    tmp_path: Path,
) -> None:
    context = RepoContext(tmp_path)
    start_workflow(
        context,
        issue_id="issue.workflow.first",
        query="first query",
        workflow_id="workflow.shared",
    )
    step_workflow(
        context,
        "workflow.shared",
        action_id="workspace.info",
        execute_local_action=False,
    )

    try:
        start_workflow(
            context,
            issue_id="issue.workflow.second",
            query="second query",
            workflow_id="workflow.shared",
        )
    except WorkflowError as exc:
        assert exc.code == "WORKFLOW_ALREADY_EXISTS"
    else:
        raise AssertionError("duplicate start_workflow must be rejected")

    loaded = load_workflow(context, "workflow.shared")
    assert loaded.issue_id == "issue.workflow.first"
    assert loaded.query == "first query"
    assert len(loaded.steps) == 1
    librarian = json.loads(
        (tmp_path / workflow_librarian_path("workflow.shared")).read_text(
            encoding="utf-8"
        )
    )
    assert librarian["issue_id"] == "issue.workflow.first"


def test_update_workflow_enforces_revision_and_append_only_history(
    tmp_path: Path,
) -> None:
    context = RepoContext(tmp_path)
    start_workflow(
        context,
        issue_id="issue.workflow.first",
        workflow_id="workflow.rev",
    )
    stale = load_workflow(context, "workflow.rev")
    assert stale.revision == 1

    step_workflow(
        context,
        "workflow.rev",
        action_id="workspace.info",
        execute_local_action=False,
    )
    current = load_workflow(context, "workflow.rev")
    assert current.revision == 2

    try:
        update_workflow(context, stale, expected_revision=1)
    except WorkflowError as exc:
        assert exc.code == "WORKFLOW_REVISION_CONFLICT"
    else:
        raise AssertionError("stale revision update must be rejected")

    try:
        update_workflow(context, current.model_copy(update={"steps": []}))
    except WorkflowError as exc:
        assert exc.code == "WORKFLOW_HISTORY_CONFLICT"
    else:
        raise AssertionError("truncating workflow steps must be rejected")

    try:
        update_workflow(
            context,
            current.model_copy(update={"issue_id": "issue.workflow.other"}),
        )
    except WorkflowError as exc:
        assert exc.code == "WORKFLOW_CONFLICT"
    else:
        raise AssertionError("changing workflow issue_id must be rejected")
