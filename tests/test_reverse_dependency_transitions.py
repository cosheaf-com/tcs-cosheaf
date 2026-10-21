from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from cosheaf.cli import app
from cosheaf.core.status import ArtifactStatus
from cosheaf.gates.gatekeeper import run_gatekeeper, validate_repository
from cosheaf.gates.promotion_readiness import build_promotion_readiness_report
from cosheaf.services import PromotionActionService
from cosheaf.storage.repo import RepoContext
from cosheaf.storage.writer import write_yaml_deterministic

runner = CliRunner()


def _artifact_payload(
    artifact_id: str,
    *,
    status: str = "accepted",
    review_state: str = "accepted",
    depends_on: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "id": artifact_id,
        "type": "claim",
        "title": f"Fixture {artifact_id}",
        "domain": ["graph_theory"],
        "status": status,
        "created_at": "2026-06-19T00:00:00Z",
        "updated_at": "2026-06-19T00:00:00Z",
        "authors": ["tester"],
        "depends_on": depends_on or [],
        "supersedes": [],
        "tags": ["graph"],
        "statement": f"Statement for {artifact_id}.",
        "evidence": [
            {
                "kind": "literature",
                "path": "docs/source.md",
                "summary": "Reference note.",
            }
        ],
        "sources": [
            {
                "kind": "book",
                "title": "Graph Theory",
                "authors": ["Diestel"],
                "year": 2017,
                "theorem_number": "Theorem 1",
                "page": "1",
                "notes": "Source note.",
            }
        ],
        "review": {"state": review_state, "notes": "Reviewed."},
        "risk": {"level": "low", "notes": "Low risk."},
    }


def _setup_base_and_conclusion(repo_root: Path) -> RepoContext:
    source_doc = repo_root / "docs" / "source.md"
    source_doc.parent.mkdir(parents=True, exist_ok=True)
    source_doc.write_text("Reference note.\n", encoding="utf-8")

    write_yaml_deterministic(
        repo_root / "kb" / "accepted" / "claims" / "claim.fixture.base.yaml",
        _artifact_payload("claim.fixture.base"),
    )
    write_yaml_deterministic(
        repo_root / "kb" / "accepted" / "claims" / "claim.fixture.conclusion.yaml",
        _artifact_payload(
            "claim.fixture.conclusion",
            depends_on=["claim.fixture.base"],
        ),
    )
    return RepoContext(repo_root)


@pytest.mark.parametrize("target_status", ["draft", "refuted", "obsolete"])
def test_move_status_blocks_withdrawing_or_refuting_upstream_accepted_artifact(
    tmp_path: Path,
    target_status: str,
) -> None:
    context = _setup_base_and_conclusion(tmp_path)
    base_accepted_path = (
        tmp_path / "kb" / "accepted" / "claims" / "claim.fixture.base.yaml"
    )

    result = runner.invoke(
        app,
        [
            "artifact",
            "move-status",
            "claim.fixture.base",
            target_status,
            "--repo-root",
            str(tmp_path),
        ],
    )

    assert result.exit_code != 0
    assert "claim.fixture.conclusion" in (result.stdout + result.stderr)
    assert base_accepted_path.is_file()
    assert validate_repository(context).ok is True


def test_dependency_gate_and_gatekeeper_reject_accepted_depending_on_refuted(
    tmp_path: Path,
) -> None:
    context = _setup_base_and_conclusion(tmp_path)
    base_accepted_path = (
        tmp_path / "kb" / "accepted" / "claims" / "claim.fixture.base.yaml"
    )
    base_accepted_path.unlink()
    write_yaml_deterministic(
        tmp_path / "kb" / "refuted" / "claim.fixture.base.yaml",
        _artifact_payload("claim.fixture.base", status="refuted"),
    )

    report = validate_repository(context)
    assert report.ok is False
    assert any(
        "accepted artifact depends on refuted artifact: claim.fixture.base"
        in failure.message
        for failure in report.failures
    )

    gate_res = run_gatekeeper(context)
    assert gate_res.report.verdict == "fail"


def test_draft_with_human_review_is_consistently_ready_and_promotable(
    tmp_path: Path,
) -> None:
    context = _setup_base_and_conclusion(tmp_path)
    write_yaml_deterministic(
        tmp_path / "kb" / "draft" / "claims" / "claim.fixture.reviewed-draft.yaml",
        _artifact_payload(
            "claim.fixture.reviewed-draft",
            status="draft",
            review_state="human_reviewed",
            depends_on=["claim.fixture.base"],
        ),
    )

    readiness = build_promotion_readiness_report(
        context,
        artifact_id="claim.fixture.reviewed-draft",
    )
    assert readiness.ready is True
    assert readiness.artifacts[0].ready is True
    assert not any(
        r.code == "draft_status" for r in readiness.artifacts[0].reasons
    )

    promotion_service = PromotionActionService(context)
    preview = promotion_service.preview(
        "claim.fixture.reviewed-draft",
        target_state="accepted",
        actor="Ada Reviewer",
    )
    assert preview.target_state is ArtifactStatus.ACCEPTED

    confirmed = promotion_service.confirm(
        "claim.fixture.reviewed-draft",
        target_state="accepted",
        actor="Ada Reviewer",
    )
    assert confirmed.target_state is ArtifactStatus.ACCEPTED
    assert validate_repository(context).ok is True
