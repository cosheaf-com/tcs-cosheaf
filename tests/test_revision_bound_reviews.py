from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from typer.testing import CliRunner

from cosheaf.cli import app
from cosheaf.core.artifact import (
    ArtifactRevision,
    BaseArtifact,
    build_artifact_revision,
)
from cosheaf.core.status import ArtifactStatus, ArtifactType
from cosheaf.gates.gatekeeper import validate_repository
from cosheaf.gates.promotion_readiness import build_promotion_readiness_report
from cosheaf.services import (
    DraftWriteService,
    DraftWriteServiceError,
    PromotionActionService,
    ReviewDecisionService,
)
from cosheaf.storage.repo import RepoContext
from cosheaf.storage.writer import write_yaml_deterministic
from cosheaf.verification.evidence import (
    VerifierEvidenceKind,
    VerifierEvidenceRecord,
)
from cosheaf.verification.result import VerificationStatus

runner = CliRunner()


def _create_base_and_draft(repo_root: Path) -> RepoContext:
    context = RepoContext(repo_root)
    base_data = {
        "id": "definition.graph.order",
        "type": "definition",
        "title": "Graph order",
        "domain": ["graph_theory"],
        "status": "accepted",
        "created_at": "2026-06-19T00:00:00Z",
        "updated_at": "2026-06-19T00:00:00Z",
        "authors": ["tester"],
        "depends_on": [],
        "supersedes": [],
        "tags": ["graph"],
        "statement": "The order of a graph $G=(V,E)$ is $|V|$.",
        "evidence": [
            {
                "kind": "literature",
                "path": "docs/source.md",
                "summary": "Standard textbook definition.",
            }
        ],
        "sources": [
            {
                "kind": "book",
                "title": "Graph Theory",
                "authors": ["Diestel"],
                "year": 2017,
                "theorem_number": "Section 1.1",
                "page": "2",
                "notes": "Order definition.",
            }
        ],
        "review": {"state": "accepted", "notes": "Reviewed base definition."},
        "risk": {"level": "low", "notes": "Standard definition."},
    }
    source_doc = repo_root / "docs" / "source.md"
    source_doc.parent.mkdir(parents=True, exist_ok=True)
    source_doc.write_text("Reference note.\n", encoding="utf-8")
    base_path = (
        repo_root / "kb" / "accepted" / "definitions" / "definition.graph.order.yaml"
    )
    write_yaml_deterministic(base_path, base_data)

    draft_service = DraftWriteService(context)
    draft_service.create_artifact(
        artifact_type=ArtifactType.CLAIM,
        artifact_id="claim.graph.nonempty-order",
        title="Nonempty graph order",
        domain=["graph_theory"],
        status=ArtifactStatus.HUMAN_REVIEWED,
        statement="Every nonempty finite graph has $|V| \\ge 1$.",
        authors=["tester"],
        depends_on=["definition.graph.order"],
        supersedes=[],
        tags=["graph"],
    )
    draft_service.append_source_metadata(
        "claim.graph.nonempty-order",
        {
            "kind": "book",
            "title": "Graph Theory",
            "authors": ["Diestel"],
            "year": 2017,
            "theorem_number": "Proposition 1.1",
            "page": "3",
            "notes": "Immediate from definition.",
        },
    )
    draft_service.append_evidence_metadata(
        "claim.graph.nonempty-order",
        {
            "kind": "literature",
            "path": "docs/source.md",
            "summary": "Verified against source.",
        },
    )
    return context


def test_edited_artifact_invalidates_prior_human_review_for_promotion_and_readiness(
    tmp_path: Path,
) -> None:
    context = _create_base_and_draft(tmp_path)
    draft_service = DraftWriteService(context)
    review_service = ReviewDecisionService(context)
    promotion_service = PromotionActionService(context)

    review_res = review_service.write_review_decision(
        {
            "artifact_id": "claim.graph.nonempty-order",
            "reviewer": "Ada Reviewer",
            "decision": "accept_for_public_candidate",
            "review_notes": "Verified original statement.",
            "scope": "public",
            "dependencies_checked": True,
            "sources_checked": True,
            "evidence_checked": True,
            "gate_state_acknowledged": True,
            "explicit_human_confirmation": True,
        }
    )
    original_revision = review_res.artifact.content_revision
    assert review_res.artifact.review.reviewed_revision == original_revision
    assert review_res.review["reviewed_revision"] == original_revision
    assert isinstance(build_artifact_revision(review_res.artifact), ArtifactRevision)

    updated_res = draft_service.update_artifact(
        artifact_id="claim.graph.nonempty-order",
        artifact_type=ArtifactType.CLAIM,
        title="Nonempty graph order",
        domain=["graph_theory"],
        status=ArtifactStatus.HUMAN_REVIEWED,
        statement="Every graph has $|V| \\ge 100$ (unreviewed modification).",
        authors=["tester"],
        depends_on=["definition.graph.order"],
        supersedes=[],
        tags=["graph"],
    )
    assert updated_res.artifact.content_revision != original_revision
    assert updated_res.artifact.review.reviewed_revision == original_revision

    readiness = build_promotion_readiness_report(
        context,
        artifact_id="claim.graph.nonempty-order",
    )
    target_readiness = readiness.artifacts[0]
    assert target_readiness.ready is False
    assert any(
        reason.code == "stale_review" and reason.blocking
        for reason in target_readiness.reasons
    )

    with pytest.raises(DraftWriteServiceError, match="stale_review") as exc_info:
        promotion_service.confirm(
            "claim.graph.nonempty-order",
            target_state="accepted",
            actor="Ada Reviewer",
        )
    assert exc_info.value.code == "promotion_blocked"

    cli_res = runner.invoke(
        app,
        [
            "artifact",
            "promote",
            "claim.graph.nonempty-order",
            "--repo-root",
            str(tmp_path),
        ],
    )
    assert cli_res.exit_code != 0
    assert "stale_review" in (cli_res.stdout + cli_res.stderr)

    # Recording a fresh human review for the new revision allows promotion.
    re_review = review_service.write_review_decision(
        {
            "artifact_id": "claim.graph.nonempty-order",
            "reviewer": "Ada Reviewer",
            "decision": "accept_for_public_candidate",
            "review_notes": "Re-reviewed modified statement.",
            "scope": "public",
            "dependencies_checked": True,
            "sources_checked": True,
            "evidence_checked": True,
            "gate_state_acknowledged": True,
            "explicit_human_confirmation": True,
        }
    )
    assert (
        re_review.artifact.review.reviewed_revision
        == updated_res.artifact.content_revision
    )

    promoted = promotion_service.confirm(
        "claim.graph.nonempty-order",
        target_state="accepted",
        actor="Ada Reviewer",
    )
    assert promoted.target_state is ArtifactStatus.ACCEPTED
    assert validate_repository(context).ok is True


def test_stale_evidence_target_revision_blocks_validation_readiness_and_promotion(
    tmp_path: Path,
) -> None:
    context = _create_base_and_draft(tmp_path)
    draft_service = DraftWriteService(context)
    promotion_service = PromotionActionService(context)

    records = validate_repository(context).records
    claim_loaded = next(
        rec for rec in records if rec.id == "claim.graph.nonempty-order"
    )
    assert isinstance(claim_loaded.record, BaseArtifact)
    initial_rev = claim_loaded.record.content_revision

    draft_service.append_evidence_metadata(
        "claim.graph.nonempty-order",
        {
            "kind": "python_checker",
            "path": "docs/source.md",
            "summary": "Bound to initial revision.",
            "target_revision": initial_rev,
        },
    )

    # Modify statement on disk and set human_reviewed so only evidence is stale.
    raw = claim_loaded.record.model_dump(mode="json")
    raw["statement"] = "Modified statement after evidence was recorded."
    raw["review"] = {
        "state": "human_reviewed",
        "notes": "Reviewed modified statement.",
    }
    raw["evidence"].append(
        {
            "kind": "python_checker",
            "path": "docs/source.md",
            "summary": "Bound to initial revision.",
            "target_revision": initial_rev,
        }
    )
    write_yaml_deterministic(context.resolve(claim_loaded.source_path), raw)

    val_report = validate_repository(context)
    assert val_report.ok is False
    assert any(
        "stale_evidence" in err.message or "target_revision" in err.message
        for err in val_report.failures
    )

    readiness = build_promotion_readiness_report(
        context,
        artifact_id="claim.graph.nonempty-order",
    )
    assert any(
        reason.code == "stale_evidence" and reason.blocking
        for reason in readiness.artifacts[0].reasons
    )

    with pytest.raises(DraftWriteServiceError):
        promotion_service.confirm(
            "claim.graph.nonempty-order",
            target_state="accepted",
            actor="Ada Reviewer",
        )

    ve = VerifierEvidenceRecord(
        evidence_id="evidence.verifier.test.001",
        artifact_id="claim.graph.nonempty-order",
        target_revision=initial_rev,
        verifier_kind=VerifierEvidenceKind.PYTHON,
        tool_name="pytest",
        result=VerificationStatus.PASS,
        reason_code="checked_ok",
        created_at=datetime(2026, 6, 19, tzinfo=UTC),
        limitations=("Bounded test only.",),
    )
    assert ve.to_dict()["target_revision"] == initial_rev
