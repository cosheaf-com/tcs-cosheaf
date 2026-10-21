from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from cosheaf.gates.gatekeeper import run_gatekeeper, validate_repository
from cosheaf.gates.promotion_readiness import build_promotion_readiness_report
from cosheaf.mcp.server import McpServerError, ReadOnlyMcpServer
from cosheaf.memory import MemoryRootScope, build_artifact_cards
from cosheaf.services import MemorySearchService
from cosheaf.site.export import export_site_data
from cosheaf.storage.repo import RepoContext
from cosheaf.storage.writer import write_yaml_deterministic


def _artifact_payload(
    artifact_id: str,
    *,
    title: str = "Secret research direction",
    status: str = "draft",
    depends_on: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "id": artifact_id,
        "type": "claim",
        "title": title,
        "domain": ["secret_domain"],
        "status": status,
        "created_at": "2026-06-19T00:00:00Z",
        "updated_at": "2026-06-19T00:00:00Z",
        "authors": ["researcher"],
        "depends_on": depends_on or [],
        "supersedes": [],
        "tags": ["unpublished"],
        "statement": "Unpublished private research statement.",
        "evidence": [],
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
        "review": {"state": "human_reviewed", "notes": "Reviewed."},
        "risk": {"level": "low", "notes": "Low risk."},
    }


@pytest.mark.parametrize(
    ("kb_path", "explicit_visibility_line"),
    [
        ("kb/private", ""),
        ("kb/research", 'visibility = "private"'),
    ],
)
def test_private_root_with_custom_name_does_not_leak_via_public_mcp_or_cards(
    tmp_path: Path,
    kb_path: str,
    explicit_visibility_line: str,
) -> None:
    tmp_path.joinpath("cosheaf.toml").write_text(
        "\n".join(
            [
                "[workspace]",
                'name = "research-workspace"',
                "",
                "[[kb]]",
                'name = "commons"',
                'path = "kb/public"',
                'visibility = "public"',
                "readonly = true",
                "priority = 10",
                "",
                "[[kb]]",
                'name = "research"',
                f'path = "{kb_path}"',
                explicit_visibility_line,
                "readonly = false",
                "priority = 20",
                "",
            ]
        ),
        encoding="utf-8",
    )

    write_yaml_deterministic(
        tmp_path / kb_path / "draft" / "claims" / "claim.private.secret.yaml",
        _artifact_payload("claim.private.secret"),
    )
    context = RepoContext(tmp_path)

    all_cards = build_artifact_cards(
        context,
        allowed_scopes=(MemoryRootScope.PRIVATE,),
    )
    assert len(all_cards) == 1
    assert all_cards[0].root_scope is MemoryRootScope.PRIVATE
    assert build_artifact_cards(context) == ()

    public_cards = MemorySearchService(context).cards()
    assert public_cards == ()

    out_dir = tmp_path / "_site_data"
    exported = export_site_data(
        context,
        out=out_dir,
        public_only=True,
    )
    assert exported.public_only is True
    artifacts_json = json.loads(
        (out_dir / "artifacts.json").read_text(encoding="utf-8")
    )
    assert artifacts_json["artifacts"] == []

    mcp_server = ReadOnlyMcpServer(context)
    with pytest.raises(McpServerError) as exc_info:
        mcp_server._handle_resource_read(
            {"uri": "cosheaf://artifacts/public/claim.private.secret/card"}
        )
    assert exc_info.value.code == "private_resource_denied"


def test_workspace_policy_enforces_private_can_depend_on_public_and_custom_visibility(
    tmp_path: Path,
) -> None:
    tmp_path.joinpath("cosheaf.toml").write_text(
        "\n".join(
            [
                "[workspace]",
                'name = "isolated-workspace"',
                "",
                "[[kb]]",
                'name = "commons"',
                'path = "kb/commons"',
                'visibility = "public"',
                "readonly = false",
                "priority = 10",
                "",
                "[[kb]]",
                'name = "research"',
                'path = "kb/research"',
                'visibility = "private"',
                "readonly = false",
                "priority = 20",
                "",
                "[policy]",
                "private_can_depend_on_public = false",
                "public_can_depend_on_private = false",
                "accepted_requires_source = true",
                "",
            ]
        ),
        encoding="utf-8",
    )

    write_yaml_deterministic(
        tmp_path
        / "kb"
        / "commons"
        / "accepted"
        / "claims"
        / "claim.commons.base.yaml",
        _artifact_payload("claim.commons.base", status="accepted"),
    )
    write_yaml_deterministic(
        tmp_path
        / "kb"
        / "research"
        / "draft"
        / "claims"
        / "claim.research.derived.yaml",
        _artifact_payload(
            "claim.research.derived",
            status="locally_tested",
            depends_on=["claim.commons.base"],
        ),
    )

    context = RepoContext(tmp_path)
    report = validate_repository(context)
    assert report.ok is False
    assert any(
        "private artifact depends on public artifact: claim.commons.base"
        in failure.message
        for failure in report.failures
    )

    gate_res = run_gatekeeper(context)
    assert gate_res.report.verdict == "fail"

    readiness = build_promotion_readiness_report(
        context,
        artifact_id="claim.research.derived",
    )
    assert readiness.ready is False
    assert any(
        r.code == "dependency_risk"
        and "private artifact depends on public artifact" in r.message
        for r in readiness.artifacts[0].reasons
    )
