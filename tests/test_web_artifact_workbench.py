from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

from cosheaf.app import open_app
from cosheaf.server import ReadOnlySiteApi


def _write_yaml(repo_root: Path, relative_path: str, data: dict[str, Any]) -> Path:
    path = repo_root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


def _read_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    return data


def _audit_entries(repo_root: Path) -> list[dict[str, Any]]:
    audit_path = repo_root / ".cosheaf" / "audit" / "web-actions.jsonl"
    return [
        json.loads(line)
        for line in audit_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _artifact_data(
    artifact_id: str = "claim.fixture.artifact-workbench",
) -> dict[str, Any]:
    return {
        "id": artifact_id,
        "type": "claim",
        "title": "Artifact workbench draft claim",
        "domain": ["testing"],
        "status": "draft",
        "created_at": "2026-06-19T00:00:00Z",
        "updated_at": "2026-06-19T00:00:00Z",
        "authors": ["tester"],
        "depends_on": [],
        "supersedes": [],
        "tags": ["artifact-workbench"],
        "statement": "A draft claim that can be edited.",
        "evidence": [{"kind": "note", "path": "docs/fixture.md", "summary": "Fixture"}],
        "review": {"state": "requested", "notes": "Preserve this review."},
        "risk": {"level": "medium", "notes": "Preserve this risk."},
        "sources": [
            {
                "kind": "paper",
                "title": "Fixture Source",
                "authors": ["Author"],
                "year": 2026,
                "doi": "",
                "arxiv": "",
                "url": "",
                "theorem_number": "",
                "page": "",
                "notes": "",
            }
        ],
    }


def _fixture_workspace(repo_root: Path) -> None:
    _write_yaml(
        repo_root,
        "kb/draft/claims/claim.fixture.artifact-workbench.yaml",
        _artifact_data(),
    )
    docs_path = repo_root / "docs" / "fixture.md"
    docs_path.parent.mkdir(parents=True, exist_ok=True)
    docs_path.write_text("Fixture evidence.\n", encoding="utf-8")


def test_web_artifact_preview_create_then_create_writes_draft_and_validates(
    tmp_path: Path,
) -> None:
    api = ReadOnlySiteApi(open_app(tmp_path))
    payload = {
        "artifact_id": "claim.fixture.web-artifact",
        "artifact_type": "claim",
        "title": "Web artifact",
        "domain": ["graph theory"],
        "status": "draft",
        "statement": "Triangle graph $K_3$ is a graph.",
        "authors": ["web-user"],
        "tags": ["graph"],
        "depends_on": [],
        "supersedes": [],
    }

    preview = api.handle(
        "POST",
        "/api/artifacts/preview-create",
        json.dumps(payload),
    )

    expected_path = "kb/draft/claims/claim.fixture.web-artifact.yaml"
    assert preview.status == 200
    assert preview.payload["kind"] == "artifact_create_preview"
    assert preview.payload["dry_run_only"] is True
    assert preview.payload["repo_writes_performed"] is False
    assert preview.payload["planned_files"] == [expected_path]
    assert "Triangle graph $K_3$" in preview.payload["yaml"]
    assert not (tmp_path / expected_path).exists()

    blocked = api.handle(
        "POST",
        "/api/artifacts/create",
        json.dumps(payload),
    )
    assert blocked.status == 400
    assert blocked.payload["code"] == "confirm_required"

    created = api.handle(
        "POST",
        "/api/artifacts/create",
        json.dumps({**payload, "confirm": True}),
    )

    assert created.status == 200
    assert created.payload["kind"] == "artifact_create"
    assert created.payload["repo_writes_performed"] is True
    assert created.payload["validation"]["ok"] is True
    written = tmp_path / expected_path
    assert written.is_file()
    assert _read_yaml(written)["status"] == "draft"

    entries = _audit_entries(tmp_path)
    assert [entry["action"] for entry in entries] == [
        "artifact.create",
        "artifact.create",
        "artifact.create",
    ]
    assert entries[0]["preview_only"] is True
    assert entries[1]["result_status"] == "confirm_required"
    assert entries[2]["repo_writes_performed"] is True


def test_web_artifact_update_preserves_review_evidence_sources_and_risk(
    tmp_path: Path,
) -> None:
    _fixture_workspace(tmp_path)
    api = ReadOnlySiteApi(open_app(tmp_path))
    artifact_path = tmp_path / "kb/draft/claims/claim.fixture.artifact-workbench.yaml"
    before = _read_yaml(artifact_path)
    payload = {
        "artifact_type": "claim",
        "title": "Updated artifact workbench draft claim",
        "domain": ["testing", "web"],
        "status": "draft",
        "statement": "Updated through the web draft editor.",
        "authors": ["tester", "web-user"],
        "tags": ["updated"],
        "depends_on": [],
        "supersedes": [],
    }

    preview = api.handle(
        "POST",
        "/api/artifacts/claim.fixture.artifact-workbench/preview-update",
        json.dumps(payload),
    )

    assert preview.status == 200
    assert preview.payload["kind"] == "artifact_update_preview"
    assert preview.payload["dry_run_only"] is True
    assert _read_yaml(artifact_path) == before
    assert "-title: Artifact workbench draft claim" in preview.payload["diff"]
    assert "+title: Updated artifact workbench draft claim" in preview.payload["diff"]

    updated = api.handle(
        "POST",
        "/api/artifacts/claim.fixture.artifact-workbench/update",
        json.dumps({**payload, "confirm": True}),
    )

    assert updated.status == 200
    assert updated.payload["kind"] == "artifact_update"
    after = _read_yaml(artifact_path)
    assert after["id"] == "claim.fixture.artifact-workbench"
    assert after["created_at"] == before["created_at"]
    assert after["title"] == "Updated artifact workbench draft claim"
    assert after["review"] == before["review"]
    assert after["evidence"] == before["evidence"]
    assert after["sources"] == before["sources"]
    assert after["risk"] == before["risk"]


def test_web_artifact_live_payload_includes_editable_text_and_metadata(
    tmp_path: Path,
) -> None:
    _fixture_workspace(tmp_path)
    api = ReadOnlySiteApi(open_app(tmp_path))

    response = api.handle("GET", "/api/artifacts/live")

    assert response.status == 200
    artifacts = {
        artifact["id"]: artifact for artifact in response.payload["artifacts"]
    }
    artifact = artifacts["claim.fixture.artifact-workbench"]
    assert artifact["statement"] == "A draft claim that can be edited."
    assert artifact["authors"] == ["tester"]
    assert artifact["supersedes"] == []


def test_web_artifact_source_metadata_preview_then_confirm_appends_source(
    tmp_path: Path,
) -> None:
    _fixture_workspace(tmp_path)
    api = ReadOnlySiteApi(open_app(tmp_path))
    artifact_path = tmp_path / "kb/draft/claims/claim.fixture.artifact-workbench.yaml"
    before = _read_yaml(artifact_path)
    payload = {
        "kind": "book",
        "title": "Graph Theory",
        "authors": ["Diestel"],
        "year": 2025,
        "page": "12",
        "notes": "Explicit source note from the web.",
    }

    preview = api.handle(
        "POST",
        "/api/artifacts/claim.fixture.artifact-workbench/preview-source",
        json.dumps(payload),
    )

    assert preview.status == 200
    assert preview.payload["kind"] == "artifact_source_preview"
    assert preview.payload["dry_run_only"] is True
    assert preview.payload["repo_writes_performed"] is False
    assert "Graph Theory" in preview.payload["diff"]
    assert "Explicit source note from the web." in preview.payload["diff"]
    assert _read_yaml(artifact_path) == before

    blocked = api.handle(
        "POST",
        "/api/artifacts/claim.fixture.artifact-workbench/source",
        json.dumps(payload),
    )
    assert blocked.status == 400
    assert blocked.payload["code"] == "confirm_required"

    confirmed = api.handle(
        "POST",
        "/api/artifacts/claim.fixture.artifact-workbench/source",
        json.dumps({**payload, "confirm": True}),
    )

    assert confirmed.status == 200
    assert confirmed.payload["kind"] == "artifact_source"
    assert confirmed.payload["validation"]["ok"] is True
    after = _read_yaml(artifact_path)
    assert after["sources"][:-1] == before["sources"]
    assert after["sources"][-1]["title"] == "Graph Theory"
    assert after["review"] == before["review"]
    entries = _audit_entries(tmp_path)
    assert entries[-1]["action"] == "source.attach"
    assert entries[-1]["repo_writes_performed"] is True


def test_web_artifact_evidence_preview_then_confirm_appends_evidence(
    tmp_path: Path,
) -> None:
    _fixture_workspace(tmp_path)
    api = ReadOnlySiteApi(open_app(tmp_path))
    artifact_path = tmp_path / "kb/draft/claims/claim.fixture.artifact-workbench.yaml"
    before = _read_yaml(artifact_path)
    payload = {
        "kind": "python_checker",
        "path": "docs/fixture.md",
        "summary": "Checker command output saved for review.",
    }

    preview = api.handle(
        "POST",
        "/api/artifacts/claim.fixture.artifact-workbench/preview-evidence",
        json.dumps(payload),
    )

    assert preview.status == 200
    assert preview.payload["kind"] == "artifact_evidence_preview"
    assert preview.payload["warnings"] == []
    assert _read_yaml(artifact_path) == before

    confirmed = api.handle(
        "POST",
        "/api/artifacts/claim.fixture.artifact-workbench/evidence",
        json.dumps({**payload, "confirm": True}),
    )

    assert confirmed.status == 200
    assert confirmed.payload["kind"] == "artifact_evidence"
    assert confirmed.payload["validation"]["ok"] is True
    after = _read_yaml(artifact_path)
    assert after["evidence"][:-1] == before["evidence"]
    assert after["evidence"][-1] == payload
    assert after["sources"] == before["sources"]
    entries = _audit_entries(tmp_path)
    assert entries[-1]["action"] == "evidence.attach"


def test_web_artifact_missing_local_evidence_warns_then_validation_refuses(
    tmp_path: Path,
) -> None:
    _fixture_workspace(tmp_path)
    api = ReadOnlySiteApi(open_app(tmp_path))
    artifact_path = tmp_path / "kb/draft/claims/claim.fixture.artifact-workbench.yaml"
    before = _read_yaml(artifact_path)
    payload = {
        "kind": "python_checker",
        "path": "docs/missing-evidence.md",
        "summary": "Missing local evidence path should not be accepted.",
    }

    preview = api.handle(
        "POST",
        "/api/artifacts/claim.fixture.artifact-workbench/preview-evidence",
        json.dumps(payload),
    )

    assert preview.status == 200
    assert preview.payload["warnings"] == [
        "missing local evidence path: docs/missing-evidence.md"
    ]
    assert _read_yaml(artifact_path) == before

    refused = api.handle(
        "POST",
        "/api/artifacts/claim.fixture.artifact-workbench/evidence",
        json.dumps({**payload, "confirm": True}),
    )

    assert refused.status == 400
    assert refused.payload["code"] == "artifact_file_validation_failed"
    assert _read_yaml(artifact_path) == before
    entries = _audit_entries(tmp_path)
    assert entries[-1]["result_status"] == "artifact_file_validation_failed"
    assert entries[-1]["repo_writes_performed"] is False


def test_web_artifact_accepted_direct_write_is_refused_and_audited(
    tmp_path: Path,
) -> None:
    api = ReadOnlySiteApi(open_app(tmp_path))
    payload = {
        "artifact_id": "claim.fixture.accepted-web-artifact",
        "artifact_type": "claim",
        "title": "Forbidden accepted artifact",
        "domain": ["testing"],
        "status": "accepted",
        "statement": "Accepted status cannot be created from the browser.",
        "authors": ["web-user"],
        "tags": [],
        "depends_on": [],
        "supersedes": [],
        "confirm": True,
    }

    refused = api.handle(
        "POST",
        "/api/artifacts/create",
        json.dumps(payload),
    )

    assert refused.status == 400
    assert refused.payload["code"] == "accepted_write_forbidden"
    assert not list((tmp_path / "kb").rglob("*.yaml"))
    entries = _audit_entries(tmp_path)
    assert entries[-1]["action"] == "artifact.create"
    assert entries[-1]["result_status"] == "accepted_write_forbidden"
    assert entries[-1]["repo_writes_performed"] is False


def test_web_artifact_write_plan_blocks_post_preview_mutation_and_stale_revision(
    tmp_path: Path,
) -> None:
    _fixture_workspace(tmp_path)
    api = ReadOnlySiteApi(open_app(tmp_path))
    payload = {
        "artifact_type": "claim",
        "title": "Previewed draft claim",
        "domain": ["testing"],
        "status": "draft",
        "statement": "Previewed statement.",
        "authors": ["tester"],
        "tags": ["previewed"],
        "depends_on": [],
        "supersedes": [],
    }

    preview = api.handle(
        "POST",
        "/api/artifacts/claim.fixture.artifact-workbench/preview-update",
        json.dumps(payload),
    )
    assert preview.status == 200
    write_plan = preview.payload["write_plan"]
    assert write_plan["action"] == "artifact_update"
    assert write_plan["artifact_id"] == "claim.fixture.artifact-workbench"
    assert write_plan["request_hash"].startswith("sha256:")
    assert write_plan["expected_revision"].startswith("sha256:")

    mutated = api.handle(
        "POST",
        "/api/artifacts/claim.fixture.artifact-workbench/update",
        json.dumps(
            {
                **payload,
                "statement": "Mutated after preview without re-previewing.",
                "confirm": True,
                "write_plan": write_plan,
            }
        ),
    )
    assert mutated.status == 400
    assert mutated.payload["code"] == "write_plan_mismatch"

    stale = api.handle(
        "POST",
        "/api/artifacts/claim.fixture.artifact-workbench/update",
        json.dumps(
            {
                **payload,
                "confirm": True,
                "write_plan": {
                    **write_plan,
                    "expected_revision": "sha256:0000000000000000",
                },
            }
        ),
    )
    assert stale.status == 400
    assert stale.payload["code"] == "stale_artifact_revision"

