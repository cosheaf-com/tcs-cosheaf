from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]
from typer.testing import CliRunner

from cosheaf.agent.model_provider import ProviderName
from cosheaf.agent.orchestrator_runner import (
    OrchestratorHostedRunConfig,
    OrchestratorHostedRunner,
)
from cosheaf.agent.providers import (
    OpenAICompatibleProvider,
    ProviderConfig,
    ProviderMode,
    ProviderTransportResult,
    ProviderTransportStatus,
)
from cosheaf.agent.task import WorkerType
from cosheaf.cli import app
from cosheaf.storage.repo import RepoContext

runner = CliRunner()


class RoleAwareTransport:
    def __init__(
        self,
        *,
        unsafe_reasoner_path: str | None = None,
    ) -> None:
        self.unsafe_reasoner_path = unsafe_reasoner_path

    def complete(self, request, config):  # type: ignore[no-untyped-def]
        if request.worker_role in {
            WorkerType.LITERATURE_SCOUT,
            WorkerType.CONSTRUCTION_SEARCHER,
        }:
            return ProviderTransportResult(
                content=json.dumps(
                    {
                        "summary": "Mocked hosted typed result.",
                        "selected_artifacts": request.context_artifact_ids,
                        "notes": ["Mocked transport only."],
                        "risk_flags": ["needs_human_review"],
                        "next_steps": ["Continue ordinary review."],
                    }
                ),
                status=ProviderTransportStatus.COMPLETED,
            )

        proposed_path = (
            self.unsafe_reasoner_path
            if request.worker_role is WorkerType.REASONER
            and self.unsafe_reasoner_path is not None
            else "kb/draft/hosted-workers/mock.yaml"
        )
        return ProviderTransportResult(
            content=json.dumps(
                {
                    "bundle_id": (
                        "bundle.issue.fixture.hosted."
                        f"{request.worker_role.value}.0001"
                    ),
                    "task_id": (
                        "task.issue.fixture.hosted."
                        f"{request.worker_role.value}"
                    ),
                    "worker_role": request.worker_role.value,
                    "created_at": "2026-06-10T00:00:00Z",
                    "summary": "Mocked hosted worker bundle.",
                    "used_artifacts": request.context_artifact_ids,
                    "used_sources": [],
                    "claims": ["Review-only mocked output."],
                    "proposed_artifacts": [
                        {
                            "path": proposed_path,
                            "summary": "Draft-only mocked proposal.",
                        }
                    ],
                    "verification_requests": ["Run validate and gate."],
                    "failures_or_counterexamples": ["No external verifier was run."],
                    "risk_flags": ["needs_human_review"],
                    "next_steps": ["Request human review."],
                    "confidence": "low",
                }
            ),
            status=ProviderTransportStatus.COMPLETED,
            raw_metadata={"transport": "mocked"},
        )


def _write_yaml(repo_root: Path, relative_path: str, data: dict[str, Any]) -> None:
    path = repo_root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def _write_workspace_config(repo_root: Path) -> None:
    repo_root.joinpath("cosheaf.toml").write_text(
        "\n".join(
            [
                "[workspace]",
                'name = "hosted-orchestrator-fixture"',
                "",
                "[[kb]]",
                'name = "public"',
                'path = "kb/public"',
                "readonly = true",
                "priority = 10",
                "",
                "[[kb]]",
                'name = "private"',
                'path = "kb/private"',
                "readonly = false",
                "priority = 20",
                "",
                "[policy]",
                "private_can_depend_on_public = true",
                "public_can_depend_on_private = false",
                "accepted_requires_source = true",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _artifact_data(
    artifact_id: str,
    *,
    title: str,
    status: str,
    statement: str,
    review_state: str = "requested",
    depends_on: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "id": artifact_id,
        "type": "claim",
        "title": title,
        "domain": ["testing"],
        "status": status,
        "created_at": "2026-06-10T00:00:00Z",
        "updated_at": "2026-06-10T00:00:00Z",
        "authors": ["tester"],
        "depends_on": depends_on or [],
        "supersedes": [],
        "tags": ["hosted-orchestrator"],
        "statement": statement,
        "evidence": [],
        "review": {
            "state": review_state,
            "notes": "Hosted orchestrator fixture review.",
        },
        "risk": {"level": "low", "notes": "Fixture risk."},
    }


def _issue_data(
    issue_id: str = "issue.fixture.hosted",
    *,
    related_artifacts: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "id": issue_id,
        "type": "issue",
        "title": "Run hosted-worker orchestrator dispatch",
        "status": "open",
        "created_at": "2026-06-10T00:00:00Z",
        "updated_at": "2026-06-10T00:00:00Z",
        "authors": ["tester"],
        "severity": "medium",
        "description": "Dispatch planned nodes through hosted-worker boundaries.",
        "related_artifacts": related_artifacts
        or [
            "claim.fixture.hosted-public",
            "claim.fixture.hosted-private",
        ],
        "tags": ["hosted-orchestrator"],
    }


def _write_repo(repo_root: Path) -> None:
    _write_workspace_config(repo_root)
    _write_yaml(
        repo_root,
        "kb/public/accepted/claims/hosted-public.yaml",
        _artifact_data(
            "claim.fixture.hosted-public",
            title="Public hosted orchestrator fixture",
            status="accepted",
            statement="Public context for hosted orchestrator dispatch.",
        ),
    )
    _write_yaml(
        repo_root,
        "kb/private/draft/claims/hosted-private.yaml",
        _artifact_data(
            "claim.fixture.hosted-private",
            title="Private hosted orchestrator fixture",
            status="draft",
            statement="Private context for hosted orchestrator dispatch.",
            depends_on=["claim.fixture.hosted-public"],
        ),
    )
    _write_yaml(repo_root, "issues/open/hosted.yaml", _issue_data())


def _assert_json(output: str) -> dict[str, Any]:
    assert "\x1b[" not in output
    payload = json.loads(output)
    assert isinstance(payload, dict)
    return payload


def test_orchestrator_run_provider_fake_json_dispatches_hosted_workers(
    tmp_path: Path,
) -> None:
    _write_repo(tmp_path)

    result = runner.invoke(
        app,
        [
            "orchestrator",
            "run",
            "--issue",
            "issue.fixture.hosted",
            "--provider",
            "fake",
            "--json",
            "--repo-root",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0, result.output
    payload = _assert_json(result.output)
    assert payload["provider"] == "fake"
    assert payload["state"] == "completed"
    assert payload["context_preview"]["private_context_included"] is False
    assert payload["context_preview"]["full_artifact_count"] == 0
    assert payload["context_preview"]["content_mode"] == "cards_only"
    assert payload["accepted_write_performed"] is False
    assert payload["hosted_network"] == "not_used"
    assert len(payload["worker_calls"]) == 4
    assert len(payload["reducer_results"]) == 2
    assert all(
        path.startswith(".cosheaf/orchestrator/issue.fixture.hosted/")
        for path in payload["provider_run_record_paths"]
    )
    assert not (tmp_path / "kb" / "accepted").exists()


def test_orchestrator_openai_compatible_requires_confirm_send(
    tmp_path: Path,
) -> None:
    _write_repo(tmp_path)

    result = runner.invoke(
        app,
        [
            "orchestrator",
            "run",
            "--issue",
            "issue.fixture.hosted",
            "--provider",
            "openai-compatible",
            "--json",
            "--repo-root",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 1
    payload = _assert_json(result.output)
    assert payload["code"] == "provider_confirm_send_required"
    assert payload["blocking"] is True
    assert not (tmp_path / "kb" / "accepted").exists()


def test_orchestrator_provider_private_context_requires_policy_and_consent(
    tmp_path: Path,
) -> None:
    _write_repo(tmp_path)

    result = runner.invoke(
        app,
        [
            "orchestrator",
            "run",
            "--issue",
            "issue.fixture.hosted",
            "--provider",
            "fake",
            "--include-private",
            "--json",
            "--repo-root",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 1
    payload = _assert_json(result.output)
    assert payload["code"] == "private_context_requires_policy"
    assert payload["blocking"] is True


def test_orchestrator_openai_compatible_confirm_send_cli_has_no_default_network(
    tmp_path: Path,
) -> None:
    _write_repo(tmp_path)

    result = runner.invoke(
        app,
        [
            "orchestrator",
            "run",
            "--issue",
            "issue.fixture.hosted",
            "--provider",
            "openai-compatible",
            "--confirm-send",
            "--json",
            "--repo-root",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 1
    payload = _assert_json(result.output)
    assert payload["provider"] == "openai-compatible"
    assert payload["hosted_network"] == "explicit_config_only"
    assert payload["state"] == "failed"
    assert payload["accepted_write_performed"] is False
    assert payload["context_preview"]["private_context_included"] is False
    assert payload["context_preview"]["full_artifact_count"] == 0
    assert payload["stop_conditions"][0]["reason"] == "hosted_worker_failed"
    assert "provider_transport_missing" in payload["stop_conditions"][0]["description"]
    assert not (tmp_path / "kb" / "accepted").exists()


def test_hosted_orchestrator_mocked_openai_path_succeeds_without_ci_network(
    tmp_path: Path,
) -> None:
    _write_repo(tmp_path)

    result = OrchestratorHostedRunner(RepoContext(tmp_path)).run_issue(
        OrchestratorHostedRunConfig(
            issue_id="issue.fixture.hosted",
            provider="openai-compatible",
            confirm_send=True,
            run_id="run.issue.fixture.hosted.mocked",
        ),
        provider_config=ProviderConfig(
            provider=ProviderName.OPENAI,
            mode=ProviderMode.OPENAI_COMPATIBLE,
            model="mock-openai-compatible",
            enabled=True,
            api_key_env=None,
        ),
        provider=OpenAICompatibleProvider(transport=RoleAwareTransport()),
    )

    assert result.run.state.value == "completed"
    assert len(result.run.worker_calls) == 4
    assert len(result.run.reducer_results) == 2
    assert result.context_preview.private_context_included is False
    assert all(path.is_file() for path in result.provider_run_record_paths)
    assert not (tmp_path / "kb" / "accepted").exists()


def test_hosted_orchestrator_reducer_rejects_unsafe_provider_bundle(
    tmp_path: Path,
) -> None:
    _write_repo(tmp_path)
    _write_yaml(
        tmp_path,
        "kb/private/draft/claims/unsafe.yaml",
        _artifact_data(
            "claim.fixture.unsafe-hosted",
            title="Unsafe hosted worker review state",
            status="draft",
            statement="Unsafe review state should be rejected by reducer.",
            review_state="human_reviewed",
        ),
    )

    result = OrchestratorHostedRunner(RepoContext(tmp_path)).run_issue(
        OrchestratorHostedRunConfig(
            issue_id="issue.fixture.hosted",
            provider="openai-compatible",
            confirm_send=True,
            run_id="run.issue.fixture.hosted.unsafe",
        ),
        provider_config=ProviderConfig(
            provider=ProviderName.OPENAI,
            mode=ProviderMode.OPENAI_COMPATIBLE,
            model="mock-openai-compatible",
            enabled=True,
            api_key_env=None,
        ),
        provider=OpenAICompatibleProvider(
            transport=RoleAwareTransport(
                unsafe_reasoner_path="kb/private/draft/claims/unsafe.yaml"
            )
        ),
    )

    assert result.run.state.value == "failed"
    assert any(
        "human_reviewed" in stop.description for stop in result.run.stop_conditions
    )
    assert not (tmp_path / "kb" / "accepted").exists()


def test_hosted_orchestrator_propagates_upstream_outputs_to_dependent_nodes(
    tmp_path: Path,
) -> None:
    _write_repo(tmp_path)
    captured_requests: list[Any] = []

    class InspectingTransport:
        def complete(self, request, config):  # type: ignore[no-untyped-def]
            captured_requests.append(request)
            role_slug = request.worker_role.value.replace("_", "-")
            if request.worker_role in {
                WorkerType.LITERATURE_SCOUT,
                WorkerType.CONSTRUCTION_SEARCHER,
            }:
                return ProviderTransportResult(
                    content=json.dumps(
                        {
                            "summary": f"Upstream {request.worker_role.value} summary.",
                            "selected_artifacts": [
                                *request.context_artifact_ids,
                                f"claim.selected.{role_slug}",
                            ],
                            "notes": [f"Note from {request.worker_role.value}."],
                            "risk_flags": ["needs_human_review"],
                            "next_steps": ["Pass to reasoner."],
                        }
                    ),
                    status=ProviderTransportStatus.COMPLETED,
                )

            proposed_path = (
                "kb/draft/hosted-workers/reasoner-proposal.yaml"
                if request.worker_role is WorkerType.REASONER
                else "kb/draft/hosted-workers/verifier-check.yaml"
            )
            return ProviderTransportResult(
                content=json.dumps(
                    {
                        "bundle_id": (
                            "bundle.issue.fixture.hosted."
                            f"{role_slug}.0001"
                        ),
                        "task_id": (
                            "task.issue.fixture.hosted."
                            f"{role_slug}"
                        ),
                        "worker_role": request.worker_role.value,
                        "created_at": "2026-06-10T00:00:00Z",
                        "summary": f"Bundle from {request.worker_role.value}.",
                        "used_artifacts": [
                            *request.context_artifact_ids,
                            f"claim.used.{role_slug}",
                        ],
                        "used_sources": [],
                        "claims": [f"Claim produced by {request.worker_role.value}."],
                        "proposed_artifacts": [
                            {
                                "path": proposed_path,
                                "summary": f"Draft from {request.worker_role.value}.",
                            }
                        ],
                        "verification_requests": [
                            f"Verify output of {request.worker_role.value}."
                        ],
                        "failures_or_counterexamples": ["None."],
                        "risk_flags": ["needs_human_review"],
                        "next_steps": ["Continue."],
                        "confidence": "low",
                    }
                ),
                status=ProviderTransportStatus.COMPLETED,
                raw_metadata={"transport": "inspecting"},
            )

    result = OrchestratorHostedRunner(RepoContext(tmp_path)).run_issue(
        OrchestratorHostedRunConfig(
            issue_id="issue.fixture.hosted",
            provider="openai-compatible",
            confirm_send=True,
            run_id="run.issue.fixture.hosted.propagation",
        ),
        provider_config=ProviderConfig(
            provider=ProviderName.OPENAI,
            mode=ProviderMode.OPENAI_COMPATIBLE,
            model="mock-openai-compatible",
            enabled=True,
            api_key_env=None,
        ),
        provider=OpenAICompatibleProvider(transport=InspectingTransport()),
    )

    assert result.run.state.value == "completed"
    assert len(captured_requests) == 4
    reasoner_req = next(
        req for req in captured_requests if req.worker_role is WorkerType.REASONER
    )
    verifier_req = next(
        req for req in captured_requests if req.worker_role is WorkerType.VERIFIER
    )
    assert "claim.selected.literature-scout" in reasoner_req.context_artifact_ids
    assert "claim.selected.literature-scout" in verifier_req.context_artifact_ids
    assert "claim.used.reasoner" in verifier_req.context_artifact_ids
    assert (
        "kb/draft/hosted-workers/reasoner-proposal.yaml" in verifier_req.prompt
    )
    assert "Claim produced by reasoner." in verifier_req.prompt
    assert "Verify output of reasoner." in verifier_req.prompt
    assert (
        "bundles/node.issue.fixture.hosted.reasoner-draft.yaml"
        in verifier_req.prompt
    )
