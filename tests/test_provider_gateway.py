from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import pytest
from pydantic import ValidationError

from cosheaf.agent.model_provider import (
    FinishReason,
    NetworkPolicy,
    ProviderName,
    ReasoningEffort,
    ToolPolicy,
)
from cosheaf.agent.providers import (
    OpenAICompatibleProvider,
    ProviderConfig,
    ProviderError,
    ProviderGateway,
    ProviderGatewayRequest,
    ProviderMode,
    ProviderTransportResult,
    ProviderTransportStatus,
)
from cosheaf.agent.task import WorkerType
from cosheaf.services.models import (
    ContextPolicyMode,
    ModelCallResult,
    ProviderConsent,
    ProviderRunStatus,
)
from cosheaf.storage.repo import RepoContext


class RecordingTransport:
    def __init__(self, result: ProviderTransportResult) -> None:
        self.result = result
        self.calls: list[tuple[ProviderGatewayRequest, ProviderConfig]] = []

    def complete(
        self,
        request: ProviderGatewayRequest,
        config: ProviderConfig,
    ) -> ProviderTransportResult:
        self.calls.append((request, config))
        return self.result


class SequenceTransport:
    def __init__(self, results: Sequence[ProviderTransportResult]) -> None:
        self.results = list(results)
        self.calls: list[tuple[ProviderGatewayRequest, ProviderConfig]] = []

    def complete(
        self,
        request: ProviderGatewayRequest,
        config: ProviderConfig,
    ) -> ProviderTransportResult:
        self.calls.append((request, config))
        if not self.results:
            raise AssertionError("transport called more times than expected")
        return self.results.pop(0)


def _public_consent() -> ProviderConsent:
    return ProviderConsent(
        consent_required=False,
        consent_granted=False,
        allow_private_context=False,
        policy_scope=ContextPolicyMode.PUBLIC,
    )


def _request(
    *,
    provider: ProviderName = ProviderName.FAKE,
    model: str = "fake-deterministic",
    prompt: str = "Return a review-only worker bundle.",
    output_kind: Literal["text", "worker_bundle", "draft", "proposal"] = "text",
    expected_output_paths: Sequence[str] = (),
    network_policy: NetworkPolicy = NetworkPolicy.DISABLED,
    reasoning_effort: ReasoningEffort | None = None,
    tool_policy: ToolPolicy = ToolPolicy.NONE,
) -> ProviderGatewayRequest:
    return ProviderGatewayRequest(
        provider=provider,
        model=model,
        worker_role=WorkerType.REASONER,
        prompt=prompt,
        consent=_public_consent(),
        context_artifact_ids=["definition.graph"],
        root_scopes=["public"],
        output_kind=output_kind,
        expected_output_paths=list(expected_output_paths),
        network_policy=network_policy,
        reasoning_effort=reasoning_effort,
        tool_policy=tool_policy,
    )


def _worker_bundle_json(*, proposed_path: str = "kb/draft/claims/provider.yaml") -> str:
    return json.dumps(
        {
            "bundle_id": "bundle.issue.provider.reasoner.0001",
            "task_id": "task.issue.provider.reasoner",
            "worker_role": "reasoner",
            "created_at": "2026-06-10T00:00:00Z",
            "summary": "Provider returned a review-only worker bundle.",
            "used_artifacts": ["definition.graph"],
            "used_sources": [],
            "claims": ["This output remains review context."],
            "proposed_artifacts": [
                {
                    "path": proposed_path,
                    "summary": "Draft-only provider proposal.",
                }
            ],
            "verification_requests": ["Run validate and gate before review."],
            "failures_or_counterexamples": ["No theorem prover was run."],
            "risk_flags": ["needs_human_review"],
            "next_steps": ["Request explicit human review."],
            "confidence": "low",
        }
    )


def _read_provider_log(repo_root: Path, result: ModelCallResult) -> dict[str, object]:
    assert result.provider_run.log_path is not None
    raw = json.loads((repo_root / result.provider_run.log_path).read_text())
    assert isinstance(raw, dict)
    return raw


def test_gateway_fake_provider_returns_redacted_run_record(tmp_path: Path) -> None:
    gateway = ProviderGateway(RepoContext(tmp_path))
    result = gateway.call(
        _request(
            prompt="Use sk-test-secret only as a redaction fixture.",
            expected_output_paths=["kb/private/draft/claims/provider-demo.yaml"],
        )
    )

    assert isinstance(result, ModelCallResult)
    assert result.status is ProviderRunStatus.COMPLETED
    assert result.provider is ProviderName.FAKE
    assert result.provider_run.private_context_sent is False
    assert result.provider_run.log_path == ".cosheaf/providers/run.provider.0001.json"
    assert "sk-test-secret" not in result.content
    assert "<redacted>" in result.content

    log_path = tmp_path / result.provider_run.log_path
    log = json.loads(log_path.read_text(encoding="utf-8"))
    assert log["provider"] == "fake"
    assert log["model"] == "fake-deterministic"
    assert log["private_context_sent"] is False
    assert log["redaction_applied"] is True
    assert "sk-test-secret" not in log_path.read_text(encoding="utf-8")
    assert log["output_paths"] == ["kb/private/draft/claims/provider-demo.yaml"]


def test_gateway_requires_enabled_openai_compatible_provider(tmp_path: Path) -> None:
    gateway = ProviderGateway(RepoContext(tmp_path))
    result = gateway.call(
        _request(
            provider=ProviderName.OPENAI,
            model="gpt-test",
            network_policy=NetworkPolicy.EXPLICIT_ALLOW,
        ),
        config=ProviderConfig(
            provider=ProviderName.OPENAI,
            mode=ProviderMode.OPENAI_COMPATIBLE,
            model="gpt-test",
            enabled=False,
            api_key_env="COSHEAF_TEST_API_KEY",
        ),
        provider=OpenAICompatibleProvider(
            transport=RecordingTransport(
                ProviderTransportResult(
                    content="unused",
                    status=ProviderTransportStatus.COMPLETED,
                )
            )
        ),
    )

    assert isinstance(result, ProviderError)
    assert result.code == "provider_disabled"
    assert result.blocking is True


def test_gateway_requires_api_key_for_openai_compatible_provider(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("COSHEAF_TEST_API_KEY", raising=False)
    transport = RecordingTransport(
        ProviderTransportResult(
            content="unused",
            status=ProviderTransportStatus.COMPLETED,
        )
    )
    gateway = ProviderGateway(RepoContext(tmp_path))

    result = gateway.call(
        _request(
            provider=ProviderName.OPENAI,
            model="gpt-test",
            network_policy=NetworkPolicy.EXPLICIT_ALLOW,
        ),
        config=ProviderConfig(
            provider=ProviderName.OPENAI,
            mode=ProviderMode.OPENAI_COMPATIBLE,
            model="gpt-test",
            enabled=True,
            api_key_env="COSHEAF_TEST_API_KEY",
        ),
        provider=OpenAICompatibleProvider(transport=transport),
    )

    assert isinstance(result, ProviderError)
    assert result.code == "provider_api_key_missing"
    assert transport.calls == []


def test_openai_compatible_provider_uses_mock_transport_and_redacts_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("COSHEAF_TEST_API_KEY", "sk-test-secret")
    transport = RecordingTransport(
        ProviderTransportResult(
            content=_worker_bundle_json(),
            status=ProviderTransportStatus.COMPLETED,
            latency_ms=42,
            input_tokens=12,
            output_tokens=8,
            cost_usd="0.0001",
            finish_reason=FinishReason.STOP,
            raw_metadata={
                "Authorization": "Bearer sk-test-secret",
                "request_id": "req_123",
            },
        )
    )
    gateway = ProviderGateway(RepoContext(tmp_path))

    result = gateway.call(
        _request(
            provider=ProviderName.OPENAI,
            model="gpt-test",
            network_policy=NetworkPolicy.EXPLICIT_ALLOW,
            output_kind="worker_bundle",
        ),
        config=ProviderConfig(
            provider=ProviderName.OPENAI,
            mode=ProviderMode.OPENAI_COMPATIBLE,
            model="gpt-test",
            enabled=True,
            api_key_env="COSHEAF_TEST_API_KEY",
            timeout_seconds=10,
            max_retries=2,
        ),
        provider=OpenAICompatibleProvider(transport=transport),
    )

    assert isinstance(result, ModelCallResult)
    assert result.status is ProviderRunStatus.COMPLETED
    assert result.provider is ProviderName.OPENAI
    assert transport.calls[0][1].api_key_env == "COSHEAF_TEST_API_KEY"

    assert result.provider_run.log_path is not None
    log = (tmp_path / result.provider_run.log_path).read_text(encoding="utf-8")
    payload = json.loads(log)
    assert "sk-test-secret" not in log
    assert "<redacted>" in log
    assert payload["latency_ms"] == 42
    assert payload["cost_usd"] == "0.0001"
    assert payload["attempt_count"] == 1
    assert payload["unsupported_parameters"] == []


def test_openai_compatible_provider_reports_unsupported_parameters(
    tmp_path: Path,
) -> None:
    transport = RecordingTransport(
        ProviderTransportResult(
            content=_worker_bundle_json(),
            status=ProviderTransportStatus.COMPLETED,
        )
    )
    gateway = ProviderGateway(RepoContext(tmp_path))

    result = gateway.call(
        _request(
            provider=ProviderName.OPENAI,
            model="gpt-test",
            output_kind="worker_bundle",
            reasoning_effort=ReasoningEffort.HIGH,
            tool_policy=ToolPolicy.READ_ONLY,
        ),
        config=ProviderConfig(
            provider=ProviderName.OPENAI,
            mode=ProviderMode.OPENAI_COMPATIBLE,
            model="gpt-test",
            enabled=True,
            api_key_env=None,
            supported_parameters=("prompt", "max_output_tokens"),
        ),
        provider=OpenAICompatibleProvider(transport=transport),
    )

    assert isinstance(result, ModelCallResult)
    assert any(
        "unsupported provider parameters" in warning for warning in result.warnings
    )
    log = _read_provider_log(tmp_path, result)
    assert log["unsupported_parameters"] == ["reasoning_effort", "tool_policy"]


def test_gateway_maps_timeout_and_rate_limit_to_provider_errors(tmp_path: Path) -> None:
    gateway = ProviderGateway(RepoContext(tmp_path))

    timeout = gateway.call(
        _request(provider=ProviderName.OPENAI, model="gpt-test"),
        config=ProviderConfig(
            provider=ProviderName.OPENAI,
            mode=ProviderMode.OPENAI_COMPATIBLE,
            model="gpt-test",
            enabled=True,
            api_key_env=None,
        ),
        provider=OpenAICompatibleProvider(
            transport=RecordingTransport(
                ProviderTransportResult(
                    content="",
                    status=ProviderTransportStatus.TIMEOUT,
                    error_code="transport_timeout",
                    error_message="request timed out",
                )
            )
        ),
    )
    rate_limited = gateway.call(
        _request(provider=ProviderName.OPENAI, model="gpt-test"),
        config=ProviderConfig(
            provider=ProviderName.OPENAI,
            mode=ProviderMode.OPENAI_COMPATIBLE,
            model="gpt-test",
            enabled=True,
            api_key_env=None,
        ),
        provider=OpenAICompatibleProvider(
            transport=RecordingTransport(
                ProviderTransportResult(
                    content="",
                    status=ProviderTransportStatus.RATE_LIMITED,
                    error_code="rate_limited",
                    error_message="rate limited",
                )
            )
        ),
    )

    assert isinstance(timeout, ProviderError)
    assert timeout.code == "provider_timeout"
    assert isinstance(rate_limited, ProviderError)
    assert rate_limited.code == "provider_rate_limited"


def test_gateway_retries_rate_limited_transport_once(tmp_path: Path) -> None:
    transport = SequenceTransport(
        [
            ProviderTransportResult(
                content="",
                status=ProviderTransportStatus.RATE_LIMITED,
                error_code="rate_limited",
                error_message="rate limited",
            ),
            ProviderTransportResult(
                content=_worker_bundle_json(),
                status=ProviderTransportStatus.COMPLETED,
                latency_ms=15,
            ),
        ]
    )
    gateway = ProviderGateway(RepoContext(tmp_path))

    result = gateway.call(
        _request(
            provider=ProviderName.OPENAI,
            model="gpt-test",
            output_kind="worker_bundle",
        ),
        config=ProviderConfig(
            provider=ProviderName.OPENAI,
            mode=ProviderMode.OPENAI_COMPATIBLE,
            model="gpt-test",
            enabled=True,
            api_key_env=None,
            max_retries=1,
        ),
        provider=OpenAICompatibleProvider(transport=transport),
    )

    assert isinstance(result, ModelCallResult)
    assert result.status is ProviderRunStatus.COMPLETED
    assert len(transport.calls) == 2

    log = _read_provider_log(tmp_path, result)
    assert log["attempt_count"] == 2
    assert log["retry_statuses"] == ["rate_limited"]


def test_gateway_maps_cancelled_transport_to_provider_error(tmp_path: Path) -> None:
    gateway = ProviderGateway(RepoContext(tmp_path))

    result = gateway.call(
        _request(provider=ProviderName.OPENAI, model="gpt-test"),
        config=ProviderConfig(
            provider=ProviderName.OPENAI,
            mode=ProviderMode.OPENAI_COMPATIBLE,
            model="gpt-test",
            enabled=True,
            api_key_env=None,
        ),
        provider=OpenAICompatibleProvider(
            transport=RecordingTransport(
                ProviderTransportResult(
                    content="",
                    status=ProviderTransportStatus.CANCELLED,
                    error_code="cancelled",
                    error_message="operator cancelled provider call sk-cancel-secret",
                    raw_metadata={"Authorization": "Bearer sk-cancel-secret"},
                )
            )
        ),
    )

    assert isinstance(result, ProviderError)
    assert result.code == "provider_cancelled"
    log_path = result.details["log_path"]
    log_text = (tmp_path / log_path).read_text(encoding="utf-8")
    log = json.loads(log_text)
    assert log["status"] == "failed"
    assert log["error_code"] == "provider_cancelled"
    assert "sk-cancel-secret" not in log_text
    assert "<redacted>" in log_text


def test_gateway_validates_worker_bundle_output_schema(tmp_path: Path) -> None:
    gateway = ProviderGateway(RepoContext(tmp_path))
    malformed = gateway.call(
        _request(
            provider=ProviderName.OPENAI,
            model="gpt-test",
            output_kind="worker_bundle",
        ),
        config=ProviderConfig(
            provider=ProviderName.OPENAI,
            mode=ProviderMode.OPENAI_COMPATIBLE,
            model="gpt-test",
            enabled=True,
            api_key_env=None,
        ),
        provider=OpenAICompatibleProvider(
            transport=RecordingTransport(
                ProviderTransportResult(
                    content="not json",
                    status=ProviderTransportStatus.COMPLETED,
                )
            )
        ),
    )

    assert isinstance(malformed, ProviderError)
    assert malformed.code == "provider_output_validation_failed"

    unsafe = gateway.call(
        _request(
            provider=ProviderName.OPENAI,
            model="gpt-test",
            output_kind="worker_bundle",
        ),
        config=ProviderConfig(
            provider=ProviderName.OPENAI,
            mode=ProviderMode.OPENAI_COMPATIBLE,
            model="gpt-test",
            enabled=True,
            api_key_env=None,
        ),
        provider=OpenAICompatibleProvider(
            transport=RecordingTransport(
                ProviderTransportResult(
                    content=_worker_bundle_json(
                        proposed_path="kb/accepted/claims/unsafe.yaml"
                    ),
                    status=ProviderTransportStatus.COMPLETED,
                )
            )
        ),
    )

    assert isinstance(unsafe, ProviderError)
    assert unsafe.code == "provider_output_validation_failed"
    assert "accepted knowledge" in unsafe.message


def test_gateway_retries_malformed_worker_bundle_once_with_schema_reminder(
    tmp_path: Path,
) -> None:
    transport = SequenceTransport(
        [
            ProviderTransportResult(
                content="not json",
                status=ProviderTransportStatus.COMPLETED,
            ),
            ProviderTransportResult(
                content=_worker_bundle_json(),
                status=ProviderTransportStatus.COMPLETED,
                latency_ms=12,
            ),
        ]
    )
    gateway = ProviderGateway(RepoContext(tmp_path))

    result = gateway.call(
        _request(
            provider=ProviderName.OPENAI,
            model="gpt-test",
            output_kind="worker_bundle",
        ),
        config=ProviderConfig(
            provider=ProviderName.OPENAI,
            mode=ProviderMode.OPENAI_COMPATIBLE,
            model="gpt-test",
            enabled=True,
            api_key_env=None,
            max_retries=1,
        ),
        provider=OpenAICompatibleProvider(transport=transport),
    )

    assert isinstance(result, ModelCallResult)
    assert len(transport.calls) == 2
    assert "Previous provider output failed WorkerBundle v2 validation" in (
        transport.calls[1][0].prompt
    )
    assert "Validation error code: provider_output_validation_failed" in (
        transport.calls[1][0].prompt
    )
    log = _read_provider_log(tmp_path, result)
    assert log["attempt_count"] == 2
    assert log["output_validation_retry_count"] == 1
    assert log["output_validation_retry_code"] == "provider_output_validation_failed"
    assert log["output_validation_retry_status"] == "completed"
    assert log["output_validation_retry_final_status"] == "completed"


def test_gateway_logs_malformed_worker_bundle_retry_failure(
    tmp_path: Path,
) -> None:
    transport = SequenceTransport(
        [
            ProviderTransportResult(
                content="not json",
                status=ProviderTransportStatus.COMPLETED,
            ),
            ProviderTransportResult(
                content=json.dumps(["still", "not", "a", "bundle"]),
                status=ProviderTransportStatus.COMPLETED,
            ),
        ]
    )
    gateway = ProviderGateway(RepoContext(tmp_path))

    result = gateway.call(
        _request(
            provider=ProviderName.OPENAI,
            model="gpt-test",
            output_kind="worker_bundle",
        ),
        config=ProviderConfig(
            provider=ProviderName.OPENAI,
            mode=ProviderMode.OPENAI_COMPATIBLE,
            model="gpt-test",
            enabled=True,
            api_key_env=None,
            max_retries=1,
        ),
        provider=OpenAICompatibleProvider(transport=transport),
    )

    assert isinstance(result, ProviderError)
    assert result.code == "provider_output_validation_failed"
    assert len(transport.calls) == 2
    log_path = result.details["log_path"]
    log = json.loads((tmp_path / log_path).read_text(encoding="utf-8"))
    assert log["status"] == "failed"
    assert log["error_code"] == "provider_output_validation_failed"
    assert log["attempt_count"] == 2
    assert log["output_validation_retry_count"] == 1
    assert log["output_validation_retry_code"] == "provider_output_validation_failed"
    assert log["output_validation_retry_status"] == "completed"
    assert log["output_validation_retry_final_status"] == "failed"
    assert not (tmp_path / "kb").exists()


def test_gateway_request_rejects_accepted_output_paths() -> None:
    with pytest.raises(ValidationError, match="accepted knowledge"):
        _request(expected_output_paths=["kb/accepted/claims/provider.yaml"])


def test_consecutive_provider_calls_preserve_distinct_run_logs(tmp_path: Path) -> None:
    context = RepoContext(tmp_path)
    first = ProviderGateway(context).call(_request(prompt="First distinct prompt."))
    second = ProviderGateway(context).call(_request(prompt="Second distinct prompt."))
    third = ProviderGateway(context).call(_request(prompt="First distinct prompt."))

    assert isinstance(first, ModelCallResult)
    assert isinstance(second, ModelCallResult)
    assert isinstance(third, ModelCallResult)
    assert first.provider_run.run_id == "run.provider.0001"
    assert second.provider_run.run_id == "run.provider.0002"
    assert third.provider_run.run_id == "run.provider.0003"
    assert first.request_id == "request.provider.0001"
    assert second.request_id == "request.provider.0002"
    assert third.request_id == "request.provider.0003"
    assert (
        first.provider_run.request_fingerprint
        != second.provider_run.request_fingerprint
    )
    assert (
        first.provider_run.request_fingerprint
        == third.provider_run.request_fingerprint
    )

    first_log = _read_provider_log(tmp_path, first)
    second_log = _read_provider_log(tmp_path, second)
    assert first_log["run_id"] == "run.provider.0001"
    assert second_log["run_id"] == "run.provider.0002"
    assert first_log["request_fingerprint"] == first.provider_run.request_fingerprint
    assert second_log["request_fingerprint"] == second.provider_run.request_fingerprint


def test_provider_gateway_supports_injected_id_generator_and_refuses_overwrite(
    tmp_path: Path,
) -> None:
    context = RepoContext(tmp_path)
    gateway = ProviderGateway(
        context,
        id_generator=lambda: ("run.provider.custom", "request.provider.custom"),
    )

    first = gateway.call(_request(prompt="Deterministic custom ID call."))
    assert isinstance(first, ModelCallResult)
    assert first.provider_run.run_id == "run.provider.custom"
    assert first.request_id == "request.provider.custom"

    with pytest.raises(ValueError, match="already exists and cannot be overwritten"):
        gateway.call(_request(prompt="Duplicate ID call must not overwrite log."))
