"""Deterministic leak scanner for operator-session runtime records."""

from __future__ import annotations

import json
import re
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

from cosheaf.core.ids import validate_artifact_id
from cosheaf.operator_session.models import (
    OPERATOR_SESSION_AUTHORITY_NOTICE,
    OperatorPolicyMode,
    OperatorSessionError,
)
from cosheaf.operator_session.storage import (
    operator_session_events_path,
    operator_session_path,
)
from cosheaf.security.provider_logs import scan_provider_log_text
from cosheaf.storage.loader import LoadError, load_artifacts
from cosheaf.storage.repo import RepoContext

OPERATOR_SESSION_SCAN_KIND = "operator_session_scan"

FINDING_MESSAGES = {
    "api_key": "operator session contains an API-key-shaped value",
    "bearer_token": "operator session contains an unredacted bearer token",
    "environment_dump": "operator session contains an environment-like dump",
    "secret_env_value": "operator session contains a secret-looking key with a value",
    "hidden_reasoning": "operator session contains hidden-reasoning marker text",
    "absolute_private_path": (
        "operator session contains an absolute user or private filesystem path"
    ),
    "private_artifact_id": (
        "public-only operator session references a private artifact ID"
    ),
    "private_path_reference": (
        "public-only operator session references a private path or scope"
    ),
    "accepted_write_attempt": (
        "operator session references an accepted KB write target"
    ),
    "provider_payload": (
        "operator session stores raw provider request or response payload data"
    ),
    "authority_claim": (
        "operator session claims human-review, verifier, accepted, or "
        "promotion authority"
    ),
    "session_json_invalid": "operator session JSON could not be parsed",
    "events_json_invalid": "operator session event JSON could not be parsed",
    "artifact_load_failed": (
        "private artifact ID scan could not load repository artifact records"
    ),
}

PRIVATE_PATH_PATTERN = re.compile(
    r"(?i)(?:^|[/\\])kb[/\\]private[/\\]|(?:^|[/\\])private[/\\]"
)
ACCEPTED_PATH_PATTERN = re.compile(r"(?i)(?:^|[/\\])kb[/\\][^\"'\s,}]*accepted[/\\]")
PROVIDER_PAYLOAD_KEYS = frozenset(
    {
        "provider_payload",
        "provider_request",
        "provider_response",
        "raw_provider_payload",
        "raw_provider_request",
        "raw_provider_response",
        "raw_request",
        "raw_response",
    }
)
AUTHORITY_BOOLEAN_KEYS = frozenset(
    {
        "accepted",
        "accepted_write_performed",
        "accepted_status",
        "gate_pass",
        "human_review",
        "human_review_created",
        "human_reviewed",
        "promote",
        "promotion_authority",
        "promotion_performed",
        "review_state",
        "verifier_pass",
        "verifier_result_mutated",
    }
)
AUTHORITY_TEXT_PATTERN = re.compile(
    r"(?i)(human_reviewed|mark\s+human\s+review|mark\s+.*reviewed|"
    r"promote\s+this|promotion_authority|"
    r"verifier_pass\s*[:=]\s*true|accepted_status\s*[:=]\s*accepted)"
)
ENVIRONMENT_DUMP_KEYS = frozenset({"env", "environ", "environment", "env_dump"})


@dataclass(frozen=True)
class OperatorSessionScanFinding:
    """One deterministic operator-session scanner finding."""

    code: str
    severity: Literal["warning", "blocker"]
    message: str
    source_path: str
    line: int | None = None
    field_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "code": self.code,
            "severity": self.severity,
            "message": self.message,
            "source_path": self.source_path,
        }
        if self.line is not None:
            payload["line"] = self.line
        if self.field_path is not None:
            payload["field_path"] = self.field_path
        return payload


@dataclass(frozen=True)
class OperatorSessionScanResult:
    """One operator-session scan report."""

    session_id: str
    policy_mode: str
    findings: tuple[OperatorSessionScanFinding, ...]
    report_path: Path
    accepted_write_performed: Literal[False] = False
    authority_notice: str = OPERATOR_SESSION_AUTHORITY_NOTICE

    @property
    def finding_count(self) -> int:
        return len(self.findings)

    @property
    def blocking_finding_count(self) -> int:
        return sum(1 for finding in self.findings if finding.severity == "blocker")

    @property
    def handoff_blocked(self) -> bool:
        return self.blocking_finding_count > 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "kind": OPERATOR_SESSION_SCAN_KIND,
            "session_id": self.session_id,
            "policy_mode": self.policy_mode,
            "finding_count": self.finding_count,
            "blocking_finding_count": self.blocking_finding_count,
            "handoff_blocked": self.handoff_blocked,
            "report_path": self.report_path.as_posix(),
            "accepted_write_performed": self.accepted_write_performed,
            "authority_notice": self.authority_notice,
            "findings": [finding.to_dict() for finding in self.findings],
        }


def scan_operator_session(
    context: RepoContext,
    session_id: str,
    *,
    write_report: bool = True,
) -> OperatorSessionScanResult:
    """Scan one operator-session runtime record and optionally persist a report."""
    resolved_session_id = validate_artifact_id(session_id.strip())
    relative_session_path = operator_session_path(resolved_session_id)
    relative_events_path = operator_session_events_path(resolved_session_id)
    session_target = context.resolve(relative_session_path)
    events_target = context.resolve(relative_events_path)
    if not session_target.is_file():
        raise OperatorSessionError(
            f"operator session not found: {resolved_session_id}",
            code="operator_session_not_found",
            remediation="Start the session first or pass an existing session_id.",
            details={"path": relative_session_path.as_posix()},
        )

    scanner = _OperatorSessionScanner(
        context=context,
        session_id=resolved_session_id,
    )
    session_text = session_target.read_text(encoding="utf-8-sig")
    session_json = scanner.scan_json_file(
        session_text,
        source_path=relative_session_path.as_posix(),
        invalid_code="session_json_invalid",
    )
    policy_mode = _policy_mode(session_json)
    if events_target.exists():
        events_text = events_target.read_text(encoding="utf-8-sig")
        scanner.scan_events_jsonl(
            events_text,
            source_path=relative_events_path.as_posix(),
        )
    scanner.scan_policy_scope(policy_mode)
    result = OperatorSessionScanResult(
        session_id=resolved_session_id,
        policy_mode=policy_mode,
        findings=tuple(scanner.findings),
        report_path=operator_session_scan_path(resolved_session_id),
    )
    if write_report:
        target = context.resolve(result.report_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(result.to_dict(), ensure_ascii=True, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
    return result


def operator_session_scan_path(session_id: str) -> Path:
    """Return runtime scan report path for one session ID."""
    resolved = validate_artifact_id(session_id.strip())
    return Path(".cosheaf") / "operator-sessions" / resolved / "scan.json"


class _OperatorSessionScanner:
    def __init__(self, *, context: RepoContext, session_id: str) -> None:
        self.context = context
        self.session_id = session_id
        self.findings: list[OperatorSessionScanFinding] = []
        self._seen: set[tuple[str, str, int | None, str | None]] = set()
        self._private_artifact_ids = _private_artifact_ids(context)

    def scan_json_file(
        self,
        text: str,
        *,
        source_path: str,
        invalid_code: str,
    ) -> object | None:
        self.scan_text(text, source_path=source_path)
        try:
            parsed = cast(object, json.loads(text))
        except json.JSONDecodeError as exc:
            self._add(
                invalid_code,
                source_path=source_path,
                line=exc.lineno,
                severity="blocker",
            )
            return None
        self.scan_json(parsed, source_path=source_path)
        return parsed

    def scan_events_jsonl(self, text: str, *, source_path: str) -> None:
        self.scan_text(text, source_path=source_path)
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                parsed = cast(object, json.loads(line))
            except json.JSONDecodeError:
                self._add(
                    "events_json_invalid",
                    source_path=source_path,
                    line=line_number,
                    severity="blocker",
                )
                continue
            self.scan_json(parsed, source_path=source_path)

    def scan_text(self, text: str, *, source_path: str) -> None:
        for finding in scan_provider_log_text(text, path=source_path):
            self._add(
                finding.kind,
                source_path=source_path,
                line=finding.line,
                field_path=finding.key,
                severity="blocker",
            )
        if ACCEPTED_PATH_PATTERN.search(text):
            self._add(
                "accepted_write_attempt",
                source_path=source_path,
                severity="blocker",
            )
    def scan_json(self, value: object, *, source_path: str) -> None:
        for field_path, scalar in _walk_json(value):
            key = field_path[-1] if field_path else ""
            normalized_key = _normalize_key(key)
            path_text = ".".join(field_path)
            if normalized_key in PROVIDER_PAYLOAD_KEYS:
                self._add(
                    "provider_payload",
                    source_path=source_path,
                    field_path=path_text,
                    severity="blocker",
                )
            if normalized_key in ENVIRONMENT_DUMP_KEYS:
                self._add(
                    "environment_dump",
                    source_path=source_path,
                    field_path=path_text,
                    severity="blocker",
                )
            if normalized_key in AUTHORITY_BOOLEAN_KEYS and _truthy_authority(scalar):
                self._add(
                    "authority_claim",
                    source_path=source_path,
                    field_path=path_text,
                    severity="blocker",
                )
            if isinstance(scalar, str) and AUTHORITY_TEXT_PATTERN.search(scalar):
                self._add(
                    "authority_claim",
                    source_path=source_path,
                    field_path=path_text,
                    severity="blocker",
                )
            if _is_accepted_path_scalar(scalar):
                self._add(
                    "accepted_write_attempt",
                    source_path=source_path,
                    field_path=path_text,
                    severity="blocker",
                )

    def scan_policy_scope(self, policy_mode: str) -> None:
        if policy_mode != OperatorPolicyMode.PUBLIC_ONLY.value:
            return
        for source_path in (
            operator_session_path(self.session_id).as_posix(),
            operator_session_events_path(self.session_id).as_posix(),
        ):
            target = self.context.resolve(Path(source_path))
            if not target.exists():
                continue
            text = target.read_text(encoding="utf-8-sig")
            for private_id in sorted(self._private_artifact_ids):
                if private_id in text:
                    self._add(
                        "private_artifact_id",
                        source_path=source_path,
                        severity="blocker",
                        field_path=private_id,
                    )
            if PRIVATE_PATH_PATTERN.search(text):
                self._add(
                    "private_path_reference",
                    source_path=source_path,
                    severity="blocker",
                )

    def _add(
        self,
        code: str,
        *,
        source_path: str,
        severity: Literal["warning", "blocker"],
        line: int | None = None,
        field_path: str | None = None,
    ) -> None:
        marker = (code, source_path, line, field_path)
        if marker in self._seen:
            return
        self._seen.add(marker)
        self.findings.append(
            OperatorSessionScanFinding(
                code=code,
                severity=severity,
                message=FINDING_MESSAGES[code],
                source_path=source_path,
                line=line,
                field_path=field_path,
            )
        )


def _policy_mode(value: object | None) -> str:
    if isinstance(value, Mapping):
        raw = value.get("policy_mode")
        if isinstance(raw, str):
            try:
                return OperatorPolicyMode(raw).value
            except ValueError:
                return raw
    return "unknown"


def _private_artifact_ids(context: RepoContext) -> frozenset[str]:
    try:
        records = load_artifacts(context)
    except LoadError:
        return frozenset()
    private_ids = {
        record.id
        for record in records
        if record.is_private_scope
    }
    return frozenset(private_ids)


def _walk_json(
    value: object,
    path: tuple[str, ...] = (),
) -> Iterator[tuple[tuple[str, ...], object]]:
    yield path, value
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key)
            yield from _walk_json(child, (*path, key))
        return
    if isinstance(value, list):
        for index, child in enumerate(value):
            yield from _walk_json(child, (*path, str(index)))
        return


def _normalize_key(key: str) -> str:
    return key.strip().lower().replace("-", "_").replace(".", "_")


def _truthy_authority(value: object) -> bool:
    if value is True:
        return True
    if not isinstance(value, str):
        return False
    normalized = value.strip().lower()
    return normalized in {
        "true",
        "accepted",
        "approved",
        "human_reviewed",
        "promote",
        "promotion_performed",
        "verifier_pass",
    }


def _is_accepted_path_scalar(value: object) -> bool:
    return isinstance(value, str) and bool(ACCEPTED_PATH_PATTERN.search(value))


__all__ = [
    "OPERATOR_SESSION_SCAN_KIND",
    "OperatorSessionScanFinding",
    "OperatorSessionScanResult",
    "operator_session_scan_path",
    "scan_operator_session",
]
