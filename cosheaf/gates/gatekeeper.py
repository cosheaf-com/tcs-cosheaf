"""Validation and gatekeeper report orchestration."""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from cosheaf.core.artifact import BaseArtifact
from cosheaf.core.paths import repo_relative_posix
from cosheaf.gates.dependency_gate import (
    validate_dependencies,
    validate_id_uniqueness,
)
from cosheaf.gates.formal_link_gate import (
    FormalLinkResult,
    validate_formal_link_policy,
)
from cosheaf.gates.reproducibility_gate import (
    ReproducibilityMetadataResult,
    validate_reproducibility_metadata,
)
from cosheaf.gates.schema_gate import (
    ValidationFailure,
    load_schema_valid_record,
    load_schema_valid_records,
    sort_failures,
)
from cosheaf.gates.source_metadata_gate import (
    SourceMetadataResult,
    validate_source_metadata_policy,
)
from cosheaf.gates.status_gate import (
    validate_evidence_paths,
    validate_status_paths,
)
from cosheaf.storage.loader import LoadedRecord
from cosheaf.storage.repo import RepoContext
from cosheaf.verification.registry import default_verifier_registry
from cosheaf.verification.result import VerificationResult, VerificationStatus

GateStatus = Literal["pass", "fail", "skipped", "not_applicable"]
GateVerdict = Literal["pass", "fail"]
REQUIRED_PR_CHECKLIST_SECTIONS = (
    "summary",
    "changed files",
    "tests run",
    "risks",
    "interface changes",
    "documentation changes",
    "artifact/schema changes",
    "gatekeeper result",
)
_MARKDOWN_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+(.+?)\s*$")


@dataclass(frozen=True)
class ValidationReport:
    """Validation result for a repository or a single artifact file."""

    records: tuple[LoadedRecord, ...]
    failures: tuple[ValidationFailure, ...]

    @property
    def ok(self) -> bool:
        return not self.failures

    @property
    def checked_count(self) -> int:
        return len(self.records)


def validate_repository(context: RepoContext) -> ValidationReport:
    """Validate a repository checkout using the implemented MVP gates."""
    schema_result = load_schema_valid_records(context)
    records = schema_result.records
    failures: list[ValidationFailure] = list(schema_result.failures)
    failures.extend(validate_id_uniqueness(records))
    failures.extend(validate_status_paths(records))
    failures.extend(
        validate_dependencies(records, policy=context.workspace_config.policy)
    )
    failures.extend(validate_evidence_paths(context, records))

    return ValidationReport(records=records, failures=sort_failures(failures))


def validate_artifact_file(context: RepoContext, path: Path) -> ValidationReport:
    """Validate one YAML file with file-local checks."""
    schema_result = load_schema_valid_record(context, path)
    records = schema_result.records
    failures: list[ValidationFailure] = list(schema_result.failures)
    failures.extend(validate_status_paths(records))
    failures.extend(validate_evidence_paths(context, records))

    return ValidationReport(records=records, failures=sort_failures(failures))


@dataclass(frozen=True)
class GateIssue:
    """A blocking or nonblocking issue recorded in a gatekeeper report."""

    gate_id: str
    gate_name: str
    source_path: str
    artifact_id: str
    message: str
    severity: Literal["blocking", "nonblocking"]

    def to_dict(self) -> dict[str, str]:
        return {
            "gate_id": self.gate_id,
            "gate_name": self.gate_name,
            "source_path": self.source_path,
            "artifact_id": self.artifact_id,
            "message": self.message,
            "severity": self.severity,
        }


@dataclass(frozen=True)
class GateResult:
    """One gate result in a gatekeeper run."""

    gate_id: str
    name: str
    status: GateStatus
    summary: str
    blocking_issues: tuple[GateIssue, ...] = ()
    nonblocking_issues: tuple[GateIssue, ...] = ()
    details: tuple[dict[str, object], ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.gate_id,
            "name": self.name,
            "status": self.status,
            "summary": self.summary,
            "blocking_issues": [
                issue.to_dict() for issue in self.blocking_issues
            ],
            "nonblocking_issues": [
                issue.to_dict() for issue in self.nonblocking_issues
            ],
            "details": list(self.details),
        }


@dataclass(frozen=True)
class GatekeeperReport:
    """Machine-readable gatekeeper report."""

    verdict: GateVerdict
    blocking_issues: tuple[GateIssue, ...]
    nonblocking_issues: tuple[GateIssue, ...]
    summary: dict[str, int | str]
    started_at: str
    ended_at: str
    gates: tuple[GateResult, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "verdict": self.verdict,
            "blocking_issues": [
                issue.to_dict() for issue in self.blocking_issues
            ],
            "nonblocking_issues": [
                issue.to_dict() for issue in self.nonblocking_issues
            ],
            "summary": self.summary,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "gates": [gate.to_dict() for gate in self.gates],
        }


@dataclass(frozen=True)
class GatekeeperRunResult:
    """Report and written report paths for a gatekeeper run."""

    report: GatekeeperReport
    json_path: Path
    markdown_path: Path
    review_json_path: Path | None = None
    review_markdown_path: Path | None = None


def run_gatekeeper(
    context: RepoContext,
    *,
    persist_review: bool = False,
    pr_checklist_path: Path | None = None,
    timestamp: str | None = None,
) -> GatekeeperRunResult:
    """Run all current gates and write machine/human-readable reports."""
    started = datetime.now(UTC)
    started_at = _format_report_time(started)
    report_timestamp = timestamp or started.strftime("%Y%m%dT%H%M%S%fZ")

    schema_result = load_schema_valid_records(context)
    records = schema_result.records
    verification_results = _run_verifiers(context, records)
    gates = (
        _gate_from_failures(
            gate_id="G1",
            name="schema gate",
            failures=schema_result.failures,
            pass_summary=(
                f"Schema/model parsing passed for {len(records)} loaded record(s)."
            ),
        ),
        _gate_from_failures(
            gate_id="G2",
            name="ID uniqueness gate",
            failures=validate_id_uniqueness(records),
            pass_summary="All loaded record IDs are globally unique.",
        ),
        _gate_from_failures(
            gate_id="G3",
            name="status/path gate",
            failures=validate_status_paths(records),
            pass_summary="All loaded artifact statuses match their paths.",
        ),
        _gate_from_failures(
            gate_id="G4",
            name="dependency gate",
            failures=validate_dependencies(
                records,
                policy=context.workspace_config.policy,
            ),
            pass_summary="All loaded artifact dependencies are valid.",
        ),
        _gate_from_failures(
            gate_id="G5",
            name="evidence path gate",
            failures=validate_evidence_paths(context, records),
            pass_summary="All local evidence paths exist.",
        ),
        _verifier_gate(verification_results),
        _reproducibility_metadata_gate(
            validate_reproducibility_metadata(records, verification_results)
        ),
        _pr_checklist_gate(context, pr_checklist_path),
        _source_metadata_gate(validate_source_metadata_policy(context, records)),
        _formal_link_gate(
            validate_formal_link_policy(
                records,
                context=context,
                verification_results=verification_results,
            )
        ),
    )

    blocking_issues = tuple(
        issue for gate in gates for issue in gate.blocking_issues
    )
    nonblocking_issues = tuple(
        issue for gate in gates for issue in gate.nonblocking_issues
    )
    ended_at = _format_report_time(datetime.now(UTC))
    report = GatekeeperReport(
        verdict="fail" if blocking_issues else "pass",
        blocking_issues=blocking_issues,
        nonblocking_issues=nonblocking_issues,
        summary={
            "records_checked": len(records),
            "gates_total": len(gates),
            "gates_passed": sum(1 for gate in gates if gate.status == "pass"),
            "gates_failed": sum(1 for gate in gates if gate.status == "fail"),
            "gates_skipped": sum(1 for gate in gates if gate.status == "skipped"),
            "gates_not_applicable": sum(
                1 for gate in gates if gate.status == "not_applicable"
            ),
            "blocking_issue_count": len(blocking_issues),
            "nonblocking_issue_count": len(nonblocking_issues),
        },
        started_at=started_at,
        ended_at=ended_at,
        gates=gates,
    )

    report_dir = context.resolve(".cosheaf/reports")
    json_path, markdown_path = _write_report_files(
        report_dir=report_dir,
        timestamp=report_timestamp,
        report=report,
    )

    review_json_path: Path | None = None
    review_markdown_path: Path | None = None
    if persist_review:
        review_dir = context.resolve("reviews/gatekeeper")
        review_json_path, review_markdown_path = _copy_report_files(
            json_path=json_path,
            markdown_path=markdown_path,
            review_dir=review_dir,
        )

    return GatekeeperRunResult(
        report=report,
        json_path=json_path,
        markdown_path=markdown_path,
        review_json_path=review_json_path,
        review_markdown_path=review_markdown_path,
    )


def _gate_from_failures(
    *,
    gate_id: str,
    name: str,
    failures: tuple[ValidationFailure, ...] | list[ValidationFailure],
    pass_summary: str,
) -> GateResult:
    issues = tuple(_issue_from_failure(gate_id, name, failure) for failure in failures)
    if issues:
        return GateResult(
            gate_id=gate_id,
            name=name,
            status="fail",
            summary=f"{len(issues)} blocking issue(s).",
            blocking_issues=issues,
        )
    return GateResult(
        gate_id=gate_id,
        name=name,
        status="pass",
        summary=pass_summary,
    )


def _pr_checklist_gate(
    context: RepoContext,
    pr_checklist_path: Path | None,
) -> GateResult:
    gate_name = "PR checklist gate"
    if pr_checklist_path is None:
        issue = GateIssue(
            gate_id="G8",
            gate_name=gate_name,
            source_path="",
            artifact_id="",
            message="No PR checklist source was provided.",
            severity="nonblocking",
        )
        return GateResult(
            gate_id="G8",
            name=gate_name,
            status="skipped",
            summary="Skipped: No PR checklist source was provided.",
            nonblocking_issues=(issue,),
            details=(
                {
                    "source_path": "",
                    "required_sections": list(REQUIRED_PR_CHECKLIST_SECTIONS),
                    "missing_sections": list(REQUIRED_PR_CHECKLIST_SECTIONS),
                },
            ),
        )

    source_path = _resolve_pr_checklist_path(context, pr_checklist_path)
    source_label = _pr_checklist_source_label(context, source_path)
    if not source_path.is_file():
        issue = GateIssue(
            gate_id="G8",
            gate_name=gate_name,
            source_path=source_label,
            artifact_id="",
            message=f"PR checklist source does not exist: {source_label}",
            severity="blocking",
        )
        return GateResult(
            gate_id="G8",
            name=gate_name,
            status="fail",
            summary="PR checklist source is unavailable.",
            blocking_issues=(issue,),
            details=(
                {
                    "source_path": source_label,
                    "required_sections": list(REQUIRED_PR_CHECKLIST_SECTIONS),
                    "missing_sections": list(REQUIRED_PR_CHECKLIST_SECTIONS),
                },
            ),
        )

    headings = _markdown_section_headings(source_path.read_text(encoding="utf-8"))
    missing_sections = tuple(
        section
        for section in REQUIRED_PR_CHECKLIST_SECTIONS
        if section not in headings
    )
    details: tuple[dict[str, object], ...] = (
        {
            "source_path": source_label,
            "required_sections": list(REQUIRED_PR_CHECKLIST_SECTIONS),
            "missing_sections": list(missing_sections),
        },
    )
    if missing_sections:
        issues = tuple(
            GateIssue(
                gate_id="G8",
                gate_name=gate_name,
                source_path=source_label,
                artifact_id="",
                message=f"missing PR checklist section: {section}",
                severity="blocking",
            )
            for section in missing_sections
        )
        return GateResult(
            gate_id="G8",
            name=gate_name,
            status="fail",
            summary=f"{len(issues)} required PR checklist section(s) missing.",
            blocking_issues=issues,
            details=details,
        )

    return GateResult(
        gate_id="G8",
        name=gate_name,
        status="pass",
        summary=(
            "PR checklist includes all "
            f"{len(REQUIRED_PR_CHECKLIST_SECTIONS)} required section(s)."
        ),
        details=details,
    )


def _source_metadata_gate(result: SourceMetadataResult) -> GateResult:
    gate_name = "source metadata gate"
    details = tuple(check.to_dict() for check in result.checks)
    if result.failures:
        issues = tuple(
            _issue_from_failure("G9", gate_name, failure)
            for failure in result.failures
        )
        return GateResult(
            gate_id="G9",
            name=gate_name,
            status="fail",
            summary=f"{len(issues)} source metadata issue(s).",
            blocking_issues=issues,
            details=details,
        )
    if result.policy_reason:
        return GateResult(
            gate_id="G9",
            name=gate_name,
            status="not_applicable",
            summary=result.policy_reason,
            details=details,
        )
    if result.applicable_count == 0:
        return GateResult(
            gate_id="G9",
            name=gate_name,
            status="not_applicable",
            summary="No accepted public artifacts require source metadata.",
            details=details,
        )
    return GateResult(
        gate_id="G9",
        name=gate_name,
        status="pass",
        summary=(
            "Source metadata passed for "
            f"{result.applicable_count} accepted public artifact(s)."
        ),
        details=details,
    )


def _formal_link_gate(result: FormalLinkResult) -> GateResult:
    gate_name = "formal link gate"
    details = tuple(check.to_dict() for check in result.checks)
    blocking_issues = tuple(
        _issue_from_failure("G10", gate_name, failure)
        for failure in result.failures
    )
    nonblocking_issues = tuple(
        _nonblocking_issue_from_failure("G10", gate_name, warning)
        for warning in result.warnings
    )
    if blocking_issues:
        return GateResult(
            gate_id="G10",
            name=gate_name,
            status="fail",
            summary=f"{len(blocking_issues)} formal-link policy issue(s).",
            blocking_issues=blocking_issues,
            nonblocking_issues=nonblocking_issues,
            details=details,
        )
    if result.applicable_count == 0:
        return GateResult(
            gate_id="G10",
            name=gate_name,
            status="not_applicable",
            summary="No formal-link policy metadata requires G10 checks.",
            details=details,
        )
    warning_suffix = (
        f" with {len(nonblocking_issues)} warning(s)"
        if nonblocking_issues
        else ""
    )
    return GateResult(
        gate_id="G10",
        name=gate_name,
        status="pass",
        summary=(
            "Formal-link metadata passed for "
            f"{result.applicable_count} artifact(s){warning_suffix}."
        ),
        nonblocking_issues=nonblocking_issues,
        details=details,
    )


def _resolve_pr_checklist_path(context: RepoContext, path: Path) -> Path:
    if path.is_absolute():
        return path.resolve()
    return context.resolve(path)


def _pr_checklist_source_label(context: RepoContext, path: Path) -> str:
    try:
        return repo_relative_posix(context.repo_root, path)
    except ValueError:
        return path.as_posix()


def _markdown_section_headings(markdown: str) -> frozenset[str]:
    headings: set[str] = set()
    for line in markdown.splitlines():
        match = _MARKDOWN_HEADING_RE.match(line)
        if match is None:
            continue
        headings.add(_normalize_markdown_heading(match.group(1)))
    return frozenset(headings)


def _normalize_markdown_heading(value: str) -> str:
    heading = value.strip().strip("#").strip()
    heading = re.sub(r"\s+", " ", heading).lower()
    return heading


def _run_verifiers(
    context: RepoContext,
    records: tuple[LoadedRecord, ...],
) -> tuple[VerificationResult, ...]:
    registry = default_verifier_registry()
    results: list[VerificationResult] = []
    for adapter in registry.adapters:
        for loaded in records:
            if not isinstance(loaded.record, BaseArtifact):
                continue
            if adapter.can_verify(loaded.record, context):
                results.append(adapter.verify(loaded.record, context))
    return _sort_verification_results(results)


def _verifier_gate(results: tuple[VerificationResult, ...]) -> GateResult:
    if not results:
        issue = GateIssue(
            gate_id="G6",
            gate_name="verifier gate",
            source_path="",
            artifact_id="",
            message="No verifier adapters were applicable.",
            severity="nonblocking",
        )
        return GateResult(
            gate_id="G6",
            name="verifier gate",
            status="skipped",
            summary="Skipped: no verifier adapters were applicable.",
            nonblocking_issues=(issue,),
        )

    blocking_issues = tuple(
        _issue_from_verification_result(result)
        for result in results
        if result.status in {VerificationStatus.FAIL, VerificationStatus.ERROR}
    )
    nonblocking_issues = tuple(
        _issue_from_verification_result(result)
        for result in results
        if result.status is VerificationStatus.SKIPPED
    )
    details = tuple(result.to_dict() for result in results)
    if blocking_issues:
        return GateResult(
            gate_id="G6",
            name="verifier gate",
            status="fail",
            summary=f"{len(blocking_issues)} blocking verifier issue(s).",
            blocking_issues=blocking_issues,
            nonblocking_issues=nonblocking_issues,
            details=details,
        )
    if nonblocking_issues:
        return GateResult(
            gate_id="G6",
            name="verifier gate",
            status="skipped",
            summary=f"{len(nonblocking_issues)} verifier result(s) skipped.",
            nonblocking_issues=nonblocking_issues,
            details=details,
        )
    return GateResult(
        gate_id="G6",
        name="verifier gate",
        status="pass",
        summary=f"Verifier gate passed for {len(results)} result(s).",
        details=details,
    )


def _reproducibility_metadata_gate(
    result: ReproducibilityMetadataResult,
) -> GateResult:
    details = tuple(check.to_dict() for check in result.checks)
    if result.failures:
        issues = tuple(
            _issue_from_failure("G7", "reproducibility metadata gate", failure)
            for failure in result.failures
        )
        return GateResult(
            gate_id="G7",
            name="reproducibility metadata gate",
            status="fail",
            summary=f"{len(issues)} blocking reproducibility metadata issue(s).",
            blocking_issues=issues,
            details=details,
        )
    if result.applicable_count == 0:
        return GateResult(
            gate_id="G7",
            name="reproducibility metadata gate",
            status="not_applicable",
            summary="No executable evidence requires reproducibility metadata.",
            details=details,
        )
    return GateResult(
        gate_id="G7",
        name="reproducibility metadata gate",
        status="pass",
        summary=(
            "Reproducibility metadata passed for "
            f"{result.applicable_count} executable evidence item(s)."
        ),
        details=details,
    )


def _sort_verification_results(
    results: list[VerificationResult],
) -> tuple[VerificationResult, ...]:
    return tuple(
        sorted(
            results,
            key=lambda result: (
                result.artifact_id,
                result.verifier,
                result.status.value,
                result.message,
            ),
        )
    )


def _issue_from_verification_result(result: VerificationResult) -> GateIssue:
    severity: Literal["blocking", "nonblocking"]
    severity = (
        "nonblocking"
        if result.status is VerificationStatus.SKIPPED
        else "blocking"
    )
    return GateIssue(
        gate_id="G6",
        gate_name="verifier gate",
        source_path="",
        artifact_id=result.artifact_id,
        message=f"{result.verifier} {result.status.value}: {result.message}",
        severity=severity,
    )


def _issue_from_failure(
    gate_id: str,
    gate_name: str,
    failure: ValidationFailure,
) -> GateIssue:
    return GateIssue(
        gate_id=gate_id,
        gate_name=gate_name,
        source_path=failure.source_path,
        artifact_id=failure.artifact_id,
        message=failure.message,
        severity="blocking",
    )


def _nonblocking_issue_from_failure(
    gate_id: str,
    gate_name: str,
    failure: ValidationFailure,
) -> GateIssue:
    return GateIssue(
        gate_id=gate_id,
        gate_name=gate_name,
        source_path=failure.source_path,
        artifact_id=failure.artifact_id,
        message=failure.message,
        severity="nonblocking",
    )


def _write_report_files(
    *,
    report_dir: Path,
    timestamp: str,
    report: GatekeeperReport,
) -> tuple[Path, Path]:
    report_dir.mkdir(parents=True, exist_ok=True)
    json_path = report_dir / f"{timestamp}-gate-report.json"
    markdown_path = report_dir / f"{timestamp}-gate-report.md"
    json_path.write_text(
        json.dumps(report.to_dict(), indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(_render_markdown_report(report), encoding="utf-8")
    return json_path, markdown_path


def _copy_report_files(
    *,
    json_path: Path,
    markdown_path: Path,
    review_dir: Path,
) -> tuple[Path, Path]:
    review_dir.mkdir(parents=True, exist_ok=True)
    review_json_path = review_dir / json_path.name
    review_markdown_path = review_dir / markdown_path.name
    shutil.copyfile(json_path, review_json_path)
    shutil.copyfile(markdown_path, review_markdown_path)
    return review_json_path, review_markdown_path


def _render_markdown_report(report: GatekeeperReport) -> str:
    lines = [
        "# Gatekeeper Report",
        "",
        f"- Verdict: {report.verdict}",
        f"- Started at: {report.started_at}",
        f"- Ended at: {report.ended_at}",
        "",
        "## Summary",
        "",
    ]
    for key, value in report.summary.items():
        lines.append(f"- {key}: {value}")

    lines.extend(["", "## Gates", ""])
    for gate in report.gates:
        lines.append(f"- {gate.gate_id} {gate.name}: {gate.status}")
        lines.append(f"  - {gate.summary}")

    lines.extend(["", "## Blocking Issues", ""])
    if report.blocking_issues:
        for issue in report.blocking_issues:
            lines.append(_format_markdown_issue(issue))
    else:
        lines.append("- None")

    lines.extend(["", "## Nonblocking Issues", ""])
    if report.nonblocking_issues:
        for issue in report.nonblocking_issues:
            lines.append(_format_markdown_issue(issue))
    else:
        lines.append("- None")

    return "\n".join(lines) + "\n"


def _format_markdown_issue(issue: GateIssue) -> str:
    location = issue.source_path or "-"
    artifact_id = issue.artifact_id or "-"
    return (
        f"- {issue.gate_id} {issue.gate_name} | {location} | "
        f"{artifact_id} | {issue.message}"
    )


def _format_report_time(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")

