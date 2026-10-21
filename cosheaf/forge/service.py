"""Dry-run forge planning service."""

from __future__ import annotations

import json
import re
import subprocess
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath, PureWindowsPath

from pydantic import ValidationError

from cosheaf.core.paths import normalize_repo_path
from cosheaf.forge.models import (
    ForgeActionResult,
    ForgePreviewResult,
    GitHubIssuePlan,
    GitHubPrPlan,
    GitHubPrStatusResult,
    LocalGitPlan,
)
from cosheaf.services import GateService, ValidationService
from cosheaf.storage.loader import IssueRecord, LoadError, load_yaml_file
from cosheaf.storage.repo import RepoContext
from cosheaf.storage.writer import write_yaml_deterministic


class ForgePreviewError(ValueError):
    """Raised when a dry-run forge preview cannot be constructed."""


class ForgeActionError(ValueError):
    """Raised when an explicit forge action cannot be completed."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class ForgeService:
    """Build dry-run plans for local git and GitHub workflow surfaces."""

    def __init__(self, context: RepoContext) -> None:
        self.context = context

    def status(self) -> ForgePreviewResult:
        """Return forge status without reading tokens or mutating git state."""
        return ForgePreviewResult(
            kind="status",
            local_git_plan=LocalGitPlan(repo_root=self.context.repo_root.as_posix()),
        )

    def issue_preview(self, source_path: str | Path) -> ForgePreviewResult:
        """Preview GitHub issue creation from a repository-local issue file."""
        loaded_path, issue = _load_issue_file(self.context, source_path)
        plan = GitHubIssuePlan(
            source_path=loaded_path.as_posix(),
            issue_id=issue.id,
            title=issue.title,
            body=issue.summary,
            labels=issue.labels,
        )
        return ForgePreviewResult(
            kind="github_issue",
            github_issue_plan=plan,
        )

    def pr_preview(self, *, base: str, head: str) -> ForgePreviewResult:
        """Preview GitHub pull request creation without touching git or GitHub."""
        normalized_base = _non_empty(base, "base")
        normalized_head = _non_empty(head, "head")
        return ForgePreviewResult(
            kind="github_pr",
            local_git_plan=LocalGitPlan(
                repo_root=self.context.repo_root.as_posix(),
                base=normalized_base,
                head=normalized_head,
            ),
            github_pr_plan=GitHubPrPlan(
                base=normalized_base,
                head=normalized_head,
                title=f"Merge {normalized_head} into {normalized_base}",
                body=(
                    "Dry-run PR preview only. No git push, GitHub PR creation, "
                    "network call, or repository mutation was performed."
                ),
            ),
        )

    def github_issue_create(
        self,
        source_path: str | Path,
        *,
        confirm: bool,
    ) -> ForgeActionResult:
        """Create a GitHub issue from local issue YAML through gh."""
        if not confirm:
            raise ForgeActionError(
                "forge_confirm_required",
                "forge issue create requires --confirm",
            )
        loaded_path, issue = _load_issue_file_for_action(self.context, source_path)
        args = [
            "issue",
            "create",
            "--title",
            issue.title,
            "--body",
            issue.summary,
        ]
        for label in issue.labels:
            args.extend(["--label", label])
        url = _run_gh(self.context, *args)
        updated_links = _unique_strings([*issue.external_links, url])
        updated = IssueRecord.model_validate(
            {
                **issue.model_dump(mode="json"),
                "updated_at": _now_utc(),
                "external_links": updated_links,
            }
        )
        write_yaml_deterministic(
            self.context.resolve(loaded_path),
            _issue_yaml_data(updated),
        )
        return ForgeActionResult(
            action="github_issue_create",
            action_performed=True,
            network_calls_performed=True,
            github_writes_performed=True,
            github_issue_created=True,
            github_issue_url=url,
            source_path=loaded_path.as_posix(),
            issue_id=issue.id,
        )

    def github_pr_create(
        self,
        *,
        base: str,
        head: str,
        draft: bool,
        confirm: bool,
    ) -> ForgeActionResult:
        """Create a GitHub PR through gh."""
        normalized_base = _git_ref(base, "base")
        normalized_head = _git_ref(head, "head")
        if not confirm:
            raise ForgeActionError(
                "forge_confirm_required",
                "forge pr create requires --confirm",
            )
        title = f"Merge {normalized_head} into {normalized_base}"
        body = f"Forge-created PR for {normalized_head} into {normalized_base}."
        args = [
            "pr",
            "create",
            "--base",
            normalized_base,
            "--head",
            normalized_head,
            "--title",
            title,
            "--body",
            body,
        ]
        if draft:
            args.append("--draft")
        url = _run_gh(self.context, *args)
        return ForgeActionResult(
            action="github_pr_create",
            action_performed=True,
            network_calls_performed=True,
            github_writes_performed=True,
            github_pr_created=True,
            github_pr_url=url,
            base=normalized_base,
            head=normalized_head,
        )

    def push(
        self,
        *,
        branch: str | None = None,
        remote: str = "origin",
        confirm: bool,
    ) -> ForgeActionResult:
        """Push one non-protected branch after explicit confirmation."""
        if not confirm:
            raise ForgeActionError(
                "forge_confirm_required",
                "forge push requires --confirm",
            )
        normalized_remote = _git_ref(remote, "remote")
        normalized_branch = (
            _git_ref(branch, "branch")
            if branch is not None
            else _current_branch(self.context)
        )
        _ensure_unprotected_head(normalized_branch)
        _run_git(self.context, "push", "-u", normalized_remote, normalized_branch)
        return ForgeActionResult(
            action="push",
            action_performed=True,
            network_calls_performed=True,
            git_writes_performed=True,
            push_performed=True,
            branch=normalized_branch,
            head=normalized_branch,
        )

    def github_pr_submit(
        self,
        *,
        base: str,
        head: str,
        draft: bool,
        confirm: bool,
        remote: str = "origin",
    ) -> ForgeActionResult:
        """Validate, gate, push a branch, and create a GitHub draft PR."""
        normalized_base = _git_ref(base, "base")
        normalized_head = _git_ref(head, "head")
        normalized_remote = _git_ref(remote, "remote")
        if not confirm:
            raise ForgeActionError(
                "forge_confirm_required",
                "forge pr submit requires --confirm",
            )
        _ensure_unprotected_head(normalized_head)
        verified_commit_sha = _verify_pr_submit_git_state(
            self.context,
            expected_head=normalized_head,
        )

        validation = ValidationService(self.context).validate_repository()
        if not validation.ok:
            raise ForgeActionError(
                "forge_validation_failed",
                "repository validation failed; fix validation before PR submit",
            )
        gate = GateService(self.context).run()
        if gate.report.verdict != "pass":
            raise ForgeActionError(
                "forge_gate_failed",
                "repository gate failed; fix gate failures before PR submit",
            )

        post_check_sha = _verify_pr_submit_git_state(
            self.context,
            expected_head=normalized_head,
            expected_commit_sha=verified_commit_sha,
        )
        _run_git(self.context, "push", "-u", normalized_remote, normalized_head)
        title = f"Merge {normalized_head} into {normalized_base}"
        body = f"Forge-created PR for {normalized_head} into {normalized_base}."
        args = [
            "pr",
            "create",
            "--base",
            normalized_base,
            "--head",
            normalized_head,
            "--title",
            title,
            "--body",
            body,
        ]
        if draft:
            args.append("--draft")
        url = _run_gh(self.context, *args)
        return ForgeActionResult(
            action="github_pr_submit",
            action_performed=True,
            network_calls_performed=True,
            git_writes_performed=True,
            github_writes_performed=True,
            push_performed=True,
            github_pr_created=True,
            github_pr_url=url,
            base=normalized_base,
            head=normalized_head,
            branch=normalized_head,
            commit_hash=post_check_sha,
            validation_performed=True,
            gate_performed=True,
        )

    def github_pr_status(
        self,
        *,
        number: int | None = None,
        base: str = "",
        head: str = "",
    ) -> GitHubPrStatusResult:
        """Read GitHub PR status without writing GitHub, git, or review records."""
        normalized_base = base.strip()
        normalized_head = head.strip()
        selector = str(number) if number is not None else normalized_head
        if not selector:
            return _degraded_pr_status(
                number=number,
                base=normalized_base,
                head=normalized_head,
                warning="PR number or head branch is required.",
            )
        try:
            payload = _run_gh_json(
                self.context,
                "pr",
                "view",
                selector,
                "--json",
                ",".join(_PR_STATUS_FIELDS),
            )
        except ForgeActionError:
            return _degraded_pr_status(
                number=number,
                base=normalized_base,
                head=normalized_head,
                warning="GitHub status unavailable; gh auth or network is missing.",
            )
        return _github_pr_status_from_payload(
            payload,
            number=number,
            base=normalized_base,
            head=normalized_head,
        )

    def sync(self) -> ForgeActionResult:
        """Return a read-only sync placeholder for future link reconciliation."""
        return ForgeActionResult(action="sync")

    def create_branch(
        self,
        branch: str,
        *,
        confirm: bool,
        allow_dirty: bool = False,
    ) -> ForgeActionResult:
        """Create and switch to a local branch after explicit confirmation."""
        normalized_branch = _git_ref(branch, "branch")
        if not confirm:
            raise ForgeActionError(
                "forge_confirm_required",
                "forge branch create requires --confirm",
            )
        _ensure_unprotected_head(normalized_branch)
        status = _git_status(self.context)
        if status and not allow_dirty:
            raise ForgeActionError(
                "forge_dirty_state",
                "dirty state blocks branch creation; commit, stash, or clean it first",
            )
        _run_git(self.context, "switch", "-c", normalized_branch)
        return ForgeActionResult(
            action="branch_create",
            action_performed=True,
            git_writes_performed=True,
            branch=normalized_branch,
            dirty_state_carried=bool(status),
        )

    def commit(
        self,
        *,
        message: str,
        confirm: bool,
        stage_all: bool = False,
    ) -> ForgeActionResult:
        """Run validation/gate and create one local git commit."""
        normalized_message = _action_non_empty(message, "message")
        if not confirm:
            raise ForgeActionError(
                "forge_confirm_required",
                "forge commit requires --confirm",
            )
        current_branch = _current_branch(self.context)
        _ensure_unprotected_head(current_branch)
        if stage_all:
            _run_git(self.context, "add", "-A")
        status = _git_status(self.context)
        staged = _staged_changes(status)
        ambiguous = _ambiguous_dirty_lines(status)
        if ambiguous:
            detail = "; ".join(ambiguous)
            raise ForgeActionError(
                "forge_dirty_state",
                f"dirty or untracked state blocks commit: {detail}",
            )
        if not staged:
            raise ForgeActionError(
                "forge_no_staged_changes",
                "forge commit requires staged changes",
            )

        validation = ValidationService(self.context).validate_repository()
        if not validation.ok:
            raise ForgeActionError(
                "forge_validation_failed",
                "repository validation failed; fix validation before committing",
            )
        gate = GateService(self.context).run()
        if gate.report.verdict != "pass":
            raise ForgeActionError(
                "forge_gate_failed",
                "repository gate failed; fix gate failures before committing",
            )

        _run_git(self.context, "commit", "-m", normalized_message)
        commit_hash = _run_git(self.context, "rev-parse", "HEAD").stdout.strip()
        return ForgeActionResult(
            action="commit",
            action_performed=True,
            git_writes_performed=True,
            branch=current_branch,
            commit_hash=commit_hash,
            validation_performed=True,
            gate_performed=True,
            staging_performed=stage_all,
        )


def _repo_local_input_path(path: str | Path) -> Path:
    value = str(path)
    normalized = normalize_repo_path(value)
    parts = PurePosixPath(normalized).parts
    if (
        not normalized
        or PureWindowsPath(value).is_absolute()
        or normalized == ".."
        or normalized.startswith("../")
        or ".." in parts
    ):
        raise ForgePreviewError("forge preview paths must be repository-local")
    return Path(normalized)


def _load_issue_file(
    context: RepoContext,
    path: str | Path,
) -> tuple[Path, IssueRecord]:
    relative_path = _repo_local_input_path(path)
    absolute_path = context.resolve(relative_path)
    _ensure_inside_repo(context, absolute_path)
    if not absolute_path.exists():
        raise ForgePreviewError(f"issue source path does not exist: {relative_path}")

    try:
        loaded = load_yaml_file(context, absolute_path)
    except (LoadError, ValidationError, ValueError) as exc:
        raise ForgePreviewError(str(exc)) from exc
    if not isinstance(loaded.record, IssueRecord):
        raise ForgePreviewError(
            f"forge issue preview requires an issue record: {relative_path}"
        )
    return loaded.source_path, loaded.record


def _load_issue_file_for_action(
    context: RepoContext,
    path: str | Path,
) -> tuple[Path, IssueRecord]:
    try:
        return _load_issue_file(context, path)
    except ForgePreviewError as exc:
        raise ForgeActionError("forge_invalid_input", str(exc)) from exc


def _ensure_inside_repo(context: RepoContext, path: Path) -> None:
    try:
        path.resolve().relative_to(context.repo_root.resolve())
    except ValueError as exc:
        raise ForgePreviewError(
            "forge preview paths must stay inside repo root"
        ) from exc


def _non_empty(value: str, field_name: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ForgePreviewError(f"{field_name} must be non-empty")
    return normalized


def _git_ref(value: str, field_name: str) -> str:
    normalized = _action_non_empty(value, field_name)
    if normalized.startswith("-") or any(char.isspace() for char in normalized):
        raise ForgeActionError("forge_invalid_git_ref", f"invalid git ref: {value}")
    return normalized


def _action_non_empty(value: str, field_name: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ForgeActionError("forge_invalid_input", f"{field_name} must be non-empty")
    return normalized


def _run_git(context: RepoContext, *args: str) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        ["git", *args],
        cwd=context.repo_root,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        message = result.stderr.strip() or result.stdout.strip() or "git command failed"
        raise ForgeActionError("forge_git_failed", message)
    return result


def _current_branch(context: RepoContext) -> str:
    branch = _run_git(context, "branch", "--show-current").stdout.strip()
    return _action_non_empty(branch, "branch")


def _resolve_commit_sha(context: RepoContext, ref: str) -> str:
    sha = _run_git(context, "rev-parse", ref).stdout.strip()
    return _action_non_empty(sha, f"{ref} commit SHA")


def _verify_pr_submit_git_state(
    context: RepoContext,
    *,
    expected_head: str,
    expected_commit_sha: str | None = None,
) -> str:
    current_branch = _current_branch(context)
    if current_branch != expected_head:
        raise ForgeActionError(
            "forge_branch_mismatch",
            f"current branch {current_branch!r} does not match target head "
            f"{expected_head!r}; switch to {expected_head!r} before PR submit",
        )
    status = _git_status(context)
    if status:
        raise ForgeActionError(
            "forge_dirty_state",
            "dirty working tree blocks PR submission; commit, stash, or clean it first",
        )
    head_sha = _resolve_commit_sha(context, "HEAD")
    target_sha = _resolve_commit_sha(context, expected_head)
    if head_sha != target_sha:
        raise ForgeActionError(
            "forge_head_sha_mismatch",
            f"HEAD ({head_sha}) does not match target head "
            f"{expected_head!r} ({target_sha})",
        )
    if expected_commit_sha is not None and head_sha != expected_commit_sha:
        raise ForgeActionError(
            "forge_head_sha_mismatch",
            "HEAD commit SHA changed after validation/gate "
            f"({expected_commit_sha} -> {head_sha})",
        )
    return head_sha


def _ensure_unprotected_head(branch: str) -> None:
    if branch.lower() in {"main", "master"}:
        raise ForgeActionError(
            "forge_protected_branch",
            "forge refuses to write directly from main or master",
        )


def _run_gh(context: RepoContext, *args: str) -> str:
    result = subprocess.run(
        ["gh", *args],
        cwd=context.repo_root,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        message = result.stderr.strip() or result.stdout.strip() or "gh command failed"
        raise ForgeActionError("forge_github_failed", message)
    output = result.stdout.strip()
    if not output:
        raise ForgeActionError("forge_github_failed", "gh did not return a URL")
    return output.splitlines()[0].strip()


_PR_STATUS_FIELDS = (
    "number",
    "title",
    "state",
    "isDraft",
    "mergeStateStatus",
    "mergeable",
    "headRefName",
    "baseRefName",
    "url",
    "author",
    "reviewDecision",
    "reviews",
    "comments",
    "closingIssuesReferences",
    "statusCheckRollup",
    "body",
    "updatedAt",
)


def _run_gh_json(context: RepoContext, *args: str) -> dict[str, object]:
    result = subprocess.run(
        ["gh", *args],
        cwd=context.repo_root,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise ForgeActionError(
            "forge_github_status_unavailable",
            "GitHub PR status is unavailable.",
        )
    try:
        payload = json.loads(result.stdout or "{}")
    except json.JSONDecodeError as exc:
        raise ForgeActionError(
            "forge_github_status_unavailable",
            "GitHub PR status did not return JSON.",
        ) from exc
    if not isinstance(payload, dict):
        raise ForgeActionError(
            "forge_github_status_unavailable",
            "GitHub PR status did not return an object.",
        )
    return payload


def _github_pr_status_from_payload(
    payload: dict[str, object],
    *,
    number: int | None,
    base: str,
    head: str,
) -> GitHubPrStatusResult:
    checks = _checks(payload.get("statusCheckRollup"))
    github_reviews = _github_reviews(payload.get("reviews"))
    comments = _review_comments(payload.get("comments"))
    return GitHubPrStatusResult(
        network_calls_performed=True,
        github_auth_available=True,
        source_of_truth="github",
        updated_at=_text(payload.get("updatedAt"), _now_iso()),
        pr={
            "number": _int_value(payload.get("number"), number),
            "title": _text(payload.get("title"), f"PR {number or head}"),
            "state": _text(payload.get("state"), "unknown").lower(),
            "url": _text(payload.get("url"), ""),
            "author": _author_login(payload.get("author")),
            "base": _text(payload.get("baseRefName"), base),
            "head": _text(payload.get("headRefName"), head),
            "is_draft": bool(payload.get("isDraft", False)),
            "merge_state": _text(payload.get("mergeStateStatus"), "unknown").lower(),
            "mergeable": _text(payload.get("mergeable"), "unknown").lower(),
            "review_decision": _text(payload.get("reviewDecision"), "unknown").lower(),
        },
        linked_issue=_linked_issue(payload.get("closingIssuesReferences")),
        checklist=_checklist(payload.get("body")),
        ci=_ci_summary(checks),
        gate=_gate_summary(checks),
        cosheaf_review=_cosheaf_review_placeholder(),
        github_reviews=github_reviews,
        review_comments=comments,
        warnings=[
            "GitHub reviews are collaboration signals only; no Cosheaf human "
            "review record was imported."
        ],
    )


def _degraded_pr_status(
    *,
    number: int | None,
    base: str,
    head: str,
    warning: str,
) -> GitHubPrStatusResult:
    return GitHubPrStatusResult(
        updated_at=_now_iso(),
        pr={
            "number": number,
            "title": f"PR {number}" if number is not None else head,
            "state": "unknown",
            "url": "",
            "author": "",
            "base": base,
            "head": head,
            "is_draft": False,
            "merge_state": "unknown",
            "mergeable": "unknown",
            "review_decision": "unknown",
        },
        linked_issue=None,
        checklist={"completed": 0, "total": 0, "items": []},
        ci={"status": "unknown", "checks": []},
        gate={
            "status": "unknown",
            "checks": [],
            "skipped_is_pass": False,
            "gate_pass_is_review": False,
        },
        cosheaf_review=_cosheaf_review_placeholder(),
        github_reviews=[],
        review_comments=[],
        warnings=[warning],
    )


def _checklist(value: object) -> dict[str, object]:
    body = _text(value, "")
    items: list[dict[str, object]] = []
    pattern = re.compile(r"^\s*[-*]\s+\[([ xX])\]\s+(.+?)\s*$")
    for line in body.splitlines():
        match = pattern.match(line)
        if match is None:
            continue
        checked = match.group(1).lower() == "x"
        items.append({"text": match.group(2), "checked": checked})
    completed = sum(1 for item in items if item["checked"] is True)
    return {"completed": completed, "total": len(items), "items": items}


def _checks(value: object) -> list[dict[str, str]]:
    if not isinstance(value, list):
        return []
    checks: list[dict[str, str]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        checks.append(
            {
                "name": _text(item.get("name"), _text(item.get("workflowName"), "")),
                "status": _text(item.get("status"), "unknown").lower(),
                "conclusion": _text(item.get("conclusion"), "unknown").lower(),
                "url": _text(item.get("detailsUrl"), ""),
            }
        )
    return checks


def _ci_summary(checks: list[dict[str, str]]) -> dict[str, object]:
    return {"status": _checks_status(checks), "checks": checks}


def _gate_summary(checks: list[dict[str, str]]) -> dict[str, object]:
    gate_checks = [check for check in checks if "gate" in check["name"].lower()]
    return {
        "status": _checks_status(gate_checks),
        "checks": gate_checks,
        "skipped_is_pass": False,
        "gate_pass_is_review": False,
    }


def _checks_status(checks: list[dict[str, str]]) -> str:
    if not checks:
        return "unknown"
    conclusions = {check["conclusion"] for check in checks}
    statuses = {check["status"] for check in checks}
    if conclusions & {"failure", "cancelled", "timed_out", "action_required"}:
        return "failure"
    if any(status not in {"completed", "success"} for status in statuses):
        return "pending"
    if conclusions <= {"success", "neutral"}:
        return "success"
    if "skipped" in conclusions:
        return "unknown"
    return "unknown"


def _github_reviews(value: object) -> list[dict[str, str]]:
    if not isinstance(value, list):
        return []
    reviews: list[dict[str, str]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        reviews.append(
            {
                "author": _author_login(item.get("author")),
                "state": _text(item.get("state"), "unknown").lower(),
                "submitted_at": _text(item.get("submittedAt"), ""),
                "body": _text(item.get("body"), ""),
            }
        )
    return reviews


def _review_comments(value: object) -> list[dict[str, str]]:
    if not isinstance(value, list):
        return []
    comments: list[dict[str, str]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        comments.append(
            {
                "author": _author_login(item.get("author")),
                "created_at": _text(item.get("createdAt"), ""),
                "url": _text(item.get("url"), ""),
                "body": _text(item.get("body"), ""),
            }
        )
    return comments


def _linked_issue(value: object) -> dict[str, object] | None:
    if not isinstance(value, list) or not value:
        return None
    first = value[0]
    if not isinstance(first, dict):
        return None
    return {
        "number": _int_value(first.get("number"), None),
        "title": _text(first.get("title"), ""),
        "state": _text(first.get("state"), "unknown").lower(),
        "url": _text(first.get("url"), ""),
    }


def _cosheaf_review_placeholder() -> dict[str, str]:
    return {
        "status": "not_imported",
        "source": "repository",
        "summary": "Cosheaf human review has not been imported from this PR.",
    }


def _author_login(value: object) -> str:
    if isinstance(value, dict):
        return _text(value.get("login"), "")
    return ""


def _text(value: object, fallback: str) -> str:
    return value.strip() if isinstance(value, str) and value.strip() else fallback


def _int_value(value: object, fallback: int | None) -> int | None:
    return value if isinstance(value, int) else fallback


def _now_iso() -> str:
    return _now_utc().isoformat().replace("+00:00", "Z")


def _git_status(context: RepoContext) -> tuple[str, ...]:
    output = _run_git(context, "status", "--porcelain=v1").stdout
    return tuple(line for line in output.splitlines() if line)


def _staged_changes(status: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(line for line in status if line[:2] != "??" and line[0] != " ")


def _ambiguous_dirty_lines(status: tuple[str, ...]) -> tuple[str, ...]:
    ambiguous: list[str] = []
    for line in status:
        if line[:2] == "??":
            ambiguous.append(f"untracked {line[3:]}")
        elif len(line) > 1 and line[1] != " ":
            ambiguous.append(f"unstaged {line[3:]}")
    return tuple(ambiguous)


def _now_utc() -> datetime:
    return datetime.now(UTC).replace(microsecond=0)


def _unique_strings(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        normalized = value.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result.append(normalized)
    return result


def _issue_yaml_data(issue: IssueRecord) -> dict[str, object]:
    data = issue.model_dump(mode="json", exclude={"severity"})
    if data.get("parent_issue") is None:
        data.pop("parent_issue", None)
    if data.get("close_reason") is None:
        data.pop("close_reason", None)
    if data.get("external_links") == []:
        data.pop("external_links", None)
    return data


__all__ = ["ForgeActionError", "ForgePreviewError", "ForgeService"]
