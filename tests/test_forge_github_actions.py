from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]
from typer.testing import CliRunner

from cosheaf.cli import app

runner = CliRunner()


def _write_issue(repo_root: Path) -> Path:
    path = repo_root / "issues" / "open" / "issue.fixture.github.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(
            {
                "id": "issue.fixture.github",
                "type": "issue",
                "title": "GitHub forge fixture",
                "status": "open",
                "summary": "Create a GitHub issue from local YAML.",
                "created_at": "2026-06-19T00:00:00Z",
                "updated_at": "2026-06-19T00:00:00Z",
                "authors": ["tester"],
                "labels": ["forge", "github"],
                "related_artifacts": [],
                "related_sources": [],
                "scope": "private",
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return path


def test_forge_issue_create_requires_confirm(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    issue_path = _write_issue(tmp_path)

    def fail_subprocess_run(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("unconfirmed GitHub issue create must not run gh")

    monkeypatch.setattr(subprocess, "run", fail_subprocess_run)

    result = runner.invoke(
        app,
        [
            "forge",
            "issue",
            "create",
            "--from",
            str(issue_path.relative_to(tmp_path)),
            "--repo-root",
            str(tmp_path),
            "--json",
        ],
    )

    assert result.exit_code == 1, result.output
    payload = json.loads(result.output)
    assert payload["code"] == "forge_confirm_required"


def test_forge_issue_create_calls_gh_and_links_local_issue(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    issue_path = _write_issue(tmp_path)
    calls: list[list[str]] = []

    def fake_subprocess_run(
        args: list[str],
        **_kwargs: object,
    ) -> subprocess.CompletedProcess[str]:
        calls.append(args)
        return subprocess.CompletedProcess(
            args=args,
            returncode=0,
            stdout="https://github.com/CheemsaDoge/tcs-cosheaf/issues/999\n",
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", fake_subprocess_run)

    result = runner.invoke(
        app,
        [
            "forge",
            "issue",
            "create",
            "--from",
            str(issue_path.relative_to(tmp_path)),
            "--confirm",
            "--repo-root",
            str(tmp_path),
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["action"] == "github_issue_create"
    assert payload["action_performed"] is True
    assert payload["github_issue_created"] is True
    assert payload["github_writes_performed"] is True
    assert payload["network_calls_performed"] is True
    assert payload["git_writes_performed"] is False
    assert payload["local_issue_closed"] is False
    assert payload["github_issue_url"].endswith("/issues/999")
    assert calls == [
        [
            "gh",
            "issue",
            "create",
            "--title",
            "GitHub forge fixture",
            "--body",
            "Create a GitHub issue from local YAML.",
            "--label",
            "forge",
            "--label",
            "github",
        ]
    ]

    updated = yaml.safe_load(issue_path.read_text(encoding="utf-8"))
    assert updated["status"] == "open"
    assert updated["external_links"] == [
        "https://github.com/CheemsaDoge/tcs-cosheaf/issues/999"
    ]


def test_forge_issue_create_reports_bad_source_as_action_error(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    def fail_subprocess_run(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("invalid GitHub issue source must not run gh")

    monkeypatch.setattr(subprocess, "run", fail_subprocess_run)

    result = runner.invoke(
        app,
        [
            "forge",
            "issue",
            "create",
            "--from",
            "issues/open/missing.yaml",
            "--confirm",
            "--repo-root",
            str(tmp_path),
            "--json",
        ],
    )

    assert result.exit_code == 1, result.output
    payload = json.loads(result.output)
    assert payload["code"] == "forge_invalid_input"
    assert "does not exist" in payload["message"]


def test_forge_pr_create_requires_confirm(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    def fail_subprocess_run(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("unconfirmed GitHub PR create must not run gh")

    monkeypatch.setattr(subprocess, "run", fail_subprocess_run)

    result = runner.invoke(
        app,
        [
            "forge",
            "pr",
            "create",
            "--base",
            "main",
            "--head",
            "feature",
            "--repo-root",
            str(tmp_path),
            "--json",
        ],
    )

    assert result.exit_code == 1, result.output
    payload = json.loads(result.output)
    assert payload["code"] == "forge_confirm_required"


def test_forge_pr_create_calls_gh_without_push_or_token_storage(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    calls: list[list[str]] = []

    def fake_subprocess_run(
        args: list[str],
        **_kwargs: object,
    ) -> subprocess.CompletedProcess[str]:
        calls.append(args)
        return subprocess.CompletedProcess(
            args=args,
            returncode=0,
            stdout="https://github.com/CheemsaDoge/tcs-cosheaf/pull/1000\n",
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", fake_subprocess_run)

    result = runner.invoke(
        app,
        [
            "forge",
            "pr",
            "create",
            "--base",
            "main",
            "--head",
            "feature",
            "--draft",
            "--confirm",
            "--repo-root",
            str(tmp_path),
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["action"] == "github_pr_create"
    assert payload["action_performed"] is True
    assert payload["github_pr_created"] is True
    assert payload["github_writes_performed"] is True
    assert payload["network_calls_performed"] is True
    assert payload["git_writes_performed"] is False
    assert payload["push_performed"] is False
    assert payload["github_pr_url"].endswith("/pull/1000")
    assert calls == [
        [
            "gh",
            "pr",
            "create",
            "--base",
            "main",
            "--head",
            "feature",
            "--title",
            "Merge feature into main",
            "--body",
            "Forge-created PR for feature into main.",
            "--draft",
        ]
    ]
    assert not any("token" in path.name.lower() for path in tmp_path.rglob("*"))


def test_forge_pr_submit_requires_confirm(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    def fail_subprocess_run(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("unconfirmed PR submit must not run git or gh")

    monkeypatch.setattr(subprocess, "run", fail_subprocess_run)

    result = runner.invoke(
        app,
        [
            "forge",
            "pr",
            "submit",
            "--base",
            "main",
            "--head",
            "feature",
            "--repo-root",
            str(tmp_path),
            "--json",
        ],
    )

    assert result.exit_code == 1, result.output
    payload = json.loads(result.output)
    assert payload["code"] == "forge_confirm_required"


def test_forge_pr_submit_refuses_protected_head(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    def fail_subprocess_run(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("protected PR submit must not run git or gh")

    monkeypatch.setattr(subprocess, "run", fail_subprocess_run)

    result = runner.invoke(
        app,
        [
            "forge",
            "pr",
            "submit",
            "--base",
            "main",
            "--head",
            "main",
            "--confirm",
            "--repo-root",
            str(tmp_path),
            "--json",
        ],
    )

    assert result.exit_code == 1, result.output
    payload = json.loads(result.output)
    assert payload["code"] == "forge_protected_branch"


def test_forge_pr_submit_validates_gates_pushes_and_creates_draft_pr(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    calls: list[list[str]] = []

    def fake_subprocess_run(
        args: list[str],
        **_kwargs: object,
    ) -> subprocess.CompletedProcess[str]:
        calls.append(args)
        if args == ["git", "branch", "--show-current"]:
            stdout = "feature\n"
        elif args == ["git", "status", "--porcelain=v1"]:
            stdout = ""
        elif args in (["git", "rev-parse", "HEAD"], ["git", "rev-parse", "feature"]):
            stdout = "deadbeef1234567890\n"
        elif args[:2] == ["git", "push"]:
            stdout = ""
        elif args[:3] == ["gh", "pr", "create"]:
            stdout = "https://github.com/CheemsaDoge/tcs-cosheaf/pull/1003\n"
        else:
            raise AssertionError(f"unexpected subprocess call: {args}")
        return subprocess.CompletedProcess(
            args=args,
            returncode=0,
            stdout=stdout,
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", fake_subprocess_run)

    result = runner.invoke(
        app,
        [
            "forge",
            "pr",
            "submit",
            "--base",
            "main",
            "--head",
            "feature",
            "--draft",
            "--confirm",
            "--repo-root",
            str(tmp_path),
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["action"] == "github_pr_submit"
    assert payload["action_performed"] is True
    assert payload["validation_performed"] is True
    assert payload["gate_performed"] is True
    assert payload["git_writes_performed"] is True
    assert payload["network_calls_performed"] is True
    assert payload["github_writes_performed"] is True
    assert payload["push_performed"] is True
    assert payload["github_pr_created"] is True
    assert payload["commit_hash"] == "deadbeef1234567890"
    assert payload["github_pr_url"].endswith("/pull/1003")
    assert calls == [
        ["git", "branch", "--show-current"],
        ["git", "status", "--porcelain=v1"],
        ["git", "rev-parse", "HEAD"],
        ["git", "rev-parse", "feature"],
        ["git", "branch", "--show-current"],
        ["git", "status", "--porcelain=v1"],
        ["git", "rev-parse", "HEAD"],
        ["git", "rev-parse", "feature"],
        ["git", "push", "-u", "origin", "feature"],
        [
            "gh",
            "pr",
            "create",
            "--base",
            "main",
            "--head",
            "feature",
            "--title",
            "Merge feature into main",
            "--body",
            "Forge-created PR for feature into main.",
            "--draft",
        ],
    ]
    assert not any("token" in path.name.lower() for path in tmp_path.rglob("*"))


def test_forge_pr_submit_refuses_branch_mismatch_against_unvalidated_branch(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    real_run = subprocess.run

    def _git(*args: str) -> str:
        res = real_run(
            ["git", *args],
            cwd=tmp_path,
            text=True,
            capture_output=True,
            check=False,
        )
        assert res.returncode == 0, res.stderr
        return res.stdout.strip()

    _git("init", "-b", "main")
    _git("config", "user.name", "Tester")
    _git("config", "user.email", "tester@example.invalid")
    (tmp_path / ".gitignore").write_text(".cosheaf/\n", encoding="utf-8")
    (tmp_path / "README.md").write_text("clean\n", encoding="utf-8")
    _git("add", ".")
    _git("commit", "-m", "Initial clean commit")

    _git("switch", "-c", "bad-branch")
    bad_dir = tmp_path / "kb" / "drafts"
    bad_dir.mkdir(parents=True, exist_ok=True)
    (bad_dir / "broken.yaml").write_text("kind: [unclosed\n", encoding="utf-8")
    _git("add", ".")
    _git("commit", "-m", "Add broken YAML on bad-branch")

    _git("switch", "-c", "clean-branch", "main")

    push_or_gh_calls: list[list[str]] = []

    def guarded_run(
        args: list[str],
        **kwargs: Any,
    ) -> subprocess.CompletedProcess[str]:
        if args[:2] == ["git", "push"] or args[:1] == ["gh"]:
            push_or_gh_calls.append(args)
            raise AssertionError(f"must not push or call gh on branch mismatch: {args}")
        res: subprocess.CompletedProcess[str] = real_run(args, **kwargs)
        return res

    monkeypatch.setattr(subprocess, "run", guarded_run)

    result = runner.invoke(
        app,
        [
            "forge",
            "pr",
            "submit",
            "--base",
            "main",
            "--head",
            "bad-branch",
            "--draft",
            "--confirm",
            "--repo-root",
            str(tmp_path),
            "--json",
        ],
    )

    assert result.exit_code == 1, result.output
    payload = json.loads(result.output)
    assert payload["code"] == "forge_branch_mismatch"
    assert "clean-branch" in payload["message"]
    assert "bad-branch" in payload["message"]
    assert push_or_gh_calls == []


def test_forge_pr_submit_refuses_dirty_working_tree(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    def fake_subprocess_run(
        args: list[str],
        **_kwargs: object,
    ) -> subprocess.CompletedProcess[str]:
        if args == ["git", "branch", "--show-current"]:
            stdout = "feature\n"
        elif args == ["git", "status", "--porcelain=v1"]:
            stdout = " M kb/drafts/uncommitted.yaml\n"
        else:
            raise AssertionError(f"unexpected subprocess call on dirty tree: {args}")
        return subprocess.CompletedProcess(
            args=args,
            returncode=0,
            stdout=stdout,
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", fake_subprocess_run)

    result = runner.invoke(
        app,
        [
            "forge",
            "pr",
            "submit",
            "--base",
            "main",
            "--head",
            "feature",
            "--confirm",
            "--repo-root",
            str(tmp_path),
            "--json",
        ],
    )

    assert result.exit_code == 1, result.output
    payload = json.loads(result.output)
    assert payload["code"] == "forge_dirty_state"


def test_forge_pr_submit_refuses_when_head_sha_changes_after_validation(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    rev_parse_calls = 0

    def fake_subprocess_run(
        args: list[str],
        **_kwargs: object,
    ) -> subprocess.CompletedProcess[str]:
        nonlocal rev_parse_calls
        if args == ["git", "branch", "--show-current"]:
            stdout = "feature\n"
        elif args == ["git", "status", "--porcelain=v1"]:
            stdout = ""
        elif args in (["git", "rev-parse", "HEAD"], ["git", "rev-parse", "feature"]):
            rev_parse_calls += 1
            stdout = "sha_before\n" if rev_parse_calls <= 2 else "sha_after\n"
        else:
            raise AssertionError(f"unexpected subprocess call on SHA drift: {args}")
        return subprocess.CompletedProcess(
            args=args,
            returncode=0,
            stdout=stdout,
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", fake_subprocess_run)

    result = runner.invoke(
        app,
        [
            "forge",
            "pr",
            "submit",
            "--base",
            "main",
            "--head",
            "feature",
            "--confirm",
            "--repo-root",
            str(tmp_path),
            "--json",
        ],
    )

    assert result.exit_code == 1, result.output
    payload = json.loads(result.output)
    assert payload["code"] == "forge_head_sha_mismatch"


def test_forge_sync_is_read_only(tmp_path: Path, monkeypatch: Any) -> None:
    def fail_subprocess_run(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("forge sync must not call gh in A4.3")

    monkeypatch.setattr(subprocess, "run", fail_subprocess_run)

    result = runner.invoke(
        app,
        ["forge", "sync", "--repo-root", str(tmp_path), "--json"],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["action"] == "sync"
    assert payload["action_performed"] is False
    assert payload["github_writes_performed"] is False
    assert payload["network_calls_performed"] is False
