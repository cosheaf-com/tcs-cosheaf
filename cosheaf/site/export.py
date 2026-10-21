"""Deterministic sanitized JSON export for the static website."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cosheaf.core.artifact import BaseArtifact
from cosheaf.graph.claim_graph import GraphIssue, build_dependency_graph
from cosheaf.issues import ISSUE_AUTHORITY_NOTICE, LocalIssueService
from cosheaf.memory import artifact_card_from_loaded_record
from cosheaf.storage.loader import LoadedRecord, LoadError, load_artifacts
from cosheaf.storage.repo import RepoContext

SITE_SCHEMA_VERSION = 1
SITE_EXPORT_GENERATED_AT = "1970-01-01T00:00:00Z"
SITE_EXPORT_AUTHORITY_NOTICE = (
    "Website export data is display context only; it is not source of truth, "
    "proof, source metadata, human review, verifier pass, gate pass, accepted "
    "status, accepted theorem/refutation status, or promotion authority."
)
DEMO_FIXTURE_TAGS = frozenset({"workspace-demo", "site-demo", "demo-only"})
REQUIRED_SITE_EXPORT_FILES = (
    "site.json",
    "workspace.json",
    "artifacts.json",
    "issues.json",
    "graph.json",
    "gates.json",
    "context_packs.json",
    "reports.json",
    "authority_boundaries.json",
)


class SiteExportError(ValueError):
    """Raised when website export cannot be completed."""


@dataclass(frozen=True)
class SiteExportResult:
    """Machine-readable website export result."""

    out: Path
    files: tuple[str, ...]
    public_only: bool
    demo: bool
    authority_notice: str = SITE_EXPORT_AUTHORITY_NOTICE

    @property
    def file_count(self) -> int:
        """Return the number of exported JSON files."""
        return len(self.files)

    def to_dict(self) -> dict[str, object]:
        """Return a deterministic JSON-serializable result."""
        return {
            "schema_version": SITE_SCHEMA_VERSION,
            "out": str(self.out),
            "files": list(self.files),
            "file_count": self.file_count,
            "public_only": self.public_only,
            "demo": self.demo,
            "authority_notice": self.authority_notice,
        }


def export_site_data(
    context: RepoContext,
    out: str | Path,
    *,
    public_only: bool = False,
    demo: bool = False,
) -> SiteExportResult:
    """Write deterministic website JSON sidecars.

    Demo exports are forced through the public-only sanitizer because demo data
    is intended for public builds.
    """
    effective_public_only = public_only or demo
    output_dir = Path(out)
    try:
        records = tuple(load_artifacts(context))
    except LoadError as exc:
        raise SiteExportError(f"cannot load repository records: {exc}") from exc

    exported_artifacts = _exported_artifact_records(
        records,
        public_only=effective_public_only,
        demo=demo,
    )
    exported_artifact_ids = frozenset(record.id for record in exported_artifacts)
    private_artifact_ids = frozenset(
        record.id for record in records if _is_private_artifact(record)
    )
    artifact_cards = [
        _artifact_payload(record, exported_artifact_ids=exported_artifact_ids)
        for record in exported_artifacts
    ]
    issues = _issue_payloads(
        context,
        public_only=effective_public_only,
        demo=demo,
        exported_artifact_ids=exported_artifact_ids,
    )
    workspace = _workspace_payload(
        context,
        public_only=effective_public_only,
        demo=demo,
    )

    payloads = {
        "site.json": _base_payload(
            "site",
            {
                "generated_at": SITE_EXPORT_GENERATED_AT,
                "demo": demo,
                "public_only": effective_public_only,
                "demo_private_fixtures_allowed": demo,
                "source_of_truth": "repository",
                "sidecar": True,
                "files": list(REQUIRED_SITE_EXPORT_FILES),
            },
        ),
        "workspace.json": _base_payload("workspace", workspace),
        "artifacts.json": _base_payload(
            "artifacts",
            {
                "artifacts": artifact_cards,
                "count": len(artifact_cards),
                "public_only": effective_public_only,
                "demo_private_fixtures_allowed": demo,
            },
        ),
        "issues.json": _base_payload(
            "issues",
            {
                "issues": issues,
                "count": len(issues),
                "authority_notice": ISSUE_AUTHORITY_NOTICE,
            },
        ),
        "graph.json": _base_payload(
            "graph",
            _graph_payload(
                exported_artifacts,
                private_artifact_ids=private_artifact_ids,
                public_only=effective_public_only,
            ),
        ),
        "gates.json": _base_payload(
            "gates",
            {
                "verdict": "not_run",
                "blocking_issues": [],
                "nonblocking_issues": [],
                "gates": [],
                "report_paths": [],
                "note": "Static export does not run gates; use cosheaf gate run.",
            },
        ),
        "context_packs.json": _base_payload(
            "context_packs",
            {
                "context_packs": [
                    {
                        "issue_id": issue["id"],
                        "public_only": effective_public_only,
                        "private_context_included": False,
                        "demo_private_context_included": any(
                            issue.get("demo_fixture", False)
                            for artifact_id in issue["related_artifacts"]
                            if artifact_id in private_artifact_ids
                        ),
                        "related_artifacts": issue["related_artifacts"],
                        "card_count": len(issue["related_artifacts"]),
                    }
                    for issue in issues
                ],
            },
        ),
        "reports.json": _base_payload(
            "reports",
            {
                "reports": [],
                "note": "No static reports are embedded in the site export.",
            },
        ),
        "authority_boundaries.json": _base_payload(
            "authority_boundaries",
            _authority_boundaries_payload(),
        ),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    for filename in REQUIRED_SITE_EXPORT_FILES:
        _write_json(output_dir / filename, payloads[filename])

    return SiteExportResult(
        out=output_dir,
        files=REQUIRED_SITE_EXPORT_FILES,
        public_only=effective_public_only,
        demo=demo,
    )


def _base_payload(kind: str, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": SITE_SCHEMA_VERSION,
        "kind": kind,
        "authority_notice": SITE_EXPORT_AUTHORITY_NOTICE,
        **payload,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _exported_artifact_records(
    records: tuple[LoadedRecord, ...],
    *,
    public_only: bool,
    demo: bool,
) -> tuple[LoadedRecord, ...]:
    artifacts = [
        record for record in records if isinstance(record.record, BaseArtifact)
    ]
    if public_only:
        artifacts = [
            record
            for record in artifacts
            if not _is_private_artifact(record)
            or (demo and _is_demo_artifact(record))
        ]
    return tuple(sorted(artifacts, key=lambda record: (record.id, _path(record))))


def _artifact_payload(
    record: LoadedRecord,
    *,
    exported_artifact_ids: frozenset[str],
) -> dict[str, Any]:
    artifact = record.record
    if not isinstance(artifact, BaseArtifact):
        raise TypeError(f"record is not an artifact: {record.id}")
    card = artifact_card_from_loaded_record(record).to_dict()
    card["authors"] = artifact.authors
    card["demo_fixture"] = _is_demo_artifact(record)
    card["depends_on"] = [
        dependency
        for dependency in card["depends_on"]
        if dependency in exported_artifact_ids or dependency.startswith("external:")
    ]
    card["supersedes"] = [
        dependency
        for dependency in artifact.supersedes
        if dependency in exported_artifact_ids or dependency.startswith("external:")
    ]
    return card


def _issue_payloads(
    context: RepoContext,
    *,
    public_only: bool,
    demo: bool,
    exported_artifact_ids: frozenset[str],
) -> list[dict[str, Any]]:
    issues = []
    for issue in LocalIssueService(context).list().issues:
        demo_fixture = _is_demo_issue(issue)
        if public_only and issue.scope == "private" and not (demo and demo_fixture):
            continue
        data = issue.model_dump(mode="json")
        data["demo_fixture"] = demo_fixture
        data["related_artifacts"] = [
            artifact_id
            for artifact_id in issue.related_artifacts
            if artifact_id in exported_artifact_ids
        ]
        issues.append(data)
    return sorted(issues, key=lambda item: item["id"])


def _workspace_payload(
    context: RepoContext,
    *,
    public_only: bool,
    demo: bool,
) -> dict[str, Any]:
    config = context.workspace_config
    roots = []
    for root in config.ordered_kb:
        private_root = _private_text(root.name, root.path)
        if public_only and private_root and not demo:
            continue
        roots.append(
            {
                "name": root.name,
                "path": root.path,
                "readonly": root.readonly,
                "priority": root.priority,
                "demo_private_fixtures_allowed": demo and private_root,
            }
        )
    return {
        "workspace_name": config.name,
        "mode": "configured" if config.configured else "legacy",
        "kb_roots": roots,
        "policy": {
            "private_can_depend_on_public": config.policy.private_can_depend_on_public,
            "public_can_depend_on_private": config.policy.public_can_depend_on_private,
            "accepted_requires_source": config.policy.accepted_requires_source,
        },
        "public_only": public_only,
        "demo_private_fixtures_allowed": demo,
    }


def _graph_payload(
    records: tuple[LoadedRecord, ...],
    *,
    private_artifact_ids: frozenset[str],
    public_only: bool,
) -> dict[str, Any]:
    graph = build_dependency_graph(records)
    exported_ids = frozenset(node.artifact_id for node in graph.nodes)
    edges = [
        edge
        for edge in graph.edges
        if edge.source_id in exported_ids and edge.target_id in exported_ids
    ]
    return {
        "nodes": [
            {
                "artifact_id": node.artifact_id,
                "artifact_type": node.artifact_type,
                "status": node.status,
                "path": node.path,
                "title": node.title,
                "domain": list(node.domain),
            }
            for node in graph.nodes
        ],
        "edges": [
            {"source_id": edge.source_id, "target_id": edge.target_id}
            for edge in edges
        ],
        "cycles": _cycles_for_edges(
            tuple((edge.source_id, edge.target_id) for edge in edges),
            exported_ids,
        ),
        "missing_dependencies": [
            _graph_issue_payload(issue)
            for issue in graph.missing_dependencies
            if not (public_only and issue.target_id in private_artifact_ids)
        ],
        "accepted_draft_violations": [
            _graph_issue_payload(issue)
            for issue in graph.accepted_draft_violations
            if not (public_only and issue.target_id in private_artifact_ids)
        ],
    }


def _graph_issue_payload(issue: GraphIssue) -> dict[str, str]:
    return {
        "source_id": issue.source_id,
        "target_id": issue.target_id,
        "source_path": issue.source_path,
        "message": issue.message,
    }


def _cycles_for_edges(
    edges: tuple[tuple[str, str], ...],
    artifact_ids: frozenset[str],
) -> list[list[str]]:
    adjacency: dict[str, list[str]] = {
        artifact_id: [] for artifact_id in sorted(artifact_ids)
    }
    for source_id, target_id in edges:
        adjacency[source_id].append(target_id)

    cycles: set[tuple[str, ...]] = set()
    stack: list[str] = []
    visited: set[str] = set()

    def visit(artifact_id: str) -> None:
        if artifact_id in stack:
            members = stack[stack.index(artifact_id) :] + [artifact_id]
            cycles.add(_canonical_cycle(tuple(members)))
            return
        if artifact_id in visited:
            return
        stack.append(artifact_id)
        for dependency_id in sorted(adjacency[artifact_id]):
            visit(dependency_id)
        stack.pop()
        visited.add(artifact_id)

    for artifact_id in sorted(artifact_ids):
        visit(artifact_id)
    return [list(cycle) for cycle in sorted(cycles)]


def _canonical_cycle(cycle: tuple[str, ...]) -> tuple[str, ...]:
    members = cycle[:-1]
    rotations = [
        members[index:] + members[:index] + (members[index],)
        for index in range(len(members))
    ]
    return min(rotations)


def _authority_boundaries_payload() -> dict[str, Any]:
    return {
        "website_is_authority": False,
        "export_is_source_of_truth": False,
        "accepted_write_performed": False,
        "frontend_credentials_allowed": False,
        "gate_pass_is_truth": False,
        "verifier_pass_is_truth": False,
        "skipped_is_pass": False,
        "ai_output_is_human_review": False,
        "github_issue_is_accepted_knowledge": False,
        "notices": [
            "Repository YAML/JSON remains source of truth.",
            "Website pages and exports are display context only.",
            (
                "Gate, verifier, AI, operator, issue, PR, and report output do "
                "not grant accepted authority."
            ),
            "Skipped verifier or checker output is not pass.",
            (
                "Lean #check only confirms import/symbol resolution, not "
                "semantic alignment."
            ),
        ],
    }


def _is_private_artifact(record: LoadedRecord) -> bool:
    return isinstance(record.record, BaseArtifact) and record.is_private_scope


def _is_demo_artifact(record: LoadedRecord) -> bool:
    if not isinstance(record.record, BaseArtifact):
        return False
    return _has_demo_tags(record.record.tags)


def _is_demo_issue(issue: Any) -> bool:
    labels = getattr(issue, "labels", [])
    return _has_demo_tags(labels)


def _has_demo_tags(values: list[str]) -> bool:
    return bool(DEMO_FIXTURE_TAGS.intersection(value.lower() for value in values))


def _private_text(*values: str) -> bool:
    for value in values:
        parts = value.lower().replace("\\", "/").split("/")
        if any("private" in part for part in parts):
            return True
    return False


def _path(record: LoadedRecord) -> str:
    return record.source_path.as_posix()
