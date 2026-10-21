import { describe, expect, it } from "vitest";

import {
  AUTHORITY_GUIDE,
  GRAPH_PAGE_SIZE,
  REQUIRED_DATA_FILES,
  RELATIONSHIP_INLINE_LIMIT,
  ROUTES,
  artifactById,
  buildWorkbenchDashboard,
  contextPackForIssue,
  dependentsFor,
  dependenciesFor,
  filterArtifacts,
  gateVerdictExplanation,
  graphNeighborhoodFor,
  graphLayoutMode,
  issueById,
  loadSiteData,
  paginateItems,
  relationshipPreview,
  shouldDrawGraphConnector
} from "../src/lib/siteData";
import { decideRuntimeMode, runtimeModeLabel } from "../src/lib/runtimeMode";
import {
  PREVIEW_API_BASE,
  PREVIEW_AUTHORITY_WARNING,
  buildPreviewRequests
} from "../src/lib/previewActions";

describe("site data contract", () => {
  const hasChinese = (value: string) => /\p{Script=Han}/u.test(value);
  const isLocalizedText = (
    value: unknown
  ): value is { en: string; zh: string } =>
    typeof value === "object" &&
    value !== null &&
    "en" in value &&
    "zh" in value;
  const expectSwitchableText = (value: unknown) => {
    expect(isLocalizedText(value)).toBe(true);
    if (!isLocalizedText(value)) {
      return;
    }
    expect(value.en).not.toContain(" / ");
    expect(value.zh).not.toContain(" / ");
    expect(hasChinese(value.zh)).toBe(true);
    expect(hasChinese(value.en)).toBe(false);
  };

  it("defines the primary Workbench routes", () => {
    expect(ROUTES.map((route) => route.path)).toEqual([
      "/",
      "/artifacts",
      "/issues",
      "/forge/submit",
      "/forge/pr-status",
      "/graph",
      "/gates",
      "/authority"
    ]);
  });

  it("loads every static export fixture file", async () => {
    const data = await loadSiteData();

    expect(REQUIRED_DATA_FILES).toEqual([
      "site.json",
      "workspace.json",
      "artifacts.json",
      "issues.json",
      "graph.json",
      "gates.json",
      "context_packs.json",
      "reports.json",
      "authority_boundaries.json"
    ]);

    expect(Object.keys(data).sort()).toEqual([
      "artifacts",
      "authority_boundaries",
      "context_packs",
      "gates",
      "graph",
      "issues",
      "reports",
      "runtime",
      "site",
      "workspace"
    ]);

    for (const payload of [
      data.site,
      data.workspace,
      data.artifacts,
      data.issues,
      data.graph,
      data.gates,
      data.context_packs,
      data.reports,
      data.authority_boundaries
    ]) {
      expect(payload.schema_version).toBe(1);
      expect(payload.authority_notice.length).toBeGreaterThan(20);
    }
  });

  it("keeps website authority boundaries explicit", async () => {
    const data = await loadSiteData();

    expect(data.authority_boundaries.website_is_authority).toBe(false);
    expect(data.authority_boundaries.frontend_credentials_allowed).toBe(false);
    expect(data.authority_boundaries.skipped_is_pass).toBe(false);
    expect(data.gates.verdict).toBe("not_run");
  });

  it("filters artifacts by status type and domain", async () => {
    const data = await loadSiteData();

    expect(
      filterArtifacts(data.artifacts.artifacts, {
        status: "draft",
        type: "definition",
        domain: "graph-theory"
      }).map((artifact) => artifact.id)
    ).toEqual(["definition.graph"]);
  });

  it("links issues to their context pack and related artifacts", async () => {
    const issue = issueById("issue.example-private-claim");
    const contextPack = contextPackForIssue("issue.example-private-claim");

    expect(issue?.related_artifacts).toEqual([
      "claim.example-private",
      "definition.graph"
    ]);
    expect(contextPack?.public_only).toBe(true);
    expect(contextPack?.related_artifacts).toEqual(issue?.related_artifacts);
  });

  it("derives dependency and reverse dependency relationships", async () => {
    const data = await loadSiteData();

    expect(dependenciesFor("claim.example-private", data.graph.edges)).toEqual([
      "definition.graph"
    ]);
    expect(dependentsFor("definition.graph", data.graph.edges)).toEqual([
      "claim.example-private"
    ]);
  });

  it("chooses force graph layouts instead of card-list graph fallbacks", () => {
    expect(graphLayoutMode(0)).toBe("empty");
    expect(graphLayoutMode(1)).toBe("force");
    expect(graphLayoutMode(2)).toBe("force");
    expect(graphLayoutMode(3)).toBe("force");

    expect(shouldDrawGraphConnector(2, 1)).toBe(false);
    expect(shouldDrawGraphConnector(2, 0)).toBe(false);
    expect(shouldDrawGraphConnector(3, 2)).toBe(false);
  });

  it("limits inline relationship tags and exposes the hidden remainder", () => {
    const relationships = ["a", "b", "c", "d", "e", "f"];
    const preview = relationshipPreview(relationships);

    expect(RELATIONSHIP_INLINE_LIMIT).toBeGreaterThanOrEqual(3);
    expect(RELATIONSHIP_INLINE_LIMIT).toBeLessThanOrEqual(5);
    expect(preview.visible).toEqual(["a", "b", "c", "d"]);
    expect(preview.hidden).toEqual(["e", "f"]);
    expect(preview.hiddenCount).toBe(2);
    expect(relationshipPreview(["only"]).hiddenCount).toBe(0);
  });

  it("paginates large UI collections without dropping source order", () => {
    const items = Array.from({ length: GRAPH_PAGE_SIZE + 3 }, (_, index) => index);

    expect(paginateItems(items, 1, GRAPH_PAGE_SIZE)).toEqual({
      items: items.slice(0, GRAPH_PAGE_SIZE),
      page: 1,
      pageCount: 2,
      total: GRAPH_PAGE_SIZE + 3
    });
    expect(paginateItems(items, 99, GRAPH_PAGE_SIZE).items).toEqual(
      items.slice(GRAPH_PAGE_SIZE)
    );
  });

  it("builds a one-hop artifact dependency graph around the selected artifact", async () => {
    const data = await loadSiteData();
    const neighborhood = graphNeighborhoodFor("definition.graph", data.graph);

    expect(neighborhood.center?.artifact_id).toBe("definition.graph");
    expect(neighborhood.dependencies).toEqual([]);
    expect(neighborhood.dependents.map((node) => node.artifact_id)).toEqual([
      "claim.example-private"
    ]);
    expect(neighborhood.nodes.map((node) => node.artifact_id)).toEqual([
      "definition.graph",
      "claim.example-private"
    ]);
    expect(neighborhood.edges).toEqual([
      { source_id: "claim.example-private", target_id: "definition.graph" }
    ]);
  });

  it("explains gate verdicts without turning pass into accepted truth", async () => {
    const data = await loadSiteData();

    expect(gateVerdictExplanation(data.gates).en).toContain("not run");
    expect(
      gateVerdictExplanation({ ...data.gates, verdict: "pass" }).en
    ).toContain("does not accept");
  });

  it("covers the authority-boundary guide without overclaiming", () => {
    expect(AUTHORITY_GUIDE.map((entry) => entry.title.en)).toEqual([
      "What Cosheaf is",
      "What Cosheaf is not",
      "Why gate pass is not truth",
      "Why AI output is review context",
      "Why skipped verifier is not pass",
      "How public and private KB roots work",
      "How a result becomes accepted"
    ]);

    const guideText = AUTHORITY_GUIDE.map((entry) => entry.body.en).join(" ");

    expect(guideText).not.toMatch(/automatic theorem proving/i);
    expect(guideText).not.toMatch(/automatic promotion/i);
    expect(guideText).not.toMatch(/production ready/i);
  });

  it("keeps primary user-facing website chrome switchable, not inline bilingual", () => {
    for (const route of ROUTES) {
      expectSwitchableText(route.label);
    }

    for (const entry of AUTHORITY_GUIDE) {
      expectSwitchableText(entry.title);
      expectSwitchableText(entry.body);
    }

    expectSwitchableText(PREVIEW_AUTHORITY_WARNING);
  });

  it("builds preview-only action requests without write endpoints or tokens", async () => {
    const data = await loadSiteData();
    const requests = buildPreviewRequests(data, {
      base: "main",
      head: "website-preview-actions"
    });

    expect(PREVIEW_API_BASE).toBe("http://127.0.0.1:8765");
    expect(PREVIEW_AUTHORITY_WARNING.en).toContain("dry-run");
    expect(requests.map((request) => request.id)).toEqual([
      "local-issue",
      "github-issue",
      "github-pr",
      "review-packet"
    ]);
    expect(requests.map((request) => request.endpoint)).toEqual([
      "/api/forge/local-issues/preview",
      "/api/forge/issues/preview",
      "/api/forge/prs/preview",
      "/api/forge/review-packets/preview"
    ]);
    expect(requests.map((request) => request.label.en)).toEqual([
      "Local issue",
      "GitHub issue",
      "Pull request",
      "Review summary"
    ]);

    for (const request of requests) {
      expect(request.method).toBe("POST");
      expect(JSON.stringify(request.payload)).not.toMatch(/token|confirm|create/i);
    }
  });

  it("chooses runtime mode with switchable labels", () => {
    expect(decideRuntimeMode({ dev: true, mode: "auto" }).mode).toBe(
      "live-local"
    );
    expect(decideRuntimeMode({ dev: false, mode: "auto" }).mode).toBe(
      "static-demo"
    );
    expect(decideRuntimeMode({ dev: false, mode: "hosted" }).mode).toBe(
      "hosted-workspace"
    );
    expect(runtimeModeLabel("live-local")).toEqual({
      en: "Live local mode",
      zh: "本地实时模式"
    });
  });

  it("loads live backend data all at once when live mode is available", async () => {
    const calls: string[] = [];
    const payloads: Record<string, unknown> = {
      "/api/health": {
        schema_version: 1,
        status: "ok",
        local_actor: "Ada Local",
        local_actor_configured: true,
        local_actor_is_auth: false
      },
      "/api/workspace/live": {
        schema_version: 1,
        kind: "workspace_live",
        source_of_truth: "repository",
        authority_notice: "display only",
        workspace: {
          name: "live-workspace",
          mode: "legacy",
          kb_roots: [
            { name: "framework", path: "kb", readonly: false, priority: 0 }
          ],
          repo_root: "H:/repo"
        }
      },
      "/api/status": {
        schema_version: 1,
        kind: "repository_status",
        source_of_truth: "repository",
        authority_notice: "display only",
        status: "ok",
        validation: { ok: true, checked_count: 2, failures: [] }
      },
      "/api/artifacts/live": {
        schema_version: 1,
        kind: "artifacts_live",
        source_of_truth: "repository",
        authority_notice: "display only",
        count: 1,
        artifacts: [
          {
            id: "definition.live",
            type: "definition",
            title: "Live definition",
            status: "accepted",
            statement: "Live definition statement.",
            domain: ["graph-theory"],
            tags: ["live"],
            depends_on: [],
            review: { state: "accepted" },
            risk: { level: "low" },
            kb_root: "framework",
            kb_root_readonly: false,
            path: "kb/accepted/definitions/live.yaml"
          },
          {
            id: "claim.live",
            type: "claim",
            title: "Live claim",
            status: "draft",
            statement: "Live statement.",
            domain: ["graph-theory"],
            tags: ["live"],
            depends_on: ["definition.live"],
            review: { state: "requested" },
            risk: { level: "low" },
            kb_root: "framework",
            kb_root_readonly: false,
            path: "kb/draft/claims/live.yaml"
          }
        ]
      },
      "/api/issues/live": {
        schema_version: 1,
        kind: "issues_live",
        source_of_truth: "repository",
        authority_notice: "display only",
        count: 1,
        issues: [
          {
            id: "issue.live",
            type: "issue",
            title: "Live issue",
            status: "open",
            scope: "private",
            severity: "medium",
            summary: "Live issue summary.",
            labels: ["live"],
            related_artifacts: ["claim.live"],
            related_sources: []
          }
        ]
      },
      "/api/gates/latest": {
        schema_version: 1,
        kind: "gate_latest",
        source_of_truth: "repository",
        authority_notice: "display only",
        gate_report: {
          exists: true,
          path: ".cosheaf/reports/live-gate-report.json",
          verdict: "pass",
          report: {
            verdict: "pass",
            gates: [{ id: "G1", status: "pass" }],
            blocking_issues: [],
            nonblocking_issues: [],
            summary: { records_checked: 2 }
          }
        }
      }
    };
    const fetcher = async (url: string | URL): Promise<Response> => {
      const pathname = new URL(url.toString()).pathname;
      calls.push(pathname);
      return new Response(JSON.stringify(payloads[pathname]), { status: 200 });
    };

    const data = await loadSiteData({
      env: { dev: true, mode: "live", apiBase: "http://127.0.0.1:8765" },
      fetcher
    });

    expect(calls).toEqual([
      "/api/health",
      "/api/workspace/live",
      "/api/status",
      "/api/artifacts/live",
      "/api/issues/live",
      "/api/gates/latest"
    ]);
    expect(data.runtime.mode).toBe("live-local");
    expect(data.runtime.fallback).toBe(false);
    expect(data.runtime.local_actor).toBe("Ada Local");
    expect(data.runtime.local_actor_is_auth).toBe(false);
    expect(data.workspace.workspace_name).toBe("live-workspace");
    expect(data.artifacts.artifacts.map((artifact) => artifact.id)).toEqual([
      "definition.live",
      "claim.live"
    ]);
    expect(data.graph.edges).toEqual([
      { source_id: "claim.live", target_id: "definition.live" }
    ]);
    expect(data.context_packs.context_packs[0].issue_id).toBe("issue.live");
    expect(data.context_packs.context_packs[0].exists).toBe(false);
    expect(data.context_packs.context_packs[0].card_count).toBe(0);
    expect(data.reports.source_of_truth).toBe("repository");
    expect(data.authority_boundaries.source_of_truth).toBe("repository");
    expect(data.gates.verdict).toBe("pass");
    expect(artifactById("claim.live", data)?.title).toBe("Live claim");
  });

  it("computes live graph diagnostics and preserves missing dependency edges", async () => {
    const payloads: Record<string, unknown> = {
      "/api/health": { schema_version: 1, status: "ok" },
      "/api/workspace/live": {
        schema_version: 1,
        kind: "workspace_live",
        source_of_truth: "repository",
        authority_notice: "display only",
        workspace: { name: "live-workspace", mode: "legacy", kb_roots: [] }
      },
      "/api/status": {
        schema_version: 1,
        kind: "repository_status",
        source_of_truth: "repository",
        authority_notice: "display only",
        status: "validation_failed",
        validation: { ok: false, checked_count: 2, failures: [{}] }
      },
      "/api/artifacts/live": {
        schema_version: 1,
        kind: "artifacts_live",
        source_of_truth: "repository",
        authority_notice: "display only",
        count: 2,
        artifacts: [
          {
            id: "theorem.live",
            type: "theorem",
            title: "Live accepted theorem",
            status: "accepted",
            statement: "Accepted theorem.",
            domain: ["graph-theory"],
            tags: [],
            depends_on: ["lemma.draft", "lemma.missing"],
            review: { state: "accepted" },
            risk: { level: "low" },
            path: "kb/accepted/theorems/theorem.live.yaml"
          },
          {
            id: "lemma.draft",
            type: "lemma",
            title: "Live draft lemma",
            status: "draft",
            statement: "Draft lemma.",
            domain: ["graph-theory"],
            tags: [],
            depends_on: ["theorem.live"],
            review: { state: "requested" },
            risk: { level: "low" },
            path: "kb/draft/lemmas/lemma.draft.yaml"
          }
        ]
      },
      "/api/issues/live": {
        schema_version: 1,
        kind: "issues_live",
        source_of_truth: "repository",
        authority_notice: "display only",
        count: 1,
        issues: [
          {
            id: "issue.live",
            type: "issue",
            title: "Live issue",
            status: "open",
            scope: "private",
            severity: "medium",
            summary: "Summary.",
            labels: [],
            related_artifacts: ["theorem.live", "lemma.draft"],
            related_sources: [],
            context_pack: {
              exists: true,
              card_count: 5,
              public_only: true,
              private_context_included: false
            }
          }
        ]
      },
      "/api/gates/latest": {
        schema_version: 1,
        kind: "gate_latest",
        source_of_truth: "repository",
        authority_notice: "display only",
        gate_report: { exists: false, verdict: "not_run", report: {} }
      }
    };
    const data = await loadSiteData({
      env: { dev: true, mode: "live", apiBase: "http://127.0.0.1:8765" },
      fetcher: async (url: string | URL) =>
        new Response(JSON.stringify(payloads[new URL(url.toString()).pathname]), {
          status: 200
        })
    });

    expect(data.graph.edges).toEqual([
      { source_id: "theorem.live", target_id: "lemma.draft" },
      { source_id: "theorem.live", target_id: "lemma.missing" },
      { source_id: "lemma.draft", target_id: "theorem.live" }
    ]);
    expect(data.graph.missing_dependencies).toEqual([
      {
        source_id: "theorem.live",
        target_id: "lemma.missing",
        source_path: "kb/accepted/theorems/theorem.live.yaml",
        message: "missing dependency: lemma.missing"
      }
    ]);
    expect(data.graph.accepted_draft_violations).toEqual([
      {
        source_id: "theorem.live",
        target_id: "lemma.draft",
        source_path: "kb/accepted/theorems/theorem.live.yaml",
        message: "accepted artifact depends on draft artifact: lemma.draft"
      }
    ]);
    expect(data.graph.cycles).toEqual([
      ["lemma.draft", "theorem.live", "lemma.draft"]
    ]);
    expect(data.context_packs.context_packs[0].exists).toBe(true);
    expect(data.context_packs.context_packs[0].card_count).toBe(5);
  });

  it("derives a compact workbench dashboard from exported state", async () => {
    const data = await loadSiteData();
    const dashboard = buildWorkbenchDashboard(data);

    expect(dashboard.activeIssues.map((issue) => issue.id)).toEqual([
      "issue.example-private-claim"
    ]);
    expect(dashboard.draftReviewArtifacts.map((artifact) => artifact.id)).toEqual([
      "claim.example-private",
      "definition.graph"
    ]);
    expect(dashboard.promotion.ready).toEqual([]);
    expect(dashboard.promotion.blocked.map((artifact) => artifact.id)).toEqual([
      "claim.example-private",
      "definition.graph"
    ]);
    expect(dashboard.gateFailureCount).toBe(0);
    expect(dashboard.recentActions).toEqual([]);
    expect(dashboard.recentPrLinks).toEqual([]);
    expect(dashboard.nextActions.map((action) => action.href)).toEqual([
      "/issues/issue.example-private-claim/",
      "/context/issue.example-private-claim/",
      "/artifacts/claim.example-private/review-packet/",
      "/gates/"
    ]);
  });

  it("keeps promotion readiness conservative", async () => {
    const data = await loadSiteData();
    const [artifact] = data.artifacts.artifacts;
    const readyArtifact = {
      ...artifact,
      review_state: "human_reviewed",
      verifier_state: "pass",
      sources: ["external:source"],
      status: "draft"
    };
    const dashboard = buildWorkbenchDashboard({
      ...data,
      artifacts: {
        ...data.artifacts,
        artifacts: [readyArtifact],
        count: 1
      },
      gates: {
        ...data.gates,
        verdict: "pass",
        blocking_issues: []
      }
    });

    expect(dashboard.promotion.ready.map((item) => item.id)).toEqual([
      "claim.example-private"
    ]);
    expect(dashboard.promotion.blocked).toEqual([]);
  });

  it("falls back entirely to fixtures when any live endpoint is unavailable", async () => {
    const data = await loadSiteData({
      env: { dev: true, mode: "live", apiBase: "http://127.0.0.1:8765" },
      fetcher: async () => {
        throw new Error("backend offline");
      }
    });

    expect(data.runtime.mode).toBe("static-demo");
    expect(data.runtime.fallback).toBe(true);
    expect(data.runtime.fallback_reason).toContain("backend offline");
    expect(data.artifacts.artifacts.map((artifact) => artifact.id)).toContain(
      "claim.example-private"
    );
    expect(artifactById("claim.live", data)).toBeUndefined();
  });
});
