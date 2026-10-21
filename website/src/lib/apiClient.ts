import type {
  ArtifactCard,
  ArtifactsPayload,
  AuthorityBoundariesPayload,
  ContextPacksPayload,
  GatesPayload,
  GraphPayload,
  IssueSummary,
  IssuesPayload,
  SiteData,
  SiteExportPayload,
  SitePayload,
  WorkspacePayload
} from "./siteData";
import type { RuntimeMetadata } from "./runtimeMode";

export type FetchLike = (
  input: string | URL,
  init?: RequestInit
) => Promise<Response>;

export interface FetchLiveSiteDataOptions {
  apiBase: string;
  fetcher?: FetchLike;
  fixtureData: SiteData;
  runtime: RuntimeMetadata;
  timeoutMs?: number;
}

interface LivePayload {
  authority_notice?: string;
  source_of_truth?: string;
}

export async function fetchLiveSiteData(
  options: FetchLiveSiteDataOptions
): Promise<SiteData> {
  const [health, workspace, status, artifacts, issues, gates] = await Promise.all(
    [
      fetchJson(options, "/api/health"),
      fetchJson(options, "/api/workspace/live"),
      fetchJson(options, "/api/status"),
      fetchJson(options, "/api/artifacts/live"),
      fetchJson(options, "/api/issues/live"),
      fetchJson(options, "/api/gates/latest")
    ]
  );

  return mapLiveSiteData({
    fixtureData: options.fixtureData,
    runtime: options.runtime,
    health,
    workspace,
    status,
    artifacts,
    issues,
    gates
  });
}

async function fetchJson(
  options: FetchLiveSiteDataOptions,
  endpoint: string
): Promise<Record<string, unknown>> {
  const fetcher = options.fetcher ?? globalThis.fetch;
  if (!fetcher) {
    throw new Error("fetch is not available");
  }
  const controller =
    typeof AbortController !== "undefined" ? new AbortController() : undefined;
  const timeout = controller
    ? globalThis.setTimeout(() => controller.abort(), options.timeoutMs ?? 5000)
    : undefined;
  try {
    const response = await fetcher(new URL(endpoint, options.apiBase), {
      signal: controller?.signal
    });
    if (!response.ok) {
      throw new Error(`${endpoint} returned HTTP ${response.status}`);
    }
    const payload = await response.json();
    if (!isRecord(payload)) {
      throw new Error(`${endpoint} did not return a JSON object`);
    }
    return payload;
  } finally {
    if (timeout !== undefined) {
      globalThis.clearTimeout(timeout);
    }
  }
}

function mapLiveSiteData(input: {
  fixtureData: SiteData;
  runtime: RuntimeMetadata;
  health: Record<string, unknown>;
  workspace: Record<string, unknown>;
  status: Record<string, unknown>;
  artifacts: Record<string, unknown>;
  issues: Record<string, unknown>;
  gates: Record<string, unknown>;
}): SiteData {
  const artifacts = mapArtifacts(input.fixtureData, input.artifacts);
  const issues = mapIssues(input.fixtureData, input.issues);
  const graph = mapGraph(input.fixtureData, artifacts.artifacts);
  const gates = mapGates(input.fixtureData, input.gates);
  return {
    site: mapSite(input.fixtureData.site),
    workspace: mapWorkspace(input.fixtureData, input.workspace),
    artifacts,
    issues,
    graph,
    gates,
    context_packs: mapContextPacks(
      input.fixtureData,
      issues.issues,
      input.issues
    ),
    reports: mapReports(input.fixtureData, input.status, gates),
    authority_boundaries: mapAuthorityBoundaries(
      input.fixtureData,
      input.health
    ),
    runtime: runtimeWithHealth(input.runtime, input.health)
  };
}

function runtimeWithHealth(
  runtime: RuntimeMetadata,
  health: Record<string, unknown>
): RuntimeMetadata {
  if (runtime.mode !== "live-local") {
    return runtime;
  }
  return {
    ...runtime,
    local_actor: textOrUndefined(health.local_actor),
    local_actor_is_auth: Boolean(health.local_actor_is_auth)
  };
}

function mapSite(site: SitePayload): SitePayload {
  return {
    ...site,
    demo: false,
    public_only: false,
    source_of_truth: "repository"
  };
}

function mapWorkspace(
  fixtureData: SiteData,
  payload: Record<string, unknown>
): WorkspacePayload {
  const workspace = asRecord(payload.workspace);
  const roots = arrayOfRecords(workspace.kb_roots);
  return {
    ...fixtureData.workspace,
    authority_notice: authorityNotice(payload, fixtureData.workspace),
    workspace_name: text(workspace.name, fixtureData.workspace.workspace_name),
    mode: text(workspace.mode, fixtureData.workspace.mode),
    public_only: false,
    kb_roots: roots.map((root) => ({
      name: text(root.name, "workspace"),
      path: text(root.path, "."),
      readonly: Boolean(root.readonly),
      priority: numberValue(root.priority, 0),
      demo_private_fixtures_allowed: false
    }))
  };
}

function mapArtifacts(
  fixtureData: SiteData,
  payload: Record<string, unknown>
): ArtifactsPayload {
  const artifacts = arrayOfRecords(payload.artifacts).map((artifact) =>
    artifactCard(artifact)
  );
  return {
    ...fixtureData.artifacts,
    authority_notice: authorityNotice(payload, fixtureData.artifacts),
    artifacts,
    count: artifacts.length,
    public_only: false,
    demo_private_fixtures_allowed: false
  };
}

function artifactCard(artifact: Record<string, unknown>): ArtifactCard {
  return {
    id: text(artifact.id, ""),
    title: text(artifact.title, text(artifact.id, "Untitled artifact")),
    type: text(artifact.type, "artifact"),
    status: text(artifact.status, "draft"),
    root_scope: text(artifact.kb_root, "workspace"),
    summary: text(artifact.summary, text(artifact.statement, "")),
    statement:
      typeof artifact.statement === "string" ? artifact.statement : undefined,
    domain: stringArray(artifact.domain),
    authors: stringArray(artifact.authors),
    tags: stringArray(artifact.tags),
    depends_on: stringArray(artifact.depends_on),
    supersedes: stringArray(artifact.supersedes),
    demo_fixture: false,
    verifier_state: "not_run",
    review_state: text(asRecord(artifact.review).state, "requested"),
    risk_flags: riskFlags(artifact),
    sources: sourceLabels(artifact.sources),
    path: text(artifact.path, "")
  };
}

function mapIssues(
  fixtureData: SiteData,
  payload: Record<string, unknown>
): IssuesPayload {
  const issues = arrayOfRecords(payload.issues).map((issue) => issueSummary(issue));
  return {
    ...fixtureData.issues,
    authority_notice: authorityNotice(payload, fixtureData.issues),
    issues,
    count: issues.length
  };
}

function issueSummary(issue: Record<string, unknown>): IssueSummary {
  return {
    id: text(issue.id, ""),
    title: text(issue.title, text(issue.id, "Untitled issue")),
    type: text(issue.type, "issue"),
    status: text(issue.status, "open"),
    scope: text(issue.scope, "private"),
    severity: text(issue.severity, "medium"),
    summary: text(issue.summary, text(issue.description, "")),
    authors: stringArray(issue.authors),
    labels: stringArray(issue.labels),
    related_artifacts: stringArray(issue.related_artifacts),
    related_sources: stringArray(issue.related_sources),
    demo_fixture: false
  };
}

function mapGraph(fixtureData: SiteData, artifacts: ArtifactCard[]): GraphPayload {
  const nodes = artifacts.map((artifact) => ({
    artifact_id: artifact.id,
    artifact_type: artifact.type,
    title: artifact.title,
    status: artifact.status,
    domain: artifact.domain,
    path: artifact.path
  }));
  const byId = new Map(artifacts.map((artifact) => [artifact.id, artifact]));
  const edges = artifacts.flatMap((artifact) =>
    artifact.depends_on
      .filter((target) => !target.startsWith("external:"))
      .map((target) => ({ source_id: artifact.id, target_id: target }))
  );
  const missing_dependencies = artifacts.flatMap((artifact) =>
    artifact.depends_on
      .filter((target) => !target.startsWith("external:") && !byId.has(target))
      .map((target) => ({
        source_id: artifact.id,
        target_id: target,
        source_path: artifact.path,
        message: `missing dependency: ${target}`
      }))
  );
  const accepted_draft_violations = artifacts
    .filter((artifact) => artifact.status === "accepted")
    .flatMap((artifact) =>
      artifact.depends_on
        .filter((target) => {
          if (target.startsWith("external:")) {
            return false;
          }
          const targetArtifact = byId.get(target);
          return (
            targetArtifact !== undefined && targetArtifact.status !== "accepted"
          );
        })
        .map((target) => {
          const targetArtifact = byId.get(target)!;
          return {
            source_id: artifact.id,
            target_id: target,
            source_path: artifact.path,
            message: `accepted artifact depends on ${targetArtifact.status} artifact: ${target}`
          };
        })
    );
  const cycles = detectCycles(
    edges.filter(
      (edge) => byId.has(edge.source_id) && byId.has(edge.target_id)
    ),
    Array.from(byId.keys())
  );
  return {
    ...fixtureData.graph,
    source_of_truth: "repository",
    nodes,
    edges,
    cycles,
    missing_dependencies,
    accepted_draft_violations
  };
}

function detectCycles(
  edges: Array<{ source_id: string; target_id: string }>,
  artifactIds: string[]
): string[][] {
  const sortedIds = [...artifactIds].sort();
  const adjacency = new Map<string, string[]>(
    sortedIds.map((id) => [id, []])
  );
  for (const edge of edges) {
    adjacency.get(edge.source_id)?.push(edge.target_id);
  }
  const cycles = new Map<string, string[]>();
  const stack: string[] = [];
  const visited = new Set<string>();

  const visit = (artifactId: string): void => {
    const stackIndex = stack.indexOf(artifactId);
    if (stackIndex !== -1) {
      const members = [...stack.slice(stackIndex), artifactId];
      const canonical = canonicalCycle(members);
      cycles.set(canonical.join("->"), canonical);
      return;
    }
    if (visited.has(artifactId)) {
      return;
    }
    stack.push(artifactId);
    const targets = [...(adjacency.get(artifactId) ?? [])].sort();
    for (const target of targets) {
      visit(target);
    }
    stack.pop();
    visited.add(artifactId);
  };

  for (const id of sortedIds) {
    visit(id);
  }
  return Array.from(cycles.entries())
    .sort(([a], [b]) => a.localeCompare(b))
    .map(([, cycle]) => cycle);
}

function canonicalCycle(cycle: string[]): string[] {
  const members = cycle.slice(0, -1);
  if (members.length === 0) {
    return cycle;
  }
  let best = [...members, members[0]];
  let bestKey = best.join("\u0000");
  for (let index = 1; index < members.length; index += 1) {
    const rotated = [
      ...members.slice(index),
      ...members.slice(0, index),
      members[index]
    ];
    const key = rotated.join("\u0000");
    if (key < bestKey) {
      best = rotated;
      bestKey = key;
    }
  }
  return best;
}

function mapContextPacks(
  fixtureData: SiteData,
  issues: IssueSummary[],
  issuesPayload?: Record<string, unknown>
): ContextPacksPayload {
  const rawById = new Map<string, Record<string, unknown>>();
  for (const rawIssue of arrayOfRecords(issuesPayload?.issues)) {
    const id = text(rawIssue.id, "");
    if (id) {
      rawById.set(id, asRecord(rawIssue.context_pack));
    }
  }
  return {
    ...fixtureData.context_packs,
    source_of_truth: "repository",
    context_packs: issues.map((issue) => {
      const rawPack = rawById.get(issue.id) ?? {};
      const exists = Boolean(rawPack.exists);
      return {
        issue_id: issue.id,
        exists,
        source_of_truth: "repository",
        card_count: exists ? numberValue(rawPack.card_count, 0) : 0,
        public_only: Boolean(rawPack.public_only),
        private_context_included: Boolean(rawPack.private_context_included),
        demo_private_context_included: false,
        related_artifacts: issue.related_artifacts
      };
    })
  };
}

function mapReports(
  fixtureData: SiteData,
  statusPayload: Record<string, unknown>,
  gates: GatesPayload
): SiteExportPayload {
  const validation = asRecord(statusPayload.validation);
  const reports: Array<Record<string, unknown>> = [
    {
      kind: "repository_validation",
      source_of_truth: "repository",
      ok: Boolean(validation.ok),
      checked_count: numberValue(validation.checked_count, 0),
      failure_count: arrayValue(validation.failures).length
    }
  ];
  for (const reportPath of gates.report_paths) {
    reports.push({
      kind: "gate_report",
      source_of_truth: "repository",
      path: reportPath,
      verdict: gates.verdict
    });
  }
  return {
    ...fixtureData.reports,
    authority_notice: authorityNotice(statusPayload, fixtureData.reports),
    source_of_truth: "repository",
    demo: false,
    note:
      gates.report_paths.length > 0
        ? `Live repository validation and ${gates.report_paths.length} gate report(s) loaded.`
        : "Live repository validation loaded; no gate report file exists yet.",
    reports
  };
}

function mapAuthorityBoundaries(
  fixtureData: SiteData,
  healthPayload: Record<string, unknown>
): AuthorityBoundariesPayload {
  return {
    ...fixtureData.authority_boundaries,
    authority_notice: authorityNotice(
      healthPayload,
      fixtureData.authority_boundaries
    ),
    source_of_truth: "repository",
    website_is_authority: false,
    export_is_source_of_truth: false,
    frontend_credentials_allowed: false,
    skipped_is_pass: false,
    gate_pass_is_truth: false,
    verifier_pass_is_truth: false,
    ai_output_is_human_review: false,
    accepted_write_performed: false
  };
}

function mapGates(
  fixtureData: SiteData,
  payload: Record<string, unknown>
): GatesPayload {
  const gateReport = asRecord(payload.gate_report);
  const report = asRecord(gateReport.report);
  const reportPath = text(gateReport.path, "");
  const verdict = text(gateReport.verdict, text(report.verdict, "not_run"));
  return {
    ...fixtureData.gates,
    authority_notice: authorityNotice(payload, fixtureData.gates),
    verdict,
    note: reportPath
      ? `Live gate report read from ${reportPath}.`
      : "No live gate report exists under .cosheaf/reports.",
    gates: arrayValue(report.gates),
    blocking_issues: arrayValue(report.blocking_issues),
    nonblocking_issues: arrayValue(report.nonblocking_issues),
    report_paths: reportPath ? [reportPath] : []
  };
}

function authorityNotice(
  payload: LivePayload,
  fallback: { authority_notice: string }
): string {
  return text(payload.authority_notice, fallback.authority_notice);
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function asRecord(value: unknown): Record<string, unknown> {
  return isRecord(value) ? value : {};
}

function arrayOfRecords(value: unknown): Array<Record<string, unknown>> {
  return Array.isArray(value) ? value.filter(isRecord) : [];
}

function arrayValue(value: unknown): unknown[] {
  return Array.isArray(value) ? value : [];
}

function stringArray(value: unknown): string[] {
  return Array.isArray(value)
    ? value.filter((item): item is string => typeof item === "string")
    : [];
}

function text(value: unknown, fallback: string): string {
  return typeof value === "string" && value.trim() ? value.trim() : fallback;
}

function textOrUndefined(value: unknown): string | undefined {
  return typeof value === "string" && value.trim() ? value.trim() : undefined;
}

function numberValue(value: unknown, fallback: number): number {
  return typeof value === "number" && Number.isFinite(value) ? value : fallback;
}

function riskFlags(artifact: Record<string, unknown>): string[] {
  const risk = asRecord(artifact.risk);
  const level = text(risk.level, "");
  return level ? [`risk:${level}`] : [];
}

function sourceLabels(value: unknown): string[] {
  return Array.isArray(value)
    ? value.map((source) => {
        if (typeof source === "string") {
          return source;
        }
        const record = asRecord(source);
        return text(record.title, text(record.url, text(record.doi, "")));
      }).filter((source) => source.length > 0)
    : [];
}
