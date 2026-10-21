import artifactsPayload from "../fixtures/site-data/artifacts.json";
import authorityBoundariesPayload from "../fixtures/site-data/authority_boundaries.json";
import contextPacksPayload from "../fixtures/site-data/context_packs.json";
import gatesPayload from "../fixtures/site-data/gates.json";
import graphPayload from "../fixtures/site-data/graph.json";
import issuesPayload from "../fixtures/site-data/issues.json";
import reportsPayload from "../fixtures/site-data/reports.json";
import sitePayload from "../fixtures/site-data/site.json";
import workspacePayload from "../fixtures/site-data/workspace.json";
import { type FetchLike, fetchLiveSiteData } from "./apiClient";
import { type LocalizedText, localized } from "./i18n";
import {
  type RuntimeMetadata,
  type RuntimeModeInput,
  decideRuntimeMode
} from "./runtimeMode";

export const REQUIRED_DATA_FILES = [
  "site.json",
  "workspace.json",
  "artifacts.json",
  "issues.json",
  "graph.json",
  "gates.json",
  "context_packs.json",
  "reports.json",
  "authority_boundaries.json"
] as const;

export const ROUTES = [
  { path: "/", label: localized("Dashboard", "总览") },
  { path: "/artifacts", label: localized("Artifacts", "工件") },
  { path: "/issues", label: localized("Issues", "议题") },
  { path: "/forge/submit", label: localized("Submit PR", "提交 PR") },
  { path: "/forge/pr-status", label: localized("PR status", "PR 状态") },
  { path: "/graph", label: localized("Graph", "图谱") },
  { path: "/gates", label: localized("Checks", "检查") },
  { path: "/authority", label: localized("Rules", "权限规则") }
] as const;

export const AUTHORITY_GUIDE = [
  {
    title: localized("What Cosheaf is", "Cosheaf 是什么"),
    body: localized(
      "Cosheaf is a Git-backed workflow for typed research artifacts, review context, evidence records, and reproducible handoff.",
      "Cosheaf 是一个基于 Git 的研究工作流，用于管理类型化研究工件、审阅上下文、证据记录和可复现交接。"
    )
  },
  {
    title: localized("What Cosheaf is not", "Cosheaf 不是什么"),
    body: localized(
      "It is not a hosted prover, a human reviewer, a source of private-public promotion, or a replacement for repository records.",
      "它不是托管证明器、人工审阅者、私有内容公开化的授权来源，也不能替代仓库记录。"
    )
  },
  {
    title: localized(
      "Why gate pass is not truth",
      "为什么准入检查通过不等于真理"
    ),
    body: localized(
      "A gate result can unblock or block workflow review, but it does not prove a statement or make an artifact accepted.",
      "准入检查结果可以放行或阻断工作流审阅，但不能证明命题，也不能让工件自动成为 accepted。"
    )
  },
  {
    title: localized(
      "Why AI output is review context",
      "为什么 AI 输出只是审阅上下文"
    ),
    body: localized(
      "AI output can propose drafts, summaries, or checks for a human to inspect; it cannot become human review by itself.",
      "AI 输出可以提供草稿、摘要或检查建议，但它本身不是人工审阅。"
    )
  },
  {
    title: localized(
      "Why skipped verifier is not pass",
      "为什么 skipped 不是 pass"
    ),
    body: localized(
      "Skipped, unavailable, and not-run verifier states mean no successful check was recorded.",
      "skipped、unavailable 和 not-run 表示没有成功检查记录，不能当作通过。"
    )
  },
  {
    title: localized(
      "How public and private KB roots work",
      "公开与私有 KB 根如何工作"
    ),
    body: localized(
      "Public KB roots are normally readonly and citable; private KB roots are writable overlays for local research and must not leak into public exports.",
      "公开 KB 根通常只读且可引用；私有 KB 根是本地研究的可写覆盖层，不能泄露到公开导出。"
    )
  },
  {
    title: localized(
      "How a result becomes accepted",
      "结果如何成为 accepted"
    ),
    body: localized(
      "Accepted status comes from repository records, source metadata, validation, gate policy, real human review, and explicit promotion.",
      "accepted 状态来自仓库记录、来源元数据、验证、准入检查策略、真实人工审阅和显式提升。"
    )
  }
] as const;

export interface SiteExportPayload {
  schema_version: 1;
  kind: string;
  authority_notice: string;
  [key: string]: unknown;
}

export interface ArtifactCard {
  id: string;
  title: string;
  type: string;
  status: string;
  root_scope: string;
  summary: string;
  statement?: string;
  domain: string[];
  authors: string[];
  tags: string[];
  depends_on: string[];
  supersedes: string[];
  demo_fixture: boolean;
  verifier_state: string;
  review_state: string;
  risk_flags: string[];
  sources: string[];
  path: string;
}

export interface IssueSummary {
  id: string;
  title: string;
  type: string;
  status: string;
  scope: string;
  severity: string;
  summary: string;
  authors: string[];
  labels: string[];
  related_artifacts: string[];
  related_sources: string[];
  external_links?: string[];
  demo_fixture: boolean;
}

export interface ContextPackSummary {
  issue_id: string;
  card_count: number;
  exists?: boolean;
  source_of_truth?: string;
  public_only: boolean;
  private_context_included: boolean;
  demo_private_context_included: boolean;
  related_artifacts: string[];
}

export interface GraphNode {
  artifact_id: string;
  artifact_type: string;
  title: string;
  status: string;
  domain: string[];
  path: string;
}

export interface GraphEdge {
  source_id: string;
  target_id: string;
}

export type GraphLayoutMode = "empty" | "force";

export const ARTIFACT_PAGE_SIZE = 12;
export const GRAPH_PAGE_SIZE = 24;
export const RELATIONSHIP_INLINE_LIMIT = 4;

export interface RelationshipPreview<T> {
  visible: T[];
  hidden: T[];
  hiddenCount: number;
}

export interface PaginatedItems<T> {
  items: T[];
  page: number;
  pageCount: number;
  total: number;
}

export interface GraphNeighborhood {
  center: GraphNode | undefined;
  dependencies: GraphNode[];
  dependents: GraphNode[];
  nodes: GraphNode[];
  edges: GraphEdge[];
}

export interface WorkbenchDashboardAction {
  label: LocalizedText;
  href?: string;
  meta?: LocalizedText;
}

export interface WorkbenchDashboard {
  activeIssues: IssueSummary[];
  draftReviewArtifacts: ArtifactCard[];
  promotion: {
    ready: ArtifactCard[];
    blocked: ArtifactCard[];
  };
  gateFailureCount: number;
  recentActions: WorkbenchDashboardAction[];
  recentPrLinks: WorkbenchDashboardAction[];
  nextActions: WorkbenchDashboardAction[];
}

export interface SitePayload extends SiteExportPayload {
  demo: boolean;
  public_only: boolean;
  files: string[];
  source_of_truth: string;
}

export interface WorkspacePayload extends SiteExportPayload {
  workspace_name: string;
  mode: string;
  public_only: boolean;
  kb_roots: Array<{
    name: string;
    path: string;
    readonly: boolean;
    priority: number;
    demo_private_fixtures_allowed: boolean;
  }>;
  policy: Record<string, boolean>;
}

export interface ArtifactsPayload extends SiteExportPayload {
  artifacts: ArtifactCard[];
  count: number;
  public_only: boolean;
  demo_private_fixtures_allowed: boolean;
}

export interface IssuesPayload extends SiteExportPayload {
  issues: IssueSummary[];
  count: number;
}

export interface GraphPayload extends SiteExportPayload {
  nodes: GraphNode[];
  edges: GraphEdge[];
  cycles: unknown[];
  missing_dependencies: unknown[];
  accepted_draft_violations: unknown[];
}

export interface GatesPayload extends SiteExportPayload {
  verdict: string;
  note: string;
  gates: unknown[];
  blocking_issues: unknown[];
  nonblocking_issues: unknown[];
  report_paths: string[];
}

export interface ContextPacksPayload extends SiteExportPayload {
  context_packs: ContextPackSummary[];
}

export interface AuthorityBoundariesPayload extends SiteExportPayload {
  website_is_authority: boolean;
  export_is_source_of_truth: boolean;
  frontend_credentials_allowed: boolean;
  skipped_is_pass: boolean;
  gate_pass_is_truth: boolean;
  verifier_pass_is_truth: boolean;
  ai_output_is_human_review: boolean;
  accepted_write_performed: boolean;
  notices: string[];
}

export interface SiteData {
  site: SitePayload;
  workspace: WorkspacePayload;
  artifacts: ArtifactsPayload;
  issues: IssuesPayload;
  graph: GraphPayload;
  gates: GatesPayload;
  context_packs: ContextPacksPayload;
  reports: SiteExportPayload;
  authority_boundaries: AuthorityBoundariesPayload;
  runtime: RuntimeMetadata;
}

export interface LoadSiteDataOptions {
  env?: RuntimeModeInput;
  fetcher?: FetchLike;
  timeoutMs?: number;
}

const fixtureData: SiteData = {
  site: sitePayload as SitePayload,
  workspace: workspacePayload as WorkspacePayload,
  artifacts: artifactsPayload as ArtifactsPayload,
  issues: issuesPayload as IssuesPayload,
  graph: graphPayload as GraphPayload,
  gates: gatesPayload as GatesPayload,
  context_packs: contextPacksPayload as ContextPacksPayload,
  reports: reportsPayload as SiteExportPayload,
  authority_boundaries: authorityBoundariesPayload as AuthorityBoundariesPayload,
  runtime: decideRuntimeMode({ mode: "static" })
};

export async function loadSiteData(
  options: LoadSiteDataOptions = {}
): Promise<SiteData> {
  const runtime = decideRuntimeMode(options.env ?? runtimeEnvFromImportMeta());
  if (runtime.mode === "static-demo") {
    return withRuntime(fixtureData, runtime);
  }
  try {
    return await fetchLiveSiteData({
      apiBase: runtime.api_base,
      fetcher: options.fetcher,
      fixtureData,
      runtime,
      timeoutMs: options.timeoutMs
    });
  } catch (error) {
    return withRuntime(
      fixtureData,
      decideRuntimeMode({
        mode: "static",
        apiBase: runtime.api_base,
        fallbackReason: errorMessage(error)
      })
    );
  }
}

export function artifactById(
  id: string,
  source: SiteData = fixtureData
): ArtifactCard | undefined {
  return source.artifacts.artifacts.find((artifact) => artifact.id === id);
}

export function issueById(
  id: string,
  source: SiteData = fixtureData
): IssueSummary | undefined {
  return source.issues.issues.find((issue) => issue.id === id);
}

export function contextPackForIssue(
  id: string,
  source: SiteData = fixtureData
): ContextPackSummary | undefined {
  return source.context_packs.context_packs.find((pack) => pack.issue_id === id);
}

export interface ArtifactFilter {
  status?: string;
  type?: string;
  domain?: string;
}

export function filterArtifacts(
  artifacts: ArtifactCard[],
  filter: ArtifactFilter
): ArtifactCard[] {
  return artifacts.filter((artifact) => {
    const statusMatch = !filter.status || artifact.status === filter.status;
    const typeMatch = !filter.type || artifact.type === filter.type;
    const domainMatch = !filter.domain || artifact.domain.includes(filter.domain);
    return statusMatch && typeMatch && domainMatch;
  });
}

export function uniqueSorted(values: string[]): string[] {
  return [...new Set(values)].sort((left, right) => left.localeCompare(right));
}

export function artifactHref(id: string): string {
  return `/artifacts/${encodeURIComponent(id)}/`;
}

export function issueHref(id: string): string {
  return `/issues/${encodeURIComponent(id)}/`;
}

export function contextHref(issueId: string): string {
  return `/context/${encodeURIComponent(issueId)}/`;
}

export function statusExplanation(status: string): LocalizedText {
  if (status === "accepted") {
    return localized(
      "Accepted status must come from repository records and promotion policy, not the website.",
      "accepted 状态必须来自仓库记录和提升策略，不由网页决定。"
    );
  }
  if (status === "draft") {
    return localized(
      "Draft means material waiting for review; it is not accepted knowledge.",
      "draft 是待审材料，不是 accepted 知识。"
    );
  }
  if (status === "not_run") {
    return localized(
      "Not checked: no verifier or gate pass is implied.",
      "尚未检查，不暗示验证器或准入检查通过。"
    );
  }
  if (status === "skipped") {
    return localized("Skipped is not pass.", "skipped 不是 pass。");
  }
  if (status === "requested") {
    return localized(
      "Review requested is not human-reviewed or accepted.",
      "requested 不等于已人工审阅或 accepted。"
    );
  }
  return localized(
    "Website badges are display context only.",
    "网页徽标只是展示上下文。"
  );
}

export function dependenciesFor(id: string, edges: GraphEdge[]): string[] {
  return uniqueSorted(
    edges.filter((edge) => edge.source_id === id).map((edge) => edge.target_id)
  );
}

export function dependentsFor(id: string, edges: GraphEdge[]): string[] {
  return uniqueSorted(
    edges.filter((edge) => edge.target_id === id).map((edge) => edge.source_id)
  );
}

export function graphLayoutMode(nodeCount: number): GraphLayoutMode {
  if (nodeCount === 0) {
    return "empty";
  }
  return "force";
}

export function shouldDrawGraphConnector(
  nodeCount: number,
  edgeCount: number
): boolean {
  return graphLayoutMode(nodeCount) === "force" && nodeCount <= 2 && edgeCount > 0
    ? false
    : false;
}

export function relationshipPreview<T>(
  items: T[],
  limit = RELATIONSHIP_INLINE_LIMIT
): RelationshipPreview<T> {
  const normalizedLimit = Math.max(1, limit);
  const visible = items.slice(0, normalizedLimit);
  const hidden = items.slice(normalizedLimit);
  return {
    visible,
    hidden,
    hiddenCount: hidden.length
  };
}

export function paginateItems<T>(
  items: T[],
  page = 1,
  pageSize = GRAPH_PAGE_SIZE
): PaginatedItems<T> {
  const normalizedPageSize = Math.max(1, pageSize);
  const pageCount = Math.max(1, Math.ceil(items.length / normalizedPageSize));
  const normalizedPage = Math.min(Math.max(1, page), pageCount);
  const start = (normalizedPage - 1) * normalizedPageSize;
  return {
    items: items.slice(start, start + normalizedPageSize),
    page: normalizedPage,
    pageCount,
    total: items.length
  };
}

export function graphNeighborhoodFor(
  id: string,
  graph: GraphPayload
): GraphNeighborhood {
  const center = graph.nodes.find((node) => node.artifact_id === id);
  const nodeById = new Map(graph.nodes.map((node) => [node.artifact_id, node]));
  const dependencies = dependenciesFor(id, graph.edges)
    .map((dependencyId) => nodeById.get(dependencyId))
    .filter((node): node is GraphNode => Boolean(node));
  const dependents = dependentsFor(id, graph.edges)
    .map((dependentId) => nodeById.get(dependentId))
    .filter((node): node is GraphNode => Boolean(node));
  const selectedIds = new Set<string>([
    ...(center ? [center.artifact_id] : []),
    ...dependencies.map((node) => node.artifact_id),
    ...dependents.map((node) => node.artifact_id)
  ]);
  const nodes = [
    ...(center ? [center] : []),
    ...dependencies,
    ...dependents
  ].filter((node, index, all) =>
    all.findIndex((candidate) => candidate.artifact_id === node.artifact_id) ===
    index
  );
  const edges = graph.edges.filter(
    (edge) => selectedIds.has(edge.source_id) && selectedIds.has(edge.target_id)
  );

  return {
    center,
    dependencies,
    dependents,
    nodes,
    edges
  };
}

export function buildWorkbenchDashboard(data: SiteData): WorkbenchDashboard {
  const activeIssues = data.issues.issues.filter(
    (issue) => !["closed", "done", "resolved"].includes(issue.status)
  );
  const draftReviewArtifacts = data.artifacts.artifacts.filter(
    (artifact) =>
      artifact.status === "draft" &&
      !["accepted", "human_reviewed"].includes(artifact.review_state)
  );
  const promotionCandidates = data.artifacts.artifacts.filter(
    (artifact) => artifact.status !== "accepted"
  );
  const promotionReady = promotionCandidates.filter((artifact) =>
    isPromotionReadyArtifact(artifact, data)
  );
  const promotionReadyIds = new Set(promotionReady.map((artifact) => artifact.id));
  const promotionBlocked = promotionCandidates.filter(
    (artifact) => !promotionReadyIds.has(artifact.id)
  );
  const gateFailureCount =
    (["fail", "error"].includes(data.gates.verdict) ? 1 : 0) +
    data.gates.blocking_issues.length;
  const recentPrLinks = data.issues.issues
    .flatMap((issue) =>
      (issue.external_links ?? [])
        .filter((link) => /github\.com\/.+\/pull\//i.test(link))
        .map((link) => ({
          label: localized(issue.title, issue.title),
          href: link,
          meta: localized("GitHub PR", "GitHub PR")
        }))
    )
    .slice(0, 3);

  return {
    activeIssues,
    draftReviewArtifacts,
    promotion: {
      ready: promotionReady,
      blocked: promotionBlocked
    },
    gateFailureCount,
    recentActions: recentReportActions(data.reports),
    recentPrLinks,
    nextActions: nextWorkbenchActions(
      activeIssues,
      draftReviewArtifacts,
      promotionReady,
      data.gates
    )
  };
}

function isPromotionReadyArtifact(
  artifact: ArtifactCard,
  data: SiteData
): boolean {
  const hasHumanReview = ["accepted", "human_reviewed"].includes(
    artifact.review_state
  );
  const verifierPassed = ["pass", "passed", "success"].includes(
    artifact.verifier_state
  );
  return (
    hasHumanReview &&
    verifierPassed &&
    artifact.sources.length > 0 &&
    data.gates.verdict === "pass" &&
    data.gates.blocking_issues.length === 0
  );
}

function recentReportActions(
  reports: SiteExportPayload
): WorkbenchDashboardAction[] {
  const rows = reports.reports;
  if (!Array.isArray(rows)) {
    return [];
  }
  return rows
    .map((row) => {
      if (typeof row === "string") {
        return {
          label: localized(row, row),
          meta: localized("Report", "报告")
        };
      }
      if (row && typeof row === "object" && "path" in row) {
        const path = String((row as { path: unknown }).path);
        return {
          label: localized(path, path),
          meta: localized("Report", "报告")
        };
      }
      return undefined;
    })
    .filter((row): row is WorkbenchDashboardAction => Boolean(row))
    .slice(0, 3);
}

function nextWorkbenchActions(
  activeIssues: IssueSummary[],
  draftArtifacts: ArtifactCard[],
  readyArtifacts: ArtifactCard[],
  gates: GatesPayload
): WorkbenchDashboardAction[] {
  const actions: WorkbenchDashboardAction[] = [];
  const issue = activeIssues[0];
  const draft = draftArtifacts[0];
  const ready = readyArtifacts[0];
  if (issue) {
    actions.push({
      label: localized("Open active issue", "打开当前议题"),
      href: issueHref(issue.id),
      meta: localized(issue.id, issue.id)
    });
    actions.push({
      label: localized("Prepare related material", "准备相关资料"),
      href: contextHref(issue.id),
      meta: localized(issue.id, issue.id)
    });
  }
  if (draft) {
    actions.push({
      label: localized("Prepare review summary", "准备审阅摘要"),
      href: `/artifacts/${encodeURIComponent(draft.id)}/review-packet/`,
      meta: localized(draft.id, draft.id)
    });
  }
  if (ready) {
    actions.push({
      label: localized("Open promotion check", "打开晋升检查"),
      href: `/artifacts/${encodeURIComponent(ready.id)}/promotion/`,
      meta: localized(ready.id, ready.id)
    });
  }
  if (gates.verdict !== "pass" || gates.blocking_issues.length > 0) {
    actions.push({
      label: localized("Run validation and gate checks", "运行验证和准入检查"),
      href: "/gates/",
      meta: localized("Preview before confirmed runs", "确认运行前先预览")
    });
  }
  return actions.slice(0, 5);
}

function withRuntime(source: SiteData, runtime: RuntimeMetadata): SiteData {
  return { ...source, runtime };
}

function runtimeEnvFromImportMeta(): RuntimeModeInput {
  const env = import.meta.env;
  const mode =
    env.PUBLIC_COSHEAF_RUNTIME_MODE ??
    env.COSHEAF_RUNTIME_MODE ??
    "auto";
  const apiBase =
    env.PUBLIC_COSHEAF_API_BASE ??
    env.COSHEAF_API_BASE ??
    undefined;
  return {
    mode: String(mode),
    apiBase: apiBase ? String(apiBase) : undefined,
    dev: Boolean(env.DEV) && env.MODE !== "test"
  };
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

export function gateVerdictExplanation(gates: GatesPayload): LocalizedText {
  if (gates.verdict === "not_run") {
    return localized(
      "Gate was not run in this static export; no pass is implied.",
      "此静态导出未运行准入检查，不暗示通过。"
    );
  }
  if (gates.verdict === "pass") {
    return localized(
      "Gate pass may unblock review workflow, but it does not accept artifacts or prove truth.",
      "准入检查通过可以放行审阅流程，但不会自动接受工件或证明真理。"
    );
  }
  if (gates.verdict === "fail" || gates.verdict === "error") {
    return localized(
      "Gate result blocks review until the reported issues are resolved.",
      "准入检查结果会阻断审阅，直到报告的问题被解决。"
    );
  }
  return localized(
    "Gate output is workflow context only, not accepted authority.",
    "准入检查输出只是工作流上下文，不是 accepted 权威。"
  );
}

export function statusTone(status: string): "neutral" | "warning" | "good" {
  if (status === "accepted") {
    return "good";
  }
  if (status === "draft" || status === "not_run" || status === "requested") {
    return "warning";
  }
  return "neutral";
}
