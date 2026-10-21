import { localized } from "./i18n";

export const ARTIFACT_ACTION_ENDPOINTS = {
  previewCreate: "/api/artifacts/preview-create",
  create: "/api/artifacts/create",
  previewUpdate: (id: string) =>
    `/api/artifacts/${encodeURIComponent(id)}/preview-update`,
  update: (id: string) => `/api/artifacts/${encodeURIComponent(id)}/update`
} as const;

export const PREACCEPTED_ARTIFACT_STATUSES = [
  "raw",
  "draft",
  "locally_tested",
  "adversarially_tested",
  "machine_checked",
  "human_reviewed"
] as const;

export const ARTIFACT_WORKBENCH_LABELS = {
  createArtifact: localized("New artifact", "新建工件"),
  editArtifact: localized("Edit artifact", "编辑工件"),
  preview: localized("Preview changes", "预览变更"),
  confirm: localized("Confirm write", "确认写入"),
  artifactId: localized("Artifact ID", "工件 ID"),
  artifactType: localized("Type", "类型"),
  status: localized("Status", "状态"),
  title: localized("Title", "标题"),
  domain: localized("Domain", "领域"),
  statement: localized("Statement", "陈述"),
  authors: localized("Authors", "作者"),
  tags: localized("Tags", "标签"),
  dependsOn: localized("Dependencies", "依赖"),
  supersedes: localized("Supersedes", "替代"),
  previewDiff: localized("YAML preview", "YAML 预览")
} as const;

export interface ArtifactFormInput {
  artifactId: string;
  artifactType: string;
  title: string;
  domain: string;
  status: string;
  statement: string;
  authors: string;
  tags: string;
  dependsOn: string;
  supersedes: string;
}

export interface ArtifactWritePlan {
  plan_id: string;
  action: string;
  artifact_id: string;
  request_hash: string;
  expected_revision: string | null;
  planned_content_revision?: string | null;
  planned_files?: string[];
}

export interface ArtifactActionPayload {
  artifact_id: string;
  artifact_type: string;
  title: string;
  domain: string[];
  status: string;
  statement: string;
  authors: string[];
  tags: string[];
  depends_on: string[];
  supersedes: string[];
  confirm?: true;
  write_plan?: ArtifactWritePlan;
}

export function parseDelimitedList(value: string): string[] {
  const seen = new Set<string>();
  const result: string[] = [];
  for (const item of value.split(/[,\n;]/)) {
    const normalized = item.trim();
    if (!normalized || seen.has(normalized)) {
      continue;
    }
    seen.add(normalized);
    result.push(normalized);
  }
  return result;
}

function preacceptedStatus(value: string): string {
  return PREACCEPTED_ARTIFACT_STATUSES.includes(
    value as (typeof PREACCEPTED_ARTIFACT_STATUSES)[number]
  )
    ? value
    : "draft";
}

export function buildArtifactActionPayload(
  input: ArtifactFormInput,
  options: { confirm?: true; writePlan?: ArtifactWritePlan } = {}
): ArtifactActionPayload {
  const payload: ArtifactActionPayload = {
    artifact_id: input.artifactId.trim(),
    artifact_type: input.artifactType.trim() || "claim",
    title: input.title.trim(),
    domain: parseDelimitedList(input.domain),
    status: preacceptedStatus(input.status.trim()),
    statement: input.statement.trim(),
    authors: parseDelimitedList(input.authors),
    tags: parseDelimitedList(input.tags),
    depends_on: parseDelimitedList(input.dependsOn),
    supersedes: parseDelimitedList(input.supersedes)
  };
  if (options.confirm) {
    payload.confirm = true;
  }
  if (options.writePlan) {
    payload.write_plan = options.writePlan;
  }
  return payload;
}

export function serializeArtifactActionFields(
  payload: ArtifactActionPayload
): string {
  return JSON.stringify({
    artifact_id: payload.artifact_id,
    artifact_type: payload.artifact_type,
    title: payload.title,
    domain: payload.domain,
    status: payload.status,
    statement: payload.statement,
    authors: payload.authors,
    tags: payload.tags,
    depends_on: payload.depends_on,
    supersedes: payload.supersedes
  });
}

export function isPreviewPayloadCurrent(
  previewed: ArtifactActionPayload | null | undefined,
  current: ArtifactActionPayload
): boolean {
  if (!previewed) {
    return false;
  }
  return (
    serializeArtifactActionFields(previewed) ===
    serializeArtifactActionFields(current)
  );
}

export function buildConfirmedArtifactActionPayload(
  previewed: ArtifactActionPayload,
  writePlan?: ArtifactWritePlan
): ArtifactActionPayload {
  const confirmed: ArtifactActionPayload = {
    ...previewed,
    confirm: true
  };
  if (writePlan) {
    confirmed.write_plan = writePlan;
  }
  return confirmed;
}
