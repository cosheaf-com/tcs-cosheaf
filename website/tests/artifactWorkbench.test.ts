import { describe, expect, it } from "vitest";

import {
  ARTIFACT_ACTION_ENDPOINTS,
  ARTIFACT_WORKBENCH_LABELS,
  PREACCEPTED_ARTIFACT_STATUSES,
  buildArtifactActionPayload,
  buildConfirmedArtifactActionPayload,
  isPreviewPayloadCurrent,
  parseDelimitedList
} from "../src/lib/artifactWorkbench";

describe("artifact workbench helpers", () => {
  const isLocalizedText = (
    value: unknown
  ): value is { en: string; zh: string } =>
    typeof value === "object" &&
    value !== null &&
    "en" in value &&
    "zh" in value;

  it("uses the exact B2.5.1 artifact action endpoints", () => {
    expect(ARTIFACT_ACTION_ENDPOINTS.previewCreate).toBe(
      "/api/artifacts/preview-create"
    );
    expect(ARTIFACT_ACTION_ENDPOINTS.create).toBe("/api/artifacts/create");
    expect(ARTIFACT_ACTION_ENDPOINTS.previewUpdate("claim.fixture.web")).toBe(
      "/api/artifacts/claim.fixture.web/preview-update"
    );
    expect(ARTIFACT_ACTION_ENDPOINTS.update("claim.fixture.web")).toBe(
      "/api/artifacts/claim.fixture.web/update"
    );
  });

  it("keeps labels switchable and excludes accepted status", () => {
    for (const value of Object.values(ARTIFACT_WORKBENCH_LABELS)) {
      expect(isLocalizedText(value)).toBe(true);
      expect(value.en).not.toContain(" / ");
      expect(value.zh).not.toContain(" / ");
    }
    expect(PREACCEPTED_ARTIFACT_STATUSES).toContain("draft");
    expect(PREACCEPTED_ARTIFACT_STATUSES).toContain("human_reviewed");
    expect(PREACCEPTED_ARTIFACT_STATUSES).not.toContain("accepted");
    expect(PREACCEPTED_ARTIFACT_STATUSES).not.toContain("refuted");
  });

  it("builds stable artifact payloads without authority or token fields", () => {
    expect(parseDelimitedList("alpha, beta\nalpha; gamma")).toEqual([
      "alpha",
      "beta",
      "gamma"
    ]);

    const payload = buildArtifactActionPayload({
      artifactId: " claim.fixture.web ",
      artifactType: "claim",
      title: " Web claim ",
      domain: "graph theory, testing",
      status: "accepted",
      statement: " Triangle graph $K_3$ is a graph. ",
      authors: "human, reviewer",
      tags: "graph",
      dependsOn: "definition.graph",
      supersedes: ""
    });

    expect(payload).toEqual({
      artifact_id: "claim.fixture.web",
      artifact_type: "claim",
      title: "Web claim",
      domain: ["graph theory", "testing"],
      status: "draft",
      statement: "Triangle graph $K_3$ is a graph.",
      authors: ["human", "reviewer"],
      tags: ["graph"],
      depends_on: ["definition.graph"],
      supersedes: []
    });
    expect(JSON.stringify(payload)).not.toMatch(/token|password|secret/i);
  });

  it("adds confirm only for explicit confirm submissions", () => {
    const payload = buildArtifactActionPayload(
      {
        artifactId: "claim.fixture.web",
        artifactType: "claim",
        title: "Web claim",
        domain: "testing",
        status: "draft",
        statement: "Draft only.",
        authors: "",
        tags: "",
        dependsOn: "",
        supersedes: ""
      },
      { confirm: true }
    );

    expect(payload.confirm).toBe(true);
  });

  it("binds preview payload and WritePlan to confirm and detects form mutations", () => {
    const previewed = buildArtifactActionPayload({
      artifactId: "claim.fixture.web",
      artifactType: "claim",
      title: "Web claim",
      domain: "testing",
      status: "draft",
      statement: "Previewed statement.",
      authors: "tester",
      tags: "web",
      dependsOn: "",
      supersedes: ""
    });
    const unchanged = buildArtifactActionPayload({
      artifactId: "claim.fixture.web",
      artifactType: "claim",
      title: "Web claim",
      domain: "testing",
      status: "draft",
      statement: "Previewed statement.",
      authors: "tester",
      tags: "web",
      dependsOn: "",
      supersedes: ""
    });
    const mutated = buildArtifactActionPayload({
      artifactId: "claim.fixture.web",
      artifactType: "claim",
      title: "Web claim",
      domain: "testing",
      status: "draft",
      statement: "Edited after preview.",
      authors: "tester",
      tags: "web",
      dependsOn: "",
      supersedes: ""
    });

    expect(isPreviewPayloadCurrent(previewed, unchanged)).toBe(true);
    expect(isPreviewPayloadCurrent(previewed, mutated)).toBe(false);

    const writePlan = {
      plan_id: "write-plan.artifact_update.1234",
      action: "artifact_update",
      artifact_id: "claim.fixture.web",
      request_hash: "sha256:abcd",
      expected_revision: "sha256:0123"
    };
    const confirmed = buildConfirmedArtifactActionPayload(previewed, writePlan);
    expect(confirmed.confirm).toBe(true);
    expect(confirmed.write_plan).toEqual(writePlan);
    expect(confirmed.statement).toBe("Previewed statement.");
  });
});
