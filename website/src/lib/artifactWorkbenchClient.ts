import {
  ARTIFACT_ACTION_ENDPOINTS,
  type ArtifactActionPayload,
  type ArtifactWritePlan,
  buildArtifactActionPayload,
  buildConfirmedArtifactActionPayload,
  isPreviewPayloadCurrent
} from "./artifactWorkbench";

type ArtifactFormAction = "create" | "update";

function formText(form: HTMLFormElement, name: string): string {
  const value = new FormData(form).get(name);
  return typeof value === "string" ? value : "";
}

function artifactPayload(form: HTMLFormElement, confirm = false) {
  return buildArtifactActionPayload(
    {
      artifactId: formText(form, "artifactId"),
      artifactType: formText(form, "artifactType"),
      title: formText(form, "title"),
      domain: formText(form, "domain"),
      status: formText(form, "status"),
      statement: formText(form, "statement"),
      authors: formText(form, "authors"),
      tags: formText(form, "tags"),
      dependsOn: formText(form, "dependsOn"),
      supersedes: formText(form, "supersedes")
    },
    confirm ? { confirm: true } : {}
  );
}

function endpointFor(
  action: ArtifactFormAction,
  artifactId: string,
  confirm: boolean
): string {
  if (action === "create") {
    return confirm
      ? ARTIFACT_ACTION_ENDPOINTS.create
      : ARTIFACT_ACTION_ENDPOINTS.previewCreate;
  }
  return confirm
    ? ARTIFACT_ACTION_ENDPOINTS.update(artifactId)
    : ARTIFACT_ACTION_ENDPOINTS.previewUpdate(artifactId);
}

function setOutput(target: Element | null, payload: unknown): void {
  if (target) {
    target.textContent = JSON.stringify(payload, null, 2);
  }
}

async function postJson(
  apiBase: string,
  endpoint: string,
  payload: unknown
): Promise<Record<string, unknown>> {
  const response = await fetch(new URL(endpoint, apiBase), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });
  const data = (await response.json()) as Record<string, unknown>;
  if (!response.ok) {
    throw new Error(String(data.message ?? data.code ?? response.status));
  }
  return data;
}

for (const form of document.querySelectorAll<HTMLFormElement>(
  "[data-artifact-workbench-form]"
)) {
  const action = form.dataset.action as ArtifactFormAction;
  const apiBase = form.dataset.apiBase ?? "http://127.0.0.1:8765";
  const output = document.querySelector(form.dataset.output ?? "");
  const confirmButton = form.querySelector<HTMLButtonElement>("[data-confirm]");
  const previewButton = form.querySelector<HTMLButtonElement>("[data-preview]");
  const redirect = form.dataset.redirect;
  if (form.dataset.mode !== "live-local") {
    continue;
  }

  let previewedState: {
    payload: ArtifactActionPayload;
    writePlan?: ArtifactWritePlan;
  } | null = null;

  const invalidatePreview = () => {
    previewedState = null;
    confirmButton?.setAttribute("disabled", "true");
  };

  form.addEventListener("input", invalidatePreview);
  form.addEventListener("change", invalidatePreview);

  previewButton?.addEventListener("click", async () => {
    previewButton.disabled = true;
    invalidatePreview();
    try {
      const payload = artifactPayload(form, false);
      const preview = await postJson(
        apiBase,
        endpointFor(action, payload.artifact_id, false),
        payload
      );
      const rawPlan = preview.write_plan;
      const writePlan =
        typeof rawPlan === "object" && rawPlan !== null
          ? (rawPlan as ArtifactWritePlan)
          : undefined;
      previewedState = { payload, writePlan };
      setOutput(output, preview);
      confirmButton?.removeAttribute("disabled");
    } catch (error) {
      setOutput(output, { error: error instanceof Error ? error.message : error });
    } finally {
      previewButton.disabled = false;
    }
  });

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    if (confirmButton?.disabled) {
      return;
    }
    const currentPayload = artifactPayload(form, false);
    if (
      !previewedState ||
      !isPreviewPayloadCurrent(previewedState.payload, currentPayload)
    ) {
      invalidatePreview();
      setOutput(output, {
        error:
          "Preview is out of date; preview changes again before confirming."
      });
      return;
    }
    confirmButton?.setAttribute("disabled", "true");
    const payload = buildConfirmedArtifactActionPayload(
      previewedState.payload,
      previewedState.writePlan
    );
    try {
      const result = await postJson(
        apiBase,
        endpointFor(action, payload.artifact_id, true),
        payload
      );
      previewedState = null;
      setOutput(output, result);
      globalThis.location.href =
        redirect ?? `/artifacts/${encodeURIComponent(payload.artifact_id)}/`;
    } catch (error) {
      setOutput(output, { error: error instanceof Error ? error.message : error });
      confirmButton?.removeAttribute("disabled");
    }
  });
}
