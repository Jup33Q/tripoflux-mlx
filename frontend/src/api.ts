// Typed client for the TripoFlux FastAPI backend (see tripoflux/server.py).

export interface GenerateRequest {
  prompt: string;
  negative_prompt: string | null;
  seed: number | null;
  width: number;
  height: number;
  num_gaussians: number;
  splat_steps: number;
  flux_quantize: number | null;
}

export interface GenerateResponse {
  job_id: string;
}

export type StageName = "flux" | "birefnet" | "triposplat";

export type JobStatus = "queued" | "running" | "completed" | "failed";

// SSE payloads emitted by GET /api/stream/{job_id}.
export interface StreamEvent {
  status?: JobStatus;
  stage?: string;
  stage_progress?: number;
  progress?: number;
  log?: string;
  image_ready?: "flux" | "rgba";
  // Intermediate splat snapshot step during the triposplat stage; the bytes
  // are served at /api/result/{job_id}/splat_preview/{step}.
  splat_preview?: number;
  error?: string;
}

export interface StreamHandlers {
  onEvent: (event: StreamEvent) => void;
  onError?: () => void;
}

export async function generate(req: GenerateRequest): Promise<string> {
  const res = await fetch("/api/generate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
  if (!res.ok) throw new Error(`generate failed: ${res.status}`);
  const data = (await res.json()) as GenerateResponse;
  return data.job_id;
}

// Wraps the SSE stream; caller owns closing via the returned EventSource.
export function stream(jobId: string, handlers: StreamHandlers): EventSource {
  const source = new EventSource(`/api/stream/${jobId}`);
  source.onmessage = (msg) => {
    handlers.onEvent(JSON.parse(msg.data) as StreamEvent);
  };
  source.onerror = () => {
    handlers.onError?.();
  };
  return source;
}

export type ResultKind = "image" | "rgba" | "splat" | "ply" | "spz";

export function resultUrl(jobId: string, kind: ResultKind): string {
  return `/api/result/${jobId}/${kind}`;
}

export function stageUrl(jobId: string, stage: "flux" | "rgba"): string {
  return `/api/result/${jobId}/stage/${stage}`;
}

export function splatPreviewUrl(jobId: string, step: number): string {
  return `/api/result/${jobId}/splat_preview/${step}`;
}
