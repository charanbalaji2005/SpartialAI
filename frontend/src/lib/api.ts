export type Quality = "fast" | "balanced" | "best";
export type GeometryStatus = "observed" | "inferred" | "generated";

export interface Project {
  id: string;
  name: string;
  status: string;
  stage: string | null;
  progress: number;
  message: string | null;
  error: string | null;
  image_url: string | null;
  image_width: number | null;
  image_height: number | null;
  original_filename: string | null;
  scene_category: string | null;
  quality: Quality;
  has_scene: boolean;
  created_at: number;
  updated_at: number;
}

export interface StageState { key: string; name: string; state: "done" | "active" | "pending" | "failed" | "cancelled" }
export interface LogLine { t: number; level: "info" | "warn" | "error" | "stage"; message: string }

export interface JobStatus {
  project_id: string;
  status: string;
  stage: string | null;
  stage_name?: string | null;
  progress: number;
  message: string | null;
  error: string | null;
  stages: StageState[];
  logs: LogLine[];
  version: number;
}

export interface SceneNode {
  node: string;
  label: string;
  kind: "object" | "region";
  status: GeometryStatus;
  group: string;
  faces: number;
  vertices: number;
  bounds: number[][];
  center: number[];
  default_visible: boolean;
  confidence?: number;
  mask_method?: string | null;
  mask_score?: number | null;
  method?: string;
  median_depth?: number;
  box?: number[];
  closed?: boolean;            // part of a validated closed shell
  element?: string;            // completeness element this node belongs to
  shape_prior?: string;
  profile?: string;
  thickness?: number;
}

export interface CompletenessElement {
  name: string; label: string | null; kind: "object" | "background"; status: "complete" | "repaired" | "incomplete";
  shape_prior: string | null; profile: string | null; thickness: number | null; method: string | null;
  watertight: boolean; boundary_edges: number; non_manifold_edges: number; winding_consistent: boolean;
  outward_normals: boolean; components: number; textured: boolean; uv_valid: boolean;
  texture_size: number[] | null; repairs: string[];
}
export interface Completeness {
  passed: boolean; objects_total: number; objects_complete: number; background_closed: boolean;
  issues: string[]; elements: CompletenessElement[];
}

export interface Asset { id: string; name: string; file: string; type: string; object?: string }

export interface SceneMeta {
  project_id: string;
  glb_url: string;
  scene: { category: string; group: string; confidence: number; method: string; near: number; far: number };
  camera: { width: number; height: number; hfov_deg: number; vfov_deg: number };
  fov_source: string;
  target: number[];
  lighting: { ambient_intensity: number; key_color: number[]; key_direction: number[] };
  ground: { normal: number[]; offset: number; inlier_ratio: number } | null;
  nodes: SceneNode[];
  assets: Asset[];
  report: Record<string, any> | null;
  notes: string[];
}

export interface GpuInfo {
  torch_installed: boolean;
  cuda_available: boolean;
  name: string | null;
  total_vram_gb: number | null;
  allocated_gb: number | null;
  free_gb: number | null;
  smi_used_gb?: number;
  smi_total_gb?: number;
  utilization_pct: number | null;
  driver: string | null;
  torch_version: string | null;
}

export interface ModelSource { source: "pretrained" | "fine-tuned"; base_model?: string; run_id?: string; metric?: string; value?: number; promoted_at?: number }
export interface ModelStatus { name: string; model_id: string; role: string; license: string; status: string; error: string | null; source?: ModelSource | null }

export interface TrainingRun { run_id: string; status?: string; best_value?: number; best_epoch?: number; model_name?: string; updated_at?: number; monitor?: { metric: string; mode: string }; metrics: Record<string, number> }
export interface TrainingTask { production: (ModelSource & { metrics?: Record<string, number>; model_name?: string }) | null; runs: TrainingRun[] }
export interface TrainingRegistry { use_finetuned: boolean; tasks: Record<"depth" | "segmentation" | "detection", TrainingTask> }

export class ApiError extends Error {
  constructor(message: string, public status: number) { super(message); }
}

async function call<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(path, init);
  } catch {
    throw new ApiError("Can't reach the SpatialAI backend. Start it with start_spatialai.bat.", 0);
  }
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch { /* not JSON */ }
    throw new ApiError(detail, res.status);
  }
  return res.json() as Promise<T>;
}

const json = (body: unknown): RequestInit => ({
  method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
});

export const api = {
  health: () => call<{ status: string; test_mode: boolean }>("/api/health"),
  gpu: () => call<GpuInfo>("/api/system/gpu"),
  models: () => call<{ models: ModelStatus[]; test_mode: boolean }>("/api/models/status"),
  projects: () => call<Project[]>("/api/projects"),
  project: (id: string) => call<Project & { report: any }>(`/api/projects/${id}`),
  deleteProject: (id: string) => call<{ deleted: string }>(`/api/projects/${id}`, { method: "DELETE" }),
  upload: (file: File, projectId?: string) => {
    const form = new FormData();
    form.append("file", file);
    if (projectId) form.append("project_id", projectId);
    return call<Project>("/api/reconstruction/upload", { method: "POST", body: form });
  },
  generate: (id: string, quality: Quality) => call<JobStatus>(`/api/reconstruction/${id}/generate`, json({ quality })),
  status: (id: string) => call<JobStatus>(`/api/reconstruction/${id}/status`),
  cancel: (id: string) => call<{ cancelling: string }>(`/api/reconstruction/${id}/cancel`, { method: "POST" }),
  registry: () => call<TrainingRegistry>("/api/training/registry"),
  scene: (id: string) => call<SceneMeta>(`/api/scenes/${id}`),
  exportUrl: (id: string, kind: "glb" | "gltf" | "metadata") => `/api/export/${id}/${kind}`,
  assetUrl: (assetId: string) => `/api/assets/${encodeURIComponent(assetId)}`,
  progressSocketUrl: (id: string) => {
    const proto = location.protocol === "https:" ? "wss" : "ws";
    return `${proto}://${location.host}/api/reconstruction/${id}/ws`;
  },
};

export const STATUS_LABEL: Record<GeometryStatus, string> = {
  observed: "Seen in photo",
  inferred: "Inferred",
  generated: "AI-generated",
};
