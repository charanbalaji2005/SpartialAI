import { useEffect, useState, type ReactNode } from "react";
import { X } from "lucide-react";
import { api, type GpuInfo, type ModelStatus, type TrainingRegistry } from "../lib/api";

function Dialog({ title, onClose, children }: { title: string; onClose: () => void; children: ReactNode }) {
  useEffect(() => {
    const esc = (e: KeyboardEvent) => { if (e.key === "Escape") onClose(); };
    window.addEventListener("keydown", esc);
    return () => window.removeEventListener("keydown", esc);
  }, [onClose]);
  return (
    <div className="backdrop" onMouseDown={(e) => { if (e.target === e.currentTarget) onClose(); }}>
      <div className="dialog" role="dialog" aria-modal="true" aria-label={title}>
        <div className="dialog-head">{title}<span className="grow" /><button className="eye" onClick={onClose} title="Close"><X size={15} /></button></div>
        <div className="dialog-body">{children}</div>
      </div>
    </div>
  );
}

const TASKS = [
  { key: "depth", title: "Depth", metric: "AbsRel (lower is better)", prepare: "python -m training.datasets.prepare_depth --source nyu_kaggle --input training/data/raw/nyu_depth/nyu_data", train: "python -m training.trainers.train_depth --config depth.yaml" },
  { key: "segmentation", title: "Scene regions (segmentation)", metric: "mIoU (higher is better)", prepare: "python -m training.datasets.prepare_segmentation --input training/data/raw/ade20k/ADEChallengeData2016", train: "python -m training.trainers.train_segmentation --config segmentation.yaml" },
  { key: "detection", title: "Object detection (YOLO)", metric: "mAP50-95 (higher is better)", prepare: "python -m training.datasets.prepare_detection --coco training/data/raw/coco --preset spatialai", train: "python -m training.trainers.train_detection --config detection.yaml" },
] as const;

const fmt = (v: unknown) => typeof v === "number" ? (Math.abs(v) < 10 ? v.toFixed(4) : v.toFixed(1)) : String(v ?? "");

export function ModelsDialog({ onClose }: { onClose: () => void }) {
  const [reg, setReg] = useState<TrainingRegistry | null>(null);
  const [models, setModels] = useState<ModelStatus[]>([]);
  const [gpu, setGpu] = useState<GpuInfo | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    api.registry().then(setReg).catch((e) => setError(e.message));
    api.models().then((m) => setModels(m.models)).catch(() => {});
    api.gpu().then(setGpu).catch(() => {});
  }, []);
  return (
    <Dialog title="Models and training" onClose={onClose}>
      {error && <p className="error-box">{error}</p>}
      <div className="task-card">
        <h3>Runtime models{gpu ? ` on ${gpu.cuda_available ? gpu.name : "CPU"}` : ""}</h3>
        <table className="grid">
          <thead><tr><th>Role</th><th>Weights</th><th>Source</th><th>Status</th></tr></thead>
          <tbody>{models.map((m) => (
            <tr key={m.name}><td>{m.role}</td><td title={m.model_id}>{m.source?.source === "fine-tuned" ? `run ${m.source.run_id}` : m.model_id}</td>
              <td>{m.source?.source === "fine-tuned" ? <span style={{ color: "var(--generated)" }}>Fine-tuned{m.source.value != null ? ` (${m.source.metric} ${fmt(m.source.value)})` : ""}</span> : "Pretrained"}</td>
              <td title={m.error ?? ""}>{m.error ? "Unavailable" : m.status.replace("_", " ")}</td></tr>
          ))}</tbody>
        </table>
      </div>
      {TASKS.map((t) => {
        const task = reg?.tasks[t.key];
        return (
          <div key={t.key} className="task-card">
            <h3>{t.title}<span className="grow" /><span className="muted" style={{ fontWeight: 400 }}>{t.metric}</span></h3>
            {task?.production ? (
              <dl className="kv num">
                <dt>In production</dt><dd>run {task.production.run_id} ({task.production.model_name})</dd>
                <dt>{task.production.metric}</dt><dd>{fmt(task.production.value)}</dd>
                <dt>Promoted</dt><dd>{task.production.promoted_at ? new Date(task.production.promoted_at * 1000).toLocaleString() : ""}</dd>
              </dl>
            ) : <p className="empty-note" style={{ padding: "8px 10px" }}>Using pretrained weights. To fine-tune, prepare data then train:</p>}
            {task && task.runs.length > 0 ? (
              <table className="grid">
                <thead><tr><th>Run</th><th>Status</th><th>Best</th><th>Epoch</th><th>Updated</th></tr></thead>
                <tbody>{task.runs.map((r) => (
                  <tr key={r.run_id}><td>{r.run_id}</td><td>{r.status}</td><td className="num">{fmt(r.best_value)}</td>
                    <td className="num">{r.best_epoch != null ? r.best_epoch + 1 : ""}</td>
                    <td>{r.updated_at ? new Date(r.updated_at * 1000).toLocaleString() : ""}</td></tr>
                ))}</tbody>
              </table>
            ) : (
              <div style={{ padding: "0 10px 10px", display: "grid", gap: 6 }}>
                <code className="cmdline">{t.prepare}</code><code className="cmdline">{t.train}</code>
                <code className="cmdline">python -m training.promote {t.key}</code>
              </div>
            )}
          </div>
        );
      })}
      <p className="muted">Restart the backend after promoting a model so it loads the new weights. See TRAINING_GUIDE.md.</p>
    </Dialog>
  );
}

const KEYS: [string, string][] = [
  ["Q / W / E / R", "Select, move, rotate, scale tool"], ["F", "Focus selection"], ["V", "Back to the photo's viewpoint"],
  ["H", "Hide or show selection"], ["G", "Toggle grid"], ["O", "Orthographic camera"], ["T", "Seen-vs-inferred view"],
  ["1 / 2 / 3 / 4", "Photo, Lit, Clay, Seen-vs-inferred"], ["Esc", "Deselect"], ["Ctrl+Space", "Content Drawer"],
  ["` (backtick)", "Focus the console"], ["Ctrl+G", "Generate scene"],
];

export function ShortcutsDialog({ onClose }: { onClose: () => void }) {
  return (
    <Dialog title="Keyboard shortcuts" onClose={onClose}>
      <table className="grid"><tbody>{KEYS.map(([k, d]) => <tr key={k}><td style={{ width: 160 }}><code>{k}</code></td><td>{d}</td></tr>)}</tbody></table>
      <p className="muted">Mouse: left-drag orbit, right-drag pan, wheel zoom. Touch: one finger orbit, two fingers pan and pinch zoom.</p>
    </Dialog>
  );
}
