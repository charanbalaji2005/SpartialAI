import { useEffect, useState } from "react";
import { Clock, FolderOpen, ImagePlus, Trash2 } from "lucide-react";
import { MenuBar } from "./Menu";
import { api, type Project } from "../lib/api";

const STATUS: Record<string, string> = {
  created: "Empty", uploaded: "Ready to build", queued: "Queued", running: "Building",
  completed: "Scene built", failed: "Failed", cancelled: "Cancelled",
};

export default function ProjectBrowser({ onOpen, onShowModels }: { onOpen: (id: string) => void; onShowModels: () => void }) {
  const [projects, setProjects] = useState<Project[] | null>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [over, setOver] = useState(false);

  const refresh = () => api.projects().then(setProjects).catch((e) => setError(e.message));
  useEffect(() => { refresh(); }, []);

  const importPhoto = async (file: File) => {
    setBusy(true); setError(null);
    try { onOpen((await api.upload(file)).id); } catch (e: any) { setError(e.message); } finally { setBusy(false); }
  };
  const remove = async () => {
    const p = projects?.find((x) => x.id === selected);
    if (!p || !confirm(`Delete “${p.name}” and all its generated files?`)) return;
    try { await api.deleteProject(p.id); setSelected(null); refresh(); } catch (e: any) { setError(e.message); }
  };
  const sel = projects?.find((p) => p.id === selected);

  return (
    <div className="app browser">
      <MenuBar title="SpatialAI Project Browser" menus={[
        { label: "File", items: [{ label: "Import photo…", onSelect: () => document.getElementById("pb-file")?.click() },
          { label: "Open selected scene", onSelect: () => selected && onOpen(selected), disabled: !selected }] },
        { label: "Tools", items: [{ label: "Models and training", onSelect: onShowModels }] },
      ]} />
      <div className="pb">
        <aside className="pb-side">
          <div className="heading">Scenes</div>
          <button aria-current="true"><Clock size={14} />Recent scenes</button>
          <button onClick={onShowModels}><FolderOpen size={14} />Models and training</button>
        </aside>
        <section className="pb-main">
          <div className="pb-top">
            <div>
              <h1>Select or create a scene</h1>
              <p>Import one photograph. SpatialAI estimates depth, separates objects and regions, and builds an editable 3D scene on this computer.</p>
            </div>
          </div>
          <div className="pb-grid">
            <div className={`pb-card pb-new${over ? " over" : ""}`}
              onDragOver={(e) => { e.preventDefault(); setOver(true); }} onDragLeave={() => setOver(false)}
              onDrop={(e) => { e.preventDefault(); setOver(false); const f = e.dataTransfer.files[0]; if (f) importPhoto(f); }}>
              <div className="img"><ImagePlus size={26} /><span>{busy ? "Importing…" : "Import photo"}</span><span className="muted">JPG, PNG, WEBP</span></div>
              <div className="name">New scene</div>
              <div className="meta">Drop a photo or click</div>
              <input id="pb-file" type="file" accept="image/jpeg,image/png,image/webp" aria-label="Import photo" disabled={busy}
                onChange={(e) => { const f = e.target.files?.[0]; if (f) importPhoto(f); e.target.value = ""; }} />
            </div>
            {projects?.map((p) => (
              <button key={p.id} className="pb-card" aria-selected={selected === p.id}
                onClick={() => setSelected(p.id)} onDoubleClick={() => onOpen(p.id)}>
                <div className="img" style={{ backgroundImage: p.image_url ? `url(${p.image_url})` : undefined }} />
                <div className="name">{p.name}</div>
                <div className="meta">{STATUS[p.status] ?? p.status}{p.scene_category ? ` · ${p.scene_category.replace(/_/g, " ")}` : ""}</div>
              </button>
            ))}
          </div>
          <div className="pb-bottom">
            {error ? <span className="error-box" style={{ flex: 1 }}>{error}</span> :
              <span className="path">{sel ? `${sel.name}, ${sel.image_width} × ${sel.image_height}, ${new Date(sel.updated_at * 1000).toLocaleString()}` : projects?.length === 0 ? "No scenes yet. Import a photo to start." : "Double-click a scene to open it."}</span>}
            <button className="btn" disabled={!selected} onClick={remove}><Trash2 size={13} />Delete</button>
            <button className="btn primary" disabled={!selected} onClick={() => selected && onOpen(selected)}>Open</button>
          </div>
        </section>
      </div>
    </div>
  );
}
