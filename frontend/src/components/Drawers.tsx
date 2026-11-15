import { useState } from "react";
import { FileBox, Folder, ImagePlus, Package, Search, X } from "lucide-react";
import { api, type Asset, type LogLine, type SceneMeta } from "../lib/api";

const FOLDERS = [
  { key: "all", label: "All" },
  { key: "scene", label: "Scene" },
  { key: "objects", label: "Objects" },
  { key: "textures", label: "Textures and maps" },
] as const;
type FolderKey = (typeof FOLDERS)[number]["key"];

function folderOf(a: Asset): FolderKey {
  if (a.object) return "objects";
  if (a.type.startsWith("image")) return "textures";
  return "scene";
}

export function ContentDrawer({ meta, projectName, onClose, onImport }: {
  meta: SceneMeta | null; projectName: string; onClose: () => void; onImport: (f: File) => void;
}) {
  const [folder, setFolder] = useState<FolderKey>("all");
  const [q, setQ] = useState("");
  const assets = (meta?.assets ?? []).filter((a) => (folder === "all" || folderOf(a) === folder) && a.name.toLowerCase().includes(q.toLowerCase()));
  return (
    <div className="drawer" role="dialog" aria-label="Content drawer">
      <div className="drawer-head">
        <strong>Content Drawer</strong>
        <label className="btn" style={{ position: "relative" }}><ImagePlus size={13} />Import photo
          <input type="file" accept="image/jpeg,image/png,image/webp" style={{ position: "absolute", inset: 0, opacity: 0 }}
            onChange={(e) => { const f = e.target.files?.[0]; if (f) onImport(f); e.target.value = ""; }} /></label>
        <div className="search grow" style={{ padding: 0 }}><div className="field"><Search size={13} className="muted" />
          <input placeholder={`Search ${projectName}`} value={q} onChange={(e) => setQ(e.target.value)} aria-label="Search assets" /></div></div>
        <button className="eye" title="Close" onClick={onClose}><X size={15} /></button>
      </div>
      <div className="drawer-body">
        <div className="folders">
          {FOLDERS.map((f) => (
            <button key={f.key} aria-current={folder === f.key} onClick={() => setFolder(f.key)}><Folder size={13} color="#d3a24a" />{f.label}</button>
          ))}
        </div>
        <div className="tiles">
          {!meta && <p className="empty-note">Assets appear after the scene is generated.</p>}
          {assets.map((a) => (
            <a key={a.id} className="tile" href={api.assetUrl(a.id)} download title={`Download ${a.name}`}>
              <div className="img" style={a.type.startsWith("image") ? { backgroundImage: `url(${api.assetUrl(a.id)})` } : undefined}>
                {!a.type.startsWith("image") && (a.object ? <FileBox size={26} /> : <Package size={26} />)}
              </div>
              <span className="name">{a.object ? a.name.replace(/^obj_(\d+)_(.*)$/, "SM_$2_$1") : a.name}</span>
              <span className="type">{a.type.startsWith("image") ? "Texture" : a.type.includes("zip") ? "glTF archive" : a.object ? "Static mesh" : "Scene"}</span>
            </a>
          ))}
        </div>
      </div>
    </div>
  );
}

export function OutputLog({ lines, onClose }: { lines: (LogLine | { t: number; level: "cmd"; message: string })[]; onClose: () => void }) {
  const [filter, setFilter] = useState<Record<string, boolean>>({ info: true, stage: true, warn: true, error: true, cmd: true });
  const [q, setQ] = useState("");
  const shown = lines.filter((l) => filter[l.level] && l.message.toLowerCase().includes(q.toLowerCase()));
  return (
    <div className="drawer" role="dialog" aria-label="Output log">
      <div className="drawer-head">
        <strong>Output Log</strong>
        {(["info", "warn", "error"] as const).map((k) => (
          <label key={k} className="sb-btn"><input type="checkbox" checked={filter[k]} onChange={(e) => setFilter({ ...filter, [k]: e.target.checked })} />
            {{ info: "Messages", warn: "Warnings", error: "Errors" }[k]}</label>
        ))}
        <div className="search grow" style={{ padding: 0 }}><div className="field"><Search size={13} className="muted" />
          <input placeholder="Search log" value={q} onChange={(e) => setQ(e.target.value)} aria-label="Search log" /></div></div>
        <button className="eye" title="Close" onClick={onClose}><X size={15} /></button>
      </div>
      <div className="log">
        {shown.length === 0 ? <span className="muted">No log lines yet.</span> :
          shown.map((l, i) => <div key={i} className={l.level}>[{new Date(l.t * 1000).toLocaleTimeString()}] {l.level === "cmd" ? "Cmd: " : l.level === "warn" ? "Warning: " : l.level === "error" ? "Error: " : "LogSpatialAI: "}{l.message}</div>)}
      </div>
    </div>
  );
}
