import { useState, type ReactNode } from "react";
import { ChevronRight, Lock, RotateCcw, Search, Unlock } from "lucide-react";
import type { Completeness, SceneMeta, SceneNode } from "../lib/api";
import { STATUS_LABEL } from "../lib/api";
import { actorName, objectName, sourceLabel } from "../lib/naming";
import type { Transform } from "../viewer/SceneViewer";

type Cat = "All" | "General" | "Transform" | "Mesh" | "Detection" | "Rendering";
const CATS: Cat[] = ["General", "Transform", "Mesh", "Detection", "Rendering", "All"];

interface Props {
  meta: SceneMeta | null; imageUrl: string | null; selected: string | null; nodes: SceneNode[];
  transform: Transform | null; movable: boolean; visible: Record<string, boolean>;
  onTransform: (t: Transform) => void; onReset: () => void; onToggle: (nodes: string[], on: boolean) => void;
}

function AxisInput({ value, axis, step, onChange, label }: { value: number; axis: "x" | "y" | "z"; step: number; onChange: (v: number) => void; label: string }) {
  const [text, setText] = useState<string | null>(null);
  const displayVal = text !== null ? text : Number((value ?? 0).toFixed(3)).toString();

  return (
    <label className="axis-field" title={`${label} ${axis.toUpperCase()}`}>
      <i style={{ background: `var(--${axis})` }} />
      <input
        type="number"
        step={step}
        value={displayVal}
        aria-label={`${label} ${axis.toUpperCase()}`}
        onFocus={() => setText(Number((value ?? 0).toFixed(3)).toString())}
        onChange={(e) => {
          setText(e.target.value);
          const n = parseFloat(e.target.value);
          if (!isNaN(n)) onChange(n);
        }}
        onBlur={() => {
          setText(null);
          const n = parseFloat(displayVal);
          onChange(isNaN(n) ? 0 : n);
        }}
      />
    </label>
  );
}

function Section({ title, children, show }: { title: string; children: ReactNode; show: boolean }) {
  if (!show) return null;
  return <details className="d-section" open><summary><ChevronRight size={12} />{title}</summary>{children}</details>;
}

function Prop({ k, children, reset, q }: { k: string; children: ReactNode; reset?: () => void; q: string }) {
  if (q && !k.toLowerCase().includes(q)) return null;
  return (
    <div className="prop"><span className="k">{k}</span><div className="v">{children}</div>
      {reset ? <button className="reset-btn" title="Reset to reconstructed value" onClick={reset}><RotateCcw size={12} /></button> : <span />}
    </div>
  );
}

function cropStyle(url: string | null, box: number[] | undefined, w: number, h: number) {
  if (!url || !box) return { backgroundImage: url ? `url(${url})` : undefined, backgroundSize: "cover", backgroundPosition: "center" };
  const [x0, y0, x1, y1] = box;
  const bw = Math.max(1, x1 - x0), bh = Math.max(1, y1 - y0), side = Math.max(bw, bh);
  const cx = (x0 + x1) / 2 - side / 2, cy = (y0 + y1) / 2 - side / 2;
  return {
    backgroundImage: `url(${url})`,
    backgroundSize: `${(w / side) * 100}% ${(h / side) * 100}%`,
    backgroundPosition: `${(cx / Math.max(w - side, 1)) * 100}% ${(cy / Math.max(h - side, 1)) * 100}%`,
  };
}

export default function Details(p: Props) {
  const [tab, setTab] = useState<"details" | "scene">("details");
  const [cat, setCat] = useState<Cat>("All");
  const [query, setQuery] = useState("");
  const [lockScale, setLockScale] = useState(true);
  const q = query.trim().toLowerCase();
  const show = (c: Cat) => cat === "All" || cat === c;
  const main = p.nodes.find((n) => n.status === "observed") ?? p.nodes[0];
  const t = p.transform;

  const setAxis = (key: "position" | "rotation" | "scale", i: number, v: number) => {
    if (!t) return;
    const next = { ...t, [key]: [...t[key]] };
    if (key === "scale" && lockScale) {
      const ratio = v / (t.scale[i] || 1);
      next.scale = t.scale.map((s, j) => (j === i ? v : s * ratio));
    } else next[key][i] = v;
    p.onTransform(next);
  };

  return (
    <section className="panel" aria-label="Details">
      <div className="panel-tabs">
        <button className="panel-tab" aria-selected={tab === "details"} onClick={() => setTab("details")}>Details</button>
        <button className="panel-tab" aria-selected={tab === "scene"} onClick={() => setTab("scene")}>Scene Settings</button>
      </div>
      <div className="panel-body">
        {tab === "scene" ? <SceneSettings meta={p.meta} /> : !main ? (
          <p className="empty-note">Select an actor in the viewport or the Outliner to see its details.</p>
        ) : (
          <>
            <div className="d-actor"><span className="name">{main.kind === "object" ? objectName(p.nodes) : actorName(main)}</span></div>
            <div className="d-instance">
              <div className="row root">{main.kind === "object" ? objectName(p.nodes) : actorName(main)} (Instance)</div>
              {p.nodes.map((n) => <div key={n.node} className="row child">{actorName(n)} ({n.status === "observed" ? "StaticMeshComponent" : n.status === "generated" ? "GeneratedMeshComponent" : "InferredMeshComponent"})</div>)}
            </div>
            <div className="search"><div className="field"><Search size={13} className="muted" />
              <input placeholder="Search details" value={query} onChange={(e) => setQuery(e.target.value)} aria-label="Search details" /></div></div>
            <div className="d-cats">{CATS.map((c) => <button key={c} aria-pressed={cat === c} onClick={() => setCat(c)}>{c}</button>)}</div>

            <Section title="Transform" show={show("Transform") || show("General")}>
              {p.movable && t ? (
                <>
                  <Prop k="Location" q={q} reset={p.onReset}><div className="xyz">{(["x", "y", "z"] as const).map((a, i) =>
                    <AxisInput key={a} axis={a} label="Location" step={0.05} value={t.position[i]} onChange={(v) => setAxis("position", i, v)} />)}</div></Prop>
                  <Prop k="Rotation" q={q} reset={p.onReset}><div className="xyz">{(["x", "y", "z"] as const).map((a, i) =>
                    <AxisInput key={a} axis={a} label="Rotation" step={5} value={t.rotation[i]} onChange={(v) => setAxis("rotation", i, v)} />)}</div></Prop>
                  <Prop k="Scale" q={q} reset={p.onReset}><div style={{ display: "flex", gap: 3 }}><div className="xyz grow">{(["x", "y", "z"] as const).map((a, i) =>
                    <AxisInput key={a} axis={a} label="Scale" step={0.05} value={t.scale[i]} onChange={(v) => setAxis("scale", i, Math.max(0.01, v))} />)}</div>
                    <button className="reset-btn" title={lockScale ? "Uniform scale on" : "Uniform scale off"} onClick={() => setLockScale(!lockScale)}>{lockScale ? <Lock size={12} /> : <Unlock size={12} />}</button></div></Prop>
                  <Prop k="Mobility" q={q}><span>Movable</span></Prop>
                </>
              ) : (
                <>
                  <Prop k="Bounds min" q={q}><span className="num">{main.bounds[0].map((v) => v.toFixed(2)).join(", ")}</span></Prop>
                  <Prop k="Bounds max" q={q}><span className="num">{main.bounds[1].map((v) => v.toFixed(2)).join(", ")}</span></Prop>
                  <Prop k="Mobility" q={q}><span>Static (environment)</span></Prop>
                </>
              )}
            </Section>

            <Section title="Static Mesh" show={show("Mesh") || show("General")}>
              <Prop k="Static mesh" q={q}>
                <div className="mesh-row"><div className="thumb" style={cropStyle(p.imageUrl, main.box, p.meta?.camera.width ?? 1, p.meta?.camera.height ?? 1)} />
                  <div style={{ display: "grid", gap: 2 }}><span>{actorName(main)}</span><span className="muted num">{p.nodes.reduce((a, n) => a + n.faces, 0).toLocaleString()} triangles</span></div></div>
              </Prop>
              {p.nodes.map((n) => (
                <Prop key={n.node} k={actorName(n).replace(/^SM_/, "")} q={q}>
                  <div style={{ display: "grid", gap: 2 }}>
                    <span className="pill-tag" style={{ color: `var(--${n.status})` }}>{STATUS_LABEL[n.status]}</span>
                    <span className="muted num">{n.faces.toLocaleString()} faces, {sourceLabel(n)}{n.method ? `: ${n.method}` : ""}</span>
                  </div>
                </Prop>
              ))}
            </Section>

            {(() => {
              const comp = p.meta?.report?.completeness as Completeness | undefined;
              const el = comp?.elements?.find((x) => x.name === main.element);
              const back = p.nodes.find((n) => n.shape_prior);
              if (!el && main.closed == null) return null;
              return (
                <Section title="Geometry" show={show("Mesh") || show("General")}>
                  <Prop k="Closed shell" q={q}><span style={{ color: (el ? el.watertight && el.outward_normals : main.closed) ? "var(--generated)" : "#ff8a80" }}>
                    {(el ? el.watertight && el.outward_normals : main.closed) ? "Yes, viewable from every side" : "No, has open edges"}</span></Prop>
                  {el && <Prop k="Status" q={q}>{el.status}{el.repairs.length ? ` (${el.repairs.join("; ")})` : ""}</Prop>}
                  {el && !el.watertight && <Prop k="Open edges" q={q}><span className="num">{el.boundary_edges}</span></Prop>}
                  {back?.shape_prior && <Prop k="Hidden-side prior" q={q}>{back.shape_prior} ({back.profile})</Prop>}
                  {back?.thickness != null && <Prop k="Inferred depth" q={q}><span className="num">≈ {back.thickness.toFixed(2)} m</span></Prop>}
                  {el?.texture_size?.length ? <Prop k="Back texture" q={q}><span className="num">{el.texture_size.join(" × ")} px, synthesized</span></Prop> : null}
                </Section>
              );
            })()}

            {main.kind === "object" && (
              <Section title="Detection" show={show("Detection")}>
                <Prop k="Label" q={q}>{main.label}</Prop>
                {main.confidence != null && <Prop k="Confidence" q={q}><span className="num">{Math.round(main.confidence * 100)}%</span></Prop>}
                {main.mask_method && <Prop k="Mask" q={q}>{main.mask_method === "sam" ? "Segment Anything" : main.mask_method === "grabcut" ? "GrabCut" : main.mask_method}{main.mask_score != null ? ` (${main.mask_score.toFixed(2)})` : ""}</Prop>}
                {main.median_depth != null && <Prop k="Distance" q={q}><span className="num">≈ {main.median_depth.toFixed(2)} m</span></Prop>}
                {main.box && <Prop k="Image box" q={q}><span className="num">{main.box.map((v) => Math.round(v)).join(", ")}</span></Prop>}
              </Section>
            )}

            <Section title="Rendering" show={show("Rendering")}>
              {p.nodes.map((n) => (
                <Prop key={n.node} k={`Visible ${actorName(n).replace(/^SM_/, "")}`} q={q}>
                  <input type="checkbox" checked={!!p.visible[n.node]} onChange={(e) => p.onToggle([n.node], e.target.checked)} />
                </Prop>
              ))}
              <Prop k="Material" q={q}>
                <div className="mesh-row"><div className="thumb" style={{ width: 40, height: 40, ...cropStyle(p.imageUrl, undefined, 1, 1) }} />
                  <span>{main.node === "bg_fill" ? "M_InpaintedPhoto" : main.node === "bg_shell" ? "M_SynthesizedBack" : "M_PhotoProjection"}{p.nodes.some((n) => n.node.endsWith("_back")) ? " + M_SynthesizedBack" : ""}</span></div>
              </Prop>
            </Section>
          </>
        )}
      </div>
    </section>
  );
}

function SceneSettings({ meta }: { meta: SceneMeta | null }) {
  if (!meta) return <p className="empty-note">Generate the scene to see its settings.</p>;
  const r = meta.report;
  const q = "";
  return (
    <>
      <Section title="Scene" show>
        <Prop k="Type" q={q}>{meta.scene.category.replace(/_/g, " ")} ({Math.round(meta.scene.confidence * 100)}%)</Prop>
        <Prop k="Depth range" q={q}>{meta.scene.near}–{meta.scene.far} m (assumed)</Prop>
        <Prop k="Coordinates" q={q}>Y up, camera at origin</Prop>
      </Section>
      {(() => {
        const c = r?.completeness as Completeness | undefined;
        if (!c) return null;
        return (
          <Section title="Completeness" show>
            <Prop k="Check" q={q}><span style={{ color: c.passed ? "var(--generated)" : "#ff8a80" }}>{c.passed ? "Passed: every part is closed" : "Issues found"}</span></Prop>
            <Prop k="Objects closed" q={q}><span className="num">{c.objects_complete} / {c.objects_total}</span></Prop>
            <Prop k="Background" q={q}>{c.background_closed ? "Closed solid" : "Open"}</Prop>
            {r?.export?.all_closed != null && <Prop k="After export" q={q}>{r.export.all_closed ? "Still closed (GLB reloaded)" : "Not closed after reload"}</Prop>}
            {c.issues.map((i, k) => <Prop key={k} k={`Issue ${k + 1}`} q={q}><span style={{ color: "#ff8a80" }}>{i}</span></Prop>)}
          </Section>
        );
      })()}
      <Section title="Camera" show>
        <Prop k="Field of view" q={q}>{meta.camera.hfov_deg}° ({meta.fov_source})</Prop>
        <Prop k="Resolution" q={q}><span className="num">{meta.camera.width} × {meta.camera.height}</span></Prop>
      </Section>
      <Section title="Lighting" show>
        <Prop k="Ambient" q={q}><span className="num">{meta.lighting.ambient_intensity}</span></Prop>
        <Prop k="Key colour" q={q}><span className="dot" style={{ width: 14, height: 14, background: `rgb(${meta.lighting.key_color.map((c) => Math.round(c * 255)).join(",")})` }} /></Prop>
        <Prop k="Ground plane" q={q}>{meta.ground ? `${Math.round(meta.ground.inlier_ratio * 100)}% fit` : "not found"}</Prop>
      </Section>
      {r && (
        <>
          <Section title="Quality" show>
            <Prop k="Seen in photo" q={q}><span className="num">{Math.round(r.geometry.observed_ratio * 100)}% of faces</span></Prop>
            <Prop k="Texture coverage" q={q}><span className="num">{Math.round(r.geometry.texture_coverage * 100)}%</span></Prop>
            <Prop k="Edges cut" q={q}><span className="num">{(r.geometry.discontinuity_cut_fraction * 100).toFixed(1)}%</span></Prop>
            <Prop k="Build time" q={q}><span className="num">{r.performance.timings_s.total}s on {r.performance.device}</span></Prop>
            <Prop k="Peak VRAM" q={q}><span className="num">{r.performance.peak_vram_gb != null ? `${r.performance.peak_vram_gb} GB` : "n/a"}</span></Prop>
          </Section>
          <Section title="Models used" show>
            {Object.entries(r.models as Record<string, string>).map(([k, v]) => <Prop key={k} k={k.replace(/_/g, " ")} q={q}><span title={v}>{v}</span></Prop>)}
          </Section>
          {r.fallbacks.length > 0 && <Section title="Fallbacks" show>{r.fallbacks.map((f: string, i: number) => <p key={i} className="empty-note" style={{ padding: "4px 10px" }}>{f}</p>)}</Section>}
        </>
      )}
    </>
  );
}
