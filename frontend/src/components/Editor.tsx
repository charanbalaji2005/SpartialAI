import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  Box, Camera, Check, CircleDashed, Cpu, Download, FolderOpen, Globe, Grid3x3, ImagePlus, Loader2, Magnet,
  Menu as MenuIcon, MousePointer2, Move, Play, Rotate3d, Save, Scaling, Settings, Square, Sun, Terminal, X, Eye,
} from "lucide-react";
import { Dropdown, MenuBar, type MenuItem } from "./Menu";
import Outliner from "./Outliner";
import Details from "./Details";
import { ContentDrawer, OutputLog } from "./Drawers";
import { ModelsDialog, ShortcutsDialog } from "./Dialogs";
import { SceneViewer, type ShadingMode, type ToolMode, type Transform, type ViewPreset } from "../viewer/SceneViewer";
import { api, type GpuInfo, type LogLine, type Project, type Quality, type SceneMeta } from "../lib/api";
import { useJobProgress } from "../lib/useJobProgress";
import { actorName } from "../lib/naming";

const QUALITY_LABEL: Record<Quality, string> = { fast: "Low", balanced: "Medium", best: "Epic" };
const SHADING_LABEL: Record<ShadingMode, string> = { photo: "Unlit", lit: "Lit", clay: "Clay", truth: "Seen vs inferred", check: "Mesh check" };
const PRESETS: { key: ViewPreset; label: string; shortcut: string }[] = [
  { key: "front", label: "Front", shortcut: "Num 1" }, { key: "back", label: "Back", shortcut: "Ctrl+Num 1" },
  { key: "right", label: "Right", shortcut: "Num 3" }, { key: "left", label: "Left", shortcut: "Ctrl+Num 3" },
  { key: "top", label: "Top", shortcut: "Num 7" }, { key: "bottom", label: "Bottom", shortcut: "Ctrl+Num 7" },
];
type Phase = "photo" | "running" | "scene";
type CmdLine = { t: number; level: "cmd"; message: string };

function download(href: string, name: string) { const a = document.createElement("a"); a.href = href; a.download = name; a.click(); }

export default function Editor({ projectId, onClose, onOpenProject }: { projectId: string; onClose: () => void; onOpenProject: (id: string) => void }) {
  const host = useRef<HTMLDivElement>(null);
  const viewportRef = useRef<HTMLDivElement>(null);
  const cmdRef = useRef<HTMLInputElement>(null);
  const viewer = useRef<SceneViewer | null>(null);
  const [project, setProject] = useState<Project | null>(null);
  const [meta, setMeta] = useState<SceneMeta | null>(null);
  const [phase, setPhase] = useState<Phase>("photo");
  const [quality, setQuality] = useState<Quality>("balanced");
  const [error, setError] = useState<string | null>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const [visible, setVisible] = useState<Record<string, boolean>>({});
  const [transform, setTransform] = useState<Transform | null>(null);
  const [tool, setTool] = useState<ToolMode>("translate");
  const [space, setSpace] = useState<"world" | "local">("world");
  const [snap, setSnap] = useState({ on: false, move: 0.1, rot: 15, scale: 0.25 });
  const [camSpeed, setCamSpeed] = useState(1);
  const [shading, setShading] = useState<ShadingMode>("photo");
  const [wire, setWire] = useState(false);
  const [grid, setGrid] = useState(true);
  const [axes, setAxes] = useState(false);
  const [ortho, setOrtho] = useState(false);
  const [drawer, setDrawer] = useState<null | "content" | "log">(null);
  const [dialog, setDialog] = useState<null | "models" | "shortcuts">(null);
  const [cmdLog, setCmdLog] = useState<CmdLine[]>([]);
  const [lastLogs, setLastLogs] = useState<LogLine[]>([]);
  const [gpu, setGpu] = useState<GpuInfo | null>(null);
  const [stats, setStats] = useState({ meshes: 0, triangles: 0, visible: 0 });
  const [cancelling, setCancelling] = useState(false);
  const job = useJobProgress(projectId, phase === "running");

  // ------------------------------------------------------------ engine lifecycle
  useEffect(() => {
    const v = new SceneViewer(host.current!);
    viewer.current = v;
    v.onSelect = (k) => { setSelected(k); setTransform(k ? v.getTransform(k) : null); };
    v.onTransform = (_g, t) => setTransform(t);
    return () => { v.dispose(); viewer.current = null; };
  }, []);

  const loadScene = useCallback(async () => {
    const m = await api.scene(projectId);
    setMeta(m);
    await viewer.current?.load(m);
    setVisible(Object.fromEntries(m.nodes.map((n) => [n.node, n.default_visible])));
    setSelected(null); setTransform(null);
    if (viewer.current) setStats(viewer.current.stats());
  }, [projectId]);

  const loadProject = useCallback(async () => {
    const p = await api.project(projectId);
    setProject(p);
    setQuality(p.quality ?? "balanced");
    if (p.status === "running" || p.status === "queued") setPhase("running");
    else if (p.has_scene) { setPhase("scene"); await loadScene(); }
    else setPhase("photo");
    if (p.status === "failed" && p.error) setError(p.error);
    api.status(projectId).then((s) => setLastLogs(s.logs ?? [])).catch(() => {});
  }, [projectId, loadScene]);

  useEffect(() => { loadProject().catch((e) => setError(e.message)); }, [loadProject]);
  useEffect(() => { const id = setInterval(() => api.gpu().then(setGpu).catch(() => {}), 3000); api.gpu().then(setGpu).catch(() => {}); return () => clearInterval(id); }, []);

  useEffect(() => {
    if (!job) return;
    setLastLogs(job.logs ?? []);
    if (["completed", "failed", "cancelled"].includes(job.status)) {
      setCancelling(false);
      if (job.status === "completed") { setPhase("scene"); loadScene().catch((e) => setError(e.message)); api.project(projectId).then(setProject); }
      else { setPhase(meta ? "scene" : "photo"); setError(job.status === "cancelled" ? "Build cancelled." : job.error ?? "Build failed."); }
    }
  }, [job, loadScene, projectId, meta]);

  // ------------------------------------------------------------ viewer settings
  useEffect(() => { viewer.current?.setShading(shading); }, [shading]);
  useEffect(() => { viewer.current?.setWireframe(wire); }, [wire]);
  useEffect(() => { viewer.current?.setGrid(grid); }, [grid]);
  useEffect(() => { viewer.current?.setAxes(axes); }, [axes]);
  useEffect(() => { viewer.current?.setOrthographic(ortho); }, [ortho]);
  useEffect(() => { viewer.current?.setTool(tool); }, [tool]);
  useEffect(() => { viewer.current?.setSpace(space); }, [space]);
  useEffect(() => { viewer.current?.setCameraSpeed(camSpeed); }, [camSpeed]);
  useEffect(() => { viewer.current?.setSnap(snap.on ? snap.move : null, snap.on ? snap.rot : null, snap.on ? snap.scale : null); }, [snap]);

  // ------------------------------------------------------------ actions
  const select = useCallback((key: string | null) => {
    viewer.current?.select(key);
    setSelected(key);
    setTransform(key ? viewer.current?.getTransform(key) ?? null : null);
  }, []);
  const toggle = useCallback((nodes: string[], on: boolean) => {
    viewer.current?.setVisible(nodes, on);
    setVisible((v) => ({ ...v, ...Object.fromEntries(nodes.map((n) => [n, on])) }));
    if (viewer.current) setStats(viewer.current.stats());
  }, []);
  const applyTransform = useCallback((t: Transform) => {
    if (!selected) return;
    viewer.current?.setTransform(selected, t);
    setTransform(viewer.current?.getTransform(selected) ?? t);
  }, [selected]);

  const generate = async (q: Quality = quality) => {
    setError(null);
    try { await api.generate(projectId, q); setQuality(q); setPhase("running"); }
    catch (e: any) { setError(e.message); }
  };
  const cancel = async () => { setCancelling(true); try { await api.cancel(projectId); } catch { setCancelling(false); } };
  const importPhoto = async (file: File) => {
    setError(null);
    try { onOpenProject((await api.upload(file)).id); } catch (e: any) { setError(e.message); }
  };
  const exportEdited = async (onlyVisible = true) => {
    if (!viewer.current || !meta) return;
    const url = URL.createObjectURL(await viewer.current.exportEdited(onlyVisible));
    download(url, `${project?.name.replace(/\s+/g, "_") ?? "scene"}_edited.glb`);
    setTimeout(() => URL.revokeObjectURL(url), 3000);
  };
  const screenshot = () => { if (viewer.current) download(viewer.current.screenshot(), `${project?.name ?? "scene"}_viewport.png`); };
  const fullscreen = () => { if (document.fullscreenElement) document.exitFullscreen(); else viewportRef.current?.requestFullscreen?.(); };
  const nodesOf = (key: string | null) => meta?.nodes.filter((n) => (n.group || n.node) === key) ?? [];
  const toggleSelected = () => { const ns = nodesOf(selected).map((n) => n.node); if (ns.length) toggle(ns, !ns.some((n) => visible[n])); };

  // ------------------------------------------------------------ console
  const say = (message: string) => setCmdLog((l) => [...l, { t: Date.now() / 1000, level: "cmd", message }]);
  const runCommand = (raw: string) => {
    const [cmd, ...rest] = raw.trim().split(/\s+/);
    const arg = rest.join(" ");
    say(raw);
    const find = (name: string) => meta?.nodes.find((n) => actorName(n).toLowerCase() === name.toLowerCase() || n.node === name || n.label === name);
    switch ((cmd || "").toLowerCase()) {
      case "help": say("Commands: generate [fast|balanced|best], cancel, view unlit|lit|clay|truth|check, cam front|back|left|right|top|bottom|photo, wire on|off, select <actor>, hide <actor|all>, show <actor|all>, focus, tool select|move|rotate|scale, export glb|gltf|edited, stat gpu, stat scene, stat mesh, clear"); break;
      case "generate": generate((["fast", "balanced", "best"].includes(arg) ? arg : quality) as Quality); break;
      case "cancel": cancel(); break;
      case "view": { const m = { unlit: "photo", photo: "photo", lit: "lit", clay: "clay", truth: "truth", check: "check" }[arg] as ShadingMode | undefined; if (m) setShading(m); else say("Usage: view unlit|lit|clay|truth|check"); break; }
      case "cam": if (arg === "photo") viewer.current?.resetCamera(); else if (PRESETS.some((p) => p.key === arg)) viewer.current?.viewPreset(arg as ViewPreset); else say("Usage: cam front|back|left|right|top|bottom|photo"); break;
      case "wire": setWire(arg ? arg === "on" : !wire); break;
      case "select": { const n = find(arg); if (n) select(n.group || n.node); else say(`No actor named ${arg}`); break; }
      case "hide": case "show": {
        const on = cmd.toLowerCase() === "show";
        if (arg === "all") toggle(meta?.nodes.map((n) => n.node) ?? [], on);
        else { const n = find(arg); if (n) toggle(nodesOf(n.group || n.node).map((x) => x.node), on); else say(`No actor named ${arg}`); }
        break;
      }
      case "focus": viewer.current?.focusSelection(); break;
      case "tool": { const t = { select: "select", move: "translate", rotate: "rotate", scale: "scale" }[arg] as ToolMode | undefined; if (t) setTool(t); break; }
      case "export": if (arg === "glb") download(api.exportUrl(projectId, "glb"), "scene.glb"); else if (arg === "gltf") download(api.exportUrl(projectId, "gltf"), "scene_gltf.zip"); else if (arg === "edited") exportEdited(); else say("Usage: export glb|gltf|edited"); break;
      case "stat":
        if (arg === "gpu") say(gpu?.cuda_available ? `${gpu.name}: ${(gpu.smi_used_gb ?? gpu.allocated_gb ?? 0).toFixed(2)} / ${(gpu.smi_total_gb ?? gpu.total_vram_gb ?? 0).toFixed(1)} GB, ${gpu.utilization_pct ?? "?"}% busy` : "Running on CPU");
        else if (arg === "scene") say(`${stats.meshes} meshes (${stats.visible} visible), ${stats.triangles.toLocaleString()} triangles`);
        else if (arg === "mesh") {
          const c = meta?.report?.completeness;
          if (!c) say("No completeness report for this scene (generate it again with this version).");
          else {
            say(`Completeness: ${c.passed ? "PASSED" : "ISSUES"} · ${c.objects_complete}/${c.objects_total} objects closed · background ${c.background_closed ? "closed" : "OPEN"}`);
            for (const el of c.elements ?? []) say(`  ${el.name}: ${el.status}, ${el.watertight ? "watertight" : `${el.boundary_edges} open edges`}${el.shape_prior ? `, ${el.shape_prior} prior` : ""}${el.repairs?.length ? ` (${el.repairs.join("; ")})` : ""}`);
            for (const i of c.issues ?? []) say(`  issue: ${i}`);
          }
        }
        else say("Usage: stat gpu|scene|mesh");
        break;
      case "clear": setCmdLog([]); break;
      default: say(`Unknown command “${cmd}”. Type help.`);
    }
    setDrawer("log");
  };

  // ------------------------------------------------------------ keyboard
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const typing = (e.target as HTMLElement).closest("input, select, textarea");
      if (e.ctrlKey && e.code === "Space") { e.preventDefault(); setDrawer((d) => (d === "content" ? null : "content")); return; }
      if (e.ctrlKey && e.key.toLowerCase() === "g") { e.preventDefault(); if (phase !== "running") generate(); return; }
      // Numpad view presets (Blender-style): 1 front, 3 right, 7 top; with Ctrl the opposite side.
      const numpad = { Numpad1: ["front", "back"], Numpad3: ["right", "left"], Numpad7: ["top", "bottom"] }[e.code] as [ViewPreset, ViewPreset] | undefined;
      if (numpad && !typing) { e.preventDefault(); viewer.current?.viewPreset(numpad[e.ctrlKey ? 1 : 0]); return; }
      if (typing || e.ctrlKey || e.metaKey || e.altKey) return;
      const k = e.key.toLowerCase();
      const map: Record<string, () => void> = {
        q: () => setTool("select"), w: () => setTool("translate"), e: () => setTool("rotate"), r: () => setTool("scale"),
        f: () => viewer.current?.focusSelection(), v: () => viewer.current?.resetCamera(), g: () => setGrid((x) => !x),
        o: () => setOrtho((x) => !x), t: () => setShading((s) => (s === "truth" ? "photo" : "truth")), h: toggleSelected,
        "1": () => setShading("photo"), "2": () => setShading("lit"), "3": () => setShading("clay"), "4": () => setShading("truth"), "5": () => setShading("check"),
        escape: () => select(null), "`": () => cmdRef.current?.focus(),
      };
      if (map[k]) { e.preventDefault(); map[k](); }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  });

  useEffect(() => { if (viewer.current && meta) setStats(viewer.current.stats()); }, [visible, meta]);

  // ------------------------------------------------------------ menus
  const hasScene = phase === "scene" && !!meta;
  const exportItems: MenuItem[] = [
    { label: "Scene as GLB", onSelect: () => download(api.exportUrl(projectId, "glb"), `${project?.name ?? "scene"}.glb`), disabled: !hasScene },
    { label: "Scene as glTF (zip)", onSelect: () => download(api.exportUrl(projectId, "gltf"), `${project?.name ?? "scene"}_gltf.zip`), disabled: !hasScene },
    { label: "Edited scene (visible parts)", onSelect: () => exportEdited(true), disabled: !hasScene },
    { label: "Edited scene (everything)", onSelect: () => exportEdited(false), disabled: !hasScene },
    { label: "Scene metadata (JSON)", onSelect: () => download(api.exportUrl(projectId, "metadata"), "metadata.json"), disabled: !hasScene },
    { type: "sep" },
    { label: "Viewport screenshot", onSelect: screenshot, disabled: !hasScene },
  ];
  const showItems: MenuItem[] = [
    { label: "Grid", checked: grid, onSelect: () => setGrid(!grid), shortcut: "G" },
    { label: "Axes", checked: axes, onSelect: () => setAxes(!axes) },
    { type: "sep" }, { type: "heading", label: "Scene layers" },
    ...(meta ? [
      { label: "Inferred surfaces", checked: meta.nodes.filter((n) => n.status === "inferred").some((n) => visible[n.node]),
        onSelect: () => { const ns = meta.nodes.filter((n) => n.status === "inferred").map((n) => n.node); toggle(ns, !ns.some((n) => visible[n])); } },
      { label: "Ground plane", checked: !!visible["ground_plane"], disabled: !meta.nodes.some((n) => n.node === "ground_plane"), onSelect: () => toggle(["ground_plane"], !visible["ground_plane"]) },
      { label: "Show everything", onSelect: () => toggle(meta.nodes.map((n) => n.node), true) },
    ] as MenuItem[] : []),
  ];
  const qualityItems: MenuItem[] = (["fast", "balanced", "best"] as Quality[]).map((q) => ({
    label: `${QUALITY_LABEL[q]} (${q === "fast" ? "512 px, 4 objects" : q === "balanced" ? "768 px, SAM, 8 objects" : "1024 px, 12 objects, Stable Fast 3D"})`,
    checked: quality === q, onSelect: () => setQuality(q),
  }));
  const menus: { label: string; items: MenuItem[] }[] = [
    { label: "File", items: [
      { label: "Import photo as new scene…", icon: <ImagePlus size={13} />, onSelect: () => document.getElementById("ed-import")?.click() },
      { label: "Open scene…", icon: <FolderOpen size={13} />, onSelect: onClose },
      { type: "sep" }, { type: "heading", label: "Export" }, ...exportItems,
      { type: "sep" }, { label: "Close scene", onSelect: onClose },
    ] },
    { label: "Edit", items: [
      { label: "Reset transform", disabled: !viewer.current?.isMovable(selected), onSelect: () => selected && setTransform(viewer.current?.resetTransform(selected) ?? null) },
      { label: "Hide or show selection", shortcut: "H", disabled: !selected, onSelect: toggleSelected },
      { label: "Deselect", shortcut: "Esc", disabled: !selected, onSelect: () => select(null) },
      { type: "sep" },
      { label: "Snap to grid", checked: snap.on, onSelect: () => setSnap({ ...snap, on: !snap.on }) },
    ] },
    { label: "Window", items: [
      { label: "Content Drawer", shortcut: "Ctrl+Space", checked: drawer === "content", onSelect: () => setDrawer(drawer === "content" ? null : "content") },
      { label: "Output Log", checked: drawer === "log", onSelect: () => setDrawer(drawer === "log" ? null : "log") },
      { label: "Models and training", onSelect: () => setDialog("models") },
    ] },
    { label: "Tools", items: [
      { label: "Select", shortcut: "Q", checked: tool === "select", onSelect: () => setTool("select") },
      { label: "Move", shortcut: "W", checked: tool === "translate", onSelect: () => setTool("translate") },
      { label: "Rotate", shortcut: "E", checked: tool === "rotate", onSelect: () => setTool("rotate") },
      { label: "Scale", shortcut: "R", checked: tool === "scale", onSelect: () => setTool("scale") },
      { type: "sep" }, { label: "Models and training", onSelect: () => setDialog("models") },
    ] },
    { label: "Build", items: [
      { label: "Generate 3D scene", shortcut: "Ctrl+G", disabled: phase === "running" || !project?.image_url, onSelect: () => generate() },
      { label: "Cancel build", disabled: phase !== "running", onSelect: cancel },
      { type: "sep" }, { type: "heading", label: "Quality" }, ...qualityItems,
    ] },
    { label: "Select", items: [
      { label: "None", onSelect: () => select(null) },
      ...(meta?.nodes.map((n) => ({ label: actorName(n), onSelect: () => select(n.kind === "object" ? n.group : n.node) })) ?? []),
    ] },
    { label: "Help", items: [
      { label: "Keyboard shortcuts", onSelect: () => setDialog("shortcuts") },
      { label: "Console commands", onSelect: () => runCommand("help") },
    ] },
  ];

  const pct = Math.round((job?.progress ?? 0) * 100);
  const used = gpu?.smi_used_gb ?? gpu?.allocated_gb;
  const total = gpu?.smi_total_gb ?? gpu?.total_vram_gb;
  const allLogs = useMemo(() => [...lastLogs, ...cmdLog].sort((a, b) => a.t - b.t), [lastLogs, cmdLog]);
  const nodes = nodesOf(selected);

  return (
    <div className="app">
      <MenuBar menus={menus} title={project ? `${project.name} – SpatialAI` : "SpatialAI"} />
      <div className="tabbar">
        <div className="doc-tab"><Globe size={13} color="#9cc3ff" />{project?.name ?? "Loading…"}
          <button className="close" title="Close scene" onClick={onClose}><X size={12} /></button></div>
      </div>

      <div className="toolbar" role="toolbar" aria-label="Main toolbar">
        <button className="tb-btn" disabled title="Scenes save automatically"><Save size={16} /></button>
        <button className="tb-btn" title="Content Drawer (Ctrl+Space)" onClick={() => setDrawer(drawer === "content" ? null : "content")}><FolderOpen size={16} /></button>
        <span className="tb-sep" />
        <Dropdown className="tb-btn boxed" trigger={<><MousePointer2 size={14} />{{ select: "Selection Mode", translate: "Move Mode", rotate: "Rotate Mode", scale: "Scale Mode" }[tool]}</>}
          items={menus[3].items.slice(0, 4)} />
        <label className="tb-btn" title="Import photo as new scene" style={{ position: "relative" }}><ImagePlus size={16} />Import
          <input id="ed-import" type="file" accept="image/jpeg,image/png,image/webp" style={{ position: "absolute", inset: 0, opacity: 0, width: "100%" }}
            onChange={(e) => { const f = e.target.files?.[0]; if (f) importPhoto(f); e.target.value = ""; }} /></label>
        <span className="tb-sep" />
        <button className="tb-btn play" title="Generate 3D scene (Ctrl+G)" disabled={phase === "running" || !project?.image_url} onClick={() => generate()}><Play size={18} fill="currentColor" />Generate</button>
        <button className="tb-btn stop" title="Cancel build" disabled={phase !== "running" || cancelling} onClick={cancel}><Square size={14} fill="currentColor" /></button>
        <Dropdown className="tb-btn" trigger={<>Quality: {QUALITY_LABEL[quality]}</>} items={qualityItems} />
        <span className="tb-sep" />
        <Dropdown className="tb-btn" trigger={<><Download size={15} />Export</>} items={exportItems} />
        <span className="grow" />
        <Dropdown className="tb-btn" align="right" trigger={<><Settings size={15} />Settings</>} items={[
          { label: "Models and training", onSelect: () => setDialog("models"), icon: <Cpu size={13} /> },
          { label: "Keyboard shortcuts", onSelect: () => setDialog("shortcuts") },
          { type: "sep" }, { type: "heading", label: "Camera speed" },
          ...[0.5, 1, 2, 4].map((s) => ({ label: `${s}×`, checked: camSpeed === s, onSelect: () => setCamSpeed(s) })),
        ]} />
      </div>

      <div className="workspace">
        <section className="viewport" ref={viewportRef} aria-label="Viewport">
          <div className="canvas-host" ref={host} />
          {phase !== "scene" && project?.image_url && (
            <div className="vp-photo"><img src={`${project.image_url}?v=${project.updated_at}`} alt="Source photo" style={phase === "running" ? { opacity: 0.45 } : undefined} /></div>
          )}
          <div className="vp-bar">
            <Dropdown className="chip icon" title="Viewport options" trigger={<MenuIcon size={14} />} items={[
              { label: "Back to photo viewpoint", shortcut: "V", onSelect: () => viewer.current?.resetCamera(), disabled: !hasScene },
              { label: "Focus selection", shortcut: "F", onSelect: () => viewer.current?.focusSelection(), disabled: !selected },
              { label: "Screenshot", onSelect: screenshot, disabled: !hasScene },
              { label: "Fullscreen", onSelect: fullscreen },
            ]} />
            <Dropdown className="chip" trigger={<><Camera size={13} />{ortho ? "Orthographic" : "Perspective"}</>} items={[
              { label: "Perspective", checked: !ortho, onSelect: () => setOrtho(false) },
              { label: "Orthographic", shortcut: "O", checked: ortho, onSelect: () => setOrtho(true) },
              { type: "sep" }, { type: "heading", label: "View from" },
              { label: "Photo viewpoint", shortcut: "V", disabled: !hasScene, onSelect: () => viewer.current?.resetCamera() },
              ...PRESETS.map((p) => ({ label: p.label, shortcut: p.shortcut, disabled: !hasScene, onSelect: () => viewer.current?.viewPreset(p.key) })),
            ]} />
            <Dropdown className="chip" trigger={<><Sun size={13} />{SHADING_LABEL[shading]}</>} items={[
              ...(["photo", "lit", "clay", "truth", "check"] as ShadingMode[]).map((m, i) => ({ label: SHADING_LABEL[m], shortcut: String(i + 1), checked: shading === m, onSelect: () => setShading(m) })),
              { type: "sep" }, { label: "Wireframe", checked: wire, onSelect: () => setWire(!wire) },
            ]} />
            <Dropdown className="chip" trigger={<><Eye size={13} />Show</>} items={showItems} />
            <Dropdown className="chip scalability" trigger={<>Scalability: {QUALITY_LABEL[quality]}</>} items={qualityItems} />
            <span className="grow" />
            <div className="cluster" role="group" aria-label="Transform tools">
              <button title="Select (Q)" aria-pressed={tool === "select"} onClick={() => setTool("select")}><MousePointer2 size={13} /></button>
              <button title="Move (W)" aria-pressed={tool === "translate"} onClick={() => setTool("translate")}><Move size={13} /></button>
              <button title="Rotate (E)" aria-pressed={tool === "rotate"} onClick={() => setTool("rotate")}><Rotate3d size={13} /></button>
              <button title="Scale (R)" aria-pressed={tool === "scale"} onClick={() => setTool("scale")}><Scaling size={13} /></button>
            </div>
            <div className="cluster" role="group" aria-label="Coordinate space and snapping">
              <button title={space === "world" ? "World space" : "Local space"} onClick={() => setSpace(space === "world" ? "local" : "world")}>{space === "world" ? <Globe size={13} /> : <Box size={13} />}</button>
              <button title="Snap" aria-pressed={snap.on} onClick={() => setSnap({ ...snap, on: !snap.on })}><Magnet size={13} /></button>
              <Dropdown className="" trigger={<span className="num"><Grid3x3 size={12} /> {snap.move}</span>} title="Move snap" items={[0.01, 0.05, 0.1, 0.5, 1].map((v) => ({ label: `${v} m`, checked: snap.move === v, onSelect: () => setSnap({ ...snap, on: true, move: v }) }))} />
              <Dropdown className="" trigger={<span className="num">∠ {snap.rot}°</span>} title="Rotation snap" items={[5, 10, 15, 45, 90].map((v) => ({ label: `${v}°`, checked: snap.rot === v, onSelect: () => setSnap({ ...snap, on: true, rot: v }) }))} />
              <Dropdown className="" trigger={<span className="num">⤢ {snap.scale}</span>} title="Scale snap" items={[0.1, 0.25, 0.5, 1].map((v) => ({ label: `${v}`, checked: snap.scale === v, onSelect: () => setSnap({ ...snap, on: true, scale: v }) }))} />
            </div>
            <Dropdown className="chip" align="right" title="Camera speed" trigger={<span className="num"><Camera size={12} /> {camSpeed}</span>}
              items={[0.5, 1, 2, 4].map((s) => ({ label: `${s}×`, checked: camSpeed === s, onSelect: () => setCamSpeed(s) }))} />
          </div>

          {phase === "photo" && project?.image_url && !error && (
            <div className="vp-hint"><Play size={14} color="var(--play)" fill="var(--play)" />Press Generate to build the 3D scene from this photo.</div>
          )}
          {error && phase !== "running" && (
            <div className="vp-hint" style={{ borderColor: "#7a2a26" }}><span style={{ color: "#ffd6d2" }}>{error}</span>
              <button className="eye" onClick={() => setError(null)} title="Dismiss"><X size={13} /></button></div>
          )}
          {shading === "truth" && hasScene && (
            <div className="vp-legend"><span><i className="dot" style={{ background: "var(--observed)" }} />Seen in photo</span>
              <span><i className="dot" style={{ background: "var(--inferred)" }} />Inferred</span>
              <span><i className="dot" style={{ background: "var(--generated)" }} />AI-generated</span></div>
          )}
          {shading === "check" && hasScene && (
            <div className="vp-legend"><span><i className="dot" style={{ background: "var(--observed)" }} />Outside surface</span>
              <span><i className="dot" style={{ background: "#eb2424" }} />Back face: hole or flipped normal</span></div>
          )}
          {hasScene && <div className="vp-stats num">{stats.visible}/{stats.meshes} meshes · {stats.triangles.toLocaleString()} tris</div>}

          {phase === "running" && (
            <div className="toast" role="status" aria-live="polite">
              <h4><Loader2 size={14} className="spin" />Building scene… {pct}%</h4>
              <div className="bar"><span style={{ width: `${pct}%` }} /></div>
              <span className="muted">{job?.message ?? "Starting"}</span>
              <ul className="stage-list">
                {(job?.stages ?? []).map((s) => (
                  <li key={s.key} className={s.state}>
                    {s.state === "done" ? <Check size={12} /> : s.state === "active" ? <Loader2 size={12} className="spin" /> : s.state === "failed" || s.state === "cancelled" ? <X size={12} /> : <CircleDashed size={12} />}
                    <span>{s.name}</span>
                  </li>
                ))}
              </ul>
              <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
                <span className="muted num grow">{gpu?.cuda_available && used != null && total ? `VRAM ${used.toFixed(1)}/${total.toFixed(1)} GB` : "CPU"}</span>
                <button className="btn" disabled={cancelling} onClick={cancel}>{cancelling ? "Cancelling…" : "Cancel"}</button>
              </div>
            </div>
          )}
        </section>
        <div className="splitter" />
        <div className="sidecol">
          <Outliner meta={hasScene ? meta : null} projectName={project?.name ?? ""} selected={selected} visible={visible}
            onSelect={select} onToggle={toggle} onFocus={() => viewer.current?.focusSelection()} />
          <div className="splitter" />
          <Details meta={hasScene ? meta : null} imageUrl={project?.image_url ?? null} selected={selected} nodes={nodes}
            transform={transform} movable={!!viewer.current?.isMovable(selected)} visible={visible}
            onTransform={applyTransform} onReset={() => selected && setTransform(viewer.current?.resetTransform(selected) ?? null)} onToggle={toggle} />
        </div>
        {drawer === "content" && <ContentDrawer meta={hasScene ? meta : null} projectName={project?.name ?? ""} onClose={() => setDrawer(null)} onImport={importPhoto} />}
        {drawer === "log" && <OutputLog lines={allLogs} onClose={() => setDrawer(null)} />}
      </div>

      <footer className="statusbar">
        <button className="sb-btn" aria-pressed={drawer === "content"} onClick={() => setDrawer(drawer === "content" ? null : "content")}><FolderOpen size={13} />Content Drawer</button>
        <button className="sb-btn" aria-pressed={drawer === "log"} onClick={() => setDrawer(drawer === "log" ? null : "log")}><Terminal size={13} />Output Log</button>
        <form className="cmd" onSubmit={(e) => { e.preventDefault(); const v = cmdRef.current?.value ?? ""; if (v.trim()) runCommand(v); if (cmdRef.current) cmdRef.current.value = ""; }}>
          <span className="muted">Cmd</span>
          <input ref={cmdRef} placeholder="Enter console command (help)" aria-label="Console command" />
        </form>
        <div className="sb-right num">
          <span title="GPU"><Cpu size={12} />{gpu?.cuda_available ? `${gpu.name?.replace("NVIDIA GeForce ", "")} ${used != null && total ? `${used.toFixed(1)}/${total.toFixed(1)} GB` : ""}` : gpu ? "CPU" : "…"}</span>
          <span>{phase === "running" ? `Building ${pct}%` : "All saved"}</span>
          <button className="sb-btn" onClick={() => setDialog("models")}>Models</button>
        </div>
      </footer>

      {dialog === "models" && <ModelsDialog onClose={() => setDialog(null)} />}
      {dialog === "shortcuts" && <ShortcutsDialog onClose={() => setDialog(null)} />}
    </div>
  );
}
