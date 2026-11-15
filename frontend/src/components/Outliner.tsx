import { useMemo, useState } from "react";
import { ChevronDown, ChevronRight, Eye, EyeOff, Folder, Box, Mountain, Search } from "lucide-react";
import type { SceneMeta, SceneNode } from "../lib/api";
import { actorName, objectName, sourceLabel, TYPE_LABEL } from "../lib/naming";

interface Props {
  meta: SceneMeta | null; projectName: string; selected: string | null; visible: Record<string, boolean>;
  onSelect: (key: string | null) => void; onToggle: (nodes: string[], on: boolean) => void; onFocus: () => void;
}

export default function Outliner({ meta, projectName, selected, visible, onSelect, onToggle, onFocus }: Props) {
  const [query, setQuery] = useState("");
  const [open, setOpen] = useState<Record<string, boolean>>({ env: true, objects: true });
  const q = query.trim().toLowerCase();

  const { regions, objects } = useMemo(() => {
    const regions = (meta?.nodes ?? []).filter((n) => n.kind === "region");
    const groups = new Map<string, SceneNode[]>();
    (meta?.nodes ?? []).filter((n) => n.kind === "object").forEach((n) => groups.set(n.group, [...(groups.get(n.group) ?? []), n]));
    return { regions, objects: [...groups.entries()] };
  }, [meta]);
  const match = (n: SceneNode) => !q || actorName(n).toLowerCase().includes(q) || n.label.toLowerCase().includes(q);
  const actors = meta?.nodes.length ?? 0;

  const EyeBtn = ({ nodes }: { nodes: string[] }) => {
    const on = nodes.some((n) => visible[n]);
    return <button className="eye" title={on ? "Hide in viewport (H)" : "Show"} onClick={(e) => { e.stopPropagation(); onToggle(nodes, !on); }}>
      {on ? <Eye size={13} /> : <EyeOff size={13} />}</button>;
  };
  const Caret = ({ k }: { k: string }) => (
    <button className="caret" onClick={(e) => { e.stopPropagation(); setOpen((o) => ({ ...o, [k]: !o[k] })); }} aria-label={open[k] ? "Collapse" : "Expand"}>
      {open[k] ? <ChevronDown size={12} /> : <ChevronRight size={12} />}</button>
  );
  const indent = (lvl: number) => ({ paddingLeft: 4 + lvl * 14 });

  return (
    <section className="panel" aria-label="Outliner">
      <div className="panel-tabs"><button className="panel-tab" aria-selected="true">Outliner</button></div>
      <div className="search"><div className="field"><Search size={13} className="muted" />
        <input placeholder="Search…" value={query} onChange={(e) => setQuery(e.target.value)} aria-label="Search actors" /></div></div>
      <div className="ol-head"><span /><span>Item label</span><span>Source</span><span>Type</span></div>
      <div className="panel-body" role="tree">
        {!meta ? <p className="empty-note">The scene appears here after you generate it.</p> : (
          <>
            <div className="ol-row"><span /><span className="label" style={indent(0)}><Folder size={13} /><span>{projectName} (Scene)</span></span><span className="cell" /><span className="cell">World</span></div>
            <div className="ol-row" onClick={() => setOpen((o) => ({ ...o, env: !o.env }))}>
              <EyeBtn nodes={regions.map((r) => r.node)} />
              <span className="label" style={indent(1)}><Caret k="env" /><Folder size={13} color="#d3a24a" /><span>Environment</span></span>
              <span className="cell" /><span className="cell">Folder</span>
            </div>
            {open.env && regions.filter(match).map((n) => (
              <div key={n.node} className={`ol-row${selected === (n.group || n.node) ? " sel" : ""}${!visible[n.node] ? " hidden" : ""}`}
                onClick={() => onSelect(n.group || n.node)} onDoubleClick={onFocus} role="treeitem" aria-selected={selected === (n.group || n.node)}>
                <EyeBtn nodes={[n.node]} />
                <span className="label" style={indent(2.6)}><Mountain size={12} color={`var(--${n.status})`} /><span>{actorName(n)}</span></span>
                <span className="cell">{sourceLabel(n)}</span><span className="cell">{TYPE_LABEL[n.status]}</span>
              </div>
            ))}
            <div className="ol-row" onClick={() => setOpen((o) => ({ ...o, objects: !o.objects }))}>
              <EyeBtn nodes={objects.flatMap(([, ns]) => ns.map((n) => n.node))} />
              <span className="label" style={indent(1)}><Caret k="objects" /><Folder size={13} color="#d3a24a" /><span>Objects ({objects.length})</span></span>
              <span className="cell" /><span className="cell">Folder</span>
            </div>
            {open.objects && objects.filter(([, ns]) => ns.some(match)).map(([g, ns]) => (
              <div key={g}>
                <div className={`ol-row${selected === g ? " sel" : ""}${!ns.some((n) => visible[n.node]) ? " hidden" : ""}`}
                  onClick={() => onSelect(g)} onDoubleClick={onFocus} role="treeitem" aria-selected={selected === g}>
                  <EyeBtn nodes={ns.map((n) => n.node)} />
                  <span className="label" style={indent(2)}><Caret k={g} /><Box size={12} color="#9cc3ff" /><span>{objectName(ns)}</span></span>
                  <span className="cell">{ns[0].confidence != null ? `${Math.round(ns[0].confidence * 100)}% conf.` : ""}</span>
                  <span className="cell">Actor</span>
                </div>
                {open[g] && ns.map((n) => (
                  <div key={n.node} className={`ol-row${selected === g ? " sel" : ""}${!visible[n.node] ? " hidden" : ""}`} onClick={() => onSelect(g)}>
                    <EyeBtn nodes={[n.node]} />
                    <span className="label" style={indent(3.6)}><span className="dot" style={{ background: `var(--${n.status})` }} /><span>{actorName(n)}</span></span>
                    <span className="cell">{sourceLabel(n)}</span><span className="cell">{TYPE_LABEL[n.status]}</span>
                  </div>
                ))}
              </div>
            ))}
          </>
        )}
      </div>
      <div className="ol-foot num">{actors} actors{selected ? " (1 selected)" : ""}</div>
    </section>
  );
}
