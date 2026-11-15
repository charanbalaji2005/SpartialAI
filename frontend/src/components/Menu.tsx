import { useEffect, useLayoutEffect, useRef, useState, type ReactNode } from "react";
import { Check } from "lucide-react";

export type MenuItem =
  | { type?: "item"; label: string; onSelect: () => void; shortcut?: string; disabled?: boolean; checked?: boolean; icon?: ReactNode }
  | { type: "sep" }
  | { type: "heading"; label: string };

/** A dropdown anchored to its trigger (menu bar menus and viewport chips). */
export function Dropdown({ trigger, items, className, title, align = "left" }: {
  trigger: ReactNode; items: MenuItem[]; className?: string; title?: string; align?: "left" | "right";
}) {
  const [open, setOpen] = useState(false);
  const btn = useRef<HTMLButtonElement>(null);
  const pop = useRef<HTMLDivElement>(null);
  const [pos, setPos] = useState({ top: 0, left: 0 });

  useLayoutEffect(() => {
    if (!open || !btn.current) return;
    const r = btn.current.getBoundingClientRect();
    const width = pop.current?.offsetWidth ?? 220;
    setPos({ top: r.bottom + 2, left: Math.max(4, Math.min(align === "right" ? r.right - width : r.left, window.innerWidth - width - 4)) });
  }, [open, align]);

  useEffect(() => {
    if (!open) return;
    const close = (e: MouseEvent) => { if (!pop.current?.contains(e.target as Node) && !btn.current?.contains(e.target as Node)) setOpen(false); };
    const esc = (e: KeyboardEvent) => { if (e.key === "Escape") setOpen(false); };
    window.addEventListener("mousedown", close);
    window.addEventListener("keydown", esc);
    return () => { window.removeEventListener("mousedown", close); window.removeEventListener("keydown", esc); };
  }, [open]);

  return (
    <>
      <button ref={btn} className={className} title={title} aria-haspopup="menu" aria-expanded={open} onClick={() => setOpen((o) => !o)}>{trigger}</button>
      {open && (
        <div ref={pop} className="popover" role="menu" style={pos}>
          {items.map((it, i) => it.type === "sep" ? <div key={i} className="sep" /> :
            it.type === "heading" ? <div key={i} className="heading">{it.label}</div> : (
              <button key={i} role="menuitem" disabled={it.disabled} onClick={() => { setOpen(false); it.onSelect(); }}>
                <span>{it.checked ? <Check size={13} /> : it.icon}</span>
                <span>{it.label}</span>
                <span className="kbd">{it.shortcut}</span>
              </button>
            ))}
        </div>
      )}
    </>
  );
}

export function MenuBar({ menus, title }: { menus: { label: string; items: MenuItem[] }[]; title: string }) {
  return (
    <nav className="menubar" aria-label="Main menu">
      <img className="logo" src="/favicon.svg" alt="SpatialAI" />
      {menus.map((m) => <Dropdown key={m.label} className="menu-btn" trigger={m.label} items={m.items} />)}
      <span className="title">{title}</span>
    </nav>
  );
}
