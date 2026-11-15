import type { SceneNode } from "./api";

const title = (s: string) => s.replace(/[_-]+/g, " ").replace(/\b\w/g, (c) => c.toUpperCase()).replace(/\s+/g, "");

/** Editor-style actor names: SM_Chair_01, SM_Chair_01_Back, SM_Wall ... */
export function actorName(n: SceneNode): string {
  if (n.kind === "object") {
    const m = n.group.match(/^obj_(\d+)_/);
    const base = `SM_${title(n.label)}_${m ? m[1] : "00"}`;
    return n.node.endsWith("_back") ? `${base}_Back` : n.node.endsWith("_gen") ? `${base}_Gen` : base;
  }
  if (n.node === "bg_fill") return "SM_HiddenBackground";
  if (n.node === "ground_plane") return "SM_GroundPlane";
  return `SM_${title(n.label)}`;
}

export function objectName(nodes: SceneNode[]): string {
  const main = nodes.find((n) => n.status === "observed") ?? nodes[0];
  const m = main.group.match(/^obj_(\d+)_/);
  return `${title(main.label)}_${m ? m[1] : "00"}`;
}

export const TYPE_LABEL: Record<SceneNode["status"], string> = {
  observed: "StaticMesh", inferred: "InferredMesh", generated: "GeneratedMesh",
};

export function sourceLabel(n: SceneNode): string {
  if (n.status === "generated") return "Stable Fast 3D";
  if (n.status === "inferred") return n.method?.split(" ")[0] ?? "Inferred";
  return n.mask_method === "segformer" ? "SegFormer" : n.kind === "object" ? (n.mask_method === "sam" ? "SAM" : "GrabCut") : "Photo";
}
