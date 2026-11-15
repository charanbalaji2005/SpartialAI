"""GLB / glTF / per-object exports, with post-export verification."""
from __future__ import annotations

import io
import zipfile
from pathlib import Path

import trimesh

from ..geometry import repair


def export_glb(scene: trimesh.Scene, path: Path) -> int:
    data = scene.export(file_type="glb", include_normals=True)
    path.write_bytes(data)
    return len(data)


def export_gltf_zip(scene: trimesh.Scene, path: Path) -> int:
    """glTF is multi-file (.gltf + .bin + textures); deliver it as a zip."""
    files = scene.export(file_type="gltf", include_normals=True)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in files.items():
            zf.writestr(name, data)
    path.write_bytes(buf.getvalue())
    return path.stat().st_size


def _node_meshes(loaded: trimesh.Scene) -> dict[str, trimesh.Trimesh]:
    """Map node name -> mesh in world coordinates."""
    out: dict[str, trimesh.Trimesh] = {}
    for node in loaded.graph.nodes_geometry:
        transform, geom_name = loaded.graph[node]
        geom = loaded.geometry.get(geom_name)
        if geom is None or not hasattr(geom, "faces"):
            continue
        m = geom.copy()
        m.apply_transform(transform)
        out[node] = m
    return out


def validate_glb(path: Path, groups: dict[str, list[str]] | None = None) -> dict:
    """Reload the GLB, confirm it parses, and re-check that every element is still closed.

    `groups` maps element name -> node names that together must form a closed
    shell (as produced by the composer). Exporting can split vertices along UV
    seams; the check welds them by position before testing.
    """
    loaded = trimesh.load(path, force="scene")
    result = {
        "valid": True,
        "geometries": len(loaded.geometry),
        "faces": int(sum(len(g.faces) for g in loaded.geometry.values() if hasattr(g, "faces"))),
        "bytes": path.stat().st_size,
        "normals": all(getattr(g, "vertex_normals", None) is not None
                       for g in loaded.geometry.values() if hasattr(g, "faces")),
    }
    if groups:
        meshes = _node_meshes(loaded)
        reloaded = {}
        for name, node_names in groups.items():
            parts = [meshes[n] for n in node_names if n in meshes]
            missing = [n for n in node_names if n not in meshes]
            rep = repair.analyze(parts)
            reloaded[name] = {"closed": rep.watertight and rep.winding_consistent and rep.outward_normals,
                              "boundary_edges": rep.boundary_edges, "missing_nodes": missing}
        result["elements"] = reloaded
        result["all_closed"] = all(v["closed"] and not v["missing_nodes"] for v in reloaded.values())
    return result
