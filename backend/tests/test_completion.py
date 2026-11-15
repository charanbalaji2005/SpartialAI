"""Hidden-surface completion, mesh repair, scene validation and foreground fallback."""
import numpy as np
import pytest
import trimesh
from PIL import Image

from app.depth_estimation import depth as depth_mod
from app.export import exporter
from app.geometry import completion as comp
from app.geometry import mesh as geo
from app.geometry import repair
from app.object_detection import foreground
from app.object_detection.detector import Detection
from app.scene_composition import composer
from app.scene_understanding import classifier
from app.segmentation import segmenter


def _disc_scene(h=120, w=120, r=40):
    cam = geo.make_camera(w, h, 60)
    yy, xx = np.mgrid[:h, :w]
    mask = (yy - h // 2) ** 2 + (xx - w // 2) ** 2 < r ** 2
    depth = np.full((h, w), 4.0, np.float32)
    bulge = np.sqrt(np.clip(1 - ((yy - h // 2) ** 2 + (xx - w // 2) ** 2) / r ** 2, 0, 1))
    depth[mask] = (2.0 - 0.3 * bulge)[mask]
    rgb = np.zeros((h, w, 3), np.uint8)
    rgb[...] = (60, 90, 160)
    # High-contrast "front content" (stripes, like text or a face) on the object.
    rgb[mask] = (200, 60, 50)
    stripes = mask & ((xx // 4) % 2 == 0)
    rgb[stripes] = (250, 240, 230)
    return cam, depth, rgb, mask


def _depth_cut(g, max_ratio=1.15):
    """Discontinuity mask computed here, independent of build_grid's own cut/heal policy."""
    z = -g.verts[:, 2][g.faces]
    return (z.max(1) / z.min(1)) < max_ratio


def _front(cam, depth, rgb, mask, step=2):
    g = geo.build_grid(depth, cam, step)
    sel = _depth_cut(g) & mask[g.face_px[:, 1], g.face_px[:, 0]]
    front = geo.submesh(g, sel, geo.material(Image.fromarray(rgb), "photo"))
    return front


@pytest.mark.parametrize("label,profile", [("person", "round"), ("chair", "box"),
                                            ("painting", "slab"), ("mystery thing", "round")])
def test_object_closure_is_closed_and_outward(label, profile):
    cam, depth, rgb, mask = _disc_scene()
    front = _front(cam, depth, rgb, mask)
    c = comp.close_object(front, mask, depth, label, rgb=rgb)
    assert c.prior.profile == profile
    r = repair.analyze([front, c.mesh])
    assert r.watertight and r.winding_consistent and r.outward_normals and r.complete, r.to_dict()
    assert r.boundary_edges == 0 and r.non_manifold_edges == 0


def test_back_lies_behind_front_on_the_same_rays():
    cam, depth, rgb, mask = _disc_scene()
    front = _front(cam, depth, rgb, mask)
    c = comp.close_object(front, mask, depth, "person", rgb=rgb)
    v = np.asarray(c.mesh.vertices)
    fz = -np.asarray(front.vertices)[:, 2].min()
    # The back reaches deeper than anything visible, and every vertex is in front of the camera.
    assert (-v[:, 2]).max() > (-np.asarray(front.vertices)[:, 2]).max() + 0.1
    assert (v[:, 2] < 0).all() and fz > 0
    # Round profile: thickest at the core, thin at the silhouette.
    assert c.thickness > 0.2


def test_object_back_stops_at_the_surface_behind_it():
    cam, depth, rgb, mask = _disc_scene()
    front = _front(cam, depth, rgb, mask)
    behind = np.full_like(depth, 2.3)                 # a wall just behind the disc (front at 1.7-2.0)
    c = comp.close_object(front, mask, depth, "chair", rgb=rgb, behind=behind)
    assert (-np.asarray(c.mesh.vertices)[:, 2]).max() <= 2.3 * 0.99 + 1e-6
    assert repair.analyze([front, c.mesh]).complete


def test_back_texture_is_synthesized_not_copied():
    cam, depth, rgb, mask = _disc_scene()
    front = _front(cam, depth, rgb, mask)
    c = comp.close_object(front, mask, depth, "person", rgb=rgb)
    tex = np.asarray(c.mesh.visual.material.baseColorTexture).astype(np.float32)
    x0, y0, x1, y1 = c.box
    crop = rgb[y0:y1, x0:x1].astype(np.float32)
    assert tex.shape[:2] == crop.shape[:2]
    inner = mask[y0:y1, x0:x1]
    # Same material colour on average...
    assert np.abs(tex[inner].mean(0) - crop[inner].mean(0)).max() < 25
    # ...but the front's detail (stripes, standing in for text or a face) is not reproduced
    # at the same place: high-frequency content of back and front is uncorrelated.
    def hp(img):
        g = img.mean(2)
        import cv2
        return g - cv2.GaussianBlur(g, (0, 0), 2.0)
    a, b = hp(tex)[inner], hp(crop)[inner]
    corr = float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum() + 1e-6))
    assert abs(corr) < 0.3, corr
    # The material's detail is still present (not a flat colour).
    assert a.std() > 0.25 * b.std()


def test_background_closure_is_closed_with_bridges_and_no_flare():
    cam, depth, rgb, _ = _disc_scene()
    g = geo.build_grid(depth, cam, 3)
    ok = _depth_cut(g)
    assert not ok.all()                              # the disc edge is a depth discontinuity
    mat = geo.material(Image.fromarray(rgb), "photo")
    surface = geo.submesh(g, np.ones(len(g.faces), bool), mat)
    observed = geo.submesh(g, ok, mat)
    c = comp.close_background(surface, depth, rgb=rgb, occluded=~ok)
    r = repair.analyze([observed, c.mesh])
    assert r.complete and r.components == 1, r.to_dict()
    # Back vertices sit on the same camera rays as the front, always deeper, and close behind it
    # (a relief shell), so the border walls are short instead of flaring to a far plane.
    sv = np.asarray(surface.vertices)
    back = np.asarray(c.mesh.vertices)
    zb_max = (-back[:, 2]).max()
    assert zb_max < depth.max() * 1.4
    assert np.ptp(back[:, 0]) <= np.ptp(sv[:, 0]) * 1.4
    assert "relief shell" in c.method


def test_object_occluded_faces_become_bridges():
    cam = geo.make_camera(60, 60, 60)
    depth = np.full((60, 60), 2.0, np.float32)
    depth[:, 30:] = 2.6                                # a step inside the object
    mask = np.zeros((60, 60), bool)
    mask[10:50, 10:50] = True
    g = geo.build_grid(depth, cam, 2)
    ok = _depth_cut(g)
    inside = mask[g.face_px[:, 1], g.face_px[:, 0]]
    mat = geo.material(Image.new("RGB", (60, 60)), "m")
    surface = geo.submesh(g, inside, mat)
    observed = geo.submesh(g, inside & ok, mat)
    occluded = ~ok[inside]
    assert occluded.any()
    c = comp.close_object(surface, mask, depth, "cabinet", occluded=occluded)
    r = repair.analyze([observed, c.mesh])
    assert r.complete and r.components == 1, r.to_dict()


def test_bowtie_vertices_are_removed_before_closing():
    # Two patches of surface that touch at a single grid vertex (a "bow-tie"), as happens
    # where depth-discontinuity cuts meet diagonally.
    cam = geo.make_camera(13, 13, 60)
    depth = np.full((13, 13), 2.0, np.float32)
    g = geo.build_grid(depth, cam, 1)
    cx, cy = g.face_px[:, 0], g.face_px[:, 1]
    sel = ((cx < 6) & (cy < 6)) | ((cx >= 6) & (cy >= 6))
    mat = geo.material(Image.new("RGB", (13, 13)), "m")
    mask = np.ones((13, 13), bool)

    naive = geo.submesh(g, sel, mat)
    bad = repair.analyze([naive, comp.close_object(naive, mask, depth, "box").mesh])
    assert bad.non_manifold_edges > 0          # the defect this guards against

    fixed_sel, removed = repair.manifold_selection(g.faces, sel, len(g.verts))
    assert removed > 0
    front = geo.submesh(g, fixed_sel, mat)
    good = repair.analyze([front, comp.close_object(front, mask, depth, "box").mesh])
    assert good.complete, good.to_dict()


def test_analyze_detects_holes_and_repair_single_fills_them():
    box = trimesh.creation.box()
    assert repair.analyze([box]).watertight
    broken = trimesh.Trimesh(box.vertices, box.faces[:-1], process=False)
    r = repair.analyze([broken])
    assert not r.watertight and r.boundary_edges == 3 and "open boundary" in r.issues[0]
    fixed, actions = repair.repair_single(broken)
    assert repair.analyze([fixed]).watertight and actions


def test_prior_lookup():
    assert comp.prior_for("person").profile == "round"
    assert comp.prior_for("dining table").profile == "box"
    assert comp.prior_for("televisions").profile == "slab"
    assert comp.prior_for("zxq") is comp.DEFAULT_PRIOR


def test_every_scene_type_searches_for_people():
    for cat in classifier.CATEGORIES:
        info = classifier._info(cat, 0.5, [], "test")
        assert "person" in info.thing_prompts, cat
    assert classifier._info("poster", 0.5, [], "test").group == "people"


def test_foreground_proposals():
    h, w = 120, 160
    disp = np.full((h, w), 0.2, np.float32)
    disp[30:100, 20:60] = 0.9
    disp[40:110, 100:140] = 0.85
    props = foreground.propose(disp, max_objects=4)
    assert len(props) == 2 and all(p.mask_method == "depth-foreground" for p in props)
    assert props[0].mask[60, 40] and props[1].mask[70, 120]
    assert foreground.propose(np.full((h, w), 0.5, np.float32)) == []


def _scene(scene_rgb):
    h, w = scene_rgb.shape[:2]
    norm = np.repeat(np.linspace(0.1, 0.6, h, dtype=np.float32)[:, None], w, 1)
    norm[int(h * 0.35):int(h * 0.8), int(w * 0.38):int(w * 0.62)] = 0.95
    depth = depth_mod.to_depth(norm, 1.0, 8.0)
    det = Detection("chair", 0.9, [w * 0.33, h * 0.3, w * 0.67, h * 0.85], "thing")
    segmenter.segment_grabcut(scene_rgb, det, norm)
    return depth, det, geo.make_camera(w, h, 60)


def test_compose_produces_closed_scene_and_survives_export(scene_rgb, tmp_path):
    depth, det, cam = _scene(scene_rgb)
    stages = []
    res = composer.compose(scene_rgb, depth, cam, 3, [det], [],
                           progress=lambda k, m: stages.append(k))
    order = list(dict.fromkeys(stages))
    assert order == ["geometry", "objects", "completion", "materials", "repair", "validation"]
    rep = res.completeness
    assert rep["passed"], rep["issues"]
    assert rep["objects_complete"] == rep["objects_total"] == 1 and rep["background_closed"]
    names = {n["node"]: n for n in res.nodes}
    assert names["bg_shell"]["status"] == "inferred" and names["bg_shell"]["closed"]
    assert names["obj_01_chair_back"]["profile"] == "box" and names["obj_01_chair"]["closed"]

    glb = tmp_path / "scene.glb"
    exporter.export_glb(res.scene, glb)
    check = exporter.validate_glb(glb, res.groups)
    assert check["all_closed"], check["elements"]
    assert check["normals"]


def test_failed_closure_is_regenerated(scene_rgb, monkeypatch):
    depth, det, cam = _scene(scene_rgb)
    real = comp.close_object

    def broken(*a, **kw):
        c = real(*a, **kw)
        c.mesh = trimesh.Trimesh(c.mesh.vertices, c.mesh.faces[: len(c.mesh.faces) // 2], process=False,
                                 visual=trimesh.visual.TextureVisuals(uv=c.mesh.visual.uv,
                                                                      material=c.mesh.visual.material))
        return c

    monkeypatch.setattr(comp, "close_object", broken)
    res = composer.compose(scene_rgb, depth, cam, 3, [det], [])
    el = next(e for e in res.completeness["elements"] if e["kind"] == "object")
    assert el["status"] == "repaired" and el["complete"]
    assert any("regenerated" in r for r in el["repairs"])
    back = next(n for n in res.nodes if n["node"].endswith("_back"))
    assert "convex closure" in back["method"]


def test_face_detection_grows_into_whole_person():
    h, w = 120, 120
    disp = np.full((h, w), 0.15, np.float32)
    disp[20:120, 35:85] = 0.8                       # the figure (head to bottom of frame)
    face = Detection("face", 0.7, [50, 25, 70, 50], "thing")
    face.mask = np.zeros((h, w), bool)
    face.mask[28:48, 52:68] = True
    hand = Detection("hand", 0.5, [40, 90, 50, 100], "thing")
    hand.mask = np.zeros((h, w), bool)
    hand.mask[90:100, 40:50] = True
    out = segmenter.merge_parts_and_grow([face, hand], disp)
    person = out[0]
    assert person.label == "person" and "depth-layer" in person.mask_method
    assert person.mask[110, 60] and person.mask[25, 40] and not person.mask[60, 10]
    # the hand belongs to the same figure: it no longer exists as a separate fragment
    assert all(not (d.mask & person.mask).any() for d in out[1:])
