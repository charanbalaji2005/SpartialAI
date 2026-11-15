import numpy as np
import trimesh

from app.depth_estimation import depth as depth_mod
from app.export import exporter
from app.geometry import mesh as geo
from app.object_detection.detector import Detection, iou, nms
from app.scene_composition import composer
from app.segmentation import segmenter


def _depth_for(rgb):
    h, w = rgb.shape[:2]
    norm = np.repeat(np.linspace(0.1, 0.6, h, dtype=np.float32)[:, None], w, 1)
    norm[int(h * 0.35):int(h * 0.8), int(w * 0.38):int(w * 0.62)] = 0.95   # box is near
    return norm


def test_grabcut_segments_object(scene_rgb):
    h, w = scene_rgb.shape[:2]
    det = Detection("box", 0.9, [w * 0.33, h * 0.3, w * 0.67, h * 0.85], "thing")
    segmenter.segment_grabcut(scene_rgb, det, _depth_for(scene_rgb))
    truth = np.zeros((h, w), bool)
    truth[int(h * 0.35):int(h * 0.8), int(w * 0.38):int(w * 0.62)] = True
    inter = (det.mask & truth).sum() / (det.mask | truth).sum()
    assert det.mask_method == "grabcut" and inter > 0.8


def test_compose_and_export(scene_rgb, tmp_path):
    h, w = scene_rgb.shape[:2]
    norm = _depth_for(scene_rgb)
    depth = depth_mod.to_depth(norm, 1.0, 8.0)
    det = Detection("box", 0.9, [w * 0.33, h * 0.3, w * 0.67, h * 0.85], "thing")
    segmenter.segment_grabcut(scene_rgb, det, norm)
    floor = Detection("floor", 0.6, [0, h * 0.6, w - 1, h - 1], "stuff")
    segmenter.segment_grabcut(scene_rgb, floor, norm)
    cam = geo.make_camera(w, h, 60)
    res = composer.compose(scene_rgb, depth, cam, 3, [det], [floor])

    names = {n["node"]: n for n in res.nodes}
    assert "obj_01_box" in names and names["obj_01_box"]["status"] == "observed"
    assert names["obj_01_box_back"]["status"] == "inferred"
    assert names["bg_fill"]["status"] == "inferred"
    assert "bg_floor" in names
    # The object sits in front of the inferred background behind it.
    assert names["obj_01_box"]["center"][2] > names["bg_fill"]["center"][2]
    assert 0 < res.stats["observed_ratio"] < 1

    glb = tmp_path / "scene.glb"
    exporter.export_glb(res.scene, glb)
    check = exporter.validate_glb(glb)
    assert check["valid"] and check["geometries"] == len(res.nodes)
    loaded = trimesh.load(glb, force="scene")
    assert "obj_01_box" in loaded.graph.nodes
    exporter.export_gltf_zip(res.scene, tmp_path / "scene.zip")
    import zipfile
    names_in_zip = zipfile.ZipFile(tmp_path / "scene.zip").namelist()
    assert any(n.endswith(".gltf") for n in names_in_zip)


def test_nms_and_iou():
    a = Detection("chair", 0.9, [0, 0, 10, 10], "thing")
    b = Detection("chair", 0.5, [1, 1, 10, 10], "thing")
    c = Detection("table", 0.4, [1, 1, 10, 10], "thing")
    assert iou(a.box, b.box) > 0.6
    kept = nms([b, a, c])
    assert [d.score for d in kept] == [0.9, 0.4]
