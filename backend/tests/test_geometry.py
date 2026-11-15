import numpy as np
from PIL import Image

from app.depth_estimation import depth as depth_mod
from app.geometry import mesh as geo


def test_disparity_to_depth_monotonic():
    norm = np.linspace(0, 1, 11, dtype=np.float32)[None, :]
    d = depth_mod.to_depth(norm, near=1.0, far=10.0)
    assert np.isclose(d[0, 0], 10.0) and np.isclose(d[0, -1], 1.0)
    assert np.all(np.diff(d[0]) < 0)


def test_refine_handles_nan(scene_rgb):
    disp = np.random.default_rng(0).random(scene_rgb.shape[:2]).astype(np.float32)
    disp[0, 0] = np.nan
    norm, stats = depth_mod.refine_disparity(disp, scene_rgb)
    assert np.isfinite(norm).all() and norm.min() >= 0 and norm.max() <= 1
    assert stats["invalid_fraction"] > 0


def test_unprojection_center_ray():
    cam = geo.make_camera(101, 81, 60)
    depth = np.full((81, 101), 2.0, np.float32)
    g = geo.build_grid(depth, cam, step=1)
    center = g.verts[40 * 101 + 50]
    assert np.allclose(center, [0, 0, -2.0], atol=1e-4)
    # Flat plane: no faces should be cut as discontinuities.
    assert g.face_ok.all()


def test_discontinuity_cut():
    cam = geo.make_camera(40, 40, 60)
    depth = np.full((40, 40), 5.0, np.float32)
    depth[:, 20:] = 1.0        # sharp step
    g = geo.build_grid(depth, cam, step=2)
    assert not g.face_ok.all() and g.face_ok.mean() > 0.8


def test_exif_fov():
    img = Image.new("RGB", (10, 10))
    exif = img.getexif()
    exif[0xA405] = 26  # 26 mm equivalent ≈ 69° horizontal
    import io
    buf = io.BytesIO()
    img.save(buf, format="JPEG", exif=exif)
    fov, src = geo.fov_from_exif(Image.open(buf))
    assert src == "exif" and 65 < fov < 72


def test_ground_plane_ransac():
    rng = np.random.default_rng(1)
    pts = np.column_stack([rng.uniform(-2, 2, 500), np.full(500, -1.5), rng.uniform(-6, -1, 500)])
    pts += rng.normal(0, 0.005, pts.shape)
    n, d, ratio = geo.fit_ground_plane(pts)
    assert n[1] > 0.99 and abs(d - 1.5) < 0.05 and ratio > 0.9
