# Reconstruction pipeline

Code: `backend/app/services/pipeline.py`, `scene_composition/composer.py`, `geometry/mesh.py`.

## 1. Image analysis
EXIF orientation is applied, transparency is flattened onto white, and the image is resized so its long side matches the quality mode (512/768/1024 px). The horizontal field of view comes from the EXIF 35 mm-equivalent focal length (`hfov = 2·atan(36 / 2f₃₅)`) or defaults to 60°. CLIP scores 17 scene descriptions; the winner's group (indoor, outdoor, object) picks the detection vocabulary and the assumed depth range. Without CLIP a colour heuristic decides.

## 2. Depth
Depth Anything V2 predicts relative disparity (larger = nearer). It is upsampled to the image size, robustly normalised to [0, 1] using the 1st and 99.5th percentiles, and smoothed with a guided filter that uses the photo as a guide so object edges stay sharp. Normalised disparity `d` becomes depth with `z = 1 / (1/far + d·(1/near − 1/far))`, which preserves the inverse relationship between disparity and distance. On a CUDA out-of-memory error, depth reruns at half resolution.

## 3. Detection and segmentation
Grounding DINO finds "things" (sofa, car, tree…) and "stuff" (wall, floor, sky…) with per-label non-maximum suppression. SAM segments each thing from its box, reusing one image embedding for all boxes and batching boxes to bound memory. Without SAM, GrabCut segments from the box and pixels much farther than the object's median depth are removed. Stuff regions use colour similarity within the box. Masks are cleaned (open/close, largest component), overlapping duplicates are removed, and the most confident objects are kept up to the mode's limit.

## 4. Geometry
A vertex grid samples every N pixels (N = 4/3/2 by mode). Each vertex is unprojected through a pinhole camera at the origin looking down −Z (glTF convention): `X = (u − cx)·z/f`, `Y = −(v − cy)·z/f`, `Z = −z`. Each grid cell becomes two triangles. A triangle is dropped when its depth ratio exceeds `1.06^N`, which removes the "rubber sheet" that otherwise stretches between a foreground object and the wall behind it.

## 5. Inferred background
Pixels under object masks (dilated) are inpainted in colour and depth (Telea). The filled depth is clamped to be no nearer than the original, since the hidden background lies behind the object. Background triangles inside that region become the `bg_fill` node, marked inferred and textured with the inpainted image.

## 6. Objects and hidden surfaces
Code: `geometry/completion.py`. Each object's visible surface comes from the original depth and photo (`observed`). Faces at the silhouette that reach background depth (masks are a few pixels wider than depth edges) are trimmed, and bow-tie vertices are removed so the surface is manifold. Faces spanning a depth discontinuity are kept as *occlusion bridges* (inferred, synthesized texture) so the surface has no internal holes.

The hidden back is inferred from a **category shape prior** (`prior_for`): `round` (people, animals, plants, vehicles, bottles) gets an elliptical thickness that falls to a thin rim at the silhouette; `box` (furniture, cabinets, buildings) gets a flat back plane; `slab` (paintings, doors, screens, rugs) gets a thin constant thickness. Thickness = prior depth/width ratio × the median silhouette row width (ignores protrusions like outstretched arms). Back vertices lie on the same camera rays as the front, behind it, and never behind the background surface at that pixel. Side walls stitch every boundary to the back, so front + back + sides form one watertight, consistently wound, outward-facing shell.

The back gets its **own synthesized texture**: the object's low-frequency colour field (masked blur) plus high-frequency detail re-assembled from randomly placed patches of the visible material. Material and colour match the front; recognisable front content (faces, text) is not reproduced.

In Best mode with Stable Fast 3D configured, the masked crop is reconstructed by the model instead (`generated`) and repaired (`repair_single`). If the detector finds no objects at all, near-layer components of the depth map are proposed as objects (`object_detection/foreground.py`) and refined with SAM.

## 6b. Background completion
The background depth is despiked (grey-scale open/close on disparity) so isolated near/far pixels don't become needles. The whole background then becomes a **relief shell** (`close_background`): its back follows a smoothed copy of the front at ~12% of the median distance, on the same camera rays, joined by short border walls. It cannot self-intersect, and the scene has no empty side from any direction (`bg_shell`, inferred).

## 7. Composition, lighting and ground
A RANSAC plane fit on the lower part of the background (normal close to +Y) gives an inferred ground plane, hidden by default. Key light colour and direction are taken from the brightest 2% of pixels, and ambient level from mean brightness; the viewer's Lit mode uses them.

## 8. Repair and validation
Code: `geometry/repair.py`. Every element (each object, the background) is checked as a closed shell: open boundary edges, non-manifold edges, winding consistency, outward normals (positive volume), degenerate faces, components, finite normals, valid UVs, texture presence. An object whose closure fails is regenerated as a convex closure; a single generated mesh is repaired (weld, fill holes, fix winding). The ground plane is a support plane and is open by design. Results go to `completeness.json` and `report.json["completeness"]`; warnings appear in the job log.

## 9. Export
trimesh writes the scene as GLB with normals and as glTF in a zip. The GLB is reloaded and every element is re-checked for closedness (`report.json["export"]["elements"]`). Each object is also exported alone. `metadata.json` lists every node with its status, method, bounds, confidence, mask information and whether it belongs to a closed shell; `report.json` holds the quality report.
