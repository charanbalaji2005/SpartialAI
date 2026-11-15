# Models

| Name | Hugging Face id | Size | License |
|---|---|---|---|
| Depth Anything V2 Small | depth-anything/Depth-Anything-V2-Small-hf | ~100 MB | Apache-2.0 |
| CLIP ViT-B/32 | openai/clip-vit-base-patch32 | ~600 MB | MIT |
| Grounding DINO Tiny | IDEA-Research/grounding-dino-tiny | ~700 MB | Apache-2.0 |
| SAM ViT-Base | facebook/sam-vit-base | ~375 MB | Apache-2.0 |
| Stable Fast 3D (optional) | stabilityai/stable-fast-3d | ~4 GB incl. deps | Stability AI Community License |

Sizes are approximate. Swap models in `.env`: for example `SPATIALAI_DEPTH_MODEL=depth-anything/Depth-Anything-V2-Base-hf` gives sharper depth but uses more memory, and the Base and Large depth checkpoints are licensed CC-BY-NC-4.0 (non-commercial). Check each license against your use.

Weights download to `models/huggingface`. After `scripts\download_models.py` succeeds you can work offline; set `HF_HUB_OFFLINE=1` to stop Transformers from checking for updates.

## Stable Fast 3D (optional, Best mode)

Stable Fast 3D needs about 6 GB VRAM at its default settings, which is tight on an 8 GB card, so SpatialAI unloads all its own models before calling it. Its Windows support is experimental upstream.

1. Accept the model license at https://huggingface.co/stabilityai/stable-fast-3d and run `huggingface-cli login` with a read token.
2. Install Visual Studio 2022 with the "Desktop development with C++" workload.
3. In a new PowerShell window from the SpatialAI folder:
   ```powershell
   git clone https://github.com/Stability-AI/stable-fast-3d.git
   py -3.11 -m venv sf3d-env
   .\sf3d-env\Scripts\Activate.ps1
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
   pip install setuptools==69.5.1 wheel
   cd stable-fast-3d
   pip install -r requirements.txt
   python run.py demo_files/examples/chair1.png --output-dir output/
   ```
   Follow the repository README if its instructions have changed. The last command must produce a GLB before SpatialAI can use it.
4. In `.env` set
   ```
   SPATIALAI_SF3D_PYTHON=C:\path\to\SpatialAI\sf3d-env\Scripts\python.exe
   SPATIALAI_SF3D_REPO=C:\path\to\SpatialAI\stable-fast-3d
   ```
5. Restart SpatialAI. The GPU panel should show "Object reconstruction: Configured". Best mode then generates objects; any per-object failure falls back to the mirrored relief and is listed in the report.

TripoSR (MIT) can replace Stable Fast 3D with the same pattern: adapt `backend/app/object_reconstruction/sf3d.py` to call its `run.py` and load its output mesh.

## Citations

- Yang et al., "Depth Anything V2", NeurIPS 2024.
- Radford et al., "Learning Transferable Visual Models From Natural Language Supervision" (CLIP), ICML 2021.
- Liu et al., "Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection", ECCV 2024.
- Kirillov et al., "Segment Anything", ICCV 2023.
- Boss et al., "SF3D: Stable Fast 3D Mesh Reconstruction with UV-unwrapping and Illumination Disentanglement", 2024.
- Telea, "An Image Inpainting Technique Based on the Fast Marching Method", 2004.
- Rother et al., "GrabCut: Interactive Foreground Extraction using Iterated Graph Cuts", SIGGRAPH 2004.

Please verify these references before citing them in a report.
