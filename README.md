# Reconstruction Experiments

A comparison harness for turning a tracked iPhone capture (video, ARKit poses, LiDAR depth and confidence) into a room mesh usable for depth masking in MikanXR. Each method produces per-frame depth, the same TSDF fuses it, and the same held-out LiDAR frames score the result.

## Decision

The LiDAR-only path is the one MikanXR will use: ARKit `sceneDepth` and its confidence, recorded by the streamer, fused by the TSDF in `recon/fusion.py`. Over ten captures no depth model beat it against held-out sensor frames, the visible gains (crisper small-object outlines at fine voxels) come from depth resolution rather than better geometry, and the end use renders the mesh as a coarse prior that image-space refinement snaps to the video's edges, which the LiDAR mesh is already accurate enough to seed. It fuses in seconds with no model, no torch, and no GPU memory.

What follows from that:

- Pose is now the dominant silhouette error at room scale (the LiDAR mesh against its own held-out frames: 1.75 cm MAE, 11 px edge distance). Refining the sweep's poses and registering the live camera well matter more than any depth model.
- Thin structures (chair legs) and moving objects are absent from every static mesh and image-space refinement cannot invent them. Per-frame depth at runtime is the only route to those, and that is a different architecture from an offline mesh.
- `promptda` stays available as an optional offline refinement (`promptda.conf0_weight: 0`) for close-range hero objects if refined edges from the LiDAR prior prove too soft. It is a Python preprocessing step, not a candidate for an in-editor port.
- `mapanything`, `da3`, and `milo` are retired from the comparison.
- An object segmenter prompted from the mesh silhouette fixes what no band method can: SAM2 small with positive points per occluding contour run puts the marked edge at 1.0 to 1.7 px median and 2.7 to 6.3 px p90 on 24 hand-marked frames across four takes, whether the mesh is 6 or 46 px off, where the band methods track the mesh's error. Details in `NOTES.md`.

The harness itself stays useful: it scores the LiDAR mesh against held-out frames, which is the measurement the pose-refinement work needs.

## Methods

| name | what it is | weights license | status |
|---|---|---|---|
| `lidar_tsdf` | raw ARKit LiDAR depth fused as-is | sensor data | production path |
| `promptda` | Prompt Depth Anything ViT-L: RGB plus the LiDAR frame as a metric prompt | Apache 2.0 | optional offline refinement |
| `da3` | Depth Anything 3 with the ARKit poses and intrinsics supplied | Base/Small Apache 2.0, Large disputed, Giant CC BY-NC | retired |
| `mapanything` | MapAnything (Apache checkpoint) with images, intrinsics, poses, and LiDAR | Apache 2.0 | retired |
| `milo` | MILo Gaussian splatting with mesh extraction, trained offline from a COLMAP export | non-commercial | retired |

## Setup

Windows, Python 3.12, an NVIDIA GPU.

```
python -m venv .venv
.venv\Scripts\python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
.venv\Scripts\python -m pip install -r requirements.txt
```

That venv runs the baseline, fusion, evaluation, and the report. Each model gets its own venv (`.venv_promptda`, `.venv_da3`, `.venv_mapanything`, `.venv_sam2`) built the same way, plus the model's own dependencies. The exact commands that worked, and the pitfalls, are at the top of each `methods/*.py`. `requirements-<method>.txt` files hold the extra packages where a method needs more than `requirements.txt`. Model repos are cloned under `third_party/` and checkpoints land in the HuggingFace cache. Run a method with its own venv's python, since the harness measures peak VRAM in-process:

```
.venv_da3\Scripts\python run.py da3
```

## Captures

Record with [Stray Scanner](https://github.com/strayrobots/scanner) and copy the capture folder into `captures/<id>/` (gitignored). The loader needs `rgb.mp4`, `depth/`, `confidence/`, and an `odometry.csv` with per-frame intrinsics.

A MikanARStreamer take loads the same way: `<take>.mp4` with its `<take>.pose.json`, `<take>.depth.bin` and `<take>.conf.bin` beside it (the format is documented in MikanXR's `docs/reference/videosources.md`). Address it by the movie, absolute or relative to `captures/`:

```
python run.py lidar_tsdf --capture D:\path\to\take_20261013_120000.mp4
```

The pose track stores raw ARKit camera transforms, which `recon.capture.load_mikan_take` rotates into the OpenCV camera axes every method expects; the Stray Scanner loader needs no such step because that app bakes the rotation in. `Capture.check_pose_consistency(a, b)` is the check for a new take: with correct poses, LiDAR reprojected between frames thirty apart agrees to a few millimetres median. `tools/check_take.py <take>.mp4` runs it over the take, and `tools/take_diagnostics.py <take>.mp4` explains a fused run's held-out error: by range and tail, against camera speed, and as start-to-end loop drift. `tools/edge_refine_eval.py <take>.mp4 --mesh <mesh.ply>` prototypes the editor's image-space edge refinement on a fused mesh and scores it against the held-out LiDAR edges, with `--sweep` over the band radius and colour threshold.

Hand-marked silhouettes are the finer measure. `tools/render_take_frames.py <take>.trackcamera <mesh.glb> --times 2,6,10` renders a mesh into chosen frames of a take offline, one folder per frame in the format `tools/mark_silhouette.py` marks and `tools/occlusion_edge_eval.py` scores, so frames can be marked without the running editor (`--poses` swaps in another pose track). The SAM2 tools run in `.venv_sam2` (recipe in `requirements-sam2.txt`): `tools/sam2_edge_eval.py` prompts SAM2 from the mesh silhouette on the marked frames and scores every prompt variant next to the band methods, `tools/sam2_track_eval.py <take>.trackcamera` tracks one prompt through the take with the video predictor and scores it on the marked frames, `tools/sam2_timing.py` times the encoder, decoder and tracker step per model size, and `tools/sam2_export_encoder.py` plus `tools/sam2_dml_timing.py` (in `.venv_sam2_dml`) time the exported encoder under DirectML. `tools/sam2_export_encoder.py` and `tools/sam2_export_decoder.py --check` are also the exports MikanXR runs (the encoder with the normalization inside, returning the predictor's image embedding and high-resolution features; the decoder with a dynamic point axis, since a padding point is not neutral to it), validated against the PyTorch predictor on the marked frames. `tools/sam2_release_bundle.py` lays them out as the per-checkpoint MikanXR/MikanSam2Onnx release folders with `manifest.json`, and `tools/score_mikan_matte.py <dump dir>` scores the labels `MikanCmd -occlusionMatte -dump` wrote against the marked frames beside the Python result, the parity check of the C++ port.

## Running

```
.venv\Scripts\python run.py lidar_tsdf
.venv\Scripts\python run.py promptda
.venv\Scripts\python run.py da3 --stage depth
.venv\Scripts\python run.py da3 --stage fuse
```

`run_all.py` runs a set of methods over every capture serially, each in its own venv, and `summarize.py` aggregates every `metrics.json` into `out/summary.md` with one table per metric (captures as rows, runs as columns):

```
.venv\Scripts\python run_all.py --methods lidar_tsdf promptda mapanything_aligned --skip-existing
```

`--stage` is one of `depth`, `fuse`, `eval`, `render`, `report`, or `all` (default). `--tag` names a variant so it gets its own output directory (`out/<capture>/<method>_<tag>/`), `--config` swaps the whole settings file (`configs/fine.yaml` fuses at 8 mm voxels for tabletop captures), and `--opt key.path=value` overrides single settings, for example `--opt da3.model=depth-anything/DA3-LARGE-1.1`. A tagged run reuses the untagged run's depth cache when the method settings match, so fusion variants do not re-run the model. Outputs land in `out/<capture>/<method>/` and the comparison table in `out/<capture>/report.md`.

## Pose refinement

`fusion.frame_to_model` in the config turns the fusion into KinectFusion's loop: before each training frame is integrated, the volume so far is ray cast from the frame's predicted pose (`DenseTSDF.raycast`) and the frame is registered against that model map by projective point-to-plane ICP (`recon/register.py`). The prediction is the ARKit pose carried forward by the previous frame's correction, so the refinement follows accumulated drift, and a prior on the deviation from that prediction (`prior_sigma_m`, `prior_sigma_deg`) holds whatever a frame's geometry leaves unconstrained, such as the directions along a wall that is all the frame sees. The refined pose of every frame, held-out frames taking their nearest training frame's correction, is saved as `poses_refined.npy` beside the mesh and used by the eval and render stages, so the mesh is scored in the frame it was built in. It is off by default:

```
python run.py lidar_tsdf --capture <take>.mp4 --opt fusion.frame_to_model.enabled=true --tag f2m
```

`tools/test_register.py` checks the ray cast and the registration on a synthetic room corner.

## How the comparison works

- Every 3rd frame is a keyframe. Every 10th keyframe is held out of fusion.
- `recon/fusion.py` is a dense GPU TSDF (1.5 cm voxels, 6 cm truncation, per-pixel confidence weights, flying-pixel cull at depth discontinuities) over the bounding box of the training LiDAR. Marching cubes extracts the mesh and small islands are removed.
- `recon/evaluate.py` ray casts each mesh from the held-out ARKit poses at LiDAR resolution and compares with that frame's confidence-2 LiDAR: silhouette edge distance, depth error, completeness, floater hits, and the fraction of mesh surface with no LiDAR support. Candidates are also compared with the baseline mesh by chamfer distance and F-score.
- `recon/render.py` writes a strip per view (video frame, shaded mesh, depth overlay) and a top-down view.

Coordinate frame: everything is in ARKit's world frame as written by Stray Scanner (gravity aligned, +Y up). Camera poses are camera-to-world with OpenCV camera axes.

## Results

Ten captures: one hotel room (`03aec96c39`, 973 training frames), one tabletop sweep (`86a1a8e3b1`, where every variant including MILo was run), and eight convention booth tabletops with screens, glass cases, plastic bins, and people walking through. 1.5 cm voxels throughout. Per-capture tables and renders are in `out/<capture>/report.md`, the cross-capture tables in `out/summary.md`, and the working observations in `NOTES.md`.

Means over the ten captures for the runs that went everywhere:

| run | edge mean px | MAE cm | within 2 cm | completeness | floater hits | unsupported | F vs LiDAR | predict s | VRAM GB | license |
|---|---|---|---|---|---|---|---|---|---|---|
| lidar_tsdf (baseline) | 8.7 | 1.87 | 85.4% | 100% | 0.29% | 0.5% | n/a | 1 | 0 | sensor |
| mapanything_aligned | 8.4 | 1.88 | 81.8% | 100% | 0.65% | 3.0% | 0.800 | 145 | 10.7 | Apache 2.0 |
| promptda | 10.1 | 2.67 | 76.8% | 99.9% | 1.01% | 27.8% | 0.685 | 161 | 3.5 | Apache 2.0 |
| promptda_conf0 (weight 0 at confidence 0) | 10.1 | 2.56 | 76.9% | 99.9% | 1.05% | 17.2% | 0.730 | 160 | 3.5 | Apache 2.0 |
| da3_large | 11.1 | 8.02 | 37.9% | 97.5% | 3.92% | 20.9% | 0.427 | 43 | 6.6 | disputed |

The tabletop capture alone, where the remaining variants ran:

| run | edge mean px | MAE cm | within 2 cm | completeness | unsupported | license |
|---|---|---|---|---|---|---|
| lidar_tsdf | 5.3 | 0.73 | 94.7% | 100% | 0.0% | sensor |
| mapanything_aligned | 5.8 | 0.83 | 92.3% | 100% | 0.1% | Apache 2.0 |
| promptda | 6.7 | 0.94 | 91.1% | 100% | 6.4% | Apache 2.0 |
| mapanything (no scale alignment) | 7.5 | 2.09 | 70.4% | 100% | 0.1% | Apache 2.0 |
| da3_large | 7.8 | 1.06 | 89.2% | 93% | 2.5% | disputed |
| da3 (Base) | 9.8 | 1.88 | 77.5% | 93% | 17.4% | Apache 2.0 |
| milo | 11.4 | 2.48 | 71.6% | 100% | 16.2% | non-commercial |
| mapanything_nolidar | 14.9 | 11.53 | 4.2% | 98% | 34.5% | Apache 2.0 |

Reading the tables:

- Every metric compares against the held-out LiDAR, so the LiDAR fused with itself sets the ceiling on the numbers, and a method that is sharper than the sensor is penalized where the sensor is blobby. The renders show PromptDA, DA3 Large, MapAnything, and above all MILo resolving the mouse and the cap bill more crisply than the baseline despite scoring below it.
- MapAnything with the per-window LiDAR scale alignment is the one model that tracks the baseline on every capture: MAE within 0.2 cm, edge px equal or better on 8 of 10, and no geometry beyond where the LiDAR saw. It is also the heaviest (4.9 GB of weights, about 8 GB of VRAM before any views).
- PromptDA is close to MapAnything in the room and on the tabletop and consistently third elsewhere. Its unsupported fraction (20 to 45 percent on the booth captures) is geometry extrapolated past the LiDAR's range into the hall behind each booth, where ARKit's depth has confidence 0 and the prompt is nearest-filled. It runs at a third of MapAnything's VRAM and double its resolution.
- DA3 Large is fastest and fails on half the booth captures (11 to 23 cm MAE where screens and glass dominate), compresses far depth on the tabletop, and shows per-window scale seams in the room.
- MILo (36 minutes of training) gives the sharpest object outlines and the worst numbers: printed patterns on the cloth and book covers become geometric relief, and its Delaunay extraction spans the poorly observed far region with large floating triangles. Not a quality ceiling, and its license rules it out for shipping anyway.
- No model works without the LiDAR prompt or input.
- At room scale the differences between the two leading models are smaller than the baseline's own drift error (MAE 1.75 cm against itself), so ARKit pose refinement is now the larger lever for silhouette accuracy than the choice of depth model.
