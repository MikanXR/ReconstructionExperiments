# Working notes

## Decision

LiDAR-only TSDF is the production path (see the README). The reasoning, in the order it was weighed:

- Against held-out LiDAR frames the baseline matched or beat every model on every one of ten captures. The metric favours the sensor on silhouettes, but the renders show the same thing at room scale: the meshes are the same within the ARKit drift they all inherit.
- The models' one visible gain, crisper small-object outlines at 8 mm voxels, is a resolution effect (1008x756 or 518x392 depth against 256x192), and the LiDAR-only fine-voxel fusion recovers part of it for free.
- The end use is a coarse depth prior rendered from the tracked camera and refined against the video's edges in image space. The LiDAR mesh is already inside that refinement band wherever the sensor saw. A sharper prior does not move the refined edge; pose error does.
- Cost side: 2.5 minutes and 3.5 to 11 GB per capture, a torch stack per model, and no realistic in-editor port, against a fusion that runs in seconds with no model.
- What no static mesh gives (thin structures, moving objects) would need per-frame depth at runtime, which is a different architecture and not improved by a better offline mesh.

Kept: PromptDA as an optional offline refinement with `conf0_weight: 0`, for hero objects if the refined edges prove too soft. The zero weight was tested on all ten captures (`promptda_conf0`): held-out depth and silhouette numbers are unchanged, triangle counts fall 10 to 25 percent, and unsupported surface drops by a third to a half on the booth captures (40 to 22, 45 to 30, 29 to 16 percent) but stays at 14 to 31 percent against the baseline's under 1. The remainder comes from medium-confidence pixels (weight 0.5) at range, where the model and the noisy LiDAR disagree. So the rule is worth keeping on, and it does not make PromptDA's far field trustworthy. Retired: MapAnything (4.9 GB of weights, 8 GB VRAM floor, its edge over PromptDA was staying inside the sensed volume, which the weight rule gives PromptDA too), DA3 (fails on screen-and-glass scenes), MILo (non-commercial and worst on the numbers).

Next levers, in order: pose refinement of the sweep and good live-camera registration, voxel size per scene scale, and the image-space edge refinement itself.

## Tabletop capture notes

Observations gathered while running the comparison on capture `86a1a8e3b1` (a tabletop sweep: cap, mouse, bottle, bag, remote on a patterned table at 0.3 to 1.2 m, with a couch and shelf at 1.5 to 3 m behind). The README's results section is the consolidated version.

## Harness

- Back-shell bug, found by inspecting `mesh.ply` in a viewer: every TSDF mesh was a thin closed slab, a real front face plus a spurious back face about two voxels behind it. Unobserved voxels hold TSDF +1, and marching cubes (even under the observed-voxel mask) formed a zero crossing where the observed negative band behind a surface met them. The fix fills each unobserved voxel with the sign of its nearest observed voxel before extraction. On the tabletop MapAnything mesh the triangle count went from 142 k to 88 k with no visible change to the front surfaces. Every fused result was regenerated after the fix; the tables in this file and the README are the post-fix numbers. Expected effect: unsupported surface and chamfer drop for every TSDF method (the back faces counted as unsupported geometry), held-out depth and silhouette numbers barely move since the ray caster hit the front face first.
- Pose convention verified: LiDAR from one frame reprojected into frames 3 and 30 later agrees to 1 to 3 mm median, so Stray Scanner poses are camera-to-world with OpenCV camera axes and no flip is needed.
- The held-out silhouette metric treats the LiDAR depth as truth. The baseline against itself scores 2.6 px median at 256x192, which is the metric's noise floor. A method sharper than the sensor is penalized where the sensor is blobby (the mouse and the cap bill are visibly crisper in the PromptDA and DA3 Large meshes while their edge numbers are worse), so the visual sheet carries the silhouette judgment and the metric catches gross misses.
- "Unsupported surface" at 3 cm measured TSDF smoothing against raw LiDAR noise (25 percent for the baseline itself). At 5 cm, and with the back-shell fix, the baseline sits at 0.03 percent on the tabletop and under 1 percent on every capture, which is the floater signal the metric is meant to carry.
- Finer voxels (8 mm, `configs/fine.yaml`) make the edge numbers slightly worse for every method because the mesh keeps more detail and noise than the smoothed LiDAR reference. Visually they are better.

## PromptDA

- Scale tracks LiDAR within 0.5 to 1.5 percent at every depth band, including 2 to 3 m. Weight never collapses because it is derived from the LiDAR confidence, not the model.
- 1008x756 output, 1.4 s per frame on the shared GPU, 3.5 GB peak. The 402-frame run took 195 s.

## DA3

- Fast: 33 s (Base) and 63 s (Large) for 402 frames at 504x378, 3.6 and 6.6 GB peak, 32-frame windows.
- Depth is right at the table (ratio to LiDAR 0.99 to 1.02 under 1 m) and compresses with distance: 0.89 (Large) and 0.95 (Base) at 1 to 1.5 m, 0.77 and 0.84 at 1.5 to 2 m, about 0.7 at 2 to 3 m. The model's own confidence collapses in the same bands (weight under 0.05 for every pixel beyond 1.5 m), so the fusion drops the far geometry: completeness 92 to 93 percent and the couch and shelf are missing from the top-down view.
- Tested whether the small baseline of 32 consecutive keyframes (about 3 s of video) was the cause: `da3.window_mode: strided` builds 13 windows of 31 frames that each span the whole sweep (`da3_large_strided`). It did not help. The far-field ratio stayed at 0.90 (1 to 1.5 m) and 0.70 (2 to 3 m), confidence still collapsed beyond 1.5 m, and completeness dropped to 90.9 percent because near-field consistency got worse. The far-field weakness is the model's behaviour on this capture (far geometry seen obliquely and never approached), not a windowing artifact.
- Near-field silhouettes (mouse, cap bill) are the crispest of the methods so far.

## MapAnything

- The Apache checkpoint is 4.9 GB fp32 and sits at about 8 GB of VRAM before any views; 24-view windows peak at 10.7 GB. 402 frames take about 180 s at 518x392.
- Metric scale is regressed per window by a scale head, not copied from the inputs. With images, intrinsics, poses, and LiDAR all supplied the output runs 3 percent large (ratio 1.03, MAE 2.1 cm). Images plus poses without LiDAR gives ratio 1.29, images alone 3.1, so the LiDAR is doing most of the metric anchoring and the pose scale the rest.
- `mapanything.align_scale_to_lidar: true` divides each window by its median predicted/LiDAR ratio over the pixels the model was fed. That removes the bias (ratio 1.00, MAE 0.83 cm) and gives the best held-out numbers of any model: edge 5.8 px mean, within 2 cm 92 percent, completeness 100 percent, unsupported 0.05 percent. Windows agree to about 2 mm on shared frames, so the per-window correction does not introduce seams.
- Completeness stays at 100 percent at every depth band, unlike DA3, so the couch and shelf are present.
- Without LiDAR (`mapanything_nolidar`) the output is unusable for masking: ratio 1.29, MAE 11 cm, 4 percent within 2 cm. The LiDAR prompt is not optional for any of the pose-conditioned models on this capture.

## MILo results

- 18,000 iterations at 960x720 plus SDF mesh extraction took 36 minutes on the 4090 Laptop (peak about 4.9 GB early in training, not tracked through the harness). The extracted mesh has 2.87 M triangles after cleanup (205 k removed as islands), about 16 times the TSDF meshes.
- It is the sharpest reconstruction where the capture looked closely: the cap visor, the bottle, the mouse, and the book edges read as crisp outlines in the top-down view. It is also the worst on every number: edge 11.4 px mean, MAE 2.5 cm, 3.1 percent floater hits, F-score 0.62 against the baseline.
- Two causes, both visible in the renders. The patterned tablecloth and the printed book covers are baked into the geometry as relief, so flat surfaces read a centimetre or two off the LiDAR. And the Delaunay-based extraction spans poorly observed regions (the couch at 1.5 to 3 m, the frame edges) with large spiky triangles that sit in free space, which is what the floater and unsupported metrics count.
- The training used ARKit poses unchanged, no exposure lock on the capture, and MILo's `fast` config with default densification. A longer schedule, the `outdoor` importance metric for the far region, or masking the far background out of training would each be reasonable next experiments, but the texture-as-relief problem is inherent to photometric reconstruction of a flat printed surface and would need depth supervision (MILo has a Depth-Anything depth-order term, unused in the paper) to fix.
- Conclusion for the comparison: as a reference it is not a ceiling on this capture. The LiDAR-anchored feed-forward methods are closer to the sensor everywhere except object silhouettes, where MILo is better and the metric cannot see it.

## Room-scale capture (`03aec96c39`)

A hotel room: bed, desk, chair, a corridor, and a bathroom seen through its door. 3,243 frames, 973 training and 108 held out, fusion box 7.1 x 3.5 x 5.6 m at 1.5 cm (472 x 232 x 374 voxels, 67 s to fuse on the GPU). Poses stay consistent over 30 frames (3 to 5 mm median), but the baseline against itself drops from 0.7 cm MAE on the tabletop to 1.7 cm here and from 5.3 to 11.4 edge px, which is accumulated ARKit drift across the sweep showing up in every method's reference.

| run | edge px | MAE cm | within 2 cm | completeness | floaters | unsupported | F vs LiDAR | predict s |
|---|---|---|---|---|---|---|---|---|
| lidar_tsdf | 11.2 | 1.75 | 80.9% | 100% | 0.25% | 0.2% | n/a | 1 |
| mapanything_aligned | 11.4 | 1.98 | 77.1% | 100% | 0.33% | 2.4% | 0.941 | 368 |
| promptda | 11.6 | 2.34 | 73.2% | 100% | 0.42% | 6.6% | 0.890 | 454 |
| da3_large | 13.3 | 5.21 | 39.3% | 100% | 3.49% | 19.8% | 0.708 | 109 |

- MapAnything (aligned) and PromptDA both track the baseline to within 0.3 to 0.6 cm of MAE at room scale, with the same ordering as the tabletop. Their meshes are indistinguishable from the baseline's in the top-down view and the per-view strips. PromptDA keeps slightly more of the chair's thin frame.
- DA3 Large now covers the whole room (completeness 100 percent, the sweep approaches everything), but the per-window scale alignment disagrees between windows by several centimetres, which shows as layered seams across the bed and floor in the top-down view: 5.2 cm MAE, 3.5 percent floater hits. The fix would be a global pose-scale solve across windows rather than per-window Umeyama, which is a change inside the model's alignment, not a fusion setting.
- At room scale the per-method differences are smaller than the baseline's own drift error, so pose refinement (keyframe bundle adjustment or ICP against the fused volume) is now the larger lever for silhouette accuracy than the choice between the two leading depth models.

## First MikanARStreamer take (`take_20261013_125556`)

The first take recorded by the streamer's own LiDAR path rather than Stray Scanner: a 20 s room sweep, 615 frames with 612 depth records (the first three are zero records from sensor warm-up), uploaded to the editor over Wi-Fi and loaded by `load_mikan_take`. `tools/check_take.py` reprojects LiDAR between frames thirty apart at 1.7 to 5.8 mm median per pair, which confirms the ARKit-to-OpenCV camera flip in the loader and the record alignment. `lidar_tsdf` with the default config (185 training frames, 20 held out, 1.5 cm voxels, 32 s to fuse on the CPU): 1.38 cm MAE, 10.2 edge px, 99.8 percent completeness, 0.58 percent floaters, 0.44 percent unsupported. That sits between the tabletop and hotel-room baselines, as a short sweep with little loop drift should. The top-down render shows the floor, a desk with objects, and the walls as one surface.

## Tabletop streamer take (`take_20261013_132047`)

A desk with two open laptops, a keyboard, a mouse, a remote, boxes, cables, and a couch behind, orbited at arm's length: 861 frames over 29 s, 6.1 m of camera path, high-confidence depth median 0.8 m (p90 1.5 m). The recording is sound: `check_take.py` gives 3.5 mm median over pairs thirty frames apart, and the held-out median error of the default fusion is 0.56 cm. The MAE is not: 2.55 cm at 1.5 cm voxels, 2.59 cm at 8 mm (`fine`), 2.98 cm at 8 mm with a 1.6 cm truncation (`thin`), against the 0.7 cm the Stray Scanner tabletop gave. Nine percent of held-out pixels with errors over 5 cm carry 72 percent of the total, and that tail does not track camera speed (correlation -0.13 between a frame's speed and its MAE), so motion blur is not the cause.

The tail is the laptop lids. On frame 855 the LiDAR returns the left screen at 0.85 m (94 percent of the patch at confidence 2, median stable within 3 cm over twenty neighbouring frames), and every fused mesh puts the surface at 1.04 m, the desk behind it. The sensor sees the lid frontally; the views from the side and behind, where a few-millimetre plate is edge-on to a sparse LiDAR, return the desk and carve the lid's voxels as free space, and the frontal votes lose the weighted average. A smaller truncation does not change that, because the carving is from beyond the plate, not from its back face. This is the thin-structure limit the decision already records (chair legs, cables), with a laptop lid as the example that matters most for a desk shot. The two mitigations are a fusion that weights free-space carving below surface votes (an untested change to `fusion.py`), or per-frame depth at composite time, which is a different architecture. A tabletop measurement comparable to the Stray Scanner one needs a scene of solid objects; the surfaces this take does fuse are as accurate as that one was.

Voxel size on this take: 8 mm costs five times the fusion time (203 s against 41 s on the CPU) and four times the triangles for no MAE change, while unsupported surface drops from 0.26 to 0.03 percent. The room take at 1.5 cm fused in 32 s.

## Room loop streamer take (`take_20261013_154718`)

A 57 s walk around the same room, returning to the start: 1646 frames, 16.6 m of camera path at about 0.3 m/s, 1.8 x 0.9 x 2.7 m of translation extent, the end pose 0.47 m from the start. Short-range consistency is looser than the slower takes, 10.3 mm median over pairs thirty frames apart against 3.5 to 5.3 mm, with fewer shared points per pair (7k to 17k against 30k) because the camera covers more ground in a second. Loop closure measured by registering the last four seconds of LiDAR against the first four: on the 56 percent of late points that overlap the early cloud the disagreement is 19 mm median and 89 mm p90 before any correction, and a rigid point-to-plane ICP finds only a 31 mm, 0.9 degree offset that brings the median to 16 mm. ARKit relocalises on the return, so the drift does not sit at the seam as one rigid error. It is spread along the sweep.

`lidar_tsdf` at 1.5 cm (494 training frames, 55 held out, 424 s on the CPU): 2.35 cm MAE, 8.7 edge px, 100 percent completeness, 0.53 percent floaters. Held-out median error 1.07 cm, against 0.39 cm on the short room sweep of the same room and 0.56 cm on the tabletop, so the body of the distribution doubled rather than the tail: pixels over 5 cm carry 51 percent of the error here against 72 percent on the tabletop, and the per-frame error does not track camera speed (correlation -0.28). That is accumulated pose drift between the views that fused a surface and the view that scores it, the same effect the hotel room showed (1.75 cm MAE, 11.2 edge px over 973 training frames), and it is what a pose refinement has to remove. The top-down render is a clean floor plan: desk, couch, chairs, the corridor, straight walls, no doubled surfaces visible at this scale.

## Ten captures (`out/summary.md`)

Nine convention booth tabletops (CRT and LCD screens, glass cases, transparent bins, cloth, people walking through) plus the hotel room, baseline and the three shipping-plausible runs on each. Means over the ten:

| run | edge px | MAE cm | within 2 cm | completeness | floaters | unsupported | F vs LiDAR | predict s |
|---|---|---|---|---|---|---|---|---|
| lidar_tsdf | 8.7 | 1.87 | 85.4% | 100% | 0.29% | 0.5% | n/a | 1 |
| mapanything_aligned | 8.4 | 1.88 | 81.8% | 100% | 0.65% | 3.0% | 0.800 | 145 |
| promptda | 10.1 | 2.67 | 76.8% | 99.9% | 1.01% | 27.8% | 0.685 | 161 |
| da3_large | 11.1 | 8.02 | 37.9% | 97.5% | 3.92% | 20.9% | 0.427 | 43 |

- MapAnything (aligned) matches the baseline on every capture: MAE within 0.2 cm, edge px equal or better on 8 of 10, unsupported surface 1 to 8 percent against the baseline's 0 to 1. It is the only model whose far-field geometry stays where the LiDAR saw it.
- PromptDA is consistently third. Its unsupported fraction (28 percent mean, 20 to 45 percent on the booth captures, 6 to 7 percent on the tabletop and in the room) is geometry extrapolated past the LiDAR's range: beyond about 4 m the ARKit depth has confidence 0, the prompt is nearest-filled from the last valid pixel, and the model paints smooth sheets across the hall behind the booth that the weight of 0.25 does not suppress. Zeroing the weight where confidence is 0 (instead of 0.25) would remove most of it and is a one-constant change.
- DA3 Large fails on half the booth captures (MAE 11 to 23 cm on `82cab9c64d` and `76a023ddb1`, within 2 cm under 10 percent on four captures). The per-window scale alignment does not survive scenes dominated by screens and glass.
- The LiDAR baseline itself is worst on the three captures with glass cases and bins (`97c6bc203d`, `76a023ddb1`, `ce6eafa2a4`: MAE 2.6 to 3.2 cm), which is where the held-out reference is least trustworthy for everyone.

## Chamfer against the LiDAR baseline

F-score at 5 cm against the baseline mesh tracks completeness and extrapolation. Tabletop: MapAnything 0.95 (aligned or not), PromptDA 0.87, DA3 0.42 to 0.49 because the far geometry is absent, strided DA3 0.35, MILo 0.62. Across the ten captures: MapAnything 0.80, PromptDA 0.69, DA3 Large 0.43.

## MILo

- The machine has no CUDA compiler: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1` holds only a runtime (bin, include, lib, no `nvcc.exe`), and no `nvcc.exe` exists anywhere on `C:`. Every Gaussian splatting trainer (MILo, 2DGS, vanilla 3DGS) compiles a rasterizer extension, so none of them can be built without a toolkit. nvdiffrast built because it compiles its kernels lazily at first use, so it will fail then.
- MILo's submodules are referenced by SSH URL in `.gitmodules`; rewriting them to HTTPS (`sed` on `.gitmodules`, `git submodule sync`) fetches them. Its `tetra_triangulation` extension needs CGAL with GMP and MPFR through CMake and links against the conda-only `cnpy`, and `mesh_extract_sdf.py` and the mesh regularizer both import it, so it is not optional.
- The COLMAP export (`out/<capture>/milo/colmap`) is complete and ready for any trainer once a toolkit is available.

Windows build, once CUDA Toolkit 12.8 was installed (all of it scripted in `.venv_milo/build_ext.cmd`):

- Toolset: `vcvars64.bat -vcvars_ver=14.44` (the VS 2022 toolset shipped inside VS 2026), since nvcc 12.8 does not accept the 14.5x compilers.
- Every extension that includes torch headers from a `.cu` file failed with `compiled_autograd.h: error C2872: 'std': ambiguous symbol` (the MSVC candidates are `valarray`'s `std` and the header's own `::std`). `NVCC_APPEND_FLAGS=-Xcompiler /permissive-` fixes it. nvcc reads that variable itself, so no setup.py edits are needed.
- The GOF rasterizer uses the POSIX `u_int16_t` and `u_int32_t`; two typedefs under `#ifdef _MSC_VER` at the top of `cuda_rasterizer/forward.cu` fix it.
- `tetra_triangulation`: CGAL 6.2 with GMP and MPFR from a local vcpkg (24 minutes), the `Visual Studio 18 2026` generator with `-T v143`, `CMAKE_POLICY_VERSION_MINIMUM=3.5`, pybind11 bumped from 2.9.2 to 2.13.6 (2.9.2 does not configure under CMake 4), the GCC-only `-flto=auto` and `-D_GLIBCXX_USE_CXX11_ABI` flags guarded behind `if(NOT MSVC)`, and the CUDA language made optional since the module has no `.cu` sources. The built `.pyd` plus `gmp-10.dll`, `gmpxx-4.dll`, and `mpfr-6.dll` are copied beside `tetranerf/utils/extension/__init__.py`. A 2000-point test returns 12,956 tetrahedra.
- Python packages missing from MILo's `requirements.txt` that the training path imports: `matplotlib`, `scikit-learn`.
- Training uses `-r 2` (960x720): MILo keeps every image on the GPU, and 402 frames at its default 1600 px cap would take about 9 GB before the model.
- nvdiffrast compiles its plugin at first use, so the training process needs the same environment as the builds (`.venv_milo/train_milo.cmd` calls `vcvars64.bat` and sets the 12.8 paths; without them it picked the 13.1 runtime headers from the system `CUDA_PATH`). With torch 2.11 the compiled plugin built but `importlib.import_module("nvdiffrast_plugin")` failed, because `torch.utils.cpp_extension.load` no longer puts the JIT build directory on `sys.path`. Patched `nvdiffrast/torch/ops.py` in the venv to keep the module object `load` returns.
- `scene/mesh.py` defaults `MeshRasterizer` to nvdiffrast's GL context. On this hybrid laptop GL binds to the Intel iGPU, so the default was switched to `RasterizeCudaContext`, which renders the same thing through CUDA.
