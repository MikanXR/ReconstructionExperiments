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

## Frame-to-model registration on the room loop (`lidar_tsdf_f2m*`)

KinectFusion's loop as `fusion.frame_to_model`: ray cast the volume from the frame's predicted pose, projective point-to-plane ICP against that map, integrate at the refined pose, carry the correction forward as the next prediction, and score the held-out frames at their nearest integrated frame's correction. Four runs on `take_20261013_154718`, all 1.5 cm voxels, 494 training frames, CPU:

| run | MAE cm | edge px | held-out median cm | end-to-end seam | fuse s |
|---|---|---|---|---|---|
| lidar_tsdf (ARKit poses) | 2.35 | 8.72 | 1.07 | 31 mm, 0.94 deg | 424 |
| f2m: ICP, no prior | 1.80 | 8.11 | 0.54 | 41 mm, 1.38 deg | 701 |
| f2mp: + prior, 2-voxel normals | 1.74 | 8.07 | 0.52 | 37 mm, 1.27 deg | 800 |
| f2mq: + skip rejected, 15 cm bound | 1.59 | 7.85 | 0.49 | 30 mm, 1.11 deg | 772 |

What each step fixed, found on synthetic planes (`tools/test_register.py`) and confirmed on the take:

- A trilinear sample at the rim of an observed patch blends the band with the unobserved +1 and reads as a zero crossing with an in-plane normal: a fake wall around every patch that the solve slides toward. Any sample touching an unobserved voxel now reads as unobserved.
- One-voxel central differences give the model normals 2.4 percent noise, which makes an unobservable direction (along a lone wall or floor) look stiff and lets the first iteration's large residual push the pose sideways. A two-voxel stencil brings that to 0.9 percent, and a prior on the deviation from the carried-forward ARKit pose (3 mm, 0.2 degrees, against a 5 mm data sigma) holds what is left: a floor-only frame then slips under 3 mm while a fully constrained 54 mm displacement is still recovered to 0.5 mm.
- ARKit jumped 56 mm in one frame at frame 844, a tracking correction, after which the model disagreed with the frames by 11 cm. Under a 10 cm bound four frames were rejected and, worse, integrated at the stale pose, which put the held-out frame between them at 8.6 cm error. Rejected frames are now skipped, and the bound is 15 cm so a jump is absorbed in one accepted step (107 mm at frame 846).

What the numbers say. The body of the distribution is back where the short sweep of the same room was (0.49 against 0.39 cm median), so the drift accumulated along the walk is what the refinement removed, and MAE fell by a third. The seam did not move: ARKit's own relocalisation already closes the loop to about 3 cm and 1 degree, and frame-to-model drifts the same way on a loop, so a loop constraint would be the next thing to add rather than a replacement. The per-step corrections have a 10 mm median across the direction of travel (not along it: the implied depth-to-pose time offset is 0 ms, and shifting the poses by a frame in either direction only makes the reprojection check worse), which is consistent with ARKit's upsampled depth warping a few millimetres per frame. The fused surface averages that out; the pose track inherits it. Fusion time rises by about 80 percent on the CPU, almost all of it the ray cast at 256x192 with 164 samples per ray.

Two things the streamer could add to make this better: the frame's `ARCamera.trackingState` in the pose track, so a fusion can treat limited-tracking frames the way it now treats rejected ones, and the raw LiDAR sample points if ARKit ever exposes them, which would replace the upsampled map's correlated noise with a few hundred independent returns.

## Image-space edge refinement (`tools/edge_refine_eval.py`)

The filter MikanXR's `DepthEdgeRefineNode` runs, prototyped here on the room loop take with the `lidar_tsdf_f2mq` mesh and its refined poses, six held-out frames at 1920x1440, scored against the held-out LiDAR's edges at 256x192. The first rule tried, a joint bilateral two-sided vote with a spatial Gaussian of half the band radius, moved nothing: 9.44 px before, 9.46 after, 2.5 percent of pixels touched. The far side's own colour-matching pixels next to the edge outvote the near side, so the edge can move two or three pixels at most whatever the radius, and the mesh's silhouettes sit ten to twenty video pixels off the picture's. The rule became a two-prototype classification: the window splits at its midpoint depth, each side's mean colour is its prototype, and the pixel joins the side its own colour is nearer to, at that side's mean depth, unless the prototypes are closer than the colour threshold. Box weights, so the reach is the whole radius.

| radius | sigma | edge px | matched | pixels changed |
|---|---|---|---|---|
| prior | | 9.44 | 38.3% | |
| 8 | 0.10 | 9.11 | 40.2% | 1.2% |
| 16 | 0.10 | 9.03 | 37.5% | 5.4% |
| 16 | 0.20 | 9.00 | 40.1% | 3.1% |
| 32 | 0.05 | 10.24 | 27.2% | 28.3% |
| 32 | 0.20 | 9.18 | 36.7% | 12.2% |

Radius 16 with a 0.2 colour threshold is the default that landed in the node. The gain reads small on this metric because the LiDAR edges are blocky at 7.5 video pixels per LiDAR pixel and carry the sensor's blur, so a refined edge that follows the picture can score level with the prior; the contact sheets show the whale plush's silhouette, which the mesh overshoots by ten to twenty pixels, pulled onto the plush. Radius 32 with a small threshold over-reaches, moving a quarter of the pixels and losing matches. Hand-marked frames are the measure that would separate the two effects.

## Live occlusion edges against hand-marked silhouettes (`out/live_occlusion`)

Six live angles of a can on the desk and four frames of the take its mesh was fused from, captured from the running editor (`tools/capture_live_occlusion.py`, `tools/capture_mesh_variants.py`: the video, the mesh depth mask Mikan renders, and Mikan's DepthEdgeRefineNode output) and the can's true outline marked by hand (`tools/mark_silhouette.py`). `tools/occlusion_edge_eval.py` scores methods on the can's occluding edges only; `tools/score_mesh_variants.py` splits a mesh's silhouette error into a shift and a width error per row.

Edge methods, live, median distance to the marked occluding edge at 1920x1440: mesh 21.9 px, Mikan's node 19.2, the numpy reference at the node's settings 12.2 (the two disagree: box versus Gaussian weights, linear versus sRGB colour), the node's vote with prototypes from a trimap's definite regions and a median filter 4.2, contour snapping by dynamic programming 4.8 with the least stray boundary of the fast methods, GrabCut 5.7 with failures, a guided filter about 21 (it cannot move an edge 20 px). Every method's p90 sits at 50 to 100 px because the prior is wrong past any band: the can's black cap is missing from the mesh and the silhouette shifts.

Error split: on the live angles the mesh can is 22 to 40 px too thin in three of six views and shifted 0 to 46 px by view. On the take's own frames, with no second session, it is still thin (2.5 to 4 mm a side) and one frame is shifted 17.5 px (about 7 mm), so the thinning is the fusion and the shift is per-frame pose error, not only the live alignment.

The thinning is the flying pixel cull. Variants fused by `MikanCmd -occlusionMesh` at 8 mm and scored on the four marked take frames, width error per side: symmetric cull (both pixels of a bad pair) -2.3 mm, keep the near pixel -1.9, keep near and cull the far side within 2 px +0.4, within 3 px +2.0, no cull -3.8. Background pixels next to a near edge carve the object's side as free space; dropping them within 2 LiDAR pixels stops it. The same cull here (`fusion.gradient_cull_keep_near`, `gradient_cull_radius`, off by default so earlier runs compare) on the room loop with frame-to-model (`lidar_tsdf_f2m_v0` against `_v3`): median 5.28 to 5.21 mm, MAE 1.59 to 1.56 cm, edge 7.85 to 7.81 px, completeness 100 percent both, registration 491 of 491 both, floaters 0.39 to 0.61 percent. MikanXR made keep near with a 2 px band its default.

## SAM2 prompted from the mesh silhouette (`tools/sam2_edge_eval.py`, `tools/sam2_track_eval.py`)

SAM2 (Apache 2.0, the `sam2` 1.1.0 PyPI release with the SAM 2.1 checkpoints, in `.venv_sam2`) prompted from the occlusion mesh's silhouette on the ten hand-marked can frames in `out/live_occlusion`, the six live angles and the four take frames, scored by `occlusion_edge_eval.py` on the occluding edges at 1920x1440 as above. The band methods stop at the band: on the can's missing cap and the shifted views the prior is wrong past any band, which is where every p90 of 36 to 100 px comes from, and the question was whether an object segmenter recovers that. The prior is the mesh region nearer than the harness's threshold, the can and the desk it stands on, so the naive prompts fail: the prior's bounding box segments can and desk together (2.3 to 5.7 px median, 40 to 100 px p90 by model), points sampled over the prior land on the desk (14 to 38 px median), and the prior handed to the decoder as its dense mask prompt is unstable across sizes (2.9 px median on small, 43 on tiny). The prompts that work follow the occluding contour: the prior's outline is probed 45 px either side along its normal and kept where the mesh depth jumps, the kept stretches are grouped into runs, and each run is prompted on its own with positive points just inside it at the thickest spot within two bands (band 24 px), the runs' masks unioned. Among the three multimask outputs the one nearest the prior inside the run's box is kept. Median / p90 of the distance from the marked occluding edge to the mask's boundary, and the share within 5 px, over the ten frames:

| prompt | tiny | small | base_plus | large |
|---|---|---|---|---|
| mesh | 17.8 / 67 / 0.19 | | | |
| trimap vote, band 45 | 3.2 / 36 / 0.68 | | | |
| contour snapping | 3.3 / 51 / 0.67 | | | |
| runs, positive points | 1.6 / 10.0 / 0.90 | 1.3 / 6.3 / 0.94 | 1.6 / 5.2 / 0.93 | 1.6 / 5.8 / 0.93 |
| runs, box and positives | 1.6 / 20.3 / 0.85 | 1.4 / 16.7 / 0.90 | 1.5 / 14.4 / 0.89 | 2.2 / 18.6 / 0.85 |
| runs, box, positives and negatives | 2.4 / 16.1 / 0.83 | 1.7 / 14.8 / 0.90 | 2.8 / 9.0 / 0.77 | 2.2 / 9.5 / 0.86 |
| runs, box and positives, refined once | 1.9 / 22.1 / 0.81 | 2.3 / 20.3 / 0.81 | 1.9 / 19.8 / 0.80 | 1.3 / 13.9 / 0.91 |
| whole prior, box | 2.3 / 39.5 / 0.78 | 4.5 / 101 / 0.71 | 5.7 / 75 / 0.63 | 3.0 / 82 / 0.72 |
| whole prior, dense mask | 42.7 / 131 / 0.35 | 2.9 / 7.9 / 0.79 | 6.0 / 43 / 0.50 | 4.8 / 39 / 0.59 |

Positive points per run are the prompt, and the model size barely matters past tiny. On the hardest view, angle 5, where the mesh sits 60 px left of the can and the mesh's mean distance is 80 px, the runs prompt lands at 10 to 13 px mean on every size and the black cap is in the mask. The per-run box costs p90 there, since a box grown by one band around a mesh that is off by more than the band cuts the can. Negatives are the risk the design predicted: where the mesh is short by more than the band they land on the object, as they do on the cap, and they hurt on every size at band 24 and more at band 45 (tiny, band 45: positives 1.7 / 18.5 / 0.89, with the box 1.5 / 12.9 / 0.91, with negatives 3.1 / 21.6 / 0.77). Feeding the chosen logits back for a second pass gains nothing except on large. The whole 1920x1440 frame through the 1024 encoder instead of the crop (tiny, positives: 1.6 / 11.2 / 0.88) costs nothing for an object this size.

Tracking: the take's 481 frames from 2 to 18 s decoded from the capture (each marked frame matched against its decoded frame at under 1 of 255 mean absolute difference), one prompt on the first marked frame from its prior, `propagate_in_video` through the rest, scored at the four marked frames, with the same prompt in image mode on each of those frames alone beside it:

| model, prompt | image mode | tracked from the first frame |
|---|---|---|
| small, positives | 1.0 / 2.9 / 0.97 | 1.3 / 5.6 / 0.92 |
| small, box and positives | 1.0 / 3.1 / 0.96 | 1.3 / 5.4 / 0.94 |
| tiny, positives | 1.1 / 3.3 / 0.95 | lost after 17 frames |
| tiny, box and positives | 0.9 / 3.3 / 0.95 | 1.3 / 3.1 / 0.97 |
| small, the prior as a mask | | 4.6 / 27.9 / 0.70 |

The video predictor picks among its multimask outputs by its own score, with no prior to steer it. Tiny's points-only prompt chose a 486,000 px region on the first frame, the can and the desk, where small chose 191,000, and lost it 17 frames in; the box pins it, and tiny with the box is the best tracked result. The mask prompt is taken as the prompted frame's output as it is (20.9 px there, the mesh's own 20.8) and refined only on the following frames: 1 to 2 px by 6 and 10 s, 12 px at 18 s.

Cost on the RTX 3090 (`tools/sam2_timing.py`, CUDA, bfloat16, the 1920x1440 frame through the 1024 encoder, median of 50): encoder plus decoder tiny 26.0 + 8.6 ms, small 29.9 + 8.5, base_plus 40.4 + 8.8, large 84.3 + 8.7. The tracker's step per frame (`propagate_in_video`, including the mask resized to the video size on the host): tiny 52 ms, small 56, base_plus 66, large 109. Under DirectML, the runtime the editor uses (the image encoder exported to ONNX by `tools/sam2_export_encoder.py`, matching torch to 1e-5, timed by `tools/sam2_dml_timing.py` on onnxruntime-directml 1.24.4): tiny 67 ms, small 75, about 2.5 times CUDA. A bare `image_size` 512 override fails in sam2 1.1.0 with a view over a non-contiguous tensor in the Hiera trunk, since the predictor hard-codes its feature sizes for 1024; the 512 configuration is scored and timed below with those sizes set from `image_size`.

The objectness claim holds on the can: the segmenter puts the edge where the mesh, the vote and the contour snap cannot, including a part the LiDAR never returned. `tools/render_take_frames.py` renders a mesh into chosen frames of a take offline in the folder format `mark_silhouette.py` marks, so other takes can be marked without the editor; against the editor's own render of the V2 can mesh at the four marked times it agrees at IoU 0.998 to 1.000 with the frame index mapping exact, and the 1 to 4 px outline distance that remains sits on the mesh's ragged border at the top of the frame, with a half-pixel principal point difference the likely cause.

Three more sets were rendered that way and marked by hand, one occluder per frame (`out/mark_cup`, `out/mark_room`, `out/mark_duster`, scored into `out/sam2_edges_<set>`): the cup take with `cup_v3.glb` at its refined poses (the book, the open laptop, the cup, the chair back; the plant at 22 s has no mesh at all inside its outline and is skipped, since without a prior there is nothing to prompt from), the room walk with `noreg.glb` at recorded poses (the chair back, a plush on the couch arm, the air filter on the shelf, a plush and the backpack, the cat tree), and the 10-29 desk take with the mesh built from it (the duster can, the plush, the box, the mouse). The room's mesh is the worst prior yet, 46 px median off the marked edges. Median / p90 / within 5 px, the prompt being positive points per occluding run:

| set | frames | mesh | best band method | tiny | small | large |
|---|---|---|---|---|---|---|
| can | 10 | 17.8 / 67 / 0.19 | 3.2 / 36 / 0.68 | 1.6 / 10.0 / 0.90 | 1.3 / 6.3 / 0.94 | 1.6 / 5.8 / 0.93 |
| cup | 4 | 11.2 / 19 / 0.25 | 0.8 / 7.1 / 0.87 | 1.4 / 7.5 / 0.90 | 1.0 / 2.7 / 0.98 | 1.6 / 6.3 / 0.91 |
| room | 5 | 46.0 / 106 / 0.13 | 23.8 / 120 / 0.42 | 1.8 / 7.8 / 0.88 | 1.4 / 6.0 / 0.91 | 2.4 / 9.3 / 0.82 |
| duster | 5 | 5.6 / 26 / 0.48 | 2.3 / 9.2 / 0.76 | 1.8 / 10.5 / 0.90 | 1.7 / 3.7 / 0.94 | 1.3 / 3.4 / 0.96 |

Small is the consistent size: under 1.5 px median and 6.3 px p90 on every set, with at least 91 percent of the marked edge within 5 px, whether the mesh is 6 or 46 px off. The band methods track the mesh's error (contour snapping is excellent on the cup, where the mesh is close, and useless in the room) and the segmenter does not. The box helps where the mesh is accurate (duster: runs with box, or box and positives refined once, 1.0 to 1.2 px median and 2.8 to 3.1 px p90 on small) and hurts where it is not (room: box and positives 2.2 / 15 / 0.78 on small against 1.4 / 6.0 / 0.91 for positives alone), as on the can, so the points-only prompt is the one that needs no knowledge of the mesh's error. Negatives hurt on every set. The whole-prior prompts fail on every set the way they failed on the can.

## The C++ port in MikanXR (`tools/score_mikan_matte.py`)

MikanXR's `MikanCmd -occlusionMatte` builds the matte in C++: the mesh rasterized to depth per frame, the occluding contours found on the depth map alone (no marked prior), each run prompted with its positive points into SAM 2.1 small under ONNX Runtime's DirectML provider, and a label map written per frame that says where the segmenter disagrees with the mesh inside a band around each contour. The dumped labels applied to the marked take frames' prior, scored next to the Python result on the same frames (median / p90 / within 5 px over `take_t02`, `t06`, `t10`, `t18`, mesh 11.6 / 37 / 0.29, Python crop prompt 1.0 / 2.9 / 0.97):

| C++ run | median | p90 | within 5 px | segmenter per frame |
|---|---|---|---|---|
| band 24, every run judged by the union of the frame's masks | 3.6 | 14.3 | 0.69 | 867 ms |
| band 24, each run judged by its own mask | 3.7 | 16.1 | 0.67 | 625 ms |
| band 24, a small run prompted on a crop of its own | 2.9 | 16.4 | 0.70 | 1737 ms |
| band 64, crops, each run its own mask (the shipped defaults) | 1.0 | 3.7 | 0.94 | 1688 ms |

Two findings. The union was wrong by design: a two-point run on the desk beside the can returned the desk, and its mask told the can's band "occluder" where the can's own mask said background, so nothing was removed there. Each run now judges its band by its own mask, and runs shorter than 40 px are dropped as mesh noise. And the band is the reach of the correction: the marked edge points still more than 10 px from the matte on the 18 s frame sit a median 32 px from the mesh contour, past the 24 px band. Emulated in the harness by applying the Python mask only within a band of the prior's contour:

| band | all ten frames median / p90 / within 5 px | take frames median / p90 |
|---|---|---|
| 24 | 4.7 / 45 / 0.69 | 1.3 / 16.8 |
| 32 | 3.8 / 40 / 0.74 | 1.3 / 13.0 |
| 48 | 1.9 / 28 / 0.83 | 1.1 / 4.6 |
| 64 | 1.4 / 21 / 0.88 | 1.0 / 2.9 |
| no band | 1.3 / 6.3 / 0.94 | 1.0 / 2.9 |

The take frames reach the unbanded result at 64 px, which is the editor's default now; the live angles, where the mesh is off by up to 46 px, keep a 20 px p90 even there, the price of touching nothing far from a contour. The crop around a small run buys the median and the share within 2 px (0.43 to 0.55), not the p90, at 2.8 times the segmenter time; it stays on by default for a build that runs once per take. The segmenter's cost under DirectML is the decoder's inputs crossing to the device per call, which the export's split at the predictor's own feature boundary cut from 88 MB to 10 MB per call (867 to 625 ms per frame with 7 runs).

## Lighter segmenters for a live matte (`tools/segmenter_sweep.py`, `tools/sam2_dml_timing.py`)

The matte builds offline at 625 to 1688 ms per frame, and a live matte on the GPU the client renderer shares has a few milliseconds per frame, so the question was whether a lighter promptable encoder holds SAM2 small's edge through the same prompt design. The candidates, all Apache 2.0, each in its own venv with the install that worked in its `requirements-<family>.txt`: EdgeTAM (a RepViT-M1 trunk, 4.9 M encoder parameters, a fork of the `sam2` package, `.venv_edgetam`), EfficientTAM ti and s at 1024 and their 512 variants (ViT-Tiny and ViT-Small, 6.2 and 22.4 M, `.venv_efficienttam`), EfficientViT-SAM l0 and l1 at 512 and xl0 at 1024 (30.7, 43.6 and 112.9 M, with a SAM1-style decoder that takes no high-resolution features and no mask round trip, `.venv_efficientvit`), and SAM2 tiny and small at 512, which runs once the predictor's `_bb_feat_sizes` is set from `image_size`. `sam2_edge_eval.py --family` runs every family through the unchanged prompts (runs of occluding contour, positive points, the candidate nearest the prior), `segmenter_sweep.py run` covers the four marked sets in the crop framing and the whole frame, and `segmenter_sweep.py table` writes `out/segmenter_sweep.md`. The refactored harness reproduces small's can row to the digit, and every new encoder export (`sam2_export_encoder.py --family`) matches torch to 2e-5. EfficientViT-SAM runs from the authors' ONNX graphs on Hugging Face. Median / p90 / within 5 px, positive points per run, band 24:

| family | model | input | framing | can | cup | room | duster |
|---|---|---|---|---|---|---|---|
| sam2 | small | 1024 | crop | 1.3 / 6.3 / 0.94 | 1.0 / 2.7 / 0.98 | 1.4 / 6.0 / 0.91 | 1.7 / 3.7 / 0.94 |
| sam2 | small | 1024 | full frame | 1.9 / 7.4 / 0.87 | 1.4 / 4.6 / 0.92 | 2.0 / 18.6 / 0.83 | 1.6 / 5.6 / 0.89 |
| sam2 | small | 512 | crop | 1.6 / 16.1 / 0.90 | 1.8 / 5.2 / 0.90 | 2.6 / 7.8 / 0.80 | 1.6 / 4.2 / 0.91 |
| sam2 | small | 512 | full frame | 3.3 / 21.1 / 0.68 | 1.8 / 5.5 / 0.88 | 3.8 / 26.6 / 0.65 | 2.9 / 8.4 / 0.76 |
| sam2 | tiny | 1024 | crop | 1.6 / 10.0 / 0.90 | 1.4 / 7.5 / 0.90 | 1.8 / 7.8 / 0.88 | 1.8 / 10.5 / 0.90 |
| sam2 | tiny | 1024 | full frame | 1.6 / 11.2 / 0.88 | 1.5 / 3.3 / 0.97 | 3.0 / 10.3 / 0.79 | 2.1 / 7.7 / 0.83 |
| sam2 | tiny | 512 | crop | 2.1 / 17.4 / 0.84 | 8.5 / 23.6 / 0.59 | 3.0 / 8.7 / 0.76 | 2.0 / 14.5 / 0.80 |
| sam2 | tiny | 512 | full frame | 2.7 / 23.1 / 0.72 | 2.8 / 9.7 / 0.78 | 3.9 / 16.0 / 0.65 | 5.0 / 14.2 / 0.58 |
| edgetam | edgetam | 1024 | crop | 2.4 / 7.9 / 0.84 | 1.1 / 2.5 / 0.99 | 1.6 / 5.6 / 0.89 | 1.6 / 5.0 / 0.90 |
| edgetam | edgetam | 1024 | full frame | 2.8 / 12.2 / 0.80 | 2.5 / 5.2 / 0.86 | 1.9 / 7.6 / 0.85 | 2.1 / 5.8 / 0.86 |
| efficienttam | ti | 1024 | crop | 1.1 / 8.9 / 0.93 | 3.0 / 11.4 / 0.85 | 2.1 / 6.9 / 0.82 | 1.0 / 3.1 / 0.96 |
| efficienttam | ti | 1024 | full frame | 2.3 / 8.2 / 0.81 | 2.2 / 6.4 / 0.84 | 2.2 / 6.8 / 0.85 | 1.8 / 9.8 / 0.84 |
| efficienttam | ti | 512 | crop | 1.8 / 17.1 / 0.88 | 2.6 / 8.8 / 0.84 | 2.2 / 6.7 / 0.84 | 1.8 / 4.8 / 0.90 |
| efficienttam | ti | 512 | full frame | 2.9 / 13.1 / 0.73 | 4.5 / 9.4 / 0.68 | 3.8 / 10.3 / 0.65 | 3.5 / 9.7 / 0.70 |
| efficienttam | s | 1024 | crop | 1.5 / 11.9 / 0.93 | 3.4 / 7.6 / 0.80 | 1.8 / 6.2 / 0.85 | 1.4 / 3.9 / 0.94 |
| efficienttam | s | 1024 | full frame | 2.0 / 9.5 / 0.85 | 3.4 / 10.5 / 0.75 | 2.9 / 8.3 / 0.74 | 1.4 / 3.6 / 0.95 |
| efficienttam | s | 512 | crop | 2.5 / 13.8 / 0.88 | 1.0 / 3.5 / 0.95 | 1.9 / 6.1 / 0.87 | 1.8 / 3.9 / 0.91 |
| efficienttam | s | 512 | full frame | 3.2 / 15.8 / 0.71 | 3.3 / 17.7 / 0.75 | 3.4 / 11.9 / 0.73 | 2.7 / 9.3 / 0.77 |
| efficientvit_sam | l0 | 512 | crop | 1.6 / 13.3 / 0.84 | 0.5 / 4.7 / 0.95 | 1.8 / 27.5 / 0.84 | 1.4 / 3.5 / 0.94 |
| efficientvit_sam | l0 | 512 | full frame | 1.6 / 6.1 / 0.89 | 1.8 / 7.7 / 0.87 | 1.8 / 17.9 / 0.83 | 1.4 / 4.2 / 0.92 |
| efficientvit_sam | l1 | 512 | crop | 1.5 / 5.6 / 0.91 | 0.8 / 5.7 / 0.93 | 2.0 / 24.1 / 0.84 | 1.4 / 5.0 / 0.91 |
| efficientvit_sam | l1 | 512 | full frame | 1.8 / 6.2 / 0.88 | 3.2 / 10.2 / 0.75 | 1.8 / 30.2 / 0.83 | 1.6 / 4.4 / 0.90 |
| efficientvit_sam | xl0 | 1024 | crop | 1.4 / 6.8 / 0.89 | 0.5 / 1.8 / 0.98 | 1.1 / 9.6 / 0.90 | 1.4 / 3.7 / 0.94 |
| efficientvit_sam | xl0 | 1024 | full frame | 1.4 / 4.8 / 0.93 | 2.9 / 10.7 / 0.76 | 1.8 / 15.3 / 0.83 | 1.4 / 3.7 / 0.92 |

Nothing beats SAM2 small. In the crop framing EfficientViT-SAM xl0 and EdgeTAM come nearest (xl0 on the cup 0.5 / 1.8 / 0.98, EdgeTAM 1.1 / 2.5 / 0.99) and both lose on the can, and the EfficientViT models carry a room p90 of 10 to 30 px from single frames the SAM2 decoder's candidates handle. The whole frame, the framing a live path encodes, is worse for every model, small included: its room p90 of 18.6 is one frame (`t180`, 14.5 px median where the other four sit at 1.7 to 4.7). Every 512 configuration degrades, worst on the whole frame, where one encoder pixel is 3.75 frame pixels and 58 to 88 percent of the edge lands within 5 px. The per-run box (`runs_box_pos`, in `out/segmenter_sweep_box_pos.md`) hurts the whole-frame rows across families as it hurt the can before. Read against the rule set before the runs, that a live candidate's DirectML encoder costs at most 10 ms and its whole-frame row sits within 1 px median and 3 px p90 of small's on every set with at least 85 percent within 5 px, none qualifies. EfficientViT-SAM l0 is nearest on accuracy (inside the margin on three sets, cup p90 7.7 against 4.6) at 26 to 40 ms of encoder and a second decoder contract; EdgeTAM is nearest on cost and misses on the can (2.8 / 12.2 / 0.80).

Cost, the RTX 3090 with the Unreal editor, two `Mikan.exe` and LM Studio running (53 to 97 percent utilization and 11 GB in use before each family, read from `nvidia-smi` into `out/sam2_timing_lighter/run.log`): the quiet numbers above (small 75 ms fp32 under DirectML, 29.9 ms under CUDA) are 0.5 to 0.6 of the same graphs here, and the pass is one command away from a quiet rerun. DirectML is onnxruntime-directml 1.20.1, the editor's version, with the 1920x1440 frame as the encoder input, 50 iterations after 10 warm-up, median ms. CUDA is torch bfloat16 through each family's predictor. The fp16 column is the ORT graph converted as described below, with the largest difference in `image_embed` against the fp32 graph on the same frame, divided by the output's value range:

| family | model | input | encoder params | CUDA encoder | DML fp32 | DML fp16 | fp16 diff |
|---|---|---|---|---|---|---|---|
| sam2 | tiny | 1024 | 27.2 M | 57.3 | 113.5 | 55.7 | 1.3e-2 |
| sam2 | tiny | 512 | 27.2 M | 19.8 | 29.3 | 15.8 | 1.0e-2 |
| sam2 | small | 1024 | 34.3 M | 65.4 | 126.7 | 63.1 | 1.6e-2 |
| sam2 | small | 512 | 34.3 M | 23.9 | 35.0 | 19.0 | 1.5e-2 |
| edgetam | edgetam | 1024 | 4.9 M | 45.8 | 21.4 | 13.5 | 1.9e-2 |
| efficienttam | ti | 1024 | 6.2 M | 29.1 | 58.6 | 30.5 | 1.7e-2 |
| efficienttam | ti | 512 | 6.2 M | 23.5 | 16.2 | 11.7 | 1.4e-2 |
| efficienttam | s | 1024 | 22.4 M | 47.1 | 119.2 | 56.8 | 2.0e-2 |
| efficienttam | s | 512 | 22.4 M | 26.4 | 28.3 | 16.4 | 1.4e-2 |
| efficientvit_sam | l0 | 512 | 30.7 M | 30.2 | 40.4 | 26.0 | nan |
| efficientvit_sam | l1 | 512 | 43.6 M | 35.3 | 52.0 | 31.4 | nan |
| efficientvit_sam | xl0 | 1024 | 112.9 M | 80.7 | 117.8 | 70.5 | nan |

EdgeTAM's torch encoder is slower than its ONNX graph because the export reparameterizes the RepViT trunk (`timm.utils.model.reparameterize_model`, checked against the unfused model) and the predictor does not. The EfficientViT rows are the authors' graphs with their neck's three bicubic `Resize` nodes switched to linear (`tools/onnx_resize_linear.py`, writing `<model>_encoder_linear.onnx`): DirectML has no cubic resize, so ORT ran those three nodes on the CPU with a host round trip, 432 of the 421 to 484 ms the unpatched graphs took (profiled, everything else on the GPU in about 105 ms), and the accuracy rows above are torch bicubic, so the linear graphs are timing only. Their fp16 graphs returned NaN on the random feed the harness gives an input not named `image`, so that column is timing only too.

The decoder, DirectML, 8 points, median ms, with the three feature tensors copied from the host per call, bound once as DirectML `OrtValue`s, and the fp16 graph:

| decoder | host copy | bound | fp16 |
|---|---|---|---|
| sam2 tiny | 12.4 | 9.9 | 9.9 |
| sam2 small | 11.1 | 9.6 | 10.0 |
| efficientvit_sam l0 | 8.7 | | 6.3 |
| efficientvit_sam xl0 | 8.6 | | 6.7 |

EdgeTAM's frame rows use SAM2 tiny's decoder, the same mask decoder architecture, since its own was not exported, and EfficientTAM's decoder, which takes no high-resolution features, was not exported either. Binding the features saves 1.5 to 2.5 ms per call with identical output, and fp16 saves about the same, with the masks within 0.2 to 0.4 percent of the fp32 range and `iou_scores` within 0.5 percent on tiny and 6 percent on small (the matte chooses candidates by IoU against the mesh, not by that score). A frame at 1, 4 and 7 runs, encoder plus bound decoder calls: small fp16 1024 72.8 / 101.7 / 130.6 ms, tiny fp16 512 25.7 / 55.3 / 84.8, EdgeTAM fp16 23.1 / 51.7 / 80.4. Past one run the decoder is the larger term on every encoder, and the seven runs a take frame carries cost about 67 ms of decoder under this load whatever the encoder, so per-frame segmentation needs the runs' prompts batched into one decoder call (the mask decoder takes a batch of prompts; the export's batch axis is 1) as well as a lighter encoder, and neither alone fits the budget.

The fp16 graphs: `onnxconverter_common` leaves `Cast(to=FLOAT)` nodes that fail type checking, and `onnxruntime.transformers.float16.convert_float_to_float16` with `keep_io_types` works. The encoder's positional-embedding `Resize` is a constant chain ORT cannot fold in fp16, and left in place it ran on the CPU at 1.2 s per call, so `--fp16` constant-folds the fp32 graph offline first (a CPU session at basic optimization writing `optimized_model_filepath`), after which the whole encoder is one DirectML fused node. The converted graph loads on onnxruntime-directml 1.20.1 and fails on 1.24.4 with `HasExternalDataInMemory` unless graph optimization is disabled, which is why `.venv_sam2_dml120` exists. Tracking under CUDA for the record, the step per frame over 59 frames under the same load: EdgeTAM 73 ms, EfficientTAM ti 512 44.5 and s 512 47.6, SAM2 small 152 (56 quiet above).

Setup notes. The sandbox refused editable installs of the EdgeTAM and EfficientTAM clones, so `sam2_edge_eval.py` puts the clone on `sys.path`, as `methods/promptda.py` does, and their venvs hold PyPI packages only. EfficientTAM's configs `torch.compile` the image encoder, which has no inductor backend on Windows, so the harness overrides `compile_image_encoder` to false. EdgeTAM's pinned timm imports names `huggingface_hub` 1.x removed, which surfaces as "Hugging Face hub model specified but package not installed". `efficientvit` imports triton at module level for a kernel the SAM models never call, satisfied by `triton-windows`.

The direction, then, is not a lighter encoder per frame but SAM2 small at 1024 in fp16 with its features kept on the device, run on keyframes, and the labels reprojected through the mesh depth on the frames between, with the runs' prompts batched per keyframe. How far a keyframe's labels stay valid is measured in the next section, and it is one frame on the can.

## Keyframe reprojection of the matte (`tools/keyframe_reproject_eval.py`)

The lighter-segmenter pass left keyframe segmentation as the direction: SAM2 small on a keyframe, its corrections carried through the mesh to the frames between. This measures how far that carry holds, on the can take (`take_20261027_232002`, `erosion_V2.glb`, the 481 frames from 2 to 18 s) and the duster take (`take_20261029_143651` with the mesh built from it, the 601 frames from 4 to 24 s). Every frame gets the mesh ray cast from its pose, a prior without a marked mask (the marked frames' occluder pixels at the mesh's depth unprojected into one point cloud of about 3,000 points, its convex hull in each frame, matching the marked regions at IoU 0.70 to 0.87 and the marked folders' depth prior, threshold and occluding contour count within ten percent), and the design's own segmentation of that frame alone (`runs_pos`, SAM2 small, crop framing). From a keyframe t, frame t+k is given four ways: `hold` (the keyframe's mask as it is), `mesh` (the mesh prior rendered at t+k, uncorrected), `mask_warp` (the keyframe's mask pixels unprojected with the keyframe's mesh depth, a pixel the mesh misses taking the nearest mesh depth, projected into t+k and splatted) and `correction_warp` (the mesh re-rendered at t+k plus the keyframe's added pixels and minus its removed pixels, each carried as 3D points the same way within 64 px of the keyframe's occluding contours, a point dropped where it lands more than 2 cm behind the mesh at t+k). Carrying the mesh prior through its own depth lands within 0 px of the mesh rendered at the target up to five frames and 2 px at thirty, so the reprojection itself is exact, and a keyframe carried to itself scores 0 px. Truth is the image-mode mask of frame t+k, for keyframes every fifth frame, and the hand-marked frames from the keyframe k before each. Median / p90 / within 5 px of the occluding edge, averaged over the pairs, with the camera's median translation and rotation over k frames, on the takes' recorded ARKit poses (the ones a live path has):

| k | motion | can `hold` | can `mesh` | can `mask_warp` | can `correction_warp` | duster `mesh` | duster `mask_warp` | duster `correction_warp` |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.8 cm, 0.8 deg | 8.7 / 14.2 / 0.47 | 15.9 / 34.1 / 0.17 | 2.4 / 9.3 / 0.85 | 2.1 / 8.4 / 0.85 | 12.1 / 34.2 / 0.32 | 1.8 / 9.0 / 0.84 | 1.7 / 9.8 / 0.85 |
| 2 | 1.6 cm, 1.6 deg | 16.5 / 26.1 / 0.26 | 16.1 / 35.9 / 0.17 | 3.5 / 13.6 / 0.74 | 3.0 / 12.1 / 0.74 | 12.6 / 36.3 / 0.30 | 2.9 / 12.4 / 0.79 | 2.6 / 14.0 / 0.78 |
| 3 | 2.3 cm, 2.4 deg | 21.7 / 34.0 / 0.17 | 15.9 / 34.6 / 0.17 | 4.0 / 15.4 / 0.66 | 3.6 / 13.6 / 0.67 | 11.9 / 36.7 / 0.32 | 3.1 / 14.5 / 0.76 | 2.8 / 15.0 / 0.76 |
| 5 | 3.8 cm, 4.0 deg | 34.1 / 55.9 / 0.13 | 16.2 / 37.1 / 0.16 | 5.4 / 22.0 / 0.56 | 4.5 / 18.7 / 0.57 | 12.3 / 36.1 / 0.31 | 4.6 / 18.0 / 0.68 | 4.0 / 17.2 / 0.70 |
| 10 | 7.5 cm, 7.8 deg | 60.2 / 100.5 / 0.09 | 16.3 / 36.9 / 0.16 | 8.8 / 30.2 / 0.43 | 6.7 / 23.8 / 0.45 | 12.4 / 36.2 / 0.31 | 7.3 / 25.2 / 0.54 | 5.2 / 21.2 / 0.60 |
| 30 | 21 cm, 25 deg | 279 / 332 / 0.04 | 16.2 / 35.0 / 0.17 | 18.4 / 47.4 / 0.21 | 11.8 / 32.1 / 0.30 | 12.4 / 36.5 / 0.31 | 14.0 / 38.1 / 0.35 | 6.7 / 25.8 / 0.51 |
| 60 | 41 cm, 50 deg | 587 / 631 / 0.03 | 16.5 / 32.0 / 0.16 | 24.2 / 54.3 / 0.13 | 14.8 / 31.2 / 0.24 | 12.5 / 37.0 / 0.32 | 20.6 / 62.2 / 0.24 | 9.1 / 29.3 / 0.41 |

Against the hand-marked frames, `correction_warp` on the can (three pairs per k) is 2.4 / 9.7 / 0.81 at one frame, 3.7 / 11.9 / 0.66 at two and 6.1 / 21.4 / 0.42 at five, where those frames' own image-mode mask is 1.0 / 3.8 / 0.94; on the duster (four pairs) it is 2.2 / 4.6 / 0.92 at one, 2.1 / 5.1 / 0.89 at two, 2.6 / 6.5 / 0.83 at three, 3.1 / 9.3 / 0.75 at five and 3.4 / 11.6 / 0.70 at ten, against 1.8 / 3.8 / 0.93 for image mode and 5.5 / 27.3 / 0.51 for the mesh. The full tables are in `out/keyframe_reproject_recorded/summary.md` and `out/keyframe_reproject_duster/summary.md`; `out/keyframe_reproject/summary.md` is the can on the refined `erosion_V2` poses (2.5 / 8.3 / 0.82 at one frame, 5.6 / 15.6 / 0.55 at five).

A keyframe is good for one frame on the can and two or three on the duster. The carried corrections are a frame behind the segmenter's own result at k = 1 on the can (2.1 px median against 1.0, 85 percent within 5 px against 97) and twice as far off by k = 2; on the duster they stay within half a pixel of median and 1.5 px of p90 of image mode through k = 2 and are a pixel behind by k = 3. The camera's motion is not the cause: the reprojection is exact, and the `hold` row shows what the motion alone would cost. The cause is that the corrections are anchored to the mesh, and the mesh slides against the image. Measured as the can's image-mode mask centroid against the projected mesh cloud's, the mesh sits 23 px off the image on median (p90 36) with the recorded poses and 14 px with the refined ones, and that offset changes by 1.5 px per frame median (7 px p90), 3.3 px over five frames and 13.6 over thirty, which is the carried rows' decline. The can's image-space velocity explains none of the offset's variance, so it is not a pose-to-frame timestamp delay (a least-squares delay of 0.3 ms) but the mesh's own misregistration, 7 mm by the earlier marked-frame measurement, projected differently as the view turns 0.8 degrees per frame. The duster's mesh, built from the take it is scored on, sits 5.5 px off its marked frames against the can mesh's 20, and its carry lasts longer in the same proportion, so the interval is set by the mesh's registration, not by the camera. The refined can poses, which pin the mesh closer to the image on average, jitter four times more frame to frame (the step between consecutive poses changes by 0.13 cm median and 0.72 cm p90 against 0.03 and 0.08 for the recorded track), so the carry degrades faster on them past the first frame. `mask_warp` is a little behind `correction_warp` at every k and falls well behind as the view turns (duster, k = 30: 14.0 against 6.7 px), since re-rendering the mesh keeps the parts of the silhouette the keyframe never saw.

The reading against the lighter segmenters: carrying SAM2 small's keyframe two frames on the can (3.0 / 12.1 / 0.74) is what EdgeTAM gives per frame on the whole frame (2.8 / 12.2 / 0.80), and EfficientViT-SAM l0 per frame (1.6 / 6.1 / 0.89) is better than carrying even one. So the keyframe design as measured does not buy quality over a lighter per-frame segmenter unless the mesh is registered to the duster's standard, and a live matte at the keyframe's quality needs either the segmenter on every frame or a per-frame image-space re-anchoring of the carried silhouette to the video's edges, which the measured slide (a smooth few pixels per frame, 7 px p90) is small enough to permit and which plan.md already names for the mesh itself. That re-anchoring is measured next.

Re-anchoring the carried silhouette to the video's edges (the same tool, on by default) recovers little. The alignment is an oriented chamfer: Canny edges (20 and 60 after a 1 px blur, where the segmenter's own silhouette has an edge within 2 px on 74 to 86 percent of its points, against 60 percent at 40 and 120) binned by gradient orientation into eight bins with one distance transform per bin truncated at 8 px, so a contour point is measured only against edges that run its way and a point with no edge near it neither flattens nor drags the search. The carried mask's occluding outline is searched over integer translations within 16 px (40 for the bare mesh, which sits farther off), refined to sub-pixel, and a shift is rejected unless it gains half a pixel of mean edge distance over no shift. Three scopes: one translation for the whole contour (`correction_anchored`, `mesh_anchored`), one per occluding contour run substituted within that run's band (`correction_anchored_runs`, `mesh_anchored_runs`), and a chain that carries one frame, re-anchors per run within 8 px, rebuilds the keyframe from the result and carries again (`correction_chained`). The image-mode mask re-anchored to its own frame does not move on 98 to 100 percent of frames. Against the image-mode mask of frame t+k, recorded poses:

| k | can carried | can anchored per run | can chained | duster carried | duster anchored per run | duster chained |
|---|---|---|---|---|---|---|
| 1 | 2.1 / 8.4 / 0.85 | 2.1 / 8.5 / 0.86 | 1.9 / 8.1 / 0.87 | 1.7 / 9.8 / 0.85 | 2.0 / 10.6 / 0.83 | 1.7 / 10.0 / 0.84 |
| 2 | 3.0 / 12.1 / 0.74 | 2.8 / 11.5 / 0.79 | 2.5 / 11.2 / 0.82 | 2.6 / 14.0 / 0.78 | 2.8 / 14.0 / 0.76 | 2.6 / 14.0 / 0.78 |
| 3 | 3.6 / 13.6 / 0.67 | 3.0 / 12.4 / 0.73 | 2.8 / 11.6 / 0.76 | 2.8 / 15.0 / 0.76 | 2.8 / 14.8 / 0.76 | 2.8 / 15.2 / 0.76 |
| 5 | 4.5 / 18.7 / 0.57 | 3.9 / 16.8 / 0.65 | 3.6 / 16.5 / 0.68 | 4.0 / 17.2 / 0.70 | 3.8 / 17.1 / 0.71 | 3.8 / 16.9 / 0.71 |
| 10 | 6.7 / 23.8 / 0.45 | 5.2 / 21.1 / 0.54 | 4.6 / 21.4 / 0.58 | 5.2 / 21.2 / 0.60 | 4.8 / 20.6 / 0.63 | 4.6 / 20.3 / 0.64 |
| 30 | 11.8 / 32.1 / 0.30 | 8.6 / 29.5 / 0.40 | | 6.7 / 25.8 / 0.51 | 6.0 / 24.6 / 0.54 | |

The global scope rarely fires (rejected on 41 percent of the can's frames and 76 percent of the duster's) and changes nothing where it does, since the carried contour mixes the object's few pixels of slide with the desk's 18 px and one translation serves neither. Per run, the gain is half a pixel of median and 5 to 10 points of the share within 5 px on the can at k up to 10, and nothing on the duster below k = 5; the chain adds a little on the can and nothing on the duster, where it rejected 69 percent of its shifts. The hand-marked frames say the same: per run at one frame the can is 1.7 / 10.2 / 0.84 and the duster 1.9 / 4.7 / 0.92, against 1.0 / 3.8 / 0.94 and 1.8 / 3.8 / 0.93 for segmenting the frame itself. The bare mesh re-anchored, plan.md's silhouette alignment on its own, goes from 16 to 10 px median on the can with its shifts at the 40 px window's edge, and from 12 to 11 on the duster. What remains after the best shift is not a translation. The carried silhouette is ragged from the splat, carries the keyframe's own segmentation errors forward (a white object behind the can that one keyframe's mask took in), finds no edge where the frame is motion blurred (the segmenter's silhouette has no edge within 5 px on a blurred frame at any threshold), and the mesh's slide differs along one contour. The proxy truth also flickers (the segmenter's mask carried one frame overlaps the next frame's at IoU 0.92), which is part of the one-frame residual on both sides. So re-anchoring by translation does not make SAM2 small keyframes viable for the live matte, and the per-frame segmenter stands as the path. If the keyframe route is taken up again, the measured residual points at a local snap of each carried contour point to its own edge, a few pixels away rather than the 20 the band methods faced before, which is unmeasured.

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
