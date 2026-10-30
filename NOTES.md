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

Cost on the RTX 3090 (`tools/sam2_timing.py`, CUDA, bfloat16, the 1920x1440 frame through the 1024 encoder, median of 50): encoder plus decoder tiny 26.0 + 8.6 ms, small 29.9 + 8.5, base_plus 40.4 + 8.8, large 84.3 + 8.7. The tracker's step per frame (`propagate_in_video`, including the mask resized to the video size on the host): tiny 52 ms, small 56, base_plus 66, large 109. Under DirectML, the runtime the editor uses (the image encoder exported to ONNX by `tools/sam2_export_encoder.py`, matching torch to 1e-5, timed by `tools/sam2_dml_timing.py` on onnxruntime-directml 1.24.4): tiny 67 ms, small 75, about 2.5 times CUDA. A bare `image_size` 512 override fails in sam2 1.1.0 with a view over a non-contiguous tensor in the Hiera trunk, so the smaller real-time configuration was not timed.

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
