# Twinfit · Milestone 1 · Perception: video to metric part tracks

How to start: run Claude Code inside `../twinfit-m1` (branch `m1-perception`) and say "Read prompts/01_m1_perception.md and carry it out exactly."

<role_and_principles>
You own Twinfit's perception stage. It turns a phone video plus its metadata into metric pose tracks for each rigid part, written in the frozen PartTracks contract.

Two other sessions build in parallel against the same contracts: identification (Milestone 2) and verification plus the agent (Milestone 3). You never need their code, and you must not touch it.

Operating principles:
1. Meta models are the perception front end, and judges score how they are used (25% of the grade).
   - Muse Wonder proposes the parts, the base, the connectivity and the release events.
   - SAM 3.1 segments and tracks the parts through the video.
   - Classical geometry turns masks into metric motion: calibration, board pose, plane homography and rigid fitting.
   - Every piece must be measurable against synthetic ground truth.
2. Read the Meta docs before writing any client code.
   - Never guess request shapes, parameter names, upload flows or output formats.
   - If the docs and a live call disagree, trust the live call and record the discrepancy.
3. Spend deliberately:
   - Cache every call.
   - Try each call on synth_tiny before synth_small.
   - Measure cost per clip before running a whole suite.
4. Report measured numbers only.
   - If you miss a target, report the actual value and the dominant error source.
   - Never tune thresholds or metrics on the evaluation set to pass.
5. Work autonomously with a task list, and stop only for <stop_conditions>.
</role_and_principles>

<context>
Twinfit turns phone video of a real mechanism into a MuJoCo model plus a trust certificate. It is built solo by Ahaan Kothari for the Meta Global AI Developer Hackathon, with a deadline of Wed Nov 4, 2026, 9:00 PM ET. The repository is public, and all work must be built during the event.

Before planning, read these files. They define frames, units, file layouts and the stage interface, and they are frozen:
- CLAUDE.md
- docs/ARCHITECTURE.md
- docs/meta_api_notes.md
- docs/TEST_DATA_PROTOCOL.md
- `src/twinfit/contracts/`

Your stage:
- Signature: `twinfit.perception.stage:run(ctx: StageContext, options: PerceptionOptions) -> StageResult`.
- Required outputs, written into `ctx.stage_dir`: `tracks.json` and `tracks.npz`.
- Optional outputs: `masks/`, `overlay.mp4` and `perception_report.json`.

Scope:
- In scope: planar mechanisms (motion_model "planar"), with the motion plane parallel to the board at `plane_offset_m`.
- Out of scope: markerless spatial motion. For spatial clips, return status failed with the message "markerless spatial perception not supported", unless `use_part_markers` is true and markers are assigned.

Data:
- Synthetic clips live in `~/twinfit-data/synth/{synth_tiny,synth_small}`, with full ground truth under `gt/`: tracks, masks and camera extrinsics. Get the paths with `twinfit.synth.suite_dir`.
- Real clips live in `~/twinfit-data/real/<clip_id>/`.
- Calibration videos live in `~/twinfit-data/real/_calibration/<camera_id>/`. They may not exist yet, because Ahaan records them during your session.

Budget: tag m1-dev has a $45 cap. The global $120 cap is shared by all sessions.
</context>

<instructions>

Phase 0. Orientation and plan
1. Create `.twinfit-session.yaml` in this worktree with `tag: m1-dev`, `tag_cap_usd: 45` and `allow_spend: true`. Then run `uv sync --all-extras --dev` and `uv run twinfit doctor`.
2. Read the files listed in <context>.
3. Read these Meta docs with WebFetch, all under https://dev.meta.ai:
   - /docs/sam/overview, /docs/sam/segmenting, /docs/sam/reading-segmentation, /docs/sam/client-libraries
   - /docs/media-segmentation, /docs/video-understanding, /docs/file-handling
   - /docs/structured-output, /docs/protocols/responses, /docs/reasoning
   - /docs/pricing-rate-limits, /docs/error-handling

   If a page does not render, use https://github.com/meta-models/meta-model-cookbook.
4. Write an "API notes" section in `docs/m1_perception.md`. Include every fact you will rely on, each with its source URL:
   - endpoints and payload shapes
   - limits and prices
   - the SAM client library name and its parsing API
5. Write a plan in `docs/ledger/m1.md`: modules, build order, how each module is verified, and the estimated cost of one synth_small pass.

Phase 1. Video ingest, calibration and board pose (no API calls)
1. Write `perception/video.py`:
   - Probe the video with the ffmpeg bundled by imageio-ffmpeg, and read per-frame timestamps.
   - Normalize any input (for example HEVC .mov with variable frame rate) to an H.264 mp4 working copy in the stage folder, keeping the source frame rate. Use `-preset veryfast`.
   - Refuse clips whose effective frame rate is below 24 fps.
   - For real clips, warn when the frame rate is below 100 fps, because dynamics needs high frame rates.
2. Write `perception/calibration.py` and the command `twinfit perception calibrate CALIB_VIDEO --camera-id ID`:
   - Detect ChArUco corners, using the board spec from the template metadata or the defaults.
   - Select up to 60 frames spread across image regions and tilts.
   - Run `cv2.calibrateCamera`.
   - Write `camera_intrinsics.json` next to the video.
   - Print the RMS reprojection error and the coverage. Warn above 1.0 px.
3. Validate calibration in `tests/perception/` with synthetic frames:
   - Warp the generated board image with a known K and 40 known poses (cv2.warpPerspective), adding mild noise and blur.
   - Recover fx and fy within 1%, and cx and cy within 3 px.
4. Write `perception/board.py`:
   - Estimate the board pose per frame: ChArUco detection, `solvePnP`, then refinement.
   - Compute a robust static estimate of T_W_C, plus a static flag.
   - The camera is not static if any frame deviates by more than 2 mm or 0.2°. In that case, keep the per-frame poses.
   - Validate against `gt/camera_extrinsics.json` on synth_small.

Phase 2. SAM 3.1 segmentation
1. Write `perception/sam.py`. Following the docs exactly:
   - Upload the working video.
   - Request video segmentation, prompted with one box per part at a given frame. Prefer box prompts over noun phrases.
   - Parse the masks with the SAM client library.
   - Return a per-frame label image: 0 for background, and part index + 1 for each part.
2. Route every call through `twinfit.meta`:
   - budget checks
   - ledger records with kind `sam_video`
   - caching, with a key that includes the video sha256, the prompts and the stride
3. Default SAM stride: the smallest integer s with fps/s ≤ 60. Masks for unsampled frames come from the nearest sampled frame. This is separate from `PerceptionOptions.frame_stride`, which subsamples the whole video before any processing (default 1, all frames); the SAM stride applies on top of it.
4. If the docs give limits on duration, file size or frame count, split long videos into overlapping chunks. Stitch identities across chunks by mask IoU in the overlap.
5. Make the first live call on one synth_tiny clip, and record its actual cost and latency.
6. Then evaluate on the synth_small planar clips, with GT-derived boxes passed as `box_prompts` by the evaluation script. The stage itself never reads gt. Targets are in <acceptance_criteria>.

Phase 3. Muse Wonder part proposal
1. Write `perception/propose.py`. It sends Muse Wonder the clip together with `metadata.parts_hint`.
   - The clip is either a downscaled working copy or a set of keyframes. Decide by measuring tokens and cost on one synth clip first.
   - Request structured output with this schema:
     - `parts`: list of {label, sam_noun_phrase, frame_index, box_xyxy_px, is_base, notes}
     - `connections`: list of {a_label, b_label, joint_type_guess (revolute, prismatic, fixed or unknown)}
     - `free_motion_segments_s`: list of [t0, t1]
     - `confidence`: 0 to 1
     - `cannot_determine`: list of str
2. Validate the response with pydantic.
   - Every box must lie inside the image and cover at least 0.05% of its area.
   - On a validation failure, retry once with the validation error appended.
   - If the retry also fails, fail the stage with the error.
3. The prompt must give Wonder explicit permission to say it cannot determine something by filling `cannot_determine`, instead of guessing. It must also tell Wonder its output is a hypothesis that is checked downstream.
4. Store the proposed connections and joint-type guesses in the `tracks.json` extras under `llm_connections`.
5. Store the proposed free-motion segments in `free_motion_segments_s`, with `free_motion_source = "llm"`, only when the metadata has none.

Phase 4. Metric planar tracking
1. Write `perception/track.py`. For each part:
   - Seed Shi-Tomasi corners inside the eroded mask.
   - Track at full frame rate with pyramidal Lucas-Kanade, using a forward-backward check (error < 0.5 px).
   - Reseed when inliers fall below 30, or every 0.5 s.
   - Reject points that fall outside the current mask.
2. Convert image points to metric plane motion:
   - Map image points to the motion plane using K, the distortion, T_W_C and `plane_offset_m`, through the `twinfit.geometry.camera` helpers.
   - Fit an in-plane SE(2) transform between the reference frame and each frame, using RANSAC plus Kabsch (no scale), then refine on the inliers.
   - Lift the result to an SE(3) T_W_P with the plane normal fixed.
   - Define the part frame P with its origin at the in-plane centroid of the part's points at the reference frame, and its axes aligned with W at the reference frame.
3. Set quality to the RMS in-plane residual of the inliers, in metres. A frame is invalid when it has fewer than 6 inliers or a residual above 3 mm.
4. For textureless parts where feature tracking fails, fall back to mask-based rigid registration: in-plane ICP on mask boundary points. Mark the frames that use this fallback in extras.
5. Set extent_P to the bounding box of the part's plane points in P at the reference frame. Use a z extent of ±0.005 m, and set `extras.extent_z_assumed = true`.
6. Set `is_base_hint` for parts that are flagged `is_base` (by the LLM or by parts_hint) and show near-zero motion.

Phase 5. Marker ground truth for real Take A clips
1. Add the command `twinfit perception marker-gt CLIP_DIR`:
   - Detect DICT_4X4_50 markers.
   - Estimate each assigned part's pose per frame with solvePnP, using IPPE_SQUARE.
   - Express each pose in W through the board pose.
   - Write `gt/tracks.json` and `gt/tracks.npz` with source "marker_gt".
2. Validate with a synthetic test: warp a marker image into frames with known poses, and recover the poses within 1 mm and 0.5°.
3. With `use_part_markers: true`, the stage may use marker poses directly. This also enables spatial clips. Flag it in extras.

Phase 6. Stage, overlay and evaluation
1. Write `perception/stage.py:run` to wire everything together. Honor every PerceptionOptions field:
   - `part_prompts`
   - `box_prompts` (skip Wonder when these are given)
   - `frame_stride` and `max_frames`
   - `exclude_labels`
   - `use_part_markers`
   - `llm_part_proposal`
2. Write `overlay.mp4`, drawn per frame, encoded as H.264 with `-preset veryfast`. It shows:
   - the masks
   - the tracked points
   - the part frame axes
   - the board axes
3. Write `perception_report.json` with:
   - each part's valid fraction and median quality
   - the number of fallback frames
   - cost by model
   - timings
4. Add the command `twinfit perception eval --suite synth_small [--boxes oracle|llm]`:
   - Run the stage on every planar clip.
   - Compute board pose error, mask IoU, motion point RMS (from `twinfit.eval.metrics`), valid fraction and cost.
   - Write `docs/media/m1_eval_<boxes>.csv`, and a summary table in `docs/m1_perception.md`.
5. Add a gt-guard test:
   - Run the stage on a copy of a clip folder with `gt/` removed, and on the original.
   - When API responses come from the cache, the two tracks must be identical.

Phase 7. Real clips (when present in ~/twinfit-data/real)
1. Calibrate each camera_id that has a calibration video.
2. For each Take A clip, run marker-gt, then run the stage markerless, and compare the result to the marker GT.
3. For each Take B clip, run the stage and record its valid fraction and quality.
4. Put a results table in `docs/m1_perception.md`. If there are no real clips yet, say so and list exactly what you need from Ahaan.

Priorities:
- P0:
  - Phases 0, 1, 2, 4 and 6 on synthetic clips, with oracle boxes.
  - Chunking, if the documented limits require it.
- P1:
  - Phase 3 (Wonder proposals).
  - Phase 5 (marker ground truth).
  - Phase 7 (real clips).
- P2:
  - The mask-registration fallback.
  - Spatial clips with markers.
</instructions>

<constraints>
- Edit only these paths:
  - src/twinfit/perception/
  - tests/perception/
  - docs/m1_perception.md
  - docs/ledger/m1.md
  - docs/media/m1_*
  - the `perception` optional-dependency group in pyproject.toml, plus uv.lock
- Contracts are frozen. If one is insufficient, write `docs/contract_change_requests/m1.md` and use extras in the meantime.
- The stage never reads gt/. Only evaluation scripts and tests may read it.
- Every Meta call goes through `twinfit.meta`, or through `raw_client` together with `record_cost` and `budget.check`. Every call is tagged m1-dev and cached.
- No Meta calls in non-live tests. Use recorded fixtures instead.
- Use only Meta models for segmentation and part proposal. Do not substitute another vendor's model.
- Do not change how Claude Code itself is configured or routed.
- Follow the code style in CLAUDE.md. Prefix commits with [m1], and commit after each phase.
</constraints>

<acceptance_criteria>
Unless stated otherwise, these apply to the 10 synth_small planar clips:
1. Board pose error is ≤ 0.5° and ≤ 5 mm on every clip.
2. Synthetic calibration recovers fx and fy within 1%, cx and cy within 3 px, with RMS ≤ 0.5 px.
3. Segmentation with oracle boxes reaches mean mask IoU ≥ 0.85 for every moving part. Report it per clip.
4. Tracking with SAM masks:
   - the median over clips of the per-part motion point RMS is ≤ 3 mm
   - each moving part has a valid fraction ≥ 0.95
5. `uv run twinfit contracts-check` passes on every run folder.
6. The gt-guard test passes.
7. Wonder proposals (with no boxes given) get the part count right on at least 8 of 10 clips. Report how IoU and RMS change compared with oracle boxes.
8. Cost per clip is measured, with SAM and Wonder reported separately. The tag m1-dev total is ≤ $45.
9. `uv run pytest -q` and `uv run ruff check .` pass in this worktree.
10. Real data (only if present):
    - calibration RMS per camera
    - for Take A clips, markerless versus marker-GT motion point RMS per part, with a target of ≤ 5 mm (report the actual value)
</acceptance_criteria>

<self_check>
Before the final report, answer each of these in docs/ledger/m1.md:
- Does a re-run of the eval with cache hits reproduce the same numbers?
- Did any threshold or metric change after you saw evaluation results? If so, revert it or justify it.
- Does `git diff --name-only main...HEAD` list only owned paths?
- Is every API fact in docs/m1_perception.md backed by a URL or a recorded live response?
- Which statements in your report are unverified? Mark them.
</self_check>

<stop_conditions>
1. BillingError or a missing key: stop live work, finish the offline phases, and report.
2. The tag cap or the global cap is reached: stop live calls, and report what remains.
3. SAM 3.1 or Wonder cannot do something this plan requires (for example, no box prompts for video):
   - Stop that path.
   - Record what the docs say.
   - Propose the nearest Meta-model alternative in the report.
   - Continue the other phases.
4. Ahaan's real clips are missing or malformed: do not stop. List the exact problems in the report.
</stop_conditions>

<output_format>
```
## Milestone 1 report
Status: COMPLETE (all P0 and acceptance met) | PARTIAL (what is missing)
Acceptance: table with columns criterion | measured | target | pass/fail
Cost: per clip (SAM, Wonder); tag m1-dev total; global total
API discrepancies found: list
What Ahaan needs to record or fix: list
Contract change requests: none, or the file path
Unverified claims: list
```
</output_format>
