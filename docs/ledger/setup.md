# Setup ledger

Session: Prompt 0, 2026-10-31. One entry per phase: plan first, result after.

## Phase 0. Preflight

Plan: check git, uv, gh, gh auth, disk space, and the secrets file. Verify by reading each command's output.

Result:
- git 2.54.0, uv 0.11.7 (Homebrew), gh 2.92.0, gh authenticated as ahaan001 (scopes gist, read:org, repo, workflow).
- Free disk: 596 GB.
- `~/.config/twinfit/.env` did not exist. Created it with `MODEL_API_KEY=` (empty) and `TWINFIT_BUDGET_USD=120`, mode 600. Live Meta calls will fail until Ahaan fills the key.
- Anaconda python is first on PATH. All project Python runs through `uv run`.
- All seven input files present and read in full.

## Phase 1. Skeleton

Plan: git init on main, move prompts, runbook and the private context file, write LICENSE, .gitignore, .env.example and docs/meta_api_notes.md (sections 1 to 11 of the context minus anything personal). Verify by listing the tree and grepping meta_api_notes.md for the removed identifiers.

Result:
- Repo initialised on `main`. Prompts in `prompts/`, runbook at `docs/RUNBOOK.md`, context file at `.private/hackathon_context.md` (ignored by git).
- `docs/meta_api_notes.md` written from sections 1 to 11. A grep for the email, team_id, project_id, key name, masked key and account statuses finds nothing.
- LICENSE (MIT), .gitignore, .env.example written.

## Phase 2. Environment

Plan: pin Python 3.11, write pyproject (hatchling, src layout, lower-bound deps, dev group, empty optional groups, pytest and ruff config), `uv sync --all-extras --dev`, then import cv2.aruco, mujoco, pinocchio, openai and run `imageio_ffmpeg.get_ffmpeg_exe()`.

Result:
- Python 3.11.15 via uv. Locked versions: numpy 2.4.6, scipy 1.17.1, pydantic 2.14.2, opencv-contrib-python 5.0.0, mujoco 3.17.0, pin 4.1.0, openai 3.42.0, imageio 2.38.1, pytest 9.2.0, ruff 0.17.3.
- All imports succeed on the first attempt, including Pinocchio. ffmpeg 7.1 bundled binary runs.
- Library facts confirmed by running code (not assumed):
  - `cv2.aruco.CharucoBoard((7, 5), 0.035, 0.026, dict).getChessboardCorners()` returns 24 corners; corner id 0 is at (0.035, 0.035, 0) and in `generateImage` with zero margin it lands at the top-left interior corner (pixel 99.5, 99.5 at 100 px per square). Pixel u grows with W x, pixel v (downward) grows with W y.
  - `cv2.aruco.CharucoDetector(board).detectBoard(img)` returns (corners, ids, marker_corners, marker_ids); `board.matchImagePoints(corners, ids)` gives (objPoints, imgPoints) for solvePnP.
  - MuJoCo `springref` is parsed in degrees unless `<compiler angle="radian"/>` is set. All generated MJCF sets radian.
  - With `<flag energy="enable"/>`, `data.energy[0]` (potential) includes the joint spring term 0.5 k (q - springref)^2 together with gravity; `data.energy[1]` is kinetic. Verified numerically (0.2474 J for k=0.5, q=1 rad, springref=0.3 deg).
  - `mujoco.Renderer(model, height, width)`; `enable_segmentation_rendering()` returns an (H, W, 2) int32 image: channel 0 is the object id, channel 1 the object type (5 = mjOBJ_GEOM), -1 for background.
  - MuJoCo's `camprojection` sensor matches K = [[fy, 0, W/2], [0, fy, H/2], [0, 0, 1]] with fy = (H/2)/tan(fovy/2) and R_cv = R_mj diag(1, -1, -1) to 1e-9 px. Pixel-centre convention is settled by the board test in Phase 6.
  - Builtin texture attributes `builtin="flat" mark="random" random=... markrgb=...` are accepted by MuJoCo 3.17.
  - openai 3.42 vendors httpx as `httpx2`. `openai.RateLimitError(message, response=httpx2.Response, body=None)` constructs test errors; `.status_code` and `.response.headers` are available. `OpenAI(base_url=..., api_key=..., max_retries=0)` disables the SDK's own retries so the client's backoff is the only one.
  - `ResponseUsage` has `input_tokens`, `output_tokens`, `input_tokens_details.cached_tokens`.

## Phase 3. Paths and session

Plan: `paths.py` (data_dir with subfolders, repo_root from git, runs_dir, ledger_path), `session.py` (session yaml with env overrides, dotenv secrets). Verify with unit tests using tmp dirs and monkeypatched env.

## Phase 4. Contracts

Plan: pydantic v2 models in `contracts/` exactly as the spec, `conventions.py` with the verbatim conventions and board constants computed at import by detecting the generated board image, `io.py` with validating load/save and a deterministic npz writer (fixed zip timestamps so files are byte-identical), `stage.py` with the stage interface and options. `twinfit contracts-check PATH`. Verify with round-trip tests and malformed-array rejection tests, plus a test that the frame constants match a hand derivation.


Result (Phases 3 and 4):
- `paths.py`, `session.py` written; the session file for this worktree has tag setup, cap $1, allow_spend true.
- Contracts: `models.py` (all six file models plus `compute_verdict`), `stage.py` (stage names, folders, output keys, context, result, options), `io.py` (validating load/save, deterministic npz), `check.py` (contracts-check), `conventions.py` (verbatim conventions, board constants, import-time frame derivation).
- Empirical board frame, from detecting the generated zero-margin board image: pixel u (right) grows with W x, pixel v (down the page) grows with W y. Hence up_W = (0, -1, 0) and normal_toward_camera_W = (0, 0, -1): +z points into the printed face. Tests compare the import-time map against a detection at a different resolution and against the hand derivation.
- 33 tests pass (round trips, malformed arrays rejected, verdict rules, options defaults, npz determinism, session and paths).
- OpenCV 5.0 quirk: `CharucoBoard.generateImage` fails an internal ROI assertion for some sizes (60 and 200 px per square); 40, 50, 70, 80, 100, 120 and 140 work. Noted in conventions.py.

## Phase 5. Geometry and simulation utilities

Plan: `geometry/se3.py`, `geometry/camera.py`, `geometry/lines.py`, `sim/replay.py` (simulate, windowed_replay_rms), `sim/energy.py`. Verify with exp/log round trips, a hand-computed projection, homography against direct projection, energy conservation of an undamped pendulum, and `simulate` against plain `mj_step`.

Result:
- All written. `simulate` lands exactly on requested times by taking full steps of the timestep and one shorter final step; `windowed_replay_rms` tiles each segment with windows [t_start, t_start + window_s) and restarts from the observed state at each window start.
- Energy: MuJoCo's potential term already includes the spring energy (test compares against m g h + 0.5 k (q - ref)^2 + 0.5 I w^2 to 1e-9 relative), so energy.py adds nothing.
- Undamped pendulum over 5 s at RK4 2e-4: relative energy drift below 1e-3 (test).
- `so3_log` needed a sign fix near theta = pi (axis sign taken from the antisymmetric part); caught by the round-trip test at pi - 1e-7.
- Head-on camera: the OpenCV camera frame of an upright view of the board coincides with W (x right, y down, z into the page), so T_W_C has R = I and a negative z. Documented in the projection test.
- 56 tests pass.

## Phase 6. Synthetic ground-truth generator

Plan: `synth/scenes.py` (six scenes, seed randomisation), `synth/mjcf.py` (mechanism and rendered
scene MJCF), `synth/generate.py` (trajectories, camera sampling, rendering, ground truth files),
`suite_dir`. Verify with the board-frame, projection, FK and simulate self-consistency tests on
synth_small, plus a byte-identity test.

Result:
- Texture mapping settled empirically: a plane geom with quat (0, 1, 0, 0) (180 degrees about x) and
  the unmodified board image renders the board so that detection plus PnP recovers the camera pose
  (0.007 degrees, 1 mm when centred). With the identity quat the image appears mirrored.
- Principal point: with cx = W/2 - 0.5 and cy = H/2 - 0.5 the true pose reprojects detected
  corners at 0.096 px RMS; with W/2, H/2 it is 0.714 px. `K_from_fovy` uses the shifted value.
- Trajectories: hold phase as (q_hold, 0, 0), then `replay.simulate` from (q_hold, 0) at
  t_release, so the GT is reproducible bitwise by the same function (acceptance criterion 6).
- Lamp springref is solved so the chosen bent pose is a static equilibrium: ref = q + g(q)/k with
  g from MuJoCo's qfrc_bias at zero velocity.
- Deviation from the spec, recorded here: seed 0 uses a 7.5 degree camera tilt (midpoint of the
  0 to 15 degree randomisation range) instead of exactly fronto-parallel. Reason: single-frame PnP
  of a fronto-parallel planar target is ill-conditioned (tilt trades off against lateral
  translation). Measured on the pendulum clip: 0.97 degrees and 15 mm error from the video frame,
  0.54 degrees on a raw render, and 0.86 degrees on a pure OpenCV warp with no MuJoCo involved,
  all with 0.1 px corner residuals. With 7.5 degrees of tilt the error is 0.21 degrees and 3.4 mm.
  Marker corners (0.8 px residuals) and corner re-refinement did not help; H.264 at crf 18 adds
  almost nothing (0.10 vs 0.093 px). Real captures are never exactly fronto-parallel either.
- Interpenetration note: the slider block (20 mm thick, offset 15 mm) and the door panel (30 mm
  tall, offset 20 mm) extend behind the board plane as specified; rendering clips them at z = 0.
  The double pendulum and lamp stack their two strips in z (one thickness apart) so the parts can
  overlap at the pin; `plane_offset_m` refers to the front part.
- Generation time: 6 s per synth_small clip (1280x720, 480 frames, RGB plus segmentation), about
  75 s for the suite; 0.5 s per synth_tiny clip.

## Phase 7. Stubs, runner, evaluation, CLI

Plan: `stubs/oracle.py`, `runner/registry.py`, `runner/pipeline.py`, `eval/metrics.py`, `cli.py`.
Verify with tests on a 64x36 generated clip (stub pipeline, contracts-check on the run folder,
oracle noise determinism, missing gt, crash handling, registry fallback and propagation) and
hand-built metric cases.

Result: all written; 15 tests pass. `twinfit doctor` reports every import OK and a successful
offscreen render. Typer parameters use `Annotated`; `B008` is ignored for cli modules so milestone
CLIs can use either style.

## Phase 8. Meta API client

Plan: delegated to a subagent with the exact Phase 8 spec; review its code and tests.

Result: `meta/{config,ledger,budget,cache,client,errors}.py`; 20 mocked tests pass (retry with
Retry-After in seconds and HTTP date, 6 attempts, billing errors for 401/402/403 and payment or
billing messages, two-process ledger concurrency, budget refusals for the global and tag caps,
zero-cost cache hits). SDK facts verified by running code: there is no 402 error class (plain
APIStatusError); `ResponseUsage.input_tokens_details` requires `cache_write_tokens` in SDK 3.42,
so cached responses fall back to `Response.construct` when validation fails. Unverified: live
response shapes (no key on this machine), whether Meta's `cached_tokens` is a subset of
`input_tokens` (assumed, as in OpenAI's convention), the 4 chars per token estimate (pre-call
estimate only; recorded costs come from returned usage).

## Phase 9. Printables and templates

Plan: delegated to a subagent with the exact Phase 9 spec; review the PDFs, the template and the
protocol document.

Result: three PDFs at exact size (US Letter landscape, inch-unit axes, sizes asserted before
saving), the metadata template (validates as ClipMetadata as written), `docs/TEST_DATA_PROTOCOL.md`
with the board axes in plain words. The board figure rendered at 100 and 300 dpi detects with the
expected 35 mm spacing within 0.15 percent. Unverified: a physical print was not measured.

## Phase 10. Repository documents

Plan: CLAUDE.md, docs/ARCHITECTURE.md, README.md skeleton, THIRD_PARTY.md from installed metadata,
CI workflow, tests/setup/test_no_secrets.py.

THIRD_PARTY.md generator (run from the repo root with `uv run python -I script.py .`): reads
`project.dependencies` and the dev group from pyproject.toml and prints `License-Expression`, the
`License` field or the license classifier of each installed distribution.

Result: all written. THIRD_PARTY.md lists 21 direct dependencies with licenses read from the
installed metadata. The secrets test uses `LLM_[0-9]+_` and `MODEL_API_KEY=` followed by a
key-like value (at least six key characters on the same line); the literal `MODEL_API_KEY=.+`
would flag the placeholders in the prompts, the runbook and this ledger.

## Phase 11. Verify, publish, branch

Plan: run every acceptance command, commit in logical `[setup]` commits, create the public GitHub
repository, check CI, create the three worktrees and sync and test each, leave synth_small in the
shared data dir, and set allow_spend false for this worktree.

Acceptance commands and results (2026-10-31):
1. `uv run twinfit doctor`: every import OK (numpy 2.4.6, scipy 1.17.1, cv2 5.0.0, mujoco 3.17.0,
   pinocchio 4.1.0, openai 3.42.0, pydantic 2.14.2, imageio_ffmpeg 0.7.0), ffmpeg found, data dir
   with 639 GB free, MODEL_API_KEY set: no, offscreen render OK.
2. `uv run ruff check .`: all checks passed. `uv run pytest -q`: 147 passed, 1 skipped (live).
3. `uv run twinfit synth --suite synth_small`: 12 clip folders; `contracts-check` passes on each.
4. Board-frame test (tests/setup/test_synth.py::test_board_pose_recovered_from_render): passes on
   all 12 clips with pooled chessboard corners from 6 frames.
5. Projection test: passes on all 12 clips, with ray-box occlusion by other parts excluded.
6. GT self-consistency: FK reproduces tracks below 1e-6 m on all 12 clips; `simulate` from the
   release state reproduces GT q within 1e-6 on the 10 dynamics clips.
7. `uv run twinfit run .../pendulum_1r__s0 --impl stub`: five stages ok; run folder passes
   `contracts-check`.
8. All six milestone groups print "not implemented yet (owned by Milestone N)".
9. `TWINFIT_LIVE=1 uv run pytest -m live`: fails with MissingApiKey ("MODEL_API_KEY is not set;
   put it in ~/.config/twinfit/.env"). The key is empty on this machine.
10. Printables (three PDFs), template and docs/TEST_DATA_PROTOCOL.md exist.
11, 12, 13: see below.

## Self-check

Could each milestone meet its acceptance criteria without editing setup-owned files?

- M1: calibration and board pose use `twinfit.contracts.charuco_board`, `geometry.camera`
  (project, homography, back-projection) and `gt/camera_extrinsics.json`; segmentation IoU uses
  `eval.metrics.label_image_ious` against `gt/masks`; tracking RMS uses
  `eval.metrics.match_parts` and `motion_point_rms_m` with GT extents; `contracts-check` on run
  folders; the gt-guard test uses the runner on a clip copy; Meta calls use `twinfit.meta.client`
  (`responses_create`, `raw_client`, `record_cost`, `budget.check`, `sam_video_usd`); cost tables
  use `ledger.summary`; the `perception` optional group and `twinfit.perception.cli:app` are
  mounted automatically. The `extras` dicts on PartTracks and TrackPart hold llm_connections and
  the extent_z_assumed flag. The marker-GT command writes tracks with source "marker_gt".
- M2: oracle tracks with noise via `RunOptions.perceive.oracle_noise` through the runner; relative
  poses from the tracks; `sim.replay` for replay; `eval.metrics.compare_kinematics` (type, axis,
  line distance, q) and `motion_point_rms_m`; the DynamicsModel contract carries identifiability,
  prior_dominated and fit; `gt/model.xml` and `gt/dynamics.json` for evaluation; Pinocchio is
  installed. `force_joint_types`, `exclude_time_ranges_s`, `planar_constraint`, `segments_s`,
  `fit_stiffness`, `mass_kg` and `window_s` are all in the options models.
- M3: `compute_verdict`, the Certificate and Check models, `sim.replay.windowed_replay_rms` (C10),
  `sim.energy.total_energy` and `energies_at` (C11), `simulate` (C12), the runner's run folders
  and `attempts/<k>/` under them (the runner accepts `runs_root` and `run_id`), oracle stubs for
  scenario corruption, `eval.metrics` for bench columns, `suite_dir` for both suites, the
  `verify` and `agent` optional groups, four auto-mounted CLI apps, and `twinfit.meta` for the
  Wonder agent under the session tag.

Is every convention covered by a sign-sensitive test?

- up_W, normal_toward_camera_W and gravity_W: `test_conventions.py` fails on any sign flip or
  axis swap (detected pixel axes compared with the hand derivation).
- R_cv = R_mj diag(1, -1, -1): `test_opencv_from_mujoco_camera`, and the rendered board test
  recovers the camera pose derived with this formula.
- T_A_B composition: the FK test rebuilds every part's motion from parent poses and joint data.
- Quaternion order (w, x, y, z): `test_quaternion_conversion_scalar_first`.
- plane_offset_m sign: `test_homography_against_direct_projection` and the mask projection test
  put part faces at z = -plane_offset.
- q right-hand rule about axis_W: FK uses exp(axis q) and matches the recorded tracks.
- Part and joint id patterns, the "world" id: contract validation tests.
- Time in seconds, increasing: array checks in io.py (tested).

Does any test touch the network without the live marker? No. The meta tests mock
`client.responses.create` and `models.list`; the only network test is `test_meta_live.py`
(marked live, skipped by conftest).

Is every unverified item marked? Yes, in the Phase 8 and Phase 9 results above and in the report.

Is anything personal in a tracked file? The commit author is the git global identity
(a GitHub noreply address). `.private/` is ignored and untracked (tested). docs/meta_api_notes.md
was grepped for the email, team_id, project_id, key name and masked key: no hits. The ledger and
the prompts name Ahaan, which is intended.

Result (Phase 11):
- Nine `[setup]` commits on main. Repository https://github.com/ahaan001/twinfit is PUBLIC.
- CI: the first two runs on main failed because the runner tests generated a clip with rendering
  (ubuntu-latest has no OpenGL: `mujoco.FatalError: gladLoadGL error`). Fixed by a `render` flag
  on `generate_clip` used by those tests; run 41225623257 on main is green (ruff clean, 98 tests,
  50 deselected as live or render).
- Worktrees: ../twinfit-m1 (m1-perception), ../twinfit-m2 (m2-identify), ../twinfit-m3
  (m3-verify-agent), all at the main head and pushed. Each synced with `uv sync --all-extras
  --dev` and passed `uv run pytest -q -m "not live and not render"` (98 passed).
- A fresh worktree exposed that the milestone package `__init__.py` files had never been written
  (empty directories are not tracked). Added them; the registry now also treats a missing parent
  package of the real module path as "absent".
- synth_small (12 clips) is in ~/twinfit-data/synth/synth_small, generated with the final code.
- Full suite on main after the last change: 147 passed, 1 skipped (live).
- History scan: no .private files, videos, .env or session files were ever tracked; no key
  patterns in `git log -p --all`; no email, team_id or project_id in tracked files.
- `.twinfit-session.yaml` in this worktree now has allow_spend false. Meta spend: $0.00 (no key).

## Addendum: live smoke (2026-10-31, after Ahaan added the key)

- `TWINFIT_LIVE=1 uv run pytest -m live`: 1 passed (138 output tokens, answer OK, $0.0006).
- `uv run twinfit meta-smoke`: models listed; the OK call returned status incomplete, reason
  max_output_tokens, 254 reasoning tokens of 256, empty text ($0.0011). A third diagnostic call
  reproduced it. Finding recorded in docs/meta_api_notes.md; `meta-smoke` now prints the status.
- Meta spend: tag setup $0.0028 over 3 calls, global $0.0028. Acceptance criterion 9: passes.
