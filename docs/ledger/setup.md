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
