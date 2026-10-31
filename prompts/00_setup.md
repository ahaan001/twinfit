# Twinfit · Prompt 0 · Repository setup and frozen interfaces

How to start: open Claude Code in the starting folder and say "Read 00_setup.md and carry it out exactly."

<role_and_principles>
You are building the shared foundation for Twinfit, a solo hackathon project by Ahaan Kothari. After this session, three Claude Code sessions will build three milestones in parallel. Each runs in its own git worktree and owns one package. They will rely only on what you build here.

Your deliverables are:
- tooling and the environment
- frozen data contracts
- a ground-truth synthetic data generator
- oracle stubs
- a pipeline runner and CLI
- a budget-guarded Meta API client
- printables for real data capture
- repository rules
- the three worktrees

Operating principles:
1. What you commit becomes a frozen interface. Design it so no milestone ever needs to edit shared code. Where a need is foreseeable but unspecified, add an optional field or route it through the `extras` dict.
2. Back every convention (frames, units, signs, layouts) with a test that fails if the convention is wrong. Prefer tests that compare two independent computations.
3. Never invent API parameters, model IDs, library signatures or results. Confirm library behavior by running code in the project environment. Confirm Meta API behavior against docs/meta_api_notes.md and the live docs at https://dev.meta.ai/docs. Mark anything you could not confirm as "unverified" in your report.
4. Keep code minimal:
   - no abstractions beyond this spec
   - no leftover print debugging
   - comments only where the math is non-obvious, citing the source or equation
5. Work autonomously and keep a task list with one item per phase. Stop only for the conditions in <stop_conditions>.
6. Before each phase, write a short plan into docs/ledger/setup.md: what you will build and how you will verify it. After each phase, record the result there.
</role_and_principles>

<context>
Twinfit turns phone video of a real mechanism into a MuJoCo simulation model, plus a certificate stating how far that model can be trusted. Example mechanisms are a pendulum, a double pendulum, a desk lamp, a hinged lid and a slider.

Pipeline and owners:
```
video + metadata.yaml
  -> perceive              Milestone 1: Muse Wonder part proposals, SAM 3.1 video segmentation,
                           OpenCV calibration, board pose, metric planar tracking      -> tracks
  -> kinematics, dynamics  Milestone 2: screw-axis joint fitting, kinematic tree,
                           physically consistent inertial/friction/spring identification
                                                     -> kinematics, dynamics, model.xml
  -> certify, report       Milestone 3: physical-consistency certificate, Muse Wonder repair
                           agent, benchmark harness, HTML report
```

Hackathon facts (full detail in hackathon_context.md):
- Event: Meta Global AI Developer Hackathon (Cerebral Valley).
- Deadline: Wed Nov 4, 2026, 9:00 PM ET.
- Entry: solo, Indie track, problem statement "Push the Frontier".
- Rules that shape the engineering:
  - The repository must be public.
  - All work must be built during the event (Oct 26 to Nov 4, 2026) and be clearly identifiable as such.
  - The project must use Meta models.
  - No code, data or assets without rights.
- Judging: Impact 30%, Use of Meta AI tooling 25%, Technical depth 20%, Demo 25%.
- Meta Model API budget: $150 in credits. Twinfit's global hard cap is $120.

Machine: Ahaan's Mac (macOS). Anaconda is installed and may be active in the shell. Do not use conda for this project. Use uv, and run all Python through `uv run`.

Ahaan has about 3 hours a day, mostly spent recording test videos. Claude Code does nearly all engineering.
</context>

<inputs>
The starting folder becomes the repository root, and its folder name becomes the repo name. It contains:
- 00_setup.md (this file) and the four other prompts: 01_m1_perception.md, 02_m2_identification.md, 03_m3_verify_agent.md, 04_integrate_and_submit.md.
  - Read the <context> and <acceptance_criteria> of 01 to 04 so the interfaces you build serve them.
  - Do not do their work.
- RUNBOOK.md: Ahaan's runbook. Its "Test data protocol" section defines how real clips are captured. Your printables, metadata template and conventions must match it exactly.
- hackathon_context.md: rules, schedule and the Meta Model API reference.
  - It contains personal account details and must never be committed.
  - If it is absent, use any other .md in the folder whose first heading mentions "Meta Global AI Developer Hackathon".

If any file is missing, list it in your report and continue with what exists.
</inputs>

<instructions>

Phase 0. Preflight (record failures and keep going)
1. Check these tools:
   - `git --version`
   - `uv --version`
   - `gh --version`
   - `gh auth status`

   If uv is missing, install it with Homebrew when present, otherwise with the official installer (`curl -LsSf https://astral.sh/uv/install.sh | sh`).
2. Check free disk space. Below 5 GB is a stop condition.
3. Check whether `~/.config/twinfit/.env` defines a non-empty `MODEL_API_KEY`, without printing the value. For example: `grep -c '^MODEL_API_KEY=..*' ~/.config/twinfit/.env`.
   - If the file is missing, create the directory.
   - Write `MODEL_API_KEY=` and `TWINFIT_BUDGET_USD=120` into the file, chmod 600, and continue.
4. Read every input file in full.

Phase 1. Skeleton
1. Run `git init -b main`, then move files:
   - the five prompt files into `prompts/`
   - RUNBOOK.md to `docs/RUNBOOK.md`
   - the hackathon context file to `.private/hackathon_context.md`
2. Write `docs/meta_api_notes.md` from sections 1 to 11 of the hackathon context.
   - Remove everything personal or account-specific:
     - the section 0 table and the participant email
     - team_id, project_id, API key names and masked keys
     - billing and credit status
     - partner account details
   - Header: the source, copy date 2026-10-31, and the line "Verify against https://dev.meta.ai/docs before relying on any detail."
3. Create this layout:
```
<repo>/
  pyproject.toml  uv.lock  .python-version  LICENSE  README.md  CLAUDE.md  THIRD_PARTY.md
  .gitignore  .env.example  .github/workflows/ci.yml
  prompts/  docs/  docs/ledger/  docs/media/  docs/contract_change_requests/  .private/
  src/twinfit/
    __init__.py  __main__.py  cli.py  paths.py  session.py
    contracts/  geometry/  sim/  synth/  stubs/  meta/  eval/  runner/  printables/
    perception/__init__.py                       (Milestone 1 owns this package)
    identify/__init__.py                         (Milestone 2)
    verify/__init__.py  agent/__init__.py  bench/__init__.py  report/__init__.py   (Milestone 3)
  tests/
    conftest.py  setup/
    perception/  identify/  verify/  agent/  bench/  report/    (each with a .gitkeep)
```
4. LICENSE: MIT, "Copyright (c) 2026 Ahaan Kothari".
5. `.gitignore` entries:
   - `.venv/`, `__pycache__/`, `.pytest_cache/`, `.ruff_cache/`, `.DS_Store`
   - `.private/`, `.env`, `.twinfit-session.yaml`, `runs/`
   - `*.mp4`, `*.mov`, `*.MOV`
   - the exception `!docs/media/**`
6. `.env.example`: contains `MODEL_API_KEY=` and `TWINFIT_BUDGET_USD=120`, with a comment that real values live in `~/.config/twinfit/.env`.

Phase 2. Environment
1. Run `uv python pin 3.11`. In pyproject:
   - name `twinfit`
   - `requires-python = ">=3.11,<3.12"`
   - hatchling build with src layout
   - console script `twinfit = "twinfit.cli:main"`
2. Dependencies, with lower bounds only (uv.lock pins exact versions):
   - numpy, scipy, pydantic>=2.6, pyyaml
   - typer, rich, jinja2
   - opencv-contrib-python>=4.8 (never also opencv-python)
   - mujoco>=3.2, pin>=3.0 (Pinocchio)
   - imageio, imageio-ffmpeg, matplotlib
   - pandas, pyarrow
   - filelock, python-dotenv, openai>=1.50

   Dev group: pytest, pytest-timeout, ruff.
3. Add empty optional-dependency groups `perception`, `identify`, `verify` and `agent`. Each milestone adds packages only to its own group.
4. pytest config:
   - `testpaths = ["tests"]`
   - `addopts = "--import-mode=importlib"`
   - markers: `live` (calls the paid Meta API), `render` (MuJoCo offscreen rendering), `slow`
   - In `tests/conftest.py`, skip `live` tests unless the environment variable `TWINFIT_LIVE=1` is set on that command.
5. ruff: line length 100, select E, F, I, B, UP.
6. Run `uv sync --all-extras --dev`. Then confirm:
   - `cv2.aruco`, `mujoco`, `pinocchio` and `openai` all import
   - `imageio_ffmpeg.get_ffmpeg_exe()` runs

   If Pinocchio will not install or import after two distinct attempts, remove it, record why, and continue. Milestone 2 has a fallback.

Phase 3. Paths and session
1. `paths.py` defines:
   - `data_dir()`: env `TWINFIT_DATA_DIR`, default `~/twinfit-data`, with subfolders `synth/`, `real/`, `real/_calibration/`, `real/_template/`, `cache/meta/` and `printables/`.
   - `repo_root()`: the git top level of the current worktree.
   - `runs_dir()`: `repo_root()/runs`.
   - `ledger_path()`: `data_dir()/cost_ledger.jsonl`.

   The data dir and ledger are deliberately shared by all worktrees.
2. `session.py` reads an optional `<repo_root>/.twinfit-session.yaml` with these keys:
   - `tag` (str)
   - `tag_cap_usd` (float or null)
   - `allow_spend` (bool)

   Defaults: tag "untagged", no cap, allow_spend false. The environment variables `TWINFIT_TAG` and `TWINFIT_ALLOW_SPEND` override the file.

   Load secrets with python-dotenv from `~/.config/twinfit/.env`, then from a repo-root `.env` if present. Never print or log secret values.
3. Create this worktree's `.twinfit-session.yaml` with `tag: setup`, `tag_cap_usd: 1.0` and `allow_spend: true`.

Phase 4. Contracts
1. Implement <contract_spec> exactly in `src/twinfit/contracts/`.
2. Add `twinfit contracts-check PATH`. It validates a clip folder or a run folder and prints one line per file.
3. Write tests that round-trip every model and reject malformed arrays.

Phase 5. Geometry and simulation utilities
1. `geometry/se3.py`:
   - inverse and compose
   - SO(3) and SE(3) exp/log
   - chordal rotation averaging
   - quaternion conversion (w, x, y, z)
   - skew
2. `geometry/camera.py`:
   - project world points given K, distortion and T_W_C
   - homography between pixels and a plane parallel to the board at signed offset d along `normal_toward_camera_W`
   - back-projection of pixels onto that plane
3. `geometry/lines.py`:
   - distance between two 3D lines, handling the parallel case
   - the point on a line closest to the origin
4. `sim/replay.py`:
   - `simulate(model_xml, q0, qd0, t_out, timestep=2e-4, integrator="RK4")` returns q and qd at `t_out`.
   - `windowed_replay_rms(model_xml, t, q_obs, qd_obs, valid, segments_s, window_s=0.5)`:
     - restarts the simulation from the observed state at the start of each window inside each segment
     - skips windows that contain invalid frames
     - returns per-joint RMS and per-window details

   Milestones 2 and 3 must both use these functions, so that fitting and certification measure the same quantity. Say so in the docstrings.
5. `sim/energy.py`: the total mechanical energy of a MuJoCo state, using MuJoCo's energy computation. Enable the energy flag. Confirm whether spring potential is included, and add it if it is not.
6. Tests:
   - exp/log round trips
   - projection against a hand-computed case
   - homography against direct projection for points on an offset plane
   - an undamped, frictionless pendulum conserves energy within 1e-3 relative over 5 s
   - `simulate` matches plain `mj_step` stepping

Phase 6. Synthetic ground-truth generator
1. Implement <synthetic_scenes> in `src/twinfit/synth/`.
2. Add these CLI forms:
   - `twinfit synth --suite {synth_tiny,synth_small} [--out DIR]`
   - `twinfit synth --scene NAME --seed N [--fps --duration --width --height --out]`
3. Default output is `data_dir()/synth/<suite>/<scene>__s<seed>/`.
4. Expose `twinfit.synth.suite_dir(name, ensure=True) -> Path`. It generates the suite if absent, and every milestone's tests use it to get shared fixtures.

Phase 7. Stubs, runner, evaluation, CLI
1. `stubs/oracle.py`: one function per stage, with the same signature as the real stages.
   - perceive copies `gt/tracks.*`. If `oracle_noise` is set, it applies an independent, seeded random rigid perturbation per part per frame: translation standard deviation `trans_m` per axis, and rotation `rot_rad` about a uniformly random axis.
   - kinematics copies `gt/kinematics.*`.
   - dynamics copies `gt/dynamics.json` and `gt/model.xml`.
   - certify writes a NOT_RUN certificate whose checks all have status not_run.
   - report writes a minimal HTML page listing the stage results.
   - A missing gt folder gives status failed with a clear message.
2. `runner/registry.py` maps each stage to its real implementation and to its oracle stub:

   | Stage | Real implementation |
   |---|---|
   | perceive | `twinfit.perception.stage:run` |
   | kinematics | `twinfit.identify.kinematics_stage:run` |
   | dynamics | `twinfit.identify.dynamics_stage:run` |
   | certify | `twinfit.verify.stage:run` |
   | report | `twinfit.report.stage:run` |

   `resolve(stage, impl)` takes impl in {auto, real, stub}:
   - auto uses the real module when importing it does not raise ModuleNotFoundError for that exact module path, and otherwise uses the stub
   - any other import error propagates
3. `runner/pipeline.py` provides `run_pipeline(clip_dir, run_id, impl_map, options: RunOptions, stop_after=None) -> list[StageResult]`.
   - It creates `runs/<run_id>/` containing `run.json`: clip_dir, the impl used per stage, git commit hash, start and end time, and total cost.
   - It also creates one stage folder per stage, holding that stage's outputs and a `stage_result.json`.
   - A failed stage marks all later stages as skipped.
   - Run ids default to `<clip_id>__<UTC yyyymmddThhmmss>`.
4. `eval/metrics.py` is shared by all milestones. Document each function with its exact definition:
   - `axis_angle_error_deg(a_est, a_gt)` = degrees(arccos(|a_est · a_gt|)).
   - `line_distance_m(p1, a1, p2, a2)`.
   - `motion_point_rms_m(T_est, T_gt, valid, points_W_ref, ref_index)`:
     - moves points with each sequence's displacement relative to its own reference pose: X(t) = T(t) T(ref)^-1 X(ref)
     - returns the RMS distance between estimate and ground truth over points and valid frames
     - default points are the 8 corners of the GT part's extent at the reference frame, mapped to W
   - `match_parts(tracks_est, tracks_gt)`: Hungarian matching on `motion_point_rms_m`, reporting unmatched parts.
   - `compare_kinematics(kin_est, kin_gt, part_map, tracks_gt)`:
     - Joint type agreement.
     - Axis angle and line distance, compared in W at the estimate's reference time. Transform the GT `axis_parent` and `point_parent` with the GT parent pose at that time; parent "world" is unchanged.
     - q compared as q_est(t) against s·(q_gt(t) - q_gt(t_ref)), where s = sign(a_est · a_gt).
     - A static GT base part is treated as equivalent to "world".
   - `mask_iou(mask_est, mask_gt)`, per part per frame.
   - Tests for each function, using hand-built cases.
5. `cli.py` uses typer.
   - Top-level commands: `doctor`, `synth`, `run`, `contracts-check`, `meta-smoke`, `printables`, and `cost` (ledger totals by tag, model and day).
   - `run` options:
     - `--impl {auto,real,stub}`
     - repeatable `--impl-stage stage=impl`
     - `--options FILE.yaml` (a RunOptions document)
     - `--run-id` and `--stop-after STAGE`
   - Mount the milestone sub-apps by importing `app` from each module:

     | Module | Group name |
     |---|---|
     | `twinfit.perception.cli` | `perception` |
     | `twinfit.identify.cli` | `identify` |
     | `twinfit.verify.cli` | `verify` |
     | `twinfit.agent.cli` | `agent` |
     | `twinfit.bench.cli` | `bench` |
     | `twinfit.report.cli` | `report` |

   - When a module is absent, register a placeholder group whose commands print "not implemented yet (owned by Milestone N)". Milestones must never need to edit cli.py.
6. `doctor` reports:
   - Python version and the import status of key packages
   - ffmpeg path
   - data dir and its free space
   - whether MODEL_API_KEY is set (yes or no only)
   - session tag and spend status
   - ledger total against the caps
   - a one-frame MuJoCo offscreen render test

Phase 8. Meta API client (`src/twinfit/meta/`)
1. `config.py`:
   - base URL `https://api.meta.ai/v1`
   - model ids `muse-wonder-1.0`, `muse-spark-1.3`, `muse-spark-1.3-contributor` and `sam-3.1`
   - prices from docs/meta_api_notes.md: USD per million tokens for input, cached input and output; SAM per 1,000 images and per 1,000 video frames, with a 50-frame minimum per video request
   - the date the prices were copied
2. `ledger.py`: an append-only JSONL file at `ledger_path()`, written under a filelock.
   - Each record holds: timestamp, tag, worktree path, model, kind (responses, sam_video, sam_image, files or other), usage, cost_usd, cache_hit and request id.
   - Provide `total(tag=None)`.
3. `budget.py` provides `check(estimate_usd, tag)`, which raises:
   - `SpendNotAllowed` when allow_spend is false, with a message that says how to enable it
   - `BudgetExceeded` when the global total plus the estimate exceeds `TWINFIT_BUDGET_USD` (default 120), or the tag total plus the estimate exceeds the tag cap
4. `cache.py`: a content-addressed disk cache in `data_dir()/cache/meta/`.
   - The key is sha256 over the model id plus the canonical JSON of the request, with any file content replaced by its sha256.
   - It stores the full response JSON.
5. `client.py` provides `responses_create(model, input, *, max_output_tokens, estimate_usd=None, cache=True, **kwargs)`. It uses the OpenAI Python SDK pointed at the base URL, with the key passed explicitly. Order of operations:
   1. Cache lookup.
   2. Budget check. The default estimate comes from input size and max_output_tokens at list price. Callers pass `estimate_usd` for image or video inputs.
   3. The call itself.
   4. Retry on 429 and 5xx with exponential backoff and full jitter: 1 s base, factor 2, 6 attempts, honoring Retry-After.
   5. Record the actual cost from the returned usage.
   6. Cache store.

   Also expose `raw_client()`, `record_cost(...)` and `budget.check`, so milestone code that uploads files or streams responses keeps the ledger accurate.

   Authentication or billing failures (401, 402, 403, or messages about payment or billing) raise `BillingError` immediately. Never retry them. The message is "Add a payment method or check the key at https://dev.meta.ai/platform".
6. `twinfit meta-smoke`:
   - List the models and print their ids.
   - Make one `muse-spark-1.3` Responses call asking for the single word OK (max_output_tokens 256), recorded under the current tag.
7. Unit tests, with the SDK mocked:
   - retry and backoff, including Retry-After
   - billing error handling
   - ledger concurrency across two processes
   - budget refusal
   - a cache hit records zero cost

   Add one `live` test that runs the smoke call.

Phase 9. Printables and real-data templates
1. `twinfit printables` writes three PDFs to `data_dir()/printables/`:
   - `charuco_board_letter.pdf`:
     - the ChArUco board from <contract_spec> at exact physical size, on US Letter landscape, centered
     - a 100 mm scale bar and a large "UP" arrow above the board's top edge
     - the dictionary, square and marker sizes, and the text "Print at 100% (Actual size). Measure squares and record them in metadata.yaml."
   - `part_markers_letter.pdf`:
     - DICT_4X4_50 markers with IDs 0 to 11, each 30 mm, with a 6 mm white border
     - each ID printed below its marker, plus cut lines and a 100 mm scale bar
   - `speckle_letter.pdf`:
     - random black dots on white, 2 to 4 mm in diameter, about 35% coverage, fixed seed
     - a 100 mm scale bar
     - used to texture parts

   Assert the drawn physical sizes in code (axes extents in inches) before saving.
2. Write `data_dir()/real/_template/metadata.yaml`.
   - Include every ClipMetadata field, with an inline comment explaining what to measure and how, matching the RUNBOOK protocol.
   - Leave `gravity_W` null, since it is derived from the board orientation.
   - Set `intrinsics` to `../_calibration/<camera_id>/camera_intrinsics.json`, which is where Milestone 1's `twinfit perception calibrate` writes it, and say so in the comment.
   - Set `video` to a placeholder filename, with a comment telling Ahaan to put the actual video filename in the clip folder there.
3. Write `docs/TEST_DATA_PROTOCOL.md`: the RUNBOOK's "Test data protocol" section, plus the empirically derived board axes in plain words. State which printed direction is +x, which is +y, and whether +z points out of or into the printed face.

Phase 10. Repository documents
1. `CLAUDE.md` is read automatically by every later session. It contains:
   - a one-paragraph project summary
   - a command cheat sheet
   - the ownership table below
   - these rules:
     - Never edit outside owned paths.
     - Contracts are frozen. A needed change goes into `docs/contract_change_requests/<milestone>.md`, with a workaround through `extras` in the meantime.
     - Real stages never read `gt/`.
     - Every stage writes only inside its own stage folder.
     - Meta API calls go only through `twinfit.meta`: never in unit tests, always tagged, cached and budget-checked.
     - Secrets are never printed, logged or committed.
     - Do not set `ANTHROPIC_BASE_URL`, `ANTHROPIC_AUTH_TOKEN` or any other variable from the hackathon notes that would route Claude Code itself through Meta's API.
   - Code style: clean and minimal, descriptive names, no print debugging, no emoji, comments only for non-obvious math with a source.
   - Documentation style: plain, direct, no em dashes.
   - Commit message prefixes: `[setup]`, `[m1]`, `[m2]`, `[m3]`, `[int]`.
   - Each milestone keeps its own ledger file, `docs/ledger/<milestone>.md`.
   - A summary of the hackathon rules.
2. Ownership table (repeat it in docs/ARCHITECTURE.md):

| Owner | Paths |
|---|---|
| Setup (frozen) | pyproject.toml (except other owners' optional groups), cli.py, paths.py, session.py, contracts/, geometry/, sim/, synth/, stubs/, meta/, eval/, runner/, printables/, tests/setup/, CLAUDE.md, docs/ARCHITECTURE.md |
| Milestone 1 | src/twinfit/perception/, tests/perception/, docs/m1_perception.md, docs/ledger/m1.md, docs/media/m1_*, optional group `perception` |
| Milestone 2 | src/twinfit/identify/, tests/identify/, docs/m2_identification.md, docs/ledger/m2.md, docs/media/m2_*, optional group `identify` |
| Milestone 3 | src/twinfit/verify/, src/twinfit/agent/, src/twinfit/bench/, src/twinfit/report/, tests/verify/, tests/agent/, tests/bench/, tests/report/, docs/m3_verify_agent.md, docs/ledger/m3.md, docs/media/m3_*, optional groups `verify` and `agent` |
| Integration | README.md, scripts/, docs/ledger/integration.md, docs/DEMO_SCRIPT.md, docs/SUBMISSION.md, merges |

3. `docs/ARCHITECTURE.md` contains:
   - a pipeline diagram (Mermaid)
   - the conventions
   - every contract in readable form
   - the stage interface
   - the run, clip and data dir layouts
   - the ownership table
4. `README.md` skeleton:
   - the title and a one-line pitch
   - "Built during the Meta Global AI Developer Hackathon (Oct 26 to Nov 4, 2026)"
   - sections: How it works, Meta models used, Results, Quickstart, Capturing your own data, Limitations, License
   - "TBD at integration" where the content is not known yet
   - put `<!-- RESULTS:BEGIN -->` and `<!-- RESULTS:END -->` markers inside the Results section
5. `THIRD_PARTY.md`: each dependency with its license, read from installed package metadata. Add a note that no third-party datasets or robot models are redistributed.
6. `.github/workflows/ci.yml`:
   - runs on ubuntu-latest with `astral-sh/setup-uv`
   - `uv sync --all-extras --dev`
   - `uv run ruff check .`
   - `uv run pytest -m "not live and not render"`

Phase 11. Verify, publish, branch
1. Run every command in <acceptance_criteria> and fix failures.
2. Commit in logical commits prefixed `[setup]`.
3. If `gh` is authenticated:
   - Run `gh repo create <folder name> --public --source=. --remote=origin --push --description "Phone video to a verified simulation twin. Built for the Meta Global AI Developer Hackathon."`
   - Then check the first CI run. Report its status if it is still pending.
4. Create the worktrees from the repo root:
   - `git worktree add ../twinfit-m1 -b m1-perception`
   - `git worktree add ../twinfit-m2 -b m2-identify`
   - `git worktree add ../twinfit-m3 -b m3-verify-agent`

   In each worktree, run `uv sync --all-extras --dev` and `uv run pytest -q -m "not live and not render"`. Push the branches if a remote exists.
5. Generate `synth_small` once into the shared data dir.
6. Set `allow_spend: false` in this worktree's `.twinfit-session.yaml`.
</instructions>

<contract_spec>
Write these as pydantic v2 models in `src/twinfit/contracts/`, following these rules:
- Every model uses `model_config = ConfigDict(extra="forbid")`.
- Every top-level file model has a `contract_version: Literal["1.0"]` field and an `extras: dict[str, Any] = {}` field.
- Arrays live in `.npz` files next to the JSON, and the JSON stores the npz filename.
- `contracts/io.py` provides `load_*` and `save_*` functions that validate on both read and write. The array checks are:
  - shape and dtype (float64 or bool)
  - finite values on valid frames
  - rotation orthonormality |R^T R - I| < 1e-6, and det(R) > 0
  - homogeneous last row [0, 0, 0, 1]

Conventions. Put these verbatim in the `contracts/conventions.py` docstring and in docs/ARCHITECTURE.md:
- Units are SI: m, kg, s, rad, N, N·m. Reports may show degrees and mm; files never do.
- `T_A_B` is the 4×4 homogeneous pose of frame B expressed in frame A, so p_A = T_A_B p_B. Same for R_A_B and p_A_B.
- Rotations are stored as matrices. Quaternions appear only at the MuJoCo boundary, scalar-first (w, x, y, z).
- Camera frames follow OpenCV: x right, y down, z forward along the optical axis. MuJoCo cameras look along -z with y up. Convert with R_cv = R_mj · diag(1, -1, -1).
- The world frame W is the ChArUco board frame, exactly as defined by the coordinates from the installed OpenCV's `CharucoBoard.getChessboardCorners()`, with z = 0 on the printed face.
  - Determine empirically, and document, which printed direction is +x and which is +y, and whether +z points out of the printed face toward the viewer or into it.
  - Derive `up_W`: the unit vector pointing up when the board is mounted vertically with the printed UP arrow on top.
  - Derive `normal_toward_camera_W`: the unit normal pointing from the printed face toward the viewer.
  - Both are constants computed at import from the board definition, with a test.
- Gravity: `gravity_W = -9.81 · up_W` for `vertical_arrow_up`, and `gravity_W = -9.81 · normal_toward_camera_W` for `horizontal_face_up`.
- `plane_offset_m` is the signed distance from the printed face to the plane containing the moving parts' camera-facing surfaces. It is positive toward the camera, along `normal_toward_camera_W`.
- Part ids match `p[0-9]+` and joint ids match `j[0-9]+`. The id `"world"` is reserved for W treated as a fixed link, and never appears in tracks.
- A part frame P is any frame rigidly attached to the part. Consumers must not assume its origin or orientation.
- Time is in seconds from the first decoded frame, taken from container timestamps. Variable frame rate is tolerated.
- The joint coordinate q is 0 at the reference frame index. Revolute q follows the right-hand rule about `axis_W`; prismatic q is the displacement along `axis_W`.

Board constants:
- ChArUco board: dictionary DICT_5X5_100, squares_x = 7, squares_y = 5, square_length_m = 0.035, marker_length_m = 0.026 (nominal).
- Part markers: DICT_4X4_50, IDs 0 to 11, marker_length_m = 0.030 (nominal).
- Real clips override these lengths with measured values.

Files and models:

1. `ClipMetadata`, in the file `metadata.yaml` in every clip folder, real or synthetic. Fields:
   - `clip_id`
   - `source`: "real" or "synthetic"
   - `mechanism`: str
   - `video`: relative path
   - `camera_id`
   - `intrinsics`: path to camera_intrinsics.json, relative to the clip folder
   - `board`: {type "charuco", dictionary, squares_x, squares_y, square_length_m, marker_length_m}
   - `board_orientation`: "vertical_arrow_up" or "horizontal_face_up"
   - `motion_model`: "planar" or "spatial"
   - `plane_offset_m`: float
   - `gravity_W`: 3 floats, or null (null means derive it from the orientation)
   - `parts_hint`: list of {label, fixed: bool|null, mass_kg: float|null}
   - `part_markers`: {dictionary, marker_length_m, assignments: dict[label, int]}, or null
   - `free_motion_segments_s`: list of [t0, t1]
   - `dynamics_gt`: bool, default false
   - `notes`: str
   - `extras`
2. `CameraIntrinsics`, in `camera_intrinsics.json`. Fields:
   - `camera_id`, `width`, `height`
   - `K`: 3×3
   - `dist`: OpenCV order k1, k2, p1, p2, k3
   - `rms_reprojection_px`: float|null
   - `n_frames_used`: int|null
   - `source`: "calibration" or "synthetic"
3. `PartTracks`, in `tracks.json` plus `tracks.npz`.
   - json fields:
     - `clip_id`
     - `source`: "perception", "synthetic_gt" or "marker_gt"
     - `fps_nominal`, `n_frames`, `motion_model`
     - `plane`: {point_W, normal_W}, or null
     - `gravity_W`
     - `camera`: {intrinsics: CameraIntrinsics, static: bool, T_W_C: 4×4 or null}. T_W_C is null when the camera is not static.
     - `parts`: list of {part_id, label, is_base_hint: bool|null, extent_P: [[xmin, ymin, zmin], [xmax, ymax, zmax]] or null}
     - `free_motion_segments_s`
     - `free_motion_source`: "metadata", "llm" or "none"
     - `arrays`: "tracks.npz"
   - npz keys:
     - `t`: (N,) float64
     - `T_W_C`: (N, 4, 4), present only when the camera is not static
     - for each part:
       - `T_W_P__<part_id>`: (N, 4, 4)
       - `valid__<part_id>`: (N,) bool
       - `quality__<part_id>`: (N,) float64, the RMS fit residual in metres for that frame, NaN where invalid
   - Invalid frames may hold any finite pose. Consumers must mask them out.
4. `KinematicModel`, in `kinematics.json` plus `kinematics.npz`.
   - json fields:
     - `clip_id`
     - `source`: "identify" or "synthetic_gt"
     - `base_part_id`: a part id or "world"
     - `reference_frame_index`
     - `links`: list of {part_id, parent_joint_id or null}
     - `joints`: a list. Each joint has:
       - `joint_id`, `parent_part_id`, `child_part_id`
       - `type`: "revolute", "prismatic" or "fixed"
       - `axis_W`: a unit vector, at the reference frame
       - `point_W`: for revolute joints, the point on the axis closest to the W origin; otherwise null
       - `axis_parent`, `point_parent`: the same quantities in the parent part frame P used in the tracks. For parent "world", they equal the W values.
       - `q_range_observed`: [lo, hi]
       - `fit`: {rms_point_m, n_frames}
       - `type_scores`: {revolute, prismatic, fixed: float|null}, lower is better, with the method name in extras
       - `confidence`: in [0, 1]
     - `arrays`: "kinematics.npz"
   - npz: `t`, and per joint `q__<joint_id>`, `qd__<joint_id>`, `qdd__<joint_id>` and `valid__<joint_id>`.
5. `DynamicsModel`, in `dynamics.json` plus `model.xml`.
   - json fields:
     - `clip_id`
     - `source`: "identify" or "synthetic_gt"
     - `mjcf`: "model.xml"
     - `gravity_W`: a vector in m/s²
     - `mass_scale_known`: bool
     - `links`: list of {part_id, mass_kg, com_P, inertia_com_P (full 3×3 tensor about the COM, expressed in P axes), prior_dominated: list of parameter names}
     - `joints`: list of {joint_id, damping, frictionloss, armature, stiffness, springref}
     - `identifiability`: {rank, n_params, condition_number, unidentifiable: list of {combination, reason}}
     - `fit`: {segments_s, replay_rms: dict[joint_id, float], window_s, method}
   - model.xml rules:
     - It loads with `mujoco.MjModel.from_xml_path`.
     - The worldbody frame equals W.
     - Body names equal part ids, and joint names equal joint ids.
     - The base part is a body welded to the world (no joint), unless the base is "world".
     - Geoms are for visualization only, with contype=0 and conaffinity=0.
     - Gravity is set to gravity_W via `<option gravity>`.
     - Inertial properties are set explicitly with `<inertial fullinertia=...>`, so MuJoCo never infers them from geoms.
6. `Certificate`, in `certificate.json` plus `certificate.md`.
   - Fields:
     - `clip_id`
     - `verdict`: "VERIFIED", "VERIFIED_WITH_CAVEATS", "REJECTED" or "NOT_RUN"
     - `checks`: list of {check_id, title, subject, status ("pass", "warn", "fail", "not_applicable" or "not_run"), value: float|null, threshold: float|null, unit: str, evidence: str}
     - `caveats`: list[str]
     - `generated_by`
   - `compute_verdict(checks, caveats)` applies these rules in order:
     1. Any fail gives REJECTED.
     2. Otherwise, any warn or any caveat gives VERIFIED_WITH_CAVEATS.
     3. Otherwise, at least one pass gives VERIFIED.
     4. Otherwise, NOT_RUN.

   Milestone 3 must use this function.
7. Stage interface, in `contracts/stage.py`:
   - `StageName = Literal["perceive","kinematics","dynamics","certify","report"]`. The stage folders are `01_perceive`, `02_kinematics`, `03_dynamics`, `04_certify` and `05_report`.
   - `StageContext` is a frozen dataclass with fields `clip_dir`, `run_dir`, `stage_dir`, `metadata: ClipMetadata`, `upstream: dict[StageName, StageResult]` and `cost_tag: str`.
   - `StageResult` is a pydantic model with fields:
     - `stage`
     - `status`: "ok", "failed" or "skipped"
     - `impl`: "real" or "stub"
     - `outputs`: dict[str, str], with paths relative to run_dir
     - `metrics`: dict[str, float]
     - `messages`: list[str]
     - `cost_usd`, `duration_s`
   - Fixed output keys per stage:

     | Stage | Required keys | Optional keys |
     |---|---|---|
     | perceive | tracks_json, tracks_npz | masks_dir, overlay_mp4, perception_report |
     | kinematics | kinematics_json, kinematics_npz | |
     | dynamics | dynamics_json, model_xml | |
     | certify | certificate_json, certificate_md | |
     | report | report_html | |

   - Options models. Every field is optional with a default, and each uses extra="forbid".
     - `PerceptionOptions`:
       - `part_prompts`: list[str]|None
       - `box_prompts`: list of {label, frame_index, box_xyxy_px}, or None
       - `frame_stride`: int = 1
       - `max_frames`: int|None
       - `exclude_labels`: list[str] = []
       - `use_part_markers`: bool = False
       - `llm_part_proposal`: bool = True
       - `oracle_noise`: {trans_m, rot_rad, seed}|None, used only by the oracle stub
     - `KinematicsOptions`:
       - `force_joint_types`: dict of "<parent>-><child>" to type, default {}
       - `force_base_part`: str|None
       - `exclude_time_ranges_s`: list of [t0, t1], default []
       - `planar_constraint`: bool|None, where None means follow tracks.motion_model
     - `DynamicsOptions`:
       - `segments_s`: list of [t0, t1], or None
       - `fit_stiffness`: bool = False
       - `fit_coulomb`: bool = True
       - `fix_params`: dict[str, float] = {}
       - `mass_kg`: dict[part_id, float] = {}
       - `window_s`: float = 0.5
     - `CertifyOptions`: `thresholds_override`: dict[str, float] = {}.
     - `RunOptions`: optional fields `perceive`, `kinematics`, `dynamics` and `certify`, each holding that stage's options model. The report stage takes no options; the runner passes `None`.
   - Every stage implementation exposes `run(ctx: StageContext, options) -> StageResult` and writes only inside `ctx.stage_dir`. Real implementations must never read `ctx.clip_dir / "gt"`.
</contract_spec>

<synthetic_scenes>
General rules:
- Simulation and world frame:
  - Build each scene programmatically as MJCF. No external assets except the generated board texture.
  - Simulate with the same settings as `twinfit.sim.replay.simulate` (RK4, 2e-4 s).
  - Render with `mujoco.Renderer` from a fixed camera.
  - The MuJoCo world frame equals W.
- Board:
  - Use the ChArUco image generated by OpenCV from the board spec, applied as a 2D texture on a finite plane geom of the board's physical size, at z = 0, facing the camera, with no collisions.
  - Determine the image-to-W mapping empirically. Generate the image with zero margin, detect its chessboard corners, and fit the affine map from pixel coordinates to `getChessboardCorners()` coordinates. Place the texture using that map.
  - A gray backdrop plane, larger than the camera view, sits 2 mm behind the board.
- Parts:
  - Moving parts are box geoms with a speckle texture (a MuJoCo builtin texture with random marks; confirm the attribute names in the installed MuJoCo).
  - The base part has a distinct flat color.
  - Moving parts sit in a plane at `plane_offset_m` in front of the board.
- Hold and release:
  - During the hold, reset qpos to the hold pose and qvel to zero at every physics step.
  - Release at t_release.
  - free_motion_segments_s = [[t_release, t_end]] unless the scene states otherwise.
- Camera:
  - Fronto-parallel to the board, with tilt randomized within 15° (35° oblique for the spatial scene).
  - Distance 0.7 to 1.1 m, fovy 45°.
  - The whole board and every part's full motion must stay visible. Verify by projecting the part extents over all frames: at least 98% of projected points must land inside the image. Resample the camera otherwise.
  - Set `<visual><global offwidth offheight>` to at least the render size.
- Seeds:
  - Seed 0 uses the nominal values.
  - Other seeds scale lengths, masses, damping, friction and stiffness by U(0.85, 1.15), shift initial angles by ±10°, and randomize camera pose, light direction and part colors.
- Suites:
  - `synth_tiny`: each scene at seed 0, 320×180, 30 fps, 2.0 s.
  - `synth_small`: each scene at seeds 0 and 1, 1280×720, 60 fps, 8.0 s. This gives 12 clips.

Scenes. In every scene, p0 is the fixed, visible base part, welded to the world.
1. `pendulum_1r`
   - Board: vertical.
   - p1: a rod, 0.30 × 0.03 × 0.004 m, 0.06 kg, uniform.
   - j0: revolute, p0 to p1, about the board normal, with the pivot 0.02 m from the rod's top end.
   - Damping 2e-4 N·m·s/rad, frictionloss 5e-4 N·m.
   - plane_offset 0.010 m.
   - Hold at 60° from hanging; release at 0.5 s.
2. `double_pendulum_2r`
   - Board: vertical.
   - p1: a strip, 0.20 × 0.025 × 0.003 m, 0.025 kg.
   - p2: a strip, 0.15 × 0.025 × 0.003 m, 0.020 kg.
   - j0: p0 to p1. j1: p1 to p2, at p1's far end. Both are about the board normal.
   - Damping 1e-4, frictionloss 2e-4.
   - plane_offset 0.008 m.
   - Hold at (90°, 0°); release at 0.5 s.
3. `lamp_spring_2r`
   - Board: vertical.
   - p1: an arm, 0.25 × 0.03 × 0.01 m, 0.12 kg.
   - p2: an arm, 0.20 × 0.03 × 0.01 m, 0.10 kg.
   - j0 and j1: about the board normal, with stiffness 0.6 and 0.4 N·m/rad. Choose springref so the rest pose is visibly bent.
   - Damping 5e-3, frictionloss 2e-3.
   - plane_offset 0.020 m.
   - Hold 25° from rest on both joints; release at 0.5 s.
4. `slider_incline_1p`
   - Board: vertical.
   - p0: a rail, 0.35 m long, inclined 25° within the board plane.
   - p1: a block, 0.04 × 0.03 × 0.02 m, 0.05 kg.
   - j0: prismatic along the rail, damping 1.0 N·s/m, frictionloss 0.02 N, range limited to the rail.
   - plane_offset 0.015 m.
   - Hold at the top until 0.5 s. The free segment ends 0.1 s before the block reaches the limit.
5. `door_spring_1r_h`
   - Board: horizontal, with the camera looking down.
   - p1: a panel, 0.25 × 0.02 × 0.03 m, 0.15 kg.
   - j0: about the board normal.
   - Stiffness 0.05 N·m/rad with springref 0, damping 0.01 N·m·s/rad, frictionloss 1e-3 N·m.
   - plane_offset 0.020 m.
   - Hold at 70°; release at 0.5 s.
6. `arm_3r_spatial`
   - Board: vertical, with the camera 35° oblique.
   - The base sits 0.10 m in front of the board.
   - Links p1 (0.15 m), p2 (0.15 m) and p3 (0.10 m).
   - j0 is about the board normal. j1 is about an axis in the board plane, perpendicular to p1. j2 is parallel to j1.
   - Motion: prescribed smooth joint trajectories. Each joint moves alone for 2 s, then all three move together.
   - motion_model spatial, dynamics_gt false, no free-motion segments.

Outputs per clip folder `<scene>__s<seed>/`:
- `metadata.yaml`:
  - source synthetic, camera_id "synthetic", intrinsics "camera_intrinsics.json"
  - board spec, orientation, motion_model and plane_offset_m
  - parts_hint, with labels and fixed flags
  - free_motion_segments_s and dynamics_gt
- `video.mp4`: H.264 via imageio-ffmpeg, `-preset veryfast`, crf 18, yuv420p, constant frame rate.
- `camera_intrinsics.json`:
  - K from fovy and resolution: fy = (H/2)/tan(fovy/2), fx = fy, cx = W/2, cy = H/2
  - confirm MuJoCo's principal point and half-pixel conventions with the board test
  - zero distortion, source synthetic
- `gt/camera_extrinsics.json`: {"T_W_C": ...}, in the OpenCV camera convention.
- `gt/tracks.json` and `.npz`:
  - source synthetic_gt
  - body frames used as part frames
  - all frames valid, quality 0
  - extent_P from the geom half-sizes
- `gt/kinematics.json` and `.npz`: source synthetic_gt, with the exact axes and with q, qd and qdd from the simulation.
- `gt/dynamics.json` and `gt/model.xml`, for dynamics scenes only:
  - `model.xml` contains the mechanism only (no board or backdrop) and follows the DynamicsModel rules.
  - `inertia_com_P` is assembled from MuJoCo's body inertia and iquat.
  - Joint damping, frictionloss, armature, stiffness and springref are all included.
  - `mass_scale_known: true`; `identifiability` is {rank = n_params, n_params, condition_number 1.0, unidentifiable: []}; `fit` is {segments_s from the metadata, replay_rms 0.0 per joint, window_s 0.5, method "synthetic_gt"}.
- `gt/masks/%06d.png`: uint8 label images from segmentation rendering. The label is the part index + 1, mapped geom to body to part.

Generation is deterministic for a given scene, seed and settings: the json and npz files are byte-identical across runs.
</synthetic_scenes>

<constraints>
- Python 3.11 via uv. No conda, no Docker.
- Touch nothing outside these locations:
  - the repo and its sibling worktrees
  - `~/twinfit-data`
  - `~/.config/twinfit`
- No network calls in unit tests. Live Meta calls happen only through `twinfit meta-smoke` and its one live test. Setup spend must stay at or below $1.
- No third-party robot models or datasets in the repository.
- Never commit `.private/`, `.env`, videos, or anything containing the API key. Add `tests/setup/test_no_secrets.py`, which scans tracked files for the patterns `LLM_[0-9]+_` and `MODEL_API_KEY=.+`.
- Do not implement any milestone's functionality beyond the oracle stubs.
- Documentation follows Ahaan's style: no em dashes, no filler.
</constraints>

<acceptance_criteria>
All of these must hold. Show each one with its command and result in your report.
1. `uv run twinfit doctor` shows every import OK and a successful offscreen render. Pinocchio may be "unavailable" only if Phase 2 recorded why.
2. `uv run ruff check .` is clean, and `uv run pytest -q` passes (live tests skipped), including the render tests.
3. `uv run twinfit synth --suite synth_small` produces 12 clip folders, and `uv run twinfit contracts-check` passes on each.
4. Board-frame test: on a rendered frame of every synth_small clip, OpenCV ChArUco detection plus solvePnP with the GT intrinsics recovers the GT T_W_C within 0.5° and 5 mm, with reprojection RMS ≤ 0.5 px.
5. Projection test: points sampled on each part's camera-facing geom surface, projected with the GT camera, fall inside that part's GT mask at least 95% of the time.
6. GT self-consistency:
   - Forward kinematics from the GT kinematics and q reproduces the GT tracks, with motion point RMS < 1e-6 m.
   - On dynamics_gt clips, `simulate` on `gt/model.xml`, started from the GT state at the beginning of each free-motion segment, reproduces the GT q within 1e-6 over that segment.
7. `uv run twinfit run <a synth_small clip> --impl stub` completes all five stages, and the run folder passes `contracts-check`.
8. `uv run twinfit perception --help`, and the same for the other five groups, prints the placeholder.
9. `TWINFIT_LIVE=1 uv run pytest -m live` passes, or the report states the exact reason (missing key or billing error).
10. The printables exist and their asserted sizes hold. The metadata template and the protocol document are written.
11. The repo is public on GitHub with CI green, or the report gives Ahaan the exact commands to finish this.
12. The three worktrees exist, and each one syncs and passes the fast test suite.
13. `git log --stat` shows no files from .private/, no videos and no secrets.
</acceptance_criteria>

<self_check>
Before writing the report, answer each of these in docs/ledger/setup.md:
- Could each milestone meet its acceptance criteria without editing any file you own? Walk through the criteria in 01, 02 and 03 one by one, and name the setup artifact each one relies on.
- Is every convention in conventions.py covered by at least one test that would fail if the convention flipped a sign or swapped axes?
- Does any test touch the network without the live marker?
- Is every unverified item marked as unverified?
- Is anything personal in a tracked file?
</self_check>

<stop_conditions>
Stop and ask Ahaan only in these cases:
1. `gh repo create` fails because the name already exists on his account.
2. A step would modify or delete anything outside the allowed locations.
3. Free disk space falls below 5 GB.

For a missing API key, a billing error or an unauthenticated gh, do not stop. Finish everything else, then list the exact actions Ahaan needs to take.
</stop_conditions>

<output_format>
Your final message uses this structure:
```
## Setup report
Status: COMPLETE | PARTIAL (reason)
Repository: URL or local path
Worktrees: path and branch for each of the three
Acceptance: table with columns criterion | command | result
Conventions: board +x, +y, +z in plain words; up_W; normal_toward_camera_W; gravity_W for each orientation
Unverified: list
Meta spend: $ for tag setup, and the global ledger total
Ahaan next: numbered actions with exact commands, including which prompt to start in which worktree
```
</output_format>
