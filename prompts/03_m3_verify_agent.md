# Twinfit · Milestone 3 · Verification certificate, Muse Wonder repair agent, benchmark and report

How to start: open Claude Code inside `../twinfit-m3` (branch `m3-verify-agent`) and say "Read prompts/03_m3_verify_agent.md and carry it out exactly."

<role_and_principles>
You own Twinfit's trust layer and its product surface. That means four things:
- the physical-consistency certificate
- the Muse Wonder agent that repairs failed runs
- the benchmark harness
- the HTML report used in the demo

Two other sessions are building perception (Milestone 1) and identification (Milestone 2) at the same time, against the same frozen contracts. You build against the setup's oracle stubs, against fault-injected variants of their outputs, and against synthetic ground truth. At integration, the registry swaps in the real stages. Your code must work with either.

Operating principles:
1. **The certificate is an independent auditor.** Prove its value by catching faults it did not create, measured with fault injection. Never by inspecting your own outputs.
2. **The agent is a controller around tools, not an author of facts.**
   - The model chooses actions and cites check ids.
   - Code produces every number a user sees.
   - Code validates every citation.
   - Code selects the best attempt with a fixed objective.
3. **The agent may say the evidence is insufficient.** Give it a `request_human` tool for exactly this.
4. **Read the Meta docs before writing agent code.** Never guess tool-calling formats or reasoning-state handling.
5. **Report measured results.** No tuning on the evaluation set.
6. **Work autonomously** with a task list. Stop only for <stop_conditions>.
</role_and_principles>

<context>
Twinfit turns phone video of a real mechanism into a MuJoCo model plus a trust certificate. It is built solo by Ahaan Kothari for the Meta Global AI Developer Hackathon. The deadline is Wed Nov 4, 2026, 9:00 PM ET. The repository is public, and all work must be built during the event.

How this milestone maps to judging:
- Muse Wonder as an agent counts toward "Use of Meta AI tooling" (25%) and the Best Agentic Use-Case prize.
- The report and overlays carry the "Demo" score (25%).

Before planning, read these frozen files:
- CLAUDE.md
- docs/ARCHITECTURE.md
- docs/meta_api_notes.md
- `src/twinfit/contracts/`, including `compute_verdict`
- `src/twinfit/sim/`
- `src/twinfit/eval/metrics.py`
- `src/twinfit/stubs/oracle.py`
- `src/twinfit/runner/`

Your entry points:

| Entry point | Signature or command |
|---|---|
| Certify stage | `twinfit.verify.stage:run(ctx, CertifyOptions)`, writes `certificate.json` and `certificate.md` |
| Report stage | `twinfit.report.stage:run(ctx, options=None)`, writes `index.html` and media into its stage folder |
| CLI apps (each exposes an `app`) | `twinfit.verify.cli`, `twinfit.agent.cli`, `twinfit.bench.cli`, `twinfit.report.cli` |

Data comes from `twinfit.synth.suite_dir`:
- synth_small: 12 clips. 10 have dynamics ground truth; the 2 `arm_3r_spatial` clips are kinematics only.
- synth_tiny: used for tests.

Real clips may appear in `~/twinfit-data/real/` during your session.

Budget: tag m3-dev has a $25 cap. The global $120 cap is shared across all sessions.
</context>

<instructions>

Phase 0. Orientation and plan
1. Create `.twinfit-session.yaml` with `tag: m3-dev`, `tag_cap_usd: 25` and `allow_spend: true`. Run `uv sync --all-extras --dev` and `uv run twinfit doctor`.
2. Read the files listed in <context>.
3. Read these pages on https://dev.meta.ai with WebFetch:
   - /docs/tool-calling
   - /docs/protocols/responses
   - /docs/reasoning
   - /docs/structured-output
   - /docs/image-understanding
   - /docs/prompt-caching
   - /docs/error-handling
   - /docs/pricing-rate-limits
   - /docs/agent-frameworks
   - the cookbook agent-patterns section

   If a page fails to load, use https://github.com/meta-models/meta-model-cookbook instead.
4. Write the facts you rely on, each with its URL, into an "API notes" section of `docs/m3_verify_agent.md`. Cover at least:
   - how function tools are declared and returned
   - how to continue a tool loop with reasoning preserved (previous_response_id or encrypted reasoning replay)
   - image input format
   - prices
5. Write your plan into `docs/ledger/m3.md`.

Phase 1. Certificate (`verify/checks.py`, `verify/thresholds.yaml`, `verify/stage.py`)

Build each check as a pure function. It takes contract objects and paths, and returns a check record. If an input is missing, it returns `not_applicable` with the reason. Put thresholds in `verify/thresholds.yaml`, with a one-line rationale per value, and honor `thresholds_override`.

| ID | Check | Rule (defaults) |
|---|---|---|
| C01 | schema_valid | every upstream contract file loads and validates; fail otherwise |
| C02 | tree_structure | single base, no cycles, every part reachable, joint ids unique; fail otherwise |
| C03 | axis_wellformed | unit norm within 1e-6, finite values, revolute joints have a point; fail otherwise |
| C04 | kinematic_reconstruction | FK from base pose and q(t) against observed part motion, motion point RMS; warn > 3 mm, fail > 10 mm |
| C05 | track_quality | valid fraction per moving part; warn < 0.90, fail < 0.60 |
| C06 | mass_positive | mass_kg > 0 for every link; fail otherwise |
| C07 | inertia_pd | eigenvalues of inertia_com_P > 0; fail otherwise |
| C08 | triangle_inequality | principal moments satisfy J1 + J2 ≥ J3 for every permutation, within 1e-9 relative; fail otherwise |
| C09 | com_inside_extent | COM inside the part extent inflated by 25%; fail otherwise; not_applicable without an extent |
| C10 | replay_windowed | `twinfit.sim.replay.windowed_replay_rms` over the free-motion segments with window 0.5 s; revolute warn > 1.5°, fail > 5°; prismatic warn > 3 mm, fail > 10 mm |
| C11 | passivity_energy | 10 s free simulation of the fitted model from the observed release state: total energy (kinetic, gravity, spring) never rises by more than 2% of its initial value within any 0.5 s window; fail otherwise |
| C12 | sim_stability | same simulation: no NaN, no MuJoCo warnings; fail otherwise |
| C13 | identifiability | one caveat per unidentifiable combination from dynamics.json; warn if condition number > 1e6 |
| C14 | singularity_margin | chains with ≥ 2 joints: min over observed configurations of σ_min/σ_max of the distal part's positional Jacobian (in-plane rows for planar models, from `mj_jacBody`); warn < 0.05 |
| C15 | cross_engine | load model.xml in MuJoCo and in Pinocchio's MJCF parser; inverse dynamics agree within 1e-6 relative at 50 random states; not_applicable if Pinocchio cannot parse it |
| C16 | rest_pose | if the clip ends at rest (|q̇| below threshold for the last 0.5 s), compare the fitted model's nearest static equilibrium to the final observed q; warn > 2°, fail > 8° (or 5 mm / 20 mm) |

Then:
1. The verdict always comes from `compute_verdict`.
2. `certificate.md` contains:
   - the verdict and caveats
   - the checks table
   - one plain-language sentence for each failed or warned check, filled from templates (never an LLM)
3. Fault injection lives in `verify/faults.py` and runs as `twinfit verify faults --suite synth_small`. Apply each fault, one at a time, to the oracle outputs of every applicable clip:

| ID | Fault |
|---|---|
| F01 | negative mass |
| F02 | inertia with a negative eigenvalue |
| F03 | triangle inequality violated with positive eigenvalues |
| F04 | COM moved to twice the extent |
| F05 | axis rotated 10° |
| F06 | revolute point shifted 20 mm |
| F07 | revolute replaced by prismatic, with a best-fit direction |
| F08 | damping × 10 |
| F09 | stiffness × 3 (spring clips), or gravity rotated 30° in-plane (other clips) |
| F10 | q trajectory sign flipped |
| F11 | duplicate parent or cycle in the tree |
| F12 | NaN in a parameter |
| F13 | track valid fraction cut to 50% for one part |

   - Record which check fires for each fault.
   - Write a confusion table to `docs/media/m3_faults.csv` and summarize it in `docs/m3_verify_agent.md`.
   - Run the same checks on clean ground truth to measure false failures.

Phase 2. Report (`report/`)

1. Build one HTML page per run.
   - Inline CSS, no network requests, so it works offline during the demo.
   - Media files sit next to it in the stage folder.
2. Panels, in order:
   1. **Header:** clip, verdict badge, caveats, total cost, runtime.
   2. **Video:** the perception overlay if present, plus a Twinfit overlay rendered here. The Twinfit overlay shows:
      - observed part outlines (extent rectangles projected with the track poses, K and T_W_C)
      - the recovered joint axes
      - a simulated ghost: forward kinematics of the windowed-replay q, drawn in a second color
   3. **Joint plots:** observed q against the replay windows, with residuals (matplotlib PNG).
   4. **Kinematic tree:** an inline SVG diagram.
   5. **Certificate table:** with evidence.
   6. **Identifiability:** a table, plus the prior-dominated parameters.
   7. **Agent trace:** a timeline, if a trace exists.
3. Any panel whose inputs are missing shows "not available: <reason>".
4. `twinfit report open RUN_DIR` opens the page in the default browser.

Phase 3. Benchmark harness (`bench/`)

1. Suites:
   - `synth_small`
   - `real_all`: every folder in `~/twinfit-data/real/` whose name does not start with "_"
2. `twinfit bench run --suite S [--impl auto|real|stub] [--impl-stage ...] [--agent off|on] [--noise TRANS_MM,ROT_DEG] [--scenario IDS]`, where `IDS` is one scenario id, a comma-separated list, or `all` (S1 to S5)
   - Runs the pipeline, or the agent, on every clip.
   - Computes metrics with `twinfit.eval.metrics` wherever gt exists.
   - Collects certificate statistics.
   - Writes `runs/bench/<bench_id>/results.parquet` and `summary.md`, with plots.
3. Columns:
   - identity: clip_id, scene, seed, impl_map, agent, scenario
   - outcome: stage statuses, verdict, n_fail, n_warn
   - accuracy: mask_iou_mean, point_rms_mm_median, joint_type_acc, axis_err_deg_median, axis_pos_err_mm_median, q_rms, replay_rms_max
   - cost: cost_usd, duration_s
4. Document the schema in `docs/m3_verify_agent.md`.

Phase 4. Agent (`agent/`)

1. **Tools.** Declare these as function tools in the documented format:

| Tool | Behavior |
|---|---|
| `run_stage(stage, options)` | reruns the stage and everything downstream inside a new attempt folder `runs/<run_id>/attempts/<k>/`; certify always runs afterwards (code does this, not the model); returns a compact summary |
| `read_summary(artifact)` | artifact is one of tracks, kinematics, dynamics, certificate, perception_report; returns a compact JSON summary computed by code, never raw arrays |
| `view_frames(times_s, overlay)` | P1; at most 4 frames, overlay one of masks, axes, ghost, none; returns rendered images for Wonder to inspect |
| `compare_attempts()` | returns the objective table |
| `request_human(message)` | ends the loop with status needs_human |
| `finish(attempt_id, claims)` | claims is a list of {check_id or metric_key, template_id}; ends the loop |

2. **Controller.** Code, not the model:
   - selects the best attempt by a lexicographic objective: n_fail, then n_warn, then max normalized replay RMS, then max kinematic RMS
   - enforces limits: 8 tool calls, $3 per clip, 3 `view_frames` calls; on reaching a limit, it finalizes with the best attempt
3. **Model.**
   - Use `muse-wonder-1.0` with reasoning preserved across turns, using whichever mechanism the docs support for tool loops.
   - Route every call through `twinfit.meta` under the current session tag (`m3-dev` here, `integration` during integration), so the tag caps always apply.
4. **System prompt** (`agent/prompts/system.md`):
   - The model's role is a repair controller for a measured pipeline.
   - Every claim must cite a check_id or metric key that appears in a tool result.
   - Include a table mapping failure patterns to the levers that fix them:

| Failure pattern | Lever |
|---|---|
| C04 or C05 failing on one part | `exclude_time_ranges_s` or `box_prompts` |
| ambiguous `type_scores` | `force_joint_types` |
| C10 failing early in a segment | `segments_s` |
| C10 or C16 on spring-like motion | `fit_stiffness` |
| known masses | `mass_kg` |

   - It may call `request_human` when no lever addresses the failure or the data are insufficient. Say plainly that this is a good outcome, not a failure.
   - It must never state a number of its own.
5. **Final summary.** Code renders it from the `finish` claims through templates, inserting numbers from JSON. Citations to ids that do not exist are rejected, and the model gets one chance to fix them.
6. **Trace.** Write `agent_trace.jsonl` with one line per step: tool, arguments, result summary, tokens, cost and duration.
7. **Scenarios** (`agent/scenarios.py`). These wrap the oracle stages and corrupt their outputs unless the fixing option is present, so the agent can be developed without Milestones 1 and 2:

| ID | Corruption | Fixed by |
|---|---|---|
| S1 | wrong joint type on one joint | `force_joint_types` |
| S2 | a 0.4 s glitch window in one part's track | `exclude_time_ranges_s` |
| S3 | free-motion segment that includes 0.3 s of hand contact | `segments_s` |
| S4 | missing spring term on lamp_spring_2r | `fit_stiffness: true` |
| S5 | one part valid only 40% of the time (unfixable) | expected outcome: `request_human` |

8. **Tests.**
   - Record live responses once into `tests/agent/cassettes/`, and replay them offline through the `twinfit.meta` cache.
   - A cache miss fails the test with instructions to re-record using `TWINFIT_LIVE=1`.
   - S1 to S4 must reach VERIFIED or VERIFIED_WITH_CAVEATS within 6 tool calls.
   - S5 must end with `request_human`.
   - A test asserts that every number in the final summary appears in some tool-result JSON.
9. **CLI.** `twinfit agent run CLIP_DIR [--scenario ID] [--impl ...] [--impl-stage ...]`, with ID one of S1 to S5.

Phase 5. Agent value experiment (P1)
1. Run `twinfit bench run --suite synth_small --scenario all --agent off`, then the same command with `--agent on`.
2. Tabulate verdicts and errors into `docs/media/m3_agent_value.csv`, with a summary in the docs.

Priorities:
- P0:
  - Phase 1 including the fault suite
  - Phase 2: header, plots, tree, certificate and identifiability panels
  - Phase 3 with stubs
  - Phase 4: controller, all tools except `view_frames`, scenarios S1 to S5, cassette tests
- P1:
  - `view_frames`
  - the ghost overlay video
  - live scenario runs
  - Phase 5
- P2:
  - Kinematic sanity check on a MuJoCo Menagerie arm loaded at runtime through robot_descriptions (check its license; never commit it).
  - Muse Spark Open as an alternative agent model, only if Ahaan confirms partner inference credits.
</instructions>

<constraints>
- Edit only these paths:
  - `src/twinfit/{verify,agent,bench,report}/`
  - `tests/{verify,agent,bench,report}/`
  - `docs/m3_verify_agent.md`, `docs/ledger/m3.md`, `docs/media/m3_*`
  - the `verify` and `agent` optional-dependency groups, plus uv.lock
- Contracts are frozen. Write change requests to `docs/contract_change_requests/m3.md`, and use extras in the meantime.
- Stages never read gt/. Bench and fault tooling may read it.
- All Meta calls go through `twinfit.meta`, are tagged, cached and budget-checked. No live calls in non-live tests.
- No number in any user-facing summary may come from model text.
- The report must work offline: no CDN scripts or fonts.
- Do not change how Claude Code itself is configured or routed.
- Follow the code style in CLAUDE.md. Prefix commits with [m3], and commit after each phase.
</constraints>

<acceptance_criteria>
1. **Faults detected.** Every fault is detected by at least one check on every clip where it applies. For any fault you exclude on a clip, write the justification in the docs (for example, the motion range is too small for F08 to be observable).
2. **No false failures.** Clean ground truth across synth_small produces zero fail statuses. Every warn is explained.
3. **Oracle certificates.** Every synth_small oracle run gets VERIFIED or VERIFIED_WITH_CAVEATS, and the caveats match dynamics.identifiability.
4. **Report.** Generated for 3 synth clips, with every panel present or explicitly marked not available.
5. **Bench.** `twinfit bench run --suite synth_small --impl stub` completes. The parquet schema is documented.
6. **Agent.**
   - Offline cassette tests pass for S1 to S5.
   - Each scenario has run live once, with cost ≤ $1 per scenario and the actual cost and steps reported.
7. **No model-authored numbers.** The test asserting this passes.
8. **Budget.** The tag m3-dev total is ≤ $25.
9. **Tests and lint.** `uv run pytest -q` and `uv run ruff check .` pass.
</acceptance_criteria>

<self_check>
In docs/ledger/m3.md, before reporting:
- Do the fault results come from the actual check functions on actual model files, not from hand-built certificate objects?
- Did you change any threshold after seeing fault or clean results? If so, justify it, or revert and report the outcome honestly.
- Could the agent ever present a number it computed itself? Trace the code path that renders the summary.
- Does `git diff --name-only main...HEAD` list only owned paths?
- Which statements in your report are unverified?
</self_check>

<stop_conditions>
1. BillingError, missing key, or a reached cap: stop live work, finish the offline phases, and report.
2. The documented tool-calling or reasoning-state mechanism does not work live:
   - record the exact error
   - implement the simplest documented alternative
   - say so in the report
3. A contract blocks a check: write the change request, mark the check not_applicable with the reason, and continue.
</stop_conditions>

<output_format>
```
## Milestone 3 report
Status: COMPLETE (all P0 and acceptance met) | PARTIAL (what is missing)
Acceptance: table (criterion | measured | target | pass/fail)
Fault matrix summary: fault -> detecting checks, misses with justification
Agent: per scenario, outcome, tool calls, cost
Cost: tag m3-dev total, global total
API discrepancies found: list
Contract change requests: none or file path
Unverified claims: list
```
</output_format>
