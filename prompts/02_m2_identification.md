# Twinfit · Milestone 2 · Identification: tracks to kinematic tree, dynamics and MJCF

How to start: run Claude Code inside `../twinfit-m2` (branch `m2-identify`) and say "Read prompts/02_m2_identification.md and carry it out exactly."

<role_and_principles>
You own Twinfit's identification stages. They take part pose tracks and produce two things:
- a kinematic tree with joint axes
- a physically consistent dynamics model that MuJoCo can simulate

This is the technical core of the project. It is deterministic math, with no LLM calls.

Two other sessions are building perception (Milestone 1) and verification plus the agent (Milestone 3) in parallel, against the same frozen contracts. You never need their code. Your input is the setup's oracle tracks with controlled noise.

Operating principles:
1. Every estimate carries a residual and an identifiability statement.
   - If the data cannot determine a parameter, report it as unidentified.
   - Never present such a parameter as identified.
2. Validate through two independent routes wherever possible:
   - regressor-based identification in Pinocchio
   - replay validation in MuJoCo
   - comparison against ground truth through `twinfit.eval.metrics`
3. Use established methods and cite them in docs/m2_identification.md:
   - Sturm, Stachniss and Burgard (2011), JAIR 41: probabilistic kinematic model learning, BIC model selection, minimum spanning tree structure.
   - Ehrig et al. (2006), J. Biomech. 39: SCoRE. Ehrig et al. (2007), J. Biomech. 40: SARA.
   - Atkeson, An and Hollerbach (1986), IJRR: inertial parameter estimation.
   - Gautier and Khalil (1990), IEEE TRA: base parameters.
   - Traversaro et al. (2016), arXiv:1610.08703: full physical consistency and the triangle inequalities.
   - Wensing, Kim and Slotine (2017), arXiv:1701.04395: LMI physical consistency.
   - Rucker and Wensing (2022), IEEE RA-L: log-Cholesky inertia parameterization.

   Check each reference's details before writing them into docs. If you cannot verify a detail, cite only the authors and title.
4. Report measured numbers. Never tune on the evaluation set. When you miss a target, report it plainly together with the dominant cause.
5. Work autonomously with a task list. Stop only for <stop_conditions>.
</role_and_principles>

<context>
Twinfit turns phone video of a real mechanism into a MuJoCo model plus a trust certificate. It is built solo by Ahaan Kothari for the Meta Global AI Developer Hackathon (deadline Wed Nov 4, 2026, 9:00 PM ET). The repository is public, and all work must be built during the event.

Before planning, read these. They are frozen:
- CLAUDE.md
- docs/ARCHITECTURE.md
- `src/twinfit/contracts/`
- `src/twinfit/sim/replay.py`
- `src/twinfit/eval/metrics.py`
- `src/twinfit/stubs/oracle.py`

Your two stages:

| Stage | Entry point | Writes |
|---|---|---|
| kinematics | `twinfit.identify.kinematics_stage:run(ctx, KinematicsOptions)` | `kinematics.json`, `kinematics.npz` |
| dynamics | `twinfit.identify.dynamics_stage:run(ctx, DynamicsOptions)` | `dynamics.json`, `model.xml` |

Development input comes from the oracle perception stub:
- Command: `uv run twinfit run CLIP --impl-stage perceive=stub ...`
- Pass `oracle_noise` through a RunOptions file. It perturbs the ground-truth tracks.

Data:
- The suites synth_small and synth_tiny come from `twinfit.synth.suite_dir`.
- There are 12 synth_small clips:
  - 5 planar dynamics scenes: `pendulum_1r`, `double_pendulum_2r`, `lamp_spring_2r`, `slider_incline_1p`, `door_spring_1r_h`.
  - 1 spatial kinematics-only scene: `arm_3r_spatial`.
  - Each scene appears with 2 seeds.

Budget: no Meta API calls. The session file sets allow_spend false.
</context>

<instructions>

Phase 0. Orientation and plan
1. Create `.twinfit-session.yaml` with `tag: m2-dev`, `tag_cap_usd: 0` and `allow_spend: false`. Then run `uv sync --all-extras --dev` and `uv run twinfit doctor`.
2. Read the files listed in <context>.
3. Write a plan into `docs/ledger/m2.md`. For each module, record the math, the tests and the order of work.

Phase 1. Joint model fitting (`identify/joints.py`)
1. Work on each ordered pair: parent i and child j.
   - Relative poses are T_ij(t) = T_W_Pi(t)^-1 T_W_Pj(t), over frames where both parts are valid.
   - "world" has the identity pose.
2. Fit three candidate models:
   - **fixed**: the chordal mean rotation and the mean translation.
   - **revolute**:
     1. Axis direction: the principal direction of the rotation vectors of R_ij(t) R_ij(r)^T.
     2. Axis point: the SARA linear least squares. Find c_i and c_j with R_ij(t) c_j + p_ij(t) ≈ c_i for all t. Resolve the along-axis null direction with the minimum-norm solution.
     3. Refinement: nonlinear least squares with scipy `least_squares` over the axis (2 DoF), the point (2 DoF perpendicular to the axis) and the per-frame angle.
   - **prismatic**:
     - Rotation is constant.
     - Direction: the principal component of p_ij(t) - p_ij(r).
     - A per-frame displacement along that direction.
3. Under the planar constraint, which follows tracks.motion_model unless the option overrides it:
   - the revolute axis is fixed to the plane normal
   - the prismatic direction lies in the plane
4. Measure residuals on child sample points, in metres:
   - Use the extent corners when an extent is available.
   - Otherwise, use the 8 corners of a 0.1 m cube centered on the child frame origin.
5. Select the model with BIC, as in Sturm et al. (2011).
   - Estimate the noise σ from track quality, or from the fixed-model residual floor.
   - Record type_scores.
   - `force_joint_types` overrides the selection.
6. Sign rule for the axis:
   - Planar: axis_W has a positive dot product with `normal_toward_camera_W`.
   - Spatial: the first nonzero component of axis_W is positive.
7. Unit tests:
   - Generate relative pose sequences in code for each joint type, at several noise levels.
   - Cover degenerate cases:
     - a tiny motion range, which should give low confidence
     - rotation about an axis that passes through the child origin
     - near-collinear prismatic noise

Phase 2. Tree, joint trajectories and the kinematics stage (`identify/tree.py`, `identify/kinematics_stage.py`)
1. Choose the base in this order:
   1. `force_base_part`.
   2. Otherwise, the part with `is_base_hint` and the least motion.
   3. Otherwise, "world" when a single moving part has no static partner.
   4. Otherwise, the part with the least motion in W.
2. Build the structure:
   - Build the complete graph over all parts plus "world".
   - The edge cost is the BIC of the best non-fixed model.
   - Take the minimum spanning tree and orient it away from the base.
   - Attach static parts with fixed joints.
3. Use the first frame where all parts are valid as the reference frame.
4. For each joint:
   - Compute q(t) from the final model, unwrapped, and zero at the reference frame.
   - Compute qd and qdd with a Savitzky-Golay filter: window about 0.1 s (rounded to an odd frame count), polynomial order 3. Mark samples near gaps as invalid.
5. Forward kinematics check:
   - Rebuild each part's motion from the base pose and q(t).
   - Store the per-part motion point RMS in `fit` and in the stage metrics.
6. `kinematics_stage.run` writes the contract files and honors every KinematicsOptions field.
7. Add the command `twinfit identify eval-kin --suite synth_small --noise TRANS_MM,ROT_DEG`.
   - It runs oracle perceive with that noise, then real kinematics, through the runner.
   - It compares against GT with `compare_kinematics`.
   - It writes `docs/media/m2_kin_eval.csv` and a summary table.

Phase 3. MJCF builder and cross-engine check (`identify/mjcf.py`, `identify/pin_model.py`)
1. Build `model.xml` following the contract:
   - Each child body's pose relative to its parent is T_Pparent_Pchild at the reference frame.
   - Joint `pos` and `axis` are expressed in the child body frame.
   - Inertials come from the current parameters via `fullinertia`.
   - Visualization geoms come from the extent.
   - Gravity comes from the tracks.
   - No contacts.
2. Build the same model in Pinocchio, either programmatically or through Pinocchio's MJCF parser. Choose one and test it.
3. Test: at 100 random states, Pinocchio RNEA must equal MuJoCo inverse dynamics (`mj_inverse`, with damping, friction and springs zeroed) within 1e-8 relative.
4. Test the regressor parameter ordering:
   - For random inertias, Y·π from `computeJointTorqueRegressor` must equal RNEA, with π from `Inertia.toDynamicParameters()`.
   - Pin down Pinocchio's parameter order with this test. Do not assume it.
5. If Pinocchio is unavailable:
   - Build the regressor numerically from `mj_inverse`, using linearity in the inertial parameters.
   - Apply the same test.
   - Document the choice.

Phase 4. Dynamics identification (`identify/dynamics.py`, `identify/dynamics_stage.py`)

Model of free motion, with no actuation:

M(q)q̈ + c(q, q̇) + g(q) + D q̇ + F_c tanh(q̇/ε) + K q - K q_ref = 0, with ε = 0.05 rad/s (or m/s for prismatic joints).

1. Free-motion segments:
   - Take `segments_s` from the options.
   - If absent, take `tracks.free_motion_segments_s`.
   - If that is also absent, detect a segment automatically: from the last time before the largest motion when every joint has |q̇| below a small threshold, to the end of the clip. Flag auto-detected segments in extras.
   - Trim half a Savitzky-Golay window from each segment edge.
2. Linear stage:
   - Stack Y(q, q̇, q̈), with 10 inertial parameters per moving link in Pinocchio order.
   - Add columns for viscous damping.
   - Add a column for Coulomb friction when `fit_coulomb` is set.
   - Add columns for stiffness and for stiffness·q_ref when `fit_stiffness` is set.
   - This gives the homogeneous system A θ = 0. Weight each joint's rows by that joint's residual scale.
3. Identifiability:
   - Take the SVD of A after column scaling. Use a numerical rank tolerance and document it.
   - Find base parameters by QR with column pivoting (Gautier and Khalil 1990).
   - Map null-space directions to readable combinations, such as "absolute mass scale", "I_xx of p1 (out-of-plane)" and "m·c_z of p1".
   - For passive data, the global scale is always a null direction. Resolve it as follows:
     - With metadata or option masses, use them.
     - Otherwise, set the first moving link's mass to its geometric prior and set `mass_scale_known = false`.
4. Physically consistent estimate:
   - Optimize over the log-Cholesky parameterization of each link's pseudo-inertia (Rucker and Wensing 2022), together with log damping, log friction and log stiffness.
   - Minimize ‖A θ‖² plus a small ridge toward the geometric prior.
   - The prior is a uniform-density box of the part's extent, at 600 kg/m³, unless a mass is given.
   - List in `prior_dominated` every parameter whose posterior is dominated by the prior. Use a documented sensitivity threshold.
5. Replay refinement:
   - Use multiple shooting on `twinfit.sim.replay.windowed_replay_rms`, with window `window_s` (default 0.5 s). This is the same function Milestone 3 uses to certify.
   - Optimize only the identifiable directions, in the same parameterization, with scipy `least_squares` (trust region reflective).
   - Stop when no iteration improves the cost by more than 1% over 3 iterations.
6. `dynamics_stage.run`:
   - writes `dynamics.json` and `model.xml`
   - reports replay RMS per joint, the rank and the condition number as metrics
   - honors every DynamicsOptions field
7. Spatial clips with `dynamics_gt false` and no free-motion segments: return status skipped with a clear message.

Phase 5. Evaluation
1. Add the command `twinfit identify eval-dyn --suite synth_small --noise TRANS_MM,ROT_DEG`, run on the dynamics_gt clips with oracle tracks. It reports:
   - windowed replay RMS
   - errors in identifiable quantities against GT:
     - natural frequency √(m g l_c / I_o) for gravity pendulums
     - √(k/I_o) for spring joints
     - damping ratio
     - stiffness
   - for 2-DoF chains, the relative error of each base-parameter combination
2. Noise sweep, at translation noise of 0, 0.5, 1, 2 and 4 mm, with rotation noise of 0.3° per mm. Plot all errors against noise in `docs/media/m2_noise_sweep.png`, and add a table to docs/m2_identification.md.
3. Run kinematics evaluation on the spatial clips.

Priorities:
- P0:
  - Phases 1 to 4 for revolute and prismatic joints, in 1-DoF and 2-DoF chains, including stiffness.
  - Phase 5 at noise levels 0 and 1 mm.
- P1:
  - the full noise sweep
  - auto segment detection
  - the spatial kinematics evaluation
- P2:
  - `twinfit identify export-urdf`
  - a cvxpy LMI solution compared against log-Cholesky
</instructions>

<constraints>
- Edit only these paths:
  - src/twinfit/identify/
  - tests/identify/
  - docs/m2_identification.md
  - docs/ledger/m2.md
  - docs/media/m2_*
  - the `identify` optional-dependency group, plus uv.lock
- No Meta API calls, and no network calls in tests.
- Contracts are frozen. Write change requests to `docs/contract_change_requests/m2.md`, and use extras in the meantime.
- Never read gt/ inside a stage. Only evaluation commands and tests may read it.
- Use `twinfit.sim.replay` for all replay, without reimplementing it, so fitting and certification agree.
- Follow the code style in CLAUDE.md. Prefix commits with [m2], and commit after each phase.
</constraints>

<acceptance_criteria>
These apply to synth_small, using oracle tracks with 1 mm / 0.3° noise unless stated otherwise:
1. Joint type is correct for every joint in every clip, including the spatial clips.
2. Axis direction error: median ≤ 0.5°, maximum ≤ 2°.
3. Revolute axis position error: median ≤ 2 mm, maximum ≤ 5 mm.
4. Base and parent-child structure are correct on every clip. "world" and a static base count as equivalent.
5. Kinematic reconstruction motion point RMS is ≤ 1.5 × the injected noise RMS.
6. Dynamics, on the 10 dynamics_gt clips:
   - windowed replay RMS (0.5 s) ≤ 1.0° for revolute joints and ≤ 2 mm for prismatic joints
   - single-DoF natural frequency within 2%
   - damping ratio within 20%
   - stiffness within 10%
   - for 2-DoF chains, base-parameter relative errors reported
7. Every identified link passes positive definiteness and the triangle inequalities by construction, and a test proves it.
8. The Pinocchio-versus-MuJoCo inverse dynamics test passes, or the numeric fallback passes the same test.
9. Identifiability on passive pendulum data:
   - Without a given mass, absolute mass scale and out-of-plane inertia entries are flagged as unidentifiable.
   - With a given mass, the mass is reproduced within 5%.
10. `contracts-check` passes on every run folder.
11. `uv run pytest -q` and `uv run ruff check .` pass.
</acceptance_criteria>

<self_check>
Answer each of these in docs/ledger/m2.md before the final report:
- Does each claim of "identified" correspond to a direction outside the numerical null space? Show the rank and condition number for each clip type.
- Were any thresholds, window lengths or regularization weights changed after you looked at evaluation results? If so, revert them or justify them.
- Do the fitting and certification paths use the same replay function with the same settings?
- Does `git diff --name-only main...HEAD` list only owned paths?
- Which statements in your report are unverified?
</self_check>

<stop_conditions>
1. Pinocchio is unavailable and the numeric regressor fails its test: report, and continue with the kinematics work.
2. A contract makes a required output impossible: write the change request, use extras, and continue.

Missing a target is not a reason to stop. Report it.
</stop_conditions>

<output_format>
```
## Milestone 2 report
Status: COMPLETE (all P0 and acceptance met) | PARTIAL (what is missing)
Acceptance: table with columns criterion | measured | target | pass/fail
Identifiability summary: per scene type, the rank / n_params and the unidentified combinations
Method notes: choices made (regressor source, rank tolerance, prior weight) and why
Contract change requests: none, or the file path
Unverified claims: list
```
</output_format>
