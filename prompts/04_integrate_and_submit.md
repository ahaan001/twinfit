# Twinfit · Prompt 4 · Integrate, evaluate and prepare the submission

How to start: open Claude Code in the main repository folder (branch `main`, for example `~/twinfit`) after the three milestone sessions have reported. Then say: "Read prompts/04_integrate_and_submit.md and carry it out exactly."

<role_and_principles>
You are the integrator. Your job is to merge three independently built milestones, run the real pipeline end to end on synthetic and real clips, and produce the submission material. Every number in that material must come from measured results.

Operating principles:
1. **The submission states only measured results.** A weak number reported honestly beats a strong number nobody can reproduce.
2. **Fix integration bugs wherever they are, and log every fix.** For each fix in a milestone's code, record the file, the reason and the commit in `docs/ledger/integration.md`.
3. **Ask before anything leaves the machine.** Get Ahaan's explicit approval before publishing anything that is his personal data. That covers real videos, release assets and social posts.
4. **Work autonomously** with a task list. Stop only for <stop_conditions>.
</role_and_principles>

<context>
Project: Twinfit. It turns a phone video of a real mechanism into a MuJoCo model plus a trust certificate. Built solo by Ahaan Kothari for the Meta Global AI Developer Hackathon.

Deadline and rules:
- Submission is due Wed Nov 4, 2026, 9:00 PM ET.
- Ahaan should be able to record the demo by Tue Nov 3 evening.
- The repository must be public.
- The demo may only show work built during the event.
- A project that fails to identify its new work can be disqualified at the judges' discretion.

Read these first:
- CLAUDE.md
- docs/ARCHITECTURE.md
- every file in docs/ledger/
- docs/m1_perception.md, docs/m2_identification.md, docs/m3_verify_agent.md
- docs/contract_change_requests/ (if present)
- .private/hackathon_context.md, sections 7 to 9: submission requirements, judging and prizes

Branches to merge:
- `m1-perception`
- `m2-identify`
- `m3-verify-agent`

Budget: tag `integration` has a $40 cap. Report the global ledger total at the end.
</context>

<instructions>

Phase 0. Preparation
1. Create `.twinfit-session.yaml` with:
   - `tag: integration`
   - `tag_cap_usd: 40`
   - `allow_spend: true`
2. Run `uv run twinfit cost` and record the spend so far.
3. Read the inputs. Summarize each milestone's status, gaps and change requests at the top of `docs/ledger/integration.md`.

Phase 1. Merge
1. Run `git fetch --all`. Merge into `main` with `--no-ff`, in this order:
   1. `m2-identify`
   2. `m3-verify-agent`
   3. `m1-perception`
2. Conflicts are expected in only two files:
   - `pyproject.toml`: combine the optional-dependency groups.
   - `uv.lock`: regenerate it with `uv lock`.
   - Any other conflict is a stop condition.
3. After each merge, run all three:
   - `uv sync --all-extras --dev`
   - `uv run ruff check .`
   - `uv run pytest -q`
   Fix failures, and log each fix.
4. Contract change requests:
   - Implement one only if every affected owner's code can be adapted and tested in this session.
   - Otherwise, record it under Limitations in the README.

Phase 2. End-to-end on synthetic data

Run each of these and save the bench ids:
- `uv run twinfit bench run --suite synth_small --impl stub` (the oracle upper bound)
- `uv run twinfit bench run --suite synth_small --impl real --agent off`
- `uv run twinfit bench run --suite synth_small --impl real --agent on`
- `uv run twinfit bench run --suite synth_small --scenario all --agent off`, then the same with `--agent on` (the agent value experiment)

The two `arm_3r_spatial` clips are expected to fail markerless perception in the real runs. Keep them in the tables with that reason.

Then compare real against stub per stage. That locates the losses: perception error against identification error.

Phase 3. Real clips
1. Calibrate every `camera_id` that has a calibration video:
   `uv run twinfit perception calibrate`
2. For every Take A clip, run `uv run twinfit perception marker-gt`.
3. Run `uv run twinfit bench run --suite real_all --impl real` twice, once with `--agent off` and once with `--agent on`.
4. If a real clip fails, keep the failure in the results with its reason. Do not drop it.

Phase 4. Results in the README
1. Write `scripts/fill_readme_results.py`. It reads the bench parquet files and replaces the content between `<!-- RESULTS:BEGIN -->` and `<!-- RESULTS:END -->` with four tables:
   - synthetic accuracy: stub against real
   - certificate fault detection: from `docs/media/m3_faults.csv`
   - agent value: off against on
   - real clips: verdicts, plus marker-GT errors where they exist
2. Each table gets a one-line definition of every column.
3. Run the script. Commit both the script and the README.

Phase 5. Demo assets
1. Choose the clips:
   - The showcase clip is the real clip with the best verdict. Break ties by the lowest replay RMS.
   - The repair clip is one agent run that either repaired a failure or correctly called `request_human`.
2. Generate their reports.
3. Copy the key media into `docs/media/`, at most 20 MB in total:
   - overlay and ghost videos
   - plots
   - screenshots
4. Write `docs/DEMO_SCRIPT.md`, a 3-minute shot list. Each shot gets its timestamps, what is on screen, the exact command or file to open, and voiceover text. Write the voiceover in Ahaan's voice: first person, direct, no filler, no em dashes.

   | Time | Shot |
   |---|---|
   | 0:00 to 0:15 | Hook: phone video in, verified simulation twin out |
   | 0:15 to 0:45 | Pipeline, and the Meta models used at each step |
   | 0:45 to 1:30 | Real clip: overlay, ghost replay and certificate |
   | 1:30 to 2:10 | The certificate catching injected faults, and the agent repairing a run |
   | 2:10 to 2:40 | Benchmark numbers |
   | 2:40 to 3:00 | Who benefits and what is next |

5. Mark every shot as built during the event.

Phase 6. README and documentation
1. Complete every README section from measured data. It must include:
   - "Meta models used", saying where each model is used and why
   - "Quickstart", with commands verified on this machine
   - "Capturing your own data", which links to docs/TEST_DATA_PROTOCOL.md
   - "Limitations", stated plainly:
     - markerless capture is planar only
     - absolute mass is unknown without a weighed part
     - plus anything the benchmarks revealed
2. Update THIRD_PARTY.md.
3. In `docs/ledger/integration.md`, add a section "Built during the event". It lists the commit ranges for each milestone with their dates, all within Oct 31 to Nov 4, 2026.

Phase 7. Publication hygiene
1. Scan the full history for secrets: `git log -p --all | grep -nE 'LLM_[0-9]+_|MODEL_API_KEY=.+'`. The scan must produce no output.
2. Confirm that no tracked file contains:
   - anything under .private/
   - an email address
   - team_id or project_id values
3. Confirm the repository is public: `gh repo view --json visibility`.
4. Confirm CI is green on main.
5. Ask Ahaan which real clips may be published. Then build a sample-data zip of at most 100 MB from only the approved clips. Do not upload anything without his yes.
6. With approval, create the GitHub release `v1.0.0` and attach the sample zip. Tag it.

Phase 8. Submission pack

Write `docs/SUBMISSION.md` containing:
- project name and track (Indie)
- problem statement ("Push the Frontier")
- a 100-word and a 250-word description, in Ahaan's voice
- the Meta models used
- repository URL, and a placeholder for the demo video URL
- team: Ahaan Kothari
- bonus prizes that apply (Best Agentic Use-Case; Most Viral, if he posts)
- an X post draft of at most 280 characters
- a final checklist:
  - [ ] repo public
  - [ ] video uploaded and accessible
  - [ ] submitted at the Cerebral Valley page before 9:00 PM ET Nov 4
</instructions>

<constraints>
- Every figure in README, DEMO_SCRIPT and SUBMISSION is produced by a script or read from bench output, and its source is cited in a comment or footnote.
- Do not delete failed runs or omit failing clips from tables.
- Do not change frozen contracts unless Phase 1 step 4 allows it.
- Do not change how Claude Code itself is configured or routed.
- Commits are prefixed [int].
</constraints>

<acceptance_criteria>
1. All three branches are merged. `uv run pytest -q` and `uv run ruff check .` pass on main, and CI is green.
2. Bench results exist for:
   - synth_small, with stub, real and agent on/off
   - the S1 to S5 agent value runs
   - real_all, when real clips exist
3. The README Results section is generated by the script, and re-running the script produces no diff.
4. docs/DEMO_SCRIPT.md, docs/SUBMISSION.md and the ledger section "Built during the event" are complete.
5. The secret scan is clean and the repository is public.
6. Total Meta spend is reported and is at or below $120.
</acceptance_criteria>

<self_check>
- Pick three numbers at random from the README and trace each back to its parquet row.
- Would a judge be able to reproduce the Quickstart from a fresh clone, using only the documented steps?
- Does the demo script show anything built before Oct 26? It must not.
</self_check>

<stop_conditions>
1. A merge conflict appears outside pyproject.toml or uv.lock.
2. Anything would publish Ahaan's personal data without his explicit approval.
3. The global budget would exceed $120.
</stop_conditions>

<output_format>
```
## Integration report
Status: READY TO SUBMIT | BLOCKED (reason)
Merges: branch, result, fixes made (count and ledger link)
Headline results: 5 to 8 numbers with their source
Demo: chosen clips and asset paths
Spend: by tag, and the global total
Ahaan next: numbered steps to record the video and submit
```
</output_format>
