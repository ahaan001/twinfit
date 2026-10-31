# Twinfit

Twinfit turns phone video of a real mechanism (a pendulum, a double pendulum, a desk lamp, a hinged lid, a slider) into a MuJoCo simulation model plus a certificate stating how far that model can be trusted. Perception (Milestone 1) uses Muse Wonder for part proposals and SAM 3.1 for segmentation, then classical geometry on a ChArUco board for metric tracks. Identification (Milestone 2) fits screw-axis joints, a kinematic tree and physically consistent dynamics. Verification (Milestone 3) issues a physical-consistency certificate, runs a Muse Wonder repair agent, a benchmark harness and an HTML report. Built solo by Ahaan Kothari for the Meta Global AI Developer Hackathon (Oct 26 to Nov 4, 2026). The setup session froze the shared foundation; each milestone builds in its own worktree against it.

## Command cheat sheet

```
uv sync --all-extras --dev                      # environment (Python 3.11, never conda)
uv run twinfit doctor                           # imports, ffmpeg, data dir, key, session, ledger, render
uv run twinfit synth --suite synth_small        # 12 synthetic clips into ~/twinfit-data/synth/synth_small
uv run twinfit run CLIP_DIR --impl stub         # full pipeline with oracle stubs (auto, real or stub)
uv run twinfit run CLIP_DIR --impl-stage perceive=stub --options opts.yaml --stop-after dynamics
uv run twinfit contracts-check PATH             # validate a clip folder or a run folder
uv run twinfit cost                             # ledger totals by tag, model and day
uv run twinfit meta-smoke                       # one tiny paid call under the session tag
uv run twinfit printables                       # PDFs and metadata template into ~/twinfit-data
uv run ruff check . && uv run pytest -q         # lint and tests (live tests need TWINFIT_LIVE=1)
uv run pytest -q -m "not live and not render"   # the fast suite CI runs
```

Python entry points every milestone relies on: `twinfit.synth.suite_dir("synth_small")` for shared fixtures, `twinfit.sim.replay.simulate` and `windowed_replay_rms`, `twinfit.eval.metrics`, `twinfit.contracts` (models, `load_*`, `save_*`, `compute_verdict`), `twinfit.meta.client` (`responses_create`, `raw_client`, `record_cost`, `budget.check`).

## Ownership

| Owner | Paths |
|---|---|
| Setup (frozen) | pyproject.toml (except other owners' optional groups), cli.py, paths.py, session.py, contracts/, geometry/, sim/, synth/, stubs/, meta/, eval/, runner/, printables/, tests/setup/, CLAUDE.md, docs/ARCHITECTURE.md |
| Milestone 1 | src/twinfit/perception/, tests/perception/, docs/m1_perception.md, docs/ledger/m1.md, docs/media/m1_*, optional group `perception` |
| Milestone 2 | src/twinfit/identify/, tests/identify/, docs/m2_identification.md, docs/ledger/m2.md, docs/media/m2_*, optional group `identify` |
| Milestone 3 | src/twinfit/verify/, src/twinfit/agent/, src/twinfit/bench/, src/twinfit/report/, tests/verify/, tests/agent/, tests/bench/, tests/report/, docs/m3_verify_agent.md, docs/ledger/m3.md, docs/media/m3_*, optional groups `verify` and `agent` |
| Integration | README.md, scripts/, docs/ledger/integration.md, docs/DEMO_SCRIPT.md, docs/SUBMISSION.md, merges |

## Rules

- Never edit outside owned paths.
- Contracts are frozen. A needed change goes into `docs/contract_change_requests/<milestone>.md`, with a workaround through `extras` in the meantime.
- Real stages never read `gt/`.
- Every stage writes only inside its own stage folder (`ctx.stage_dir`).
- Meta API calls go only through `twinfit.meta`: never in unit tests, always tagged, cached and budget-checked.
- Secrets are never printed, logged or committed. The key lives in `~/.config/twinfit/.env`.
- Do not set `ANTHROPIC_BASE_URL`, `ANTHROPIC_AUTH_TOKEN` or any other variable from the hackathon notes that would route Claude Code itself through Meta's API.
- Each milestone's CLI module exposes `app` (a typer.Typer); `twinfit.cli` mounts it automatically. Milestones never edit cli.py.
- Each session sets its own `.twinfit-session.yaml` (tag, tag_cap_usd, allow_spend); the file is ignored by git.

## Style

Code: clean and minimal, descriptive names, no print debugging, no emoji, comments only for non-obvious math with a source. Run `uv run ruff format` and `uv run ruff check` before committing (line length 100). Typer parameters use `Annotated[...]`; `B008` is ignored in `cli.py` files.

Documentation: plain, direct, no em dashes, no filler.

Commit message prefixes: `[setup]`, `[m1]`, `[m2]`, `[m3]`, `[int]`.

Each milestone keeps its own ledger file, `docs/ledger/<milestone>.md`: plan before each phase, result after.

## Hackathon rules in brief

Public repository. Everything shown must be built during the event (Oct 26 to Nov 4, 2026) and be identifiable as such. The project must use Meta models (Muse Wonder, Muse Spark, SAM 3.1 via the Meta Model API). No code, data or assets without rights. Judging: Impact 30%, Use of Meta AI tooling 25%, Technical depth 20%, Demo 25%. Submission due Wed Nov 4, 2026, 9:00 PM ET. Meta API credits: $150; Twinfit's hard cap is $120 with tag caps setup $1, m1-dev $45, m2-dev $0, m3-dev $25, integration $40. Full notes: `docs/meta_api_notes.md`.
