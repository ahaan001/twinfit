"""Command line entry point: `twinfit`.

Milestone sub-apps are mounted by importing `app` from their cli modules; a placeholder group
is registered when a module is absent, so milestones never need to edit this file.
"""

from __future__ import annotations

import importlib
import os
import platform
import shutil
import sys
from pathlib import Path
from typing import Annotated

import typer
import yaml
from rich.console import Console
from rich.table import Table

from twinfit import paths, session
from twinfit.contracts import RunOptions
from twinfit.contracts.stage import STAGES, StageName

app = typer.Typer(no_args_is_help=True, help="Twinfit: phone video to a verified simulation twin.")
console = Console()

MILESTONE_APPS = {
    "perception": ("twinfit.perception.cli", 1),
    "identify": ("twinfit.identify.cli", 2),
    "verify": ("twinfit.verify.cli", 3),
    "agent": ("twinfit.agent.cli", 3),
    "bench": ("twinfit.bench.cli", 3),
    "report": ("twinfit.report.cli", 3),
}


def _placeholder(group: str, milestone: int) -> typer.Typer:
    sub = typer.Typer(help=f"{group}: not implemented yet (owned by Milestone {milestone})")

    @sub.callback(invoke_without_command=True)
    def _main(ctx: typer.Context) -> None:
        console.print(f"{group}: not implemented yet (owned by Milestone {milestone})")

    return sub


for _group, (_module, _milestone) in MILESTONE_APPS.items():
    try:
        _sub = importlib.import_module(_module).app
    except ModuleNotFoundError as e:
        if e.name != _module:
            raise
        _sub = _placeholder(_group, _milestone)
    app.add_typer(_sub, name=_group)


@app.command()
def doctor() -> None:
    """Environment, data dir, key presence, session, ledger and an offscreen render check."""
    table = Table(show_header=False, box=None)
    table.add_row("python", f"{platform.python_version()} ({sys.executable})")
    for name in (
        "numpy",
        "scipy",
        "cv2",
        "mujoco",
        "pinocchio",
        "openai",
        "pydantic",
        "imageio_ffmpeg",
    ):
        try:
            mod = importlib.import_module(name)
            version = getattr(mod, "__version__", "")
            table.add_row(f"import {name}", f"OK {version}")
        except Exception as e:  # report and continue
            table.add_row(f"import {name}", f"unavailable ({type(e).__name__}: {e})")
    try:
        import imageio_ffmpeg

        table.add_row("ffmpeg", imageio_ffmpeg.get_ffmpeg_exe())
    except Exception as e:
        table.add_row("ffmpeg", f"unavailable ({e})")
    data = paths.data_dir()
    usage = shutil.disk_usage(data)
    table.add_row("data dir", f"{data} ({usage.free / 1e9:.1f} GB free)")
    table.add_row("MODEL_API_KEY set", "yes" if session.model_api_key() else "no")
    sess = session.load_session()
    table.add_row(
        "session", f"tag {sess.tag}, cap {sess.tag_cap_usd}, allow_spend {sess.allow_spend}"
    )
    try:
        from twinfit.meta import ledger

        total = ledger.total()
        tag_total = ledger.total(sess.tag)
        cap = f"{sess.tag_cap_usd}" if sess.tag_cap_usd is not None else "none"
        table.add_row(
            "ledger",
            f"global ${total:.4f} of ${session.budget_usd():.2f}; tag ${tag_total:.4f} of {cap}",
        )
    except Exception as e:
        table.add_row("ledger", f"unavailable ({e})")
    table.add_row("render", _render_check())
    console.print(table)


def _render_check() -> str:
    try:
        import mujoco
        import numpy as np

        model = mujoco.MjModel.from_xml_string(
            "<mujoco><visual><global offwidth='64' offheight='64'/></visual><worldbody>"
            "<light pos='0 0 1'/><geom type='box' size='0.1 0.1 0.1' rgba='1 0 0 1'/>"
            "<camera name='c' pos='0 0 1'/></worldbody></mujoco>"
        )
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        renderer = mujoco.Renderer(model, 64, 64)
        renderer.update_scene(data, camera="c")
        img = renderer.render()
        renderer.close()
        return f"OK offscreen {img.shape}, mean {float(np.mean(img)):.1f}"
    except Exception as e:
        return f"FAILED ({type(e).__name__}: {e})"


@app.command()
def synth(
    suite: Annotated[str | None, typer.Option(help="synth_tiny or synth_small")] = None,
    scene: Annotated[str | None, typer.Option(help="scene name")] = None,
    seed: Annotated[int, typer.Option()] = 0,
    fps: Annotated[float, typer.Option()] = 30.0,
    duration: Annotated[float, typer.Option()] = 2.0,
    width: Annotated[int, typer.Option()] = 320,
    height: Annotated[int, typer.Option()] = 180,
    out: Annotated[Path | None, typer.Option(help="output folder")] = None,
) -> None:
    """Generate synthetic ground-truth clips."""
    from twinfit.synth.generate import SUITES, SuiteSettings, generate_clip, generate_suite

    if suite is not None:
        if suite not in SUITES:
            raise typer.BadParameter(f"suite must be one of {sorted(SUITES)}")
        clips = generate_suite(suite, out)
        for clip in clips:
            console.print(str(clip))
    elif scene is not None:
        settings = SuiteSettings((seed,), width, height, fps, duration)
        target = out or paths.data_dir() / "synth" / "custom" / f"{scene}__s{seed}"
        console.print(str(generate_clip(scene, seed, target, settings)))
    else:
        raise typer.BadParameter("give --suite or --scene")


def _parse_impl_stage(values: list[str]) -> dict[StageName, str]:
    out: dict[StageName, str] = {}
    for item in values:
        if "=" not in item:
            raise typer.BadParameter(f"--impl-stage expects stage=impl, got {item!r}")
        stage, impl = item.split("=", 1)
        if stage not in STAGES:
            raise typer.BadParameter(f"unknown stage {stage!r}; stages are {STAGES}")
        out[stage] = impl  # type: ignore[index]
    return out


@app.command()
def run(
    clip_dir: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
    impl: Annotated[str, typer.Option(help="auto, real or stub for every stage")] = "auto",
    impl_stage: Annotated[
        list[str] | None, typer.Option("--impl-stage", help="stage=impl, repeatable")
    ] = None,
    options: Annotated[Path | None, typer.Option(help="RunOptions YAML file")] = None,
    run_id: Annotated[str | None, typer.Option()] = None,
    stop_after: Annotated[str | None, typer.Option(help="last stage to run")] = None,
) -> None:
    """Run the pipeline on one clip folder."""
    from twinfit.runner import run_pipeline

    impl_map = {s: impl for s in STAGES}
    impl_map.update(_parse_impl_stage(impl_stage or []))
    run_options = RunOptions()
    if options is not None:
        run_options = RunOptions.model_validate(yaml.safe_load(options.read_text()) or {})
    if stop_after is not None and stop_after not in STAGES:
        raise typer.BadParameter(f"unknown stage {stop_after!r}")
    results = run_pipeline(clip_dir, run_id, impl_map, run_options, stop_after)  # type: ignore[arg-type]
    table = Table("stage", "status", "impl", "cost_usd", "duration_s", "messages")
    for r in results:
        table.add_row(
            r.stage,
            r.status,
            r.impl,
            f"{r.cost_usd:.4f}",
            f"{r.duration_s:.2f}",
            "; ".join(r.messages)[:120],
        )
    console.print(table)
    if results:
        console.print(f"run folder: {paths.runs_dir() / (run_id or _last_run_id(results))}")
    if any(r.status == "failed" for r in results):
        raise typer.Exit(code=1)


def _last_run_id(results) -> str:
    runs = sorted(paths.runs_dir().glob("*"), key=lambda p: p.stat().st_mtime)
    return runs[-1].name if runs else ""


@app.command("contracts-check")
def contracts_check(path: Annotated[Path, typer.Argument(exists=True)]) -> None:
    """Validate a clip folder or a run folder, one line per file."""
    from twinfit.contracts.check import check_path

    lines = check_path(path)
    for line in lines:
        mark = "[green]OK [/]" if line.ok else "[red]BAD[/]"
        console.print(f"{mark} {line.file}: {line.message}")
    if not all(line.ok for line in lines):
        raise typer.Exit(code=1)


@app.command("meta-smoke")
def meta_smoke() -> None:
    """List models and make one small muse-spark-1.3 call under the current tag."""
    from twinfit.meta import client

    result = client.smoke()
    for model_id in result["models"]:
        console.print(model_id)
    console.print(f"response: {result['output_text']!r}")
    console.print(f"cost: ${result['cost_usd']:.6f} (tag {session.load_session().tag})")


@app.command()
def printables(out: Annotated[Path | None, typer.Option(help="output folder")] = None) -> None:
    """Write the ChArUco board, part marker and speckle PDFs, and the metadata template."""
    from twinfit.printables import write_metadata_template, write_printables

    for path in write_printables(out):
        console.print(str(path))
    console.print(str(write_metadata_template()))


@app.command()
def cost() -> None:
    """Ledger totals by tag, model and day."""
    from twinfit.meta import ledger

    summary = ledger.summary()
    console.print(
        f"total ${summary['total_usd']:.4f} over {summary['n_records']} records "
        f"({summary['cache_hits']} cache hits); budget ${session.budget_usd():.2f}"
    )
    for title, key in (("tag", "by_tag"), ("model", "by_model"), ("day", "by_day")):
        table = Table(title, "usd")
        for name, value in sorted(summary[key].items()):
            table.add_row(str(name), f"{value:.4f}")
        console.print(table)


def main() -> None:
    os.environ.setdefault("MUJOCO_GL", os.environ.get("MUJOCO_GL", ""))
    app()


if __name__ == "__main__":
    main()
