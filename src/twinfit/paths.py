"""Filesystem locations shared by every worktree.

The data dir and the cost ledger are deliberately shared by all worktrees so that
synthetic fixtures are generated once and spend is counted once.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

DATA_SUBDIRS = (
    "synth",
    "real",
    "real/_calibration",
    "real/_template",
    "cache/meta",
    "printables",
)


def data_dir(ensure: bool = True) -> Path:
    root = Path(os.environ.get("TWINFIT_DATA_DIR", "~/twinfit-data")).expanduser()
    if ensure:
        for sub in DATA_SUBDIRS:
            (root / sub).mkdir(parents=True, exist_ok=True)
    return root


def repo_root(start: Path | None = None) -> Path:
    """Git top level of the current worktree, or the current directory outside git."""
    cwd = Path(start or Path.cwd())
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True,
        )
        return Path(out.stdout.strip())
    except (subprocess.CalledProcessError, FileNotFoundError):
        return cwd


def runs_dir() -> Path:
    return repo_root() / "runs"


def ledger_path() -> Path:
    return data_dir() / "cost_ledger.jsonl"
