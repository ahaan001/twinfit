"""Per-worktree session settings and secret loading.

`<repo_root>/.twinfit-session.yaml` holds `tag`, `tag_cap_usd` and `allow_spend`.
The environment variables TWINFIT_TAG and TWINFIT_ALLOW_SPEND override the file.
Secrets come from ~/.config/twinfit/.env, then a repo-root .env. Values already present
in the process environment win over both files. Secret values are never printed or logged.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml
from dotenv import load_dotenv

from twinfit.paths import repo_root

SESSION_FILE = ".twinfit-session.yaml"
SECRETS_FILE = Path("~/.config/twinfit/.env").expanduser()
DEFAULT_BUDGET_USD = 120.0


@dataclass(frozen=True)
class Session:
    tag: str = "untagged"
    tag_cap_usd: float | None = None
    allow_spend: bool = False


def _truthy(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


def load_session(root: Path | None = None) -> Session:
    root = root or repo_root()
    path = root / SESSION_FILE
    data: dict = {}
    if path.exists():
        loaded = yaml.safe_load(path.read_text()) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"{path} must contain a mapping")
        data = loaded
    tag = str(data.get("tag", "untagged"))
    cap = data.get("tag_cap_usd")
    tag_cap_usd = None if cap is None else float(cap)
    allow_spend = bool(data.get("allow_spend", False))
    if "TWINFIT_TAG" in os.environ:
        tag = os.environ["TWINFIT_TAG"]
    if "TWINFIT_ALLOW_SPEND" in os.environ:
        allow_spend = _truthy(os.environ["TWINFIT_ALLOW_SPEND"])
    return Session(tag=tag, tag_cap_usd=tag_cap_usd, allow_spend=allow_spend)


def load_secrets(root: Path | None = None) -> None:
    load_dotenv(SECRETS_FILE, override=False)
    repo_env = (root or repo_root()) / ".env"
    if repo_env.exists():
        load_dotenv(repo_env, override=False)


def model_api_key() -> str | None:
    load_secrets()
    key = os.environ.get("MODEL_API_KEY", "").strip()
    return key or None


def budget_usd() -> float:
    load_secrets()
    raw = os.environ.get("TWINFIT_BUDGET_USD", "").strip()
    return float(raw) if raw else DEFAULT_BUDGET_USD
