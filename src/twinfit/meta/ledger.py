"""Append-only JSONL cost ledger shared by every worktree.

One record per API call (or cache hit), appended under a file lock so parallel processes
never interleave lines.
"""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import UTC, datetime
from typing import Any

from filelock import FileLock

from twinfit import paths, session
from twinfit.meta.config import usage_dict

KINDS = ("responses", "sam_video", "sam_image", "files", "other")


def _lock(path):
    return FileLock(str(path) + ".lock")


def append(record: dict[str, Any]) -> None:
    path = paths.ledger_path()
    line = json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
    with _lock(path), open(path, "a", encoding="utf-8") as f:
        f.write(line)
        f.flush()


def append_record(
    *,
    model: str,
    kind: str,
    usage: Any,
    cost_usd: float,
    cache_hit: bool,
    request_id: str | None,
    tag: str | None = None,
) -> dict[str, Any]:
    """Build a record with timestamp, tag and worktree, append it and return it."""
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
    record = {
        "timestamp": datetime.now(UTC).isoformat(),
        "tag": tag or session.load_session().tag,
        "worktree": str(paths.repo_root()),
        "model": model,
        "kind": kind,
        "usage": usage_dict(usage),
        "cost_usd": float(cost_usd),
        "cache_hit": bool(cache_hit),
        "request_id": request_id,
    }
    append(record)
    return record


def read_records() -> list[dict[str, Any]]:
    path = paths.ledger_path()
    if not path.exists():
        return []
    records = []
    with _lock(path), open(path, encoding="utf-8") as f:
        for number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{number}: invalid ledger line") from exc
    return records


def total(tag: str | None = None) -> float:
    records = read_records()
    if tag is not None:
        records = [r for r in records if r.get("tag") == tag]
    return sum(float(r.get("cost_usd", 0.0)) for r in records)


def summary() -> dict[str, Any]:
    """Totals overall, by tag, by model and by UTC day, for the `twinfit cost` command."""
    records = read_records()
    by_tag: dict[str, float] = defaultdict(float)
    by_model: dict[str, float] = defaultdict(float)
    by_day: dict[str, float] = defaultdict(float)
    cache_hits = 0
    for r in records:
        cost = float(r.get("cost_usd", 0.0))
        by_tag[str(r.get("tag"))] += cost
        by_model[str(r.get("model"))] += cost
        by_day[str(r.get("timestamp", ""))[:10]] += cost
        cache_hits += bool(r.get("cache_hit"))
    return {
        "total_usd": sum(by_tag.values()),
        "n_records": len(records),
        "cache_hits": cache_hits,
        "by_tag": dict(sorted(by_tag.items())),
        "by_model": dict(sorted(by_model.items())),
        "by_day": dict(sorted(by_day.items())),
    }
