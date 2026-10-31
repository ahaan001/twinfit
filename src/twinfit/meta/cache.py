"""Content-addressed response cache in data_dir()/cache/meta/.

The key is sha256 over the model id and the canonical JSON of the request. Inline file
content (bytes, data: URIs and long strings) is replaced by its sha256 before hashing so
keys stay short and identical requests with identical files hit the same entry.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from twinfit import paths

LONG_STRING_CHARS = 1024


def cache_dir() -> Path:
    return paths.data_dir() / "cache" / "meta"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _is_file_content(value: str) -> bool:
    return value.startswith("data:") or len(value) > LONG_STRING_CHARS


def canonicalize(value: Any) -> Any:
    """Return a JSON-serializable copy with file content replaced by `sha256:<hex>`."""
    if isinstance(value, dict):
        return {str(k): canonicalize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [canonicalize(v) for v in value]
    if isinstance(value, (bytes, bytearray)):
        return f"sha256:{_sha256(bytes(value))}"
    if isinstance(value, str) and _is_file_content(value):
        return f"sha256:{_sha256(value.encode('utf-8'))}"
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if hasattr(value, "model_dump"):
        return canonicalize(value.model_dump(mode="json"))
    return repr(value)


def canonical_json(value: Any) -> str:
    return json.dumps(canonicalize(value), sort_keys=True, separators=(",", ":"))


def key(model: str, request: dict[str, Any]) -> str:
    payload = model + "\n" + canonical_json(request)
    return _sha256(payload.encode("utf-8"))


def path_for(key: str) -> Path:
    return cache_dir() / f"{key}.json"


def get(key: str) -> dict[str, Any] | None:
    path = path_for(key)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def put(key: str, response_json: dict[str, Any]) -> Path:
    """Write atomically so a concurrent reader never sees a partial file."""
    path = path_for(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(response_json, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)
    return path
