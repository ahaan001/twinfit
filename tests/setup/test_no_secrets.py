"""Tracked files must not contain Meta API keys or key assignments."""

import re
import subprocess
from pathlib import Path

from twinfit.paths import repo_root

# A Meta key looks like LLM_<digits>_<secret>. The second pattern catches an assignment with a
# key-like value; a bare "MODEL_API_KEY=" placeholder in documentation is allowed.
PATTERNS = [re.compile(r"LLM_[0-9]+_"), re.compile(r"MODEL_API_KEY=[ \t]*[A-Za-z0-9_\-]{6,}")]


def test_tracked_files_have_no_secrets():
    root = repo_root()
    out = subprocess.run(["git", "ls-files", "-z"], cwd=root, capture_output=True, check=True)
    files = [f for f in out.stdout.decode().split("\0") if f]
    assert files, "no tracked files; run inside the repository"
    hits = []
    for rel in files:
        path = Path(root) / rel
        if not path.is_file():
            continue
        try:
            text = path.read_text(errors="ignore")
        except OSError:
            continue
        for pattern in PATTERNS:
            for m in pattern.finditer(text):
                hits.append((rel, m.group(0)[:20]))
    assert not hits, hits


def test_private_folder_is_not_tracked():
    root = repo_root()
    out = subprocess.run(
        ["git", "ls-files", ".private"], cwd=root, capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == ""
