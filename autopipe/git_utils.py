from __future__ import annotations

import subprocess
from pathlib import Path

from autopipe.io_utils import now_ts


def git_diff(repo_root: Path, max_bytes: int = 2 * 1024 * 1024) -> str:
    """Return git diff output, truncated to max_bytes to avoid OOM on large diffs."""
    try:
        out = subprocess.check_output(["git", "diff"], cwd=str(repo_root))
        text = out.decode("utf-8", errors="replace")
        if len(text) > max_bytes:
            text = text[:max_bytes] + "\n... [truncated]\n"
        return text
    except Exception as exc:
        return f"[git_diff_error] {repr(exc)}\n"


def git_status(repo_root: Path) -> str:
    try:
        out = subprocess.check_output(["git", "status", "--porcelain"], cwd=str(repo_root))
        return out.decode("utf-8", errors="replace")
    except Exception as exc:
        return f"[git_status_error] {repr(exc)}\n"


def snapshot_git(repo_root: Path, exp_dir: Path, tag: str) -> None:
    exp_dir.mkdir(parents=True, exist_ok=True)
    stamp = now_ts().replace(":", "").replace(" ", "_")
    (exp_dir / f"git_status_{tag}_{stamp}.txt").write_text(git_status(repo_root), encoding="utf-8")
    (exp_dir / f"git_diff_{tag}_{stamp}.patch").write_text(git_diff(repo_root), encoding="utf-8")

