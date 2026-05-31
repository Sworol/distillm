from __future__ import annotations

import subprocess
from pathlib import Path

from autopipe.io_utils import now_ts


def git_diff(repo_root: Path, max_bytes: int = 2 * 1024 * 1024) -> str:
    """Return ``git diff HEAD`` output, truncated to max_bytes to avoid OOM on large diffs.

    Uses ``git diff HEAD`` (not bare ``git diff``) to capture both staged and
    unstaged changes.  A bare ``git diff`` shows only unstaged modifications and
    would miss staged-but-uncommitted agent edits.

    Truncation happens in two stages:
    1. **Byte-level truncation**: ``git`` output exceeding *max_bytes* is
       truncated BEFORE decoding to prevent OOM from huge binary diffs.
    2. **Char-level truncation**: after UTF-8 decoding, the text is capped at
       *max_bytes* characters (handles replacement-char inflation from binary
       data).
    """
    try:
        proc = subprocess.Popen(
            ["git", "diff", "HEAD"],
            cwd=str(repo_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        out, _ = proc.communicate(timeout=30)
        if proc.returncode != 0:
            return f"[git_diff_error] git diff exit code={proc.returncode}\n"
        # Stage 1: byte-level truncation before decode (OOM protection for
        # binary-heavy diffs).
        if len(out) > max_bytes:
            out = out[:max_bytes]
        text = out.decode("utf-8", errors="replace")
        # Stage 2: char-level truncation after decode (handles replacement-char
        # inflation from null bytes in binary data).
        if len(text) > max_bytes:
            text = text[:max_bytes] + "\n... [truncated]\n"
        return text
    except subprocess.TimeoutExpired:
        return "[git_diff_error] git diff timed out after 30s\n"
    except FileNotFoundError:
        return "[git_diff_error] git not found on PATH\n"
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
    # Check if repo_root is actually a git repository before running git
    # commands.  Without this check, git_status/git_diff would still work
    # (they catch exceptions), but the error messages would be confusing.
    git_dir = repo_root / ".git"
    is_git = git_dir.is_dir() or git_dir.is_file()  # .git can be a file for worktrees
    status_text = git_status(repo_root) if is_git else "[snapshot_git] not a git repository\n"
    diff_text = git_diff(repo_root) if is_git else "[snapshot_git] not a git repository\n"
    (exp_dir / f"git_status_{tag}_{stamp}.txt").write_text(status_text, encoding="utf-8")
    (exp_dir / f"git_diff_{tag}_{stamp}.patch").write_text(diff_text, encoding="utf-8")

