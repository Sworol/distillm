from __future__ import annotations

from pathlib import Path


def write_log(tmp_path: Path, name: str, *lines: str) -> Path:
    """Write a synthetic log file under *tmp_path* and return its path."""
    p = tmp_path / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(lines), encoding="utf-8")
    return p


def make_log_with_tail(tmp_path: Path, name: str, body: str, tail: str) -> Path:
    """Write a large-enough log so scan_log_tail_mid hits multiple regions.

    The body is repeated to reach > 512 KB, then *tail* is appended.
    """
    p = tmp_path / name
    p.parent.mkdir(parents=True, exist_ok=True)
    padding = (body + "\n") * 30000  # ~30000 lines should exceed 512 KB
    p.write_text(padding + "\n" + tail, encoding="utf-8")
    return p
