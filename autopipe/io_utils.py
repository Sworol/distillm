from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional


def atomic_write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=True, indent=2)
        f.write("\n")
    os.replace(tmp, path)


def read_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def now_ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


class Lock:
    def __init__(self, path: Path, stale_seconds: int = 24 * 3600):
        self.path = path
        self.stale_seconds = stale_seconds

    def acquire(self) -> bool:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(f"pid={os.getpid()}\n")
                f.write(f"ts={now_ts()}\n")
            return True
        except FileExistsError:
            # Best-effort stale cleanup
            try:
                st = self.path.stat()
                if time.time() - st.st_mtime > self.stale_seconds:
                    self.path.unlink(missing_ok=True)
            except FileNotFoundError:
                pass
            return False

    def release(self) -> None:
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass


def tail_text(path: Path, max_bytes: int = 64 * 1024) -> str:
    try:
        with open(path, "rb") as f:
            try:
                f.seek(0, os.SEEK_END)
                size = f.tell()
                f.seek(max(0, size - max_bytes), os.SEEK_SET)
            except OSError:
                pass
            data = f.read()
        return data.decode("utf-8", errors="replace")
    except FileNotFoundError:
        return ""


def _scan_log_chunk(path: Path, start_pos: int, chunk_size: int) -> str:
    """Read a chunk of the log file at a specific byte offset."""
    try:
        with open(path, "rb") as f:
            f.seek(start_pos, os.SEEK_SET)
            return f.read(chunk_size).decode("utf-8", errors="replace")
    except (FileNotFoundError, OSError):
        return ""


def classify_failure(log_path: Path) -> str:
    """
    Classify training failure by scanning the log for known error patterns.

    Strategy: DDP training logs have a two-part structure:
    1. Actual Python traceback (OOM, FileNotFound, etc.) — in the middle of the log
    2. torchrun ChildFailedError wrapper — always at the very end

    We scan three regions to catch errors regardless of log size:
    - Tail (last 256 KB): covers torchrun wrapper + nearby traceback
    - Mid section (256 KB around the 70% mark): covers early traceback in long logs
    - Full-file keyword scan: catches OOM scattered across DDP ranks
    """
    try:
        with open(log_path, "rb") as f:
            f.seek(0, os.SEEK_END)
            file_size = f.tell()
    except (FileNotFoundError, OSError):
        return "other"

    if file_size == 0:
        return "other"

    tail_size = min(256 * 1024, file_size)
    tail = _scan_log_chunk(log_path, file_size - tail_size, tail_size).lower()

    mid_text = ""
    if file_size > 512 * 1024:
        mid_start = int(file_size * 0.65)
        mid_size = min(256 * 1024, file_size - mid_start)
        mid_text = _scan_log_chunk(log_path, mid_start, mid_size).lower()

    combined = tail + "\n" + mid_text

    if "cuda out of memory" in combined or ("out of memory" in combined and "cuda" in combined):
        return "oom"
    if "huggingface.co" in combined and ("timeout" in combined or "timed out" in combined or "rate limit" in combined):
        return "hf"
    if "temporary failure in name resolution" in combined or "connection reset by peer" in combined:
        return "net"
    if "module not found" in combined or "modulenotfounderror" in combined or "importerror" in combined:
        return "import"
    if "address already in use" in combined or "eaddrinuse" in combined:
        return "port"
    if "nccl" in combined and ("error" in combined or "unhandled" in combined or "abort" in combined):
        return "nccl"
    if "filenotfounderror" in combined or "no such file or directory" in combined:
        return "path"

    return "other"


def parse_int_list(csv: str) -> list[int]:
    out: list[int] = []
    for raw in csv.split(","):
        raw = raw.strip()
        if raw:
            out.append(int(raw))
    return out
