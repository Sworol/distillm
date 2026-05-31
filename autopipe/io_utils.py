from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict


def atomic_write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")
    os.replace(tmp, path)


def patch_exp(path: Path, base: Dict[str, Any] | None = None, **updates: Any) -> Dict[str, Any]:
    """Atomically patch a JSON file, updating only *updates* keys.

    Used by both the scheduler and stale-worker recovery to safely update
    ``exp.json`` without clobbering fields written by the worker or agent.
    """
    if path.exists():
        exp = read_json(path)
    else:
        exp = dict(base or {})
    for k, v in updates.items():
        exp[k] = v
    atomic_write_json(path, exp)
    return exp


def read_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def now_ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


class Lock:
    def __init__(self, path: Path, stale_seconds: int = 24 * 3600):
        self.path = path
        self.stale_seconds = stale_seconds
        self._pid: int | None = None

    @staticmethod
    def _read_lock_pid(lock_path: Path) -> int | None:
        try:
            txt = lock_path.read_text(encoding="utf-8", errors="replace")
        except FileNotFoundError:
            return None
        for line in txt.splitlines():
            line = line.strip()
            if line.startswith("pid="):
                raw = line.split("=", 1)[1].strip()
                try:
                    return int(raw)
                except ValueError:
                    return None
        return None

    @staticmethod
    def _pid_alive(pid: int) -> bool:
        try:
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True

    def acquire(self) -> bool:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        for _ in range(2):
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    f.write(f"pid={os.getpid()}\n")
                    f.write(f"ts={now_ts()}\n")
                self._pid = os.getpid()
                return True
            except FileExistsError:
                # Best-effort stale cleanup.
                # Before unlinking, verify that the lock holder PID is dead.
                # This closes the TOCTOU window between stat() and unlink():
                # if a new process acquired the lock in that window, its PID
                # will be alive, and we leave it alone.
                try:
                    st = self.path.stat()
                    if time.time() - st.st_mtime > self.stale_seconds:
                        lock_pid = self._read_lock_pid(self.path)
                        if lock_pid is not None and not self._pid_alive(lock_pid):
                            self.path.unlink(missing_ok=True)
                            continue
                except FileNotFoundError:
                    pass
                return False
        return False

    def owned(self) -> bool:
        """Return True if the lock file exists and contains our PID."""
        if self._pid is None:
            return False
        try:
            txt = self.path.read_text(encoding="utf-8", errors="replace")
        except FileNotFoundError:
            return False
        for line in txt.splitlines():
            if line.strip().startswith(f"pid={self._pid}"):
                return True
        return False

    def heartbeat(self) -> None:
        """Update the lock file timestamp so stale detection works correctly.

        Uses O_RDWR (not O_TRUNC) to avoid a TOCTOU race: if another process
        stole the lock between the owned() check and the write, O_RDWR opens
        whatever inode the path currently points to. We read the PID from the
        opened fd, and only rewrite if we still own it.

        After writing, we re-verify ownership via self.path (which may point
        to a different inode if the lock was stolen). If the post-write check
        fails, we mark eviction by clearing self._pid.
        """
        if self._pid is None:
            return
        try:
            fd = os.open(self.path, os.O_RDWR)
            try:
                data = os.read(fd, 4096).decode("utf-8", errors="replace")
                owned = False
                for line in data.splitlines():
                    if line.strip().startswith(f"pid={self._pid}"):
                        owned = True
                        break
                if not owned:
                    self._pid = None
                    return
                os.ftruncate(fd, 0)
                os.lseek(fd, 0, os.SEEK_SET)
                os.write(fd, f"pid={self._pid}\n".encode())
                os.write(fd, f"ts={now_ts()}\n".encode())
                os.fsync(fd)
            finally:
                os.close(fd)
            # Post-write verification: if the lock was stolen (new inode),
            # our write went to the orphaned inode and self.owned() will
            # return False when reading the current path.
            if not self.owned():
                self._pid = None
        except Exception:
            import sys
            print(f"[Lock.heartbeat] warning: failed to update lock file {self.path}", file=sys.stderr)
            pass

    def release(self) -> None:
        if not self.owned():
            return
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass


def scan_log_chunk(path: Path, start_pos: int, chunk_size: int) -> str:
    """Read a chunk of the log file at a specific byte offset."""
    try:
        with open(path, "rb") as f:
            f.seek(start_pos, os.SEEK_SET)
            return f.read(chunk_size).decode("utf-8", errors="replace")
    except (FileNotFoundError, OSError):
        return ""


def scan_log_tail_mid(log_path: Path) -> list[str]:
    """Return up to three text chunks from *log_path* for error scanning.

    Coverage strategy — three overlapping windows so errors are found
    regardless of where they land in large logs:

    * tail: last 256 KB (always included — torchrun ChildFailedError lands here)
    * mid:  256 KB around the 65 % mark (for large logs > 512 KB)
    * low:  256 KB around the 25 % mark (for large logs > 512 KB)

    Used by both ``classify_failure`` and ``_last_error_hash`` so the
    scanning strategy stays consistent.
    """
    try:
        file_size = log_path.stat().st_size
    except OSError:
        return []
    if file_size == 0:
        return []
    chunks: list[str] = []
    tail_size = min(256 * 1024, file_size)
    chunks.append(scan_log_chunk(log_path, file_size - tail_size, tail_size))
    if file_size > 512 * 1024:
        mid_start = int(file_size * 0.65)
        mid_size = min(256 * 1024, file_size - mid_start)
        chunks.append(scan_log_chunk(log_path, mid_start, mid_size))
        low_start = int(file_size * 0.25)
        low_size = min(256 * 1024, file_size - low_start)
        chunks.append(scan_log_chunk(log_path, low_start, low_size))
    return chunks


def get_classification_text(log_path: Path) -> str:
    """Return noise-filtered, lowercased text from *log_path* for error analysis.

    Combines ``scan_log_tail_mid`` + ``_filter_noise`` into a single call so
    ``classify_failure`` and ``_last_error_hash`` share the same cleaned view
    of the log.
    """
    chunks = scan_log_tail_mid(log_path)
    return "\n".join(_filter_noise(c).lower() for c in chunks)


def _proximity_match(text: str, a: str, b: str, max_distance: int = 500) -> bool:
    """Return True if strings a and b both appear in text within max_distance chars.

    Uses re.finditer to check ALL occurrences, not just the first.  When the
    text is built from concatenated log chunks (tail + mid), keyword "a" may
    appear first in mid while keyword "b" appears first in tail, producing a
    spurious large distance and a false negative.  Checking every pair of
    positions guarantees we find the true proximity match (e.g. the single-line
    kernel OOM message ``Out of memory: Killed process 12345``).
    """
    pos_a = [m.start() for m in re.finditer(re.escape(a), text)]
    if not pos_a:
        return False
    pos_b = [m.start() for m in re.finditer(re.escape(b), text)]
    if not pos_b:
        return False
    for pa in pos_a:
        for pb in pos_b:
            if abs(pb - pa) <= max_distance:
                return True
    return False


def _filter_noise(text: str) -> str:
    """Remove WARNING/INFO lines that can cause false positives in keyword matching.

    Matches both plain "WARNING: ..." and torchrun-style "[rank0] WARNING: ..."
    by looking for warning/info as the first meaningful word after any optional
    bracket-prefixed rank tag.

    Lines are NOT dropped if they also contain error-level keywords (e.g. a
    WARNING-level CUDA OOM message), preventing false negatives.
    """
    lines = []
    for line in text.splitlines():
        lower = line.strip().lower()
        if re.search(r'(?:\[.*?\]\s*:?\s*)?(warning|info)\b', lower):
            if not re.search(r'\b(error|exception|traceback|fatal|critical|oom|out of memory|nan)\b', lower):
                continue
        lines.append(line)
    return "\n".join(lines)


def classify_failure(log_path: Path) -> str:
    """
    Classify training failure by scanning the log for known error patterns.

    Strategy: DDP training logs have a two-part structure:
    1. Actual Python traceback (OOM, FileNotFound, etc.) — in the middle of the log
    2. torchrun ChildFailedError wrapper — always at the very end

    We scan three overlapping regions so errors are found regardless of
    where they land in large files:
    - Tail (last 256 KB): covers torchrun wrapper + nearby traceback
    - Mid section (256 KB around the 65% mark): covers early traceback
    - Low section (256 KB around the 25% mark): covers errors in the
      lower portion of large logs
    """
    combined = get_classification_text(log_path)
    if not combined:
        return "other"

    # Check loss_scale BEFORE oom — gradient overflow is often the root cause
    # of OOM-like symptoms, and OOM batch reduction won't fix it.
    if "loss scale" in combined and ("minimum" in combined or "cannot decrease" in combined):
        return "loss_scale"
    if "cuda out of memory" in combined or ("out of memory" in combined and "cuda" in combined):
        return "oom"
    # Detect system OOM-killer: kernel logs "Out of memory: Killed process" to dmesg/stderr.
    # We require "out of memory" and "killed" close together (within 500 chars) to avoid
    # false positives from unrelated occurrences scattered across different log regions.
    if _proximity_match(combined, "killed", "out of memory", max_distance=500):
        return "oom"
    # Check NaN BEFORE assertion — NaN loss/weight is often the root cause,
    # and subsequent AssertionErrors are just symptoms. Classifying as "nan"
    # directs the agent to inspect gradients and learning rates first.
    if (re.search(r"\bnan\b", combined) and
            ("loss" in combined or "gradient" in combined or "tensor" in combined)):
        return "nan"
    if ("no space left on device" in combined or "disk full" in combined
            or "file write failed" in combined or "inline_container" in combined):
        return "disk_full"
    if "huggingface.co" in combined and ("timeout" in combined or "timed out" in combined or "rate limit" in combined):
        return "hf"
    if "temporary failure in name resolution" in combined or "connection reset by peer" in combined:
        return "net"
    if "module not found" in combined or "modulenotfounderror" in combined or "importerror" in combined:
        return "import"
    if "address already in use" in combined or "eaddrinuse" in combined:
        return "port"
    if "nccl" in combined and ("error" in combined or "unhandled" in combined or "abort" in combined or "timeout" in combined):
        return "nccl"
    if "filenotfounderror" in combined or "no such file or directory" in combined:
        return "path"
    if "jsondecodeerror" in combined or ("keyerror" in combined and ("json" in combined or "data" in combined)):
        return "data"
    if "checkpoint" in combined and ("missing" in combined or "unexpected" in combined or "size mismatch" in combined):
        return "ckpt"
    if ("size mismatch" in combined or "expected size" in combined or "mat1 and mat2" in combined
            or "invalid shape" in combined):
        return "shape"
    if "assertionerror" in combined or ("assertion" in combined and ("error" in combined or "failed" in combined)):
        return "assert"
    if "sigterm" in combined or "keyboardinterrupt" in combined or "process killed" in combined:
        return "killed"

    return "other"

