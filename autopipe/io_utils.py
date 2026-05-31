from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict


def log_event(**kwargs: Any) -> None:
    """Write a structured JSON-line log event to stderr (auto-flush).

    All autopipe diagnostic output goes through this function so logs are
    machine-parseable.  A ``ts`` field is auto-added if the caller does not
    supply one.
    """
    kwargs.setdefault("ts", time.strftime("%Y-%m-%d %H:%M:%S"))
    try:
        print(json.dumps(kwargs, ensure_ascii=False), file=sys.stderr, flush=True)
    except (BrokenPipeError, OSError):
        # stderr may be closed or broken (e.g. parent process died).
        # There is nothing we can do — swallow silently.
        pass


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
                # Before unlinking, verify that the lock holder PID is dead
                # AND that the inode hasn't changed between our stat+read and
                # the unlink (closes the TOCTOU where a new process could
                # acquire the lock in that window).
                try:
                    st = self.path.stat()
                    if time.time() - st.st_mtime > self.stale_seconds:
                        inode_before = st.st_ino
                        lock_pid = self._read_lock_pid(self.path)
                        if lock_pid is not None and not self._pid_alive(lock_pid):
                            # Re-check inode: if it changed, a new process
                            # acquired the lock and we must not touch it.
                            if self.path.stat().st_ino == inode_before:
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

        Writes to a temporary file then atomically renames over the lock path.
        This avoids the empty-file window that ftruncate+write creates on crash:
        if the process dies between ftruncate and the first write, the lock
        file exists but is empty, making stale detection fall through to
        mtime-based checks.

        We verify ownership by reading the current lock's PID before writing
        the replacement.
        """
        if self._pid is None:
            return
        try:
            # Verify we still own the lock before writing.
            lock_pid = self._read_lock_pid(self.path)
            if lock_pid != self._pid:
                self._pid = None
                return
            # Write to temp file, then atomically rename to avoid empty-window.
            tmp = self.path.with_suffix(self.path.suffix + f".hb.{os.getpid()}")
            with open(tmp, "w", encoding="utf-8") as f:
                f.write(f"pid={self._pid}\n")
                f.write(f"ts={now_ts()}\n")
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, self.path)
            # Post-rename verification: ensure we still own it.
            if not self.owned():
                self._pid = None
        except Exception:
            log_event(source="lock", event="heartbeat_failed", path=str(self.path))
            pass

    def release(self) -> None:
        if not self.owned():
            return
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass
        self._pid = None


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
        if re.search(r'(?:\[[^\]]*\]\s*:?\s*)*\b(warning|info)\b', lower):
            if not re.search(r'\b(error|exception|traceback|fatal|critical|oom|out of memory|nan|sigterm|signal|killed|timeout|disk|connection|assert)\b', lower):
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
    if "cuda out of memory" in combined or _proximity_match(combined, "out of memory", "cuda", max_distance=500):
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
    if "sigterm" in combined or "keyboardinterrupt" in combined:
        return "killed"
    # "process killed" is common in kernel OOM logs ("Out of memory: Killed
    # process 12345").  Only classify as "killed" if it is NOT in the
    # proximity of an OOM message (which would already be caught above).
    if "process killed" in combined and not _proximity_match(combined, "killed", "out of memory", max_distance=500):
        return "killed"

    return "other"

