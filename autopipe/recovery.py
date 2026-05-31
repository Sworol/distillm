from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict

from autopipe.io_utils import Lock, atomic_write_json, log_event, now_ts, read_json, patch_exp


def recover_stale_worker(
    run_exp_path: Path,
    queue_exp: Dict[str, Any],
    lock_path: Path,
    status_path: Path,
) -> bool:
    """Check worker health and recover if stale.

    Returns True if the worker was recovered (status updated to failed),
    False if it appears healthy and should be counted as running.
    """
    # Pre-read current bookkeeping fields so stale recovery can increment them
    # for proper exponential backoff in the scheduler retry loop.
    curr_consecutive = 0
    curr_attempt = 0
    if run_exp_path.exists():
        try:
            curr = read_json(run_exp_path)
            curr_consecutive = int(curr.get("consecutive_failures", 0))
            curr_attempt = int(curr.get("attempt", 0))
        except Exception:
            pass
    next_consecutive = curr_consecutive + 1
    last_failed_ts = str(time.time())

    # ---- Pre-check: clean orphaned lock BEFORE any case analysis. -----------
    # If another process acquired this lock, its PID will be alive and we will
    # leave it alone.  Cleaning dead-PID locks here makes Case 2 (below)
    # self-contained — it does not depend on the caller having already done
    # this cleanup in a prior loop iteration.
    if lock_path.exists():
        pid = Lock._read_lock_pid(lock_path)
        if pid is not None and not Lock._pid_alive(pid):
            try:
                lock_path.unlink(missing_ok=True)
            except Exception:
                pass

    # Case 1: status.json exists with a terminal status — sync exp.json + clean lock.
    try:
        st = read_json(status_path)
        st_status = st.get("status")
        if st_status in {"failed", "success", "hard_failure"}:
            patch_kwargs: Dict[str, Any] = dict(
                status=st_status,
                attempt=max(curr_attempt, int(st.get("attempt", 0))),
                updated_at=now_ts(),
                last_exit_code=st.get("exit_code"),
                last_reason=st.get("reason"),
            )
            if st_status == "success":
                patch_kwargs["consecutive_failures"] = 0
            elif st_status in {"failed", "hard_failure"}:
                patch_kwargs["last_failed_at"] = last_failed_ts
            patch_exp(run_exp_path, base=queue_exp, **patch_kwargs)
            if lock_path.exists():
                pid = Lock._read_lock_pid(lock_path)
                if pid is None or not Lock._pid_alive(pid):
                    try:
                        lock_path.unlink(missing_ok=True)
                    except Exception:
                        pass
            return True
        # Case 2: status.json says "running" but worker lock is missing → stale.
        # The pre-check cleanup (above) already removes dead-PID locks, so a
        # missing lock at this point means the worker truly is gone.
        if st_status == "running" and not lock_path.exists():
            patch_exp(
                run_exp_path,
                base=queue_exp,
                status="failed",
                updated_at=now_ts(),
                last_reason="stale_worker",
                consecutive_failures=next_consecutive,
                last_failed_at=last_failed_ts,
            )
            atomic_write_json(
                status_path,
                {
                    "status": "failed",
                    "updated_at": now_ts(),
                    "attempt": st.get("attempt", 0),
                    "exit_code": 1,
                    "reason": "stale_worker",
                },
            )
            return True
    except FileNotFoundError:
        pass
    except (ValueError, OSError):
        # status.json exists but is malformed (ValueError from json.JSONDecodeError)
        # or unreadable (OSError). Fall through to Case 3 to attempt recovery
        # via staleness detection rather than crashing the scheduler loop.
        pass

    # Case 3: No status.json (or could not be read) and no lock after grace period.
    if not lock_path.exists():
        age = 0.0
        try:
            age = time.time() - status_path.stat().st_mtime
        except FileNotFoundError:
            # status_path doesn't exist; fall back to run_exp.json mtime.
            # If exp was set to "running" >120s ago with no status or lock file,
            # the worker died before creating any artifacts.
            if run_exp_path.exists():
                try:
                    age = time.time() - run_exp_path.stat().st_mtime
                except FileNotFoundError:
                    # TOCTOU race: file was deleted between exists() check and
                    # the stat() call. Treat as if the file never existed.
                    pass
        # age == 0 means neither file exists — do NOT recover (the experiment
        # may not have started yet, or the directory was just created).
        # 300s grace to allow large-model init (compilation + data loading can
        # easily take 2+ minutes before any log output or status update).
        if age > 300 and age > 0:
            stale_attempt = max(curr_attempt, 1)  # preserve attempt counter, floor at 1
            patch_exp(
                run_exp_path,
                base=queue_exp,
                status="failed",
                updated_at=now_ts(),
                last_reason="stale_worker",
                consecutive_failures=next_consecutive,
                last_failed_at=last_failed_ts,
            )
            atomic_write_json(
                status_path,
                {
                    "status": "failed",
                    "updated_at": now_ts(),
                    "attempt": stale_attempt,
                    "exit_code": 1,
                    "reason": "stale_worker",
                },
            )
            return True

    # Case 4: Lock exists but recorded PID is dead → clean lock + mark stale.
    # NOTE: When called from the main loop this case is defensive dead code —
    # Phase 1 orphan-lock cleanup (lines 310-317) already removes locks whose
    # recorded PID is dead before recover_stale_worker is reached.  Case 4
    # remains here so the function is independently safe if called from other
    # contexts or if the Phase-1 cleanup is ever moved.
    pid = Lock._read_lock_pid(lock_path)
    if pid is not None and not Lock._pid_alive(pid):
        try:
            lock_path.unlink(missing_ok=True)
        except Exception:
            pass
        stale_attempt = max(curr_attempt, 1)  # preserve attempt counter, floor at 1
        patch_exp(
            run_exp_path,
            base=queue_exp,
            status="failed",
            updated_at=now_ts(),
            last_reason="stale_worker",
            consecutive_failures=next_consecutive,
            last_failed_at=last_failed_ts,
        )
        atomic_write_json(
            status_path,
            {
                "status": "failed",
                "updated_at": now_ts(),
                "attempt": stale_attempt,
                "exit_code": 1,
                "reason": "stale_worker",
            },
        )
        return True

    # Worker appears healthy.
    return False
