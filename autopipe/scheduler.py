from __future__ import annotations

import argparse
import datetime
import functools
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

from autopipe.config import CONFIG_MERGE_KEYS, Paths, default_paths
from autopipe.io_utils import Lock, atomic_write_json, log_event, now_ts, patch_exp, read_json
from autopipe.recovery import recover_stale_worker


def list_queue(queue_dir: Path) -> List[Path]:
    return sorted(queue_dir.glob("*.json"))


def load_exp(path: Path) -> Dict[str, Any]:
    """Load experiment JSON with resilience to corrupted/missing files.

    Returns an empty dict on error so callers can skip entries by checking
    ``exp_id`` rather than crashing the scheduler loop.
    """
    try:
        exp = read_json(path)
    except (FileNotFoundError, OSError, ValueError):
        log_event(source="scheduler", event="load_exp_failed", path=str(path))
        return {}
    exp.setdefault("status", "pending")
    exp.setdefault("attempt", 0)
    return exp


def _load_run_exp(queue_exp: Dict[str, Any], run_exp_path: Path) -> Dict[str, Any]:
    """
    Source-of-truth rules:
    - `autopipe/queue/*.json` defines the initial experiment config.
    - `autopipe/runs/<exp_id>/exp.json` is the mutable working copy (agents may edit it).
    - `autopipe/runs/<exp_id>/status.json` is the authoritative last-attempt outcome.

    Applies CONFIG_MERGE_KEYS (new-keys-only via ``setdefault``) so phase 1
    (stale recovery) sees newly-added queue keys.  Phase 2 applies full
    overrides (existing keys get hotfixed values) in ``_phase2_spawn_workers``.
    """
    if run_exp_path.exists():
        try:
            exp = read_json(run_exp_path)
        except (FileNotFoundError, OSError, ValueError):
            # TOCTOU guard: the file was deleted between exists() and read,
            # or it is corrupted (ValueError from json.JSONDecodeError).
            # Fall back to the queue definition; next iteration will re-read.
            return dict(queue_exp)
        # Guard: if run exp_id is missing or mismatches, fall back to queue
        # to avoid cross-file contamination.
        run_exp_id = exp.get("exp_id")
        queue_exp_id = queue_exp.get("exp_id")
        if not run_exp_id or run_exp_id != queue_exp_id:
            return dict(queue_exp)
        # Apply queue-level config updates (hotfix detection).
        for k in CONFIG_MERGE_KEYS:
            if k in queue_exp:
                exp.setdefault(k, queue_exp[k])
        return exp
    return dict(queue_exp)


def _phase1_stale_recovery(paths: "Paths", q: List[Path]) -> int:
    """Recover stale workers and return count of healthy 'running' experiments.

    For each experiment in *q*:
    - Clean orphaned ``.lock_worker`` files for non-"running" statuses.
    - Delegate ``recover_stale_worker`` for experiments marked "running".
    - Count experiments that appear to have a healthy worker.
    """
    running = 0
    for exp_path in q:
        queue_exp = load_exp(exp_path)
        if not queue_exp.get("exp_id"):
            continue  # skip corrupted entries
        run_root = paths.runs_dir / queue_exp["exp_id"]
        run_exp_path = run_root / "exp.json"
        exp = _load_run_exp(queue_exp, run_exp_path)

        # Clean orphaned worker locks for non-"running" statuses.
        # For "running" statuses _recover_stale_worker handles this
        # internally so there is no ordering dependency between the
        # two steps.
        worker_lock_path = run_root / ".lock_worker"
        if exp.get("status") != "running" and worker_lock_path.exists():
            pid = Lock._read_lock_pid(worker_lock_path)
            # Clean up if: (1) lock has no valid PID (empty/corrupt), or
            # (2) the PID it references is no longer alive.
            if pid is None or not Lock._pid_alive(pid):
                try:
                    worker_lock_path.unlink(missing_ok=True)
                except Exception:
                    pass

        if exp.get("status") != "running":
            continue

        status_path = run_root / "status.json"
        if recover_stale_worker(run_exp_path, queue_exp, worker_lock_path, status_path):
            continue

        running += 1
    return running


def _phase2_spawn_workers(
    paths: "Paths",
    q: List[Path],
    repo_root: Path,
    running: int,
    _workers: Dict[str, subprocess.Popen],
    max_parallel: int,
) -> int:
    """Try to spawn workers for pending/failed/aborted experiments.

    Returns the updated *running* count (may be higher if new workers were
    successfully spawned).  Modifies ``_workers`` in-place.
    """
    for exp_path in q:
        queue_exp = load_exp(exp_path)
        if not queue_exp.get("exp_id"):
            continue  # skip corrupted entries
        run_root = paths.runs_dir / queue_exp["exp_id"]
        run_exp_path = run_root / "exp.json"
        exp = _load_run_exp(queue_exp, run_exp_path)
        status = exp.get("status")

        if status in {"success", "hard_failure"}:
            continue
        if status == "running":
            continue  # Already counted in phase 1.

        # Avoid spawning a duplicate worker when one was already
        # launched in a previous iteration but hasn't updated
        # exp.json to "running" yet (async startup).
        if queue_exp.get("exp_id") in _workers:
            running += 1
            continue

        if running >= max_parallel:
            break

        # ---- Retry policy ------------------------------------------------
        attempt = int(exp.get("attempt", 0))
        max_retries = int(exp.get("max_retries", 2))

        if status == "failed":
            # Exponential backoff: skip if still in cooldown period.
            retry_sleep_base = int(exp.get("retry_sleep", 60))
            consecutive_failures = int(exp.get("consecutive_failures", 0))
            retry_sleep = min(retry_sleep_base * (2 ** consecutive_failures), 900)
            last_failed = exp.get("last_failed_at", "")
            if last_failed:
                try:
                    elapsed = time.time() - float(last_failed)
                    if elapsed < retry_sleep:
                        continue  # cooling down
                except Exception:
                    pass

            if attempt > max_retries:
                patch_exp(run_exp_path, base=queue_exp, status="aborted", updated_at=now_ts())
                continue

        if status == "aborted":
            # Only auto-retry an aborted experiment if the queue config
            # has been modified since the run config was last written
            # (e.g., a hotfix).
            qmtime = exp_path.stat().st_mtime if exp_path.exists() else 0
            rmtime = run_exp_path.stat().st_mtime if run_exp_path.exists() else 0
            if qmtime > rmtime:
                patch_exp(run_exp_path, base=queue_exp,
                          attempt=0, consecutive_failures=0,
                          last_failed_at="",
                          status="failed", updated_at=now_ts())
                status = "failed"
            else:
                continue  # no changes detected, keep aborted

        # ---- Sync per-exp working config --------------------------------
        if not run_exp_path.exists():
            run_root.mkdir(parents=True, exist_ok=True)
            atomic_write_json(run_exp_path, exp)
        else:
            try:
                cur = read_json(run_exp_path)
            except Exception:
                cur = exp
            changed = False
            for k in CONFIG_MERGE_KEYS:
                if k in queue_exp and cur.get(k) != queue_exp.get(k):
                    cur[k] = queue_exp.get(k)
                    changed = True
            if changed:
                atomic_write_json(run_exp_path, cur)

        # ---- Spawn worker -----------------------------------------------
        # NOTE: The worker sets its own status to "running" after acquiring
        # the worker lock. We do NOT set it here — if the worker fails to
        # start (lock busy), exp.json would incorrectly remain "running"
        # and block future retries.
        cmd = [
            sys.executable, "-m", "autopipe.worker",
            "--repo-root", str(repo_root),
            "--exp-json", str(exp_path),
            "--agent-timeout", "600",
        ]
        env = os.environ.copy()
        ctx_conda = queue_exp.get("conda_env")
        if ctx_conda is None:
            ctx_conda = exp.get("conda_env")
        if ctx_conda and isinstance(ctx_conda, str):
            # Find the positional args (--repo-root and beyond) — avoid
            # hardcoding cmd[3:] which breaks if the worker invocation changes.
            try:
                arg_start = cmd.index("--repo-root")
            except ValueError:
                arg_start = 3  # fallback: [python, -m, autopipe.worker, ...]
            cmd = ["conda", "run", "-n", ctx_conda, "python", "-m", "autopipe.worker"] + cmd[arg_start:]
        log_path = paths.logs_dir / f"scheduler_{exp['exp_id']}.log"
        with open(log_path, "ab", buffering=0) as f:
            f.write(f"\n==== {now_ts()} START {' '.join(cmd)}\n".encode())
            f.flush()
            proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=str(repo_root), env=env)
            _workers[exp["exp_id"]] = proc
        running += 1

    return running


def _reap_workers(_workers: Dict[str, subprocess.Popen]) -> None:
    """Remove finished workers from the tracking dict."""
    stale = [eid for eid, p in _workers.items() if p.poll() is not None]
    for eid in stale:
        del _workers[eid]


def _parse_time_window(window_str: str) -> Tuple[int, int] | None:
    """Parse HH:MM-HH:MM into (start_minutes, end_minutes) or None.

    Example: "22:00-08:00" -> (1320, 480) (overnight window).
    Returns None for any invalid format (wrong separators, non-numeric values,
    missing fields, etc.).
    """
    import re as _re
    m = _re.match(r"^(\d{1,2}):(\d{2})-(\d{1,2}):(\d{2})$", window_str.strip())
    if not m:
        return None
    try:
        start_min = int(m.group(1)) * 60 + int(m.group(2))
        end_min = int(m.group(3)) * 60 + int(m.group(4))
    except (ValueError, TypeError):
        return None
    if start_min == end_min:
        return None
    return start_min, end_min


def _in_active_window(window: Tuple[int, int] | None) -> bool:
    """Return True if the current local time falls within *window*. Handles overnight spans.

    Uses local system time (``datetime.datetime.now()``) without pytz awareness.
    For typical overnight scheduling (e.g. "22:00-08:00"), DST transitions are
    unlikely to cause meaningful issues.  If this function ever needs to handle
    sub-hour windows or DST-critical schedules, add pytz support.
    """
    if window is None:
        return True
    start, end = window
    now = datetime.datetime.now()
    now_min = now.hour * 60 + now.minute
    if start <= end:
        return start <= now_min < end
    else:
        return now_min >= start or now_min < end


def _terminate_workers(_workers: Dict[str, subprocess.Popen]) -> None:
    """SIGTERM all running workers for graceful checkpoint-and-exit.

    Uses ``proc.send_signal()`` rather than ``os.killpg()`` because the worker
    is spawned without ``start_new_session`` and shares the scheduler's process
    group.  ``killpg`` would send SIGTERM back to the scheduler itself.
    """
    for eid, proc in list(_workers.items()):
        try:
            if proc.poll() is None:
                proc.send_signal(signal.SIGTERM)
        except Exception:
            pass


def _kill_workers_signaller(
    _workers: Dict[str, subprocess.Popen], signum: int | None = None, frame: Any = None
) -> None:
    """Best-effort terminate all tracked workers on scheduler shutdown.

    Uses ``proc.send_signal()`` not ``os.killpg()`` — see ``_terminate_workers``.
    """
    for eid, proc in list(_workers.items()):
        try:
            if proc.poll() is None:
                proc.send_signal(signal.SIGTERM)
        except Exception:
            pass
    # Sleep in short increments so we can still respond to signals during
    # the grace period (time.sleep blocks signal delivery).
    for _ in range(6):
        alive = any(p.poll() is None for p in _workers.values())
        if not alive:
            break
        time.sleep(0.5)
    for eid, proc in list(_workers.items()):
        try:
            if proc.poll() is None:
                proc.send_signal(signal.SIGKILL)
        except Exception:
            pass
    sys.exit(128 + (signum if signum else signal.SIGTERM))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--poll-seconds", type=int, default=30)
    ap.add_argument("--max-parallel", type=int, default=1, help="Keep 1 for single-node multi-GPU training.")
    ap.add_argument("--once", action="store_true", help="Run a single scan iteration and exit.")
    ap.add_argument(
        "--no-spawn", action="store_true",
        help="Do not start new workers (repair/refresh statuses only).",
    )
    ap.add_argument(
        "--force-steal-lock", action="store_true",
        help="If `.lock_scheduler` exists but the recorded PID is dead, remove it and start.",
    )
    ap.add_argument(
        "--active-window", default="",
        help="Time window for training, e.g. '22:00-08:00' (overnight). Empty = always active.",
    )
    ap.add_argument(
        "--window-kill", action="store_true",
        help="SIGTERM running workers when the active window ends.",
    )
    args = ap.parse_args()

    time_window = _parse_time_window(args.active_window) if args.active_window else None

    repo_root = Path(args.repo_root).resolve()
    paths = default_paths(repo_root)
    paths.queue_dir.mkdir(parents=True, exist_ok=True)
    paths.runs_dir.mkdir(parents=True, exist_ok=True)
    paths.logs_dir.mkdir(parents=True, exist_ok=True)

    lock_path = paths.root / ".lock_scheduler"
    sched_lock = Lock(lock_path, stale_seconds=12 * 3600)
    if not sched_lock.acquire():
        if args.force_steal_lock:
            pid = Lock._read_lock_pid(lock_path)
            if pid is not None and not Lock._pid_alive(pid):
                try:
                    lock_path.unlink(missing_ok=True)
                except Exception:
                    pass
                if not sched_lock.acquire():
                    log_event(source="scheduler", event="lock_busy", detail="failed_to_steal")
                    sys.exit(2)
            else:
                log_event(source="scheduler", event="lock_busy", detail="pid_alive_or_unreadable")
                sys.exit(2)
        else:
            log_event(source="scheduler", event="lock_busy")
            sys.exit(2)

    # Track spawned worker processes so we can terminate them on shutdown
    # and avoid leaving orphaned GPU processes if the scheduler is killed.
    _workers: Dict[str, subprocess.Popen] = {}
    _close_workers = functools.partial(_kill_workers_signaller, _workers)
    signal.signal(signal.SIGTERM, _close_workers)
    signal.signal(signal.SIGINT, _close_workers)

    _was_active = True
    _tick = 0
    _disk_warned = False
    try:
        while True:
            try:
                if not sched_lock.owned():
                    log_event(source="scheduler", event="lock_lost", detail="stolen")
                    sys.exit(2)
                sched_lock.heartbeat()
                _reap_workers(_workers)

                # Check disk space before doing any I/O; warn and skip spawn
                # if dangerously low (< 1 GB) to avoid cascading failures.
                st = os.statvfs(str(paths.runs_dir)) if paths.runs_dir.exists() else None
                free_mb = (st.f_frsize * st.f_bavail) / (1024 * 1024) if st else float("inf")
                if free_mb < 1024:
                    if not _disk_warned:
                        log_event(source="scheduler", event="low_disk",
                                  free_mb=round(free_mb))
                        _disk_warned = True
                    time.sleep(max(1, args.poll_seconds))
                    continue
                _disk_warned = False

                in_window = _in_active_window(time_window)
                if not in_window:
                    if _was_active and _workers and args.window_kill:
                        log_event(source="scheduler", event="active_window_ended",
                                  worker_count=len(_workers))
                        _terminate_workers(_workers)
                        time.sleep(3)
                        _reap_workers(_workers)
                    _was_active = False

                q = list_queue(paths.queue_dir)
                running = _phase1_stale_recovery(paths, q)

                if not args.no_spawn and in_window:
                    running = _phase2_spawn_workers(
                        paths, q, repo_root, running, _workers, args.max_parallel,
                    )
                elif not in_window:
                    running = len(_workers)

                if in_window and not _was_active:
                    log_event(source="scheduler", event="active_window_started")
                _was_active = in_window

                # Heartbeat every 10 cycles (~5 min at default 30s poll)
                _tick += 1
                if _tick % 10 == 0:
                    rkeys = list(_workers.keys()) if _workers else []
                    done = 0
                    pending = 0
                    for ep in q:
                        exp = load_exp(ep)
                        if not exp.get("exp_id"):
                            continue  # skip corrupted entries
                        rp = paths.runs_dir / exp["exp_id"] / "exp.json"
                        if rp.exists():
                            try:
                                st = read_json(rp).get("status", "")
                            except (FileNotFoundError, ValueError):
                                pending += 1
                                continue
                            if st in ("success", "hard_failure"):
                                done += 1
                            elif st in ("pending", "failed"):
                                pending += 1
                        else:
                            pending += 1
                    log_event(source="scheduler", event="heartbeat",
                              running_ids=rkeys if rkeys else [],
                              done=done, total=len(q), pending=pending)

                if args.once:
                    break
            except OSError as e:
                log_event(source="scheduler", event="io_error", error=str(e))
                time.sleep(60)
            time.sleep(max(1, args.poll_seconds))
    finally:
        sched_lock.release()


if __name__ == "__main__":
    main()
