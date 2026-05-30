from __future__ import annotations

import argparse
import functools
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

from autopipe.config import default_paths
from autopipe.io_utils import Lock, atomic_write_json, now_ts, patch_exp, read_json
from autopipe.recovery import recover_stale_worker


def list_queue(queue_dir: Path) -> List[Path]:
    return sorted(queue_dir.glob("*.json"))


def load_exp(path: Path) -> Dict[str, Any]:
    exp = read_json(path)
    exp.setdefault("status", "pending")
    exp.setdefault("attempt", 0)
    return exp


def _load_run_exp(queue_exp: Dict[str, Any], run_exp_path: Path) -> Dict[str, Any]:
    """
    Source-of-truth rules:
    - `autopipe/queue/*.json` defines the initial experiment config.
    - `autopipe/runs/<exp_id>/exp.json` is the mutable working copy (agents may edit it).
    - `autopipe/runs/<exp_id>/status.json` is the authoritative last-attempt outcome.
    """
    if run_exp_path.exists():
        exp = read_json(run_exp_path)
        # Guard: if run exp_id is missing or mismatches, fall back to queue
        # to avoid cross-file contamination.
        run_exp_id = exp.get("exp_id")
        queue_exp_id = queue_exp.get("exp_id")
        if not run_exp_id or run_exp_id != queue_exp_id:
            return dict(queue_exp)
        return exp
    return dict(queue_exp)


# Config keys that the scheduler may merge from queue definition into the
# per-experiment working copy.  Bookkeeping fields (attempt, status,
# updated_at, last_reason, error_hash, etc.) and train_opts are intentionally
# excluded — the agent edits train_opts and merges would clobber those fixes.
MERGE_KEYS = [
    "cfg_path", "trainer", "cmd", "cmd_type", "key", "conda_env",
    "gpus", "nproc", "master_port", "hf_endpoint",
    "train_timeout", "hang_timeout", "vis_timeout", "vis_opts",
    "skip_vis", "retry_sleep", "oom_batch_candidates",
    "max_retries", "agent_cli", "hard_failure_threshold",
]


def _phase1_stale_recovery(paths, q: List[Path]) -> int:
    """Recover stale workers and return count of healthy 'running' experiments.

    For each experiment in *q*:
    - Clean orphaned ``.lock_worker`` files for non-"running" statuses.
    - Delegate ``recover_stale_worker`` for experiments marked "running".
    - Count experiments that appear to have a healthy worker.
    """
    running = 0
    for exp_path in q:
        queue_exp = load_exp(exp_path)
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
            if pid is not None and not Lock._pid_alive(pid):
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
    paths,
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
            for k in MERGE_KEYS:
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
        ctx_conda = queue_exp.get("conda_env") or exp.get("conda_env")
        if ctx_conda:
            cmd = ["conda", "run", "-n", str(ctx_conda), "python", "-m", "autopipe.worker"] + cmd[3:]
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


def _kill_workers_signaller(
    _workers: Dict[str, subprocess.Popen], signum: int | None = None, frame: Any = None
) -> None:
    """Best-effort terminate all tracked workers on scheduler shutdown."""
    for eid, proc in list(_workers.items()):
        try:
            if proc.poll() is None:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except Exception:
            pass
    time.sleep(3)
    for eid, proc in list(_workers.items()):
        try:
            if proc.poll() is None:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:
            pass
    sys.exit(128 + (signum if signum else signal.SIGTERM))


def main():
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
    args = ap.parse_args()

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
                    print("[scheduler] lock busy (failed to steal)", file=sys.stderr)
                    sys.exit(2)
            else:
                print("[scheduler] already running (pid alive or unreadable)", file=sys.stderr)
                sys.exit(2)
        else:
            print("[scheduler] already running", file=sys.stderr)
            sys.exit(2)

    # Track spawned worker processes so we can terminate them on shutdown
    # and avoid leaving orphaned GPU processes if the scheduler is killed.
    _workers: Dict[str, subprocess.Popen] = {}
    _close_workers = functools.partial(_kill_workers_signaller, _workers)
    signal.signal(signal.SIGTERM, _close_workers)
    signal.signal(signal.SIGINT, _close_workers)

    try:
        while True:
            if not sched_lock.owned():
                print("[scheduler] lock lost (stolen by another process), exiting", file=sys.stderr)
                sys.exit(2)
            sched_lock.heartbeat()
            _reap_workers(_workers)

            q = list_queue(paths.queue_dir)
            running = _phase1_stale_recovery(paths, q)

            if not args.no_spawn:
                running = _phase2_spawn_workers(
                    paths, q, repo_root, running, _workers, args.max_parallel,
                )

            if args.once:
                break
            time.sleep(max(1, args.poll_seconds))
    finally:
        sched_lock.release()


if __name__ == "__main__":
    main()
