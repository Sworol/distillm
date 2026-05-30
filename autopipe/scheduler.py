from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

from autopipe.config import default_paths
from autopipe.io_utils import Lock, atomic_write_json, now_ts, read_json


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


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        # Best-effort: if we can't signal it, assume it's alive.
        return True


def list_queue(queue_dir: Path) -> List[Path]:
    return sorted(queue_dir.glob("*.json"))


def load_exp(path: Path) -> Dict[str, Any]:
    exp = read_json(path)
    exp.setdefault("status", "pending")
    exp.setdefault("attempt", 0)
    return exp


def patch_exp(path: Path, base: Dict[str, Any] | None = None, **updates: Any) -> Dict[str, Any]:
    """
    Patch an exp.json atomically without clobbering unrelated keys.

    This reduces scheduler/worker write races by only updating specific fields.
    """
    if path.exists():
        exp = read_json(path)
    else:
        exp = dict(base or {})
    for k, v in updates.items():
        exp[k] = v
    atomic_write_json(path, exp)
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
        # Guard: if run exp_id mismatches, fall back to queue to avoid cross-file contamination.
        if exp.get("exp_id") and exp.get("exp_id") != queue_exp.get("exp_id"):
            return dict(queue_exp)
        return exp
    return dict(queue_exp)


def _recover_stale_worker(
    run_exp_path: Path,
    queue_exp: Dict[str, Any],
    lock_path: Path,
    status_path: Path,
) -> bool:
    """Check worker health and recover if stale.

    Returns True if the worker was recovered (status updated to failed),
    False if it appears healthy and should be counted as running.
    """
    # Case 1: status.json exists with a terminal status — sync exp.json + clean lock.
    try:
        st = read_json(status_path)
        st_status = st.get("status")
        if st_status in {"failed", "success", "hard_failure"}:
            patch_exp(
                run_exp_path,
                base=queue_exp,
                status=st_status,
                attempt=st.get("attempt", 0),
                updated_at=now_ts(),
                last_exit_code=st.get("exit_code"),
                last_reason=st.get("reason"),
            )
            if lock_path.exists():
                pid = _read_lock_pid(lock_path)
                if pid is None or not _pid_alive(pid):
                    try:
                        lock_path.unlink(missing_ok=True)
                    except Exception:
                        pass
            return True
        # Case 2: status.json says "running" but worker lock is missing → stale.
        if st_status == "running" and not lock_path.exists():
            patch_exp(
                run_exp_path,
                base=queue_exp,
                status="failed",
                updated_at=now_ts(),
                last_reason="stale_worker",
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

    # Case 3: No status.json (or could not be read) and no lock after grace period.
    if not lock_path.exists():
        try:
            age = time.time() - status_path.stat().st_mtime
        except FileNotFoundError:
            age = 0
        if age > 120:
            patch_exp(
                run_exp_path,
                base=queue_exp,
                status="failed",
                updated_at=now_ts(),
                last_reason="stale_worker",
            )
            atomic_write_json(
                status_path,
                {
                    "status": "failed",
                    "updated_at": now_ts(),
                    "attempt": 0,
                    "exit_code": 1,
                    "reason": "stale_worker",
                },
            )
            return True

    # Case 4: Lock exists but recorded PID is dead → clean lock + mark stale.
    pid = _read_lock_pid(lock_path)
    if pid is not None and not _pid_alive(pid):
        try:
            lock_path.unlink(missing_ok=True)
        except Exception:
            pass
        patch_exp(
            run_exp_path,
            base=queue_exp,
            status="failed",
            updated_at=now_ts(),
            last_reason="stale_worker",
        )
        atomic_write_json(
            status_path,
            {
                "status": "failed",
                "updated_at": now_ts(),
                "attempt": 0,
                "exit_code": 1,
                "reason": "stale_worker",
            },
        )
        return True

    # Worker appears healthy.
    return False
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--poll-seconds", type=int, default=30)
    ap.add_argument("--max-parallel", type=int, default=1, help="Keep 1 for single-node multi-GPU training.")
    ap.add_argument("--once", action="store_true", help="Run a single scan iteration and exit.")
    ap.add_argument(
        "--no-spawn",
        action="store_true",
        help="Do not start new workers (repair/refresh statuses only).",
    )
    ap.add_argument(
        "--force-steal-lock",
        action="store_true",
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
            pid = _read_lock_pid(lock_path)
            if pid is not None and not _pid_alive(pid):
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

    try:
        while True:
            # Health check: if another scheduler stole our lock, exit immediately
            if not sched_lock.owned():
                print("[scheduler] lock lost (stolen by another process), exiting", file=sys.stderr)
                sys.exit(2)
            sched_lock.heartbeat()

            q = list_queue(paths.queue_dir)
            # Phase 1: refresh statuses / recover stale running workers.
            running = 0
            for exp_path in q:
                queue_exp = load_exp(exp_path)
                run_root = paths.runs_dir / queue_exp["exp_id"]
                run_exp_path = run_root / "exp.json"
                exp = _load_run_exp(queue_exp, run_exp_path)

                if exp.get("status") != "running":
                    continue

                status_path = run_root / "status.json"
                worker_lock_path = run_root / ".lock_worker"
                if _recover_stale_worker(run_exp_path, queue_exp, worker_lock_path, status_path):
                    continue

                running += 1

            for exp_path in q:
                if args.no_spawn:
                    continue
                queue_exp = load_exp(exp_path)
                run_root = paths.runs_dir / queue_exp["exp_id"]
                run_exp_path = run_root / "exp.json"
                exp = _load_run_exp(queue_exp, run_exp_path)
                status = exp.get("status")
                if status in {"success", "hard_failure"}:
                    continue
                if status == "running":
                    # Already counted in phase 1.
                    continue

                if running >= args.max_parallel:
                    break

                # Start or retry
                attempt = int(exp.get("attempt", 0))
                max_retries = int(exp.get("max_retries", 2))

                if status == "failed":
                    # Exponential backoff: skip if still in cooldown period.
                    retry_sleep_base = int(exp.get("retry_sleep", 60))
                    consecutive_failures = int(exp.get("consecutive_failures", 0))
                    retry_sleep = min(retry_sleep_base * (2 ** consecutive_failures), 900)  # max 15min
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
                    # Only auto-retry an aborted experiment if the underlying config has been
                    # modified (e.g., a hotfix). Otherwise, require manual intervention.
                    qmtime = exp_path.stat().st_mtime if exp_path.exists() else 0
                    rmtime = run_exp_path.stat().st_mtime if run_exp_path.exists() else 0
                    if qmtime > rmtime:
                        # Queue config was updated — retry is likely intentional.
                        attempt = 0
                        status = "failed"
                    else:
                        continue  # no changes detected, keep aborted

                # Ensure per-exp working config exists.
                if not run_exp_path.exists():
                    run_root.mkdir(parents=True, exist_ok=True)
                    atomic_write_json(run_exp_path, exp)
                else:
                    # Keep per-exp working config fresh with queue updates.
                    # This is important when we hotfix queue configs (e.g., add master_port
                    # or offline pretrained paths) and want retries to pick them up.
                    #
                    # We intentionally only merge known config keys to avoid clobbering
                    # worker/agent-written bookkeeping fields like `attempt`, `status`,
                    # `updated_at`, `last_reason`, etc.
                    try:
                        cur = read_json(run_exp_path)
                    except Exception:
                        cur = exp
                    merge_keys = [
                        "cfg_path",
                        "trainer",
                        "conda_env",
                        "gpus",
                        "nproc",
                        "master_port",
                        "hf_endpoint",
                        "train_timeout",
                        "vis_timeout",
                        "vis_opts",
                        "skip_vis",
                        "retry_sleep",
                        "oom_batch_candidates",
                        "max_retries",
                        "agent_cli",
                        "hard_failure_threshold",
                    ]
                    # Note: train_opts is intentionally NOT in merge_keys.
                    # The agent edits train_opts to fix training failures, and
                    # scheduler merges would clobber those fixes with queue defaults.
                    changed = False
                    for k in merge_keys:
                        if k in queue_exp and cur.get(k) != queue_exp.get(k):
                            cur[k] = queue_exp.get(k)
                            changed = True
                    if changed:
                        atomic_write_json(run_exp_path, cur)

                # NOTE: The worker sets its own status to "running" after acquiring
                # the worker lock. We do NOT set it here because if the worker fails
                # to start (lock busy), exp.json would incorrectly remain "running"
                # and block future retries.

                cmd = [
                    sys.executable,
                    "-m",
                    "autopipe.worker",
                    "--repo-root",
                    str(repo_root),
                    "--exp-json",
                    str(exp_path),
                    "--agent-timeout",
                    "600",
                ]
                env = os.environ.copy()
                # If conda environment is specified in the queue JSON, run the worker under it.
                # Important: call `python` (resolved inside the env), not an absolute python path.
                # This avoids import mismatches (e.g. tensorboardX missing) and ensures the
                # repo root is on sys.path via cwd=repo_root.
                conda_env = queue_exp.get("conda_env") or exp.get("conda_env")
                if conda_env:
                    cmd = ["conda", "run", "-n", str(conda_env), "python", "-m", "autopipe.worker"] + cmd[3:]
                log_path = paths.logs_dir / f"scheduler_{exp['exp_id']}.log"
                with open(log_path, "ab", buffering=0) as f:
                    f.write(f"\n==== {now_ts()} START {' '.join(cmd)}\n".encode())
                    f.flush()
                    subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=str(repo_root), env=env)
                running += 1

            if args.once:
                break
            time.sleep(max(1, args.poll_seconds))
    finally:
        sched_lock.release()


if __name__ == "__main__":
    main()
