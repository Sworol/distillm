from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import socket
import subprocess
import signal
import sys
import time
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from autopipe.config import default_paths
from autopipe.io_utils import (
    Lock,
    atomic_write_json,
    classify_failure,
    now_ts,
    parse_int_list,
    read_json,
)
from autopipe.agent import run_agent
from autopipe.guard import snapshot_git


def run_cmd(cmd: List[str], log_path: Path, env: Dict[str, str]) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "ab", buffering=0) as f:
        f.write(f"\n==== {now_ts()} COMMAND: {' '.join(cmd)}\n".encode())
        f.flush()
        # Create a new process group so we can terminate all children on shutdown.
        proc = subprocess.Popen(
            cmd,
            stdout=f,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )
        try:
            return proc.wait()
        except KeyboardInterrupt:
            # Best-effort terminate the whole process group.
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except Exception:
                pass
            try:
                return proc.wait(timeout=30)
            except Exception:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except Exception:
                    pass
                return 130


def run_cmd_with_timeout(cmd: List[str], log_path: Path, env: Dict[str, str], timeout_seconds: int) -> int:
    """
    Run a command with an overall wall-clock timeout.

    Returns 124 on timeout (like GNU timeout), otherwise the subprocess return code.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "ab", buffering=0) as f:
        f.write(f"\n==== {now_ts()} COMMAND: {' '.join(cmd)}\n".encode())
        f.flush()
        proc = subprocess.Popen(
            cmd,
            stdout=f,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )
        try:
            return proc.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            f.write(f"\n==== {now_ts()} TIMEOUT after {timeout_seconds}s\n".encode())
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except Exception:
                pass
            try:
                return proc.wait(timeout=30)
            except Exception:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except Exception:
                    pass
                return 124


def run_cmd_maybe_timeout(
    cmd: List[str],
    log_path: Path,
    env: Dict[str, str],
    timeout_seconds: int | None,
) -> int:
    """
    Run a command with an optional timeout.

    If timeout_seconds is None or <= 0, run without a wall-clock timeout.
    """
    if timeout_seconds is None:
        return run_cmd(cmd, log_path, env)
    try:
        val = int(timeout_seconds)
    except Exception:
        val = 0
    if val <= 0:
        return run_cmd(cmd, log_path, env)
    return run_cmd_with_timeout(cmd, log_path, env, timeout_seconds=val)


def _strip_opt_prefix(opt: str) -> str:
    return opt.split("=", 1)[0].strip() if "=" in opt else opt.strip()


def _parse_kv_opt(opt: str) -> tuple[str, str] | None:
    if "=" not in opt:
        return None
    k, v = opt.split("=", 1)
    return k.strip(), v.strip()


def apply_oom_batch_backoff(exp: Dict[str, Any]) -> bool:
    """
    If an experiment fails with CUDA OOM, reduce batch size for the next attempt.

    This is intentionally conservative and only touches CLI overrides in `train_opts`:
    - `trainer.data.batch_size=<N>`

    Returns True if an update was applied (i.e., exp was modified in-memory).
    """
    candidates = exp.get("oom_batch_candidates")
    if not candidates:
        return False
    try:
        cand_list = [int(x) for x in candidates]
    except Exception:
        return False

    # Determine current batch size override if present.
    current: int | None = None
    for opt in exp.get("train_opts", []):
        kv = _parse_kv_opt(str(opt))
        if not kv:
            continue
        k, v = kv
        if k == "trainer.data.batch_size":
            try:
                current = int(v.strip("'\""))
            except Exception:
                current = None
            break

    # Choose the next smaller candidate.
    cand_list = sorted(set(cand_list), reverse=True)
    next_bs: int | None = None
    if current is None:
        # No explicit override; fall back to the smallest candidate as a safe choice.
        next_bs = min(cand_list)
    else:
        for bs in cand_list:
            if bs < current:
                next_bs = bs
                break
    if next_bs is None:
        return False

    # Rewrite train_opts: drop existing batch override, append the new one.
    new_opts: list[str] = []
    for opt in exp.get("train_opts", []):
        if _strip_opt_prefix(str(opt)) == "trainer.data.batch_size":
            continue
        new_opts.append(str(opt))
    new_opts.append(f"trainer.data.batch_size={next_bs}")
    exp["train_opts"] = new_opts
    exp["last_oom_batch_size"] = next_bs
    return True


def pick_free_tcp_port() -> int:
    """
    Best-effort free port picker for torchrun rendezvous.

    Note: there is an inherent TOCTOU race (another process could grab the port
    after we pick it). In practice this avoids the common failure mode where the
    default port 29500 is already occupied.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("", 0))
        return int(s.getsockname()[1])


def resolve_master_port(exp: Dict[str, Any]) -> int:
    """
    Resolve the rendezvous port for torch.distributed.run.

    - If exp.json specifies a positive integer `master_port`, use it.
    - If `master_port` is 0/"auto"/None/missing, pick a free port.
    """
    raw = exp.get("master_port", None)
    if raw is None:
        return pick_free_tcp_port()
    if isinstance(raw, str) and raw.strip().lower() in {"", "auto"}:
        return pick_free_tcp_port()
    try:
        val = int(raw)
    except Exception:
        return pick_free_tcp_port()
    return val if val > 0 else pick_free_tcp_port()


def ensure_exp_sane(exp: Dict[str, Any]) -> None:
    required = ["exp_id"]
    cmd_type = exp.get("cmd_type", "torchrun")
    if cmd_type != "bash":
        required.extend(["cfg_path", "trainer"])
    missing = [k for k in required if not exp.get(k)]
    if missing:
        raise ValueError(f"exp.json missing required keys: {missing}")
    if cmd_type == "bash" and "cmd" not in exp:
        raise ValueError("bash cmd_type requires 'cmd' key")
    if "nproc" in exp:
        nproc = int(exp["nproc"])
        if nproc <= 0:
            raise ValueError(f"invalid nproc: {nproc}")
    if "master_port" in exp:
        raw = exp["master_port"]
        if isinstance(raw, str) and raw.strip().lower() in {"", "auto"}:
            return
        try:
            _ = int(raw)
        except Exception:
            raise ValueError(f"invalid master_port: {raw!r}")


def latest_run_dir(repo_root: Path, trainer: str, cfg_path: str) -> Path:
    runs_dir = repo_root / "runs"
    slug = cfg_path.replace(".py", "").replace("/", ".").replace(".", "_")
    pattern = f"{trainer}_{slug}_*"
    candidates = sorted(runs_dir.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"no run dir for pattern runs/{pattern}")
    return candidates[0]


def choose_vis_checkpoint(
    repo_root: Path,
    trainer: str,
    cfg_path: str,
) -> tuple[Path, Path]:
    """
    Pick the checkpoint for visualization.

    Today we rely on the repo's run.py naming convention, so the best we can do
    without modifying core training code is: select the most-recent run dir for
    this trainer+cfg.
    """
    run_dir = latest_run_dir(repo_root, trainer, cfg_path)
    ckpt = run_dir / "net.pth"
    return run_dir, ckpt


def _read_autopipe_run_dir(run_dir: Path) -> Path | None:
    p = run_dir / "autopipe_run.json"
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except Exception:
        return None
    raw = str(data.get("run_dir", "")).strip()
    if not raw:
        return None
    return Path(raw)


def _read_autopipe_ckpt_path(run_dir: Path) -> Path | None:
    p = run_dir / "autopipe_run.json"
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except Exception:
        return None
    raw = str(data.get("ckpt_path", "")).strip()
    if not raw:
        return None
    return Path(raw)


def _last_error_hash(train_log: Path, vis_log: Path | None = None) -> str:
    """Extract error fingerprint from train.log and vis.log for agent dedup."""
    try:
        lines = []
        for log_path in [train_log, vis_log]:
            if log_path is None or not log_path.exists():
                continue
            with open(log_path, "r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    lower = line.lower()
                    if "error" in lower or "traceback" in lower or "exitcode" in lower:
                        lines.append(line.strip())
        return hashlib.md5("\n".join(lines[-5:]).encode()).hexdigest() if lines else ""
    except Exception:
        return ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=".", help="Repo root")
    ap.add_argument("--exp-json", required=True, help="Path to experiment json in autopipe/queue")
    ap.add_argument("--agent-timeout", type=int, default=600, help="Seconds for agent repair (default: 600)")
    args = ap.parse_args()

    repo_root = Path(args.repo_root).resolve()
    queue_exp_path = Path(args.exp_json).resolve()
    queue_exp = read_json(queue_exp_path)

    paths = default_paths(repo_root)
    run_root = paths.runs_dir / queue_exp["exp_id"]
    run_root.mkdir(parents=True, exist_ok=True)
    run_exp_path = run_root / "exp.json"
    if not run_exp_path.exists():
        # First run: copy queue definition into per-exp working config.
        atomic_write_json(run_exp_path, queue_exp)
    exp = read_json(run_exp_path)
    ensure_exp_sane(exp)

    lock = Lock(run_root / ".lock_worker", stale_seconds=12 * 3600)
    if not lock.acquire():
        print(f"[worker] lock busy: {run_root}", file=sys.stderr)
        sys.exit(2)

    try:
        status_path = run_root / "status.json"
        attempt = int(exp.get("attempt", 0)) + 1
        attempt_dir = run_root / f"attempt_{attempt:03d}"
        attempt_dir.mkdir(parents=True, exist_ok=True)

        atomic_write_json(
            status_path,
            {
                "status": "running",
                "updated_at": now_ts(),
                "attempt": attempt,
                "exp_id": exp.get("exp_id"),
                "cfg_path": exp.get("cfg_path"),
                "trainer": exp.get("trainer"),
            },
        )
        # Update attempt counter in the working exp config so agent/scheduler see progress.
        exp["attempt"] = attempt
        exp["status"] = "running"
        exp["updated_at"] = now_ts()
        exp["queue_exp_json"] = str(queue_exp_path)
        atomic_write_json(run_exp_path, exp)

        # Freeze exp config used for this attempt (copy from updated run_root/exp.json).
        # This avoids confusing snapshots where `attempt_XXX/exp.json` contains the
        # previous attempt number.
        atomic_write_json(attempt_dir / "exp.json", exp)

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = exp.get("gpus", "0,1,2,3")
        # Force hf mirror endpoint to avoid relying on user shell environment.
        # (If you want to disable this, set HF_ENDPOINT explicitly in your process wrapper and
        # remove this line.)
        env["HF_ENDPOINT"] = exp.get("hf_endpoint", "https://hf-mirror.com")
        env["AUTOPIPE_EXP_ID"] = str(exp.get("exp_id", ""))
        env["AUTOPIPE_ATTEMPT"] = str(attempt)

        cmd_type = exp.get("cmd_type", "torchrun")
        nproc = int(exp.get("nproc", 4))
        master_port = resolve_master_port(exp) if cmd_type != "bash" else 0
        vis_cmd = None
        run_dir = None
        ckpt = None
        if cmd_type == "bash":
            # Run a bash script (e.g., distillm shell scripts with their own torchrun)
            train_cmd = ["bash", exp["cmd"]]
        else:
            train_cmd = [
                sys.executable,
                "-m",
                "torch.distributed.run",
                f"--nproc_per_node={nproc}",
                f"--master_port={master_port}",
                "run.py",
                "-c",
                exp["cfg_path"],
                "-m",
                "train",
            ] + list(exp.get("train_opts", []))

        train_log = attempt_dir / "train.log"
        try:
            atomic_write_json(
                status_path,
                {
                    "status": "running",
                    "updated_at": now_ts(),
                    "attempt": attempt,
                    "exp_id": exp.get("exp_id"),
                    "cfg_path": exp.get("cfg_path"),
                    "trainer": exp.get("trainer"),
                    "train_cmd": train_cmd,
                    "cuda_visible_devices": env.get("CUDA_VISIBLE_DEVICES"),
                    "hf_endpoint": env.get("HF_ENDPOINT"),
                    "cmd_type": cmd_type,
                },
            )
            # Guard against "silent hangs" during model build / extension compilation.
            # If the training process doesn't make progress for too long, terminate and
            # let the agent/scheduler retry with an updated config.
            rc = run_cmd_maybe_timeout(
                train_cmd,
                train_log,
                env,
                timeout_seconds=exp.get("train_timeout", 60 * 60),
            )
        except KeyboardInterrupt:
            # Treat Ctrl+C as a "failed" attempt so autopipe can record status and
            # optionally trigger the auto-repair agent.
            with open(train_log, "ab", buffering=0) as f:
                f.write(f"\n==== {now_ts()} INTERRUPTED (KeyboardInterrupt)\n".encode())
            rc = 130

        # If training succeeded, run visualization using that run's net.pth.
        vis_log = attempt_dir / "vis.log"
        if rc == 0 and not exp.get("skip_vis", False) and cmd_type != "bash":
            run_dir, ckpt = choose_vis_checkpoint(
                repo_root=repo_root,
                trainer=exp["trainer"],
                cfg_path=exp["cfg_path"],
            )
            # If training recorded an explicit run_dir pointer, prefer it.
            recorded = _read_autopipe_run_dir(run_dir)
            if recorded is not None:
                run_dir = recorded
                # Prefer an explicitly recorded ckpt_path if present.
                ckpt_recorded = _read_autopipe_ckpt_path(run_dir)
                ckpt = ckpt_recorded if ckpt_recorded is not None else (run_dir / "net.pth")
            if not ckpt.exists():
                raise FileNotFoundError(f"missing net.pth: {ckpt}")
            vis_dir = attempt_dir / "vis"
            if vis_dir.exists():
                shutil.rmtree(vis_dir)
            vis_dir.mkdir(parents=True, exist_ok=True)
            vis_cmd = [
                sys.executable,
                "run.py",
                "-c",
                exp["cfg_path"],
                "-m",
                "test",
                f"vis_dir={vis_dir.as_posix()}",
                f"model.checkpoint_path={ckpt.as_posix()}",
            ] + list(exp.get("vis_opts", []))
            try:
                atomic_write_json(
                    status_path,
                    {
                        "status": "running",
                        "updated_at": now_ts(),
                        "attempt": attempt,
                        "exp_id": exp.get("exp_id"),
                        "cfg_path": exp.get("cfg_path"),
                        "trainer": exp.get("trainer"),
                        "train_cmd": train_cmd,
                        "vis_cmd": vis_cmd,
                        "run_dir": str(run_dir),
                        "ckpt_path": str(ckpt),
                        "cuda_visible_devices": env.get("CUDA_VISIBLE_DEVICES"),
                        "hf_endpoint": env.get("HF_ENDPOINT"),
                        "nproc": nproc,
                        "master_port": master_port,
                    },
                )
                rc = run_cmd_with_timeout(
                    vis_cmd,
                    vis_log,
                    env,
                    timeout_seconds=int(exp.get("vis_timeout", 60 * 60)),
                )
            except KeyboardInterrupt:
                with open(vis_log, "ab", buffering=0) as f:
                    f.write(f"\n==== {now_ts()} INTERRUPTED (KeyboardInterrupt)\n".encode())
                rc = 130

        if rc == 0:
            atomic_write_json(
                status_path,
                {
                    "status": "success",
                    "updated_at": now_ts(),
                    "attempt": attempt,
                    "run_dir": str(run_dir) if "run_dir" in locals() else None,
                    "ckpt_path": str(ckpt) if "ckpt" in locals() else None,
                },
            )
            exp["status"] = "success"
            exp["updated_at"] = now_ts()
            exp["consecutive_failures"] = 0
            exp["error_hash"] = ""
            exp["last_failed_at"] = ""
            if "run_dir" in locals():
                exp["last_run_dir"] = str(run_dir)
            atomic_write_json(run_exp_path, exp)
            sys.exit(0)

        if rc == 130:
            reason = "interrupted"
        elif rc == 124:
            reason = "timeout"
        else:
            reason = classify_failure(train_log)
        atomic_write_json(
            status_path,
            {
                "status": "failed",
                "updated_at": now_ts(),
                "attempt": attempt,
                "exit_code": rc,
                "reason": reason,
                "train_cmd": train_cmd,
                "vis_cmd": vis_cmd if "vis_cmd" in locals() else None,
                "run_dir": str(run_dir) if "run_dir" in locals() else None,
                "ckpt_path": str(ckpt) if "ckpt" in locals() else None,
                "cuda_visible_devices": env.get("CUDA_VISIBLE_DEVICES"),
                "hf_endpoint": env.get("HF_ENDPOINT"),
                "nproc": nproc,
                "master_port": master_port,
            },
        )
        exp["status"] = "failed"
        exp["updated_at"] = now_ts()
        exp["last_exit_code"] = rc
        exp["last_reason"] = reason

        # If we hit OOM, prefer a deterministic, config-driven batch backoff over
        # an LLM "repair" attempt. This keeps retries fast and reproducible.
        if reason == "oom":
            if apply_oom_batch_backoff(exp):
                exp["status"] = "pending"
                exp["updated_at"] = now_ts()
                atomic_write_json(run_exp_path, exp)
                sys.exit(rc)
        atomic_write_json(run_exp_path, exp)

        # Optional auto-repair: run agent inside the experiment run directory.
        # Agent is sandboxed to workspace-write for this experiment directory and should
        # only adjust exp.json for the next attempt.
        if attempt < int(exp.get("max_retries", 2)):
            error_hash = _last_error_hash(train_log, vis_log if vis_log.exists() else None)

            # Track how many times the agent has attempted to fix each error hash.
            # If the same hash persists after agent fixes, we bail out to avoid
            # wasting compute on unfixable errors.
            agent_fix_hashes = exp.get("agent_fix_hashes", {})

            prev_hash = ""
            try:
                prev_status = read_json(status_path)
                prev_hash = prev_status.get("error_hash", "")
            except Exception:
                pass

            hard_failure_threshold = int(exp.get("hard_failure_threshold", 2))

            if error_hash and error_hash == prev_hash:
                # Agent already ran for this error; check hard failure threshold.
                agent_fix_count = int(agent_fix_hashes.get(error_hash, 0))
                if agent_fix_count >= hard_failure_threshold:
                    exp["status"] = "hard_failure"
                    exp["updated_at"] = now_ts()
                    exp["last_reason"] = f"agent failed to fix '{reason}' after {agent_fix_count} attempts (hash={error_hash})"
                    atomic_write_json(run_exp_path, exp)
                    atomic_write_json(
                        status_path,
                        {
                            "status": "hard_failure",
                            "updated_at": now_ts(),
                            "attempt": attempt,
                            "exit_code": rc,
                            "reason": f"agent_fix_exhausted:{reason}",
                            "error_hash": error_hash,
                            "agent_fix_count": agent_fix_count,
                        },
                    )
                    sys.exit(rc)

                with open(run_root / "agent_skip.txt", "a", encoding="utf-8") as f:
                    f.write(f"{now_ts()} skipped agent (duplicate error {error_hash}, agent_fix_count={agent_fix_count})\n")
            else:
                try:
                    snapshot_git(repo_root, run_root, "pre_agent")
                    run_agent(run_root, timeout_seconds=args.agent_timeout, agent_cli=exp.get("agent_cli", "claude"))
                    snapshot_git(repo_root, run_root, "post_agent")
                    # Record that agent ran for this error.
                    agent_fix_hashes[error_hash] = int(agent_fix_hashes.get(error_hash, 0)) + 1
                    exp["agent_fix_hashes"] = agent_fix_hashes
                except Exception as exc:
                    # Agent is best-effort; keep the failure recorded and let scheduler retry.
                    with open(run_root / "agent_error.txt", "a", encoding="utf-8") as f:
                        f.write(f"{now_ts()} {repr(exc)}\n")

            # Update error tracking for backoff and dedup.
            exp["error_hash"] = error_hash
            exp["consecutive_failures"] = int(exp.get("consecutive_failures", 0)) + 1
            exp["last_failed_at"] = str(time.time())
            atomic_write_json(run_exp_path, exp)

            # Also update status.json so next attempt can dedup.
            try:
                st = read_json(status_path)
                st["error_hash"] = error_hash
                atomic_write_json(status_path, st)
            except Exception:
                pass
        sys.exit(rc)
    finally:
        lock.release()


if __name__ == "__main__":
    main()
