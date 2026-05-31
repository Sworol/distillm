from __future__ import annotations

import argparse
import enum
import hashlib
import os
import re
import shutil
import shlex
import socket
import subprocess
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List

from autopipe.config import HARD_FAILURE_THRESHOLD, default_paths
from autopipe.io_utils import (
    Lock,
    atomic_write_json,
    classify_failure,
    get_classification_text,
    log_event,
    now_ts,
    read_json,
)
from autopipe.agent import run_agent
from autopipe.git_utils import snapshot_git

# Module-level references to the currently-running subprocess and worker lock,
# so the SIGTERM handler can clean up GPU processes and release the lock before
# the worker exits. If we don't kill the subprocess, the scheduler's _kill_workers
# sends SIGTERM to the conda process group, but the training subprocess
# (start_new_session=True) is in a separate session and becomes an orphan that
# holds GPU memory.
_current_subprocess: subprocess.Popen | None = None
_worker_lock: "Lock | None" = None


def _sigterm_handler(signum: int, frame: Any) -> None:
    # Prevent re-entry: if the handler fires recursively (e.g. os.kill at
    # the end sends SIGTERM to self), the second invocation sees this flag
    # and falls through to SIG_DFL.
    if getattr(_sigterm_handler, "_handled", False):
        signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)
        return
    _sigterm_handler._handled = True
    proc = _current_subprocess
    if proc is not None and proc.poll() is None:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except Exception:
            pass
        try:
            proc.wait(timeout=2)
        except Exception:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except Exception:
                pass
    # Release the worker lock so the scheduler can clean up immediately
    # instead of waiting for the 12h stale timeout. Python's default SIGTERM
    # handler calls _exit() which skips finally blocks, so we must do this here.
    lock = _worker_lock
    if lock is not None:
        try:
            lock.release()
        except Exception:
            pass
    # _handled=True above guards the re-entry from this kill.
    os.kill(os.getpid(), signum)


def _hang_watcher(log_path: Path, proc: subprocess.Popen, hang_timeout: int) -> None:
    """Daemon thread: kill the training process if the log file stops growing.

    Many training failures manifest as silent hangs (GPU utilization drops to 0,
    no log output). A wall-clock timeout alone can waste hours of GPU time on a
    process that is already dead. This watcher checks log file size periodically
    and sends SIGTERM if no progress is detected within hang_timeout seconds.
    """
    last_size = 0
    try:
        last_size = log_path.stat().st_size
    except OSError:
        pass
    last_change = time.time()
    check_interval = min(300, max(30, hang_timeout // 4))
    while proc.poll() is None:
        time.sleep(check_interval)
        try:
            cur = log_path.stat().st_size
        except OSError:
            last_change = time.time()
            continue
        if cur != last_size:
            last_size = cur
            last_change = time.time()
        elif time.time() - last_change > hang_timeout:
            # Process may have exited naturally during sleep/stat. Re-check
            # before sending any signal to avoid a PID-reuse race: if the
            # original process died and the kernel recycled its PID to a new
            # process group, os.killpg would SIGTERM an unrelated workload.
            if proc.poll() is not None:
                return
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except Exception:
                pass
            return


def _kill_and_wait(proc: subprocess.Popen, fallback_code: int) -> int:
    """Send SIGTERM, wait up to 30 s, then SIGKILL if still alive.

    Always returns *fallback_code* — never the process exit code — because
    callers use the return value to classify outcomes (124=timeout, 130=interrupt).
    Returning the process signal code (-15/-9) would cause ``_handle_outcome`` to
    misclassify a timeout as ``"killed"``.
    """
    pgid = os.getpgid(proc.pid)
    try:
        os.killpg(pgid, signal.SIGTERM)
    except Exception:
        pass
    try:
        proc.wait(timeout=30)
    except Exception:
        try:
            os.killpg(pgid, signal.SIGKILL)
        except Exception:
            pass
        try:
            proc.wait(timeout=10)
        except Exception:
            pass
    return fallback_code


def run_cmd(
    cmd: List[str],
    log_path: Path,
    env: Dict[str, str],
    timeout_seconds: int | None = None,
    hang_timeout: int | None = None,
) -> int:
    """Run a command with optional wall-clock timeout and hang detection.

    - timeout_seconds: max total wall-clock time (None = no limit).
    - hang_timeout: max seconds without log growth before killing (None = no hang detection).
    """
    global _current_subprocess
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "ab", buffering=0) as f:
        f.write(f"\n==== {now_ts()} COMMAND: {' '.join(cmd)}\n".encode())
        f.flush()
        # Create a new process group so we can terminate all children on shutdown.
        # There is an unavoidable race between the kernel creating the process
        # (in Popen) and _current_subprocess becoming visible to the SIGTERM
        # handler.  If SIGTERM lands in this window the handler sees the
        # previous (already reaped) _current_subprocess, which is acceptable.
        proc = subprocess.Popen(
            cmd,
            stdout=f,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )
        _current_subprocess = proc

        # Start hang-detection watcher if configured.
        watcher = None
        if hang_timeout and hang_timeout > 0:
            watcher = threading.Thread(
                target=_hang_watcher,
                args=(log_path, proc, hang_timeout),
                daemon=True,
            )
            watcher.start()

        try:
            return proc.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            f.write(f"\n==== {now_ts()} TIMEOUT after {timeout_seconds}s\n".encode())
            return _kill_and_wait(proc, 124)
        except KeyboardInterrupt:
            return _kill_and_wait(proc, 130)
        finally:
            _current_subprocess = None


def _prepare_environment(
    exp: Dict[str, Any],
    repo_root: Path,
    attempt: int,
) -> tuple[Dict[str, str], List[str]]:
    """Build env vars and training command for a single attempt.

    Returns (env, train_cmd) where env is the full os.environ copy with
    CUDA_VISIBLE_DEVICES, HF_ENDPOINT, PATH injection, PYTHONPATH, and
    TRAIN_* exports set, and train_cmd is the argv list to execute.
    """
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = exp.get("gpus", "0,1,2,3")
    env["HF_ENDPOINT"] = exp.get("hf_endpoint", "https://hf-mirror.com")
    env["AUTOPIPE_EXP_ID"] = str(exp.get("exp_id", ""))
    env["AUTOPIPE_ATTEMPT"] = str(attempt)

    # Inject conda env bin into PATH so bash scripts find the right torchrun/python.
    conda_env = exp.get("conda_env", "")
    if conda_env:
        conda_exe = shutil.which("conda")
        if conda_exe:
            conda_root = Path(conda_exe).resolve().parent.parent
            conda_bin = str(conda_root / "envs" / conda_env / "bin")
        else:
            conda_bin = f"/anaconda3/envs/{conda_env}/bin"
        if os.path.isdir(conda_bin):
            env["PATH"] = f"{conda_bin}:{env.get('PATH', '')}"
    # Ensure repo root is on sys.path for scripts that import data_utils etc.
    env["PYTHONPATH"] = str(repo_root) + (f":{env['PYTHONPATH']}" if env.get("PYTHONPATH") else "")

    # Export train_opts from exp.json as env vars so shell scripts pick them up.
    for k, v in exp.get("train_opts", {}).items():
        env[f"TRAIN_{k.upper()}"] = str(v)

    # Build training command.
    cmd_type = exp.get("cmd_type", "torchrun")
    nproc = int(exp.get("nproc", 4))
    master_port = resolve_master_port(exp) if cmd_type != "bash" else 0
    if cmd_type == "bash":
        train_cmd = ["bash"] + shlex.split(exp["cmd"])
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
        ] + [f"{k}={v}" for k, v in exp.get("train_opts", {}).items()]

    return env, train_cmd


def _init_attempt(
    run_root: Path,
    exp: Dict[str, Any],
    queue_exp_path: str,
) -> tuple[int, Path, Path]:
    """Set up a new attempt: bump counter, create dirs, write running status.

    Returns (attempt, attempt_dir, status_path).
    Side-effects: writes exp.json and status.json.
    """
    attempt = int(exp.get("attempt", 0)) + 1
    attempt_dir = run_root / f"attempt_{attempt:03d}"
    attempt_dir.mkdir(parents=True, exist_ok=True)
    status_path = run_root / "status.json"

    exp["attempt"] = attempt
    exp["status"] = "running"
    exp["updated_at"] = now_ts()
    exp["queue_exp_json"] = str(queue_exp_path)
    atomic_write_json(run_root / "exp.json", exp)

    # Freeze exp config used for this attempt so snapshots are self-consistent.
    atomic_write_json(attempt_dir / "exp.json", exp)
    return attempt, attempt_dir, status_path


def apply_oom_batch_backoff(exp: Dict[str, Any]) -> bool:
    """
    If an experiment fails with CUDA OOM, reduce batch size for the next attempt.

    Handles dict-type `train_opts` (the canonical format from make_queue.py).
    Looks for batch-size-related keys and sets them to the next smaller candidate
    from `oom_batch_candidates`.

    If the current batch_size is already the smallest candidate (typically 1),
    no further reduction is possible — returns False so the normal failure
    path can escalate to the agent or hard_failure.

    Returns True if an update was applied (i.e., exp was modified in-memory).
    """
    candidates = exp.get("oom_batch_candidates")
    if not candidates:
        return False
    try:
        cand_list = [int(x) for x in candidates]
    except Exception:
        return False

    train_opts = exp.get("train_opts", {})
    if not isinstance(train_opts, dict):
        return False

    # Determine current batch size from dict keys.
    batch_keys = ["batch_size", "per_device_train_batch_size", "micro_batch_size"]
    current: int | None = None
    current_key: str | None = None
    for key in batch_keys:
        if key in train_opts:
            try:
                current = int(train_opts[key])
                current_key = key
            except (ValueError, TypeError):
                pass
            break

    if current is None:
        # No train_opts batch_size key found; can't safely reduce.
        # Shell scripts have their own hardcoded defaults, and guessing
        # min(cand_list) may conflict with them, wasting an attempt.
        return False

    # Choose the next smaller candidate.
    cand_list = sorted(set(cand_list), reverse=True)
    next_bs: int | None = None
    for bs in cand_list:
        if bs < current:
            next_bs = bs
            break
    if next_bs is None:
        return False

    # Update the dict in-place.
    train_opts[current_key] = next_bs
    exp["train_opts"] = train_opts
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
    if cmd_type == "bash":
        # Validate that the script referenced by cmd exists.
        # Handles both "/path/to/script.sh" and "bash /path/to/script.sh" formats.
        cmd_parts = shlex.split(exp["cmd"])
        if cmd_parts:
            first = Path(cmd_parts[0])
            # Skip leading shell interpreter (e.g. "bash /path/to/script.sh").
            if first.name in ("bash", "sh", "zsh") and len(cmd_parts) > 1:
                if cmd_parts[1] == "-c":
                    return  # inline command, no script to validate
                script_path = Path(cmd_parts[1])
            else:
                script_path = first
            if not script_path.is_absolute():
                script_path = Path.cwd() / script_path
            if not script_path.exists():
                raise ValueError(f"bash cmd script not found: {script_path}")
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


def _last_error_hash(run_log: Path) -> str:
    """Extract error fingerprint from log for agent dedup.

    Uses ``get_classification_text`` (tail+mid+low chunks, noise-filtered,
    lowercased) so the hash reflects the same cleaned view that
    ``classify_failure`` sees.  Avoids full-file scan for large logs.
    """
    try:
        if not run_log.exists():
            return ""
        text = get_classification_text(run_log)
        if not text:
            return ""
        lines: list[str] = []
        for line in text.splitlines():
            # Skip torchrun boilerplate that always contains "error".
            if "error_file" in line or "childfailederror" in line:
                continue
            if "error" in line or "traceback" in line or "exitcode" in line:
                # Strip per-rank prefixes (e.g. "[rank3]", "[rank0]:") so the
                # hash is stable across DDP configurations with different GPU
                # counts — torchrun interleaves per-rank output non-deterministically.
                stripped = re.sub(r"^\s*\[rank\d+\]\s*:?\s*", "", line.strip())
                lines.append(stripped)
        # Use last 20 error-bearing lines for a stable hash.
        return hashlib.md5("\n".join(lines[-20:]).encode(), usedforsecurity=False).hexdigest() if lines else ""
    except Exception:
        return ""


class RecoveryAction(enum.Enum):
    """Enumeration of recovery outcomes returned by ``RecoveryManager.handle_failure``.

    The caller inspects the action and performs the appropriate file writes +
    ``sys.exit()``.  Keeping file I/O in the caller (instead of buried inside
    ``handle_failure``) makes the three exit paths explicit and testable.
    """
    OOM_BACKOFF = "oom_backoff"
    HARD_FAILURE = "hard_failure"
    FAILED = "failed"


class FailureContext:
    """Parameter object for ``RecoveryManager.handle_failure``.

    Groups the seven arguments that were previously passed positionally so the
    call site is readable and the signature is stable across changes.
    """

    __slots__ = ("exp", "run_exp_path", "status_path", "attempt", "rc", "reason", "error_hash")

    def __init__(
        self,
        exp: Dict[str, Any],
        run_exp_path: Path,
        status_path: Path,
        attempt: int,
        rc: int,
        reason: str,
        error_hash: str,
    ):
        self.exp = exp
        self.run_exp_path = run_exp_path
        self.status_path = status_path
        self.attempt = attempt
        self.rc = rc
        self.reason = reason
        self.error_hash = error_hash


class AttemptContext:
    """Parameter object for ``_handle_outcome``.

    Replaces the 13 positional arguments that were previously passed to
    ``_handle_outcome`` so the call site is readable and the signature is
    stable across changes.
    """

    __slots__ = (
        "rc", "run_log", "exp", "run_root", "run_exp_path", "status_path",
        "attempt", "train_cmd", "env", "cmd_type", "nproc", "repo_root", "agent_timeout",
    )

    def __init__(
        self,
        rc: int,
        run_log: Path,
        exp: Dict[str, Any],
        run_root: Path,
        run_exp_path: Path,
        status_path: Path,
        attempt: int,
        train_cmd: List[str],
        env: Dict[str, str],
        cmd_type: str,
        nproc: int,
        repo_root: Path,
        agent_timeout: int,
    ):
        self.rc = rc
        self.run_log = run_log
        self.exp = exp
        self.run_root = run_root
        self.run_exp_path = run_exp_path
        self.status_path = status_path
        self.attempt = attempt
        self.train_cmd = train_cmd
        self.env = env
        self.cmd_type = cmd_type
        self.nproc = nproc
        self.repo_root = repo_root
        self.agent_timeout = agent_timeout


class RecoveryManager:
    """Manages training failure recovery: OOM backoff, agent repair, error dedup.

    Extracted from ``main()`` to keep the worker's execution and recovery
    concerns separate.  Handles three paths:

    * **OOM backoff** — reduces batch_size deterministically and returns
      ``RecoveryAction.OOM_BACKOFF`` so the scheduler retries immediately.
    * **Hard failure** — when the same error hash survives the agent too many
      times, returns ``RecoveryAction.HARD_FAILURE``.
    * **Agent dispatch** — invokes the LLM agent for first-seen errors, or
      skips it for duplicates below the hard-failure threshold.
    * **Failed** — returns ``RecoveryAction.FAILED`` and the caller writes the
      final ``"failed"`` status.

    The caller is responsible for all file writes and ``sys.exit()`` — the
    manager only mutates the in-memory *exp* dict and returns an action.
    """

    def __init__(self, run_root: Path, repo_root: Path, agent_timeout: int):
        self._run_root = run_root
        self._repo_root = repo_root
        self._agent_timeout = agent_timeout

    def handle_failure(self, ctx: FailureContext) -> tuple[RecoveryAction, Dict[str, Any]]:
        """Process a training failure.

        Returns ``(action, exp)`` where *action* is one of
        ``RecoveryAction.OOM_BACKOFF``, ``RecoveryAction.HARD_FAILURE``, or
        ``RecoveryAction.FAILED``, and *exp* is the (possibly agent-modified)
        experiment dict.  The caller must write *exp* to disk and call
        ``sys.exit(ctx.rc)``.

        Works on a shallow copy of ``ctx.exp`` — top-level keys (status,
        attempt, etc.) are independent, but nested containers (train_opts dict)
        are shared with the caller until overwritten.  The returned dict is
        always the authoritative version.
        """
        exp = dict(ctx.exp)  # shallow copy — top-level keys are independent

        # ---- OOM deterministic backoff (before agent) -----------------------
        if ctx.reason == "oom" and exp.get("oom_batch_candidates"):
            # Save train_opts before apply_oom_batch_backoff mutates it.
            # If backoff succeeds but we have exhausted oom retries, we must
            # roll back the mutation so the stale smaller batch_size doesn't
            # leak into the next attempt with a stale oom_backoff_count.
            train_opts_snapshot = dict(exp.get("train_opts", {}))
            if apply_oom_batch_backoff(exp):
                oom_count = int(exp.get("oom_backoff_count", 0)) + 1
                max_oom = int(exp.get("max_oom_retries", len(exp.get("oom_batch_candidates", [])) or 0))
                if oom_count <= max_oom:
                    exp["oom_backoff_count"] = oom_count
                    exp["attempt"] = ctx.attempt - 1
                    exp["consecutive_failures"] = 0
                    exp["status"] = "pending"
                    exp["updated_at"] = now_ts()
                    exp["error_hash"] = ctx.error_hash
                    exp["last_failed_at"] = str(time.time())
                    return RecoveryAction.OOM_BACKOFF, exp
                # oom_count > max_oom: rollback the batch_size mutation so we
                # don't leak a stale smaller value into agent / hard_failure.
                exp["train_opts"] = train_opts_snapshot
                exp.pop("last_oom_batch_size", None)
                # fall through to agent / hard_failure.
            else:
                # Already at minimum batch_size — track OOM-at-min to bound retries.
                # Without this counter a no-agent experiment (max_retries=0) OOMing
                # at batch_size=1 would retry forever.
                oom_at_min = int(exp.get("oom_at_min_count", 0)) + 1
                exp["oom_at_min_count"] = oom_at_min
                max_oom = int(exp.get("max_oom_retries", len(exp.get("oom_batch_candidates", [])) or 0))
                if max_oom > 0 and oom_at_min > max_oom:
                    exp["status"] = "hard_failure"
                    exp["updated_at"] = now_ts()
                    exp["last_reason"] = (
                        f"OOM at minimum batch_size persists after {oom_at_min} attempts"
                    )
                    return RecoveryAction.HARD_FAILURE, exp

        # ---- Reset OOM counters on non-OOM failures -----------------------
        # If the previous failure was OOM but this one is not, the OOM
        # backoff issues were resolved (manually or by the agent). Reset
        # counters so future OOMs get a fresh backoff cycle.
        else:
            if exp.get("oom_backoff_count") or exp.get("oom_at_min_count"):
                exp.pop("oom_backoff_count", None)
                exp.pop("oom_at_min_count", None)
                exp.pop("last_oom_batch_size", None)

        # ---- Agent dispatch (only when retries remain) --------------------
        if ctx.attempt <= int(exp.get("max_retries", 2)):
            agent_fix_hashes = exp.get("agent_fix_hashes", {})
            prev_hash = exp.get("error_hash", "")
            hard_limit = int(exp.get("hard_failure_threshold", HARD_FAILURE_THRESHOLD))

            if ctx.error_hash and ctx.error_hash == prev_hash:
                # Agent already ran for this hash — increment counter.
                agent_fix_hashes[ctx.error_hash] = int(agent_fix_hashes.get(ctx.error_hash, 0)) + 1
                exp["agent_fix_hashes"] = agent_fix_hashes
                agent_fix_count = agent_fix_hashes[ctx.error_hash]
                if agent_fix_count >= hard_limit:
                    exp["status"] = "hard_failure"
                    exp["updated_at"] = now_ts()
                    exp["last_reason"] = (
                        f"agent failed to fix '{ctx.reason}' "
                        f"after {agent_fix_count} attempts (hash={ctx.error_hash})"
                    )
                    return RecoveryAction.HARD_FAILURE, exp

                with open(self._run_root / "agent_skip.txt", "a", encoding="utf-8") as f:
                    f.write(
                        f"{now_ts()} skipped agent "
                        f"(duplicate error {ctx.error_hash}, agent_fix_count={agent_fix_count})\n"
                    )
            else:
                # First time seeing this error — run the agent.
                try:
                    snapshot_git(self._repo_root, self._run_root, "pre_agent")
                    agent_rc = run_agent(
                        self._run_root,
                        repo_root=self._repo_root,
                        timeout_seconds=self._agent_timeout,
                        agent_cli=exp.get("agent_cli", "claude"),
                        conda_env=exp.get("conda_env", "llm_train"),
                    )
                    snapshot_git(self._repo_root, self._run_root, "post_agent")
                    if agent_rc != 0:
                        with open(self._run_root / "agent_error.txt", "a", encoding="utf-8") as f:
                            f.write(
                                f"{now_ts()} agent exited non-zero (rc={agent_rc}) — "
                                f"fixes may be incomplete\n"
                            )
                    agent_fix_hashes[ctx.error_hash] = int(agent_fix_hashes.get(ctx.error_hash, 0)) + 1
                    # Re-read exp.json so agent edits (train_opts, etc.) are visible.
                    # IMPORTANT: if the re-read fails we keep the old exp AND log a
                    # warning — previously we silently clobbered agent fixes on error.
                    try:
                        exp = read_json(ctx.run_exp_path)
                    except Exception as read_exc:
                        with open(self._run_root / "agent_error.txt", "a", encoding="utf-8") as f:
                            f.write(
                                f"{now_ts()} WARNING: failed to re-read exp.json after agent — "
                                f"agent fixes may be lost: {repr(read_exc)}\n"
                            )
                    exp.setdefault("agent_fix_hashes", {}).update(agent_fix_hashes)
                    # Reset last_failed_at so the retry cooldown starts from
                    # AFTER the agent fix, not from the original crash time.
                    exp["last_failed_at"] = str(time.time())
                except Exception as exc:
                    with open(self._run_root / "agent_error.txt", "a", encoding="utf-8") as f:
                        f.write(f"{now_ts()} {repr(exc)}\n")

        return RecoveryAction.FAILED, exp


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
    try:
        ensure_exp_sane(exp)
    except ValueError as exc:
        # Validation errors are not retryable — mark hard_failure immediately
        # so the scheduler doesn't loop forever on a misconfigured experiment.
        exp["status"] = "hard_failure"
        exp["last_reason"] = f"validation_error: {exc}"
        exp["updated_at"] = now_ts()
        atomic_write_json(run_exp_path, exp)
        atomic_write_json(
            run_root / "status.json",
            {
                "status": "hard_failure",
                "updated_at": now_ts(),
                "attempt": 0,
                "exit_code": 1,
                "reason": f"validation_error: {exc}",
            },
        )
        log_event(source="worker", event="hard_failure", error=str(exc))
        sys.exit(1)

    global _worker_lock
    # Register SIGTERM handler BEFORE acquiring the lock so the handler can
    # release it even if SIGTERM arrives between lock.acquire() and the
    # handler registration below.
    signal.signal(signal.SIGTERM, _sigterm_handler)

    lock = Lock(run_root / ".lock_worker", stale_seconds=12 * 3600)
    if not lock.acquire():
        log_event(source="worker", event="lock_busy", run_root=str(run_root))
        sys.exit(2)
    _worker_lock = lock

    try:
        cmd_type = exp.get("cmd_type", "torchrun")
        nproc = int(exp.get("nproc", 4))
        # Resolve master_port once for both the training command and status.json.
        # _prepare_environment will pick up the same value via resolve_master_port.
        if cmd_type != "bash":
            exp["master_port"] = resolve_master_port(exp)

        attempt, attempt_dir, status_path = _init_attempt(run_root, exp, str(queue_exp_path))

        env, train_cmd = _prepare_environment(exp, repo_root, attempt)

        run_log = attempt_dir / "run.log"
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
            # Wall-clock timeout caps total run time; hang_timeout kills the process
            # if the log file stops growing (GPU deadlock, NCCL hang, etc.).
            rc = run_cmd(
                train_cmd,
                run_log,
                env,
                timeout_seconds=exp.get("train_timeout", 60 * 60),
                hang_timeout=exp.get("hang_timeout", 3600),
            )
        except KeyboardInterrupt:
            # Treat Ctrl+C as a "failed" attempt so autopipe can record status and
            # optionally trigger the auto-repair agent.
            with open(run_log, "ab", buffering=0) as f:
                f.write(f"\n==== {now_ts()} INTERRUPTED (KeyboardInterrupt)\n".encode())
            rc = 130

        _handle_outcome(AttemptContext(
            rc=rc, run_log=run_log, exp=exp, run_root=run_root,
            run_exp_path=run_exp_path, status_path=status_path,
            attempt=attempt, train_cmd=train_cmd, env=env,
            cmd_type=cmd_type, nproc=nproc, repo_root=repo_root,
            agent_timeout=args.agent_timeout,
        ))
    finally:
        lock.release()


def _handle_outcome(ctx: AttemptContext) -> None:
    """Process the training result.  Always ends with ``sys.exit()``.

    Status is written to ``status.json`` exactly once — classification and
    recovery happen before any disk write, eliminating the previous race
    window where two writes could leave stale intermediate state on disk.
    """
    if ctx.rc == 0:
        atomic_write_json(
            ctx.status_path,
            {"status": "success", "updated_at": now_ts(), "attempt": ctx.attempt},
        )
        ctx.exp["status"] = "success"
        ctx.exp["updated_at"] = now_ts()
        ctx.exp["consecutive_failures"] = 0
        ctx.exp["error_hash"] = ""
        ctx.exp["last_failed_at"] = ""
        atomic_write_json(ctx.run_exp_path, ctx.exp)
        sys.exit(0)

    # ---- Failure classification (BEFORE any disk writes) ----------------
    if ctx.rc == 130:
        reason = "interrupted"
    elif ctx.rc == 124:
        reason = "timeout"
    elif ctx.rc < 0:
        # Negative exit code = killed by signal (e.g. -9 = SIGKILL, -15 = SIGTERM).
        # POSIX: exit code = -signum.  Already handled above for 130/124, but
        # catch other signals (SIGKILL, SIGSEGV, SIGBUS, etc.) here.
        reason = "killed"
    else:
        reason = classify_failure(ctx.run_log)
    error_hash = _last_error_hash(ctx.run_log)

    # ---- Recovery (decides final status) ---------------------------------
    recovery = RecoveryManager(ctx.run_root, ctx.repo_root, ctx.agent_timeout)
    fctx = FailureContext(
        exp=ctx.exp, run_exp_path=ctx.run_exp_path, status_path=ctx.status_path,
        attempt=ctx.attempt, rc=ctx.rc, reason=reason, error_hash=error_hash,
    )
    action, exp = recovery.handle_failure(fctx)

    # ---- Write status.json EXACTLY ONCE ----------------------------------
    if action == RecoveryAction.OOM_BACKOFF:
        atomic_write_json(ctx.run_exp_path, exp)
        atomic_write_json(
            ctx.status_path,
            {
                "status": "pending", "updated_at": now_ts(),
                "attempt": exp["attempt"],
                "reason": "oom_backoff",
                "oom_backoff_count": exp.get("oom_backoff_count", 0),
                "error_hash": error_hash,
            },
        )
        sys.exit(ctx.rc)

    if action == RecoveryAction.HARD_FAILURE:
        atomic_write_json(ctx.run_exp_path, exp)
        atomic_write_json(
            ctx.status_path,
            {
                "status": "hard_failure", "updated_at": now_ts(),
                "attempt": ctx.attempt, "exit_code": ctx.rc,
                "reason": exp.get("last_reason", f"hard_failure:{reason}"),
                "error_hash": error_hash,
            },
        )
        sys.exit(ctx.rc)

    # RecoveryAction.FAILED — single write with full metadata.
    exp["status"] = "failed"
    exp["updated_at"] = now_ts()
    exp["last_exit_code"] = ctx.rc
    exp["last_reason"] = reason
    exp["error_hash"] = error_hash
    exp["consecutive_failures"] = int(exp.get("consecutive_failures", 0)) + 1
    exp["last_failed_at"] = str(time.time())
    atomic_write_json(ctx.run_exp_path, exp)

    atomic_write_json(
        ctx.status_path,
        {
            "status": "failed",
            "updated_at": now_ts(),
            "attempt": ctx.attempt,
            "exit_code": ctx.rc,
            "reason": reason,
            "train_cmd": ctx.train_cmd,
            "cuda_visible_devices": ctx.env.get("CUDA_VISIBLE_DEVICES"),
            "hf_endpoint": ctx.env.get("HF_ENDPOINT"),
            "nproc": ctx.nproc,
            "master_port": exp.get("master_port", 0),
            "error_hash": error_hash,
        },
    )
    sys.exit(ctx.rc)


if __name__ == "__main__":
    main()
