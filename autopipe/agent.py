from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
from pathlib import Path

from autopipe.io_utils import log_event, now_ts


AGENT_NAME = "distillm_debugger"

TASK_PROMPT = """\
Fix the training failure in this experiment directory. Steps:
1. Read status.json, then attempt_N/run.log (and previous attempts if relevant).
2. Find the root cause from the traceback (for torchrun: look BEFORE ChildFailedError).
3. Apply the smallest fix. Prefer editing exp.json "train_opts" (worker exports as TRAIN_* env vars). Then edit shell scripts, install packages, fix data, or edit Python source.
4. Write a one-line fix summary to fix_summary.txt.
5. Exit. Do NOT start training."""


def _build_system_prompt(repo_root: str, conda_env: str) -> str:
    """Build the agent system prompt with project context injected.

    NOTE: This prompt intentionally omits hardcoded model/GPU details (unlike
    earlier versions).  The agent should derive hardware and model information
    from the actual logs, exp.json, and shell scripts — not from a cached
    prompt that may go stale when experiments change.
    """
    return (
        "You are a senior ML engineer debugging failed training experiments for "
        "the DistiLLM project (LLM knowledge distillation with DeepSpeed). "
        "You have FULL autonomy to read logs, diagnose root causes, AND apply fixes.\n\n"
        "PROJECT CONTEXT:\n"
        f"- Repo root: {repo_root}\n"
        f"- Conda env: {conda_env}\n"
        "- Training uses DeepSpeed ZeRO with FP16 mixed precision\n"
        "- Experiments use cmd_type bash - the cmd field in exp.json points to the shell script\n"
        "- Worker exports exp.json train_opts as TRAIN_* env vars before running scripts\n"
        "- Shell scripts read TRAIN_LR, TRAIN_BATCH_SIZE, TRAIN_EPOCHS, TRAIN_GRADIENT_ACCUMULATION_STEPS with fallback defaults\n"
        "- Data: processed_data/, checkpoints: checkpoints/, results: results/\n\n"
        "FIX PRINCIPLES:\n"
        "- First: edit exp.json -> train_opts dict (lr, batch_size, epochs, gradient_accumulation_steps). "
        "Worker exports as TRAIN_* env vars. Fixes persist across retries.\n"
        "- Then: shell script hyperparameters -> install packages -> free disk -> fix data -> Python source\n"
        "- Make surgical edits. Do not restructure files.\n"
        "- Trust the traceback. Read the actual error.\n"
        "- If error is transient (port conflict, network, killed), note it and exit.\n"
        "- Check agent.log / fix_summary.txt before fixing - don't repeat failed fixes."
    )


def _build_agent_spec(repo_root: str, conda_env: str) -> str:
    """Build JSON agent spec for the --agents CLI flag."""
    system_prompt = _build_system_prompt(repo_root, conda_env)
    return json.dumps({AGENT_NAME: {"description": "DistiLLM training failure debugger", "prompt": system_prompt}})


def _resolve_agent(agent_cli: str = "auto") -> str:
    """Determine which agent CLI to use. Returns 'claude' or 'codex'.

    Raises RuntimeError if the requested CLI is not found or is not executable
    (e.g., ``shutil.which`` found a broken symlink, or the binary is corrupt).
    """

    def _check_binary(name: str) -> str:
        path = shutil.which(name)
        if path is None:
            raise RuntimeError(f"agent_cli='{name}' but '{name}' CLI not found on PATH")
        # Verify the binary is executable (detect broken symlinks, non-executable
        # files that shutil.which might still return on some systems).
        if not os.access(path, os.X_OK):
            raise RuntimeError(
                f"agent_cli='{name}' found at '{path}' but is not executable"
            )
        return name

    def _check_binary_graceful(name: str) -> bool:
        """Return True if *name* is on PATH and executable, False otherwise."""
        path = shutil.which(name)
        if path is None:
            return False
        try:
            return os.access(path, os.X_OK)
        except Exception:
            return False

    if agent_cli == "claude":
        return _check_binary("claude")
    if agent_cli == "codex":
        return _check_binary("codex")
    # auto: probe for available CLIs, prefer claude
    if _check_binary_graceful("claude"):
        return "claude"
    if _check_binary_graceful("codex"):
        return "codex"
    raise RuntimeError("No agent CLI found on PATH (tried: claude, codex). "
                       "Install one or pass agent_cli='claude'/'codex' explicitly.")


def run_agent(
    exp_dir: Path,
    repo_root: Path,
    timeout_seconds: int = 600,
    sandbox: str = "danger-full-access",
    agent_cli: str = "auto",
    conda_env: str = "llm_train",
) -> int:
    """Run agent CLI in exp_dir. Returns exit code."""
    cli = _resolve_agent(agent_cli)

    if cli == "claude":
        agent_spec = _build_agent_spec(str(repo_root), conda_env)
        cmd = [
            "claude",
            "-p",
            "--no-session-persistence",
            "--dangerously-skip-permissions",
            "--add-dir", str(repo_root),
            "--agents", agent_spec,
            "--agent", AGENT_NAME,
            "--allowedTools",
            "Glob,Grep,Edit,Write,Read,Bash(ls:*,find:*,cat:*,head:*,tail:*,grep:*,rg:*,wc:*,cp:*,mv:*,mkdir:*,rm:*,rmdir:*,pip:*,pip3:*,python:*,python3:*,df:*,du:*,nvidia-smi:*,conda:*,git:*)",
            "-",  # read task from stdin
        ]
    else:
        # codex: inject project context directly into the prompt (same context
        # that the claude path gets via --agents).
        # NOTE: The full prompt appears as a positional argument and is visible
        # in /proc/*/cmdline.  The claude path passes via stdin, which is more
        # private.  This is an inherent codex CLI design limitation.
        system_prompt = _build_system_prompt(str(repo_root), conda_env)
        full_prompt = system_prompt + "\n\nTASK:\n" + TASK_PROMPT
        cmd = [
            "codex",
            "exec",
            f"--sandbox={sandbox}",
            "--dangerously-bypass-approvals-and-sandbox",
            full_prompt,
        ]

    log_path = exp_dir / "agent.log"
    with open(log_path, "ab", buffering=0) as f:
        f.write(f"\n==== {now_ts()} AGENT_START ({cli}): {' '.join(shlex.quote(str(x)) for x in cmd)}\n".encode())
        # buffering=0 above already ensures unbuffered writes; no need for flush()
        try:
            use_text = cli == "claude"
            completed = subprocess.run(
                cmd,
                cwd=str(exp_dir),
                stdout=f,
                stderr=subprocess.STDOUT,
                timeout=timeout_seconds,
                check=False,
                input=TASK_PROMPT if use_text else None,
                text=use_text,
            )
            return int(completed.returncode)
        except subprocess.TimeoutExpired:
            f.write(f"\n==== {now_ts()} AGENT_TIMEOUT after {timeout_seconds}s\n".encode())
            return 124
        except (OSError, ValueError, UnicodeError) as exc:
            f.write(f"\n==== {now_ts()} AGENT_IO_ERROR: {repr(exc)}\n".encode())
            return 1


