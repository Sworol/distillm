from __future__ import annotations

import json
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Optional

from autopipe.io_utils import now_ts


AGENT_NAME = "distillm_debugger"

TASK_PROMPT = """\
Fix the training failure in this experiment directory. Steps:
1. Read status.json, then attempt_N/train.log (and previous attempts if relevant).
2. Find the root cause from the traceback (for torchrun: look BEFORE ChildFailedError).
3. Apply the smallest fix. Prefer editing exp.json "train_opts" (worker exports as TRAIN_* env vars). Then edit shell scripts, install packages, fix data, or edit Python source.
4. Write a one-line fix summary to fix_summary.txt.
5. Exit. Do NOT start training."""


def _build_agent_spec(repo_root: str, conda_env: str) -> str:
    """Build JSON agent spec with project context injected into the system prompt."""
    system_prompt = (
        "You are a senior ML engineer debugging failed training experiments for "
        "the DistiLLM project (LLM knowledge distillation with DeepSpeed). "
        "You have FULL autonomy to read logs, diagnose root causes, AND apply fixes.\n\n"
        "PROJECT CONTEXT:\n"
        f"- Repo root: {repo_root}\n"
        f"- Conda env: {conda_env}\n"
        "- Training: DeepSpeed ZeRO-1 FP16, 4x RTX 4090 (24GB), gpt2-base (124M) <- gpt2-xlarge (1.5B)\n"
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
    return json.dumps({AGENT_NAME: {"description": "DistiLLM training failure debugger", "prompt": system_prompt}})


def _resolve_agent(exp_dir: Path, agent_cli: str = "auto") -> str:
    """Determine which agent CLI to use. Returns 'claude' or 'codex'."""
    if agent_cli == "claude":
        return "claude"
    if agent_cli == "codex":
        return "codex"
    # auto: probe for available CLIs, prefer claude
    if shutil.which("claude"):
        return "claude"
    if shutil.which("codex"):
        return "codex"
    # Fallback to claude (will fail fast with clear error if missing)
    return "claude"


def run_agent(
    exp_dir: Path,
    repo_root: Path,
    timeout_seconds: int = 600,
    sandbox: str = "danger-full-access",
    agent_cli: str = "auto",
    conda_env: str = "llm_train",
) -> int:
    """Run agent CLI in exp_dir. Returns exit code."""
    cli = _resolve_agent(exp_dir, agent_cli)
    agent_spec = _build_agent_spec(str(repo_root), conda_env)

    if cli == "claude":
        cmd = [
            "claude",
            "-p",
            "--no-session-persistence",
            "--permission-mode", "bypassPermissions",
            "--add-dir", str(repo_root),
            "--agents", agent_spec,
            "--agent", AGENT_NAME,
            "--allowedTools",
            "Edit,Write,Read,Bash(ls:*,find:*,cat:*,head:*,tail:*,grep:*,wc:*,cp:*,mv:*,mkdir:*,pip:*,pip3:*,python:*,python3:*,df:*,du:*,nvidia-smi:*)",
            "-",  # read task from stdin
        ]
    else:
        cmd = [
            "codex",
            "exec",
            f"--sandbox={sandbox}",
            "--dangerously-bypass-approvals-and-sandbox",
            TASK_PROMPT,
        ]

    log_path = exp_dir / "agent.log"
    with open(log_path, "ab", buffering=0) as f:
        f.write(f"\n==== {now_ts()} AGENT_START ({cli}): {' '.join(shlex.quote(str(x)) for x in cmd)}\n".encode())
        f.flush()
        try:
            completed = subprocess.run(
                cmd,
                cwd=str(exp_dir),
                stdout=f,
                stderr=subprocess.STDOUT,
                timeout=timeout_seconds,
                check=False,
                input=TASK_PROMPT if cli == "claude" else None,
                text=True if cli == "claude" else False,
            )
            return int(completed.returncode)
        except subprocess.TimeoutExpired:
            f.write(f"\n==== {now_ts()} AGENT_TIMEOUT after {timeout_seconds}s\n".encode())
            return 124


# Backward-compatible alias
run_codex = run_agent
