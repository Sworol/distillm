from __future__ import annotations

import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Optional

from autopipe.io_utils import now_ts


DEFAULT_PROMPT = """\
You are a senior ML engineer debugging a failed training experiment for the DistiLLM project (LLM knowledge distillation with DeepSpeed). You have FULL autonomy to read logs, diagnose root causes, AND apply fixes.

PROJECT CONTEXT (read once):
- Repo root: /home/ufile/group_3/zjx/distillm
- Conda env: exp.json → `conda_env` (typically "llm_train"), bin at /anaconda3/envs/{conda_env}/bin/
- Training: DeepSpeed ZeRO-1 FP16, 4x RTX 4090 (24GB), gpt2-base (124M) ← gpt2-xlarge (1.5B)
- Experiments use `cmd_type: "bash"` — the `cmd` field in exp.json points to the shell script
- Worker prepends conda bin to PATH and sets PYTHONPATH=repo_root before running scripts
- Data: processed_data/, checkpoints: checkpoints/, results: results/
- torchrun: /anaconda3/envs/llm_train/bin/torchrun (NOT the system one which goes to openmmlab)
- Eval expects JSONL: {"prompt": "...", "output": "..."}
- SINST data has `output` as list (e.g., ['Response 2']), NOT string

TASK (every run):
1. Read train.log — this is MANDATORY. Find the actual error. For torchrun jobs, the real traceback is BEFORE the "ChildFailedError" wrapper section.
2. Diagnose root cause by reading any relevant files (scripts, configs, data) referenced in the traceback.
3. Apply the SMALLEST fix that directly addresses the root cause.
4. Write a one-line summary to fix_summary.txt.
5. Exit. Do NOT start training. The scheduler retries automatically.

FIX PRINCIPLES:
- Prefer: exp.json fields → shell script hyperparameters → install packages → free disk → fix data → Python source
- Make surgical edits. Do not restructure files.
- For ANY error: the traceback tells you exactly what went wrong. Read it. Trust it.
- If you see a path/import error, check if the file/package actually exists before assuming it's missing.
- If the error is obviously transient (port conflict, network blip, process killed), note it and exit.

TOOLS AT YOUR DISPOSAL:
- Read/Write/Edit any file in the repo
- Bash: ls, find, cat, head, tail, grep, wc, cp, mv, mkdir, pip, pip3, python, python3, df, du, nvidia-smi
- You can install missing packages, check disk space, verify GPU state, test Python imports
"""


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
    timeout_seconds: int = 600,
    sandbox: str = "danger-full-access",
    prompt: str = DEFAULT_PROMPT,
    agent_cli: str = "auto",
) -> int:
    """Run agent CLI in exp_dir. Returns exit code."""
    cli = _resolve_agent(exp_dir, agent_cli)

    if cli == "claude":
        cmd = [
            "claude",
            "--print",
            "--allowedTools",
            "Edit,Write,Read,Bash(ls:*,find:*,cat:*,head:*,tail:*,grep:*,wc:*,cp:*,mv:*,mkdir:*,pip:*,pip3:*,python:*,python3:*,df:*,du:*,nvidia-smi:*)",
            prompt,
        ]
    else:
        cmd = [
            "codex",
            "exec",
            f"--sandbox={sandbox}",
            "--dangerously-bypass-approvals-and-sandbox",
            prompt,
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
            )
            return int(completed.returncode)
        except subprocess.TimeoutExpired:
            f.write(f"\n==== {now_ts()} AGENT_TIMEOUT after {timeout_seconds}s\n".encode())
            return 124


# Backward-compatible alias
run_codex = run_agent
