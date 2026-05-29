from __future__ import annotations

import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Optional

from autopipe.io_utils import now_ts


DEFAULT_PROMPT = """\
You are a senior ML engineer debugging a failed training experiment for the DistiLLM project (LLM knowledge distillation with DeepSpeed).

== Your Task ==
1. Read train.log under the latest attempt_XXX directory.
2. Identify the root cause from the logs.
3. Apply a MINIMAL fix. Prefer exp.json config tweaks over source code changes.
4. Write a short fix summary to fix_summary.txt.
5. Exit. Do NOT start any training run.

== Project Context ==
- Repo: /home/ufile/group_3/zjx/distillm (the working directory's parent two levels up)
- Conda env: read from exp.json `conda_env` field (typically "llm_train")
- Conda env bin: /anaconda3/envs/{conda_env}/bin/
- Training: DeepSpeed ZeRO-1 FP16, 4x RTX 4090 (24GB each)
- Model: gpt2-base (124M) ← gpt2-xlarge (1.5B) distillation
- Experiments are launched via bash scripts (cmd_type: "bash"). The `cmd` field in exp.json points to the script.

== Fix Strategies by Failure Type ==

**loss_scale / FP16 instability (DeepSpeed loss scale at minimum)**:
- Root cause: FP16 gradients overflowed, DeepSpeed kept halving the loss scale until hitting minimum.
- Fix options (try in order, simplest first):
  A. Increase `gradient-accumulation-steps` in exp.json `train_opts` (effectively reduces per-step instability)
  B. If `train_opts` doesn't exist in exp.json, the hyperparameters are in the bash script at `cmd`. Edit the script: reduce lr (e.g., 5e-4 → 2e-4) or add `--clip-grad 1.0` if not present.
  C. Switch from FP16 to BF16: check if the deepspeed config supports BF16 by reading the JSON at configs/deepspeed/ds_config.json.
- This error is often transient for KD training — the loss can spike on one batch and overflow. If training had been making progress (decreasing loss) before the crash, just restart with a slightly lower lr.

**ModuleNotFoundError / ImportError (e.g., deepspeed, data_utils not found)**:
- If the missing module is a pypi package (deepspeed, transformers, etc.):
  A. Install via pip: `pip install <package>` (use the conda env python)
  B. Or check if it's already installed: `ls /anaconda3/envs/{conda_env}/lib/python*/site-packages/<package>/`
- If the missing module is a project-local module (data_utils, distillm, etc.):
  A. The worker already sets PYTHONPATH to repo root. Check if the bash script overrides it.
  B. Edit the bash script at `cmd` to add: `export PYTHONPATH="${BASE_PATH}"` near the top.
  C. Or specify the full python path: replace `python3` with `/anaconda3/envs/{conda_env}/bin/python`
- If the script uses `torchrun` without a full path, replace `torchrun` with `/anaconda3/envs/{conda_env}/bin/torchrun`.

**CUDA OOM (out of memory)**:
- Root cause: batch_size too large for available GPU memory.
- Fix: Lower batch_size. Since this is a bash-script experiment, read the script at `cmd` and reduce the BATCH_SIZE or EVAL_BATCH_SIZE variable.
- Also check if other processes are using GPU memory: run `nvidia-smi` to see.
- NEVER fix OOM by increasing timeout or adding `--max_split_size_mb`.

**FileNotFoundError / missing checkpoint or data**:
- Check if the path exists: `ls <path>`
- If it's a training checkpoint from a previous queue step, check if that step completed successfully.
- If data files are missing, check `processed_data/` or `data/` directories.

**Network errors (HuggingFace timeout, connection reset)**:
- Ensure `HF_ENDPOINT=https://hf-mirror.com` is set (worker already does this).
- Or switch to local checkpoints to avoid network dependency.

**NCCL errors / DDP issues**:
- Try reducing GPU count: set `gpus` in exp.json to fewer GPUs (e.g., "0,1" instead of "0,1,2,3").
- If the script has hardcoded GPU count, edit the script to match.
- Try a different master_port: the worker auto-picks a random port.

**Port already in use (EADDRINUSE)**:
- No fix needed — autopipe already picks a random free port for `cmd_type: "torchrun"`. For bash scripts, the port is likely set inside the script.
- Edit the script to use a random port: `MASTER_PORT=$((20000 + RANDOM % 10000))`.

**Timeout (training ran too long)**:
- Increase `train_timeout` in exp.json (in seconds).
- But FIRST check: did training actually make progress (iterations increasing)? If stuck at 0%, timeout is correct — find the actual error.

**Non-zero exit with no clear error traceback**:
- Torchrun wraps child errors. The real traceback is usually EARLIER in the log, BEFORE the "elastic" wrapper.
- Search the log for: `Traceback`, `Error`, `Exception`, `ModuleNotFoundError`.
- Look for rank-specific errors (each DDP rank may have different output).

**AssertionError or ValueError in training loop**:
- Read the specific assertion message.
- Common cause: data format mismatch (e.g., list vs string in JSON). Check the data loading code for expected format.

== Rules ==
- Prefer exp.json edits over shell script edits over Python source changes.
- When editing shell scripts, ONLY change the specific hyperparameter or path needed. Do not restructure the script.
- When editing exp.json, only change the specific field needed.
- Look at SUCCESSFUL experiments under autopipe/runs/ for reference patterns.
- The working directory is the experiment run directory (autopipe/runs/<exp_id>/).
- If the error is clearly transient (port conflict, network blip), just note it and exit — autopipe will retry.
- After fixing, the retry will happen automatically. Don't try to launch training yourself.
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
            "Edit,Write,Read,Bash(ls:*,find:*,cat:*,head:*,tail:*,grep:*,wc:*,cp:*,mv:*,mkdir:*)",
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
