from __future__ import annotations

import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Optional

from autopipe.io_utils import now_ts


DEFAULT_PROMPT = """\
You are a senior ML engineer debugging a failed training+visualization experiment.

== Your Task ==
1. Read train.log and vis.log (if present) under the latest attempt_XXX directory.
2. Identify the root cause from the logs.
3. Apply a MINIMAL fix. Prefer exp.json config tweaks over source code changes.
4. Write a short fix summary to fix_summary.txt.
5. Exit. Do NOT start any training run.

== Fix Strategies by Failure Type ==

**CUDA OOM (out of memory)**:
- Root cause: batch_size too large for available GPU memory.
- Fix: Update exp.json `oom_batch_candidates`. Pick the NEXT SMALLER value (e.g., if current batch=16, try [8,4,2]). Lower the first entry in `train_opts` for `trainer.data.batch_size` to match.
- Also consider: reduce `nproc` if multi-GPU (less parallelism = less per-GPU memory per worker), or reduce image size via `data.resize_shape`.
- NEVER fix OOM by increasing timeout or adding `--max_split_size_mb`. Those are not root-cause fixes.

**FileNotFoundError / missing pretrained weights**:
- Check if the file exists at the given path. If it's a timm/huggingface model:
  - Option A: Download via HF mirror: `HF_ENDPOINT=https://hf-mirror.com huggingface-cli download ...`
  - Option B: Set `pretrained=False` and point `checkpoint_path` to a local .pth file in `model/pretrain/`.
  - Option C: Use torchvision's built-in weights (e.g., `torchvision.models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)`).
- Check existing files in `model/pretrain/` first — the weights may already be there under a different name.

**state_dict mismatch / Missing keys in state_dict**:
- The checkpoint file exists but model architecture doesn't match.
- If `strict=False` is already set, this is just a warning — the model will train from scratch on missing layers.
- If `strict=True`, change to `strict=False` in the config (as a CLI override in train_opts), or fix the checkpoint_path to point to the correct architecture.

**ModuleNotFoundError / ImportError**:
- Missing pip/conda package. Install it.
- If the import is inside a try/except fallback, make sure the fallback path works.

**Network errors (HuggingFace timeout, connection reset)**:
- Ensure `HF_ENDPOINT=https://hf-mirror.com` is set (already in exp.json env).
- Or switch to local pretrained weights to avoid network dependency entirely.

**NCCL errors / DDP issues**:
- Try reducing `nproc` in exp.json.
- Try adding `trainer.find_unused_parameters=True` to train_opts.
- Try setting `master_port=auto` (autopipe picks a free port).

**Port already in use (EADDRINUSE)**:
- No fix needed — autopipe already picks a random free port on retry.
- Just note in fix_summary.txt and exit cleanly.

**Timeout (training ran too long)**:
- Increase `train_timeout` in exp.json (in seconds).
- But FIRST check: did training actually make progress (iterations increasing)? If it was stuck at 0%, timeout is correct behavior.

**ValueError: not enough values to unpack (expected 2, got 1)**:
- A `train_opts` entry doesn't have `key=value` format (e.g., a comment-like string without `=`).
- Check train_opts in exp.json, remove or fix any entry that isn't a valid `key=value` pair.

== Rules ==
- Prefer exp.json edits over source code changes.
- When editing exp.json, only change the specific field needed. Do NOT restructure the entire file.
- Look at SUCCESSFUL experiments under autopipe/runs/ for reference patterns.
- Do NOT modify training config .py files unless absolutely necessary (they affect other experiments).
- The working directory is the experiment run directory. Repo root is two levels up.
- If the error is clearly transient (port conflict, network blip), just note it and exit — autopipe will retry.
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
