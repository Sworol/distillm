from __future__ import annotations

import argparse
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, List

from autopipe.config import HARD_FAILURE_THRESHOLD, default_paths
from autopipe.io_utils import atomic_write_json, now_ts


def distillm_specs(base: str | None = None) -> List[Dict[str, Any]]:
    """Defines all DistiLLM baseline experiments in execution order.

    Args:
        base: Repo root path. If None, inferred from this file's location.
    """

    if base is None:
        base = str(Path(__file__).resolve().parent.parent)
    BASE = base
    specs: List[Dict[str, Any]] = []

    # ============================================================
    # 1. KD Baseline: forward KL distillation
    # ============================================================
    specs.append(dict(
        key="kd_train",
        cmd=f"{BASE}/scripts/run_kd_multitask.sh",
        conda_env="llm_train",
        gpus="0,1,2,3",
        train_timeout=86400,  # 24h
        skip_vis=True,
        retry_sleep=60,
        max_retries=1,
        train_opts=dict(lr=0.0005, batch_size=8, epochs=20, gradient_accumulation_steps=1),
        oom_batch_candidates=[8, 4, 2, 1],
    ))

    specs.append(dict(
        key="kd_eval",
        cmd=f"{BASE}/scripts/run_eval_kd_multitask.sh",
        conda_env="llm_train",
        gpus="0",  # single GPU eval
        train_timeout=86400,
        skip_vis=True,
        retry_sleep=60,
        max_retries=1,
    ))

    # ============================================================
    # 2. SeqKD: teacher pseudo-label generation
    # ============================================================
    specs.append(dict(
        key="seqkd_gen",
        cmd=f"{BASE}/scripts/gpt2/tools/generate_data_seqkd_multitask.sh",
        conda_env="llm_train",
        gpus="0,1,2,3",
        train_timeout=172800,  # 48h for 60K generation
        skip_vis=True,
        retry_sleep=60,
        max_retries=0,
    ))

    specs.append(dict(
        key="seqkd_process",
        cmd=f"{BASE}/scripts/process_seqkd_data.sh",
        conda_env="llm_train",
        gpus="",  # CPU only
        train_timeout=3600,
        skip_vis=True,
        retry_sleep=60,
        max_retries=1,
    ))

    specs.append(dict(
        key="seqkd_train",
        cmd=f"{BASE}/scripts/gpt2/seqkd/seqkd_multitask_base.sh",
        conda_env="llm_train",
        gpus="0,1,2,3",
        train_timeout=86400,
        skip_vis=True,
        retry_sleep=60,
        max_retries=1,
        train_opts=dict(lr=0.0005, batch_size=2, epochs=20, gradient_accumulation_steps=1),
        oom_batch_candidates=[2, 1],
    ))

    specs.append(dict(
        key="seqkd_eval",
        cmd=f"{BASE}/scripts/run_eval_seqkd_multitask.sh",
        conda_env="llm_train",
        gpus="0",
        train_timeout=86400,
        skip_vis=True,
        retry_sleep=60,
        max_retries=1,
    ))

    # ============================================================
    # 3. MiniLLM: PPO-based baseline
    # ============================================================
    specs.append(dict(
        key="minillm_train",
        cmd=f"{BASE}/scripts/gpt2/minillm/train_multitask_base_xl.sh",
        conda_env="llm_train",
        gpus="0,1,2,3",
        train_timeout=172800,  # 48h for PPO training
        skip_vis=True,
        retry_sleep=60,
        max_retries=1,
        train_opts=dict(lr=5e-6, batch_size=8, epochs=10, total_iters=5000,
                        gradient_accumulation_steps=1, chunk_size=16),
        oom_batch_candidates=[8, 4, 2, 1],
    ))

    specs.append(dict(
        key="minillm_eval",
        cmd=f"{BASE}/scripts/run_eval_minillm_multitask.sh",
        conda_env="llm_train",
        gpus="0",
        train_timeout=86400,
        skip_vis=True,
        retry_sleep=60,
        max_retries=1,
    ))

    return specs


def _extract_script_path(exp: Dict[str, Any]) -> Path | None:
    """Return the Path to the bash script referenced by *exp*, or None."""
    if exp.get("cmd_type") != "bash":
        return None
    parts = exp["cmd"].split()
    return Path(parts[0]) if parts else Path(exp["cmd"])


def build_exp(spec: Dict[str, Any], seq: int) -> Dict[str, Any]:
    exp_id = f"{spec['key']}_{uuid.uuid4().hex[:8]}"
    return {
        "exp_id": exp_id,
        "seq": seq,  # execution order
        "key": spec["key"],
        "cmd_type": "bash",
        "cmd": spec["cmd"],
        "created_at": now_ts(),
        "status": "pending",
        "attempt": 0,
        "max_retries": spec.get("max_retries", 2),
        "retry_sleep": spec.get("retry_sleep", 60),
        "gpus": spec.get("gpus", "0,1,2,3"),
        "train_timeout": spec.get("train_timeout", 86400),
        "skip_vis": spec.get("skip_vis", True),
        "conda_env": spec.get("conda_env", "llm_train"),
        "hard_failure_threshold": spec.get("hard_failure_threshold", HARD_FAILURE_THRESHOLD),
        "train_opts": spec.get("train_opts", {}),  # hyperparams that agent can edit
        "oom_batch_candidates": spec.get("oom_batch_candidates", []),  # batch sizes to try on OOM
        "max_oom_retries": spec.get("max_oom_retries", len(spec.get("oom_batch_candidates", []))),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=".", help="Path to repo root (default: .)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Validate script paths only; do not generate queue files.")
    args = ap.parse_args()

    repo_root = Path(args.repo_root).resolve()
    paths = default_paths(repo_root)

    specs = distillm_specs(base=str(repo_root))
    errors = 0
    for i, spec in enumerate(specs):
        exp = build_exp(spec, seq=i + 1)
        script_path = _extract_script_path(exp)
        if script_path is not None and not script_path.exists():
            print(f"ERROR: [{spec['key']}] script not found: {script_path}", file=sys.stderr)
            errors += 1
        elif args.dry_run:
            print(f"OK: [{spec['key']}] {exp['seq']:02d} {script_path or '(no script)'}")

    if args.dry_run:
        if errors:
            print(f"\nDry run: {errors} error(s) found.", file=sys.stderr)
            sys.exit(1)
        print(f"\nDry run: all {len(specs)} experiment(s) validated.")
        return

    paths.queue_dir.mkdir(parents=True, exist_ok=True)
    for i, spec in enumerate(specs):
        exp = build_exp(spec, seq=i + 1)
        script_path = _extract_script_path(exp)
        if script_path is not None and not script_path.exists():
            print(f"ERROR: [{spec['key']}] script not found: {script_path} — skipping", file=sys.stderr)
            errors += 1
            continue
        out = paths.queue_dir / f"{exp['seq']:02d}_{exp['exp_id']}.json"
        atomic_write_json(out, exp)
        print(f"[{spec['key']}] {out}")

    if errors:
        print(f"\n{errors} error(s) — queue generated but {errors} experiment(s) skipped.", file=sys.stderr)


if __name__ == "__main__":
    main()
