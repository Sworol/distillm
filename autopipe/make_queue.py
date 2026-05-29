from __future__ import annotations

import argparse
import uuid
from pathlib import Path
from typing import Any, Dict, List

from autopipe.config import default_paths
from autopipe.io_utils import atomic_write_json, now_ts


def distillm_specs() -> List[Dict[str, Any]]:
    """Defines all DistiLLM baseline experiments in execution order."""

    BASE = "/home/ufile/group_3/zjx/distillm"
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


def build_exp(spec: Dict[str, Any]) -> Dict[str, Any]:
    exp_id = f"{spec['key']}_{uuid.uuid4().hex[:8]}"
    return {
        "exp_id": exp_id,
        "key": spec["key"],
        "cmd_type": "bash",
        "cmd": spec["cmd"],
        "created_at": now_ts(),
        "status": "pending",
        "attempt": 0,
        "max_retries": spec.get("max_retries", 1),
        "retry_sleep": spec.get("retry_sleep", 60),
        "gpus": spec.get("gpus", "0,1,2,3"),
        "train_timeout": spec.get("train_timeout", 86400),
        "skip_vis": spec.get("skip_vis", True),
        "conda_env": spec.get("conda_env", "llm_train"),
        "hard_failure_threshold": 0,  # disable LLM agent auto-repair
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=".", help="Path to repo root (default: .)")
    args = ap.parse_args()

    repo_root = Path(args.repo_root).resolve()
    paths = default_paths(repo_root)
    paths.queue_dir.mkdir(parents=True, exist_ok=True)

    specs = distillm_specs()
    for spec in specs:
        exp = build_exp(spec)
        out = paths.queue_dir / f"{exp['exp_id']}.json"
        atomic_write_json(out, exp)
        print(f"[{spec['key']}] {out}")


if __name__ == "__main__":
    main()
