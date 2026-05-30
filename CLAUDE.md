# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DistiLLM (ICML 2024) is a knowledge distillation framework for Large Language Models. It trains a smaller "student" model to match a larger "teacher" model's output distribution using **Skewed KL Divergence** loss with an **adaptive off-policy replay buffer**. The repo builds on top of MiniLLM (ICLR 2024) and supports GPT-2, OPT, OpenLLaMA, LLaMA, Mistral, and Qwen model families.

The repo contains **two independent training systems** that share data loading and evaluation infrastructure but use fundamentally different algorithms:

## Two Training Systems

### 1. DistiLLM / SFT / KD (`finetune.py`)
Standard supervised and distillation training using cross-entropy + KL-divergence losses. DeepSpeed handles distributed training. The student is trained to match teacher log-probabilities (or hard labels, for SFT).

**Training types** (`--type`): `lm` (SFT), `kd` (forward KL), `adaptive-srkl` (DistiLLM default), `fkl`, `rkl`, `jsd`, `tvd`, `sfkl`, `srkl`, `mixed`, `adaptive`

**Core DistiLLM components** in `distillm/`:
- `losses.py`: skewed KL divergence variants — mixes teacher/student probability distributions with a skew parameter `--skew-alpha`
- `sampler.py`: `SampleGenerator` — generates on-policy student outputs during training
- `buffer.py`: `ReplayBuffer` — stores past generations for off-policy replay, controlled by `--replay-ratio` and `--capacity`

**Adaptive mechanism** (`--adaptive-srkl`): An adaptive threshold controls whether to use fresh on-policy samples vs replay buffer samples. The threshold adjusts based on validation loss trends.

**Key training flow** (`finetune.py:finetune()`):
1. Forward pass to get student logits
2. Compute LM loss (cross-entropy on labels, with -100 masking for prompt tokens)
3. Compute distillation loss against teacher logits (via `get_distil_loss()`)
4. Optionally add auxiliary PT loss on pretrain corpus (`--lm-data-dir`)
5. Optionally generate on-policy student samples → push to ReplayBuffer
6. `model.backward(loss)` then `model.step()` (DeepSpeed handles optimizer/scheduler/gradients)

### 2. MiniLLM (`train_minillm.py` + `minillm/`)
PPO-based reinforcement learning approach. The student generates responses, a teacher-derived reward scores each token, and PPO policy gradient updates the student. This is the ICLR 2024 MiniLLM baseline preserved in this repo.

**Key flow**: `PPOSampler` collects rollouts → `Reward` scores tokens using teacher log-probs → `PPOTrainer` applies clipped policy gradient + auxiliary losses (PT loss, single-step regularization)

## Architecture Patterns

### Argument system (`arguments.py`)
All configuration is via CLI arguments, not config files. `get_args()` parses args and auto-constructs the save path from model names, hyperparameters, and seeds. When adding args, follow the existing grouping pattern (ModelConfig, RuntimeConfig, DataConfig, etc.). DeepSpeed configs (JSON in `configs/deepspeed/`) are loaded separately but overridden by CLI args for key fields.

### Data pipeline
1. Raw text → tokenization scripts (`tools/process_data_*.py`) → binary format (`.bin` + `.idx` pair, memory-mapped via `data_utils/indexed_dataset.py`)
2. At training time, `LMTrainDataset` returns a tuple of `(model_batch, no_model_batch, gen_data)`:
   - `model_batch`: tensors fed to the model forward pass (input_ids, attention_mask)
   - `no_model_batch`: metadata not fed to the model (labels with -100 masking, loss_mask)
   - `gen_data`: prompt-only data for student generation (left-padded, separate from training batch)
3. Documents in binary format are separated by token id `65535` between prompt and response

### Model loading pattern
- Student: loaded via `utils.get_model()` → wrapped in DeepSpeed via `deepspeed.initialize()`
- Teacher: loaded via `finetune.py:get_teacher_model()` → kept in eval mode, no DeepSpeed, optional LoRA merge
- PEFT/LoRA: supported via `--peft lora`, uses huggingface PEFT library

### Script conventions (`scripts/`)
All training/eval is launched via shell scripts organized as `scripts/{model_family}/{task}/`. Scripts take 3 positional args: `PATH_TO_DistiLLM`, `MASTER_PORT`, `GPU_NUM`. They use `torchrun` for distributed launch. GPU count is embedded in per-script `CUDA_VISIBLE_DEVICES` and `nproc_per_node`.

### Checkpoint selection convention
- DistiLLM / SFT / KD: select best checkpoint by **ROUGE-L** score
- Student initialization (init): select best checkpoint by **validation loss**

## autopipe — Experiment Orchestration

The `autopipe/` module provides a self-healing queue-based pipeline runner with LLM agent auto-repair. It manages long-running experiments without manual intervention.

### Architecture

```
make_queue.py          →  generates queue entries in autopipe/queue/ (numeric prefixed for ordering)
scheduler.py           →  polls queue, spawns workers (max 1 parallel for single-node multi-GPU)
  ├─ Phase 1           →  refresh statuses, recover stale workers, clean orphaned locks
  └─ Phase 2           →  spawn new workers for pending/failed items (exp backoff)
worker.py              →  runs bash scripts under conda env, records outcome, classifies failures
  ├─ classify_failure  →  15 error patterns (oom, loss_scale, nan, disk_full, hf, net, import,
  │                       port, nccl, path, data, shape, assert, killed, ckpt)
  ├─ OOM auto-reduce   →  halve batch_size, cap at batch_size=1
  └─ agent.py          →  invoke claude CLI with --agents spec for LLM-powered diagnosis and repair
```

### How it works
1. `make_queue.py` → generates JSON queue entries in `autopipe/queue/` (numbered `01_`-`08_` for ordering)
2. `scheduler.py` → polls queue, picks `pending`/`failed` items, spawns workers via `subprocess.Popen`
3. `worker.py` → injects conda bin to PATH, exports `train_opts` as `TRAIN_*` env vars, runs bash script
4. On failure: exponential backoff retry, OOM batch-size auto-reduction, optional LLM agent repair
5. Agent (`agent.py`): invokes `claude` CLI with `--agents '<json-spec>' --agent distillm_debugger`, reads logs → diagnoses root cause → applies fix → writes `fix_summary.txt` → exits
6. `hard_failure_threshold=3` limits agent repair attempts per unique error hash (prevents infinite repair loops)
7. Aborted tasks only auto-retry if queue config `mtime > run config mtime` (hotfix detection)

### Lock & singleton safety
- `Lock.owned()` — scheduler checks each loop iteration that it still owns `.lock_scheduler`; exits if stolen
- `Lock.heartbeat()` — updates lock file timestamp each loop so stale detection works correctly
- Scheduler Phase 1 cleans up stale `.lock_worker` files when status.json shows terminal state but lock file persists (e.g. SIGKILL bypassed worker's `finally` block)
- **Worker sets its own `status="running"`** after acquiring `.lock_worker` — scheduler does NOT set it, preventing "running but dead" state when worker fails to start

### `train_opts` mechanism
- Single source of truth: `exp.json → train_opts` dict
- Worker exports each key as `TRAIN_{KEY_UPPER}` env var before running bash scripts
- Shell scripts read with fallback: `LR=${TRAIN_LR:-0.0005}`
- Agent edits `train_opts` in exp.json → fix persists across retries
- **`train_opts` is excluded from scheduler `merge_keys`** to prevent clobbering agent fixes

### Agent prompt strategy
- Principle-based, not exhaustive: agent reads logs itself rather than matching against a pre-enumerated error catalog
- Project context dynamically injected via `repo_root` and `conda_env` parameters (no hardcoded paths in prompt)
- Agent invoked via `--agents` JSON spec + `--agent distillm_debugger` (clean, no long prompt on command line)
- Full repo access: `--add-dir` grants access to scripts, configs, data outside the run directory
- Fix precedence: exp.json `train_opts` → shell script hyperparameters → install packages → free disk → fix data → Python source
- Surgical edits only (no restructuring). Agent writes one-line summary, does NOT restart training.

### Usage
```bash
# Generate experiment queue
cd /home/ufile/group_3/zjx/distillm
python3 -m autopipe.make_queue --repo-root .

# Start scheduler (poll every 30s, run 1 experiment at a time)
python3 -m autopipe.scheduler --repo-root . --poll-seconds 30 --max-parallel 1
```

### Experiment JSON format (for `cmd_type: "bash"`)
```json
{
  "exp_id": "kd_train_xxx",
  "key": "kd_train",
  "seq": 1,
  "cmd_type": "bash",
  "cmd": "/path/to/script.sh",
  "conda_env": "llm_train",
  "gpus": "0,1,2,3",
  "train_timeout": 86400,
  "skip_vis": true,
  "max_retries": 1,
  "retry_sleep": 60,
  "hard_failure_threshold": 3
}
```

### Artifacts
- Queue: `autopipe/queue/<seq>_<exp_id>.json`
- Run artifacts: `autopipe/runs/<exp_id>/attempt_N/` (train.log, status.json)
- Per-run: `exp.json`, `status.json`, `agent.log`, `fix_summary.txt`, `agent_skip.txt`
- Agent diffs: `git_diff_pre_agent_*.patch`, `git_diff_post_agent_*.patch`
- Scheduler log: `autopipe/logs/scheduler_<exp_id>.log`
- Locks: `.lock_scheduler` (scheduler singleton), `.lock_worker` (per-experiment worker singleton)

## Key dependencies
- conda env `llm_train` with PyTorch 2.4.0, CUDA 12.4
- transformers 4.43.4
- DeepSpeed (ZeRO-1 with FP16)
- rouge-score, datasets, accelerate, PEFT
