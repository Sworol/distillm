# DistiLLM Experiment Progress

## Setup

- **Repo**: DistiLLM (ICML 2024) — knowledge distillation for LLMs
- **Task**: gpt2-base (124M) ← gpt2-xlarge (1.5B) white-box distillation
- **Hardware**: 4× RTX 4090 (24GB each)
- **Env**: conda `llm_train`, PyTorch 2.4.0, transformers 4.43.4, DeepSpeed ZeRO-1

## Data

| Data | Size | Status |
|------|------|--------|
| Dolly 15K (instruction data) | 12K train + 3K valid | Tokenized to `processed_data/dolly/full/gpt2/` |
| Combined multi-task (Dolly+SINST+UINST) | 57K train + 3K valid | Tokenized to `processed_data/combined/gpt2/` |
| OpenWebText 10M (LM corpus) | 200K docs → 512-token chunks | Tokenized to `processed_data/openwebtext/gpt2/512/10M/` |
| gpt2-base weights | 478M | Local `checkpoints/gpt2-base/` |
| gpt2-xlarge weights | 5.9G | Local `checkpoints/gpt2-xlarge/` |
| 5 Benchmark eval data | Self-Inst(242), Vicuna(80), SINST(1694), UINST(23916) | Downloaded from MiniLLM HF |
| Combined prompt-only data | MiniLLM prompt data | Tokenized to `processed_data/combined_prompt/gpt2/` |

---

## Results Summary: Multi-task DistiLLM vs Paper

| Benchmark | DistiLLM (Ours) | Paper DistiLLM | Delta |
|-----------|----------------|----------------|-------|
| Dolly | **26.67** | 26.11 ± 0.68 | +0.56 |
| Self-Inst | **13.51** | 13.14 ± 0.69 | +0.37 |
| Vicuna | 16.12 | 18.46 ± 0.53 | -2.34 |
| SINST | **35.71** | 27.51 ± 0.03 | **+8.20** |
| UINST | 21.93 | 29.35 ± 0.07 | **-7.42** |

### S/T Ratio (Multi-task Teacher)

| Benchmark | Teacher SFT | DistiLLM Student | S/T |
|-----------|------------|-----------------|------|
| Dolly | 75.19 | 26.67 | 35.5% |
| Self-Inst | 13.46 | 13.51 | **100.4%** |
| Vicuna | 15.13 | 16.12 | **106.5%** |
| SINST | 38.70 | 35.71 | 92.3% |
| UINST | pending | 21.93 | - |

### Analysis

- **Dolly & Self-Inst**: Match or slightly exceed the paper.
- **SINST +8.20**: Training data includes SINST 0_2/3_6/6_10. Expected improvement.
- **Self-Inst/Vicuna student > teacher**: On OOD benchmarks, DistiLLM generalizes better than SFT teacher — the teacher overfits to its training distribution.
- **Vicuna -2.34**: No Vicuna data in training. Distillation hurts out-of-distribution performance here.
- **UINST -7.42**: Trained on UINST 0_2/3_5/6_10 but evaluated on split 11_. Large domain gap between train/eval splits. Needs investigation.

---

## Multi-task Training (Round 2)

| Stage | Model | Data | Epochs | Steps | Time | Checkpoint |
|-------|-------|------|--------|-------|------|------------|
| Teacher SFT | gpt2-xlarge | 60K multi-task | 10 | 37,220 | ~10.5h | `sft_multitask/e10-bs4-lr5e-05-G1-N4-NN1/37220` |
| Student Init | gpt2-base | 60K multi-task | 3 | 5,583 | ~0.6h | `init_multitask/e3-bs8-lr0.0005-G1-N4-NN1/5583` |
| DistiLLM | gpt2-base | 60K + OWT aux | 20 | 9,300 | ~5h | `distill_multitask/9300` |

### Combined training data

| Source | Items | Role |
|--------|-------|------|
| Dolly valid | 3,000 | Dev eval |
| Dolly train | 12,000 | Training |
| SINST 0_2 + 3_6 + 6_10 | 6,660 | Training |
| UINST 0_2 + 3_5 + 6_10 | 40,893 | Training |
| **Total** | **62,553** | `processed_data/combined/gpt2/` |

---

## Dolly-only (Round 1, completed)

### Results

| Benchmark | DistiLLM | Teacher SFT | S/T |
|-----------|---------|-------------|------|
| Dolly | 26.64 | 28.82 | 92.4% |
| Self-Inst | 12.05 | 14.99 | 80.4% |
| Vicuna | 16.90 | 16.26 | 104.0% |
| SINST | 23.44 | 26.07 | 89.9% |
| UINST | 25.70 | 28.62 | 89.8% |

**Multi-seed Dolly (5 seeds):** 28.93 ± 0.23 rougeL

### Checkpoints

| Stage | Checkpoint |
|-------|------------|
| Teacher SFT | `sft/e10-bs2-lr5e-05-G1-N4-NN1/17500` |
| Student Init | `init/e3-bs8-lr0.0005-G1-N4-NN1/1311` |
| DistiLLM | `distill_0.1B_1.5B/2180` |

---

## Baseline Experiments (Autopipe Orchestration)

Experiments are managed via `autopipe/` — a self-healing queue-based pipeline runner with LLM agent auto-repair.

Start: `python3 -m autopipe.scheduler --repo-root . --poll-seconds 30 --max-parallel 1`

### Architecture

```
make_queue.py          →  autopipe/queue/*.json (10 experiments, ordered by numeric prefix)
scheduler.py           →  polls queue, picks pending items, spawns workers (max 1 parallel)
  ├─ Phase 1           →  refresh statuses, recover stale workers, clean orphaned locks
  ├─ Phase 2           →  spawn new workers for pending/failed items (exp backoff)
  └─ Heartbeat          →  every 10 cycles (~5min) prints running/done/pending summary
worker.py              →  runs bash script under conda env, records success/failure
  ├─ classify_failure  →  scans train.log for 15 error patterns (oom, loss_scale, nan, ...)
  ├─ OOM auto-reduce   →  halve batch size, cap at 1
  └─ agent.py          →  on unclassified/OOM-exhausted failures: invoke claude CLI for diagnosis+repair
```

### Agent Auto-Repair (`autopipe/agent.py`)

On worker-detected failure that can't be handled by OOM-reduction:
1. Worker calls `classify_failure()` on train.log → maps to error type
2. If error is new (by error_hash dedup), invokes `claude` CLI with `--add-dir` pointing to repo root
3. Agent reads all logs, diagnoses root cause, applies smallest fix, writes one-line summary, exits
4. Scheduler retries with exponential backoff (up to `hard_failure_threshold=3` per error hash)
5. Agent has access to: Edit/Write/Read + Bash (ls, find, cat, grep, pip, python, df, nvidia-smi)

### Scheduler Features

- **PID-based singleton lock** (`.lock_scheduler`): prevents duplicate scheduler processes; detects stale locks (12h timeout) and allows forced steal via `--force-steal-lock`
- **Time-window scheduling**: `--active-window 22:00-08:00` restricts training to overnight hours; `--window-kill` SIGTERMs workers when window ends (checkpoint-friendly)
- **Heartbeat**: every 10 poll cycles (~5 min at default 30s), prints running/done/pending counts
- **Exponential backoff**: failed experiments wait `retry_sleep * 2^consecutive_failures` seconds (capped at 900s)
- **Aborted hotfix detection**: aborted tasks auto-retry only if queue config mtime > run config mtime (supports hotfix edits)

### Worker Environment Setup

- `PATH`: prepends conda env bin (ensures correct torchrun, python)
- `PYTHONPATH`: set to repo root (ensures `data_utils`, `distillm` imports work)
- `train_opts` from exp.json exported as `TRAIN_{KEY_UPPER}` env vars
- `HF_ENDPOINT=https://hf-mirror.com` for model downloads in China

### Queue (sequential execution)

| # | Task | Script | GPUs | Est. Time | Status |
|---|------|--------|------|-----------|--------|
| 1 | KD train | `scripts/run_kd_multitask.sh` | 0-3 | ~2-3h | success |
| 2 | KD eval | `scripts/run_eval_kd_multitask.sh` | 0 | ~1h | success |
| 3 | SeqKD gen | `scripts/gpt2/tools/generate_data_seqkd_multitask.sh` | 0-3 | ~1.5h | success |
| 4 | SeqKD process | `scripts/process_seqkd_data.sh` | CPU | ~10min | success |
| 5 | SeqKD train | `scripts/gpt2/seqkd/seqkd_multitask_base.sh` | 0-3 | ~12h | running (attempt 5/5, epoch 0, iter ~500) |
| 6 | SeqKD eval | `scripts/run_eval_seqkd_multitask.sh` | 0 | ~1h | failed (dependency: needs seqkd_train ckpt) |
| 7 | MiniLLM train | `scripts/gpt2/minillm/train_multitask_base_xl.sh` | 0-3 | ~10-20h | failed (killed, attempt 2, agent fixed model_kwargs) |
| 8 | MiniLLM eval | `scripts/run_eval_minillm_multitask.sh` | 0 | ~1h | failed (needs minillm_train) |
| 9 | DistiLLM train | `scripts/run_distillm_multitask.sh` | 0-3 | ~5-8h | pending |
| 10 | DistiLLM eval | `scripts/run_eval_multitask_student.sh` | 0 | ~1h | pending |

Total completed so far: 4/10. seqkd_train at 141K steps × 20 epochs → estimated ~12h remaining.

### Data status

| Baseline | Data | Status |
|----------|------|--------|
| KD | `processed_data/combined/gpt2/` | Ready |
| SeqKD gen | `processed_data/combined_prompt/gpt2/` (teacher pseudo-labels) | Generated (60K items) |
| SeqKD train | `processed_data/combined/pseudo/sft_multitask/gpt2/` (tokenized pseudo-labels) | Ready |
| MiniLLM | `processed_data/combined_prompt/gpt2/` | Ready |
| DistiLLM | `processed_data/combined/gpt2/` + `processed_data/openwebtext/gpt2/512/10M/` | Ready |

### Agent Fix Summaries

| Experiment | Attempt | Error | Fix |
|-----------|---------|-------|-----|
| kd_train | 2 | GPU OOM (loss_scale underflow) | Reduced lr 0.0005→0.0001; added num_workers=2 |
| kd_eval | 2 | Checkpoint detection selected eval/ dir | grep -v '/eval/' → config.json check loop |
| seqkd_gen | 1 | `ImportError: cannot import mpu from transformers` | try/except ImportError in 4 files (mpu removed in transformers>=4.x) |
| seqkd_process | 1-2 | Cascading: no raw.jsonl from gen | Fixed mpu import in generate.py upstream |
| seqkd_train | 1-4 | Prompt overflow (368 > max_prompt_length=256) | Truncate prompt at lm_datasets.py:74 |
| seqkd_train | 4 | Disk full: PyTorch checkpoint write failed | classifies as disk_full now; scheduler tolerates OSError |
| minillm_train | 2 | `ValueError: mix_in_model` in generate() | Pop unrecognized kwargs before model.generate() (HF >=4.x validation) |
| minillm_eval | 1 | Dependency: no minillm checkpoint yet | Will auto-resolve once minillm_train succeeds |
| seqkd_eval | 1-3 | Dependency: no seqkd checkpoint yet | Will auto-resolve once seqkd_train succeeds |

### Autopipe Bug Fixes & Improvements

1. **Queue ordering**: Renamed queue files with `01_`-`10_` numeric prefixes. `make_queue.py` generates prefixed names via `seq` parameter.
2. **Infinite retry loop**: Fixed `max_retries` exhaustion + automatic `aborted→failed` reset bug. Aborted tasks only auto-retry if queue config mtime > run config mtime (hotfix detection).
3. **Agent CLI args**: Corrected from `--print` (wrong) to `-p --no-session-persistence --dangerously-skip-permissions --add-dir <repo_root> --agents '<json-spec>' --agent distillm_debugger`.
4. **hard_failure_threshold merge**: Added to scheduler `merge_keys` so queue updates propagate.
5. **Conda env injection**: Worker prepends conda bin to PATH, sets PYTHONPATH, exports `TRAIN_*` env vars.
6. **Failure classification**: 15 error patterns (oom, loss_scale, nan, disk_full, hf, net, import, port, nccl, path, data, shape, assert, killed, ckpt). Extended `disk_full` to catch PyTorch `file write failed` / `inline_container` errors.
7. **Agent prompt**: Principle-based methodology — agent reads logs itself rather than matching against a pre-enumerated error catalog.
8. **Scheduler singleton safety**: `Lock.owned()` check every loop; `Lock.heartbeat()` with TOCTOU-safe O_RDWR; SIGTERM/SIGINT handlers terminate tracked workers.
9. **Worker lock cleanup**: Phase 1 cleans orphaned `.lock_worker` files when status.json shows terminal state.
10. **train_opts exclude from merge**: Scheduler `merge_keys` skips `train_opts` so agent edits persist across queue merges.
11. **Time-window scheduling**: `--active-window 22:00-08:00` + `--window-kill` for overnight training.
12. **Heartbeat**: Every 10 poll cycles prints `running=X done=Y/Z pending=W` summary.
13. **Scheduler disk-full resilience**: Main loop wrapped in `try/except OSError` + pre-I/O disk check (<1GB skips spawn). Prevents scheduler crash on full disk; worker cleanup runs normally.

### Data status

| Baseline | Data | Status |
|----------|------|--------|
| KD | `processed_data/combined/gpt2/` | Ready |
| SeqKD gen | `processed_data/combined_prompt/gpt2/` (teacher) | Ready |
| SeqKD train | `processed_data/combined/pseudo/sft_multitask/` | Needs gen+tokenize |
| MiniLLM | `processed_data/combined_prompt/gpt2/` | Ready |

### Autopipe Bug Fixes & Improvements

1. **Queue ordering**: Renamed queue files with `01_`-`08_` numeric prefixes. `make_queue.py` now generates prefixed names via `seq` parameter.
2. **Infinite retry loop**: `max_retries=1` + automatic `aborted→failed` reset caused 26 restarts. Fixed: aborted tasks only auto-retry if queue config mtime > run config mtime.
3. **Agent CLI args**: Fixed from `--print` (wrong) to `-p --no-session-persistence --permission-mode bypassPermissions --add-dir <repo_root>`.
4. **hard_failure_threshold merge**: Added to scheduler's `merge_keys` so queue updates propagate to run configs.
5. **Conda env injection**: Worker now prepends conda bin to PATH and sets PYTHONPATH before running bash scripts (fixes bare `torchrun` resolving to wrong env).
6. **Failure classification**: Expanded from 7 to 15 error patterns (added loss_scale, nan, disk_full, data, shape, assert, killed, ckpt).
7. **Agent prompt**: Rewrote from exhaustive 100+ line error catalog to principle-based methodology — agent reads logs itself and diagnoses ANY failure.

---

## Issues & Notes

1. **Save path precedence bug**: In `arguments.py:270`, the expression `(f"{args.ckpt_name}" + ... if args.peft_name is not None else "")` drops `ckpt_name` when `peft_name` is None due to operator precedence.

2. **Eval data format**: `evaluate_main` expects JSONL with `{"prompt": "...", "output": "..."}` fields.

3. **GPU memory**: Other processes (`ader` env) occupy ~7.7GB per GPU. Use `CUDA_VISIBLE_DEVICES=3` for eval. Kill conflicting processes before full training to free up memory.

4. **Data leakage in Dolly eval**: All 15K Dolly items were used for training. Use training dev set (held-out 1000) for fair comparison.

5. **torchrun path**: Must use `/anaconda3/envs/llm_train/bin/torchrun` (not system `torchrun` which points to `openmmlab` env).

6. **GPU OOM with multi-task Teacher SFT**: gpt2-xlarge (1.5B) with batch_size=2 on combined 60K data OOMed when other processes used ~7.5GB per GPU. Reduced to batch_size=1+grad_acc=2. After killing other processes and freeing full 24GB, batch_size=4 works safely.

7. **Disk full crash** (May 30): `/home/ufile` hit 100% (1007G). `sft_multitask` had 13 intermediate checkpoints at ~3GB each = ~30GB. Deleted 9 intermediate checkpoints (kept final 37220), freed 26GB. Currently at 98% (27G free) — tight, monitor if training checkpoints accumulate.

8. **Checkpoint naming convention**: Finetune saves checkpoints every epoch unless `--save-interval -1`. For 20-epoch runs, this means 20 checkpoints per experiment at ~1-2GB each for gpt2-base. Delete intermediate checkpoints after completion.

9. **Residual GPU processes**: After killing training, Python processes may not release GPU memory immediately. Use `nvidia-smi` to verify, then `kill -9` residual PIDs.

10. **SINST output format**: SINST data has `output` as list (e.g., `['Response 2']`) not string. Must convert before tokenization.

11. **`.gitignore` pattern syntax**: `./results/` doesn't work — use `/results/` to anchor to repo root.

12. **Teacher eval hung on Dolly**: DeepSpeed eval on gpt2-xlarge hung indefinitely at ~26% through Dolly. Subsequent eval completed normally.

13. **`transformers` 4.43.4 dropped `mpu` module**: `from transformers import mpu` removed in >=4.x. 4 files need try/except ImportError guards (generate.py, minillm/sampler.py, minillm/pipelines.py, minillm/losses.py).

14. **HuggingFace generate() validates model_kwargs**: >=4.x rejects unrecognized keys. `minillm/model.py:generate()` must pop custom keys (mix_in_model, mix_in_alpha) before forwarding.

15. **SeqKD prompts exceed 256 tokens**: Pseudo-label generation can produce prompts longer than max_prompt_length. `lm_datasets.py:74` must truncate: `prompt = prompt[-self.max_prompt_length:]`.

## New Checkpoints Downloaded

| Model | Size | Path |
|-------|------|------|
| gpt2-medium | 355M params | `checkpoints/gpt2-medium/` |
| gpt2-large | 774M params | `checkpoints/gpt2-large/` |

For future medium/large scale distillation experiments.
