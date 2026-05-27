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

## Pipeline — Round 1: Dolly-only (completed)

### Stage 1: Teacher SFT ✅

| Config | Value |
|--------|-------|
| Model | gpt2-xlarge (1.5B) |
| Epochs | 10 |
| Batch size | 2 (effective global = 8) |
| LR | 5e-5, cosine decay |
| Weight decay | 1e-2 |

**Results (Dolly dev, 1000 items):**
- rougeL: 28.82
- exact_match: 3.3%
- Checkpoint: `results/gpt2/train/sft/e10-bs2-lr5e-05-G1-N4-NN1/17500`

### Stage 2: Student Init ✅

| Config | Value |
|--------|-------|
| Model | gpt2-base (124M) |
| Epochs | 3 |
| Batch size | 8 (effective global = 32) |
| LR | 5e-4, cosine decay |

- Checkpoint: `results/gpt2/train/init/e3-bs8-lr0.0005-G1-N4-NN1/1311`

### Stage 3: DistiLLM ✅

| Config | Value |
|--------|-------|
| Student | gpt2-base (from Stage 2) |
| Teacher | gpt2-xlarge (from Stage 1) |
| Epochs | 20 |
| Batch size | 4 × grad_acc 8 (effective global = 128) |
| LR | 5e-4, cosine decay |
| KD ratio | 1.0 |
| Type | `adaptive-sfkl` (skewed forward KL) |
| Replay buffer | capacity=1000, init_threshold=0.0 |
| LM data | OpenWebText 10M for auxiliary PT loss |

**Training progression (Dolly dev, per-epoch eval):**
- Epoch 0: rougeL 19.5
- Epoch 10: rougeL 26.2
- Epoch 20: rougeL 26.6

- Checkpoint: `results/gpt2/train/distill_0.1B_1.5B/2180`

## Evaluation — Round 1 (Dolly-only teacher)

### 5-Benchmark Results

| Benchmark | Items | DistiLLM Student | Teacher SFT | S/T Ratio |
|-----------|-------|-----------------|-------------|-----------|
| Dolly | 3000 | 29.06 (leaked) / **26.64** (held-out) | 83.98 (leaked) / **28.82** (held-out) | 92.4% |
| Self-Inst | 242 | 12.05 | 14.99 | 80.4% |
| Vicuna | 80 | 16.90 | 16.26 | 104.0% |
| SINST (11_) | 1694 | 23.44 | 26.07 | 89.9% |
| UINST (11_) | 10000 | 25.70 | 28.62 | 89.8% |

**Multi-seed Dolly (5 seeds):** 28.93 ± 0.23 rougeL

### Comparison with Paper

Paper uses multi-task instruction teacher; our teacher is Dolly-only → absolute numbers lower on SINST/UINST (~4 points). But S/T retention is 80-104%.

| Benchmark | Our DistiLLM | Paper DistiLLM | Paper KD |
|-----------|-------------|----------------|----------|
| Dolly | 26.64 (held-out) | 26.11 ± 0.68 | 23.52 |
| Self-Inst | 12.05 | 13.14 ± 0.69 | 11.23 |
| Vicuna | 16.90 | 18.46 ± 0.53 | 15.92 |
| SINST | 23.44 | 27.51 ± 0.03 | 20.68 |
| UINST | 25.70 | 29.35 ± 0.07 | 23.38 |

## Pipeline — Round 2: Multi-task training (in progress)

### Combined training data ✅

| Source | Items | Role |
|--------|-------|------|
| Dolly valid | 3,000 | Dev eval |
| Dolly train | 12,000 | Training |
| SINST 0_2 + 3_6 + 6_10 | 6,660 | Training |
| UINST 0_2 + 3_5 + 6_10 | 40,893 | Training |
| **Total train** | **59,553** | |
| **Total** | **62,553** | ← `processed_data/combined/gpt2/` |

### Multi-task training script

Script: `run_multitask_experiment.sh`
- Stage 1: Teacher SFT — 10 epochs, ~74K steps (estimated ~2 days on 4x 4090)
- Stage 2: Student Init — 3 epochs
- Stage 3: DistiLLM — 20 epochs, adaptive-sfkl + replay buffer + OWT auxiliary

## Issues & Notes

1. **Save path precedence bug**: In `arguments.py:270`, the expression `(f"{args.ckpt_name}" + ... if args.peft_name is not None else "")` drops `ckpt_name` when `peft_name` is None due to operator precedence.

2. **Eval data format**: `evaluate_main` expects JSONL with `{"prompt": "...", "output": "..."}` fields.

3. **GPU memory**: Other processes (`ader` env) occupy ~7.7GB per GPU. Use `CUDA_VISIBLE_DEVICES=3` for eval.

4. **Data leakage in Dolly eval**: All 15K Dolly items were used for training. Use training dev set (held-out 1000) for fair comparison.

5. **torchrun path**: Must use `/anaconda3/envs/llm_train/bin/torchrun` (not system `torchrun` which points to `openmmlab` env).

## Remaining

- [x] Teacher eval on Dolly test set
- [x] Download 5-benchmark eval data
- [x] Multi-seed Dolly eval (rougeL 28.93 ± 0.23, 5 seeds)
- [x] 5-benchmark evaluation (DistiLLM student + Teacher SFT)
- [x] Paper comparison
- [x] Download SINST/UINST training splits
- [x] Merge + tokenize combined training data (62,553 items)
- [ ] Multi-task: Teacher SFT (script ready: `run_multitask_experiment.sh`)
- [ ] Multi-task: Student Init
- [ ] Multi-task: DistiLLM
- [ ] Multi-task: 5-benchmark re-evaluation
- [ ] KD baseline training (script ready: `scripts/run_kd_baseline.sh`)
- [ ] SeqKD baseline (needs teacher data generation)
- [ ] MiniLLM baseline (needs PPO data prep)
