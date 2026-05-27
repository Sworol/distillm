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

## Key dependencies
- PyTorch 2.1.2 with CUDA 12.1
- transformers 4.42.4 (was pinned to a specific commit following MiniLLM)
- DeepSpeed (ZeRO-1 or ZeRO-2 with FP16)
- vLLM 0.5.0, PEFT, accelerate, rouge-score, datasets
