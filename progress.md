# Progress Log

## 2026-06-01

### Checkpoint Resume (`--load`)
- `utils.py`: `get_model()` now supports `--load` to resume from a prior training directory.
  - Auto-discovers the latest step checkpoint by scanning numeric subdirectories.
  - Validates `pytorch_model.bin` size (>100MB) to skip corrupt checkpoints.
- `worker.py`: Exports `AUTOPIPE_LOAD_PATH` from `train_opts.load_path` so shell scripts pick up the checkpoint path.
- **Caveat**: Only loads model weights (not optimizer state). DeepSpeed FP16 can be unstable without proper optimizer resume. Best used for evaluation or as weight init; full training resume needs DeepSpeed `load_checkpoint`.

### Scheduler Bug Fix
- `scheduler.py`: Phase 1 stale lock cleanup now handles empty/corrupt `.lock_worker` files.
  - Changed condition from `pid is not None` to `pid is None` — empty lock files (0 bytes) are now deleted.
  - Prevents infinite `lock_busy` retry loops when a worker fails before writing its PID.

### MiniLLM Training — 4-round Agent Repair
- `minillm_train`: failed 4 times, each with a different root cause:
  1. `ImportError: cannot import name 'mpu' from 'transformers'` → agent fixed `trainer.py` + `reward.py`
  2. `ValueError: model_kwargs not used: ['mix_in_model', 'mix_in_alpha']` → agent fixed `model.py`
  3. CUDA init error in DataLoader (empty diff, agent didn't fix effectively)
  4. Same CUDA init error → agent fixed `minillm/storages.py`: added `.cpu()` to 10 tensor attributes in `PPORolloutStorage.collate()` so DataLoader fork workers don't need CUDA
- **Root cause**: `pad_sequence()` called on GPU-resident tensors inside DataLoader worker processes (num_workers=4) that don't have CUDA initialized after fork.
- **Fix applied**, awaiting retry (attempt 5/5, backoff 15 min). DistillM training must finish first.

### Queue Status (Jun 1 ~10:00)
- kd_eval: success
- seqkd_gen: success
- seqkd_process: success
- seqkd_train: success
- seqkd_eval: success
- **distillm_train: running** (epoch 17/20, iter 8020/9300, ~86%, loss=3.63, eta ~10-15 min)
- minillm_train: failed (fix ready, awaits retry after distillm)
- minillm_eval: failed (depends on minillm_train)
- distillm_eval: pending
