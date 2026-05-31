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

### Queue Status (Jun 1 ~02:30)
- kd_eval: success
- seqkd_gen: success
- seqkd_process: success
- seqkd_train: success (attempt 6, trained from scratch, ~5.7h)
- seqkd_eval: running
- minillm_train: failed (awaits retry)
- minillm_eval: failed
- distillm_train: pending
- distillm_eval: pending
