#!/bin/bash
# Wait for teacher eval (UINST) to finish, then start autopipe scheduler
RESULTS_FILE="/home/ufile/group_3/zjx/distillm/results/gpt2/eval_main/multitask_teacher_seed10.txt"
BASE_PATH="/home/ufile/group_3/zjx/distillm"

echo "[wait] Waiting for teacher eval to complete (UINST benchmark)..."
while ! grep -q "uinst_11_" "$RESULTS_FILE" 2>/dev/null; do
    sleep 30
done
echo "[wait] Teacher eval done at $(date)"

echo "[scheduler] Starting autopipe scheduler..."
cd "$BASE_PATH"
python3 -m autopipe.scheduler --repo-root . --poll-seconds 30 --max-parallel 1
