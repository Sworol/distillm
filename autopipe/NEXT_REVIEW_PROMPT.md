# Autopipe Next Review Prompt

## 核心理念
用多个子agent做review+fix，每轮聚焦不同维度，前一轮修复的代码作为下一轮的基线。禁止降低agent权限。

## 6轮分工模板

### Round 1 — 代码质量和正确性
```
Do a rigorous code review of autopipe/ (8 files). Focus on:
- bugs, race conditions, error handling gaps, resource leaks
- Logic bugs, type safety, crash paths
Fix ALL real bugs. Run tests after fixing.
IMPORTANT: Do NOT reduce agent permissions in agent.py.
```

### Round 2 — 架构和并发
```
Deep architectural review. Focus on:
- State machine correctness (all status transitions)
- Concurrency: scheduler+worker lock semantics, signal handler edge cases
- Resource management: subprocess lifecycle, file descriptors, thread safety
- Crash-only design: what breaks if any component crashes at any point
- Data integrity: exp.json/status.json consistency, CONFIG_MERGE_KEYS
- Error classification: false positives/negatives in classify_failure()
- Signal handler: module-level globals _current_subprocess/_worker_lock safe?
- hang_watcher daemon thread race conditions
- Exponential backoff formula correctness
- Disk full: every I/O path covered?
```

### Round 3 — 测试覆盖和边界条件
```
Test coverage + edge cases. Focus on:
- Read ALL test files, find coverage gaps for critical paths
- classify_failure() multi-error priority, ambiguous patterns
- RecoveryManager race conditions (agent writes vs recovery reads)
- make_queue.py: UUID collision, script path, duplicate seq
- io_utils Lock.acquire() TOCTOU
- Error hash stability: can same error produce different hashes?
- Signal handler during agent.run_agent()
- Disk full at atomic_write_json
- Corrupted queue JSON handling
Add tests for uncovered critical paths.
```

### Round 4 — 未审模块和跨模块集成
```
Focused module review + integration. Focus on:
1. agent.py: --add-dir memory impact, allowedTools completeness, broken binary detection
2. make_queue.py: shlex.split, queue cleanup, dual-pass consistency
3. git_utils.py: binary diff OOM, non-git repo, byte-level truncation
4. Cross-module: scheduler->worker spawn, worker->agent invocation,
   agent->exp.json edit, recovery->scheduler sync
5. config.py: CONFIG_MERGE_KEYS consistency, Paths dataclass
```

### Round 5 — 生产加固
```
Production hardening. Focus on:
- Subprocess lifecycle: torchrun nested process tree cleanup
- Memory leaks in scheduler (runs 24/7 for weeks)
- Log file rotation (multi-GB run.log)
- Network partition (HF timeout mid-training)
- Conda env edge cases (binary disappeared, activation failure)
- Race condition re-review: ALL lock/file/status transitions
- Python version compatibility (| type unions, missing_ok)
- Signal safety: os.kill(os.getpid(), signum) correctness
- Unused imports, dead code, misleading comments
```

### Round 6 — 最终签字验收
```
Final sign-off checklist:
- Correctness: ALL status transitions valid and complete
- Consistency: classify_failure and _last_error_hash use SAME get_classification_text
- Robustness: every file path has error handling
- No unbounded growth in long-running process
- Subprocess cleanup guaranteed in all exit paths
- Agent permissions: --dangerously-skip-permissions PRESERVED
- Dead code / unused imports / misleading comments
- All tests pass
Read ALL 8 source files + ALL test files one more time.
```

## 已知已修复的问题（不要再重复）
1. recovery.py: ValueError on malformed status.json
2. recovery.py: TOCTOU race on run_exp_path.stat()
3. io_utils.py: Lock.release() leaks stale _pid
4. worker.py: PID-reuse race in _hang_watcher
5. scheduler.py: TOCTOU on read_json in _load_run_exp
6. scheduler.py: corrupted queue JSON crashes load_exp
7. scheduler.py: corrupted exp.json crashes _load_run_exp
8. scheduler.py: heartbeat crash on corrupted status.json
9. agent.py: missing Glob/Grep in --allowedTools
10. agent.py: no broken-binary detection
11. make_queue.py: str.split() instead of shlex.split()
12. make_queue.py: no queue cleanup between runs
13. git_utils.py: OOM on binary diff (unlimited check_output)
14. git_utils.py: no early-exit for non-git repos
15. worker.py: ensure_exp_sane false failure for bash-prefixed cmd
16. recovery.py: dead code in Case 2

## 状态机参考
```
pending -> running -> success
pending -> running -> failed -> pending (OOM backoff)
pending -> running -> failed -> failed -> ... -> failed -> aborted (max_retries)
pending -> running -> failed -> hard_failure (agent exhausted)
failed -> aborted (hotfix reopens to failed)
aborted -> failed (queue mtime > run mtime)
```

## 不可修改项
- agent.py 的 `--dangerously-skip-permissions` 必须保留
- agent.py 的 `danger-full-access` sandbox 必须保留
- agent.py 的 `--allowedTools` 只能扩充不能缩减
- 所有现有测试必须继续通过

## 运行测试的命令
```bash
cd /home/ufile/group_3/zjx/distillm && PYTHONPATH=. python -m pytest autopipe/tests/ -v -p no:launch_testing -p no:launch -p no:launch_testing_ros -c /dev/null 2>&1 | tail -40
```

## 当前状态
- 263 tests, all passing
- 8 source files + 8 test files
- 审查轮次: 6 rounds, 16 bugs fixed, +192 tests

## 未解决但可接受的设计权衡
- 真实生产环境新错误模式需要实际日志来迭代
- DeepSpeed/NCCL/PyTorch 新版本的 breaking change
- 单机多GPU调度策略在实际负载下的调优
- classify_failure 对非英文错误消息的覆盖不足
- recover_stale_worker Case 3 有300秒grace window（超慢模型初始化可能误判）
