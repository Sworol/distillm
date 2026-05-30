"""Tests for autopipe core utilities: Lock, error hashing, OOM backoff, etc."""

from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path
from typing import Any, Dict

import pytest

from autopipe.io_utils import (
    Lock,
    _filter_noise,
    _proximity_match,
)
from autopipe.worker import (
    _last_error_hash,
    _prepare_environment,
    apply_oom_batch_backoff,
    resolve_master_port,
)
from autopipe.make_queue import build_exp


# ============================================================================
# Lock tests
# ============================================================================

class TestLock:
    def test_acquire_release(self, tmp_path: Path):
        lock = Lock(tmp_path / ".lock_test")
        assert lock.acquire()
        assert lock.owned()
        lock.release()
        assert not lock.owned()

    def test_acquire_second_fails(self, tmp_path: Path):
        lock1 = Lock(tmp_path / ".lock_test")
        lock2 = Lock(tmp_path / ".lock_test")
        assert lock1.acquire()
        assert not lock2.acquire()
        lock1.release()

    def test_heartbeat_updates_timestamp(self, tmp_path: Path):
        lock = Lock(tmp_path / ".lock_test")
        lock.acquire()
        mtime_before = lock.path.stat().st_mtime
        time.sleep(0.1)
        lock.heartbeat()
        assert lock.path.stat().st_mtime > mtime_before
        lock.release()

    def test_heartbeat_noop_when_not_owned(self, tmp_path: Path):
        lock = Lock(tmp_path / ".lock_test")
        lock._pid = None
        lock.heartbeat()  # should not raise

    def test_stale_lock_acquire(self, tmp_path: Path):
        """Acquire a stale lock whose PID is dead."""
        lock_path = tmp_path / ".lock_stale"
        # Write a dead PID
        dead_pid = 99999
        # Ensure PID is dead
        try:
            os.kill(dead_pid, 0)
            pytest.skip(f"PID {dead_pid} is unexpectedly alive")
        except (ProcessLookupError, PermissionError):
            pass

        lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock_path.write_text(f"pid={dead_pid}\nts=old\n", encoding="utf-8")
        # Set old mtime
        os.utime(lock_path, (time.time() - 48 * 3600, time.time() - 48 * 3600))

        lock = Lock(lock_path, stale_seconds=24 * 3600)
        assert lock.acquire()
        assert lock.owned()
        lock.release()

    def test_owned_returns_false_when_pid_none(self, tmp_path: Path):
        lock = Lock(tmp_path / ".lock_test")
        assert not lock.owned()

    def test_owned_returns_false_when_file_missing(self, tmp_path: Path):
        lock = Lock(tmp_path / ".lock_ghost")
        lock._pid = os.getpid()
        assert not lock.owned()

    def test_heartbeat_detects_stolen_lock(self, tmp_path: Path):
        lock1 = Lock(tmp_path / ".lock_test")
        lock1.acquire()
        # Simulate stolen lock: overwrite with different PID
        lock1.path.write_text(f"pid={os.getpid() + 99999}\nts=x\n", encoding="utf-8")
        lock1.heartbeat()
        assert lock1._pid is None  # should detect eviction
        lock1.release()


# ============================================================================
# _proximity_match tests
# ============================================================================

class TestProximityMatch:
    def test_basic_match(self):
        assert _proximity_match("killed x out of memory y", "killed", "out of memory")

    def test_too_far(self):
        a = "killed"
        b = "out of memory"
        gap = "x" * 600
        text = f"{a}{gap}{b}"
        assert not _proximity_match(text, a, b)

    def test_within_distance(self):
        a = "killed"
        b = "out of memory"
        gap = "x" * 100
        text = f"{a}{gap}{b}"
        assert _proximity_match(text, a, b)


# ============================================================================
# _filter_noise tests
# ============================================================================

class TestFilterNoise:
    def test_removes_warning_line(self):
        text = "WARNING: disk full check passed\nTraining error here\n"
        result = _filter_noise(text)
        assert "WARNING" not in result
        assert "Training error here" in result

    def test_removes_rank_warning_line(self):
        text = "[rank0] WARNING: overflow detected\nreal error\n"
        result = _filter_noise(text)
        assert "WARNING" not in result
        assert "real error" in result

    def test_removes_info_line(self):
        text = "[rank0]: INFO: checkpoint saved\nreal error\n"
        result = _filter_noise(text)
        assert "INFO" not in result
        assert "real error" in result

    def test_preserves_non_noise(self):
        text = "Error: disk full on device\n"
        result = _filter_noise(text)
        assert "Error: disk full" in result

    def test_re_search_matches_midline(self):
        """re.search should match warning even mid-line after whitespace."""
        text = "  [rank0] WARNING: some warning\nreal error\n"
        result = _filter_noise(text)
        assert "WARNING" not in result


# ============================================================================
# _last_error_hash tests
# ============================================================================

class TestLastErrorHash:
    def test_empty_file(self, tmp_path: Path):
        log = tmp_path / "empty.log"
        log.write_text("", encoding="utf-8")
        assert _last_error_hash(log) == ""

    def test_non_existent_file(self, tmp_path: Path):
        assert _last_error_hash(tmp_path / "nope.log") == ""

    def test_reproducible_hash(self, tmp_path: Path):
        log = tmp_path / "err.log"
        log.write_text(
            "Traceback (most recent call last):\n"
            "  File 'train.py', line 42, in forward\n"
            "RuntimeError: CUDA out of memory\n",
            encoding="utf-8",
        )
        h1 = _last_error_hash(log)
        h2 = _last_error_hash(log)
        assert h1 == h2
        assert h1 != ""

    def test_different_errors_produce_different_hash(self, tmp_path: Path):
        log1 = tmp_path / "err1.log"
        log1.write_text("RuntimeError: CUDA out of memory\n")
        log2 = tmp_path / "err2.log"
        log2.write_text("FileNotFoundError: no such file\n")
        assert _last_error_hash(log1) != _last_error_hash(log2)

    def test_skips_torchrun_boilerplate(self, tmp_path: Path):
        """Lines with error_file or ChildFailedError should not contribute to hash."""
        log = tmp_path / "torchrun.log"
        log.write_text(
            "error_file: /tmp/error.json\n"
            "ChildFailedError: rank 0 exited with code 1\n"
            "Traceback (most recent call last):\n"
            "RuntimeError: actual training error\n",
            encoding="utf-8",
        )
        h = _last_error_hash(log)
        assert h != ""


# ============================================================================
# apply_oom_batch_backoff tests
# ============================================================================

class TestOomBackoff:
    def test_reduces_batch_size(self):
        exp: Dict[str, Any] = {
            "train_opts": {"batch_size": 8},
            "oom_batch_candidates": [8, 4, 2, 1],
        }
        assert apply_oom_batch_backoff(exp)
        assert exp["train_opts"]["batch_size"] == 4
        assert exp["last_oom_batch_size"] == 4

    def test_at_minimum_no_change(self):
        exp: Dict[str, Any] = {
            "train_opts": {"batch_size": 1},
            "oom_batch_candidates": [4, 2, 1],
        }
        assert not apply_oom_batch_backoff(exp)

    def test_current_none_returns_false(self):
        """
        When batch_size isn't in train_opts, we can't guess the real value
        (shell scripts have their own defaults), so we must return False.
        """
        exp: Dict[str, Any] = {
            "train_opts": {"lr": 0.0005},
            "oom_batch_candidates": [8, 4, 2, 1],
        }
        assert not apply_oom_batch_backoff(exp)

    def test_no_candidates_returns_false(self):
        exp: Dict[str, Any] = {
            "train_opts": {"batch_size": 8},
            "oom_batch_candidates": [],
        }
        assert not apply_oom_batch_backoff(exp)

    def test_other_batch_key(self):
        exp: Dict[str, Any] = {
            "train_opts": {"per_device_train_batch_size": 16},
            "oom_batch_candidates": [16, 8, 4],
        }
        assert apply_oom_batch_backoff(exp)
        assert exp["train_opts"]["per_device_train_batch_size"] == 8


# ============================================================================
# resolve_master_port tests
# ============================================================================

class TestResolveMasterPort:
    def test_positive_integer(self):
        exp = {"master_port": 12345}
        assert resolve_master_port(exp) == 12345

    def test_zero_uses_random(self):
        exp = {"master_port": 0}
        port = resolve_master_port(exp)
        assert 1024 <= port <= 65535

    def test_string_auto(self):
        exp = {"master_port": "auto"}
        port = resolve_master_port(exp)
        assert 1024 <= port <= 65535

    def test_empty_string(self):
        exp = {"master_port": ""}
        port = resolve_master_port(exp)
        assert 1024 <= port <= 65535

    def test_missing_key(self):
        exp: Dict[str, Any] = {}
        port = resolve_master_port(exp)
        assert 1024 <= port <= 65535

    def test_invalid_value(self):
        exp = {"master_port": "not_a_number"}
        port = resolve_master_port(exp)
        assert 1024 <= port <= 65535  # falls back to random


# ============================================================================
# build_exp tests
# ============================================================================

class TestBuildExp:
    def test_minimal_spec(self):
        spec = {"key": "test_exp", "cmd": "/tmp/test.sh"}
        exp = build_exp(spec, seq=1)
        assert exp["key"] == "test_exp"
        assert exp["seq"] == 1
        assert exp["status"] == "pending"
        assert exp["attempt"] == 0
        assert "exp_id" in exp

    def test_overrides_defaults(self):
        spec = {
            "key": "test_exp",
            "cmd": "/tmp/test.sh",
            "max_retries": 5,
            "train_timeout": 3600,
        }
        exp = build_exp(spec, seq=3)
        assert exp["max_retries"] == 5
        assert exp["train_timeout"] == 3600

    def test_train_opts_preserved(self):
        spec = {"key": "test_exp", "cmd": "/tmp/test.sh", "train_opts": {"lr": 0.001}}
        exp = build_exp(spec, seq=1)
        assert exp["train_opts"] == {"lr": 0.001}


# ============================================================================
# _prepare_environment tests
# ============================================================================

class TestPrepareEnvironment:
    def test_env_vars_set(self, tmp_path: Path):
        exp = {
            "exp_id": "test_123",
            "cmd_type": "bash",
            "cmd": "echo hello",
            "gpus": "0,1",
            "train_opts": {"lr": 0.0005, "batch_size": 8},
        }
        env, train_cmd = _prepare_environment(exp, tmp_path, 1)
        assert env["CUDA_VISIBLE_DEVICES"] == "0,1"
        assert env["HF_ENDPOINT"] == "https://hf-mirror.com"
        assert env["AUTOPIPE_EXP_ID"] == "test_123"
        assert env["AUTOPIPE_ATTEMPT"] == "1"
        assert env["TRAIN_LR"] == "0.0005"
        assert env["TRAIN_BATCH_SIZE"] == "8"

    def test_bash_cmd_type(self, tmp_path: Path):
        exp = {
            "exp_id": "test",
            "cmd_type": "bash",
            "cmd": "/tmp/test.sh --lr 0.001",
        }
        env, train_cmd = _prepare_environment(exp, tmp_path, 1)
        assert train_cmd[0] == "bash"
        assert "/tmp/test.sh" in train_cmd

    def test_torchrun_cmd_type(self, tmp_path: Path):
        exp = {
            "exp_id": "test",
            "cmd_type": "torchrun",
            "cfg_path": "configs/test.json",
            "train_opts": {"lr": 0.001},
            "nproc": 2,
        }
        env, train_cmd = _prepare_environment(exp, tmp_path, 1)
        assert "torch.distributed.run" in train_cmd
        assert "--nproc_per_node=2" in train_cmd

    def test_pythonpath_set(self, tmp_path: Path):
        repo_root = tmp_path / "repo"
        exp = {
            "exp_id": "test",
            "cmd_type": "bash",
            "cmd": "echo x",
        }
        env, _ = _prepare_environment(exp, repo_root, 1)
        assert str(repo_root) in env["PYTHONPATH"]

    def test_existing_pythonpath_preserved(self, tmp_path: Path, monkeypatch):
        monkeypatch.setenv("PYTHONPATH", "/some/existing/path")
        import importlib, autopipe.worker
        importlib.reload(autopipe.worker)
        # The function uses os.environ.copy(), so we need to make sure it's set
        # in the env passed in. Since _prepare_environment does os.environ.copy(),
        # it will pick up the monkeypatched env.

        exp = {
            "exp_id": "test",
            "cmd_type": "bash",
            "cmd": "echo x",
        }
        env, _ = autopipe.worker._prepare_environment(exp, tmp_path, 1)
        assert "/some/existing/path" in env["PYTHONPATH"]
        assert str(tmp_path) in env["PYTHONPATH"]


# ============================================================================
# RecoveryManager.handle_failure tests
# ============================================================================

class TestRecoveryManager:
    """Test the three action paths through handle_failure."""

    @pytest.fixture
    def mock_agent(self, monkeypatch):
        """Replace run_agent & snapshot_git with no-ops."""
        monkeypatch.setattr("autopipe.worker.run_agent", lambda **kwargs: 0)
        monkeypatch.setattr("autopipe.worker.snapshot_git", lambda *args, **kwargs: None)

    def _make_ctx(self, exp, tmp_path, reason="other", error_hash="abc123", attempt=1, rc=1):
        from autopipe.worker import FailureContext
        run_exp_path = tmp_path / "exp.json"
        status_path = tmp_path / "status.json"
        return FailureContext(
            exp=exp,
            run_exp_path=run_exp_path,
            status_path=status_path,
            attempt=attempt,
            rc=rc,
            reason=reason,
            error_hash=error_hash,
        )

    def test_oom_backoff_path(self, tmp_path, mock_agent):
        """OOM with batch_size reducible → OOM_BACKOFF."""
        from autopipe.worker import RecoveryManager, RecoveryAction
        exp = {
            "train_opts": {"batch_size": 8},
            "oom_batch_candidates": [8, 4, 2, 1],
            "max_retries": 2,
            "consecutive_failures": 0,
        }
        ctx = self._make_ctx(exp, tmp_path, reason="oom")
        mgr = RecoveryManager(tmp_path, tmp_path, 600)
        action, out_exp = mgr.handle_failure(ctx)
        assert action == RecoveryAction.OOM_BACKOFF
        assert out_exp["status"] == "pending"
        assert out_exp["train_opts"]["batch_size"] == 4
        assert out_exp["attempt"] == 0  # decremented so scheduler retries

    def test_oom_at_min_hard_failure(self, tmp_path, mock_agent):
        """OOM at minimum batch_size repeatedly → HARD_FAILURE."""
        from autopipe.worker import RecoveryManager, RecoveryAction
        exp = {
            "train_opts": {"batch_size": 1},
            "oom_batch_candidates": [2, 1],
            "max_oom_retries": 2,
            "max_retries": 2,
            "oom_at_min_count": 2,  # already failed twice at min
        }
        ctx = self._make_ctx(exp, tmp_path, reason="oom")
        mgr = RecoveryManager(tmp_path, tmp_path, 600)
        action, out_exp = mgr.handle_failure(ctx)
        assert action == RecoveryAction.HARD_FAILURE
        assert out_exp["status"] == "hard_failure"

    def test_agent_duplicate_hash_hard_failure(self, tmp_path, mock_agent):
        """Same error hash hitting hard_failure_threshold → HARD_FAILURE."""
        from autopipe.worker import RecoveryManager, RecoveryAction
        exp = {
            "train_opts": {"lr": 0.0005},
            "max_retries": 3,
            "hard_failure_threshold": 2,
            "error_hash": "abc123",  # same hash
            "agent_fix_hashes": {"abc123": 1},  # already tried once
        }
        ctx = self._make_ctx(exp, tmp_path, reason="nan", error_hash="abc123", attempt=2)
        mgr = RecoveryManager(tmp_path, tmp_path, 600)
        action, out_exp = mgr.handle_failure(ctx)
        assert action == RecoveryAction.HARD_FAILURE
        assert "agent failed to fix" in out_exp["last_reason"]

    def test_failed_path_retries_remaining(self, tmp_path, mock_agent):
        """First-seen error with retries → FAILED (let scheduler retry)."""
        from autopipe.worker import RecoveryManager, RecoveryAction
        exp = {
            "train_opts": {"lr": 0.0005},
            "max_retries": 3,
            "hard_failure_threshold": 3,
        }
        ctx = self._make_ctx(exp, tmp_path, reason="nan", error_hash="def456", attempt=1)
        mgr = RecoveryManager(tmp_path, tmp_path, 600)
        action, out_exp = mgr.handle_failure(ctx)
        assert action == RecoveryAction.FAILED

    def test_failed_path_no_retries(self, tmp_path, mock_agent):
        """Attempt > max_retries → FAILED (scheduler will mark aborted)."""
        from autopipe.worker import RecoveryManager, RecoveryAction
        exp = {
            "train_opts": {"lr": 0.0005},
            "max_retries": 1,
        }
        ctx = self._make_ctx(exp, tmp_path, reason="nan", error_hash="ghi789", attempt=2)
        mgr = RecoveryManager(tmp_path, tmp_path, 600)
        action, out_exp = mgr.handle_failure(ctx)
        assert action == RecoveryAction.FAILED
        # Agent should NOT have run (attempt > max_retries).
        assert "agent_fix_hashes" not in out_exp

    def test_oom_backoff_exhausted_falls_through(self, tmp_path, mock_agent):
        """oom_backoff_count >= max_oom_retries → rollback batch_size, go to FAILED."""
        from autopipe.worker import RecoveryManager, RecoveryAction
        exp = {
            "train_opts": {"batch_size": 4},
            "oom_batch_candidates": [8, 4, 2],
            "max_oom_retries": 2,
            "oom_backoff_count": 2,
            "max_retries": 2,
        }
        ctx = self._make_ctx(exp, tmp_path, reason="oom")
        mgr = RecoveryManager(tmp_path, tmp_path, 600)
        action, out_exp = mgr.handle_failure(ctx)
        # Backoff exhausted → batch_size mutation is rolled back so the stale
        # smaller value doesn't leak into agent / hard_failure path.
        assert action == RecoveryAction.FAILED
        assert out_exp["train_opts"]["batch_size"] == 4  # rolled back, not reduced
        assert "last_oom_batch_size" not in out_exp        # cleaned up
