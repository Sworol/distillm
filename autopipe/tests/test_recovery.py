from __future__ import annotations

import json
import os
import time
from pathlib import Path

from autopipe.recovery import recover_stale_worker


class TestRecoverStaleWorker:
    """Tests for recover_stale_worker covering all 4 cases and edge cases."""

    def test_case1_terminal_status_success(self, tmp_path: Path) -> None:
        """Case 1: status.json says 'success' — sync exp.json, clean lock."""
        run_root = tmp_path / "runs" / "test_001"
        run_root.mkdir(parents=True, exist_ok=True)
        run_exp_path = run_root / "exp.json"
        run_exp_path.write_text('{"exp_id": "test_001"}', encoding="utf-8")
        status_path = run_root / "status.json"
        status_path.write_text(
            json.dumps({"status": "success", "attempt": 3, "exit_code": 0}),
            encoding="utf-8",
        )
        lock_path = run_root / ".lock_worker"
        queue_exp = {"exp_id": "test_001", "key": "test"}

        assert recover_stale_worker(run_exp_path, queue_exp, lock_path, status_path) is True

        updated = json.loads(run_exp_path.read_text())
        assert updated["status"] == "success"
        assert updated["attempt"] == 3

    def test_case1_terminal_status_failed(self, tmp_path: Path) -> None:
        """Case 1: status.json says 'failed' — sync exp.json, clean lock."""
        run_root = tmp_path / "runs" / "test_001"
        run_root.mkdir(parents=True, exist_ok=True)
        run_exp_path = run_root / "exp.json"
        run_exp_path.write_text('{"exp_id": "test_001"}', encoding="utf-8")
        status_path = run_root / "status.json"
        status_path.write_text(
            json.dumps({"status": "failed", "attempt": 2, "exit_code": 1, "reason": "oom"}),
            encoding="utf-8",
        )
        lock_path = run_root / ".lock_worker"
        lock_path.write_text("pid=99999\nts=old\n")
        queue_exp = {"exp_id": "test_001", "key": "test"}

        assert recover_stale_worker(run_exp_path, queue_exp, lock_path, status_path) is True

        updated = json.loads(run_exp_path.read_text())
        assert updated["status"] == "failed"
        assert updated["last_exit_code"] == 1
        assert updated["last_reason"] == "oom"

    def test_case2_running_no_lock_marks_failed(self, tmp_path: Path) -> None:
        """Case 2: status.json says 'running' but worker lock is missing."""
        run_root = tmp_path / "runs" / "test_001"
        run_root.mkdir(parents=True, exist_ok=True)
        run_exp_path = run_root / "exp.json"
        run_exp_path.write_text('{"exp_id": "test_001"}', encoding="utf-8")
        status_path = run_root / "status.json"
        status_path.write_text(
            json.dumps({"status": "running", "attempt": 1}),
            encoding="utf-8",
        )
        lock_path = run_root / ".lock_worker"  # does not exist
        queue_exp = {"exp_id": "test_001", "key": "test"}

        assert recover_stale_worker(run_exp_path, queue_exp, lock_path, status_path) is True

        updated = json.loads(run_exp_path.read_text())
        assert updated["status"] == "failed"
        assert updated["last_reason"] == "stale_worker"

        updated_status = json.loads(status_path.read_text())
        assert updated_status["status"] == "failed"
        assert updated_status["reason"] == "stale_worker"

    def test_case3_no_status_no_lock_old_mtime(self, tmp_path: Path) -> None:
        """Case 3: no status.json, no lock, old mtime on run_exp."""
        run_root = tmp_path / "runs" / "test_001"
        run_root.mkdir(parents=True, exist_ok=True)
        run_exp_path = run_root / "exp.json"
        run_exp_path.write_text('{"exp_id": "test_001"}', encoding="utf-8")
        old_time = time.time() - 600
        os.utime(str(run_exp_path), (old_time, old_time))

        status_path = run_root / "status.json"  # does not exist
        lock_path = run_root / ".lock_worker"  # does not exist
        queue_exp = {"exp_id": "test_001", "key": "test"}

        assert recover_stale_worker(run_exp_path, queue_exp, lock_path, status_path) is True

        updated = json.loads(run_exp_path.read_text())
        assert updated["status"] == "failed"
        assert updated["last_reason"] == "stale_worker"
        assert updated["consecutive_failures"] == 1

    def test_case3_fresh_exp_not_recovered(self, tmp_path: Path) -> None:
        """Case 3: no status.json, no lock, but exp is fresh — do NOT recover."""
        run_root = tmp_path / "runs" / "test_001"
        run_root.mkdir(parents=True, exist_ok=True)
        run_exp_path = run_root / "exp.json"
        run_exp_path.write_text('{"exp_id": "test_001"}', encoding="utf-8")

        status_path = run_root / "status.json"  # does not exist
        lock_path = run_root / ".lock_worker"  # does not exist
        queue_exp = {"exp_id": "test_001", "key": "test"}

        assert recover_stale_worker(run_exp_path, queue_exp, lock_path, status_path) is False

    def test_healthy_worker_returns_false(self, tmp_path: Path) -> None:
        """Worker with running status and live lock should not be recovered."""
        run_root = tmp_path / "runs" / "test_001"
        run_root.mkdir(parents=True, exist_ok=True)
        run_exp_path = run_root / "exp.json"
        run_exp_path.write_text('{"exp_id": "test_001", "status": "running"}', encoding="utf-8")
        status_path = run_root / "status.json"
        status_path.write_text(
            json.dumps({"status": "running", "attempt": 1}),
            encoding="utf-8",
        )
        lock_path = run_root / ".lock_worker"
        lock_path.write_text(f"pid={os.getpid()}\nts=now\n")
        queue_exp = {"exp_id": "test_001", "key": "test"}

        assert recover_stale_worker(run_exp_path, queue_exp, lock_path, status_path) is False

    def test_corrupted_status_json_handled(self, tmp_path: Path) -> None:
        """Corrupted status.json should not crash — falls through to Case 3."""
        run_root = tmp_path / "runs" / "test_001"
        run_root.mkdir(parents=True, exist_ok=True)
        run_exp_path = run_root / "exp.json"
        run_exp_path.write_text('{"exp_id": "test_001"}', encoding="utf-8")
        old_time = time.time() - 600
        os.utime(str(run_exp_path), (old_time, old_time))

        status_path = run_root / "status.json"
        status_path.write_text("{broken json", encoding="utf-8")
        os.utime(str(status_path), (old_time, old_time))  # also make status.mtime old

        lock_path = run_root / ".lock_worker"
        queue_exp = {"exp_id": "test_001", "key": "test"}
        result = recover_stale_worker(run_exp_path, queue_exp, lock_path, status_path)
        assert result is True

        updated = json.loads(run_exp_path.read_text())
        assert updated["status"] == "failed"
        assert updated["last_reason"] == "stale_worker"

    def test_case4_dead_lock_no_status_file(self, tmp_path: Path) -> None:
        """Case 4: lock exists with dead PID, no status.json — marks stale.

        This path is reachable when the worker died mid-startup (created
        the lock file but never wrote status.json). The lock file is cleaned
        and the experiment is marked as failed (stale_worker).
        """
        run_root = tmp_path / "runs" / "test_001"
        run_root.mkdir(parents=True, exist_ok=True)
        run_exp_path = run_root / "exp.json"
        run_exp_path.write_text('{"exp_id": "test_001"}', encoding="utf-8")
        old_time = time.time() - 600
        os.utime(str(run_exp_path), (old_time, old_time))

        status_path = run_root / "status.json"  # does not exist
        lock_path = run_root / ".lock_worker"
        lock_path.write_text("pid=99999\nts=old\n")  # dead PID
        os.utime(str(lock_path), (old_time, old_time))

        queue_exp = {"exp_id": "test_001", "key": "test"}
        result = recover_stale_worker(run_exp_path, queue_exp, lock_path, status_path)
        assert result is True

        updated = json.loads(run_exp_path.read_text())
        assert updated["status"] == "failed"
        assert updated["last_reason"] == "stale_worker"
        assert not lock_path.exists()  # lock cleaned

    def test_orphan_lock_cleaned_before_case_analysis(self, tmp_path: Path) -> None:
        """Dead-PID lock should be cleaned even when status.json says 'running'.

        The pre-check in recover_stale_worker removes dead-PID locks
        so that Case 2 (running + no lock) triggers correctly.
        """
        run_root = tmp_path / "runs" / "test_001"
        run_root.mkdir(parents=True, exist_ok=True)
        run_exp_path = run_root / "exp.json"
        run_exp_path.write_text('{"exp_id": "test_001"}', encoding="utf-8")
        status_path = run_root / "status.json"
        status_path.write_text(
            json.dumps({"status": "running", "attempt": 1}),
            encoding="utf-8",
        )
        lock_path = run_root / ".lock_worker"
        lock_path.write_text("pid=99999\nts=old\n")  # dead PID

        queue_exp = {"exp_id": "test_001", "key": "test"}
        assert recover_stale_worker(run_exp_path, queue_exp, lock_path, status_path) is True
        assert not lock_path.exists()
