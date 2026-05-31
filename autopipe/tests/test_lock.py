from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from autopipe.io_utils import Lock


class TestLock:
    """Tests for Lock: acquire, release, owned, heartbeat, stale detection."""

    def test_acquire_and_release(self, tmp_path: Path) -> None:
        lock_path = tmp_path / ".lock_test"
        lock = Lock(lock_path)
        assert lock.acquire()
        assert lock.owned()
        assert lock_path.exists()
        lock.release()
        assert not lock_path.exists()

    def test_double_acquire_fails(self, tmp_path: Path) -> None:
        lock_path = tmp_path / ".lock_test"
        lock1 = Lock(lock_path)
        assert lock1.acquire()
        lock2 = Lock(lock_path)
        assert not lock2.acquire()
        lock1.release()

    def test_acquire_after_release(self, tmp_path: Path) -> None:
        lock_path = tmp_path / ".lock_test"
        lock1 = Lock(lock_path)
        assert lock1.acquire()
        lock1.release()
        lock2 = Lock(lock_path)
        assert lock2.acquire()
        lock2.release()

    def test_owned_false_when_no_lock(self, tmp_path: Path) -> None:
        lock = Lock(tmp_path / ".lock_nonexist")
        assert not lock.owned()

    def test_owned_false_after_release(self, tmp_path: Path) -> None:
        lock_path = tmp_path / ".lock_test"
        lock = Lock(lock_path)
        lock.acquire()
        lock.release()
        assert not lock.owned()

    def test_owned_false_when_different_pid(self, tmp_path: Path) -> None:
        lock_path = tmp_path / ".lock_test"
        # Write a lock file with a dead PID
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock_path.write_text("pid=99999\nts=2024-01-01 00:00:00\n")
        lock = Lock(lock_path)
        assert not lock.owned()

    def test_heartbeat_updates_timestamp(self, tmp_path: Path) -> None:
        lock_path = tmp_path / ".lock_test"
        lock = Lock(lock_path)
        lock.acquire()
        old_mtime = lock_path.stat().st_mtime
        time.sleep(0.1)
        lock.heartbeat()
        new_mtime = lock_path.stat().st_mtime
        assert new_mtime >= old_mtime
        lock.release()

    def test_stale_lock_acquired_by_new_process(self, tmp_path: Path) -> None:
        lock_path = tmp_path / ".lock_test"
        # Write a stale lock with a dead PID
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock_path.write_text("pid=99999\nts=2024-01-01 00:00:00\n")
        # Set old mtime
        old_time = time.time() - 25 * 3600  # 25 hours ago
        os.utime(str(lock_path), (old_time, old_time))

        lock = Lock(lock_path, stale_seconds=24 * 3600)
        assert lock.acquire()
        assert lock._pid == os.getpid()
        lock.release()

    def test_fresh_lock_not_stolen(self, tmp_path: Path) -> None:
        lock_path = tmp_path / ".lock_test"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock_path.write_text(f"pid={os.getpid()}\nts=now\n")

        lock = Lock(lock_path, stale_seconds=1)  # 1s stale timeout
        # We own the lock already (by PID), but acquire should fail
        # because the file already exists with a live PID
        assert not lock.acquire()

    def test_heartbeat_noop_when_pid_none(self, tmp_path: Path) -> None:
        lock = Lock(tmp_path / ".lock_test")
        # _pid is None — heartbeat should be a no-op without error
        lock.heartbeat()  # should not raise

    def test_read_lock_pid_valid(self, tmp_path: Path) -> None:
        lock_path = tmp_path / ".lock_test"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock_path.write_text("pid=42\nts=2024-01-01\n")
        assert Lock._read_lock_pid(lock_path) == 42

    def test_read_lock_pid_missing_file(self, tmp_path: Path) -> None:
        assert Lock._read_lock_pid(tmp_path / ".nonexistent") is None

    def test_read_lock_pid_no_pid_line(self, tmp_path: Path) -> None:
        lock_path = tmp_path / ".lock_test"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock_path.write_text("some garbage\nno pid here\n")
        assert Lock._read_lock_pid(lock_path) is None

    def test_pid_alive_self(self) -> None:
        assert Lock._pid_alive(os.getpid())

    def test_pid_alive_dead(self) -> None:
        # PID 1 is usually init/systemd — it exists
        # Use a very high PID that is extremely unlikely to exist
        assert not Lock._pid_alive(99999999)

    def test_acquire_cleans_stale_lock(self, tmp_path: Path) -> None:
        lock_path = tmp_path / ".lock_test"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        # Write a lock with a dead PID and old mtime
        lock_path.write_text("pid=99999\nts=2024-01-01 00:00:00\n")
        old_time = time.time() - 25 * 3600
        os.utime(str(lock_path), (old_time, old_time))

        lock = Lock(lock_path, stale_seconds=1)
        # Should be able to acquire by replacing the stale lock
        assert lock.acquire()
        assert lock._pid == os.getpid()
        lock.release()
