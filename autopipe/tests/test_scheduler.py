from __future__ import annotations

import datetime
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from unittest import mock

import pytest

from autopipe.scheduler import (
    _in_active_window,
    _kill_workers_signaller,
    _parse_time_window,
    _load_run_exp,
    _reap_workers,
    _terminate_workers,
    list_queue,
    load_exp,
)


class TestListQueue:
    def test_returns_sorted_json_files(self, tmp_path: Path) -> None:
        (tmp_path / "02_bar.json").write_text("{}", encoding="utf-8")
        (tmp_path / "01_foo.json").write_text("{}", encoding="utf-8")
        (tmp_path / "03_baz.json").write_text("{}", encoding="utf-8")
        # Non-JSON files should be excluded
        (tmp_path / "notes.txt").write_text("hello", encoding="utf-8")
        files = list_queue(tmp_path)
        assert len(files) == 3
        assert files[0].name == "01_foo.json"
        assert files[1].name == "02_bar.json"
        assert files[2].name == "03_baz.json"

    def test_empty_directory(self, tmp_path: Path) -> None:
        assert list_queue(tmp_path) == []


class TestLoadExp:
    def test_loads_valid_json(self, tmp_path: Path) -> None:
        p = tmp_path / "exp.json"
        p.write_text('{"exp_id": "test_001", "key": "test"}', encoding="utf-8")
        exp = load_exp(p)
        assert exp["exp_id"] == "test_001"
        assert exp["status"] == "pending"
        assert exp["attempt"] == 0

    def test_preserves_existing_status(self, tmp_path: Path) -> None:
        p = tmp_path / "exp.json"
        p.write_text('{"exp_id": "test_001", "status": "success", "attempt": 5}', encoding="utf-8")
        exp = load_exp(p)
        assert exp["status"] == "success"
        assert exp["attempt"] == 5

    def test_missing_file(self, tmp_path: Path) -> None:
        exp = load_exp(tmp_path / "nonexistent.json")
        assert exp == {}

    def test_corrupted_json(self, tmp_path: Path) -> None:
        p = tmp_path / "corrupted.json"
        p.write_text('{"exp_id": "broken', encoding="utf-8")  # truncated JSON
        exp = load_exp(p)
        assert exp == {}

    def test_invalid_json(self, tmp_path: Path) -> None:
        p = tmp_path / "invalid.json"
        p.write_text("not json at all", encoding="utf-8")
        exp = load_exp(p)
        assert exp == {}

    def test_empty_file(self, tmp_path: Path) -> None:
        p = tmp_path / "empty.json"
        p.write_text("", encoding="utf-8")
        exp = load_exp(p)
        assert exp == {}


class TestLoadRunExp:
    def test_no_run_exp_yet(self, tmp_path: Path) -> None:
        queue_exp = {"exp_id": "test_001", "key": "test"}
        run_exp_path = tmp_path / "runs" / "test_001" / "exp.json"
        result = _load_run_exp(queue_exp, run_exp_path)
        assert result == queue_exp
        assert result is not queue_exp  # must be a copy

    def test_loads_existing_run_exp(self, tmp_path: Path) -> None:
        queue_exp = {"exp_id": "test_001", "key": "test"}
        run_root = tmp_path / "runs" / "test_001"
        run_root.mkdir(parents=True, exist_ok=True)
        run_exp_path = run_root / "exp.json"
        run_exp = {"exp_id": "test_001", "key": "test", "attempt": 3, "status": "running"}
        run_exp_path.write_text(json.dumps(run_exp), encoding="utf-8")
        result = _load_run_exp(queue_exp, run_exp_path)
        assert result["exp_id"] == "test_001"
        assert result["attempt"] == 3
        assert result["status"] == "running"

    def test_mismatched_exp_id_falls_back(self, tmp_path: Path) -> None:
        queue_exp = {"exp_id": "queue_001"}
        run_root = tmp_path / "runs" / "other_001"
        run_root.mkdir(parents=True, exist_ok=True)
        run_exp_path = run_root / "exp.json"
        run_exp_path.write_text('{"exp_id": "other_001"}', encoding="utf-8")
        result = _load_run_exp(queue_exp, run_exp_path)
        assert result == queue_exp

    def test_missing_exp_id_in_run_falls_back(self, tmp_path: Path) -> None:
        queue_exp = {"exp_id": "queue_001"}
        run_root = tmp_path / "runs" / "queue_001"
        run_root.mkdir(parents=True, exist_ok=True)
        run_exp_path = run_root / "exp.json"
        run_exp_path.write_text('{"status": "running"}', encoding="utf-8")
        result = _load_run_exp(queue_exp, run_exp_path)
        assert result == queue_exp

    def test_corrupted_run_exp_falls_back(self, tmp_path: Path) -> None:
        queue_exp = {"exp_id": "test_001", "key": "test"}
        run_root = tmp_path / "runs" / "test_001"
        run_root.mkdir(parents=True, exist_ok=True)
        run_exp_path = run_root / "exp.json"
        run_exp_path.write_text('{"exp_id": "test_001", broken', encoding="utf-8")  # truncated
        result = _load_run_exp(queue_exp, run_exp_path)
        assert result == queue_exp  # falls back gracefully

    def test_deleted_between_exists_and_read(self, tmp_path: Path) -> None:
        """Simulate TOCTOU: file exists but read raises FileNotFoundError."""
        queue_exp = {"exp_id": "test_001"}
        run_root = tmp_path / "runs" / "test_001"
        run_root.mkdir(parents=True, exist_ok=True)
        run_exp_path = run_root / "exp.json"
        # Write then immediately remove so exists() returns True but read_json fails
        run_exp_path.write_text('{"exp_id": "test_001"}', encoding="utf-8")
        run_exp_path.unlink()
        result = _load_run_exp(queue_exp, run_exp_path)
        assert result == queue_exp


class TestParseTimeWindow:
    def test_valid_window(self) -> None:
        assert _parse_time_window("22:00-08:00") == (1320, 480)

    def test_same_day_window(self) -> None:
        assert _parse_time_window("09:00-17:00") == (540, 1020)

    def test_invalid_format_no_match(self) -> None:
        assert _parse_time_window("invalid") is None

    def test_invalid_format_wrong_separator(self) -> None:
        assert _parse_time_window("22:00~08:00") is None

    def test_empty_string(self) -> None:
        assert _parse_time_window("") is None

    def test_non_numeric(self) -> None:
        assert _parse_time_window("ab:cd-ef:gh") is None

    def test_start_equals_end(self) -> None:
        assert _parse_time_window("12:00-12:00") is None

    def test_single_digit_hour(self) -> None:
        assert _parse_time_window("6:00-22:00") == (360, 1320)


class TestInActiveWindow:
    def test_none_window_always_active(self) -> None:
        assert _in_active_window(None) is True

    def test_currently_in_window(self) -> None:
        """Use a very wide window that definitely contains the current time."""
        window = (0, 1439)  # 00:00-23:59
        assert _in_active_window(window) is True

    def test_currently_outside_window(self) -> None:
        """Use a window that does not contain the current time."""
        import datetime
        now = datetime.datetime.now()
        target_end = (now.hour * 60 + now.minute) - 1
        if target_end < 0:
            target_end = 0
        window = (0, target_end)
        # If target_end is 0, always true, so skip in that case
        if target_end > 0:
            assert _in_active_window(window) is False

    def test_overnight_window(self) -> None:
        """Overnight window 22:00-08:00: active at 23:00, inactive at 14:00."""
        with mock.patch("autopipe.scheduler.datetime") as mock_dt:
            mock_dt.datetime.now.return_value = datetime.datetime(2024, 1, 1, 23, 0)
            # 22:00-08:00 overnight — 23:00 is inside the window
            assert _in_active_window((1320, 480)) is True

        with mock.patch("autopipe.scheduler.datetime") as mock_dt:
            mock_dt.datetime.now.return_value = datetime.datetime(2024, 1, 1, 14, 0)
            # 14:00 is NOT inside 22:00-08:00
            assert _in_active_window((1320, 480)) is False

    def test_same_day_window(self) -> None:
        """Same-day window 09:00-17:00: active at 12:00, inactive at 08:00."""
        with mock.patch("autopipe.scheduler.datetime") as mock_dt:
            mock_dt.datetime.now.return_value = datetime.datetime(2024, 1, 1, 12, 0)
            assert _in_active_window((540, 1020)) is True

        with mock.patch("autopipe.scheduler.datetime") as mock_dt:
            mock_dt.datetime.now.return_value = datetime.datetime(2024, 1, 1, 8, 0)
            assert _in_active_window((540, 1020)) is False


class TestTerminateWorkers:
    """Tests for _terminate_workers — window-kill termination path."""

    def test_sends_sigterm_to_running_workers(self) -> None:
        """_terminate_workers sends SIGTERM to each running worker via send_signal."""
        p1 = mock.MagicMock(spec=subprocess.Popen)
        p1.poll.return_value = None  # still running
        p2 = mock.MagicMock(spec=subprocess.Popen)
        p2.poll.return_value = None  # still running
        workers = {"exp_1": p1, "exp_2": p2}

        _terminate_workers(workers)

        p1.send_signal.assert_called_once_with(signal.SIGTERM)
        p2.send_signal.assert_called_once_with(signal.SIGTERM)

    def test_skips_exited_workers(self) -> None:
        """Workers that have already exited should not receive SIGTERM."""
        p1 = mock.MagicMock(spec=subprocess.Popen)
        p1.poll.return_value = 0  # already exited
        workers = {"exp_1": p1}

        _terminate_workers(workers)

        p1.send_signal.assert_not_called()

    def test_handles_exception_gracefully(self) -> None:
        """Exception during send_signal should not propagate."""
        p1 = mock.MagicMock(spec=subprocess.Popen)
        p1.poll.return_value = None
        p1.send_signal.side_effect = OSError("no such process")
        workers = {"exp_1": p1}

        _terminate_workers(workers)  # should not raise

    def test_empty_workers_no_error(self) -> None:
        """Empty workers dict should not cause errors."""
        _terminate_workers({})  # should not raise


class TestKillWorkersSignaller:
    """Tests for _kill_workers_signaller — scheduler shutdown handler."""

    def test_sends_sigterm_then_sigkill_after_timeout(self) -> None:
        """Signaller sends SIGTERM, waits, then SIGKILLs survivors."""
        p1 = mock.MagicMock(spec=subprocess.Popen)
        # Never exits — stays running through all 6 poll checks, forcing SIGKILL
        p1.poll.return_value = None
        workers = {"exp_1": p1}

        with mock.patch.object(sys, "exit") as mock_exit:
            with mock.patch("time.sleep"):  # speed up test
                _kill_workers_signaller(workers, signum=signal.SIGTERM)

        # send_signal called: SIGTERM first, then SIGKILL
        sigterm_calls = [c for c in p1.send_signal.call_args_list if c == mock.call(signal.SIGTERM)]
        sigkill_calls = [c for c in p1.send_signal.call_args_list if c == mock.call(signal.SIGKILL)]
        assert len(sigterm_calls) >= 1
        assert len(sigkill_calls) >= 1
        mock_exit.assert_called_once_with(128 + signal.SIGTERM)

    def test_all_workers_exit_gracefully_no_sigkill(self) -> None:
        """If workers exit after SIGTERM, no SIGKILL needed."""
        p1 = mock.MagicMock(spec=subprocess.Popen)
        p1.poll.side_effect = [None, 0, 0, 0, 0, 0, 0]  # exits after first check
        workers = {"exp_1": p1}

        with mock.patch.object(sys, "exit") as mock_exit:
            with mock.patch("time.sleep"):
                _kill_workers_signaller(workers, signum=signal.SIGINT)

        # Only SIGTERM (no SIGKILL)
        p1.send_signal.assert_called_once_with(signal.SIGTERM)
        mock_exit.assert_called_once_with(128 + signal.SIGINT)

    def test_no_signum_defaults_to_sigterm_exit_code(self) -> None:
        """When signum is None, exit code defaults to 128 + SIGTERM."""
        workers: dict = {}

        with mock.patch.object(sys, "exit") as mock_exit:
            _kill_workers_signaller(workers)

        mock_exit.assert_called_once_with(128 + signal.SIGTERM)

    def test_handles_exception_gracefully(self) -> None:
        """Exceptions during send_signal should not propagate."""
        p1 = mock.MagicMock(spec=subprocess.Popen)
        p1.poll.side_effect = [None, None, 0]
        p1.send_signal.side_effect = OSError("no such process")
        workers = {"exp_1": p1}

        with mock.patch.object(sys, "exit"):
            _kill_workers_signaller(workers, signum=signal.SIGTERM)  # should not raise


class TestReapWorkers:
    """Tests for _reap_workers."""

    def test_removes_finished_workers(self) -> None:
        p1 = mock.MagicMock(spec=subprocess.Popen)
        p1.poll.return_value = 0  # finished
        p2 = mock.MagicMock(spec=subprocess.Popen)
        p2.poll.return_value = None  # still running
        workers = {"exp_1": p1, "exp_2": p2}

        _reap_workers(workers)

        assert "exp_1" not in workers
        assert "exp_2" in workers

    def test_empty_dict_no_error(self) -> None:
        workers: dict = {}
        _reap_workers(workers)
        assert workers == {}


class TestLoadRunExpMergeKeys:
    """Tests that CONFIG_MERGE_KEYS are applied correctly in _load_run_exp."""

    def test_new_keys_from_queue_applied_via_setdefault(self, tmp_path: Path) -> None:
        """Phase 1: new keys from queue are added via setdefault (don't override)."""
        from autopipe.config import CONFIG_MERGE_KEYS

        queue_exp = {k: f"q_{k}" for k in list(CONFIG_MERGE_KEYS)[:3]}
        queue_exp["exp_id"] = "test_001"
        run_root = tmp_path / "runs" / "test_001"
        run_root.mkdir(parents=True, exist_ok=True)
        run_exp_path = run_root / "exp.json"
        run_exp = {"exp_id": "test_001", "attempt": 3}
        run_exp_path.write_text(json.dumps(run_exp), encoding="utf-8")

        result = _load_run_exp(queue_exp, run_exp_path)

        # Existing keys preserved
        assert result["attempt"] == 3
        # New keys from queue applied
        for k in list(CONFIG_MERGE_KEYS)[:3]:
            assert k in result, f"Key {k} should be in result"
            assert result[k] == f"q_{k}"


class TestLoadExpCorruptedEdgeCases:
    """Tests for heartbeat handling of corrupted/missing status files."""

    def test_heartbeat_with_corrupted_status_json(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test the heartbeat path doesn't crash on corrupted status.json."""
        from autopipe.io_utils import read_json

        # Create a synthetic experiment with corrupted status.json
        queue_dir = tmp_path / "queue"
        queue_dir.mkdir()
        runs_dir = tmp_path / "runs"
        runs_dir.mkdir()

        # Write queue entry
        queue_path = queue_dir / "01_test.json"
        queue_path.write_text(
            json.dumps({"exp_id": "test_001", "key": "test", "status": "pending"})
        )

        # Write a corrupted status.json
        exp_dir = runs_dir / "test_001"
        exp_dir.mkdir(parents=True)
        exp_path = exp_dir / "exp.json"
        exp_path.write_text(
            json.dumps({"exp_id": "test_001", "key": "test", "status": "pending"})
        )
        status_path = exp_dir / "status.json"
        status_path.write_text("{corrupted json", encoding="utf-8")

        # The heartbeat read_json should not raise ValueError
        # Verify read_json raises on corrupted data
        with pytest.raises((ValueError)):
            read_json(status_path)

        # Heartbeat handling: load_exp wraps in try/except and returns empty dict
        from autopipe.scheduler import load_exp as sched_load_exp
        result = sched_load_exp(status_path)
        assert result == {}  # corrupted files return empty dict
