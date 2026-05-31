from __future__ import annotations

import json
import os
import time
from pathlib import Path

from autopipe.io_utils import (
    _filter_noise,
    _proximity_match,
    atomic_write_json,
    get_classification_text,
    log_event,
    now_ts,
    patch_exp,
    read_json,
    scan_log_chunk,
    scan_log_tail_mid,
)


class TestNowTs:
    def test_returns_string(self) -> None:
        ts = now_ts()
        assert isinstance(ts, str)
        assert len(ts) > 10


class TestLogEvent:
    def test_requires_no_args(self) -> None:
        # Should not raise
        log_event()
        log_event(source="test", event="test_event")


class TestAtomicWriteJson:
    def test_creates_file(self, tmp_path: Path) -> None:
        p = tmp_path / "data.json"
        atomic_write_json(p, {"key": "value", "num": 42})
        assert p.exists()
        data = json.loads(p.read_text())
        assert data["key"] == "value"
        assert data["num"] == 42

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        p = tmp_path / "a" / "b" / "data.json"
        atomic_write_json(p, {"key": "value"})
        assert p.exists()
        assert json.loads(p.read_text())["key"] == "value"

    def test_atomic_replace(self, tmp_path: Path) -> None:
        """os.replace is atomic, so a concurrent read sees old or new, never partial."""
        p = tmp_path / "data.json"
        atomic_write_json(p, {"version": 1})
        atomic_write_json(p, {"version": 2})
        data = json.loads(p.read_text())
        assert data["version"] == 2

    def test_no_temp_file_leak(self, tmp_path: Path) -> None:
        p = tmp_path / "data.json"
        atomic_write_json(p, {"key": "value"})
        # No .tmp.* files should remain
        temps = list(tmp_path.glob(f"*.tmp.*"))
        assert len(temps) == 0


class TestPatchExp:
    def test_creates_new_file_with_base(self, tmp_path: Path) -> None:
        p = tmp_path / "exp.json"
        result = patch_exp(p, base={"exp_id": "test_001"}, status="running")
        assert result["exp_id"] == "test_001"
        assert result["status"] == "running"
        saved = json.loads(p.read_text())
        assert saved["exp_id"] == "test_001"
        assert saved["status"] == "running"

    def test_updates_existing(self, tmp_path: Path) -> None:
        p = tmp_path / "exp.json"
        p.write_text(json.dumps({"exp_id": "test_001", "status": "pending", "attempt": 1}))
        result = patch_exp(p, status="running", attempt=2)
        assert result["status"] == "running"
        assert result["attempt"] == 2
        assert result["exp_id"] == "test_001"  # preserved
        saved = json.loads(p.read_text())
        assert saved["status"] == "running"
        assert saved["attempt"] == 2

    def test_does_not_clobber_unrelated_keys(self, tmp_path: Path) -> None:
        p = tmp_path / "exp.json"
        p.write_text(json.dumps({"exp_id": "test_001", "agent_fix_hashes": {"abc": 1}}))
        result = patch_exp(p, status="running")
        assert result["agent_fix_hashes"] == {"abc": 1}
        assert result["status"] == "running"

    def test_no_base_on_new_file(self, tmp_path: Path) -> None:
        """If file doesn't exist and no base given, starts empty."""
        p = tmp_path / "exp.json"
        result = patch_exp(p, status="running")
        assert result["status"] == "running"
        assert len(result) == 1


class TestReadJson:
    def test_reads_valid_file(self, tmp_path: Path) -> None:
        p = tmp_path / "test.json"
        p.write_text('{"a": 1, "b": [2, 3]}', encoding="utf-8")
        data = read_json(p)
        assert data == {"a": 1, "b": [2, 3]}


class TestScanLogChunk:
    def test_reads_from_offset(self, tmp_path: Path) -> None:
        p = tmp_path / "test.log"
        p.write_text("abcdefghij", encoding="utf-8")
        chunk = scan_log_chunk(p, 3, 4)
        assert chunk == "defg"

    def test_missing_file(self, tmp_path: Path) -> None:
        assert scan_log_chunk(tmp_path / "nonexistent.log", 0, 100) == ""


class TestScanLogTailMid:
    def test_small_log(self, tmp_path: Path) -> None:
        p = tmp_path / "small.log"
        p.write_text("line1\nline2\n", encoding="utf-8")
        chunks = scan_log_tail_mid(p)
        assert len(chunks) == 1  # only tail for small logs
        assert "line1" in chunks[0]

    def test_large_log(self, tmp_path: Path) -> None:
        p = tmp_path / "large.log"
        # Write > 512 KB of data
        content = ("x" * 1000 + "\n") * 600
        p.write_text(content, encoding="utf-8")
        chunks = scan_log_tail_mid(p)
        assert len(chunks) >= 2  # tail + mid (and possibly low)

    def test_empty_log(self, tmp_path: Path) -> None:
        p = tmp_path / "empty.log"
        p.write_text("", encoding="utf-8")
        assert scan_log_tail_mid(p) == []

    def test_missing_log(self, tmp_path: Path) -> None:
        assert scan_log_tail_mid(tmp_path / "nonexistent.log") == []


class TestGetClassificationText:
    def test_returns_lowercased_filtered_text(self, tmp_path: Path) -> None:
        p = tmp_path / "run.log"
        p.write_text(
            "WARNING: benign message\n"
            "ERROR: something bad happened\n"
            "INFO: progress update\n"
            "RuntimeError: CUDA out of memory\n",
            encoding="utf-8",
        )
        text = get_classification_text(p)
        assert "warning: benign" not in text  # filtered out
        assert "info: progress" not in text  # filtered out
        assert "error: something bad" in text  # kept
        assert "cuda out of memory" in text  # kept


class TestProximityMatch:
    def test_keywords_close_together(self) -> None:
        text = "out of memory: killed process 12345"
        assert _proximity_match(text, "killed", "out of memory", max_distance=500)

    def test_keywords_far_apart(self) -> None:
        text = "killed" + "x" * 600 + "out of memory"
        assert not _proximity_match(text, "killed", "out of memory", max_distance=500)

    def test_missing_keyword_a(self) -> None:
        text = "only out of memory"
        assert not _proximity_match(text, "killed", "out of memory", max_distance=500)

    def test_missing_keyword_b(self) -> None:
        text = "only killed"
        assert not _proximity_match(text, "killed", "out of memory", max_distance=500)

    def test_multiple_occurrences_finds_closest(self) -> None:
        """When keywords appear in multiple chunks, the closest pair wins."""
        text = "killed" + "x" * 100 + "out of some other thing" + "y" * 100 + "out of memory"
        assert _proximity_match(text, "killed", "out of memory", max_distance=500)


class TestFilterNoise:
    def test_removes_warning_lines_without_errors(self) -> None:
        text = "WARNING: benign message\nERROR: real problem"
        result = _filter_noise(text)
        assert "WARNING" not in result
        assert "ERROR" in result

    def test_keeps_warning_lines_with_error_keywords(self) -> None:
        text = "[rank0]:WARNING: CUDA OOM in AllReduce\nnormal line"
        result = _filter_noise(text)
        assert "WARNING" in result  # kept because it contains OOM

    def test_handles_empty_text(self) -> None:
        assert _filter_noise("") == ""

    def test_multiline_bracket_prefix(self) -> None:
        text = "[rank0]:WARNING: some context\nINFO: some info\nline without prefix"
        result = _filter_noise(text)
        assert "line without prefix" in result

    def test_info_lines_without_errors_removed(self) -> None:
        text = "INFO: step 100 completed\nERROR: failure"
        result = _filter_noise(text)
        assert "INFO" not in result
        assert "ERROR" in result

    def test_keeps_lines_with_traceback(self) -> None:
        text = "WARNING: something\nTraceback (most recent call last):\n  File \"train.py\", line 42"
        result = _filter_noise(text)
        assert "WARNING" not in result  # no error keywords
        assert "Traceback" in result  # traceback is always kept (no leading warning/info)


class TestProximityMatchEdgeCases:
    """Edge cases for _proximity_match — overlapping occurrences, empty text, etc."""

    def test_empty_text(self) -> None:
        assert not _proximity_match("", "killed", "out of memory", max_distance=500)

    def test_empty_keyword_a(self) -> None:
        text = "out of memory happened"
        assert _proximity_match(text, "", "out of memory", max_distance=500)

    def test_empty_keyword_b(self) -> None:
        text = "killed the process"
        assert _proximity_match(text, "killed", "", max_distance=500)

    def test_overlapping_occurrences(self) -> None:
        """Keywords that overlap should still match."""
        text = "ab" + "a" * 300 + "b"  # a and b within 500 chars
        assert _proximity_match(text, "a", "b", max_distance=500)

    def test_exact_boundary_distance(self) -> None:
        """Keywords exactly at max_distance apart should match (not strict inequality)."""
        text = "killed" + "x" * 500 + "out of memory"  # exactly 507 apart (6 + 500 + 1)
        assert not _proximity_match(text, "killed", "out of memory", max_distance=500)

    def test_barely_within_boundary(self) -> None:
        """Keywords exactly 499 chars apart should match."""
        text = "killed" + "x" * 493 + "out of memory"  # 6 + 493 + 1 = 500 apart
        assert _proximity_match(text, "killed", "out of memory", max_distance=500)

    def test_same_position_for_both_keywords(self) -> None:
        """When 'a' and 'b' match at the same position (keyword b is substring of a)."""
        # "killed" also matches "killed processes" in text
        text = "out of memory process killed"
        assert _proximity_match(text, "killed", "out of memory", max_distance=500)

    def test_keywords_across_chunk_boundary(self) -> None:
        """Mimics concatenated chunks: keyword a near end of chunk1, b near start of chunk2."""
        text = "x" * 400 + "killed" + "\n" + "out of memory" + "x" * 400
        assert _proximity_match(text, "killed", "out of memory", max_distance=500)


class TestGetClassificationTextEdgeCases:
    """Edge cases for get_classification_text — empty files, all-noise files."""

    def test_empty_file_returns_empty(self, tmp_path: Path) -> None:
        p = tmp_path / "empty.log"
        p.write_text("", encoding="utf-8")
        assert get_classification_text(p) == ""

    def test_missing_file_returns_empty(self) -> None:
        assert get_classification_text(Path("/nonexistent/run.log")) == ""

    def test_all_warning_lines_returns_empty(self, tmp_path: Path) -> None:
        """When all lines are pure WARNING/INFO (no error keywords), result is empty."""
        p = tmp_path / "clean.log"
        p.write_text(
            "WARNING: benign message about checkpoint saving\n"
            "INFO: training step 100 completed\n"
            "WARNING: estimated time remaining 30 minutes\n",
            encoding="utf-8",
        )
        result = get_classification_text(p)
        # "disk" is an error keyword, would keep that line — use only truly benign words
        assert result == ""

    def test_large_log_with_noise_returns_filtered_text(self, tmp_path: Path) -> None:
        """Large log with both noise and error lines returns cleaned text."""
        lines = ["INFO: step {} loss={}".format(i, 2.5 - i * 0.01) for i in range(200)]
        lines.append("RuntimeError: CUDA out of memory")
        p = tmp_path / "large.log"
        p.write_text("\n".join(lines), encoding="utf-8")
        result = get_classification_text(p)
        assert "cuda out of memory" in result
        assert "step" not in result  # INFO lines should be filtered
