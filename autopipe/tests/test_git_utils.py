from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path
from unittest import mock

from autopipe.git_utils import git_diff, git_status, snapshot_git


class TestGitDiff:
    """Tests for git_diff — error handling and truncation."""

    def test_not_a_git_repo(self, tmp_path: Path) -> None:
        """Non-git directory returns error message, not an exception."""
        result = git_diff(tmp_path)
        assert isinstance(result, str)
        assert "git_diff_error" in result

    def test_git_not_on_path(self, tmp_path: Path) -> None:
        """When git binary is not found, return error message."""
        with mock.patch("autopipe.git_utils.subprocess.Popen") as mock_popen:
            mock_popen.side_effect = FileNotFoundError("git not found")
            result = git_diff(tmp_path)
            assert "git not found" in result

    def test_timeout_returns_error(self, tmp_path: Path) -> None:
        """Git diff that times out returns error message."""
        with mock.patch("autopipe.git_utils.subprocess.Popen") as mock_popen:
            mock_proc = mock.MagicMock()
            mock_proc.communicate.side_effect = subprocess.TimeoutExpired("cmd", 30)
            mock_popen.return_value = mock_proc
            result = git_diff(tmp_path)
            assert "timed out" in result

    def test_non_zero_exit(self, tmp_path: Path) -> None:
        """Git diff with non-zero exit returns error message."""
        with mock.patch("autopipe.git_utils.subprocess.Popen") as mock_popen:
            mock_proc = mock.MagicMock()
            mock_proc.returncode = 128
            mock_proc.communicate.return_value = (b"", b"fatal: not a git repo")
            mock_popen.return_value = mock_proc
            result = git_diff(tmp_path)
            assert "exit code=128" in result

    def test_byte_level_truncation(self, tmp_path: Path) -> None:
        """Large binary diff is truncated at byte level before decode."""
        with mock.patch("autopipe.git_utils.subprocess.Popen") as mock_popen:
            mock_proc = mock.MagicMock()
            mock_proc.returncode = 0
            # Create a 3MB output with null bytes (binary)
            mock_proc.communicate.return_value = (b"\x00" * (3 * 1024 * 1024), b"")
            mock_popen.return_value = mock_proc
            result = git_diff(tmp_path, max_bytes=1024)
            # Should not OOM — result should be truncated
            assert len(result) <= 1024 + 50  # allow for replacement chars + truncation note

    def test_empty_diff(self, tmp_path: Path) -> None:
        """Empty diff (no changes) returns empty string-ish."""
        with mock.patch("autopipe.git_utils.subprocess.Popen") as mock_popen:
            mock_proc = mock.MagicMock()
            mock_proc.returncode = 0
            mock_proc.communicate.return_value = (b"", b"")
            mock_popen.return_value = mock_proc
            result = git_diff(tmp_path)
            assert result == ""


class TestGitStatus:
    """Tests for git_status — error handling."""

    def test_not_a_git_repo(self, tmp_path: Path) -> None:
        """Non-git directory returns error string."""
        result = git_status(tmp_path)
        assert isinstance(result, str)
        assert "git_status_error" in result

    def test_empty_status(self, tmp_path: Path) -> None:
        """Clean repo returns empty string."""
        with mock.patch("autopipe.git_utils.subprocess.check_output") as mock_co:
            mock_co.return_value = b""
            result = git_status(tmp_path)
            assert result == ""


class TestSnapshotGit:
    """Tests for snapshot_git — error handling and file creation."""

    def test_creates_files_in_exp_dir(self, tmp_path: Path) -> None:
        """snapshot_git creates status and diff files in exp_dir."""
        exp_dir = tmp_path / "exp"
        with mock.patch("autopipe.git_utils.git_status", return_value="M file.py\n"):
            with mock.patch("autopipe.git_utils.git_diff", return_value="diff --git a/file.py b/file.py\n"):
                snapshot_git(tmp_path, exp_dir, "pre_agent")
                files = list(exp_dir.iterdir())
                assert len(files) >= 2
                status_files = [f for f in files if "git_status" in f.name]
                diff_files = [f for f in files if "git_diff" in f.name]
                assert len(status_files) == 1
                assert len(diff_files) == 1
                assert "pre_agent" in status_files[0].name
                assert "pre_agent" in diff_files[0].name

    def test_not_a_git_repo_writes_notice(self, tmp_path: Path) -> None:
        """When repo_root is not a git repo, snapshot writes informative notice."""
        exp_dir = tmp_path / "exp"
        snapshot_git(tmp_path, exp_dir, "pre_agent")
        status_files = list(exp_dir.glob("git_status_*"))
        diff_files = list(exp_dir.glob("git_diff_*"))
        assert len(status_files) == 1
        assert len(diff_files) == 1
        status_text = status_files[0].read_text()
        assert "not a git repository" in status_text or "git_status_error" in status_text
        diff_text = diff_files[0].read_text()
        assert "not a git repository" in diff_text or "git_diff_error" in diff_text

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        """snapshot_git creates intermediate directories in exp_dir."""
        deep_dir = tmp_path / "a" / "b" / "c" / "exp"
        snapshot_git(deep_dir, deep_dir, "test")  # repo_root doesn't exist either
        # Should not crash; deep_dir was created with a status file
        assert deep_dir.is_dir()
        assert len(list(deep_dir.glob("git_status_*"))) == 1

    def test_tag_in_filenames(self, tmp_path: Path) -> None:
        """Tag is embedded in the generated filenames."""
        exp_dir = tmp_path / "exp"
        with mock.patch("autopipe.git_utils.git_status", return_value=""):
            with mock.patch("autopipe.git_utils.git_diff", return_value=""):
                snapshot_git(tmp_path, exp_dir, "my_custom_tag")
                diff_files = list(exp_dir.glob("git_diff_*"))
                assert len(diff_files) == 1
                assert "my_custom_tag" in diff_files[0].name
