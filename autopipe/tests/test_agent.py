from __future__ import annotations

import os
from pathlib import Path
from unittest import mock

import pytest

from autopipe.agent import (
    AGENT_NAME,
    TASK_PROMPT,
    _build_agent_spec,
    _build_system_prompt,
    _resolve_agent,
    run_agent,
)


class TestResolveAgent:
    """Unit tests for _resolve_agent covering all branches + edge cases."""

    def test_claude_explicit(self) -> None:
        with mock.patch("autopipe.agent.shutil.which", return_value="/usr/bin/claude"):
            assert _resolve_agent("claude") == "claude"

    def test_codex_explicit(self) -> None:
        with mock.patch("autopipe.agent.shutil.which", return_value="/usr/bin/codex"):
            with mock.patch("autopipe.agent.os.access", return_value=True):
                assert _resolve_agent("codex") == "codex"

    def test_auto_prefers_claude(self) -> None:
        with mock.patch("autopipe.agent.shutil.which", side_effect=lambda x: f"/usr/bin/{x}"):
            with mock.patch("autopipe.agent.os.access", return_value=True):
                assert _resolve_agent("auto") == "claude"

    def test_auto_falls_back_to_codex(self) -> None:
        def _which(name: str) -> str | None:
            return "/usr/bin/codex" if name == "codex" else None

        with mock.patch("autopipe.agent.shutil.which", side_effect=_which):
            with mock.patch("autopipe.agent.os.access", return_value=True):
                assert _resolve_agent("auto") == "codex"

    def test_auto_raises_when_neither_found(self) -> None:
        with mock.patch("autopipe.agent.shutil.which", return_value=None):
            with pytest.raises(RuntimeError, match="No agent CLI found"):
                _resolve_agent("auto")

    def test_claude_not_found_raises(self) -> None:
        with mock.patch("autopipe.agent.shutil.which", return_value=None):
            with pytest.raises(RuntimeError, match="not found on PATH"):
                _resolve_agent("claude")

    def test_claude_broken_binary_raises(self) -> None:
        """Binary found by shutil.which but not executable raises RuntimeError."""
        with mock.patch("autopipe.agent.shutil.which", return_value="/usr/bin/claude"):
            with mock.patch("autopipe.agent.os.access", return_value=False):
                with pytest.raises(RuntimeError, match="not executable"):
                    _resolve_agent("claude")

    def test_codex_not_found_raises(self) -> None:
        with mock.patch("autopipe.agent.shutil.which", return_value=None):
            with pytest.raises(RuntimeError, match="not found on PATH"):
                _resolve_agent("codex")

    def test_codex_broken_binary_raises(self) -> None:
        """Broken codex binary raises RuntimeError."""
        with mock.patch("autopipe.agent.shutil.which", return_value="/usr/bin/codex"):
            with mock.patch("autopipe.agent.os.access", return_value=False):
                with pytest.raises(RuntimeError, match="not executable"):
                    _resolve_agent("codex")


class TestBuildSystemPrompt:
    """Tests for _build_system_prompt content."""

    def test_contains_repo_root(self) -> None:
        prompt = _build_system_prompt("/home/user/distillm", "llm_train")
        assert "/home/user/distillm" in prompt
        assert "llm_train" in prompt

    def test_contains_task_instructions(self) -> None:
        prompt = _build_system_prompt("/repo", "env")
        assert "TRAIN_" in prompt
        assert "DeepSpeed" in prompt
        assert "train_opts" in prompt


class TestBuildAgentSpec:
    """Tests for _build_agent_spec."""

    def test_is_valid_json(self) -> None:
        import json
        spec = _build_agent_spec("/repo", "env")
        parsed = json.loads(spec)
        assert AGENT_NAME in parsed
        assert "description" in parsed[AGENT_NAME]
        assert "prompt" in parsed[AGENT_NAME]

    def test_contains_project_context(self) -> None:
        spec = _build_agent_spec("/custom/path", "custom_env")
        assert "/custom/path" in spec
        assert "custom_env" in spec


class TestRunAgent:
    """Tests for run_agent edge cases (mocking subprocess)."""

    def test_returns_exit_code(self, tmp_path: Path) -> None:
        """Successful agent run returns 0."""
        with mock.patch("autopipe.agent._resolve_agent", return_value="claude"):
            with mock.patch("autopipe.agent.subprocess.run") as mock_run:
                mock_run.return_value.returncode = 0
                rc = run_agent(tmp_path, tmp_path, timeout_seconds=60)
                assert rc == 0

    def test_returns_non_zero_exit_code(self, tmp_path: Path) -> None:
        """Agent that fails returns its exit code."""
        with mock.patch("autopipe.agent._resolve_agent", return_value="claude"):
            with mock.patch("autopipe.agent.subprocess.run") as mock_run:
                mock_run.return_value.returncode = 1
                rc = run_agent(tmp_path, tmp_path, timeout_seconds=60)
                assert rc == 1

    def test_timeout_returns_124(self, tmp_path: Path) -> None:
        """Subprocess timeout returns exit code 124."""
        with mock.patch("autopipe.agent._resolve_agent", return_value="claude"):
            with mock.patch("autopipe.agent.subprocess.run") as mock_run:
                from subprocess import TimeoutExpired
                mock_run.side_effect = TimeoutExpired("cmd", 60)
                rc = run_agent(tmp_path, tmp_path, timeout_seconds=60)
                assert rc == 124

    def test_io_error_returns_1(self, tmp_path: Path) -> None:
        """OSError during agent run returns exit code 1."""
        with mock.patch("autopipe.agent._resolve_agent", return_value="claude"):
            with mock.patch("autopipe.agent.subprocess.run") as mock_run:
                mock_run.side_effect = OSError("Broken pipe")
                rc = run_agent(tmp_path, tmp_path, timeout_seconds=60)
                assert rc == 1

    def test_writes_agent_log(self, tmp_path: Path) -> None:
        """Agent run writes start/end markers to agent.log."""
        with mock.patch("autopipe.agent._resolve_agent", return_value="claude"):
            with mock.patch("autopipe.agent.subprocess.run") as mock_run:
                mock_run.return_value.returncode = 0
                rc = run_agent(tmp_path, tmp_path, timeout_seconds=60)
                assert rc == 0
                agent_log = tmp_path / "agent.log"
                assert agent_log.exists()
                content = agent_log.read_text()
                assert "AGENT_START" in content

    def test_captures_io_error_in_log(self, tmp_path: Path) -> None:
        """IO error writes AGENT_IO_ERROR marker to log."""
        with mock.patch("autopipe.agent._resolve_agent", return_value="claude"):
            with mock.patch("autopipe.agent.subprocess.run") as mock_run:
                mock_run.side_effect = ValueError("embedded null byte")
                rc = run_agent(tmp_path, tmp_path, timeout_seconds=60)
                assert rc == 1
                agent_log = tmp_path / "agent.log"
                assert agent_log.exists()
                content = agent_log.read_text()
                assert "AGENT_IO_ERROR" in content

    def test_captures_timeout_in_log(self, tmp_path: Path) -> None:
        """Timeout writes AGENT_TIMEOUT marker to log."""
        with mock.patch("autopipe.agent._resolve_agent", return_value="claude"):
            with mock.patch("autopipe.agent.subprocess.run") as mock_run:
                from subprocess import TimeoutExpired
                mock_run.side_effect = TimeoutExpired("cmd", 60)
                rc = run_agent(tmp_path, tmp_path, timeout_seconds=60)
                assert rc == 124
                agent_log = tmp_path / "agent.log"
                assert agent_log.exists()
                content = agent_log.read_text()
                assert "AGENT_TIMEOUT" in content

    def test_codex_path_constructs_correctly(self, tmp_path: Path) -> None:
        """Codex path constructs command with sandbox and prompt."""
        with mock.patch("autopipe.agent._resolve_agent", return_value="codex"):
            with mock.patch("autopipe.agent.subprocess.run") as mock_run:
                mock_run.return_value.returncode = 0
                rc = run_agent(tmp_path, tmp_path, timeout_seconds=60)
                assert rc == 0
                # Verify the command structure
                cmd = mock_run.call_args[0][0]
                assert cmd[0] == "codex"
                assert cmd[1] == "exec"
                assert "--sandbox" in str(cmd[2])
                assert "--dangerously-bypass-approvals-and-sandbox" in cmd

    def test_claude_path_constructs_correctly(self, tmp_path: Path) -> None:
        """Claude path constructs command with --allowedTools and stdin."""
        with mock.patch("autopipe.agent._resolve_agent", return_value="claude"):
            with mock.patch("autopipe.agent.subprocess.run") as mock_run:
                mock_run.return_value.returncode = 0
                rc = run_agent(tmp_path, tmp_path, timeout_seconds=60)
                assert rc == 0
                cmd = mock_run.call_args[0][0]
                assert cmd[0] == "claude"
                assert "--allowedTools" in cmd
                assert "Glob" in str(cmd)
                assert "Grep" in str(cmd)
                assert "-" in cmd  # stdin flag

    def test_claude_path_passes_task_prompt_as_input(self, tmp_path: Path) -> None:
        """Claude path passes TASK_PROMPT as input to subprocess."""
        with mock.patch("autopipe.agent._resolve_agent", return_value="claude"):
            with mock.patch("autopipe.agent.subprocess.run") as mock_run:
                mock_run.return_value.returncode = 0
                run_agent(tmp_path, tmp_path, timeout_seconds=60)
                kwargs = mock_run.call_args[1]
                assert kwargs.get("input") == TASK_PROMPT
                assert kwargs.get("text") is True

    def test_allowed_tools_contains_glob_and_grep(self) -> None:
        """Verify --allowedTools includes Glob and Grep tools."""
        with mock.patch("autopipe.agent._resolve_agent", return_value="claude"):
            with mock.patch("autopipe.agent.subprocess.run") as mock_run:
                mock_run.return_value.returncode = 0
                run_agent(Path("/tmp"), Path("/tmp"), timeout_seconds=60)
                cmd = mock_run.call_args[0][0]
                tools_idx = cmd.index("--allowedTools") + 1
                tools_str = cmd[tools_idx]
                # Glob and Grep should be listed as tool names (not inside Bash parens)
                assert "Glob," in tools_str or tools_str.startswith("Glob")
                assert "Grep," in tools_str
                assert "Bash(" in tools_str
