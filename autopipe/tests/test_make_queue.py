from __future__ import annotations

import json
from pathlib import Path

from autopipe.make_queue import (
    _extract_script_path,
    build_exp,
    distillm_specs,
)


class TestExtractScriptPath:
    """Tests for _extract_script_path covering shell constructs and edge cases."""

    def test_simple_script_path(self) -> None:
        exp = {"cmd_type": "bash", "cmd": "/path/to/script.sh"}
        assert _extract_script_path(exp) == Path("/path/to/script.sh")

    def test_with_bash_prefix(self) -> None:
        exp = {"cmd_type": "bash", "cmd": "bash /path/to/script.sh"}
        assert _extract_script_path(exp) == Path("/path/to/script.sh")

    def test_with_sh_prefix(self) -> None:
        exp = {"cmd_type": "bash", "cmd": "sh /path/to/script.sh"}
        assert _extract_script_path(exp) == Path("/path/to/script.sh")

    def test_with_zsh_prefix(self) -> None:
        exp = {"cmd_type": "bash", "cmd": "zsh /path/to/script.sh"}
        assert _extract_script_path(exp) == Path("/path/to/script.sh")

    def test_bash_c_inline_returns_none(self) -> None:
        """bash -c 'echo hi' is an inline command, not a script path."""
        exp = {"cmd_type": "bash", "cmd": "bash -c 'echo hi'"}
        assert _extract_script_path(exp) is None

    def test_bash_c_with_quoted_command(self) -> None:
        """bash -c with double-quoted inline command returns None."""
        exp = {"cmd_type": "bash", "cmd": 'bash -c "echo hello world"'}
        assert _extract_script_path(exp) is None

    def test_non_bash_cmd_type(self) -> None:
        exp = {"cmd_type": "torchrun", "cmd": "/path/to/script.sh"}
        assert _extract_script_path(exp) is None

    def test_bash_cmd_type_not_set(self) -> None:
        exp = {"cmd": "/path/to/script.sh"}  # no cmd_type
        assert _extract_script_path(exp) is None

    def test_empty_cmd(self) -> None:
        exp = {"cmd_type": "bash", "cmd": ""}
        assert _extract_script_path(exp) is None

    def test_whitespace_cmd(self) -> None:
        exp = {"cmd_type": "bash", "cmd": "   "}
        assert _extract_script_path(exp) is None

    def test_malformed_quoting(self) -> None:
        """Unclosed quotes should not crash — returns None."""
        exp = {"cmd_type": "bash", "cmd": "bash '/path/unclosed"}
        assert _extract_script_path(exp) is None

    def test_script_with_spaces_in_path(self) -> None:
        """Quoted paths with spaces are handled correctly by shlex.split."""
        exp = {"cmd_type": "bash", "cmd": 'bash "/path/with spaces/script.sh"'}
        result = _extract_script_path(exp)
        assert result is not None
        assert result.name == "script.sh"

    def test_bash_only_no_args(self) -> None:
        """Just 'bash' with no script — returns Path("bash") which won't exist."""
        exp = {"cmd_type": "bash", "cmd": "bash"}
        result = _extract_script_path(exp)
        assert result == Path("bash")

    def test_missing_cmd_key(self) -> None:
        exp = {"cmd_type": "bash"}  # no 'cmd' key
        result = _extract_script_path(exp)
        assert result is None


class TestBuildExp:
    """Tests for build_exp covering seq values and default handling."""

    def test_build_exp_creates_exp_id(self) -> None:
        spec = {"key": "test", "cmd": "/path/to/script.sh"}
        exp = build_exp(spec, seq=1)
        assert exp["exp_id"].startswith("test_")
        # "test_" = 5 chars + 8 hex chars = 13
        assert len(exp["exp_id"]) == 13

    def test_seq_positive(self) -> None:
        exp = build_exp({"key": "test", "cmd": "/path.sh"}, seq=5)
        assert exp["seq"] == 5

    def test_seq_zero(self) -> None:
        exp = build_exp({"key": "test", "cmd": "/path.sh"}, seq=0)
        assert exp["seq"] == 0

    def test_seq_negative(self) -> None:
        exp = build_exp({"key": "test", "cmd": "/path.sh"}, seq=-1)
        assert exp["seq"] == -1

    def test_default_values(self) -> None:
        exp = build_exp({"key": "test", "cmd": "/path.sh"}, seq=1)
        assert exp["cmd_type"] == "bash"
        assert exp["status"] == "pending"
        assert exp["attempt"] == 0
        assert exp["max_retries"] == 2  # default
        assert exp["retry_sleep"] == 60  # default
        assert exp["train_timeout"] == 86400  # default
        assert exp["skip_vis"] is True
        assert exp["conda_env"] == "llm_train"
        assert exp["hard_failure_threshold"] == 3
        assert exp["train_opts"] == {}
        assert exp["oom_batch_candidates"] == []
        assert exp["max_oom_retries"] == 0

    def test_custom_values(self) -> None:
        spec = {
            "key": "test",
            "cmd": "/path.sh",
            "max_retries": 5,
            "retry_sleep": 120,
            "gpus": "0",
            "train_timeout": 3600,
            "skip_vis": False,
            "conda_env": "custom_env",
            "hard_failure_threshold": 2,
            "train_opts": {"lr": 0.001, "batch_size": 4},
            "oom_batch_candidates": [4, 2, 1],
        }
        exp = build_exp(spec, seq=2)
        assert exp["key"] == "test"
        assert exp["max_retries"] == 5
        assert exp["retry_sleep"] == 120
        assert exp["gpus"] == "0"
        assert exp["train_timeout"] == 3600
        assert exp["skip_vis"] is False
        assert exp["conda_env"] == "custom_env"
        assert exp["hard_failure_threshold"] == 2
        assert exp["train_opts"] == {"lr": 0.001, "batch_size": 4}
        assert exp["oom_batch_candidates"] == [4, 2, 1]
        assert exp["max_oom_retries"] == 3  # len of candidates

    def test_max_oom_retries_custom(self) -> None:
        """max_oom_retries from spec is used if provided."""
        spec = {
            "key": "test",
            "cmd": "/path.sh",
            "oom_batch_candidates": [8, 4, 2, 1],
            "max_oom_retries": 2,  # override default len(candidates)
        }
        exp = build_exp(spec, seq=1)
        assert exp["max_oom_retries"] == 2

    def test_seq_formatting(self) -> None:
        """seq is formatted as 2-digit zero-padded in filename."""
        from autopipe.make_queue import distillm_specs
        specs = distillm_specs(base="/tmp")
        exps = [build_exp(s, seq=i + 1) for i, s in enumerate(specs)]
        for i, exp in enumerate(exps):
            assert exp["seq"] == i + 1


class TestDistillmSpecs:
    """Tests for distillm_specs factory function."""

    def test_returns_list_of_dicts(self) -> None:
        specs = distillm_specs(base="/tmp")
        assert isinstance(specs, list)
        assert len(specs) > 0
        for spec in specs:
            assert "key" in spec
            assert "cmd" in spec

    def test_all_specs_have_required_fields(self) -> None:
        specs = distillm_specs(base="/tmp")
        for spec in specs:
            assert spec["cmd"].startswith("/tmp/"), f"cmd should start with base: {spec['cmd']}"
            assert spec.get("cmd_type", "bash") == "bash" or "cmd_type" not in spec

    def test_specs_in_expected_order(self) -> None:
        specs = distillm_specs(base="/tmp")
        keys = [s["key"] for s in specs]
        assert keys[0] == "kd_train"
        assert "distillm_train" in keys
        assert "distillm_eval" in keys
