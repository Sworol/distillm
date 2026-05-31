from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from unittest import mock

import pytest

from autopipe.worker import (
    AttemptContext,
    FailureContext,
    RecoveryAction,
    RecoveryManager,
    _handle_outcome,
    _last_error_hash,
    apply_oom_batch_backoff,
    ensure_exp_sane,
    pick_free_tcp_port,
    resolve_master_port,
)


def _make_log_with_error(tmp_path: Path, error_text: str) -> Path:
    """Write a synthetic run.log with a recognizable error."""
    p = tmp_path / "run.log"
    p.write_text(error_text, encoding="utf-8")
    return p


class TestHandleOutcome:
    """Integration tests for _handle_outcome covering all exit paths.

    Each test patches ``sys.exit`` to prevent the test process from actually
    terminating and to capture the exit code.
    """

    def test_success_path(self, tmp_path: Path) -> None:
        """rc==0 writes 'success' status and exits 0."""
        run_log = tmp_path / "run.log"
        run_log.write_text("training complete\n")
        status_path = tmp_path / "status.json"
        run_exp_path = tmp_path / "exp.json"

        exp = {"exp_id": "test", "train_opts": {}, "master_port": 29500}
        run_exp_path.write_text(json.dumps(exp))

        ctx = AttemptContext(
            rc=0, run_log=run_log, exp=exp, run_root=tmp_path,
            run_exp_path=run_exp_path, status_path=status_path,
            attempt=1, train_cmd=["bash", "test.sh"], env={},
            cmd_type="bash", nproc=4, repo_root=tmp_path, agent_timeout=600,
        )

        with pytest.raises(SystemExit) as exc_info:
            _handle_outcome(ctx)
        assert exc_info.value.code == 0

        st = json.loads(status_path.read_text())
        assert st["status"] == "success"
        assert st["attempt"] == 1

        written_exp = json.loads(run_exp_path.read_text())
        assert written_exp["status"] == "success"
        assert written_exp["consecutive_failures"] == 0

    def test_interrupted_path(self, tmp_path: Path) -> None:
        """rc==130 with no retries writes 'failed' with reason='interrupted'."""
        run_log = tmp_path / "run.log"
        run_log.write_text("interrupted\n")
        status_path = tmp_path / "status.json"
        run_exp_path = tmp_path / "exp.json"

        exp = {"exp_id": "test", "max_retries": 0, "train_opts": {}, "master_port": 29500}
        run_exp_path.write_text(json.dumps(exp))

        ctx = AttemptContext(
            rc=130, run_log=run_log, exp=exp, run_root=tmp_path,
            run_exp_path=run_exp_path, status_path=status_path,
            attempt=1, train_cmd=["bash", "test.sh"], env={},
            cmd_type="bash", nproc=4, repo_root=tmp_path, agent_timeout=600,
        )

        with pytest.raises(SystemExit) as exc_info:
            _handle_outcome(ctx)
        assert exc_info.value.code == 130

        st = json.loads(status_path.read_text())
        assert st["status"] == "failed"
        assert st["reason"] == "interrupted"

    def test_timeout_path(self, tmp_path: Path) -> None:
        """rc==124 with no retries writes 'failed' with reason='timeout'."""
        run_log = tmp_path / "run.log"
        run_log.write_text("timeout\n")
        status_path = tmp_path / "status.json"
        run_exp_path = tmp_path / "exp.json"

        exp = {"exp_id": "test", "max_retries": 0, "train_opts": {}, "master_port": 29500}
        run_exp_path.write_text(json.dumps(exp))

        ctx = AttemptContext(
            rc=124, run_log=run_log, exp=exp, run_root=tmp_path,
            run_exp_path=run_exp_path, status_path=status_path,
            attempt=1, train_cmd=["bash", "test.sh"], env={},
            cmd_type="bash", nproc=4, repo_root=tmp_path, agent_timeout=600,
        )

        with pytest.raises(SystemExit) as exc_info:
            _handle_outcome(ctx)
        assert exc_info.value.code == 124

        st = json.loads(status_path.read_text())
        assert st["status"] == "failed"
        assert st["reason"] == "timeout"

    def test_status_json_contains_full_metadata_on_failure(self, tmp_path: Path) -> None:
        """Failed status.json includes train_cmd, env vars, nproc, master_port."""
        run_log = _make_log_with_error(tmp_path,
            "RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB")
        status_path = tmp_path / "status.json"
        run_exp_path = tmp_path / "exp.json"

        exp = {"exp_id": "test", "max_retries": 0, "train_opts": {},
               "master_port": 12345, "oom_batch_candidates": [8, 4, 2, 1]}
        run_exp_path.write_text(json.dumps(exp))

        ctx = AttemptContext(
            rc=1, run_log=run_log, exp=exp, run_root=tmp_path,
            run_exp_path=run_exp_path, status_path=status_path,
            attempt=2, train_cmd=["bash", "train.sh"], env={"CUDA_VISIBLE_DEVICES": "0,1"},
            cmd_type="bash", nproc=4, repo_root=tmp_path, agent_timeout=600,
        )

        with pytest.raises(SystemExit) as exc_info:
            _handle_outcome(ctx)
        assert exc_info.value.code == 1

        st = json.loads(status_path.read_text())
        assert st["status"] == "failed"
        assert st["train_cmd"] == ["bash", "train.sh"]
        assert st["cuda_visible_devices"] == "0,1"
        assert st["nproc"] == 4
        assert st["master_port"] == 12345
        assert "error_hash" in st

    def test_single_write_on_failure(self, tmp_path: Path) -> None:
        """status.json is written exactly once (not a preliminary 'failed'
        write followed by a second overwrite)."""
        run_log = _make_log_with_error(tmp_path,
            "RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB")
        status_path = tmp_path / "status.json"
        run_exp_path = tmp_path / "exp.json"

        exp = {"exp_id": "test", "max_retries": 0, "train_opts": {},
               "oom_batch_candidates": [8, 4, 2, 1]}
        run_exp_path.write_text(json.dumps(exp))

        ctx = AttemptContext(
            rc=1, run_log=run_log, exp=exp, run_root=tmp_path,
            run_exp_path=run_exp_path, status_path=status_path,
            attempt=1, train_cmd=["bash", "train.sh"], env={},
            cmd_type="bash", nproc=4, repo_root=tmp_path, agent_timeout=600,
        )

        # Wrap atomic_write_json to count writes to status_path.
        import autopipe.worker as worker_module
        original = worker_module.atomic_write_json
        status_writes = []

        def counting_write(path, data):
            if path == status_path:
                status_writes.append(dict(data))
            return original(path, data)

        with mock.patch.object(worker_module, "atomic_write_json", side_effect=counting_write):
            with pytest.raises(SystemExit):
                _handle_outcome(ctx)

        assert len(status_writes) == 1, (
            f"Expected 1 write to status.json, got {len(status_writes)}: {status_writes}"
        )
        assert status_writes[0]["status"] == "failed"
        assert "train_cmd" in status_writes[0]


class TestHandleOutcomeKilled:
    """Tests for _handle_outcome with negative exit codes (signal termination)."""

    def test_negative_exit_code_classifies_as_killed(self, tmp_path: Path) -> None:
        """rc < 0 (killed by signal) should be classified as 'killed'."""
        run_log = tmp_path / "run.log"
        run_log.write_text("SIGKILL\n", encoding="utf-8")
        status_path = tmp_path / "status.json"
        run_exp_path = tmp_path / "exp.json"

        exp = {"exp_id": "test", "max_retries": 0, "train_opts": {}, "master_port": 29500}
        run_exp_path.write_text(json.dumps(exp))

        ctx = AttemptContext(
            rc=-9, run_log=run_log, exp=exp, run_root=tmp_path,
            run_exp_path=run_exp_path, status_path=status_path,
            attempt=1, train_cmd=["bash", "test.sh"], env={},
            cmd_type="bash", nproc=4, repo_root=tmp_path, agent_timeout=600,
        )

        with pytest.raises(SystemExit) as exc_info:
            _handle_outcome(ctx)
        assert exc_info.value.code == -9

        st = json.loads(status_path.read_text())
        assert st["status"] == "failed"
        assert st["reason"] == "killed"


class TestLastErrorHash:
    """Tests for _last_error_hash edge cases."""

    def test_empty_log_returns_empty(self, tmp_path: Path) -> None:
        p = tmp_path / "run.log"
        p.write_text("", encoding="utf-8")
        assert _last_error_hash(p) == ""

    def test_missing_log_returns_empty(self) -> None:
        assert _last_error_hash(Path("/nonexistent/run.log")) == ""

    def test_only_torchrun_boilerplate_returns_empty(self, tmp_path: Path) -> None:
        """Torchrun boilerplate (error_file, ChildFailedError) should be filtered."""
        p = tmp_path / "run.log"
        p.write_text(
            "torchrun error_file: /tmp/torchrun_errors.txt\n"
            "ChildFailedError: rank 0 failed\n"
            "INFO: step 100 completed normally\n",
            encoding="utf-8",
        )
        assert _last_error_hash(p) == ""

    def test_actual_error_returns_hash(self, tmp_path: Path) -> None:
        p = tmp_path / "run.log"
        p.write_text(
            "Step 100: loss=2.5\n"
            "RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB\n"
            "torchrun ChildFailedError: root cause in rank 0\n",
            encoding="utf-8",
        )
        h = _last_error_hash(p)
        assert h != ""
        assert len(h) == 32  # md5 hex digest length

    def test_rank_prefix_is_stripped_for_stable_hash(self, tmp_path: Path) -> None:
        """[rank0] and [rank3] prefixes should produce the same hash."""
        p1 = tmp_path / "run1.log"
        p1.write_text(
            "[rank0]: RuntimeError: CUDA out of memory\n",
            encoding="utf-8",
        )
        p2 = tmp_path / "run2.log"
        p2.write_text(
            "[rank3]: RuntimeError: CUDA out of memory\n",
            encoding="utf-8",
        )

        h1 = _last_error_hash(p1)
        h2 = _last_error_hash(p2)
        assert h1 == h2
        assert h1 != ""

    def test_many_lines_only_last_20_used(self, tmp_path: Path) -> None:
        """Only the last 20 error lines are hashed (prevents unbounded memory)."""
        lines = []
        for i in range(50):
            lines.append(f"Error line {i}: something failed")
        p = tmp_path / "run.log"
        p.write_text("\n".join(lines), encoding="utf-8")
        h = _last_error_hash(p)
        assert h != ""
        # Verify the hash is stable (repeatable)
        assert _last_error_hash(p) == h


class TestEnsureExpSane:
    """Tests for ensure_exp_sane covering bash cmd edge cases."""

    def test_bash_cmd_type_with_script_path(self) -> None:
        """Simple script path with bash cmd_type passes validation."""
        exp = {
            "exp_id": "test",
            "cmd_type": "bash",
            "cmd": "/bin/echo",
        }
        ensure_exp_sane(exp)  # should not raise

    def test_bash_cmd_type_with_bash_prefix_and_script(self) -> None:
        """'bash /path/to/script.sh' format should work."""
        exp = {
            "exp_id": "test",
            "cmd_type": "bash",
            "cmd": "bash /bin/echo",
        }
        ensure_exp_sane(exp)  # should not raise

    def test_bash_cmd_type_with_inline_command(self) -> None:
        """'bash -c \"echo hi\"' inline command should not crash."""
        exp = {
            "exp_id": "test",
            "cmd_type": "bash",
            "cmd": "bash -c 'echo hi'",
        }
        ensure_exp_sane(exp)  # should not raise

    def test_bash_cmd_type_missing_cmd_key(self) -> None:
        """Missing 'cmd' key for bash type raises ValueError."""
        exp = {
            "exp_id": "test",
            "cmd_type": "bash",
        }
        with pytest.raises(ValueError, match="requires.*cmd"):
            ensure_exp_sane(exp)

    def test_bash_cmd_type_empty_cmd(self) -> None:
        """Empty 'cmd' string: shlex.split('') returns [] so no script to validate."""
        exp = {
            "exp_id": "test",
            "cmd_type": "bash",
            "cmd": "",
        }
        ensure_exp_sane(exp)  # empty cmd passes (no parts to validate)

    def test_non_bash_cmd_type_skips_cmd_validation(self) -> None:
        """torchrun cmd_type doesn't require 'cmd' key."""
        exp = {
            "exp_id": "test",
            "cmd_type": "torchrun",
            "trainer": "default",
            "cfg_path": "config.yaml",
        }
        ensure_exp_sane(exp)  # should not raise

    def test_missing_exp_id_raises(self) -> None:
        exp = {"cmd_type": "bash", "cmd": "/bin/echo"}
        with pytest.raises(ValueError, match="missing required keys"):
            ensure_exp_sane(exp)

    def test_negative_nproc_raises(self) -> None:
        exp = {
            "exp_id": "test",
            "cmd_type": "bash",
            "cmd": "/bin/echo",
            "nproc": -1,
        }
        with pytest.raises(ValueError, match="invalid nproc"):
            ensure_exp_sane(exp)

    def test_invalid_master_port_raises(self) -> None:
        exp = {
            "exp_id": "test",
            "cmd_type": "bash",
            "cmd": "/bin/echo",
            "master_port": "not_a_number",
        }
        with pytest.raises(ValueError, match="invalid master_port"):
            ensure_exp_sane(exp)

    def test_auto_master_port_is_valid(self) -> None:
        exp = {
            "exp_id": "test",
            "cmd_type": "bash",
            "cmd": "/bin/echo",
            "master_port": "auto",
        }
        ensure_exp_sane(exp)  # should not raise


class TestPickFreeTcpPort:
    def test_returns_positive_integer(self) -> None:
        port = pick_free_tcp_port()
        assert isinstance(port, int)
        assert port > 0
        assert port < 65536


class TestResolveMasterPort:
    def test_explicit_positive_port(self) -> None:
        assert resolve_master_port({"master_port": 29500}) == 29500

    def test_explicit_string_port(self) -> None:
        assert resolve_master_port({"master_port": "29500"}) == 29500

    def test_auto_string(self) -> None:
        port = resolve_master_port({"master_port": "auto"})
        assert isinstance(port, int)
        assert port > 0

    def test_none_or_missing(self) -> None:
        port = resolve_master_port({"master_port": None})
        assert isinstance(port, int)
        assert port > 0
        port2 = resolve_master_port({})
        assert isinstance(port2, int)
        assert port2 > 0

    def test_zero_returns_free_port(self) -> None:
        port = resolve_master_port({"master_port": 0})
        assert isinstance(port, int)
        assert port > 0


class TestApplyOomBatchBackoff:
    def test_reduces_batch_size_to_next_candidate(self) -> None:
        exp = {
            "train_opts": {"batch_size": 8},
            "oom_batch_candidates": [8, 4, 2, 1],
        }
        assert apply_oom_batch_backoff(exp) is True
        assert exp["train_opts"]["batch_size"] == 4

    def test_no_more_candidates_returns_false(self) -> None:
        exp = {
            "train_opts": {"batch_size": 1},
            "oom_batch_candidates": [8, 4, 2, 1],
        }
        assert apply_oom_batch_backoff(exp) is False
        assert exp["train_opts"]["batch_size"] == 1

    def test_no_candidates_returns_false(self) -> None:
        exp = {
            "train_opts": {"batch_size": 8},
        }
        assert apply_oom_batch_backoff(exp) is False

    def test_non_dict_train_opts_returns_false(self) -> None:
        exp = {
            "train_opts": "string_instead_of_dict",
            "oom_batch_candidates": [8, 4, 2, 1],
        }
        assert apply_oom_batch_backoff(exp) is False

    def test_multiple_batch_keys_first_matched_key_used(self) -> None:
        """batch_keys order is ['batch_size', 'per_device_train_batch_size', 'micro_batch_size'].
        'batch_size: 8' is found first; micro_batch_size is never reached."""
        exp = {
            "train_opts": {"micro_batch_size": 16, "batch_size": 8},
            "oom_batch_candidates": [16, 8, 4],
        }
        assert apply_oom_batch_backoff(exp) is True
        # batch_size (key order priority) got reduced from 8 -> 4
        assert exp["train_opts"]["batch_size"] == 4
        # micro_batch_size was never touched
        assert exp["train_opts"]["micro_batch_size"] == 16
