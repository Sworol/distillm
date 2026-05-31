from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest import mock

import pytest

from autopipe.worker import AttemptContext, _handle_outcome


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
