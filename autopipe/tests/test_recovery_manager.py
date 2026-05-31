from __future__ import annotations

import time
from pathlib import Path
from unittest import mock

from autopipe.worker import (
    FailureContext,
    RecoveryAction,
    RecoveryManager,
)


def make_ctx(**overrides) -> FailureContext:
    """Create a FailureContext with sensible defaults."""
    defaults: dict = dict(
        exp={
            "exp_id": "test_exp_001",
            "max_retries": 3,
            "train_opts": {"batch_size": 8, "lr": 0.0005},
            "oom_batch_candidates": [8, 4, 2, 1],
            "max_oom_retries": 3,
        },
        run_exp_path=Path("/tmp/test_exp/exp.json"),
        status_path=Path("/tmp/test_exp/status.json"),
        attempt=1,
        rc=1,
        reason="oom",
        error_hash="abc123",
    )
    defaults.update(overrides)
    return FailureContext(**defaults)


class TestRecoveryManagerOOM:
    """Tests for OOM batch-size backoff path."""

    def test_oom_backoff_reduces_batch_size(self, tmp_path: Path) -> None:
        mgr = RecoveryManager(tmp_path, tmp_path, 600)
        ctx = make_ctx(reason="oom")
        action, exp = mgr.handle_failure(ctx)

        assert action == RecoveryAction.OOM_BACKOFF
        assert exp["train_opts"]["batch_size"] == 4
        assert exp["status"] == "pending"
        assert exp["attempt"] == 0  # ctx.attempt - 1
        assert exp["oom_backoff_count"] == 1
        assert exp["consecutive_failures"] == 0

    def test_oom_at_min_batch_escalates_to_hard_failure(self, tmp_path: Path) -> None:
        exp = make_ctx(
            reason="oom",
            exp={
                "exp_id": "test_exp",
                "max_retries": 3,
                "train_opts": {"batch_size": 1},
                "oom_batch_candidates": [4, 2, 1],
                "max_oom_retries": 2,
                "oom_at_min_count": 2,  # already tried twice at min
            },
        ).exp
        ctx = FailureContext(
            exp=exp,
            run_exp_path=tmp_path / "exp.json",
            status_path=tmp_path / "status.json",
            attempt=3,
            rc=1,
            reason="oom",
            error_hash="abc",
        )
        mgr = RecoveryManager(tmp_path, tmp_path, 600)
        action, exp = mgr.handle_failure(ctx)

        assert action == RecoveryAction.HARD_FAILURE
        assert "OOM at minimum batch_size persists" in exp.get("last_reason", "")

    def test_oom_backoff_rollback_on_exhausted(self, tmp_path: Path) -> None:
        """When oom_count > max_oom backoff, batch_size mutation is rolled back."""
        exp_data = {
            "exp_id": "test_exp",
            "max_retries": 3,
            "train_opts": {"batch_size": 2},
            "oom_batch_candidates": [4, 2, 1],
            "max_oom_retries": 0,  # no backoff allowed at all
        }
        ctx = FailureContext(
            exp=exp_data,
            run_exp_path=tmp_path / "exp.json",
            status_path=tmp_path / "status.json",
            attempt=1,
            rc=1,
            reason="oom",
            error_hash="abc",
        )
        mgr = RecoveryManager(tmp_path, tmp_path, 600)
        with mock.patch("autopipe.worker.run_agent"):
            with mock.patch("autopipe.worker.snapshot_git"):
                action, exp = mgr.handle_failure(ctx)

        # Batch size rolled back; fall through to agent/failed path.
        assert exp["train_opts"]["batch_size"] == 2
        assert "last_oom_batch_size" not in exp
        # Since max_retries allows agent attempt, it should fall through to FAILED
        # (agent runs first time, then returns FAILED)
        assert action in (RecoveryAction.FAILED, RecoveryAction.HARD_FAILURE)

    def test_non_oom_failure_skips_oom_path(self, tmp_path: Path) -> None:
        """Non-OOM failures should not trigger batch_size reduction."""
        mgr = RecoveryManager(tmp_path, tmp_path, 600)
        exp_data = {
            "exp_id": "test_exp",
            "max_retries": 0,
            "train_opts": {"batch_size": 8},
            "oom_batch_candidates": [8, 4, 2, 1],
        }
        ctx = FailureContext(
            exp=exp_data,
            run_exp_path=tmp_path / "exp.json",
            status_path=tmp_path / "status.json",
            attempt=1,
            rc=1,
            reason="nccl",
            error_hash="xyz",
        )
        action, exp = mgr.handle_failure(ctx)

        # Batch size must remain unchanged for non-OOM failures.
        assert exp["train_opts"]["batch_size"] == 8
        assert action == RecoveryAction.FAILED

    def test_oom_backoff_caller_exp_not_mutated(self, tmp_path: Path) -> None:
        """Design #6: ctx.exp must not be mutated by handle_failure."""
        exp_data = {
            "exp_id": "test_exp",
            "max_retries": 3,
            "train_opts": {"batch_size": 8, "lr": 0.0005},
            "oom_batch_candidates": [8, 4, 2, 1],
            "max_oom_retries": 3,
        }
        mgr = RecoveryManager(tmp_path, tmp_path, 600)
        ctx = FailureContext(
            exp=exp_data,
            run_exp_path=tmp_path / "exp.json",
            status_path=tmp_path / "status.json",
            attempt=1,
            rc=1,
            reason="oom",
            error_hash="abc",
        )
        original_exp = dict(ctx.exp)  # snapshot
        action, returned_exp = mgr.handle_failure(ctx)

        # ctx.exp must be unchanged
        assert ctx.exp == original_exp
        # returned_exp must have the modifications
        assert returned_exp["train_opts"]["batch_size"] == 4


class TestRecoveryManagerAgent:
    """Tests for agent dispatch and dedup."""

    def test_first_seen_error_runs_agent(self, tmp_path: Path) -> None:
        mgr = RecoveryManager(tmp_path, tmp_path, 600)
        ctx = make_ctx(reason="shape", error_hash="new_error_hash")
        ctx.exp["error_hash"] = ""  # no previous error

        with mock.patch("autopipe.worker.run_agent") as mock_agent:
            with mock.patch("autopipe.worker.snapshot_git"):
                action, exp = mgr.handle_failure(ctx)
                mock_agent.assert_called_once()

        assert "agent_fix_hashes" in exp
        assert "new_error_hash" in exp["agent_fix_hashes"]

    def test_duplicate_error_below_threshold_skips_agent(self, tmp_path: Path) -> None:
        exp_data = {
            "exp_id": "test_exp",
            "max_retries": 3,
            "error_hash": "dup_hash_123",
            "agent_fix_hashes": {"dup_hash_123": 1},
            "train_opts": {},
        }
        ctx = FailureContext(
            exp=exp_data,
            run_exp_path=tmp_path / "exp.json",
            status_path=tmp_path / "status.json",
            attempt=2,
            rc=1,
            reason="shape",
            error_hash="dup_hash_123",
        )
        mgr = RecoveryManager(tmp_path, tmp_path, 600)

        with mock.patch("autopipe.worker.run_agent") as mock_agent:
            with mock.patch("autopipe.worker.snapshot_git"):
                action, exp = mgr.handle_failure(ctx)
                mock_agent.assert_not_called()

        assert exp["agent_fix_hashes"]["dup_hash_123"] == 2
        # Should have written agent_skip.txt
        assert (tmp_path / "agent_skip.txt").exists()

    def test_duplicate_error_exceeds_threshold_becomes_hard_failure(self, tmp_path: Path) -> None:
        exp_data = {
            "exp_id": "test_exp",
            "max_retries": 3,
            "error_hash": "stubborn_hash",
            "agent_fix_hashes": {"stubborn_hash": 2},  # already tried twice
            "hard_failure_threshold": 3,
            "train_opts": {},
        }
        ctx = FailureContext(
            exp=exp_data,
            run_exp_path=tmp_path / "exp.json",
            status_path=tmp_path / "status.json",
            attempt=2,
            rc=1,
            reason="shape",
            error_hash="stubborn_hash",
        )
        mgr = RecoveryManager(tmp_path, tmp_path, 600)

        action, exp = mgr.handle_failure(ctx)

        assert action == RecoveryAction.HARD_FAILURE
        assert "agent failed to fix" in exp.get("last_reason", "")

    def test_agent_failure_does_not_crash(self, tmp_path: Path) -> None:
        mgr = RecoveryManager(tmp_path, tmp_path, 600)
        ctx = make_ctx(reason="shape", error_hash="new_hash")
        ctx.exp["error_hash"] = ""

        with mock.patch("autopipe.worker.run_agent", side_effect=RuntimeError("agent crash")):
            with mock.patch("autopipe.worker.snapshot_git"):
                action, exp = mgr.handle_failure(ctx)
                # Should gracefully return FAILED, not raise
                assert action == RecoveryAction.FAILED

    def test_attempt_exceeds_max_retries_no_agent(self, tmp_path: Path) -> None:
        exp_data = {
            "exp_id": "test_exp",
            "max_retries": 1,
            "error_hash": "",
            "train_opts": {},
        }
        ctx = FailureContext(
            exp=exp_data,
            run_exp_path=tmp_path / "exp.json",
            status_path=tmp_path / "status.json",
            attempt=3,  # > max_retries
            rc=1,
            reason="nccl",
            error_hash="some_hash",
        )
        mgr = RecoveryManager(tmp_path, tmp_path, 600)

        with mock.patch("autopipe.worker.run_agent") as mock_agent:
            action, exp = mgr.handle_failure(ctx)
            mock_agent.assert_not_called()

        assert action == RecoveryAction.FAILED
