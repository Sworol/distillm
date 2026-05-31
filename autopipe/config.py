from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# Default max number of agent repair attempts for a single unique error hash
# before marking the experiment as hard_failure.  Used by both make_queue and
# the worker; keep them in sync via this single constant.
HARD_FAILURE_THRESHOLD = 3

# Config keys that the scheduler may merge from queue definition into the
# per-experiment working copy.  Bookkeeping fields (attempt, status,
# updated_at, last_reason, error_hash, etc.) and train_opts are intentionally
# excluded — the agent edits train_opts and merges would clobber those fixes.
CONFIG_MERGE_KEYS: frozenset[str] = frozenset([
    "cfg_path", "trainer", "cmd", "cmd_type", "key", "conda_env",
    "gpus", "nproc", "master_port", "hf_endpoint",
    "train_timeout", "hang_timeout", "vis_timeout", "vis_opts",
    "skip_vis", "retry_sleep", "oom_batch_candidates",
    "max_retries", "agent_cli", "hard_failure_threshold",
])


@dataclass(frozen=True)
class Paths:
    root: Path

    @property
    def queue_dir(self) -> Path:
        return self.root / "queue"

    @property
    def runs_dir(self) -> Path:
        return self.root / "runs"

    @property
    def logs_dir(self) -> Path:
        return self.root / "logs"


def default_paths(repo_root: Path) -> Paths:
    return Paths(root=repo_root / "autopipe")

