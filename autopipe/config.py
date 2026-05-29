from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


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

