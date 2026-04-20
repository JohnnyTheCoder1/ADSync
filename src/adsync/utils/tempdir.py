"""Temp directory helpers."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path


class WorkDir:
    """Managed temporary working directory.

    If *root* is given the directory is created there and NOT automatically
    cleaned up (the caller owns it).  Otherwise a system tempdir is used and
    will be cleaned on :meth:`cleanup` or context-manager exit.
    """

    def __init__(self, root: Path | None = None) -> None:
        if root is not None:
            self.path = Path(root)
            self.path.mkdir(parents=True, exist_ok=True)
            self._auto_clean = False
        else:
            self._tmpdir = tempfile.mkdtemp(prefix="adsync_")
            self.path = Path(self._tmpdir)
            self._auto_clean = True

    # convenience sub-path creators
    def child(self, name: str) -> Path:
        p = self.path / name
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def subdir(self, name: str) -> Path:
        p = self.path / name
        p.mkdir(parents=True, exist_ok=True)
        return p

    # lifecycle
    def cleanup(self) -> None:
        if self._auto_clean and self.path.exists():
            shutil.rmtree(self.path, ignore_errors=True)

    def __enter__(self) -> "WorkDir":
        return self

    def __exit__(self, *_: object) -> None:
        self.cleanup()
