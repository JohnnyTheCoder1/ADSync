"""Content-bound, disposable analysis artifacts. No media is ever published here."""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
import re
import tempfile
from typing import Any

import numpy as np

ANALYSIS_REVISION = "analysis-20260922-1"
log = logging.getLogger("adsync")


def content_key(source: Path, options: dict[str, Any], *, revision: str = ANALYSIS_REVISION) -> str:
    """Hash actual source bytes, semantic options, and the analysis revision."""
    before = source.stat()
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    after = source.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f"Source changed while computing analysis identity: {source}")
    return derived_key(digest.hexdigest(), options, revision=revision)


def derived_key(parent: str, options: dict[str, Any], *, revision: str = ANALYSIS_REVISION) -> str:
    payload = json.dumps({"source": parent, "options": options, "revision": revision},
                         sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def default_cache() -> ArtifactCache | None:
    """An explicit empty/disabled value turns caching off."""
    setting = os.environ.get("ADSYNC_CACHE_DIR")
    if setting is not None and setting.strip().lower() in {"", "off", "none", "0"}:
        return None
    if setting:
        root = Path(setting).expanduser()
    elif os.name == "nt":
        root = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")) / "ADSync" / "analysis"
    else:
        root = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "adsync" / "analysis"
    return ArtifactCache(root)


class ArtifactCache:
    def __init__(self, root: Path):
        self.root = Path(root)

    def _path(self, key: str) -> Path:
        if not re.fullmatch(r"[0-9a-f]{64}", key):
            raise ValueError("Cache key must be a SHA256 hex digest")
        return self.root / key[:2] / f"{key}.npz"

    def get(self, key: str) -> dict[str, np.ndarray] | None:
        path = self._path(key)
        try:
            with np.load(path, allow_pickle=False) as archive:
                arrays = {name: archive[name] for name in archive.files}
            self._validate(arrays)
            return arrays
        except FileNotFoundError:
            return None
        except (OSError, ValueError, EOFError, KeyError) as exc:
            log.warning("Ignoring unreadable analysis cache %s: %s", path, exc)
            return None
        except Exception as exc:
            # ZIP CRC failures are disposable cache misses, never media failures.
            import zipfile
            if isinstance(exc, zipfile.BadZipFile):
                log.warning("Ignoring corrupt analysis cache %s", path)
                return None
            raise

    @staticmethod
    def _validate(arrays: dict[str, np.ndarray]) -> None:
        if not arrays:
            raise ValueError("An empty artifact cannot be cached")
        for name, value in arrays.items():
            if not name or not isinstance(value, np.ndarray) or value.dtype.kind not in "biufc":
                raise ValueError("Only named numeric arrays may be cached")
            if not np.all(np.isfinite(value)):
                raise ValueError("Cached analysis must contain finite values")

    def put(self, key: str, arrays: dict[str, np.ndarray]) -> None:
        path = self._path(key)
        self._validate(arrays)
        temporary: Path | None = None
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=path.parent, prefix=key + ".", suffix=".tmp", delete=False) as handle:
                temporary = Path(handle.name)
                np.savez(handle, **arrays)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        except OSError as exc:
            # A full/unavailable cache must not prevent processing valid sources.
            log.warning("Could not save analysis cache %s: %s", path, exc)
        finally:
            if temporary is not None:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    log.debug("Could not remove disposable cache temporary %s", temporary)
