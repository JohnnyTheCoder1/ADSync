"""I/O convenience helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_json(path: Path | str, data: Any) -> Path:
    """Serialize *data* to pretty JSON at *path*."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    return path


def read_json(path: Path | str) -> Any:
    """Read JSON from *path*."""
    return json.loads(Path(path).read_text(encoding="utf-8"))
