"""Structured logging helpers."""

from __future__ import annotations

import logging
import sys

from rich.console import Console

console = Console(stderr=True, emoji=False)

_LOG_FORMAT = "%(message)s"


def setup_logging(*, verbose: bool = False) -> logging.Logger:
    """Configure and return the root adsync logger."""
    level = logging.DEBUG if verbose else logging.INFO
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(asctime)s %(message)s", datefmt="%H:%M:%S"))
    handler.setLevel(level)

    logger = logging.getLogger("adsync")
    logger.setLevel(level)
    logger.handlers.clear()
    logger.addHandler(handler)
    return logger


log = logging.getLogger("adsync")
