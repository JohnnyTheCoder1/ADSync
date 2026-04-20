"""Tests for the CLI commands."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from adsync.cli import app

runner = CliRunner()


def test_version() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "adsync" in result.stdout


def test_sync_missing_file() -> None:
    result = runner.invoke(app, ["sync", "nonexistent.mkv", "nonexistent.m4a"])
    assert result.exit_code == 2


def test_analyze_missing_file() -> None:
    result = runner.invoke(app, ["analyze", "nonexistent.mkv", "nonexistent.m4a"])
    assert result.exit_code == 2


def test_mux_missing_file() -> None:
    result = runner.invoke(app, ["mux", "nonexistent.mkv", "nonexistent.m4a"])
    assert result.exit_code == 2


def test_debug_missing_file() -> None:
    result = runner.invoke(app, ["debug", "nonexistent.mkv", "nonexistent.m4a"])
    assert result.exit_code == 2
