"""Completed media becomes visible atomically, with recoverable publication errors."""

from __future__ import annotations

import errno
import os
from pathlib import Path

import pytest


def test_output_destination_precedence(tmp_path, monkeypatch):
    from adsync.media.output import resolve_output_path

    source = tmp_path / "inputs" / "Episode.mkv"
    monkeypatch.delenv("ADSYNC_OUTPUT_DIR", raising=False)
    assert resolve_output_path(source) == source.with_name("Episode.synced.mkv")
    monkeypatch.setenv("ADSYNC_OUTPUT_DIR", str(tmp_path / "library"))
    assert resolve_output_path(source) == tmp_path / "library" / "Episode.synced.mkv"
    assert resolve_output_path(source, output_dir=tmp_path / "override") == tmp_path / "override" / "Episode.synced.mkv"
    assert resolve_output_path(source, tmp_path / "chosen.mkv", tmp_path / "ignored") == tmp_path / "chosen.mkv"
    assert resolve_output_path(source, suffix=".prepped.mkv") == tmp_path / "library" / "Episode.prepped.mkv"


@pytest.mark.skipif(os.name != "nt", reason="Windows UNC path semantics")
def test_unc_output_dir_keeps_share_and_filename(monkeypatch):
    from adsync.media.output import resolve_output_path

    monkeypatch.delenv("ADSYNC_OUTPUT_DIR", raising=False)
    result = resolve_output_path(Path(r"C:\input\Episode.mkv"), output_dir=Path(r"\\laptop\Jellyfin\Movies"))
    assert str(result) == r"\\laptop\Jellyfin\Movies\Episode.synced.mkv"


def test_local_render_replaces_existing_only_after_completion(tmp_path):
    from adsync.media.output import staged_output

    destination = tmp_path / "movie.mkv"
    destination.write_bytes(b"previous complete media")
    with staged_output(destination) as local:
        assert local != destination
        assert local.suffix == ".mkv"
        local.write_bytes(b"new complete media")
        assert destination.read_bytes() == b"previous complete media"
    assert destination.read_bytes() == b"new complete media"
    assert not local.exists()
    assert not local.parent.exists()


def test_failed_render_preserves_existing_output_and_cleans_stage(tmp_path):
    from adsync.media.output import staged_output

    destination = tmp_path / "movie.mkv"
    destination.write_bytes(b"previous complete media")
    with pytest.raises(RuntimeError, match="encoder failed"):
        with staged_output(destination) as local:
            local.write_bytes(b"unfinished")
            raise RuntimeError("encoder failed")
    assert destination.read_bytes() == b"previous complete media"
    assert not local.parent.exists()


@pytest.mark.parametrize("hardlink", [False, True])
def test_input_alias_is_rejected_before_render(tmp_path, hardlink):
    from adsync.media.output import staged_output

    source = tmp_path / "source.mkv"
    source.write_bytes(b"source media")
    destination = tmp_path / "alias.mkv" if hardlink else source
    if hardlink:
        os.link(source, destination)
    with pytest.raises(ValueError, match="input"):
        with staged_output(destination, inputs=[source]):
            pytest.fail("Unsafe destination was allowed to render")
    assert source.read_bytes() == b"source media"


def test_publication_failure_keeps_completed_local_file(tmp_path):
    from adsync.media.output import OutputPublicationError, staged_output

    blocker = tmp_path / "unavailable_share"
    blocker.write_bytes(b"not a directory")
    destination = blocker / "movie.mkv"
    with pytest.raises(OutputPublicationError) as caught:
        with staged_output(destination) as local:
            local.write_bytes(b"completed media")
    try:
        assert caught.value.local_path == local
        assert caught.value.destination == destination
        assert local.read_bytes() == b"completed media"
        assert str(local) in str(caught.value)
    finally:
        local.unlink()
        local.parent.rmdir()


def _force_cross_volume(monkeypatch, output_module, destination):
    """An ordinary temp directory cannot reproduce a second/network volume."""
    real_replace = output_module.os.replace

    def replace(source, target):
        if Path(source).parent != destination.parent:
            raise OSError(errno.EXDEV, "Different filesystem")
        return real_replace(source, target)

    monkeypatch.setattr(output_module.os, "replace", replace)


def test_cross_volume_copy_uses_nonmedia_partial_then_renames(tmp_path, monkeypatch):
    from adsync.media import output

    destination = tmp_path / "library" / "movie.mkv"
    destination.parent.mkdir()
    destination.write_bytes(b"old complete media")
    _force_cross_volume(monkeypatch, output, destination)
    real_copy = output.shutil.copyfileobj

    def observe_copy(source, target, length):
        partials = list(destination.parent.iterdir())
        assert destination.read_bytes() == b"old complete media"
        assert len(partials) == 2
        assert next(p for p in partials if p != destination).suffix == ".adsync-part"
        return real_copy(source, target, length)

    monkeypatch.setattr(output.shutil, "copyfileobj", observe_copy)
    with output.staged_output(destination) as local:
        local.write_bytes(b"new complete media")
    assert destination.read_bytes() == b"new complete media"
    assert list(destination.parent.iterdir()) == [destination]
    assert not local.parent.exists()


def test_interrupted_cross_volume_copy_keeps_local_and_existing_output(tmp_path, monkeypatch):
    from adsync.media import output

    destination = tmp_path / "movie.mkv"
    destination.write_bytes(b"old complete media")
    _force_cross_volume(monkeypatch, output, destination)

    def disconnected_copy(source, target, length):
        target.write(source.read(3))
        raise OSError("Network disconnected")

    monkeypatch.setattr(output.shutil, "copyfileobj", disconnected_copy)
    with pytest.raises(output.OutputPublicationError) as caught:
        with output.staged_output(destination) as local:
            local.write_bytes(b"new complete media")
    try:
        assert destination.read_bytes() == b"old complete media"
        assert local.read_bytes() == b"new complete media"
        assert caught.value.local_path == local
        assert list(destination.parent.iterdir()) == [destination]
    finally:
        local.unlink()
        local.parent.rmdir()


def test_failed_final_rename_keeps_local_and_existing_output(tmp_path, monkeypatch):
    from adsync.media import output

    destination = tmp_path / "movie.mkv"
    destination.write_bytes(b"old complete media")

    def unavailable_replace(source, target):
        if Path(source).parent != destination.parent:
            raise OSError(errno.EXDEV, "Different filesystem")
        raise PermissionError("Destination locked")

    monkeypatch.setattr(output.os, "replace", unavailable_replace)
    with pytest.raises(output.OutputPublicationError):
        with output.staged_output(destination) as local:
            local.write_bytes(b"new complete media")
    try:
        assert destination.read_bytes() == b"old complete media"
        assert local.read_bytes() == b"new complete media"
        assert list(destination.parent.iterdir()) == [destination]
    finally:
        local.unlink()
        local.parent.rmdir()


def test_empty_render_is_never_published(tmp_path):
    from adsync.media.output import staged_output

    destination = tmp_path / "movie.mkv"
    destination.write_bytes(b"old complete media")
    with pytest.raises(ValueError, match="empty"):
        with staged_output(destination) as local:
            local.touch()
    assert destination.read_bytes() == b"old complete media"
    assert not local.parent.exists()


def test_publish_existing_file_copies_atomically_without_consuming_source(tmp_path, monkeypatch):
    from adsync.media import output

    source = tmp_path / "prepped.mkv"
    source.write_bytes(b"ready prepped media")
    destination = tmp_path / "library" / "movie.mkv"
    assert output.publish_completed_file(source, destination) == destination
    assert source.read_bytes() == destination.read_bytes() == b"ready prepped media"
    assert list(destination.parent.iterdir()) == [destination]


def test_publish_existing_file_failure_keeps_source_and_old_destination(tmp_path, monkeypatch):
    from adsync.media import output

    source = tmp_path / "prepped.mkv"
    source.write_bytes(b"ready prepped media")
    destination = tmp_path / "movie.mkv"
    destination.write_bytes(b"old complete media")

    def disconnected_copy(source, target, length):
        target.write(source.read(2))
        raise OSError("Network disconnected")

    monkeypatch.setattr(output.shutil, "copyfileobj", disconnected_copy)
    with pytest.raises(output.OutputPublicationError) as caught:
        output.publish_completed_file(source, destination)
    assert source.read_bytes() == b"ready prepped media"
    assert caught.value.local_path == source
    assert destination.read_bytes() == b"old complete media"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["movie.mkv", "prepped.mkv"]


def test_silent_copy_corruption_is_rejected_before_replacing_destination(tmp_path, monkeypatch):
    from adsync.media import output

    source = tmp_path / "source.mkv"
    source.write_bytes(b"complete correct media")
    destination = tmp_path / "library" / "movie.mkv"
    destination.parent.mkdir()
    destination.write_bytes(b"previous good media")

    def corrupt_copy(source, target, length):
        target.write(b"complete corrupt media")

    monkeypatch.setattr(output.shutil, "copyfileobj", corrupt_copy)
    with pytest.raises(output.OutputPublicationError, match="hash|integrity"):
        output.publish_completed_file(source, destination)
    assert destination.read_bytes() == b"previous good media"
    assert source.read_bytes() == b"complete correct media"


def test_publications_to_one_destination_volume_are_serialized(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    import threading
    import time
    from adsync.media import output

    source = tmp_path / "source.mkv"
    source.write_bytes(b"complete media")
    active = peak = 0
    mutex = threading.Lock()
    original = output.shutil.copyfileobj

    def slow_copy(source, target, length):
        nonlocal active, peak
        with mutex:
            active += 1
            peak = max(peak, active)
        try:
            time.sleep(0.05)
            original(source, target, length)
        finally:
            with mutex:
                active -= 1

    monkeypatch.setattr(output.shutil, "copyfileobj", slow_copy)
    with ThreadPoolExecutor(max_workers=3) as pool:
        destinations = [tmp_path / f"library{i}" / "movie.mkv" for i in range(3)]
        list(pool.map(lambda path: output.publish_completed_file(source, path), destinations))
    assert peak == 1
    assert all(path.read_bytes() == b"complete media" for path in destinations)
