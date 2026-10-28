"""Crashed publications clean up after themselves; unreadable outputs are not mistaken for changed ones."""

import json
import os

import pytest

from adsync.batch.transaction import commit_state, content_signature
from adsync.media.output import publish_completed_file


def test_publish_removes_partials_left_by_a_killed_run(tmp_path):
    local = tmp_path / "local" / "Show.S01E01.AD.mkv"
    local.parent.mkdir()
    local.write_bytes(b"rendered" * 1000)
    season = tmp_path / "library" / "Season 01"
    season.mkdir(parents=True)
    (season / ".adsync-k2j3h4.adsync-part").write_bytes(b"left by a killed run")
    (season / ".Show.S01E02.AD.mkv.adsync-part").write_bytes(b"left by a killed run")
    publish_completed_file(local, season / "Show.S01E01.AD.mkv")
    assert sorted(p.name for p in season.iterdir()) == ["Show.S01E01.AD.mkv"]


def _committed_episode(tmp_path):
    fingerprint = {"video": "v", "ad": "a"}
    files = {name: tmp_path / name for name in ("episode.AD.mkv", "episode.report.json", "episode.qc.json")}
    for path in files.values():
        path.write_bytes(b"published bytes" * 100)
    commit = {"phase": "transfer_verified", "fingerprint": fingerprint,
              "output_signature": content_signature(files["episode.AD.mkv"]),
              "report_signature": content_signature(files["episode.report.json"]),
              "qc_signature": content_signature(files["episode.qc.json"])}
    commit_path = tmp_path / "episode.commit.json"
    commit_path.write_text(json.dumps(commit), encoding="utf-8")
    return commit_path, fingerprint, files["episode.AD.mkv"]


def test_verified_commit_matches(tmp_path):
    commit_path, fingerprint, _ = _committed_episode(tmp_path)
    assert commit_state(commit_path, fingerprint) == "match"


def test_changed_output_is_a_mismatch(tmp_path):
    commit_path, fingerprint, output = _committed_episode(tmp_path)
    output.write_bytes(b"replaced by someone else" * 100)
    assert commit_state(commit_path, fingerprint) == "mismatch"


@pytest.mark.skipif(os.name != "nt", reason="byte-range lock semantics are Windows-specific")
def test_output_that_cannot_be_read_right_now_is_unreadable_not_changed(tmp_path):
    import msvcrt

    commit_path, fingerprint, output = _committed_episode(tmp_path)
    with output.open("rb+") as handle:
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 64)
        try:
            assert commit_state(commit_path, fingerprint) == "unreadable"
        finally:
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 64)
