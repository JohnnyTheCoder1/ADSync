"""Receipts written before a setting existed stay valid while that setting is unset."""

import json

from test_season_runner import season


def test_receipts_from_before_ad_stream_existed_still_resume(season):
    first = season.run()
    assert first.counts == {"completed": 4}
    for commit in season.state_dir.rglob("*.commit.json"):
        record = json.loads(commit.read_text(encoding="utf-8"))
        record["fingerprint"]["config"].pop("ad_stream", None)
        commit.write_text(json.dumps(record), encoding="utf-8")
    assert season.run().counts == {"skipped_completed": 4}
