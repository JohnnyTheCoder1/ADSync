from io import StringIO

from rich.console import Console

from adsync.models import SyncReport
from adsync.report import json_report


def test_partial_summary_lists_matches_and_unmeasured_regions(monkeypatch):
    stream = StringIO()
    monkeypatch.setattr(json_report, "console", Console(file=stream, width=120, color_system=None))
    report = SyncReport(mode="partial", confidence=.99, alignment_review_required=True,
                        timing_debug={"partial_alignment": {
                            "playback_offset_adjust_sec": 1.0,
                            "matched_intervals": [{"source_start": 40., "source_end": 46.,
                                                   "target_start": 44., "target_end": 50.}],
                            "source_gaps": [{"start_sec": 46., "end_sec": 50.}],
                            "target_gaps": [],
                            "ambiguous_ranges": [{"start_sec": 41., "end_sec": 44.}],
                        }})
    json_report.print_summary(report)
    text = stream.getvalue()
    assert "AD 00:40" in text and "video 00:44" in text
    assert "before playback adjustment +1.00 s" in text
    assert "Unmeasured AD" in text and "00:46" in text
    assert "Competing" in text
    assert "Review required" in text
    assert "High alignment confidence" not in text
