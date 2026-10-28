import numpy as np

from adsync.align.fingerprint import _offset_spans


def test_short_nonzero_votes_survive_for_independent_qc_after_inlier_filtering():
    times = np.r_[np.arange(0.1, 60, 0.05), np.arange(42.0, 44.0, 0.05)].astype(np.float32)
    offsets = np.r_[np.zeros(1198), np.full(40, -0.8)].astype(np.float32)
    # A dominant zero-offset ten-second bucket filters the minority -0.8s
    # votes from steering. QC still needs those original observations.
    targets = times + offsets
    result = _offset_spans(times, targets, 60.0)
    assert result.match_t_ad is not None
    assert not np.any(np.isclose(result.match_t_vid - result.match_t_ad, -0.8, atol=0.005))
    raw_ad = getattr(result, "raw_match_t_ad", None)
    raw_video = getattr(result, "raw_match_t_vid", None)
    assert raw_ad is not None and raw_video is not None
    assert len(raw_ad) == len(times)
    assert np.count_nonzero(np.isclose(raw_video - raw_ad, -0.8, atol=0.005)) == 40
