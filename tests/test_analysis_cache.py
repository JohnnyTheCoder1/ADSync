from pathlib import Path

import numpy as np
import pytest

from adsync.cache import ArtifactCache, content_key
from test_pipeline_regressions import pipeline_case


def test_pipeline_reuses_analysis_and_recomputes_only_changed_source(pipeline_case, monkeypatch, tmp_path):
    monkeypatch.setenv("ADSYNC_CACHE_DIR", str(tmp_path / "artifacts"))
    Path("video.mkv").write_bytes(b"video source")
    Path("ad.wav").write_bytes(b"audio source")
    pipeline_case.run()
    assert len(pipeline_case.extractions) == 2
    pipeline_case.run()
    assert len(pipeline_case.extractions) == 2
    Path("ad.wav").write_bytes(b"new recording")
    pipeline_case.run()
    assert len(pipeline_case.extractions) == 3


def test_source_mutation_during_decode_cannot_poison_audio_cache(pipeline_case, monkeypatch, tmp_path):
    from adsync.media import extract

    cache_root = tmp_path / "artifacts"
    monkeypatch.setenv("ADSYNC_CACHE_DIR", str(cache_root))
    Path("video.mkv").write_bytes(b"video source")
    Path("ad.wav").write_bytes(b"old recording")
    options = {"kind": "preprocessed-audio", "stream": None, "sr": 8000,
               "mono": True, "highpass_hz": 80.0, "trim_silence": False}
    original_key = content_key(Path("ad.wav"), options)
    original_extract = extract.extract_audio
    def changing_extract(info, output_path, **kwargs):
        result = original_extract(info, output_path, **kwargs)
        if Path(info.path).name == "ad.wav":
            Path("ad.wav").write_bytes(b"new recording")
        return result
    monkeypatch.setattr(extract, "extract_audio", changing_extract)
    with pytest.raises(RuntimeError, match="Source changed"):
        pipeline_case.run()
    assert ArtifactCache(cache_root).get(original_key) is None


def test_cache_write_and_cleanup_failures_do_not_abort_processing(tmp_path, monkeypatch):
    cache = ArtifactCache(tmp_path)
    def unavailable(*args, **kwargs):
        raise OSError("Cache disconnected")
    monkeypatch.setattr("adsync.cache.os.replace", unavailable)
    monkeypatch.setattr(Path, "unlink", unavailable)
    cache.put("d" * 64, {"audio": np.zeros(4)})


def test_reuses_arrays_but_rejects_changed_source_with_same_size_and_timestamp(tmp_path):
    import os

    source = tmp_path / "source.bin"
    source.write_bytes(b"original")
    original_stat = source.stat()
    cache = ArtifactCache(tmp_path / "cache")
    key = content_key(source, {"stream": 1, "sr": 16000})
    cache.put(key, {"audio": np.arange(10, dtype=np.float32)})
    np.testing.assert_array_equal(cache.get(key)["audio"], np.arange(10))
    source.write_bytes(b"modified")
    os.utime(source, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
    changed = content_key(source, {"stream": 1, "sr": 16000})
    assert cache.get(changed) is None


def test_revision_stream_and_analysis_settings_invalidate_cache(tmp_path):
    source = tmp_path / "source.bin"
    source.write_bytes(b"sound")
    base = content_key(source, {"stream": 1, "sr": 16000}, revision="a")
    assert base != content_key(source, {"stream": 2, "sr": 16000}, revision="a")
    assert base != content_key(source, {"stream": 1, "sr": 8000}, revision="a")
    assert base != content_key(source, {"stream": 1, "sr": 16000}, revision="b")


def test_corrupt_cache_is_a_miss_and_can_be_rebuilt(tmp_path):
    cache = ArtifactCache(tmp_path)
    cache.put("a" * 64, {"audio": np.ones(4, dtype=np.float32)})
    path = next(tmp_path.rglob("*.npz"))
    path.write_bytes(b"interrupted transfer")
    assert cache.get("a" * 64) is None
    cache.put("a" * 64, {"audio": np.zeros(4, dtype=np.float32)})
    assert cache.get("a" * 64)["audio"].sum() == 0


def test_nonfinite_or_pickled_arrays_never_enter_cache(tmp_path):
    cache = ArtifactCache(tmp_path)
    for array in (np.array([float("nan")]), np.array([object()], dtype=object)):
        with pytest.raises(ValueError):
            cache.put("b" * 64, {"audio": array})
    assert cache.get("b" * 64) is None


def test_rejects_path_traversal_cache_keys(tmp_path):
    cache = ArtifactCache(tmp_path)
    with pytest.raises(ValueError):
        cache.put("../escape", {"audio": np.ones(1)})
