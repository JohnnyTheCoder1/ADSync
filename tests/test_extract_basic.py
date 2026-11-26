"""Blockwise feature extraction agrees with the full-spectrogram reference."""
from __future__ import annotations

from unittest import TestCase
from unittest.mock import patch

import librosa
import numpy as np

from adsync.features.extract_basic import extract_basic_features


def _reference(y, sr, n_fft, hop, n_mels, n_mfcc):
    rms = librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop)[0]
    power = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop))
    np.square(power, out=power)
    mel = librosa.power_to_db(
        librosa.feature.melspectrogram(S=power, sr=sr, n_mels=n_mels), ref=np.max,
    )
    full_mel = librosa.power_to_db(librosa.feature.melspectrogram(S=power, sr=sr))
    return {
        "rms": rms,
        "mel": mel,
        "onset": librosa.onset.onset_strength(S=full_mel, sr=sr, hop_length=hop),
        "mfcc": librosa.feature.mfcc(S=full_mel, sr=sr, n_mfcc=n_mfcc),
    }


class BasicFeatureTests(TestCase):
    def test_matches_full_spectrogram_across_block_boundaries(self) -> None:
        rng = np.random.default_rng(54)
        cases = (
            (2048, 512, 256017, np.float32, 13),
            (513, 128, 81659, np.float64, 13),
            (2048, 512, 513, np.float32, 13),
            (512, 128, 18943, np.float32, 132),
        )
        for n_fft, hop, length, dtype, n_mfcc in cases:
            with self.subTest(n_fft=n_fft, hop=hop, length=length, dtype=dtype, n_mfcc=n_mfcc):
                audio = rng.standard_normal(length).astype(dtype)
                audio[:length // 3] *= 0.005  # Whole-track reference must survive quiet blocks.
                audio[length // 3:length // 2] = 0
                expected = _reference(audio, 16000, n_fft, hop, 48, n_mfcc)
                with (patch("adsync.features.extract_basic._BLOCK_FRAMES", 37),
                      patch("adsync.features.extract_basic.librosa.stft", wraps=librosa.stft) as fft):
                    actual = extract_basic_features(
                        audio, 16000, n_fft=n_fft, hop_length=hop, n_mels=48, n_mfcc=n_mfcc,
                    )
                self.assertEqual(fft.call_count, (len(expected["rms"]) + 36) // 37)
                self.assertTrue(all(len(call.args[0]) <= n_fft + 36 * hop for call in fft.call_args_list))
                self.assertEqual(actual.duration, length / 16000)
                for name, reference in expected.items():
                    np.testing.assert_allclose(
                        getattr(actual, name), reference, rtol=1e-4, atol=2e-4, err_msg=name,
                    )
