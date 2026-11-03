import unittest

import numpy as np

from common.project_audio import prepare
from analyzer.services import audio


class ProjectAudioTests(unittest.TestCase):
    def test_training_input_matches_app_ingest_and_float32_storage(self):
        for sample_rate, seconds in [(16000, .2), (44100, 1.5), (96000, 10.2)]:
            with self.subTest(sample_rate=sample_rate, seconds=seconds):
                times = np.arange(int(sample_rate * seconds)) / sample_rate
                source = np.stack([.2 * np.sin(2 * np.pi * 1000 * times),
                                   .1 * np.cos(2 * np.pi * 1700 * times)], axis=1).astype('float32')
                ingested = audio.prepare(source.T, sample_rate)
                stored = ingested.x.astype('float32')
                from_training = prepare(source, sample_rate)
                from_application = prepare(stored, ingested.sr)
                np.testing.assert_array_equal(from_training, from_application)
                self.assertEqual(from_training.shape, (160000,))
                self.assertEqual(from_training.dtype, np.float32)
                self.assertTrue(np.isfinite(from_training).all())

    def test_short_audio_is_zero_padded_as_waveform(self):
        source = np.ones(3200, dtype='float32') * .1
        result = prepare(source, 16000)
        np.testing.assert_array_equal(result[:3200], source)
        np.testing.assert_array_equal(result[3200:], 0.)


if __name__ == '__main__':
    unittest.main()
