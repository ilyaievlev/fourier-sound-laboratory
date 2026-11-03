import unittest
from pathlib import Path
import tempfile
from unittest.mock import patch

import numpy as np
import torch

from common.anomaly_metrics import evaluate_scores
from common.anomaly_model import true_id_score
from common.anomaly_model import from_config
from common.features import pump_logmel, sliding_crops


class AnomalyTests(unittest.TestCase):
    def test_real_pump_loop_resumes_at_committed_epoch(self):
        import train_anomaly
        from common.checkpoint import load_checkpoint
        from train_baselines import seed_all
        config = dict(seed=2026, batch_size=2, learning_rate=.001, benchmark_steps=6,
                      epochs=3, patience=5, threshold_quantile=.9, crop_frames=64,
                      crop_hop_frames=32, id_list=['id_00', 'id_02', 'id_04', 'id_06'])
        train_rows = [dict(file=f'{machine}_{label}', machine_id=machine, label=label)
                      for machine in config['id_list'] for label in [0, 1]]
        manifest = dict(splits=dict(train=train_rows, validation=train_rows, test=train_rows))
        class ToyData(torch.utils.data.Dataset):
            def __init__(self, rows, _config):
                self.rows = rows
            def __len__(self):
                return len(self.rows)
            def __getitem__(self, index):
                # Exercise both torch and numpy RNG, as real random crop selection does.
                return torch.randn(8) + float(np.random.random()), index % 4
        def scores(model, data, _config, device):
            model.eval()
            return [dict(row, score=.1 + row['label']) for row in data.rows], .5
        original_save = train_anomaly.save_checkpoint
        def interrupted_save(*args):
            original_save(*args)
            if args[-1]['epoch'] == 1:
                raise RuntimeError('simulated stop after commit')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            continuous, interrupted = root / 'continuous', root / 'interrupted'
            continuous.mkdir(); interrupted.mkdir()
            with patch.multiple(train_anomaly, ROOT=root, PumpData=ToyData, cnn_scores=scores,
                                from_config=lambda _: torch.nn.Linear(8, 4)):
                seed_all(config['seed'])
                train_anomaly.train_cnn(continuous, manifest, config, torch.device('cpu'))
                with patch.object(train_anomaly, 'save_checkpoint', interrupted_save):
                    with self.assertRaisesRegex(RuntimeError, 'simulated stop'):
                        train_anomaly.train_cnn(interrupted, manifest, config, torch.device('cpu'))
                train_anomaly.train_cnn(interrupted, manifest, config, torch.device('cpu'))
            expected = load_checkpoint(continuous / 'checkpoint')
            actual = load_checkpoint(interrupted / 'checkpoint')
            for key, value in expected['model'].items():
                torch.testing.assert_close(actual['model'][key], value, rtol=0, atol=0)
            self.assertEqual(expected['progress']['history'], actual['progress']['history'])

    def test_exported_cnn_preserves_clip_score(self):
        from safetensors.torch import load_file, save_file
        torch.set_num_threads(2)
        config = dict(architecture='PumpIDCNN', num_ids=4, channels=[8, 16, 32, 48], input_offset=40., input_scale=40.)
        model = from_config(config).eval()
        crops = torch.randn(3, 128, 64) * 10 - 40
        with tempfile.TemporaryDirectory() as directory, torch.inference_mode():
            before = true_id_score(model(crops), 2)
            path = str(Path(directory) / 'model.safetensors')
            save_file(model.state_dict(), path)
            restored = from_config(config).eval()
            restored.load_state_dict(load_file(path))
            torch.testing.assert_close(true_id_score(restored(crops), 2), before, rtol=0, atol=0)

    def test_silence_floor_and_sliding_tail(self):
        mel = pump_logmel(np.zeros(160000, dtype=np.float32))
        self.assertEqual(mel.shape, (128, 313))
        np.testing.assert_allclose(mel, -100., atol=1e-5)
        crops = sliding_crops(np.tile(np.arange(313, dtype=np.float32), (128, 1)))
        self.assertEqual(crops.shape[1:], (128, 64))
        self.assertEqual(crops[-1, 0, -1], 312)

    def test_score_is_mean_negative_log_not_log_mean(self):
        probabilities = torch.tensor([[.9, .1], [.1, .9]])
        score = float(true_id_score(probabilities.log(), 0))
        self.assertAlmostEqual(score, -.5 * (np.log(.9) + np.log(.1)), places=6)
        self.assertNotAlmostEqual(score, -np.log(.5), places=3)

    def test_perfect_ranking_auc_and_pauc(self):
        rows = [dict(machine_id='id_00', label=label, score=score)
                for label, score in [(0, .1), (0, .2), (1, .8), (1, .9)]]
        metrics, roc = evaluate_scores(rows, {'id_00': .5}, ['id_00'])
        self.assertEqual(metrics['mean_auc'], 1.)
        self.assertEqual(metrics['mean_pauc'], 1.)
        self.assertEqual(metrics['id_00']['confusion'], [[2, 0], [0, 2]])
        self.assertIsNone(roc['id_00']['thresholds'][0])


if __name__ == '__main__':
    unittest.main()
