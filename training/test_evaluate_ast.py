from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch.utils.data import TensorDataset

import evaluate_ast


class NoiseEvaluationTests(unittest.TestCase):
    def test_noise_is_repeatable_and_has_requested_snr(self):
        torch.set_num_threads(2)
        wave = torch.arange(1, 129, dtype=torch.float32).reshape(2, 64) / 128
        dataset = TensorDataset(wave, torch.tensor([0, 0]))
        config = dict(seed=2026, test_fold=1, mixed_precision=False, num_classes=2)
        class Model:
            def __call__(self, input_values):
                return SimpleNamespace(logits=torch.tensor([[1., 0.]]))
        observed = []
        def capture(wave, *args):
            observed.append(wave.clone())
            return wave
        with patch.object(evaluate_ast, 'features', capture):
            evaluate_ast.predict(Model(), None, dataset, config, torch.device('cpu'), None)
            torch.testing.assert_close(torch.cat(observed), wave, rtol=0, atol=0)
            observed.clear()
            first = evaluate_ast.predict(Model(), None, dataset, config, torch.device('cpu'), 10)
            noisy = torch.cat(observed)
            observed.clear()
            second = evaluate_ast.predict(Model(), None, dataset, config, torch.device('cpu'), 10)
            torch.testing.assert_close(torch.cat(observed), noisy, rtol=0, atol=0)
            snr = 10 * torch.log10(wave.square().mean(1) / (noisy - wave).square().mean(1))
            torch.testing.assert_close(snr, torch.full((2,), 10.), atol=1e-5, rtol=0)
            self.assertEqual(first['predicted'], second['predicted'])


if __name__ == '__main__':
    unittest.main()
