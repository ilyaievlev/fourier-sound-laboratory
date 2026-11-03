"""Verify exported weights reproduce predictions and evaluation is deterministic."""
from pathlib import Path
import tempfile
import unittest

from safetensors.torch import load_file, save_file
import torch
import yaml

from common.baseline_models import make_model


class ExportTests(unittest.TestCase):
    def test_safetensors_roundtrip_and_eval_determinism(self):
        config = yaml.safe_load((Path(__file__).parent / 'configs/baselines.yaml').read_text())
        torch.set_num_threads(2)
        torch.manual_seed(2026)
        wave = torch.randn(2, 80000)
        with tempfile.TemporaryDirectory() as temp:
            for name in config['models']:
                with self.subTest(model=name), torch.inference_mode():
                    original = make_model(name, config).eval()
                    expected = original(wave)
                    self.assertEqual(tuple(expected.shape), (2, 50))
                    self.assertTrue(torch.isfinite(expected).all())
                    torch.testing.assert_close(original(wave), expected, rtol=0, atol=0)
                    path = str(Path(temp) / f'{name}.safetensors')
                    save_file(original.state_dict(), path)
                    restored = make_model(name, config).eval()
                    restored.load_state_dict(load_file(path), strict=True)
                    torch.testing.assert_close(restored(wave), expected, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
