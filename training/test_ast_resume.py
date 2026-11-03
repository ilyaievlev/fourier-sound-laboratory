"""Exercise the real two-phase loop across an epoch interruption on synthetic data."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch.utils.data import TensorDataset
from transformers import ASTConfig, ASTFeatureExtractor, ASTForAudioClassification
import yaml

from common.checkpoint import load_checkpoint
import train_ast


class ASTResumeTests(unittest.TestCase):
    def test_loop_resume_matches_all_remaining_epochs(self):
        config = yaml.safe_load((Path(__file__).parent / 'configs/ast_esc50.yaml').read_text())
        config.update(num_classes=2, max_length=32, sample_rate=16, duration_seconds=32,
                      batch_size=2, gradient_accumulation=2, head_epochs=2, finetune_epochs=2,
                      gradient_checkpointing=False, mixed_precision=False, time_shift_seconds=0,
                      specaugment_time_width=2, specaugment_frequency_width=2)
        def create(config, labels):
            model = ASTForAudioClassification(ASTConfig(hidden_size=16, num_hidden_layers=1,
                num_attention_heads=2, intermediate_size=32, num_labels=2,
                max_length=32, num_mel_bins=16, patch_size=8, time_stride=8, frequency_stride=8))
            return model, ASTFeatureExtractor(max_length=32, num_mel_bins=16), {}, {}
        def dataset(rows, config):
            wave = torch.arange(len(rows) * 512, dtype=torch.float32).reshape(len(rows), 512) / 1000
            return TensorDataset(wave, torch.tensor([row['target'] for row in rows]))
        def features(wave, *args, **kwargs):
            return wave.reshape(len(wave), 32, 16)
        def provenance(output, config, splits):
            output.mkdir(parents=True)
            (output / 'config.yaml').write_text(yaml.safe_dump(config))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [dict(filename=f'{i}.wav', target=i % 2, category=str(i % 2)) for i in range(8)]
            (root / 'splits.json').write_text(json.dumps({'1': dict(train=rows, validation=rows[:4], test=rows[:2])}))
            config['split_manifest'] = 'splits.json'
            def interrupt(output, detail):
                if detail.get('phase') == 'head' and detail.get('epoch') == 1:
                    self.partial = output
                    raise RuntimeError('simulated interruption')
            with patch.multiple(train_ast, ROOT=root, StreamingAudio=dataset, features=features,
                                create_ast=create, provenance=provenance,
                                environment=lambda: {'device': 'cpu'}):
                continuous = train_ast.run(config, 1, 'train')
                # Different directory name while leaving the saved config identical.
                with patch.object(train_ast.time, 'strftime', return_value='interrupted'):
                    with self.assertRaisesRegex(RuntimeError, 'simulated interruption'):
                        train_ast.run(config, 1, 'train', on_progress=interrupt)
                resumed = train_ast.run(config, 1, 'train', resume_dir=self.partial)
                expected = load_checkpoint(Path(continuous['output']) / 'checkpoint')
                actual = load_checkpoint(Path(resumed['output']) / 'checkpoint')
                for key, value in expected['model'].items():
                    torch.testing.assert_close(actual['model'][key], value, rtol=0, atol=0)
                self.assertEqual(actual['scheduler'], expected['scheduler'])
                self.assertEqual(resumed['best_validation_macro_f1'], continuous['best_validation_macro_f1'])
                self.assertEqual(len(resumed['history']), 4)


if __name__ == '__main__':
    unittest.main()
