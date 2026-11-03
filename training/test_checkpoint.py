"""Interrupted vs continuous training must produce identical next-epoch weights."""
from pathlib import Path
import random
import tempfile
import unittest

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from common.checkpoint import load_checkpoint, restore_rng, save_checkpoint


class CheckpointTests(unittest.TestCase):
    def test_epoch_resume_matches_continuous_training(self):
        torch.set_num_threads(2)
        torch.manual_seed(2026)
        np.random.seed(2026)
        random.seed(2026)
        data = TensorDataset(torch.randn(20, 4), torch.randn(20, 2))
        def components():
            model = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.Dropout(.2), torch.nn.Linear(8, 2))
            optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: .95 ** step)
            scaler = torch.amp.GradScaler('cuda', enabled=False)
            return model, optimizer, scheduler, scaler
        def epoch(model, optimizer, scheduler):
            losses = []
            for x, target in DataLoader(data, batch_size=4, shuffle=True):
                optimizer.zero_grad()
                perturbed = x + torch.randn_like(x) * np.random.uniform(0, .1) + random.random() * .01
                loss = (model(perturbed) - target).square().mean()
                loss.backward()
                optimizer.step()
                scheduler.step()
                losses.append(loss.item())
            return losses
        model, optimizer, scheduler, scaler = components()
        epoch(model, optimizer, scheduler)
        with tempfile.TemporaryDirectory() as directory:
            save_checkpoint(directory, model, optimizer, scheduler, scaler, {'epoch': 1})
            expected_loss = epoch(model, optimizer, scheduler)
            expected = {k: v.clone() for k, v in model.state_dict().items()}
            restored, new_optimizer, new_scheduler, new_scaler = components()
            state = load_checkpoint(directory)
            restored.load_state_dict(state['model'])
            new_optimizer.load_state_dict(state['optimizer'])
            new_scheduler.load_state_dict(state['scheduler'])
            new_scaler.load_state_dict(state['scaler'])
            restore_rng(state['rng'])
            self.assertEqual(epoch(restored, new_optimizer, new_scheduler), expected_loss)
            for key, actual in restored.state_dict().items():
                torch.testing.assert_close(actual, expected[key], rtol=0, atol=0)
            self.assertEqual(new_scheduler.state_dict(), scheduler.state_dict())
            # A subsequent checkpoint replaces the pointer and keeps only one committed blob.
            save_checkpoint(directory, restored, new_optimizer, new_scheduler, new_scaler, {'epoch': 2})
            self.assertEqual(load_checkpoint(directory)['progress']['epoch'], 2)
            self.assertEqual(len(list(Path(directory).glob('*.safetensors'))), 1)

    def test_corrupt_blob_is_rejected(self):
        import json
        model = torch.nn.Linear(1, 1)
        optimizer = torch.optim.AdamW(model.parameters())
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.)
        with tempfile.TemporaryDirectory() as directory:
            save_checkpoint(directory, model, optimizer, scheduler, torch.amp.GradScaler('cuda', enabled=False), {})
            pointer = json.loads((Path(directory) / 'latest.json').read_text())
            (Path(directory) / pointer['blob']).write_bytes(b'corrupt')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                load_checkpoint(directory)


if __name__ == '__main__':
    unittest.main()
