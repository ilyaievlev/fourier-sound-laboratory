from pathlib import Path
import tempfile
import unittest

import torch
from transformers import ASTConfig, ASTForAudioClassification

from common.ast_model import crop_positions


class ASTPositionTests(unittest.TestCase):
    def test_frequency_time_crop_and_saved_reload(self):
        torch.set_num_threads(2)
        model = ASTForAudioClassification(ASTConfig(hidden_size=16, num_hidden_layers=1,
            num_attention_heads=2, intermediate_size=32, num_labels=50, max_length=1024))
        embedding = model.audio_spectrogram_transformer.embeddings
        with torch.no_grad():
            embedding.position_embeddings.copy_(torch.arange(1214 * 16).reshape(1, 1214, 16) / 10000)
        original = embedding.position_embeddings.detach().clone()
        layout = crop_positions(model, 512)
        self.assertEqual(layout['tokens'], 602)
        self.assertEqual(layout['time_patches'], 50)
        expected = original[:, 2:].reshape(1, 12, 101, 16)[:, :, :50].reshape(1, 600, 16)
        torch.testing.assert_close(embedding.position_embeddings[:, 2:], expected)
        torch.testing.assert_close(embedding.position_embeddings[:, :2], original[:, :2])
        model.eval()
        inputs = torch.zeros(1, 512, 128)
        with tempfile.TemporaryDirectory() as directory, torch.inference_mode():
            prediction = model(inputs).logits
            model.save_pretrained(directory, safe_serialization=True)
            self.assertTrue((Path(directory) / 'model.safetensors').exists())
            restored, info = ASTForAudioClassification.from_pretrained(directory, output_loading_info=True)
            self.assertFalse(any(info.get(key) for key in ['missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs']))
            restored.eval()
            torch.testing.assert_close(restored(inputs).logits, prediction)


if __name__ == '__main__':
    unittest.main()
