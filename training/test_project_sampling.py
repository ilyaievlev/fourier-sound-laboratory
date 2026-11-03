import unittest
import torch
from torch.utils.data import Dataset

from train_project import balanced_loader


class ProjectSamplingTests(unittest.TestCase):
    def test_imbalanced_classes_receive_equal_total_sampling_mass(self):
        class Toy(Dataset):
            rows = [dict(target=0)] * 90 + [dict(target=1)] * 10
            def __len__(self):
                return len(self.rows)
            def __getitem__(self, index):
                return index
        dataset = Toy()
        loader = balanced_loader(dataset, batch_size=4, shuffle=True, num_workers=0)
        self.assertAlmostEqual(float(loader.sampler.weights[:90].sum()), float(loader.sampler.weights[90:].sum()))
        validation = balanced_loader(dataset, batch_size=4, shuffle=False, num_workers=0)
        self.assertEqual(torch.cat(list(validation)).tolist(), list(range(100)))


if __name__ == '__main__':
    unittest.main()
