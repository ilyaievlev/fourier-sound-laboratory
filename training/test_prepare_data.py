"""Synthetic metadata tests; no real test audio is opened."""
import unittest

from prepare_data import build_splits


class SplitTests(unittest.TestCase):
    def setUp(self):
        self.rows = [dict(filename=f'{fold}-{label}.wav', fold=str(fold),
                          target=str(label), category=f'class_{label}',
                          src_file=f'{fold}-{label}')
                     for fold in range(1, 6) for label in range(2)]

    def build(self):
        return build_splits(self.rows, expected_clips=10, expected_classes=2)

    def test_five_disjoint_rotating_folds(self):
        splits = self.build()
        all_test = []
        for k, split in splits.items():
            sets = [{r['filename'] for r in split[p]} for p in ['train', 'validation', 'test']]
            self.assertEqual([len(s) for s in sets], [6, 2, 2])
            self.assertFalse(sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
            self.assertEqual({r['fold'] for r in split['validation']}, {int(k) % 5 + 1})
            all_test.extend(sets[2])
        self.assertEqual(len(set(all_test)), 10)

    def test_source_leakage_rejected(self):
        self.rows[2]['src_file'] = self.rows[0]['src_file']
        with self.assertRaisesRegex(ValueError, 'multiple folds'):
            self.build()

    def test_duplicate_rejected(self):
        self.rows[1]['filename'] = self.rows[0]['filename']
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            self.build()

    def test_purge_preserves_test_and_removes_source_overlap(self):
        self.rows[2]['src_file'] = self.rows[0]['src_file']
        splits = build_splits(self.rows, 10, 2, 5, 'purge_training_and_validation')
        for split in splits.values():
            sources = [{r['src_file'] for r in split[p]} for p in ['train', 'validation', 'test']]
            self.assertFalse(sources[0] & sources[1] or sources[0] & sources[2] or sources[1] & sources[2])
            self.assertEqual(len(split['test']), 2)
        self.assertEqual(len(splits['1']['validation']), 1)
        self.assertEqual(len(splits['5']['train']), 5)

    def test_path_traversal_rejected(self):
        self.rows[0]['filename'] = '../escape.wav'
        with self.assertRaisesRegex(ValueError, 'Invalid'):
            self.build()


if __name__ == '__main__':
    unittest.main()
