import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn

from alexNet.finetune import fine_tune, validate_splits


class FineTuningTests(unittest.TestCase):
    def rows(self):
        return [dict(patient_id=f'{split}_{i}', split=split, label=label,
                     image=f'{split}_{i}.png')
                for split in ('train', 'val', 'test')
                for i, label in enumerate(('lesion_absent', 'lesion_present'))]

    def test_requires_both_classes_and_disjoint_validation(self):
        rows = self.rows()
        with self.assertRaisesRegex(ValueError, 'val must contain'):
            validate_splits([r for r in rows if r['image'] != 'val_1.png'])
        rows[-1]['patient_id'] = rows[0]['patient_id']
        with self.assertRaisesRegex(ValueError, 'Patient leakage'):
            validate_splits(rows)

    def test_updates_backbone_and_reloads_checkpoint_without_loading_test(self):
        torch.manual_seed(1)
        backbone = nn.Sequential(nn.Conv2d(3, 256, 1), nn.ReLU())
        before = backbone[0].weight.detach().clone()
        loaded = []

        def load(path):
            self.assertFalse(path.startswith('test'))
            loaded.append(path)
            return torch.full((3, 8, 8), 0.8 if '_1' in path else 0.2)

        with tempfile.TemporaryDirectory() as folder, \
             patch('alexNet.finetune.build_backbone', return_value=backbone), \
             patch('alexNet.finetune.load_image', side_effect=load):
            result, report = fine_tune(self.rows(), Path(folder), batch_size=2,
                                      epochs=2, patience=1, learning_rate=0.001)
            saved = torch.load(Path(folder) / 'alexnet_features.pt', weights_only=True)
            self.assertFalse(torch.equal(before, result[0].weight))
            self.assertTrue(all(not p.requires_grad for p in result.parameters()))
            self.assertFalse(result.training)
            self.assertEqual(report['best_epoch'], saved['epoch'])
            for name, value in result.state_dict().items():
                torch.testing.assert_close(value, saved['backbone'][name])
            self.assertEqual(set(loaded), {'train_0.png', 'train_1.png', 'val_0.png', 'val_1.png'})
