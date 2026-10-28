import json
from pathlib import Path
import tempfile
import unittest
import warnings

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader

from training.radio_dataset import RadioDataset, build_manifest
from training.loss import EDMLoss
from training.networks import EDMPrecond


class RadioDataTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / 'png' / '06.DenseUrban').mkdir(parents=True)
        (self.root / 'npz').mkdir()
        for map_index in range(12):
            for tx in range(2):
                stem = f'T06C1D{map_index:04d}_n{tx:02d}'
                for frequency in ('f00', 'f03'):
                    values = np.arange(64, dtype=np.uint8).reshape(8, 8)
                    if frequency == 'f03':
                        values += 100
                    for height in ('z00', 'z01', 'z02'):
                        Image.fromarray(values if height == 'z01' else np.zeros_like(values)).save(
                            self.root / 'png' / '06.DenseUrban' / f'{stem}_{frequency}_ss_{height}.png')
                building = np.ones((3, 8, 8), dtype=np.uint8)
                building[1] = 0
                building[1, 0, 0] = 1
                np.savez(self.root / 'npz' / f'{stem}_bdtr.npz',
                         inBldg_zyx=building, terrain_yx=np.full((8, 8), 100.0))

    def dataset(self, **kwargs):
        return RadioDataset(path=self.root, **kwargs)

    def test_z01_pairs_and_map_split(self):
        info = build_manifest(self.root)
        self.assertEqual(info['counts']['paired'], 24)
        split_maps = {}
        for r in info['records']:
            self.assertTrue(r['source'].endswith('_z01.png'))
            self.assertTrue(r['target'].endswith('_z01.png'))
            self.assertTrue(Path(r['source']).name.startswith(r['sample_id']))
            split_maps.setdefault(r['split'], set()).add(r['map_id'])
        self.assertFalse(split_maps['train'] & split_maps['test'])
        self.assertFalse(split_maps['train'] & split_maps['val'])
        self.assertFalse(split_maps['test'] & split_maps['val'])
        sample = self.dataset(split='test')[0]
        self.assertEqual(int(sample['valid_mask'].sum()), 63)
        self.assertAlmostEqual(sample['target'][0, 0, 0].item(), 100 / 255)
        self.assertTrue(torch.allclose(sample['terrain'], torch.full((1, 8, 8), 0.1)))

    def test_missing_pairs_reported_and_manifest_reusable(self):
        (self.root / 'png' / '06.DenseUrban' / 'T06C1D0000_n00_f00_ss_z01.png').unlink()
        (self.root / 'npz' / 'T06C1D0000_n01_bdtr.npz').unlink()
        info = build_manifest(self.root)
        self.assertEqual(info['counts']['missing_bands'], 1)
        self.assertEqual(info['counts']['missing_environment'], 1)
        manifest = self.root / 'manifest.json'
        manifest.write_text(json.dumps(info))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            self.assertEqual(len(self.dataset(split='all', manifest=manifest)), 22)
        with self.assertRaises(ValueError):
            self.dataset(manifest=manifest, source_frequency='f02')

    def test_masks_conditions_and_paired_observations(self):
        cross = self.dataset(split='test', source_fraction=0.5, target_fraction=0.1)
        baseline = self.dataset(split='test', use_source=False, source_fraction=0, target_fraction=0.1)
        a, b = cross[0], baseline[0]
        self.assertEqual(a['condition'].shape, (5, 8, 8))
        self.assertEqual(b['condition'].shape, (3, 8, 8))
        self.assertTrue(torch.equal(a['target_mask'], b['target_mask']))
        self.assertTrue(torch.equal(a['target_observations'], b['target_observations']))
        self.assertTrue(torch.equal(a['condition'][:3], b['condition']))
        self.assertTrue(torch.equal(a['source_mask'], cross[0]['source_mask']))
        self.assertEqual(a['target_mask'].sum(), 6)
        self.assertEqual(a['source_mask'].sum(), 32)
        self.assertFalse((a['target_mask'] & ~a['valid_mask']).any())
        self.assertFalse((a['evaluation_mask'] & a['target_mask']).any())
        self.assertTrue((a['source_observations'][~a['source_mask']] == 0).all())
        self.assertTrue(torch.equal(a['target_observations'], a['target'] * a['target_mask']))
        # Changing the target truth cannot change the prior's spatial condition.
        target_path = self.root / cross.records[0]['target']
        Image.fromarray(np.zeros((8, 8), dtype=np.uint8)).save(target_path)
        self.assertTrue(torch.equal(a['condition'], cross[0]['condition']))

    def test_aligned_flips(self):
        plain = self.dataset(source_fraction=1)
        flipped = self.dataset(source_fraction=1, p_x_flip=1, p_y_flip=1)
        a, b = plain[0], flipped[0]
        for key in ('target', 'condition', 'valid_mask', 'target_mask', 'source_mask', 'evaluation_mask'):
            self.assertTrue(torch.equal(a[key].flip(-1, -2), b[key]), key)

    def test_conditioned_training_and_rain_compatibility(self):
        torch.set_num_threads(1)
        batch = next(iter(DataLoader(self.dataset(), batch_size=2)))
        model = EDMPrecond(img_resolution=8, img_channels=1, condition_channels=5,
                           model_type='SongUNet', model_channels=8, channel_mult=[1, 2],
                           num_blocks=1, attn_resolutions=[])
        loss = EDMLoss()(model, batch['target'], condition=batch['condition'], valid_mask=batch['valid_mask'])
        self.assertEqual(tuple(loss.shape), (2,))
        loss.mean().backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
        with self.assertRaises(ValueError):
            model(batch['target'], torch.ones(2))
        rain = EDMPrecond(img_resolution=8, img_channels=1, model_type='SongUNet',
                          model_channels=8, channel_mult=[1], num_blocks=1, attn_resolutions=[])
        self.assertEqual(EDMLoss()(rain, batch['target']).shape, batch['target'].shape)

    def test_loss_excludes_buildings_and_noises_target_only(self):
        class Probe:
            def __call__(self, x, sigma, labels, condition=None, **kwargs):
                self.condition = condition
                return torch.zeros_like(x)
        net = Probe()
        target = torch.zeros(1, 1, 8, 8)
        target[0, 0, 0, 0] = 1000
        mask = torch.ones_like(target, dtype=torch.bool)
        mask[0, 0, 0, 0] = False
        condition = torch.rand(1, 5, 8, 8)
        loss = EDMLoss()(net, target, condition=condition, valid_mask=mask)
        self.assertEqual(loss.item(), 0)
        self.assertTrue(torch.equal(net.condition, condition))


if __name__ == '__main__':
    unittest.main()
