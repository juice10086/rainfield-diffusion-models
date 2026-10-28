"""Paired PPData5D maps at z01 (30 m), loaded lazily from PNG/NPZ."""

import hashlib
import json
from pathlib import Path
import re
import warnings

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


RADIO_NAME = re.compile(
    r'(?P<map>T\d{2}C\d+D\d+)_(?P<tx>n\d+)_'
    r'(?P<frequency>f\d{2})_ss_z01\.png$'
)


def build_manifest(path, source_frequency='f00', target_frequency='f03',
                   seed=42, val_fraction=0.1, test_fraction=0.1):
    """Pair only matching transmitters; split entire maps before filtering pairs."""
    if source_frequency == target_frequency:
        raise ValueError('Source and target frequencies must differ')
    if any(not re.fullmatch(r'f\d{2}', f) for f in (source_frequency, target_frequency)):
        raise ValueError('Frequency IDs must have the form f00, f01, ...')
    if not (0 <= val_fraction < 1 and 0 <= test_fraction < 1
            and val_fraction + test_fraction < 1):
        raise ValueError('Split fractions must be nonnegative and sum to less than 1')
    root = Path(path).resolve()
    if not (root / 'png').is_dir() or not (root / 'npz').is_dir():
        raise FileNotFoundError(f'Expected png/ and npz/ under {root}')
    groups = {}
    for filename in sorted((root / 'png').rglob('*_z01.png')):
        match = RADIO_NAME.fullmatch(filename.name)
        if match is None:
            raise ValueError(f'Unrecognized z01 filename: {filename}')
        map_id, tx, frequency = match.group('map', 'tx', 'frequency')
        bands = groups.setdefault((map_id, tx), {})
        if frequency in bands:
            raise ValueError(f'Duplicate map/transmitter/frequency: {filename}')
        bands[frequency] = str(filename.relative_to(root))
    if not groups:
        raise ValueError(f'No *_ss_z01.png maps found under {root / "png"}')
    maps = sorted({key[0] for key in groups})
    np.random.RandomState(seed).shuffle(maps)
    n_test, n_val = int(len(maps) * test_fraction), int(len(maps) * val_fraction)
    splits = {m: ('test' if i < n_test else 'val' if i < n_test + n_val else 'train')
              for i, m in enumerate(maps)}
    records, missing_bands, missing_environment = [], 0, 0
    for (map_id, tx), bands in sorted(groups.items()):
        if source_frequency not in bands or target_frequency not in bands:
            missing_bands += 1
            continue
        environment = Path('npz') / f'{map_id}_{tx}_bdtr.npz'
        if not (root / environment).is_file():
            missing_environment += 1
            continue
        records.append(dict(sample_id=f'{map_id}_{tx}', map_id=map_id, transmitter_id=tx,
                            source=bands[source_frequency], target=bands[target_frequency],
                            environment=str(environment), split=splits[map_id]))
    if not records:
        raise ValueError('No complete source/target/environment pairs at z01')
    return dict(version=1, root=str(root), height_id='z01', height_m=30,
                source_frequency=source_frequency, target_frequency=target_frequency,
                seed=seed, val_fraction=val_fraction, test_fraction=test_fraction,
                counts=dict(candidates=len(groups), paired=len(records),
                            missing_bands=missing_bands, missing_environment=missing_environment),
                records=records)


def sample_observation_mask(valid_mask, fraction, generator):
    """Select an exact fraction (rounded) of valid pixels, without replacement."""
    if not 0 <= fraction <= 1:
        raise ValueError('Observation fraction must be in [0, 1]')
    indices = torch.nonzero(valid_mask.flatten(), as_tuple=False).flatten()
    count = int(round(len(indices) * fraction))
    mask = torch.zeros_like(valid_mask, dtype=torch.bool).flatten()
    chosen = torch.randperm(len(indices), generator=generator)[:count]
    mask[indices[chosen]] = True
    return mask.reshape(valid_mask.shape)


class RadioDataset(Dataset):
    """Target X_t and C[, Y_s, M_s]; target observations never enter C.

    PNG intensities are divided by 255, without assuming a dB calibration.
    NPZ z-axis index 1 is the only building layer read into model inputs.
    Both paths use the same complete pairs and map-level splits.
    """

    def __init__(self, path='/home/czh/PPData5D-success', manifest=None,
                 source_frequency='f00', target_frequency='f03', split='train',
                 use_source=True, source_fraction=1.0, target_fraction=0.01,
                 seed=42, val_fraction=0.1, test_fraction=0.1,
                 terrain_scale=1000.0, exclude_buildings=True,
                 p_x_flip=0.0, p_y_flip=0.0):
        if split not in ('train', 'val', 'test', 'all'):
            raise ValueError('split must be train, val, test or all')
        if not all(0 <= f <= 1 for f in (source_fraction, target_fraction, p_x_flip, p_y_flip)):
            raise ValueError('Fractions and flip probabilities must be in [0, 1]')
        if not np.isfinite(terrain_scale) or terrain_scale <= 0:
            raise ValueError('terrain_scale must be finite and positive')
        self.root = Path(path).resolve()
        if manifest is None:
            info = build_manifest(path, source_frequency, target_frequency, seed,
                                  val_fraction, test_fraction)
        else:
            with open(manifest, encoding='utf-8') as stream:
                info = json.load(stream)
            if (info['version'], info['height_id'], info['source_frequency'], info['target_frequency']) != (
                    1, 'z01', source_frequency, target_frequency):
                raise ValueError('Manifest version, height or frequencies do not match dataset options')
        self.manifest = info
        if info['counts']['missing_bands'] or info['counts']['missing_environment']:
            warnings.warn(f"Skipped incomplete z01 samples: {info['counts']}", stacklevel=2)
        self.records = [r for r in info['records'] if split == 'all' or r['split'] == split]
        if not self.records:
            raise ValueError(f'No paired samples in split {split!r}')
        self.split, self.seed = split, seed
        self.use_source, self.source_fraction = use_source, source_fraction
        self.target_fraction, self.terrain_scale = target_fraction, terrain_scale
        self.exclude_buildings = exclude_buildings
        self.p_x_flip, self.p_y_flip = p_x_flip, p_y_flip
        self.source_frequency, self.target_frequency = source_frequency, target_frequency
        self.num_channels, self.label_dim, self.has_labels = 1, 0, False
        self.condition_channels = 5 if use_source else 3
        first = self._read_png(self.records[0]['target'])
        self.image_shape = list(first.shape)
        if first.shape[-2] != first.shape[-1]:
            raise ValueError('Radio training requires square maps')
        self.resolution = first.shape[-1]
        self.name = f'ppdata5d-z01-{source_frequency}-{target_frequency}'

    def __len__(self):
        return len(self.records)

    def _read_png(self, relative_path):
        with Image.open(self.root / relative_path) as image:
            array = np.array(image)
        if array.ndim != 2 or array.dtype != np.uint8:
            raise ValueError(f'Expected uint8 grayscale radio map: {relative_path}')
        return torch.from_numpy(array.astype(np.float32) / 255.0).unsqueeze(0)

    def _generator(self, sample_id, purpose):
        digest = hashlib.sha256(f'{self.seed}:{sample_id}:{purpose}'.encode()).digest()
        return torch.Generator().manual_seed(int.from_bytes(digest[:8], 'little') % (2 ** 63))

    def __getitem__(self, index):
        record = self.records[index]
        target = self._read_png(record['target'])
        source = self._read_png(record['source'])
        with np.load(self.root / record['environment'], allow_pickle=False) as env:
            building_3d = env['inBldg_zyx']
            terrain = env['terrain_yx'].astype(np.float32)
            if building_3d.shape != (3, *target.shape[-2:]) or terrain.shape != target.shape[-2:]:
                raise ValueError(f'Unexpected environment shape: {record["environment"]}')
            building = building_3d[1].copy()
        if list(target.shape) != self.image_shape or source.shape != target.shape:
            raise ValueError(f'Unaligned radio maps: {record["sample_id"]}')
        if not np.isfinite(terrain).all() or not np.isin(building, [0, 1]).all():
            raise ValueError(f'Invalid environment values: {record["environment"]}')
        building = torch.from_numpy(building.astype(np.float32)).unsqueeze(0)
        terrain = torch.from_numpy(terrain / self.terrain_scale).unsqueeze(0)
        valid = (building == 0) if self.exclude_buildings else torch.ones_like(building, dtype=torch.bool)
        if not valid.any():
            raise ValueError(f'No valid pixels: {record["sample_id"]}')
        # Target RNG is independent of the source fraction and the chosen path.
        target_mask = sample_observation_mask(valid, self.target_fraction,
                                             self._generator(record['sample_id'], 'target'))
        source_rng = None if self.split == 'train' else self._generator(record['sample_id'], 'source')
        source_mask = sample_observation_mask(valid, self.source_fraction, source_rng)
        tensors = dict(target=target, building=building, terrain=terrain, valid_mask=valid,
                       source_observations=source * source_mask, source_mask=source_mask,
                       target_observations=target * target_mask, target_mask=target_mask)
        if self.split == 'train':
            dims = []
            if torch.rand(()) < self.p_y_flip:
                dims.append(-2)
            if torch.rand(()) < self.p_x_flip:
                dims.append(-1)
            if dims:
                tensors = {k: torch.flip(v, dims) for k, v in tensors.items()}
        condition = [tensors['building'], tensors['terrain'], tensors['valid_mask'].float()]
        if self.use_source:
            condition += [tensors['source_observations'], tensors['source_mask'].float()]
        tensors['condition'] = torch.cat(condition, dim=0)
        tensors['evaluation_mask'] = tensors['valid_mask'] & ~tensors['target_mask']
        tensors.update(sample_id=record['sample_id'], map_id=record['map_id'],
                       transmitter_id=record['transmitter_id'])
        return tensors
