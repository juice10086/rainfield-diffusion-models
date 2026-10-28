"""Reconstruct z01 target radio maps with point-observation DPS likelihoods."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from training.radio_dataset import RadioDataset
from dm_ps.inv_prob.create_inv_prob_radio import create_inv_prob
from dm_ps.models.edm_wrapper import EDMRadio
from dm_ps.samplers.dps_ve import dps


def check_checkpoint(model, dataset, use_source):
    expected = dict(class_name='training.radio_dataset.RadioDataset',
                    source_frequency=dataset.source_frequency,
                    target_frequency=dataset.target_frequency,
                    terrain_scale=dataset.terrain_scale,
                    exclude_buildings=dataset.exclude_buildings,
                    use_source=use_source)
    for key, value in expected.items():
        if model.dataset_kwargs.get(key) != value:
            raise ValueError(f'Checkpoint dataset option {key} must be {value!r}; '
                             f'got {model.dataset_kwargs.get(key)!r}')


def reconstruct(sample, dataset, checkpoints, *, device='cpu', observations=None,
                obs_std=0.01, n_samples=4, seed=42, n_steps=50, gamma=0.1,
                sigma_min=0.002, sigma_max=80.0, rho=7.0, eta=1.0,
                normalize_grad=True):
    """Run both priors against exactly one shared likelihood and common noise.

    ``observations`` may be a mapping of measured positions [N,2], values [N],
    and optional obs_std (scalar or [N]). Truth is used only for simulated data
    and for reporting held-out errors, never as a denoiser condition.
    """
    if not checkpoints or set(checkpoints) - {'baseline', 'source'}:
        raise ValueError('Provide baseline and/or source radio checkpoints')
    if n_samples < 1:
        raise ValueError('n_samples must be positive')
    common = dict(device=device, valid_mask=sample['valid_mask'])
    if observations is None:
        problem = create_inv_prob(
            field=sample['target'], mask=sample['target_mask'], obs_std=obs_std,
            generator=torch.Generator(device=device).manual_seed(seed), **common)
    else:
        problem = create_inv_prob(
            positions=observations['positions'], observations=observations['values'],
            obs_std=observations.get('obs_std', obs_std),
            height=dataset.resolution, width=dataset.resolution, **common)
    evaluation_mask = sample['valid_mask'].to(device) & ~problem.H_func.support_mask()
    target = sample['target'].to(device)
    initial_noise = torch.randn(
        (n_samples, *dataset.image_shape), device=device,
        generator=torch.Generator(device=device).manual_seed(seed + 1))
    result = dict(
        sample_id=sample['sample_id'], source_frequency=dataset.source_frequency,
        target_frequency=dataset.target_frequency, height_id='z01', units='PNG/255',
        observation_kind='simulated' if observations is None else 'measured',
        positions=problem.H_func.positions.cpu(), observations=problem.obs.cpu(),
        obs_std=problem.obs_std.cpu(), evaluation_mask=evaluation_mask.cpu(),
        valid_mask=sample['valid_mask'].cpu(),
        target=target.cpu(), seed=seed, checkpoints=dict(checkpoints),
        sampling=dict(n_steps=n_steps, gamma=gamma, sigma_min=sigma_min,
                      sigma_max=sigma_max, rho=rho, eta=eta, normalize_grad=normalize_grad),
        paths={})
    for name, checkpoint in checkpoints.items():
        condition = sample['condition'][:3] if name == 'baseline' else sample['condition']
        model = EDMRadio(checkpoint, condition, device=device)
        check_checkpoint(model, dataset, use_source=(name == 'source'))
        samples = dps(
            initial_noise.clone(), problem, model, gamma=gamma,
            n_steps=n_steps, sigma_min=sigma_min, sigma_max=sigma_max,
            rho=rho, eta=eta, normalize_grad=normalize_grad, guidance='likelihood',
            generator=torch.Generator(device=device).manual_seed(seed + 2))
        mean = samples.mean(0)
        difference = (mean - target)[evaluation_mask]
        metrics = dict(
            n_evaluation_pixels=int(evaluation_mask.sum()),
            mae=float(difference.abs().mean()) if difference.numel() else None,
            rmse=float(difference.square().mean().sqrt()) if difference.numel() else None,
            observation_rmse=float((problem.H_func.H(mean) - problem.obs).square().mean().sqrt()))
        result['paths'][name] = dict(samples=samples.cpu(), mean=mean.cpu(),
                                     variance=samples.var(0, unbiased=False).cpu(),
                                     condition=condition.cpu(), metrics=metrics)
        del model
    return result


def save_comparison_plots(result, output_dir):
    """Save posterior-mean / truth / absolute-error panels for each prior.

    Use an Agg canvas so reconstruction also works on servers without a display.
    Plot scaling never modifies the tensors or metrics saved in the result.
    """
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib import cm

    def as_image(tensor):
        return tensor.detach().cpu().numpy()[0]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    target = as_image(result['target'])
    valid = as_image(result['valid_mask']).astype(bool)
    differences = {name: np.abs(as_image(path['mean']) - target)
                   for name, path in result['paths'].items()}
    # Both paths share the same error color scale; avoid a zero-width scale.
    error_max = max([1e-6] + [float(diff[valid].max())
                            for diff in differences.values() if valid.any()])
    field_cmap = cm.get_cmap('viridis').copy()
    error_cmap = cm.get_cmap('Reds').copy()
    field_cmap.set_bad('lightgray')
    error_cmap.set_bad('lightgray')
    saved = {}
    for name, path in result['paths'].items():
        fig = Figure(figsize=(18, 5))
        FigureCanvasAgg(fig)
        axes = fig.subplots(1, 3)
        images = (as_image(path['mean']), target, differences[name])
        titles = (f'Generated ({name}, mean of {len(path["samples"])} draws)',
                  'Ground Truth', 'Absolute Difference')
        for index, (ax, values, title) in enumerate(zip(axes, images, titles)):
            im = ax.imshow(np.ma.array(values, mask=~valid), origin='upper',
                           interpolation='nearest',
                           cmap=error_cmap if index == 2 else field_cmap,
                           vmin=0, vmax=error_max if index == 2 else 1)
            ax.set_title(title)
            ax.axis('off')
            fig.colorbar(im, ax=ax, label=result['units'],
                         extend='neither' if index == 2 else 'both')
        metrics = path['metrics']
        scores = ', '.join(f'{key.upper()}={metrics[key]:.4f}'
                           for key in ('mae', 'rmse') if metrics[key] is not None)
        fig.suptitle(f'{result["sample_id"]} | target {result["target_frequency"]}'
                     f' | held-out valid pixels: {scores or "N/A"}')
        fig.tight_layout()
        filename = output_dir / f'{result["sample_id"]}_{name}_comparison.png'
        fig.savefig(filename, dpi=150, bbox_inches='tight')
        fig.clear()
        saved[name] = str(filename)
    return saved


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', default='/home/czh/PPData5D-success')
    parser.add_argument('--manifest')
    parser.add_argument('--baseline-checkpoint')
    parser.add_argument('--source-checkpoint')
    parser.add_argument('--source-frequency', default='f00')
    parser.add_argument('--target-frequency', default='f03')
    parser.add_argument('--source-fraction', type=float, default=1.0)
    parser.add_argument('--target-fraction', type=float, default=0.01)
    parser.add_argument('--terrain-scale', type=float, default=1000.0)
    parser.add_argument('--include-buildings', action='store_true')
    parser.add_argument('--split', choices=['val', 'test'], default='test')
    parser.add_argument('--field-index', type=int, default=0)
    parser.add_argument('--observations', help='Measured NPZ: positions [N,2], values [N], optional obs_std')
    parser.add_argument('--obs-std', type=float, default=0.01, help='Noise std in PNG/255 units; positive')
    parser.add_argument('--split-seed', type=int, default=42)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n-samples', type=int, default=4)
    parser.add_argument('--n-steps', type=int, default=50)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--sigma-min', type=float, default=0.002)
    parser.add_argument('--sigma-max', type=float, default=80.0)
    parser.add_argument('--rho', type=float, default=7.0)
    parser.add_argument('--eta', type=float, default=1.0)
    parser.add_argument('--no-normalize-grad', action='store_true')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--save-dir', default='out_radio_points')
    args = parser.parse_args()
    checkpoints = {name: str(Path(path).resolve()) for name, path in
                   [('baseline', args.baseline_checkpoint), ('source', args.source_checkpoint)] if path}
    if not checkpoints:
        parser.error('Supply --baseline-checkpoint and/or --source-checkpoint')
    dataset = RadioDataset(
        path=args.data, manifest=args.manifest, split=args.split, seed=args.split_seed,
        source_frequency=args.source_frequency, target_frequency=args.target_frequency,
        use_source=True, source_fraction=args.source_fraction, target_fraction=args.target_fraction,
        terrain_scale=args.terrain_scale, exclude_buildings=not args.include_buildings)
    if not 0 <= args.field_index < len(dataset):
        parser.error(f'field-index must be between 0 and {len(dataset) - 1}')
    measured = None
    if args.observations:
        with np.load(args.observations, allow_pickle=False) as data:
            measured = {key: data[key] for key in data.files}
    result = reconstruct(
        dataset[args.field_index], dataset, checkpoints, device=args.device,
        observations=measured, obs_std=args.obs_std, n_samples=args.n_samples, seed=args.seed,
        n_steps=args.n_steps, gamma=args.gamma, sigma_min=args.sigma_min,
        sigma_max=args.sigma_max, rho=args.rho, eta=args.eta,
        normalize_grad=not args.no_normalize_grad)
    output = Path(args.save_dir)
    output.mkdir(parents=True, exist_ok=True)
    path = output / f'{result["sample_id"]}.pt'
    torch.save(result, path)
    plots = save_comparison_plots(result, output / 'comparisons')
    print(json.dumps(dict(output=str(path), comparisons=plots, sample_id=result['sample_id'],
                          metrics={name: value['metrics'] for name, value in result['paths'].items()}), indent=2))


if __name__ == '__main__':
    main()
