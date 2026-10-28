import pickle
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from dm_ps.inv_prob.create_inv_prob_radio import PointSamplingOperator, create_inv_prob
from dm_ps.models.edm_wrapper import EDMRadio
from dm_ps.samplers import AVAILABLE_SAMPLERS
from dm_ps.samplers.dps_ve import dps
from experiments.radio_point_runner import reconstruct
from training.networks import EDMPrecond


class ScalingPrior:
    def mk_sigmas_fn(self, **kwargs):
        return torch.tensor([0.1, 1.0])

    def denoiser_fn(self, x, sigma):
        return 2 * x

    def decode(self, x):
        return x


class RadioInverseProblemTest(unittest.TestCase):
    def test_integer_fractional_and_boundary_sampling(self):
        operator = PointSamplingOperator([[0, 0], [1.5, 1.5], [3, 2]], height=3, width=4)
        x = torch.arange(12, dtype=torch.float32).reshape(1, 3, 4).requires_grad_()
        torch.testing.assert_close(operator.H(x), torch.tensor([0., 7.5, 11.]))
        operator.H(x)[1].backward()
        expected = torch.zeros_like(x)
        expected[0, 1:3, 1:3] = 0.25
        torch.testing.assert_close(x.grad, expected)
        batch = x.detach().expand(2, 3, 1, 3, 4)
        self.assertEqual(operator.H(batch).shape, (2, 3, 3))
        negative = -torch.ones(1, 3, 4)
        torch.testing.assert_close(operator.H(negative), -torch.ones(3))

    def test_mask_order_and_single_pixel_grid(self):
        mask = torch.tensor([[[0, 1, 0], [1, 0, 1]]], dtype=torch.bool)
        operator = PointSamplingOperator.from_mask(mask)
        torch.testing.assert_close(operator.positions, torch.tensor([[1., 0.], [0., 1.], [2., 1.]]))
        torch.testing.assert_close(operator.support_mask(), mask)
        single = PointSamplingOperator([[0, 0]], 1, 1)
        torch.testing.assert_close(single.H(torch.tensor([[[0.6]]])), torch.tensor([0.6]))

    def test_gaussian_likelihood_and_analytic_gradient(self):
        problem = create_inv_prob(positions=[[0, 0], [1, 0]], observations=[0.8, 0.2],
                                  obs_std=[0.1, 0.2], height=2, width=2)
        x = torch.full((2, 1, 2, 2), 0.6, requires_grad=True)
        torch.testing.assert_close(problem.log_pot_batch(x), torch.full((2,), -4.))
        grad, = torch.autograd.grad(problem.log_pot(x), x)
        expected = torch.zeros_like(x)
        expected[:, 0, 0, 0], expected[:, 0, 0, 1] = 20, -10
        torch.testing.assert_close(grad, expected)
        self.assertIsNone(problem.x_ref)

    def test_subpixel_likelihood_gradcheck(self):
        problem = create_inv_prob(positions=[[0.2, 0.7]], observations=[0.8],
                                  obs_std=0.15, height=2, width=2, dtype=torch.float64)
        state = torch.randn(2, 1, 2, 2, dtype=torch.float64, requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(lambda z: problem.log_pot(2 * z), (state,)))

    def test_measured_values_are_not_regenerated_or_clipped(self):
        values = torch.tensor([-0.2, 1.2])
        problem = create_inv_prob(field=torch.full((1, 2, 2), 100.), positions=[[0, 0], [1, 1]],
                                  observations=values, obs_std=0.01)
        values.zero_()
        torch.testing.assert_close(problem.obs, torch.tensor([-0.2, 1.2]))
        a = create_inv_prob(field=torch.ones(1, 2, 2), mask=torch.ones(1, 2, 2),
                            generator=torch.Generator().manual_seed(123))
        b = create_inv_prob(field=torch.ones(1, 2, 2), mask=torch.ones(1, 2, 2),
                            generator=torch.Generator().manual_seed(123))
        torch.testing.assert_close(a.obs, b.obs)

    def test_invalid_inputs_fail_explicitly(self):
        for positions in ([], [[-1, 0]], [[2, 0]], [[float('nan'), 0]]):
            with self.assertRaises(ValueError):
                PointSamplingOperator(positions, 2, 2)
        for std in (0, -0.1, float('nan'), [0.1, 0.2]):
            with self.assertRaises(ValueError):
                create_inv_prob(positions=[[0, 0]], observations=[0.8], obs_std=std, height=2, width=2)
        with self.assertRaises(ValueError):
            create_inv_prob(field=torch.zeros(1, 2, 2), mask=torch.zeros(1, 2, 2))
        valid = torch.ones(1, 2, 2, dtype=torch.bool)
        valid[0, 1, 1] = False
        with self.assertRaises(ValueError):
            create_inv_prob(positions=[[0.5, 0.5]], observations=[0.8], height=2, width=2, valid_mask=valid)

    def test_dps_backpropagates_likelihood_through_denoiser(self):
        problem = create_inv_prob(positions=[[0, 0]], observations=[0.8], obs_std=0.2, height=2, width=2)
        # D(z)=2z: initial gradient is 2*0.8/0.2^2=40. One update
        # with gamma=.001 gives z=.04 and final denoising gives .08.
        with torch.no_grad():
            samples = AVAILABLE_SAMPLERS['dps'](
                torch.zeros(1, 1, 2, 2), problem, ScalingPrior(), n_steps=2,
                gamma=0.001, eta=0, sigma_min=0.1, sigma_max=1)
        expected = torch.zeros_like(samples)
        expected[0, 0, 0, 0] = 0.08
        torch.testing.assert_close(samples, expected)
        duplicate = dps(torch.zeros(2, 1, 2, 2), problem, ScalingPrior(), n_steps=2,
                        gamma=0.001, eta=0, sigma_min=0.1, sigma_max=1)
        torch.testing.assert_close(duplicate, expected.expand_as(duplicate))

    def test_zero_gradient_normalization_is_finite(self):
        problem = create_inv_prob(positions=[[0, 0]], observations=[0.0], height=2, width=2)
        result = dps(torch.zeros(1, 1, 2, 2), problem, ScalingPrior(), n_steps=2,
                     normalize_grad=True, eta=0)
        torch.testing.assert_close(result, torch.zeros_like(result))

    def test_paired_runner_and_conditioned_checkpoint(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            dataset = SimpleNamespace(source_frequency='f00', target_frequency='f03',
                                      terrain_scale=1000., exclude_buildings=True,
                                      resolution=8, image_shape=[1, 8, 8])
            checkpoints = {}
            for name, channels in [('baseline', 3), ('source', 5)]:
                net = EDMPrecond(img_resolution=8, img_channels=1, condition_channels=channels,
                                 model_type='SongUNet', model_channels=8, channel_mult=[1],
                                 num_blocks=1, attn_resolutions=[])
                with torch.no_grad():
                    for key, block in net.model.dec.items():
                        if key.endswith('_aux_conv'):
                            block.weight.zero_()
                            block.bias.zero_()
                metadata = dict(class_name='training.radio_dataset.RadioDataset',
                                source_frequency='f00', target_frequency='f03',
                                terrain_scale=1000., exclude_buildings=True, use_source=(name == 'source'))
                path = Path(directory) / f'{name}.pkl'
                with path.open('wb') as f:
                    pickle.dump(dict(ema=net, dataset_kwargs=metadata), f)
                checkpoints[name] = str(path)
            valid = torch.ones(1, 8, 8, dtype=torch.bool)
            target_mask = torch.zeros_like(valid)
            target_mask[0, 1, 2] = True
            sample = dict(sample_id='test', condition=torch.rand(5, 8, 8),
                          valid_mask=valid, target_mask=target_mask, target=torch.full((1, 8, 8), 0.6))
            seen = []
            def record_problem(initial, problem, model, **kwargs):
                seen.append(problem)
                return dps(initial, problem, model, **kwargs)
            with patch('experiments.radio_point_runner.dps', side_effect=record_problem):
                result = reconstruct(sample, dataset, checkpoints, n_samples=2, n_steps=3,
                                     sigma_max=1, sigma_min=0.1, gamma=0.01)
            self.assertIs(seen[0], seen[1])
            # Zero-initialized final UNet convolutions give identical denoisers;
            # common initial and transition noise must therefore give identical samples.
            torch.testing.assert_close(result['paths']['baseline']['samples'], result['paths']['source']['samples'])
            self.assertEqual(result['paths']['baseline']['metrics']['n_evaluation_pixels'], 63)
            self.assertFalse(result['evaluation_mask'][0, 1, 2])
            model = EDMRadio(checkpoints['source'], sample['condition'])
            state = torch.randn(2, 1, 8, 8, requires_grad=True)
            output = model.denoiser_fn(state, torch.tensor(0.1))
            output.sum().backward()
            self.assertTrue(torch.isfinite(state.grad).all())
            self.assertTrue(all(p.grad is None for p in model.parameters()))
            with self.assertRaises(ValueError):
                EDMRadio(checkpoints['source'], sample['condition'][:3])
            # A measured run must use the supplied values without adding noise.
            measured = dict(positions=[[2.25, 1.5]], values=[0.8], obs_std=0.02)
            measured_result = reconstruct(sample, dataset, {'source': checkpoints['source']},
                                          observations=measured, n_samples=1, n_steps=2,
                                          sigma_max=1, sigma_min=0.1, gamma=0.01)
            torch.testing.assert_close(measured_result['observations'], torch.tensor([0.8]))
            self.assertEqual(measured_result['paths']['source']['metrics']['n_evaluation_pixels'], 60)


if __name__ == '__main__':
    unittest.main()
