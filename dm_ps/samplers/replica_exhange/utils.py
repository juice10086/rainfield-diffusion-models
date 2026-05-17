import torch
from torch import Tensor
from dm_ps.models.edm_wrapper import EDMWrapper


def _get_adjacents(path_x: Tensor, sigmas: Tensor, parity_pt_iter: bool):
    # NOTE the procedure below for selecting the tuples is valid
    # regardless of the parity of ``len(sigmas)``

    if parity_pt_iter:
        path_x_prev = path_x[:-1:2]
        path_x_next = path_x[1::2]

        sigmas_prev = sigmas[:-1:2]
        sigmas_next = sigmas[1::2]
    else:
        path_x_prev = path_x[1:-1:2]
        path_x_next = path_x[2::2]

        sigmas_prev = sigmas[1:-1:2]
        sigmas_next = sigmas[2::2]

    return path_x_prev, path_x_next, sigmas_prev, sigmas_next


def _update_adjacents(path_x, path_x_prev, path_x_next, parity_pt_iter):
    # NOTE same, the procedure below for selecting the tuples is valid
    # regardless of the parity of ``len(sigmas)``

    if parity_pt_iter:
        path_x[:-1:2] = path_x_prev
        path_x[1::2] = path_x_next
    else:
        path_x[1:-1:2] = path_x_prev
        path_x[2::2] = path_x_next


def _get_denoised(path_x: Tensor, sigmas: Tensor, model: EDMWrapper):
    # NOTE as denoiser may not be compatible with shape of input (n_steps, batch, *dim)
    #  this func reshapes inputs as (n_steps * batch, *dim), apply denoiser and then reshapes back
    # NOTE but this may make algo slower as we don't broadcast over batch of samples with the same sigma
    #  refer to the example of Gaussian and Gaussian Mixtures where ``_get_denoiser`` can be removed

    # path_x has shape (n_steps, batch, *dim)
    # sigmas has shape (n_steps, 1, ...])
    n_steps, n_samples, *dim = path_x.shape

    input_x = path_x.reshape(-1, *dim)
    input_sigmas = sigmas.squeeze().repeat_interleave(n_samples)
    denoised = model.denoiser_fn(input_x, input_sigmas)

    return denoised.view(n_steps, n_samples, *dim)

    # XXX to speed-up runs on Gaussians
    # return model.denoiser_fn(path_x, sigmas)


def _init_path_normal(initial_noise, sigmas, model):
    # NOTE path_x has shape (n_steps, batch, *dim)
    x_t = initial_noise
    path_x = [x_t[None]]

    for sigma_t, sigma_t_prev in zip(sigmas[1:], sigmas[:-1]):
        x_t = x_t + (sigma_t**2 - sigma_t_prev**2).sqrt() * torch.randn_like(x_t)

        path_x.append(x_t[None])
    path_x = torch.cat(path_x)

    return path_x
