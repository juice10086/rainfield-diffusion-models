import pickle
from typing import List

import torch
from torch import Tensor

import dnnlib
from torch_utils import distributed as dist


class EDMWrapper(torch.nn.Module):

    def __init__(self, network_pkl, device: str = "cpu", vae=None):
        super().__init__()

        self.device = device
        self.network_pkl = network_pkl

        # load network
        # explicitly cast network_pkl to str as edm match the file name with re
        with dnnlib.util.open_url(
            str(network_pkl), verbose=(dist.get_rank() == 0)
        ) as f:
            snapshot = pickle.load(f)
            net = snapshot['ema'].to(device)
            self.dataset_kwargs = snapshot.get('dataset_kwargs', {})

        # set network to inference model
        self.net = _prepare_model_to_eval(net, device)

        self.x_shape = (
            net.img_channels,
            net.img_resolution,
            net.img_resolution,
        )

        # in case we switch latent diffusion models
        self.vae = vae

    def denoiser_fn(self, x, sigma, ctx=None, use_cfg: bool = False):
        net = self.net
        n_samples = x.shape[0]

        # preprocess sigma to be passed in to the denoiser
        # handle the case where the denoiser is applied to different times steps
        if len(sigma.shape) == 0:
            sigmas = sigma * torch.ones((n_samples, 1), device=x.device)
        elif sigma.shape[0] != n_samples:
            raise ValueError(
                "parameter `sigma` must be a scaler or have the same size as `x`"
            )
        else:
            sigmas = sigma

        x_0s = net(x, sigmas, ctx)
        return x_0s

    def make_ctx(self, ctx: List[int]):
        class_label = torch.nn.functional.one_hot(
            torch.tensor(ctx), num_classes=self.cond_model.label_dim
        )
        return class_label

    def mk_sigmas_fn(
        self,
        n_steps: int,
        sigma_min=0.002,
        sigma_max=80.0,
        rho=7.0,
    ) -> torch.Tensor:
        sigmas = (
            sigma_max ** (1 / rho)
            + torch.linspace(1, 0, n_steps, device=self.device)
            * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
        ) ** rho
        return sigmas

    def decode(self, x):
        # VAE not supported yet
        return x

    def encode(self, x):
        # VAE not supported yet
        return x

    # NOTE: this is for compatibility with torch NN
    def forward(self, x, sigma, ctx=None):
        return self.denoiser_fn(x, sigma, ctx)

    def pred_noise(self, x, sigma, ctx=None):
        x_0 = self.denoiser_fn(x, sigma, ctx)
        return (x - x_0) / sigma


class EDMRadio(EDMWrapper):
    """Bind one scene's conditions; keep maps in differentiable PNG/255 units."""

    def __init__(self, network_pkl, condition, device='cpu'):
        super().__init__(network_pkl, device=device)
        condition = torch.as_tensor(condition, dtype=torch.float32, device=device).detach().clone()
        if condition.ndim == 3:
            condition = condition.unsqueeze(0)
        expected = (1, getattr(self.net, 'condition_channels', 0), *self.x_shape[-2:])
        if expected[1] not in (3, 5) or tuple(condition.shape) != expected:
            raise ValueError(f'Radio checkpoint requires spatial condition {expected}, got {tuple(condition.shape)}')
        if self.x_shape[0] != 1 or not torch.isfinite(condition).all():
            raise ValueError('Radio prior must output one channel and receive finite conditions')
        self.register_buffer('condition', condition)

    def denoiser_fn(self, x, sigma, ctx=None, use_cfg=False):
        if ctx is not None or use_cfg:
            raise ValueError('Radio conditions are bound to the scene; class labels/CFG are unsupported')
        sigmas = torch.as_tensor(sigma, device=x.device, dtype=x.dtype)
        if sigmas.numel() == 1:
            sigmas = sigmas.reshape(1).expand(x.shape[0])
        elif sigmas.shape[0] != x.shape[0]:
            raise ValueError('sigma must be scalar or have the same batch size as x')
        condition = self.condition.to(x).expand(x.shape[0], -1, -1, -1)
        return self.net(x, sigmas, condition=condition)


class EDMOpenMRG(EDMWrapper):

    def __init__(
        self,
        network_pkl: str,
        use_threshold: bool = True,
        threshold: float = 1e-2,
        device: str = "cpu",
    ):
        super().__init__(network_pkl, device, vae=None)

        self.x_shape = (1, 48, 36)  # channels, height, width

        # rain below threshold (in mm/hr) will be considered zero
        self.threshold = threshold
        self.use_threshold = use_threshold

    def encode(self, x: Tensor) -> Tensor:
        log_10 = torch.tensor(10.0, device=x.device).log()
        return torch.log1p(x) / log_10

    def decode(self, x: Tensor) -> Tensor:
        log_10 = torch.tensor(10.0, device=x.device).log()
        out = torch.expm1(x * log_10)

        if self.use_threshold:
            out = torch.where(out > self.threshold, out, 0.0)

        return out


class DecoderFreeEDMOpenMRG(EDMWrapper):

    def __init__(
        self,
        network_pkl: str,
        use_threshold: bool = True,
        threshold: float = 1e-2,
        device: str = "cpu",
    ):
        super().__init__(network_pkl, device, vae=None)

        self.x_shape = (1, 48, 36)  # channels, height, width
        self.cst = 8.830393756358223  # 0.999 quantile, see RainData class

        # rain below threshold (in mm/hr) will be considered zero
        self.threshold = threshold
        self.use_threshold = use_threshold

    def encode(self, x: Tensor) -> Tensor:
        return x / self.cst

    def decode(self, x: Tensor) -> Tensor:

        out = x * self.cst

        if self.use_threshold:
            out = torch.where(out > self.threshold, out, 0.0)

        return out


def _prepare_model_to_eval(model, device):
    model.to(device)
    model.eval()
    model.requires_grad_(False)
    return model
