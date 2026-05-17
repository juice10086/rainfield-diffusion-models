import torch
from torch import Tensor


def rmse(samples: Tensor, reference: Tensor):
    # samples   : B, 1, H, W
    # references:    1, H, W
    n_samples = samples.shape[0]
    reference = reference[None]

    diff = samples - reference
    out = (diff.view(n_samples, -1) ** 2).mean(dim=-1).sqrt()

    return out


def mae(samples: Tensor, reference: Tensor):
    # samples   : B, 1, H, W
    # references:    1, H, W
    n_samples = samples.shape[0]
    reference = reference[None]

    diff = samples - reference
    out = (diff.view(n_samples, -1)).abs().mean(dim=-1)

    return out


def pearson_corr(samples: Tensor, reference: Tensor):
    n_samples = samples.shape[0]

    # flatten last dims
    samples = samples.reshape(n_samples, -1)
    ref = reference.reshape(-1)

    # center
    f_samples_centered = samples - samples.mean(dim=-1)[:, None]
    f_ref_centered = ref - ref.mean(dim=-1)

    # compute sqrt(var)
    f_samples_std = f_samples_centered.norm(dim=-1)
    f_ref_std = f_ref_centered.norm(dim=-1)

    # compute correlation: cov(samples, ref) / sqrt(var(samples) * var(ref))
    pcc = torch.einsum(
        "bi,i,b->b", f_samples_centered, f_ref_centered, 1 / (f_samples_std * f_ref_std)
    )

    return pcc


def diff_cum_rain_rate(samples: Tensor, reference: Tensor):
    n_samples = samples.shape[0]

    # flatten last dims
    samples = samples.reshape(n_samples, -1)
    ref = reference.reshape(-1)

    cum_ref = ref.sum()
    cum_samples = samples.sum(dim=-1)

    return cum_samples - cum_ref


# %%
# basic checks for the metrics
if __name__ == "__main__":
    import scipy, torch

    N, H, W = 2, 3, 4

    samples = (
        torch.arange(N * 1 * H * W, dtype=torch.float32).flip(0).reshape(N, 1, H, W)
    )
    ref = torch.arange(1 * 1 * H * W, dtype=torch.float32).reshape(1, H, W)

    # these can be manually inspected
    out_rmse = rmse(samples, ref)
    out_mae = mae(samples, ref)
    out_cum = mae(samples, ref)
    # ---

    # check againt scipy
    ref = torch.randn((1, H, W))
    samples = ref * (
        torch.arange(N * 1 * H * W, dtype=torch.float32).flip(0).reshape(N, 1, H, W)
    )

    print(
        pearson_corr(samples, ref),
        [
            scipy.stats.pearsonr(samples[i].view(-1), ref.view(-1)).statistic
            for i in range(N)
        ],
    )

# %%
