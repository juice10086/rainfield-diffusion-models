import math
import numpy as np

import torch
from torch.utils.data import TensorDataset


class RainDataset(TensorDataset):

    def __init__(
        self,
        path: str,
        dtype=torch.float32,
        p_x_flip: float = 0.0,
        p_y_flip: float = 0.0,
        divide_by_quantile: bool = False,
        **kwargs
    ):
        # must be a npy file
        self.path_data = path

        self.p_x_flip = p_x_flip
        self.p_y_flip = p_y_flip
        self.divide_by_quantile = divide_by_quantile

        # load data
        data = torch.from_numpy(np.load(path))
        data = data[:, None, :, :]  # number of channels equals 1
        data = data.to(dtype)

        # XXX saved train data stats
        self.cst = 8.830393756358223  # quantile 0.999 of train
        self.std = 0.07938836772571102  # std after dividing by cst
        # ----

        if self.divide_by_quantile:
            data = data / self.cst

        self.data = data

        super().__init__(data)

        # for compatibility with EDM
        self.num_channels = 1

        self._name = "rain"
        self._raw_shape = data.shape
        self._use_label = False
        self.label_dim = 0
        # ---

    # for compatibility with EDM
    @property
    def name(self):
        return self._name

    @property
    def image_shape(self):
        return list(self._raw_shape[1:])

    @property
    def resolution(self):
        # take sqrt(H x W) as they might differ
        spacial_res = self.data.shape[-2:]
        spacial_res = int(math.sqrt(math.prod(spacial_res)))

        # NOTE set it to an even number to not break NN architectures
        spacial_res = spacial_res - spacial_res % 2

        return spacial_res

    # flip image
    def __getitem__(self, index):
        sample = super().__getitem__(index)[0]

        if torch.rand(1).item() < self.p_y_flip:
            sample = torch.flip(sample, dims=[-2])

        if torch.rand(1).item() < self.p_x_flip:
            sample = torch.flip(sample, dims=[-1])

        return (sample,)
