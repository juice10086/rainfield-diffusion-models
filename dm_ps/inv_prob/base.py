"""Observation-model independent interface used by posterior samplers."""

from dataclasses import dataclass
from typing import Callable, Optional, Union

from torch import Tensor


@dataclass
class InverseProblem:
    obs: Tensor
    obs_img: Optional[Tensor]
    obs_std: Union[float, Tensor]
    H_func: object
    log_pot: Callable[[Tensor], Tensor]
    log_pot_batch: Callable[[Tensor], Tensor]
    x_ref: Optional[Tensor]
