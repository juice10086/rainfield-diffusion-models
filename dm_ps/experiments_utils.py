import random
import torch
import numpy as np
import subprocess


def fix_seed(seed: int):
    """Fix seed torch, cuda, and numpy."""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True


def get_gpu_memory_consumption(device: str) -> int:
    """Get the current gpu usage.

    Code adapted from:
    https://discuss.pytorch.org/t/access-gpu-memory-usage-in-pytorch/3192/4

    Parameters
    ----------
    device : str
        name of the device, for example: 'cuda:0'

    Returns
    -------
    usage: int
        memory usage in MB.

    Notes
    -----
    - Normally this function should be called during the execution of a scripts but
      it is possible to call it at the end as GPU computation is cached.
    """
    # get device id
    try:
        device_id = int(device.replace("cuda:", ""))
    except ValueError:
        raise ValueError(f"Expected device to be of the form 'cuda:ID', got {device}")

    result = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,nounits,noheader"],
        encoding="utf-8",
    )
    # Convert lines into a dictionary
    gpu_memory = [int(x) for x in result.strip().split("\n")]
    gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))

    memory = gpu_memory_map.get(device_id, None)
    if memory is None:
        available_devices = [f"cuda:{i}" for i in gpu_memory_map]
        raise ValueError(
            "Unknown device name.\n"
            f"Expected device to be {available_devices}\n"
            f"got {device}"
        )

    return memory
