"""Compute device selection: CUDA when available, logged every time.

Override with the environment variable URBAN_HEAT_DEVICE=cpu (or cuda).
"""

from __future__ import annotations

import logging
import os

log = logging.getLogger("device")


def get_device():
    """torch.device('cuda') if usable, else 'cpu'. Logs the choice and the GPU."""
    import torch

    want = os.environ.get("URBAN_HEAT_DEVICE", "").lower()
    if want == "cpu":
        log.info("using CPU (forced by URBAN_HEAT_DEVICE=cpu)")
        return torch.device("cpu")
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        log.info("using CUDA - %s, %.1f GB, %d SMs, compute %d.%d (torch %s, CUDA %s)",
                 p.name, p.total_memory / 1e9, p.multi_processor_count, p.major, p.minor,
                 torch.__version__, torch.version.cuda)
        return torch.device("cuda")
    if want == "cuda":
        raise RuntimeError("URBAN_HEAT_DEVICE=cuda but CUDA is not available to PyTorch")
    log.warning("using CPU - CUDA not available to PyTorch (torch %s, CUDA build %s)",
                torch.__version__, torch.version.cuda)
    return torch.device("cpu")
