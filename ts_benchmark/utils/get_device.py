# -*- coding: utf-8 -*-
"""
Device utility functions for PyTorch operations.
This module has no dependencies to avoid circular imports.
"""
import torch


def get_device() -> torch.device:
    """
    Get the best available device for PyTorch operations.
    
    Checks for device availability in the following order:
    1. CUDA (NVIDIA GPUs)
    2. MPS (Apple Silicon GPUs)
    3. CPU (fallback)
    
    :return: torch.device object representing the best available device
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")
