"""Utilities for reproducible experiment runs."""

import os
import random

import numpy as np
import torch


def seed_everything(seed):
    """Seed the local random-number generators used by the experiment."""
    if seed is None:
        return None

    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)

    return seed
