"""Shared single-genome padding/masking for BacPT inference.

Builds the standard 2-D [batch, seq] attention mask convention -- confirmed
(see inference/jacobian.py and the project plan) to differ from
inference.bacpt.prepare_bacpt_context's 3-D mask, and to be the one that
reproduces the original research notebooks' cached output for both the
Jacobian and genome-scaffolding analyses.
"""

import numpy as np
import torch
import torch.nn.functional as F

MAX_PROTEINS = 5000


def prepare_2d_context(normalized, device="cpu"):
    """Zero-pad to 5,000 positions and build a standard 2-D attention mask."""
    if not isinstance(normalized, np.ndarray):
        raise TypeError("BacPT input must be a NumPy array")
    if normalized.ndim != 2 or normalized.shape[1] != 480 or normalized.dtype != np.float32:
        raise ValueError("BacPT input must be float32 [proteins, 480]")
    count = min(normalized.shape[0], MAX_PROTEINS)
    inputs = torch.from_numpy(normalized[:count]).unsqueeze(0).to(device)
    inputs = F.pad(inputs, (0, 0, 0, MAX_PROTEINS - count))
    mask = torch.zeros((1, MAX_PROTEINS), dtype=torch.int, device=device)
    mask[:, :count] = 1
    return inputs, mask, count
