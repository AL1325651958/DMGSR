"""Area-weighted grid operators shared by every DMGSR stage.

The domain is a 10x10 latitude/longitude grid reduced to 5x5 area-weighted
2x2 blocks.  ``coarse`` performs that reduction and ``expand`` the inverse
nearest-neighbour expansion; ``project`` rescales a daily field so that its
valid-day average reproduces a prescribed coarse monthly target.
"""

from __future__ import annotations

import torch


def coarse(x, weights):
    """Area-weighted 2x2 block average of the last two dimensions (10x10 -> 5x5)."""
    return (x * weights).reshape(*x.shape[:-2], 5, 2, 5, 2).sum((-1, -3)) / weights.reshape(5, 2, 5, 2).sum((-1, -3))


def expand(x):
    """Nearest-neighbour 2x expansion of the last two dimensions (5x5 -> 10x10)."""
    return x.repeat_interleave(2, -2).repeat_interleave(2, -1)


def project(x, target, weights, mask):
    """Rescale ``x`` multiplicatively so its coarse mean matches ``target``.

    ``x`` is ``[..., days, 10, 10]``, ``target`` is ``[..., 5, 5]`` and
    ``mask`` is broadcastable to ``x`` marking valid padded-free days.  Blocks
    with no valid day fall back to the target itself, and every output value is
    non-negative.
    """
    x = x.clamp_min(0) * mask
    avg = coarse(x.sum(-3) / mask.sum(-3), weights)
    scaled = x * expand(target / avg.clamp_min(1e-8)).unsqueeze(-3)
    fallback = expand(target).unsqueeze(-3) * mask
    return torch.where(expand(avg < 1e-8).unsqueeze(-3), fallback, scaled)
