"""Velocity-diffusion schedule and masked monthly projection.

These are the two numerical primitives the masked temporal model depends on:

* a 64-step cosine ``alpha`` schedule with the velocity parameterisation
  (``recover`` / ``consistent_noise``) used to rebuild the latent after a
  constraint update, and
* ``partial_project``, which rescales only the unobserved cells so that the
  valid-day mean intensity of a padded 31-day month reproduces a coarse 5x5
  monthly target without disturbing observed cells.

``partial_project`` deliberately weights by the number of *valid* days rather
than the 31-day padded length: for a short month the padded length would
overstate the required monthly energy.  ``bounded_partial_project`` adds a
non-negativity and physical upper-bound clamp while preserving observations.
"""

from __future__ import annotations

import numpy as np
import torch


def alphas(device):
    """Cumulative signal-retention schedule for 64 reverse steps."""
    t = torch.linspace(0, 1, 65, device=device)
    a = torch.cos((t + 0.008) / 1.008 * np.pi / 2) ** 2
    return (a[1:] / a[0]).clamp_min(1e-6)


def recover(z, v, a):
    """Recover the clean signal from a latent ``z`` and predicted velocity ``v``."""
    return a.sqrt() * z - (1 - a).sqrt() * v


def consistent_noise(z, clean, a):
    """Noise implied by ``clean`` at level ``a`` that reproduces the current ``z``."""
    return (z - a.sqrt() * clean) / (1 - a).sqrt().clamp_min(1e-8)


def diffusion_chunks(steps=64, chunk_size=8):
    """Inclusive ``(low, high)`` timestep bounds for reverse-diffusion chunks."""
    steps = int(steps)
    chunk_size = int(chunk_size)
    if steps <= 0 or chunk_size <= 0:
        raise ValueError("steps and chunk_size must be positive")
    return [(low, high) for high in range(steps - 1, -1, -chunk_size)
            for low in [max(0, high - chunk_size + 1)]]


def partial_project(values, target, weights, valid_days, observed):
    """Project missing cells onto a monthly target while retaining observations.

    ``values`` is ``[batch, days, lat, lon]``; ``observed`` is either
    ``[batch, days, 1, 1]`` (whole-field observation) or the full
    ``[batch, days, lat, lon]`` mask; ``valid_days`` is ``[batch, days]`` with
    one for calendar days inside the month and zero for padding.

    Coarse quantities such as ``observed_energy`` live on the 5x5 block grid
    and are marked with a ``10`` suffix once expanded back to 10x10; they are
    inserted as ``[batch, 1, 10, 10]`` so they broadcast over the day axis.
    """
    target = target.to(values.device)
    valid_days = valid_days.to(values.device)
    observed = observed.to(values.device)
    days = values.shape[1]
    valid_days = valid_days.reshape(valid_days.shape[0], days)
    valid = valid_days[:, :, None, None]
    # Callers may pass a mask with redundant singleton dimensions, e.g.
    # [batch, day, 1, 1, 1, 1]. Normalize it before broadcasting.
    observed = observed.reshape(observed.shape[0], days, -1)
    if observed.shape[-1] != 1:
        observed = observed.reshape(observed.shape[0], days, values.shape[-2], values.shape[-1])
    else:
        observed = observed.reshape(observed.shape[0], days, 1, 1)
    observed = observed.expand_as(values) * valid
    missing = (1.0 - observed) * valid
    values = values.clamp_min(0.0) * valid
    area = weights.to(values.device)
    full_weight = area.reshape(5, 2, 5, 2).sum((-1, -3))
    observed_energy = (values * observed * area).reshape(values.shape[0], days, 5, 2, 5, 2).sum((-1, -3))
    missing_energy = (values * missing * area).reshape(values.shape[0], days, 5, 2, 5, 2).sum((-1, -3)).sum(1)
    missing_area = missing.reshape(values.shape[0], days, 5, 2, 5, 2).sum(1)
    missing_area = (missing_area * area.reshape(5, 2, 5, 2)).sum((-1, -3))
    observed_energy = observed_energy.sum(1)
    # Energy is required over valid calendar days only; padded days carry no
    # energy and must not inflate the monthly total.
    target_energy = target * valid_days.sum(1)[:, None, None] * full_weight
    remaining = (target_energy - observed_energy).clamp_min(0.0)
    scale = remaining / missing_energy.clamp_min(1e-6)
    scale = scale.repeat_interleave(2, -2).repeat_interleave(2, -1)
    # A candidate with zero missing energy carries no information to scale, so
    # it is filled uniformly from the remaining energy per unit hidden area.
    fallback = (remaining / missing_area.clamp_min(1e-6)).repeat_interleave(2, -2).repeat_interleave(2, -1)
    zero_energy = (missing_energy <= 1e-6).repeat_interleave(2, -2).repeat_interleave(2, -1)
    projected_missing = values * scale[:, None]
    projected_missing = torch.where(zero_energy[:, None], fallback[:, None], projected_missing)
    # Iterative callers clamp the result between passes. A block whose observed
    # cells already exceed the requirement would otherwise have its hidden
    # cells driven to zero one pass at a time, because a zeroed hidden cell
    # leaves the observed energy above the target forever. Hidden cells whose
    # contribution is not needed are therefore held at their supplied value
    # instead of being blanked out. All terms below are [batch, 5, 5].
    surplus = (observed_energy >= target_energy) & (missing_area > 0)
    surplus10 = surplus.repeat_interleave(2, -2).repeat_interleave(2, -1)
    projected_missing = torch.where(surplus10[:, None, :, :], values, projected_missing)
    projected = values * observed + missing * projected_missing
    # If all cells in a coarse block are observed, the observations are kept.
    no_missing10 = (missing_area <= 0).repeat_interleave(2, -2).repeat_interleave(2, -1)
    projected = torch.where(no_missing10[:, None, :, :], values, projected)
    return projected * valid


def bounded_partial_project(values, target, weights, valid_days, observed, upper, iterations=4):
    """Project missing cells with an upper bound while preserving observed cells.

    ``partial_project`` already accounts for the valid-day count, so the raw
    coarse ``target`` is passed through unchanged across iterations.
    """
    result = values.clamp_min(0.0)
    for _ in range(iterations):
        result = partial_project(result, target, weights, valid_days, observed)
        result = torch.minimum(result, upper)
        result = torch.where(observed > 0, values, result)
    return result.clamp_min(0.0)
