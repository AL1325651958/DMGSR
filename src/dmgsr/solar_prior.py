"""Train-only GSR solar prior; extraterrestrial radiation is not clear-sky GSR."""
from __future__ import annotations

import calendar
import datetime
import math

import numpy as np
import torch

from .numerics import project


def extraterrestrial(keys, latitude):
    """FAO-56 daily extraterrestrial horizontal radiation, W/m2, 31 padded days.

    Padded slots for short months are exactly zero; they must not inherit the
    previous month's tail days.
    """
    phi = torch.deg2rad(torch.as_tensor(latitude, dtype=torch.float32))[:, None]
    output = []
    for key in keys:
        year, month = map(int, key.split("-"))
        days = calendar.monthrange(year, month)[1]
        values = []
        for day in range(1, days + 1):
            j = datetime.date(year, month, day).timetuple().tm_yday
            dr = 1 + 0.033 * math.cos(2 * math.pi * j / 365)
            delta = 0.409 * math.sin(2 * math.pi * j / 365 - 1.39)
            sunset = torch.acos((-torch.tan(phi) * math.tan(delta)).clamp(-1, 1))
            energy = 24 * 60 / math.pi * 0.0820 * dr * (
                sunset * torch.sin(phi) * math.sin(delta)
                + torch.cos(phi) * math.cos(delta) * torch.sin(sunset)
            )
            values.append((energy * 1e6 / 86400).clamp_min(0).expand(-1, 10))
        field = torch.zeros(31, len(latitude), 10)
        field[:days] = torch.stack(values)
        output.append(field)
    return torch.stack(output)


class SolarPrior:
    def __init__(self, latitude, template):
        self.latitude = np.asarray(latitude)
        self.template = template.cpu()

    @classmethod
    def fit(cls, latitude, x, mask, keys, training_indices):
        """Fit month-wise surface-to-extraterrestrial ratios on training rows only.

        Every calendar month must contribute at least one valid training day.
        A month with no valid day has no defined ratio, and substituting a
        fallback would silently blank out or distort a whole season, so this
        raises instead.
        """
        toa = extraterrestrial([keys[i] for i in training_indices], latitude)
        ratio = x[training_indices] / toa.clamp_min(20)
        template = []
        month = np.array([int(keys[i][5:]) for i in training_indices])
        for k in range(1, 13):
            ids = np.flatnonzero(month == k)
            m = mask[training_indices[ids]]
            weight = m.sum((0, 1))
            if float(weight.sum()) <= 0:
                raise ValueError(f"Training split has no valid day in calendar month {k}; cannot fit the solar prior")
            # Both the numerator and the normaliser must honour the valid-day
            # mask, otherwise padded days dilute the fitted seasonal pattern.
            template.append((ratio[ids] * m).sum((0, 1)) / weight)
        stacked = torch.stack(template)
        if not bool(torch.isfinite(stacked).all()):
            raise ValueError("Fitted solar prior contains non-finite values")
        return cls(latitude, stacked)

    def fields(self, target, keys, weights, mask):
        toa = extraterrestrial(keys, self.latitude).to(target.device)
        spatial = self.template[[int(k[5:]) - 1 for k in keys]].to(target.device)
        baseline = project(toa * spatial[:, None], target, weights, mask)
        # Floor avoids low-sun division; this is not a hard physical upper bound.
        factor = toa.clamp_min(20) / 400
        return baseline, factor
