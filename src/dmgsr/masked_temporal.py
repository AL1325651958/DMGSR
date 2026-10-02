"""Masked conditional temporal diffusion with daily weather features."""

from __future__ import annotations

import json
import math
import calendar
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .daily_features import FEATURES, open_daily_feature_bundle, write_feature_manifest
from .diffusion_ops import (
    alphas,
    bounded_partial_project,
    consistent_noise,
    diffusion_chunks,
    recover,
)
from .numerics import coarse, expand
from .solar_prior import SolarPrior


OUT = Path("outputs/masked_v6")
DATA_ROOT = Path("data/nasa_power/pilot_east_asia_features")


class MaskedTemporalVelocityNet(nn.Module):
    """Temporal denoiser with explicit daily features, values and observation mask."""

    def __init__(self, width: int = 24, daily_feature_count: int = 3):
        super().__init__()
        # z, four monthly channels, daily weather, observed residual, observed mask,
        # valid-day mask, diffusion time, and within-month sin/cos coordinates.
        input_channels = 1 + 4 + daily_feature_count + 1 + 1 + 1 + 1 + 2
        self.stem = nn.Conv3d(input_channels, width, 1)
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv3d(width, width, (1, 5, 5), padding=(0, 2, 2)),
                    nn.SiLU(),
                    nn.Conv3d(width, width, (3, 1, 1), padding=(1, 0, 0)),
                    nn.SiLU(),
                )
                for _ in range(4)
            ]
        )
        self.output = nn.Conv3d(width, 1, 1)

    def forward(self, z, monthly, daily_features, observed_values, observed_mask, t, valid):
        n, days, height, width = z.shape
        valid5 = valid[:, None].expand(-1, 1, -1, height, width)
        monthly5 = monthly[:, :, None].expand(-1, -1, days, -1, -1)
        daily5 = daily_features.permute(0, 2, 1, 3, 4)
        time5 = t[:, None, None, None, None].expand(-1, 1, days, height, width)
        position = torch.arange(days, device=z.device, dtype=z.dtype) / max(days - 1, 1)
        angle = position * 2 * math.pi
        day_sin = angle.sin()[None, None, :, None, None].expand(n, 1, days, height, width)
        day_cos = angle.cos()[None, None, :, None, None].expand(n, 1, days, height, width)
        inputs = torch.cat(
            (
                z[:, None] * valid5,
                monthly5,
                daily5 * valid5,
                observed_values[:, None] * valid5,
                observed_mask[:, None] * valid5,
                valid5,
                time5,
                day_sin,
                day_cos,
            ),
            dim=1,
        )
        hidden = self.stem(inputs) * valid5
        for block in self.blocks:
            hidden = (hidden + block(hidden) * valid5) * valid5
        return self.output(hidden)[:, 0] * valid


def prepare_masked_data(root: str | Path = DATA_ROOT):
    bundle = open_daily_feature_bundle(root)
    dates = bundle.gsr.time.values.astype("datetime64[D]")
    if not np.all(np.diff(dates) == np.timedelta64(1, "D")):
        raise ValueError("Daily dates must be continuous")
    weights = torch.tensor(np.cos(np.deg2rad(bundle.gsr.lat.values))[:, None] * np.ones((1, bundle.gsr.sizes["lon"])), dtype=torch.float32)
    fields, weather, masks, targets, keys = [], [], [], [], []
    for key in np.unique(dates.astype("datetime64[M]")):
        selected = dates.astype("datetime64[M]") == key
        days = int(selected.sum())
        expected = int(((key + 1).astype("datetime64[D]") - key.astype("datetime64[D]")).astype(int))
        if days != expected:
            raise ValueError(f"Incomplete month {key}")
        fields.append(np.pad(bundle.gsr.values[selected], ((0, 31 - days), (0, 0), (0, 0))))
        weather.append(np.pad(bundle.features.values[selected], ((0, 31 - days), (0, 0), (0, 0), (0, 0))))
        masks.append(np.arange(31) < days)
        targets.append(coarse(fields[-1].sum(0) / days, weights.numpy()))
        keys.append(str(key))
    x = torch.tensor(np.stack(fields), dtype=torch.float32)
    daily_features = torch.tensor(np.stack(weather), dtype=torch.float32)
    # Keep spatial dimensions explicit so temporal and spatial artificial gaps
    # use the same interface throughout the model.
    valid = torch.tensor(np.stack(masks), dtype=torch.float32)[:, :, None, None].expand(-1, -1, x.shape[-2], x.shape[-1]).clone()
    target = torch.tensor(np.stack(targets), dtype=torch.float32)
    return x, daily_features, valid, target, weights, keys, bundle.gsr.lat.values


def random_observation_mask(
    valid,
    rng,
    min_gap=1,
    max_gap=7,
    spatial_probability=0.35,
    mode="mixed",
):
    """Create one reproducible gap for temporal, spatial, endpoint, or mixed tests."""
    if mode not in {"mixed", "temporal", "spatial", "endpoint"}:
        raise ValueError("mode must be mixed, temporal, spatial, or endpoint")
    n, days, height, width = valid.shape
    result = valid.clone()
    for i in range(n):
        month_days = int(valid[i, :, 0, 0].sum())
        length = int(rng.integers(min_gap, min(max_gap, month_days) + 1))
        if mode == "endpoint":
            start = 0 if rng.random() < 0.5 else month_days - length
        else:
            start = int(rng.integers(0, month_days - length + 1))
        spatial = mode == "spatial" or (mode == "mixed" and rng.random() < spatial_probability)
        if spatial:
            h = int(rng.integers(2, 6))
            w = int(rng.integers(2, 6))
            lat = int(rng.integers(0, height - h + 1))
            lon = int(rng.integers(0, width - w + 1))
            result[i, start : start + length, lat : lat + h, lon : lon + w] = 0.0
        else:
            result[i, start : start + length] = 0.0
    return result


def summarize_masked_metrics(samples, truth, observed, valid):
    """Return pooled and equal-month metrics for an ensemble and a gap mask."""
    missing = (valid > 0) & (observed <= 0)
    if not bool(missing.any()):
        raise ValueError("The observation mask contains no missing valid cells")
    pooled = samples.permute(1, 0, 2, 3, 4)[:, missing].reshape(samples.shape[1], -1)
    target = truth[missing]
    ordered = pooled.sort(0).values
    count = pooled.shape[0]
    coeff = (2 * torch.arange(1, count + 1) - count - 1).to(pooled)[:, None]
    crps = (pooled - target).abs().mean(0) - (coeff * ordered).sum(0) / (count * count)
    lower, upper = torch.quantile(pooled, 0.05, 0), torch.quantile(pooled, 0.95, 0)
    mean = pooled.mean(0)
    rows = masked_metrics(samples, truth, observed, valid)
    return {
        "pooled_missing_fraction": float(missing.float().mean()),
        "pooled_rmse": float((mean - target).square().mean().sqrt()),
        "pooled_mae": float((mean - target).abs().mean()),
        "pooled_crps": float(crps.mean()),
        "pooled_coverage90": float(((target >= lower) & (target <= upper)).float().mean()),
        "pooled_width90": float((upper - lower).mean()),
        "monthly_rmse": float(np.mean([row["rmse"] for row in rows])),
        "monthly_mae": float(np.mean([row["mae"] for row in rows])),
        "monthly_crps": float(np.mean([row["crps"] for row in rows])),
        "monthly_coverage90": float(np.mean([row["coverage90"] for row in rows])),
        "monthly_width90": float(np.mean([row["width90"] for row in rows])),
        "observed_mae": float(np.mean([row["observed_mae"] for row in rows])),
    }


def graph_laplacian(values):
    """Four-neighbour graph Laplacian on the regular spatial grid."""
    shape = values.shape
    q = values.reshape(-1, shape[-2], shape[-1])
    p = torch.nn.functional.pad(q[:, None], (1, 1, 1, 1), mode="replicate")[:, 0]
    out = 4.0 * q - (p[:, 1:-1, :-2] + p[:, 1:-1, 2:] + p[:, :-2, 1:-1] + p[:, 2:, 1:-1])
    return out.reshape(shape)


def graph_structure_loss(pred, truth, mask=None, edge_weight=0.5, laplacian_weight=0.5):
    """Graph-signal regularizer matching edges and Laplacian responses."""
    dxp = pred[..., :, 1:] - pred[..., :, :-1]
    dxt = truth[..., :, 1:] - truth[..., :, :-1]
    dyp = pred[..., 1:, :] - pred[..., :-1, :]
    dyt = truth[..., 1:, :] - truth[..., :-1, :]
    edge = (dxp - dxt).abs().mean() + (dyp - dyt).abs().mean()
    lap = (graph_laplacian(pred) - graph_laplacian(truth)).abs()
    if mask is not None:
        lap = lap * mask
        lap = lap.sum() / mask.sum().clamp_min(1.0)
    else:
        lap = lap.mean()
    return edge_weight * edge + laplacian_weight * lap


def graph_boundary_projection(values, observed, valid, strength=0.15):
    """One explicit graph-consistency correction after a diffusion chunk."""
    if strength <= 0:
        return values
    obs = observed * valid
    p = torch.nn.functional.pad(values, (1, 1, 1, 1), mode="replicate")
    q = torch.nn.functional.pad(obs, (1, 1, 1, 1), mode="replicate")
    neigh = (p[..., 1:-1, :-2] + p[..., 1:-1, 2:] + p[..., :-2, 1:-1] + p[..., 2:, 1:-1]) / 4.0
    support = (q[..., 1:-1, :-2] + q[..., 1:-1, 2:] + q[..., :-2, 1:-1] + q[..., 2:, 1:-1]) / 4.0
    boundary = ((obs <= 0) & (support > 0) & (valid > 0)).to(values.dtype)
    corrected = values + strength * boundary * (neigh - values)
    return torch.where(obs > 0, values, corrected).clamp_min(0.0) * valid


def spatial_boundary_blend(values, observed, valid, blend=0.25):
    """Propagate observed boundary context into spatial gaps without changing observations."""
    if blend <= 0:
        return values
    observed = observed * valid
    padded = torch.nn.functional.pad(values, (1, 1, 1, 1), mode="replicate")
    padded_obs = torch.nn.functional.pad(observed, (1, 1, 1, 1), mode="replicate")
    neighbors = (
        padded[..., 1:-1, :-2] + padded[..., 1:-1, 2:] +
        padded[..., :-2, 1:-1] + padded[..., 2:, 1:-1]
    ) / 4.0
    neighbor_obs = (
        padded_obs[..., 1:-1, :-2] + padded_obs[..., 1:-1, 2:] +
        padded_obs[..., :-2, 1:-1] + padded_obs[..., 2:, 1:-1]
    ) / 4.0
    boundary = ((neighbor_obs > 0) & (observed <= 0) & (valid > 0)).to(values.dtype)
    result = values * (1.0 - blend * boundary) + neighbors * (blend * boundary)
    return torch.where(observed > 0, values, result) * valid


def spatial_diffusion_blend(values, observed, valid, blend=0.35, iterations=4):
    """Diffuse boundary observations into block gaps with distance decay."""
    if blend <= 0 or iterations <= 0:
        return values
    obs = observed * valid
    propagated = torch.where(obs > 0, values, torch.zeros_like(values))
    support = obs.clone()
    for _ in range(iterations):
        padded_values = torch.nn.functional.pad(propagated, (1, 1, 1, 1), mode="replicate")
        padded_support = torch.nn.functional.pad(support, (1, 1, 1, 1), mode="replicate")
        neighbor_values = (
            padded_values[..., 1:-1, :-2] + padded_values[..., 1:-1, 2:] +
            padded_values[..., :-2, 1:-1] + padded_values[..., 2:, 1:-1]
        ) / 4.0
        neighbor_support = (
            padded_support[..., 1:-1, :-2] + padded_support[..., 1:-1, 2:] +
            padded_support[..., :-2, 1:-1] + padded_support[..., 2:, 1:-1]
        ) / 4.0
        fillable = (support <= 0) & (neighbor_support > 0) & (valid > 0)
        propagated = torch.where(fillable, neighbor_values, propagated)
        support = torch.maximum(support, fillable.to(support.dtype))
    result = values * (1.0 - blend * support) + propagated * (blend * support)
    return torch.where(obs > 0, values, result) * valid


def normalize_features(features, train_indices):
    selected = features[train_indices]
    mean = selected.mean((0, 1, 3, 4))
    centered = selected - mean[None, None, :, None, None]
    std = centered.square().mean((0, 1, 3, 4)).sqrt().clamp_min(1e-6)
    return mean, std


@torch.no_grad()
def sample_masked(
    net,
    target,
    keys,
    valid,
    weather,
    observed_values,
    observed,
    weights,
    mean_scale,
    residual_scale,
    base,
    factor,
    seed=11,
    members=12,
    chunk_size=8,
    boundary_blend=0.25,
    graph_strength=0.15,
    chunk_projection=True,
    gain_mode="adaptive",
    return_trace=False,
):
    """Sample masked months with hard observations and chunked feedback.

    ``graph_strength`` scales the four-neighbour boundary correction applied at
    every chunk boundary; ``boundary_blend`` scales the final spatial boundary
    and diffusion blends. Setting either to zero removes that operation, which
    is what the graph ablation varies.

    ``chunk_projection`` selects whether the bounded monthly projection runs
    inside every chunk (default) or only once at the end of reverse diffusion.
    ``gain_mode`` selects the feedback gain: ``adaptive`` uses the ensemble
    variance ratio, ``uniform`` a fixed 0.5, and ``full`` a gain of 1.
    """
    if gain_mode not in {"adaptive", "uniform", "full"}:
        raise ValueError("gain_mode must be adaptive, uniform or full")
    device = target.device
    torch.manual_seed(seed)
    batch = target.shape[0]
    target_members = target.repeat_interleave(members, 0)
    valid_members = valid.repeat_interleave(members, 0)
    weather_members = weather.repeat_interleave(members, 0)
    observed_members = observed.repeat_interleave(members, 0)
    observed_values_members = observed_values.repeat_interleave(members, 0)
    base_members = base.repeat_interleave(members, 0)
    factor_members = factor.repeat_interleave(members, 0)
    # The solar prior uses factor = max(Ra, 20) / 400.  Use the same Ra
    # quantity as a conservative daily physical ceiling for generated GSR.
    upper_members = (factor_members * 400.0 * 1.05).clamp_min(50.0)
    monthly = monthly_conditions(
        target_members,
        [key for key in keys for _ in range(members)],
        mean_scale,
        base_members,
        valid_members,
    )

    alpha_top = alphas(device)[-1]
    z = torch.randn_like(observed_values_members) * valid_members
    observed_noise = torch.randn_like(z)
    z = torch.where(
        observed_members > 0,
        alpha_top.sqrt() * observed_values_members + (1 - alpha_top).sqrt() * observed_noise,
        z,
    ) * valid_members
    traces = []
    raw = None
    schedule = alphas(device)
    for low, high in diffusion_chunks(64, chunk_size):
        for step in range(high, low - 1, -1):
            alpha = schedule[step]
            velocity = net(
                z,
                monthly,
                weather_members,
                observed_values_members,
                observed_members,
                torch.full((z.shape[0],), step / 63, device=device),
                valid_members,
            )
            clean = recover(z, velocity, alpha)
            raw = torch.minimum((clean * residual_scale * factor_members + base_members).clamp_min(0.0), upper_members) * valid_members
            raw = torch.where(observed_members > 0, observed_values_members * residual_scale * factor_members + base_members, raw)
            clean = ((raw - base_members) / (residual_scale * factor_members).clamp_min(1e-4)) * valid_members
            eps = consistent_noise(z, clean, alpha) * valid_members
            previous = schedule[step - 1] if step else torch.tensor(1.0, device=device)
            z = (previous.sqrt() * clean + (1 - previous).sqrt() * eps) * valid_members
            if step:
                observed_noise = torch.randn_like(z)
                noisy_observed = previous.sqrt() * observed_values_members + (1 - previous).sqrt() * observed_noise
                z = torch.where(observed_members > 0, noisy_observed, z)

        # Apply the monthly constraint once per chunk, then feed the corrected
        # state back into the next chunk through a consistent noisy latent.
        # `chunk_projection=False` keeps the mechanism but defers it to the end
        # of sampling, so the two are separable.
        if gain_mode == "adaptive":
            spread = raw.reshape(batch, members, *raw.shape[1:]).var(1, unbiased=False).repeat_interleave(members, 0)
            gain = (spread / (spread + (residual_scale * factor_members).square())).clamp(0.15, 0.85)
        elif gain_mode == "uniform":
            gain = torch.full_like(raw, 0.5)
        else:
            gain = torch.ones_like(raw)
        if chunk_projection:
            correction = bounded_partial_project(
                raw, target_members, weights, valid_members[:, :, 0, 0], observed_members, upper_members
            )
            progress = 1.0 - low / 64.0
            raw = raw + progress * gain * (correction - raw)
        raw = graph_boundary_projection(raw, observed_members, valid_members, strength=graph_strength)
        raw = torch.where(observed_members > 0, observed_values_members * residual_scale * factor_members + base_members, raw)
        raw = torch.minimum(raw.clamp_min(0.0), upper_members) * valid_members
        boundary = schedule[low - 1] if low else torch.tensor(1.0, device=device)
        clean = ((raw - base_members) / (residual_scale * factor_members).clamp_min(1e-4)) * valid_members
        if low:
            eps = consistent_noise(z, clean, boundary) * valid_members
            z = (boundary.sqrt() * clean + (1 - boundary).sqrt() * eps) * valid_members
            observed_noise = torch.randn_like(z)
            noisy_observed = boundary.sqrt() * observed_values_members + (1 - boundary).sqrt() * observed_noise
            z = torch.where(observed_members > 0, noisy_observed, z)
        else:
            z = clean
        if return_trace:
            traces.append(raw.detach().cpu())

    result = torch.minimum((z * residual_scale * factor_members + base_members).clamp_min(0.0), upper_members) * valid_members
    result = bounded_partial_project(
        result, target_members, weights, valid_members[:, :, 0, 0], observed_members, upper_members
    )
    result = spatial_boundary_blend(result, observed_members, valid_members, boundary_blend)
    result = spatial_diffusion_blend(result, observed_members, valid_members, boundary_blend, iterations=4)
    result = bounded_partial_project(
        result, target_members, weights, valid_members[:, :, 0, 0], observed_members, upper_members
    )
    result = torch.where(observed_members > 0, observed_values_members * residual_scale * factor_members + base_members, result)
    result = torch.minimum(result.clamp_min(0.0), upper_members) * valid_members
    result = result.reshape(batch, members, *result.shape[1:]).cpu()
    return (result, traces) if return_trace else result


def masked_metrics(samples, truth, observed, valid):
    """Score only artificially missing cells, avoiding padded and observed cells."""
    rows = []
    for i in range(truth.shape[0]):
        missing = (valid[i] > 0) & (observed[i] <= 0)
        if not bool(missing.any()):
            continue
        s = samples[i][:, missing]
        y = truth[i][missing]
        ordered = s.sort(0).values
        count = s.shape[0]
        coeff = (2 * torch.arange(1, count + 1) - count - 1).to(s)[:, None]
        crps = (s - y).abs().mean(0) - (coeff * ordered).sum(0) / (count * count)
        lower, upper = torch.quantile(s, 0.05, 0), torch.quantile(s, 0.95, 0)
        mean = s.mean(0)
        rows.append({
            "missing_fraction": float(missing.float().mean()),
            "rmse": float((mean - y).square().mean().sqrt()),
            "mae": float((mean - y).abs().mean()),
            "crps": float(crps.mean()),
            "coverage90": float(((y >= lower) & (y <= upper)).float().mean()),
            "width90": float((upper - lower).mean()),
            "observed_mae": float((samples[i][:, (valid[i] > 0) & (observed[i] > 0)].mean(0) - truth[i][(valid[i] > 0) & (observed[i] > 0)]).abs().mean()),
        })
    return rows


def evaluate_masked(
    output: str | Path = OUT,
    root: str | Path = DATA_ROOT,
    split: str = "validation",
    members: int = 12,
    chunk_size: int = 8,
    seed: int = 20260911,
    gap_mode: str = "mixed",
    min_gap: int = 1,
    max_gap: int = 7,
    spatial_probability: float = 0.35,
    boundary_blend: float = 0.25,
):
    """Evaluate a saved V6 checkpoint on synthetic temporal/spatial gaps."""
    output = Path(output)
    checkpoint = torch.load(output / "model_11.pt", map_location="cpu", weights_only=False)
    x, daily_features, valid, target, weights, keys, latitudes = prepare_masked_data(root)
    years = np.array([int(key[:4]) for key in keys])
    split_ids = {
        "train": np.flatnonzero(years <= 2009),
        "validation": np.flatnonzero((years >= 2010) & (years <= 2011)),
        "test": np.flatnonzero(years >= 2012),
    }
    if split not in split_ids:
        raise ValueError(f"Unknown split {split!r}; choose train, validation, or test")
    ids_all = split_ids[split]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = MaskedTemporalVelocityNet().to(device)
    net.load_state_dict(checkpoint["model"])
    net.eval()
    physics = SolarPrior(checkpoint["latitude"], checkpoint["template"])
    valid_device = valid.to(device)
    target_device = target.to(device)
    weights_device = weights.to(device)
    base, factor = physics.fields(target_device, keys, weights_device, valid_device)
    residual = ((x.to(device) - base) / factor.clamp_min(1e-4)) * valid_device
    residual = residual / max(float(checkpoint["residual_scale"]), 1e-6)
    feature_mean = checkpoint["feature_mean"].to(torch.float32)
    feature_std = checkpoint["feature_std"].to(torch.float32)
    weather = ((daily_features - feature_mean[None, None, :, None, None]) / feature_std[None, None, :, None, None]).to(device)
    mean_scale = float(checkpoint["mean_scale"])
    rng = np.random.default_rng(seed)
    samples, observed_rows = [], []
    for ids_np in np.array_split(ids_all, max(1, len(ids_all) // 6)):
        ids = torch.tensor(ids_np, device=device)
        observed = random_observation_mask(
            valid[ids_np], rng, min_gap, max_gap, spatial_probability, gap_mode
        ).to(device)
        generated = sample_masked(
            net,
            target_device[ids],
            [keys[i] for i in ids_np],
            valid_device[ids],
            weather[ids],
            residual[ids] * observed,
            observed,
            weights_device,
            mean_scale,
            float(checkpoint["residual_scale"]),
            base[ids],
            factor[ids],
            seed=seed + int(ids_np[0]),
            members=members,
            chunk_size=chunk_size,
            boundary_blend=boundary_blend,
        )
        samples.append(generated)
        observed_rows.append(observed.cpu())
    samples = torch.cat(samples)
    observed = torch.cat(observed_rows)
    rows = masked_metrics(samples, x[ids_all], observed.cpu(), valid[ids_all])
    summary = summarize_masked_metrics(samples, x[ids_all], observed.cpu(), valid[ids_all])
    payload = {
        "split": split,
        "members": members,
        "chunk_size": chunk_size,
        "seed": seed,
        "gap_mode": gap_mode,
        "gap_days": [min_gap, max_gap],
        "spatial_probability": spatial_probability,
        "device": str(device),
        "summary": summary,
        "by_month": rows,
        "note": "Metrics are computed only on synthetic missing cells; test split is exploratory until an untouched period is reserved.",
    }
    (output / f"masked_metrics_{split}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    np.savez_compressed(
        output / f"masked_ensemble_{split}.npz",
        gsr=samples.numpy(),
        truth=x[ids_all].numpy(),
        observed=observed.numpy(),
        valid=valid[ids_all].numpy(),
        months=np.array(keys)[ids_all],
    )
    return payload


def monthly_conditions(target, keys, mean_scale, base, valid=None):
    angle = torch.tensor([2 * math.pi * (int(key[5:]) - 1) / 12 for key in keys], device=target.device)
    # The daily baseline is represented by its monthly spatial average here;
    # its day-varying component is retained in the residual target.
    if valid is None:
        day_counts = torch.tensor(
            [calendar.monthrange(*map(int, key.split("-")))[1] for key in keys],
            device=base.device,
            dtype=base.dtype,
        )
        monthly_base = base.sum(dim=1) / day_counts[:, None, None, None]
    else:
        monthly_base = (base * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
    return torch.stack(
        (
            expand(target) / mean_scale,
            angle.sin()[:, None, None].expand(-1, 10, 10),
            angle.cos()[:, None, None].expand(-1, 10, 10),
            monthly_base / mean_scale,
        ),
        dim=1,
    )


def train_masked(
    epochs: int = 30,
    seeds: tuple[int, ...] = (11,),
    root: str | Path = DATA_ROOT,
    output: str | Path = OUT,
    spatial_probability: float = 0.65,
    missing_weight: float = 4.0,
    gradient_weight: float = 0.08,
    training_mode: str = "spatial_curriculum",
    graph_weight: float = 0.05,
):
    """Train the masked velocity network.

    ``graph_weight`` scales the four-neighbour graph-structure loss. Setting it
    to zero is the training-side graph ablation; every other term, the data
    order and the sampled masks are unchanged.
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x, daily_features, valid, target, weights, keys, latitudes = prepare_masked_data(root)
    years = np.array([int(key[:4]) for key in keys])
    train = np.flatnonzero(years <= 2009)
    validation = np.flatnonzero((years >= 2010) & (years <= 2011))
    test = np.flatnonzero(years >= 2012)
    mean_scale = float(x[train].sum() / valid[train].sum())
    physics = SolarPrior.fit(latitudes, x, valid, keys, train)
    valid_device = valid.to(device)
    target_device = target.to(device)
    base, factor = physics.fields(target_device, keys, weights.to(device), valid_device)
    residual = ((x.to(device) - base) / factor.clamp_min(1e-4)) * valid_device
    residual_scale = float((residual[train].square().sum() / valid_device[train].sum()).sqrt())
    residual = residual / max(residual_scale, 1e-6)
    normalized_features_mean, normalized_features_std = normalize_features(daily_features, train)
    weather = ((daily_features - normalized_features_mean[None, None, :, None, None]) / normalized_features_std[None, None, :, None, None]).to(device)
    monthly = monthly_conditions(target_device, keys, mean_scale, base, valid_device)
    ab = alphas(device)
    rng = np.random.default_rng(20260911)
    history = {}
    # The per-month tensors are indexed with device tensors below; no host-side
    # copy of the full residual stack is made per batch.
    for seed in seeds:
        torch.manual_seed(seed)
        net = MaskedTemporalVelocityNet().to(device)
        optimizer = torch.optim.AdamW(net.parameters(), lr=3e-4)
        records = []
        for epoch in range(epochs):
            net.train()
            losses = []
            for ids_np in np.array_split(rng.permutation(train), max(1, len(train) // 12)):
                ids = torch.as_tensor(ids_np, device=device)
                res_b = residual.index_select(0, ids)
                mon_b = monthly.index_select(0, ids)
                wea_b = weather.index_select(0, ids)
                val_b = valid_device.index_select(0, ids)
                diffusion_step = torch.randint(0, 64, (len(ids),), device=device)
                alpha = ab[diffusion_step, None, None, None]
                noise = torch.randn_like(res_b) * val_b
                z = (alpha.sqrt() * res_b + (1 - alpha).sqrt() * noise) * val_b
                if training_mode == "spatial_curriculum":
                    gap_mode = "mixed" if epoch < max(1, epochs // 3) else "spatial"
                    probability = spatial_probability if gap_mode == "mixed" else 1.0
                else:
                    gap_mode, probability = "mixed", spatial_probability
                observed = random_observation_mask(
                    valid[ids_np], rng, min_gap=1, max_gap=7,
                    spatial_probability=probability, mode=gap_mode
                ).to(device)
                observed_values = res_b * observed
                velocity = alpha.sqrt() * noise - (1 - alpha).sqrt() * res_b
                prediction = net(z, mon_b, wea_b, observed_values, observed, diffusion_step / 63, val_b)
                pixel_weight = 1.0 + missing_weight * (1.0 - observed)
                loss = ((prediction - velocity).square() * val_b * pixel_weight).sum() / (val_b * pixel_weight).sum().clamp_min(1.0)
                # Encourage realistic spatial texture specifically where the
                # target is hidden. This auxiliary term acts on the recovered
                # clean residual and uses horizontal finite differences.
                clean_prediction = recover(z, prediction, alpha)
                if gradient_weight > 0:
                    dx_pred = clean_prediction[..., :, 1:] - clean_prediction[..., :, :-1]
                    dx_true = res_b[..., :, 1:] - res_b[..., :, :-1]
                    dy_pred = clean_prediction[..., 1:, :] - clean_prediction[..., :-1, :]
                    dy_true = res_b[..., 1:, :] - res_b[..., :-1, :]
                    miss_x = (1.0 - observed[..., :, 1:] * observed[..., :, :-1]) * val_b[..., :, 1:] * val_b[..., :, :-1]
                    miss_y = (1.0 - observed[..., 1:, :] * observed[..., :-1, :]) * val_b[..., 1:, :] * val_b[..., :-1, :]
                    gradient_loss = (
                        ((dx_pred - dx_true).square() * miss_x).sum() / miss_x.sum().clamp_min(1.0)
                        + ((dy_pred - dy_true).square() * miss_y).sum() / miss_y.sum().clamp_min(1.0)
                    )
                    loss = loss + gradient_weight * gradient_loss
                # Explicit graph-signal constraint on the recovered clean state.
                if graph_weight > 0:
                    graph_mask = (1.0 - observed) * val_b
                    loss = loss + graph_weight * graph_structure_loss(clean_prediction, res_b, graph_mask)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                optimizer.step()
                losses.append(float(loss.detach()))
            records.append({"epoch": epoch + 1, "loss": float(np.mean(losses))})
        history[str(seed)] = records
        torch.save(
            {
                "model": net.state_dict(),
                "mean_scale": mean_scale,
                "residual_scale": residual_scale,
                "feature_mean": normalized_features_mean,
                "feature_std": normalized_features_std,
                "latitude": physics.latitude,
                "template": physics.template,
                "feature_names": FEATURES,
            },
            output / f"model_{seed}.pt",
        )
    protocol = {
        "name": "masked_v6_daily_weather_conditioned_diffusion",
        "target_provider": "NASA POWER ALLSKY_SFC_SW_DWN, converted from kWh m-2 day-1 to daily mean W m-2",
        "conditioning_provider": "NASA POWER daily T2M, PRECTOTCORR, CLOUD_AMT",
        "conditioning_units": {"T2M": "degC", "PRECTOTCORR": "mm/day", "CLOUD_AMT": "%"},
        "train": "2000-2009", "validation": "2010-2011", "test": "2012-2014 exploratory",
        "random_missing_masks": {"temporal_gap_days": [1, 7], "spatial_block_probability": 0.35, "mask_is_explicit_network_input": True},
        "spatial_curriculum": {"training_spatial_probability": spatial_probability, "missing_pixel_weight": missing_weight, "gradient_loss_weight": gradient_weight, "graph_structure_weight": graph_weight},
        "training_mode": training_mode,
        "observed_value_input": "normalized solar residual at observed cells; zero elsewhere",
        "network": "MaskedTemporalVelocityNet, 64-step velocity diffusion, four factorized space-time blocks",
        "device": str(device), "epochs": epochs, "seeds": list(seeds), "test_used_for_selection": False,
        "scientific_boundary": "NASA POWER is satellite/reanalysis-derived, not station measurements; CMIP6 transfer still requires bias-adjusted monthly inputs.",
    }
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    (output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    write_feature_manifest(open_daily_feature_bundle(root), output / "feature_manifest.json")
    return output


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--root", type=Path, default=DATA_ROOT)
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--split", choices=("train", "validation", "test"), default="validation")
    parser.add_argument("--members", type=int, default=12)
    parser.add_argument("--chunk-size", type=int, default=8)
    parser.add_argument("--gap-mode", choices=("mixed", "temporal", "spatial", "endpoint"), default="mixed")
    parser.add_argument("--min-gap", type=int, default=1)
    parser.add_argument("--max-gap", type=int, default=7)
    parser.add_argument("--spatial-probability", type=float, default=0.35)
    parser.add_argument("--boundary-blend", type=float, default=0.25)
    parser.add_argument("--train-spatial-probability", type=float, default=0.65)
    parser.add_argument("--missing-weight", type=float, default=4.0)
    parser.add_argument("--gradient-weight", type=float, default=0.08)
    parser.add_argument("--training-mode", choices=("mixed", "spatial_curriculum"), default="spatial_curriculum")
    args = parser.parse_args()
    if args.evaluate:
        print(json.dumps(evaluate_masked(
            args.output, args.root, args.split, args.members, args.chunk_size,
            20260911, args.gap_mode, args.min_gap, args.max_gap, args.spatial_probability,
            args.boundary_blend,
        ), indent=2), flush=True)
    else:
        train_masked(
            epochs=args.epochs, root=args.root, output=args.output,
            spatial_probability=args.train_spatial_probability,
            missing_weight=args.missing_weight,
            gradient_weight=args.gradient_weight,
            training_mode=args.training_mode,
        )
