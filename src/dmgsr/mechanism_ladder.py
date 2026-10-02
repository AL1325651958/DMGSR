"""Sampling-mechanism ladder: how much does the chunked constraint feedback buy?

Three axes are swept on frozen checkpoints, one at a time, with everything else
held fixed. Each measures one component of the sampling-time mechanism that the
manuscript describes but never isolates.

    chunk      chunk_size in {1, 2, 4, 8, 16, 32, 64}, i.e. a constraint applied
               every 1, 2, 4, 8, 16, 32 or 64 reverse steps
    projection `full` (monthly projection inside each chunk), `final` (only after
               the last step), `none` (no monthly projection at all)
    gain       `adaptive` (current), `uniform` (fixed 0.5), `full` (gain 1.0)

Usage:
    python -m dmgsr.mechanism_ladder --split validation
    python -m dmgsr.mechanism_ladder --split validation --axes chunk
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from .masked_temporal import (
    DATA_ROOT,
    MaskedTemporalVelocityNet,
    prepare_masked_data,
    random_observation_mask,
    sample_masked,
)
from .solar_prior import SolarPrior

OUT = Path("outputs/mechanism_ladder")
SEED = 11
MASK_SEED = 880001
SPATIAL_PROB = 0.35
CHUNKS = (1, 2, 4, 8, 16, 32, 64)
PROJECTIONS = ("full", "final", "none")
GAINS = ("adaptive", "uniform", "full")


def metric(samples, y):
    n = samples.shape[1]
    mean = samples.mean(1)
    order = np.sort(samples, axis=1)
    coeff = (2 * np.arange(1, n + 1) - n - 1)[None, :]
    crps = np.abs(samples - y[:, None]).mean(1) - (coeff * order).sum(1) / n**2
    lo, hi = np.quantile(samples, 0.05, axis=1), np.quantile(samples, 0.95, axis=1)
    return {"rmse": float(np.sqrt(np.mean((mean - y) ** 2))),
            "mae": float(np.mean(np.abs(mean - y))),
            "crps": float(np.mean(crps)),
            "coverage90": float(np.mean((y >= lo) & (y <= hi))),
            "width90": float(np.mean(hi - lo)),
            "n": int(y.size)}


def build(bundle, ids, device):
    """Precompute the conditioning tensors once for the split."""
    ck = torch.load(f"outputs/masked_v9_graph/model_{SEED}.pt",
                    map_location="cpu", weights_only=False)
    physics = SolarPrior(ck["latitude"], ck["template"])
    x, features, valid, target, weights, keys = (
        bundle["x"], bundle["features"], bundle["valid"], bundle["target"],
        bundle["weights"], bundle["keys"])
    vd, td = valid.to(device), target.to(device)
    base, factor = physics.fields(td, keys, weights.to(device), vd)
    scale = float(ck["residual_scale"])
    residual = ((x.to(device) - base) / factor.clamp_min(1e-4)) * vd / scale
    fmean = ck["feature_mean"].to(device).to(torch.float32)
    fstd = ck["feature_std"].to(device).to(torch.float32)
    weather = (features.to(device) - fmean[None, None, :, None, None]) / fstd[None, None, :, None, None]
    masks = []
    for i in ids:
        index = int(keys[i][:4]) * 12 + int(keys[i][5:7])
        rng = np.random.default_rng(MASK_SEED + index)
        masks.append(random_observation_mask(valid[i][None], rng, 1, 7, SPATIAL_PROB, "mixed")[0])
    masks = torch.stack(masks).to(device)
    net = MaskedTemporalVelocityNet().to(device)
    net.load_state_dict(ck["model"])
    net.eval()
    return dict(net=net, base=base, factor=factor, residual=residual, weather=weather,
                masks=masks, scale=scale, mean_scale=float(ck["mean_scale"]),
                vd=vd, td=td, weights=weights.to(device))


@torch.no_grad()
def run(pre, bundle, ids, device, members=12, chunk_size=8, graph_strength=0.15,
        boundary_blend=0.25, chunk_projection=True, gain_mode="adaptive"):
    pieces = []
    for a in range(0, len(ids), 6):
        g = ids[a:a + 6]
        ii = torch.as_tensor(g, device=device)
        ob = pre["masks"][a:a + 6]
        pieces.append(sample_masked(
            pre["net"], pre["td"][ii], [bundle["keys"][i] for i in g], pre["vd"][ii],
            pre["weather"][ii], pre["residual"][ii] * ob, ob, pre["weights"],
            pre["mean_scale"], pre["scale"], pre["base"][ii], pre["factor"][ii],
            seed=1000 + SEED + int(g[0]), members=members, chunk_size=chunk_size,
            graph_strength=graph_strength, boundary_blend=boundary_blend,
            chunk_projection=chunk_projection, gain_mode=gain_mode))
    return torch.cat(pieces).cpu().numpy()


def score(samples, masks, valid_ids, truth):
    hidden = (masks.cpu().numpy() <= 0) & (valid_ids > 0)
    pooled = []
    for i in range(samples.shape[0]):
        sel = hidden[i].reshape(-1)
        pooled.append(samples[i].reshape(samples.shape[1], -1)[:, sel].T)
    s = np.concatenate(pooled, axis=0)
    return metric(s, truth[hidden])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=("validation", "report"), default="validation")
    ap.add_argument("--axes", nargs="+", default=["chunk", "projection", "gain"])
    ap.add_argument("--members", type=int, default=12)
    ap.add_argument("--root", type=Path, default=DATA_ROOT)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()

    x, features, valid, target, weights, keys, lat = prepare_masked_data(args.root)
    bundle = dict(x=x, features=features, valid=valid, target=target,
                  weights=weights, keys=keys, lat=lat)
    years = np.array([int(k[:4]) for k in keys])
    ids = np.flatnonzero(years == 2010) if args.split == "validation" else np.flatnonzero(years == 2011)
    pre = build(bundle, ids, device)
    truth = x[ids].numpy()
    valid_ids = valid[ids].numpy()
    print(f"[data] split={args.split} months={len(ids)} cells="
          f"{int(((pre['masks'].cpu().numpy() <= 0) & (valid_ids > 0)).sum())}", flush=True)

    payload = {"split": args.split, "seed": SEED, "members": args.members,
               "mask_seed": MASK_SEED, "note": "frozen checkpoint, one axis varied at a time"}

    if "chunk" in args.axes:
        payload["chunk"] = {}
        for cs in CHUNKS:
            s = run(pre, bundle, ids, device, args.members, chunk_size=cs)
            m = score(s, pre["masks"], valid_ids, truth)
            payload["chunk"][str(cs)] = m
            print(f"[chunk={cs:2d}] rmse {m['rmse']:.3f} crps {m['crps']:.3f} "
                  f"cov {m['coverage90']:.3f} width {m['width90']:.2f}", flush=True)

    if "projection" in args.axes:
        payload["projection"] = {}
        for name, cp, cs in (("per_chunk", True, 8), ("final_only", False, 8),
                             ("per_chunk_1step", True, 1)):
            s = run(pre, bundle, ids, device, args.members, chunk_size=cs,
                    chunk_projection=cp)
            m = score(s, pre["masks"], valid_ids, truth)
            payload["projection"][name] = m
            print(f"[projection={name:16s}] rmse {m['rmse']:.3f} crps {m['crps']:.3f} "
                  f"cov {m['coverage90']:.3f}", flush=True)

    if "gain" in args.axes:
        payload["gain"] = {}
        for name in GAINS:
            s = run(pre, bundle, ids, device, args.members, gain_mode=name)
            m = score(s, pre["masks"], valid_ids, truth)
            payload["gain"][name] = m
            print(f"[gain={name:9s}] rmse {m['rmse']:.3f} crps {m['crps']:.3f} "
                  f"cov {m['coverage90']:.3f}", flush=True)

    payload["seconds"] = time.time() - t0
    (OUT / f"ladder_{args.split}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[done] {payload['seconds']:.1f} s", flush=True)


if __name__ == "__main__":
    main()
