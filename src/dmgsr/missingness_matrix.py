"""Missingness-regime matrix: where each method actually works.

Evaluates the frozen diffusion checkpoint and the U-Net ensemble on the same
stored masks across the four gap families the manuscript defines
(temporal / spatial / mixed / endpoint) and three duration bands
(1-3, 4-7, 8-14 days), on the 2010 calibration period. Every cell is scored on
identical hidden cells for both branches, and several mask realisations are
drawn per cell so the spread is not a single draw.

This is the experiment the manuscript's abstract asserts qualitatively
("adjacent locations missing on consecutive days are the most demanding") but
never measures.

Usage:
    python -m dmgsr.missingness_matrix
    python -m dmgsr.missingness_matrix --masks 5 --split validation
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from .confirm import metric
from .masked_temporal import (
    DATA_ROOT,
    MaskedTemporalVelocityNet,
    prepare_masked_data,
    random_observation_mask,
    sample_masked,
)
from .solar_prior import SolarPrior
from .spatial_unet import SpatialUNet, make_spatial_inputs

OUT = Path("outputs/missingness_matrix")
MODES = ("temporal", "spatial", "mixed", "endpoint")
BANDS = ((1, 3), (4, 7), (8, 14))
MASK_BASE = 990001
SEED = 11
N_UNET = 8


def build(bundle, ids, device):
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
    net = MaskedTemporalVelocityNet().to(device)
    net.load_state_dict(ck["model"])
    net.eval()
    return dict(net=net, base=base, factor=factor, residual=residual, weather=weather,
                scale=scale, mean_scale=float(ck["mean_scale"]), vd=vd, td=td,
                weights=weights.to(device))


@torch.no_grad()
def diffusion(pre, bundle, ids, masks, device, members=12):
    pieces = []
    for a in range(0, len(ids), 6):
        g = ids[a:a + 6]
        ii = torch.as_tensor(g, device=device)
        ob = masks[a:a + 6]
        pieces.append(sample_masked(
            pre["net"], pre["td"][ii], [bundle["keys"][i] for i in g], pre["vd"][ii],
            pre["weather"][ii], pre["residual"][ii] * ob, ob, pre["weights"],
            pre["mean_scale"], pre["scale"], pre["base"][ii], pre["factor"][ii],
            seed=500 + SEED + int(g[0]), members=members))
    return torch.cat(pieces).cpu().numpy()


@torch.no_grad()
def unet(bundle, ids, masks, device, x):
    """Per-day U-Net at hidden cells, 8 members."""
    valid, target, weights, keys, lat = (bundle["valid"], bundle["target"],
                                         bundle["weights"], bundle["keys"], bundle["lat"])
    train = np.flatnonzero(np.array([int(k[:4]) for k in keys]) <= 2009)
    prior = SolarPrior.fit(lat, x, valid, keys, train)
    base, _ = prior.fields(target.to(device), keys, weights.to(device), valid.to(device))
    truth = x[ids]
    preds = np.empty((len(ids), N_UNET) + tuple(x.shape[1:]), dtype=np.float32)
    for m in range(N_UNET):
        ck = torch.load(f"outputs/spatial_unet_ensemble/model_{m}.pt",
                        map_location=device, weights_only=False)
        netm = SpatialUNet().to(device)
        netm.load_state_dict(ck["model"] if "model" in ck else ck)
        netm.eval()
        restored = torch.where(masks.cpu() > 0, truth, torch.zeros_like(truth))
        for i, mid in enumerate(ids):
            miss_days = ((valid[mid] > 0) & (masks[i].cpu() <= 0)).any((-1, -2)).nonzero().flatten().tolist()
            for d in miss_days:
                obs = masks[i, d].cpu()
                y = truth[i, d].to(device)
                wd = bundle["features"][mid, d].to(device)
                monthly = target[mid].mean().expand_as(y) / 300.0
                inp = make_spatial_inputs(y * obs.to(device), obs.to(device), wd, monthly, base[mid, d])
                p = (netm(inp[None])[0] * 400.0).clamp(0.0, 400.0).cpu()
                restored[i, d] = torch.where(obs > 0, truth[i, d], p)
        preds[:, m] = restored.numpy()
    return preds


def score(samples, masks, valid_ids, truth):
    hidden = (masks.cpu().numpy() <= 0) & (valid_ids > 0)
    pooled = []
    for i in range(samples.shape[0]):
        sel = hidden[i].reshape(-1)
        if sel.any():
            pooled.append(samples[i].reshape(samples.shape[1], -1)[:, sel].T)
    if not pooled:
        return None
    s = np.concatenate(pooled, axis=0)
    return metric(s, truth[hidden])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=("validation", "report"), default="validation")
    ap.add_argument("--masks", type=int, default=3, help="mask realisations per cell")
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

    payload = {"split": args.split, "seed": SEED, "members": args.members,
               "unet_members": N_UNET, "mask_realisations": args.masks,
               "note": "matched masks across methods; hidden cells only"}
    rows = []
    for mode in MODES:
        for lo, hi in BANDS:
            agg = {"diffusion": [], "unet": []}
            for rep in range(args.masks):
                masks = torch.stack([
                    random_observation_mask(
                        valid[i][None], np.random.default_rng(MASK_BASE + rep * 100000 + int(keys[i][:4]) * 12 + int(keys[i][5:7])),
                        lo, hi, 0.35, mode)[0]
                    for i in ids]).to(device)
                if float((masks <= 0).float().mean()) <= 0:
                    continue
                d = diffusion(pre, bundle, ids, masks, device, args.members)
                u = unet(bundle, ids, masks, device, x)
                agg["diffusion"].append(score(d, masks, valid_ids, truth))
                agg["unet"].append(score(u, masks, valid_ids, truth))
            cell = {}
            for name, ms in agg.items():
                ms = [m for m in ms if m]
                if not ms:
                    continue
                cell[name] = {k: float(np.mean([m[k] for m in ms])) for k in ms[0]
                              if k != "n"}
                cell[name]["n"] = int(np.mean([m["n"] for m in ms]))
            rows.append({"mode": mode, "band": [lo, hi], **cell})
            line = (f"[{mode:8s} {lo}-{hi:2d}d] "
                    f"diffusion rmse {cell['diffusion']['rmse']:7.3f} crps {cell['diffusion']['crps']:7.3f} "
                    f"cov {cell['diffusion']['coverage90']:.3f} | "
                    f"unet rmse {cell['unet']['rmse']:7.3f} crps {cell['unet']['crps']:7.3f} "
                    f"cov {cell['unet']['coverage90']:.3f}")
            print(line, flush=True)
    payload["cells"] = rows
    payload["seconds"] = time.time() - t0
    (OUT / f"matrix_{args.split}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # Regime summary: RMSE of the two branches and the coverage gap.
    lines = ["# Missingness-regime matrix", "",
             f"Frozen checkpoints, split `{args.split}`, {args.masks} mask realisation(s) per cell, "
             f"{args.members} diffusion members and {N_UNET} U-Net members.",
             "", "| Mode | Days | n | Diffusion RMSE | U-Net RMSE | Diffusion CRPS | U-Net CRPS | Diffusion cov | U-Net cov |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['mode']} | {r['band'][0]}-{r['band'][1]} | {r['diffusion']['n']} | "
                     f"{r['diffusion']['rmse']:.3f} | {r['unet']['rmse']:.3f} | "
                     f"{r['diffusion']['crps']:.3f} | {r['unet']['crps']:.3f} | "
                     f"{100*r['diffusion']['coverage90']:.1f}% | {100*r['unet']['coverage90']:.1f}% |")
    (OUT / f"REPORT_{args.split}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[done] {payload['seconds']:.1f} s", flush=True)


if __name__ == "__main__":
    main()
