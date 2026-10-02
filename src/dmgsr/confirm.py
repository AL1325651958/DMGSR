"""Confirmatory evaluation on the untouched 2012-2014 period.

Frozen weights only: no training, no fine-tuning, no seed search. Calibration
uses 2010 alone (2011 is excluded because it selected the diffusion checkpoint),
and every quantity is then computed on 2012-2014, which no earlier analysis in
this repository has touched.

The decision rules are fixed in paper/PREREGISTRATION_2012_2014.md and were
written before this module was executed.

Usage:
    python -m dmgsr.confirm                     # full run
    python -m dmgsr.confirm --members 4         # cheaper smoke run
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
from .spatial_unet import SpatialUNet, make_spatial_inputs

OUT = Path("outputs/confirm_2012_2014")
CAL_YEAR = 2010
CONFIRM_YEARS = (2012, 2013, 2014)
N_SEEDS = (11, 22, 33)
N_UNET = 8
MASK_SEED = 770001          # fixed before any score was computed
SPATIAL_PROB = 0.35


# --------------------------------------------------------------------- metrics
def metric(samples, y):
    """samples [cells, members]; y [cells]."""
    n = samples.shape[1]
    mean = samples.mean(1)
    order = np.sort(samples, axis=1)
    coeff = (2 * np.arange(1, n + 1) - n - 1)[None, :]
    crps = np.abs(samples - y[:, None]).mean(1) - (coeff * order).sum(1) / n**2
    lo, hi = np.quantile(samples, 0.05, axis=1), np.quantile(samples, 0.95, axis=1)
    return {
        "rmse": float(np.sqrt(np.mean((mean - y) ** 2))),
        "mae": float(np.mean(np.abs(mean - y))),
        "crps": float(np.mean(crps)),
        "coverage90": float(np.mean((y >= lo) & (y <= hi))),
        "width90": float(np.mean(hi - lo)),
        "n": int(y.size),
    }


def mask_for(key, valid_i):
    """One stored mask per month, from a fixed pre-registered seed."""
    index = int(key[:4]) * 12 + int(key[5:7])
    rng = np.random.default_rng(MASK_SEED + index)
    return random_observation_mask(
        valid_i[None], rng, 1, 7, SPATIAL_PROB, "mixed"
    )[0]


def pool_months(samples, mask, valid):
    """[months, members, days, H, W] restricted to valid hidden cells.

    Padded days fall outside the month, so they must be excluded explicitly:
    the per-day U-Net branch writes zeros there and would otherwise pollute
    every metric.
    """
    out = []
    for i in range(mask.shape[0]):
        hidden = np.broadcast_to(mask[i], samples.shape[2:]) <= 0
        inside = np.broadcast_to(valid[i], samples.shape[2:]) > 0
        sel = (hidden & inside).reshape(-1)
        if sel.any():
            out.append(samples[i].reshape(samples.shape[1], -1)[:, sel].T)
    return np.concatenate(out, axis=0)


# ------------------------------------------------------------------ ensembles
@torch.no_grad()
def diffusion_ensemble(bundle, ids, device, seed, members, chunk_size=8,
                       graph_strength=0.15, boundary_blend=0.25, cache=None):
    if cache is not None and cache.exists():
        with np.load(cache) as d:
            return d["gsr"], d["observed"]
    ck = torch.load(f"outputs/masked_v9_graph/model_{seed}.pt",
                    map_location="cpu", weights_only=False)
    net = MaskedTemporalVelocityNet().to(device)
    net.load_state_dict(ck["model"])
    net.eval()
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

    masks = torch.stack([mask_for(keys[i], valid[i]) for i in ids]).to(device)
    pieces = []
    for a in range(0, len(ids), 6):
        g = list(range(a, min(a + 6, len(ids))))
        ii = torch.as_tensor([ids[j] for j in g], device=device)
        ob = masks[g]
        pieces.append(sample_masked(
            net, td[ii], [keys[ids[j]] for j in g], vd[ii], weather[ii],
            residual[ii] * ob, ob, weights.to(device), float(ck["mean_scale"]),
            scale, base[ii], factor[ii], seed=1000 + seed + int(ids[g[0]]),
            members=members, chunk_size=chunk_size,
            graph_strength=graph_strength, boundary_blend=boundary_blend))
    samples = torch.cat(pieces)
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, gsr=samples.numpy(), observed=masks.cpu().numpy(),
                            months=np.array(keys)[ids], truth=x[ids].numpy(),
                            valid=valid[ids].numpy())
    return samples.numpy(), masks.cpu().numpy()


@torch.no_grad()
def unet_ensemble(bundle, ids, device, cache=None):
    """Per-day U-Net inference at every missing cell; observed values copied."""
    if cache is not None and cache.exists():
        with np.load(cache) as d:
            return d["gsr"], d["observed"]
    x, features, valid, target, weights, keys, lat = (
        bundle["x"], bundle["features"], bundle["valid"], bundle["target"],
        bundle["weights"], bundle["keys"], bundle["lat"])
    train = np.flatnonzero(np.array([int(k[:4]) for k in keys]) <= 2009)
    prior = SolarPrior.fit(lat, x, valid, keys, train)
    base, _ = prior.fields(target.to(device), keys, weights.to(device), valid.to(device))
    masks = torch.stack([mask_for(keys[i], valid[i]) for i in ids]).to(device)
    preds = np.empty((len(ids), N_UNET) + tuple(x.shape[1:]), dtype=np.float32)
    truth = x[ids]
    for m in range(N_UNET):
        ck = torch.load(f"outputs/spatial_unet_ensemble/model_{m}.pt",
                        map_location=device, weights_only=False)
        net = SpatialUNet().to(device)
        # These checkpoints store the bare state_dict, unlike the diffusion ones.
        net.load_state_dict(ck["model"] if "model" in ck else ck)
        net.eval()
        restored = torch.where(masks.cpu() > 0, truth, torch.zeros_like(truth))
        for i, mid in enumerate(ids):
            miss_days = ((valid[mid] > 0) & (masks[i].cpu() <= 0)).any((-1, -2)).nonzero().flatten().tolist()
            for d in miss_days:
                obs = masks[i, d].cpu()
                y = truth[i, d].to(device)
                wd = features[mid, d].to(device)
                monthly = target[mid].mean().expand_as(y) / 300.0
                inp = make_spatial_inputs(y * obs.to(device), obs.to(device), wd, monthly,
                                          base[mid, d])
                p = (net(inp[None])[0] * 400.0).clamp(0.0, 400.0).cpu()
                restored[i, d] = torch.where(obs > 0, truth[i, d], p)
        preds[:, m] = restored.numpy()
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, gsr=preds, observed=masks.cpu().numpy(),
                            months=np.array(keys)[ids], truth=truth.numpy(),
                            valid=valid[ids].numpy())
    return preds, masks.cpu().numpy()


# ----------------------------------------------------------------- calibration
def fuse(su, sd, w, g):
    """Fused ensemble from pooled U-Net and diffusion members.

    The two branches have different ensemble sizes (8 and 12 members). The
    definition concatenates the two centred-deviation blocks **without**
    equalising their counts: the fused mean is then exactly
    ``w * mean(u) + (1 - w) * mean(d)``, because the deviations carry zero
    within-block mean. Repeated members must not be used to pad the counts,
    since that would break the identity.
    """
    um = su.mean(1, keepdims=True)
    dm = sd.mean(1, keepdims=True)
    dev = np.concatenate((w * (su - um), (1 - w) * (sd - dm)), axis=1)
    return w * um + (1 - w) * dm + g * dev


def calibrate(u, d, mask, valid, truth):
    """Search the published grid on 2010 only."""
    su = pool_months(u, mask, valid)
    sd = pool_months(d, mask, valid)
    y = truth[(mask <= 0) & (valid > 0)]
    best = None
    for w in np.linspace(0, 1, 21):
        for g in np.linspace(0.5, 8, 31):
            m = metric(fuse(su, sd, w, g), y)
            obj = m["crps"] + 8 * abs(m["coverage90"] - 0.9)
            if best is None or obj < best["objective"]:
                best = {"w": float(w), "g": float(g), "objective": float(obj), **m}
    um = su.mean(1, keepdims=True)
    ucal = None
    for g in np.linspace(0.5, 8, 31):
        m = metric(um + g * (su - um), y)
        obj = m["crps"] + 8 * abs(m["coverage90"] - 0.9)
        if ucal is None or obj < ucal["objective"]:
            ucal = {"g": float(g), "objective": float(obj), **m}
    return best, ucal


def bootstrap(a_rows, b_rows, draws=2000, seed=20260912):
    rng = np.random.default_rng(seed)
    n = len(a_rows)
    out = []
    for _ in range(draws):
        idx = rng.integers(0, n, n)
        sa = np.concatenate([a_rows[i][0] for i in idx], axis=0)
        ya = np.concatenate([a_rows[i][1] for i in idx], axis=0)
        sb = np.concatenate([b_rows[i][0] for i in idx], axis=0)
        yb = np.concatenate([b_rows[i][1] for i in idx], axis=0)
        out.append([metric(sa, ya)["rmse"] - metric(sb, yb)["rmse"],
                    metric(sa, ya)["crps"] - metric(sb, yb)["crps"]])
    out = np.array(out)
    return {"rmse_95": np.quantile(out[:, 0], [0.025, 0.975]).tolist(),
            "crps_95": np.quantile(out[:, 1], [0.025, 0.975]).tolist(), "draws": draws}


def month_rows_for(samples, mask, valid, truth, transform):
    """Per-month (fused members, truth) pairs for the paired block bootstrap.

    ``samples`` is one branch with shape [months, members, days, H, W] and
    ``transform`` maps a cell-selected (n, members) block to the fused
    ensemble, so the fusion is applied within each month and the month
    structure needed by the bootstrap is preserved.
    """
    rows = []
    for i in range(mask.shape[0]):
        hidden = np.broadcast_to(mask[i], samples.shape[-3:]) <= 0
        inside = np.broadcast_to(valid[i], samples.shape[-3:]) > 0
        sel = (hidden & inside).reshape(-1)
        if sel.any():
            block = samples[i].reshape(samples.shape[1], -1)[:, sel].T
            rows.append((transform(block), truth[i].reshape(-1)[sel]))
    return rows


def main():
    ap = argparse.ArgumentParser()
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
    cal_ids = np.flatnonzero(years == CAL_YEAR)
    conf_ids = np.flatnonzero(np.isin(years, CONFIRM_YEARS))
    print(f"[data] calibration {len(cal_ids)} months ({CAL_YEAR}); "
          f"confirmatory {len(conf_ids)} months {CONFIRM_YEARS}", flush=True)

    payload = {"pre_registration": "paper/PREREGISTRATION_2012_2014.md",
               "calibration_year": CAL_YEAR, "confirm_years": list(CONFIRM_YEARS),
               "mask_seed": MASK_SEED, "members": args.members,
               "diffusion_seeds": list(N_SEEDS), "unet_members": N_UNET,
               "note": "2012-2014 was never evaluated before this run"}

    # ---- calibration period
    du = {}
    for s in N_SEEDS:
        d, obs = diffusion_ensemble(bundle, cal_ids, device, s, args.members,
                                    cache=OUT / f"diff_cal_seed{s}.npz")
        du[s] = d
        do_cal, valid_cal = obs, valid[cal_ids].numpy()
    uu, _ = unet_ensemble(bundle, cal_ids, device, cache=OUT / "unet_cal.npz")
    truth_cal = x[cal_ids].numpy()
    sel, usel = calibrate(uu, du[11], do_cal, valid_cal, truth_cal)
    payload["calibration_2010"] = {"fusion": sel, "unet_only": usel}
    print(f"[calib] w={sel['w']:.2f} gamma={sel['g']:.2f} | unet gamma={usel['g']:.2f}", flush=True)

    # ---- confirmatory period
    dc = {}
    for s in N_SEEDS:
        d, obs = diffusion_ensemble(bundle, conf_ids, device, s, args.members,
                                    cache=OUT / f"diff_conf_seed{s}.npz")
        dc[s] = d
        do_conf, valid_conf = obs, valid[conf_ids].numpy()
    uc, _ = unet_ensemble(bundle, conf_ids, device, cache=OUT / "unet_conf.npz")
    truth_conf = x[conf_ids].numpy()
    hidden_conf = (do_conf <= 0) & (valid_conf > 0)
    y = truth_conf[hidden_conf]

    w, g, gu = sel["w"], sel["g"], usel["g"]
    results = {}
    for s in N_SEEDS:
        su = pool_months(uc, do_conf, valid_conf)
        sd = pool_months(dc[s], do_conf, valid_conf)
        um = su.mean(1, keepdims=True)
        results[f"seed{s}"] = {
            "diffusion": metric(sd, y),
            "unet": metric(su, y),
            "unet_calibrated": metric(um + gu * (su - um), y),
            "fusion": metric(fuse(su, sd, w, g), y),
        }
        print(f"[confirm] seed {s}: fusion RMSE {results[f'seed{s}']['fusion']['rmse']:.3f} "
              f"CRPS {results[f'seed{s}']['fusion']['crps']:.3f} | "
              f"unet_cal RMSE {results[f'seed{s}']['unet_calibrated']['rmse']:.3f} "
              f"CRPS {results[f'seed{s}']['unet_calibrated']['crps']:.3f}", flush=True)

    payload["per_seed_confirmatory"] = results

    # ---- primary comparison at the pre-selected seed
    su = pool_months(uc, do_conf, valid_conf)
    sd = pool_months(dc[11], do_conf, valid_conf)
    um = su.mean(1, keepdims=True)
    fus = fuse(su, sd, w, g)
    ucal = um + gu * (su - um)
    # The fused mean must equal the weighted branch means; checked, not assumed.
    assert np.allclose(fus.mean(1), w * su.mean(1) + (1 - w) * sd.mean(1), atol=1e-6)

    # Per-month fusion for the bootstrap: the two branches are pooled together
    # inside each month so the month blocks survive.
    def month_mask(i):
        hidden = np.broadcast_to(do_conf[i], uc.shape[-3:]) <= 0
        inside = np.broadcast_to(valid_conf[i], uc.shape[-3:]) > 0
        return (hidden & inside).reshape(-1)

    frows, urows = [], []
    for i in range(len(conf_ids)):
        sel = month_mask(i)
        if not sel.any():
            continue
        bu = uc[i].reshape(uc.shape[1], -1)[:, sel].T
        bd = dc[11][i].reshape(dc[11].shape[1], -1)[:, sel].T
        yb = truth_conf[i].reshape(-1)[sel]
        frows.append((fuse(bu, bd, w, g), yb))
        urows.append((bu.mean(1, keepdims=True) + gu * (bu - bu.mean(1, keepdims=True)), yb))
    bs = bootstrap(frows, urows)
    payload["primary"] = {
        "n_cells": int(y.size), "months": int(len(conf_ids)),
        "fusion": metric(fus, y), "unet_calibrated": metric(ucal, y),
        "raw_diffusion": metric(sd, y), "raw_unet": metric(su, y),
        "bootstrap_fusion_minus_unet_cal": bs,
    }
    # ---- decision rules
    p = payload["primary"]
    d_rmse = p["unet_calibrated"]["rmse"] - p["fusion"]["rmse"]
    rel = 100 * d_rmse / p["unet_calibrated"]["rmse"]
    lo, hi = bs["rmse_95"]
    cov = p["fusion"]["coverage90"]
    payload["decisions"] = {
        "C1_point_accuracy": (
            "confirmed" if (hi < 0 and rel >= 2) else
            "partially_confirmed" if hi < 0 else "not_confirmed"),
        "C1_detail": {"delta_rmse": d_rmse, "relative_percent": rel,
                      "bootstrap_95": [lo, hi]},
        "C2_reliability": (
            "confirmed" if 0.87 <= cov <= 0.93 else
            "partially_confirmed" if 0.85 <= cov <= 0.95 else "not_confirmed"),
        "C2_detail": {"fusion_coverage90": cov},
        "C3_not_only_width": (
            "confirmed" if p["fusion"]["crps"] < p["unet_calibrated"]["crps"] else
            "equal" if abs(p["fusion"]["crps"] - p["unet_calibrated"]["crps"]) <= 0.005 * p["unet_calibrated"]["crps"]
            else "not_confirmed"),
        "C3_detail": {"fusion_crps": p["fusion"]["crps"],
                      "unet_cal_crps": p["unet_calibrated"]["crps"]},
        "C4_underdispersion": (
            "confirmed" if (p["raw_diffusion"]["coverage90"] < 0.70 and p["raw_unet"]["coverage90"] < 0.70)
            else "partially_confirmed" if (p["raw_diffusion"]["coverage90"] < 0.80 and p["raw_unet"]["coverage90"] < 0.80)
            else "not_confirmed"),
        "C4_detail": {"raw_diffusion_coverage90": p["raw_diffusion"]["coverage90"],
                      "raw_unet_coverage90": p["raw_unet"]["coverage90"]},
    }

    payload["seconds"] = time.time() - t0
    (OUT / "confirmatory_results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["decisions"], indent=2), flush=True)
    print(f"[done] {payload['seconds']:.1f} s", flush=True)


if __name__ == "__main__":
    main()
