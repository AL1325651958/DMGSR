"""Is the reported 2011 diffusion score a stable quantity or a single draw?

The manuscript reports one sampling realisations of the seed-11 checkpoint on the
stored 2011 masks: RMSE 24.51. A fresh realisation with the same checkpoint and
the same masks gives 70.49. This script repeats the sampling under several seeds
and under the exact batching of paper/build_evidence.py, and reports the spread.

Usage: python -m dmgsr.seed_variance --members 12
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .confirm import metric
from .masked_temporal import (
    DATA_ROOT,
    MaskedTemporalVelocityNet,
    prepare_masked_data,
    sample_masked,
)
from .solar_prior import SolarPrior

OUT = Path("outputs/seed_variance")
CKPT = 11


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--members", type=int, default=12)
    ap.add_argument("--seeds", type=int, nargs="+",
                    default=[4242, 11, 99, 20260912, 20260916, 20260920])
    ap.add_argument("--chunk-size", type=int, default=8)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    arch = np.load("outputs/masked_v9_graph/masked_ensemble_validation.npz")
    months = [str(m) for m in arch["months"]]
    ids_2011 = [i for i, m in enumerate(months) if m.startswith("2011")]
    stored_obs = arch["observed"][ids_2011]
    stored_valid = arch["valid"][ids_2011]
    stored_truth = arch["truth"][ids_2011]
    hidden = (stored_obs <= 0) & (stored_valid > 0)
    y = stored_truth[hidden]

    x, features, valid, target, weights, keys, lat = prepare_masked_data(DATA_ROOT)
    ids = np.array([keys.index(m) for m in months])[ids_2011]
    ck = torch.load(f"outputs/masked_v9_graph/model_{CKPT}.pt",
                    map_location="cpu", weights_only=False)
    physics = SolarPrior(ck["latitude"], ck["template"])
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
    masks = torch.from_numpy(stored_obs).to(device)

    def run(batching, seed_base):
        chunks = []
        groups = ([[list(range(len(ids)))] if batching == "all"
                   else [list(r) for r in np.array_split(np.arange(len(ids)), 3)]])
        for group in groups[0]:
            ii = torch.as_tensor(ids[group], device=device)
            ob = masks[group]
            chunks.append(sample_masked(
                net, td[ii], [keys[i] for i in ids[group]], vd[ii], weather[ii],
                residual[ii] * ob, ob, weights.to(device), float(ck["mean_scale"]),
                scale, base[ii], factor[ii],
                seed=seed_base + int(ids[group][0]), members=args.members,
                chunk_size=args.chunk_size))
        return torch.cat(chunks).cpu().numpy()

    rows = []
    for batching in ("all", "split3"):
        for sb in args.seeds:
            s = run(batching, sb)
            pooled = np.concatenate([
                s[i].reshape(args.members, -1)[:, hidden[i].reshape(-1)].T
                for i in range(len(ids))], axis=0)
            m = metric(pooled, y)
            rows.append({"batching": batching, "seed_base": sb, **m})
            print(f"[{batching:6s} seed={sb:>9d}] rmse {m['rmse']:7.3f} mae {m['mae']:7.3f} "
                  f"crps {m['crps']:7.3f} cov {m['coverage90']:.3f}", flush=True)

    rm = np.array([r["rmse"] for r in rows])
    cr = np.array([r["crps"] for r in rows])
    summary = {
        "members": args.members, "checkpoint_seed": CKPT, "hidden_cells": int(y.size),
        "manuscript_table1": {"rmse": 24.51, "crps": 14.46},
        "rmse": {"mean": float(rm.mean()), "sd": float(rm.std()), "min": float(rm.min()),
                 "max": float(rm.max())},
        "crps": {"mean": float(cr.mean()), "sd": float(cr.std()), "min": float(cr.min()),
                 "max": float(cr.max())},
        "runs": rows,
    }
    (OUT / "seed_variance.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\n" + json.dumps({k: v for k, v in summary.items() if k != "runs"}, indent=2))


if __name__ == "__main__":
    main()
