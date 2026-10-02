"""Reproduce the manuscript's 2011 numbers from the stored masks.

The archived validation ensemble in outputs/masked_v9_graph scores RMSE 72.56
against its own stored mask, while Table 1 reports 24.51 for the same branch.
This script asks whether a fresh sampling run, using the same frozen checkpoint
and the *stored* masks, reproduces the reported number instead.

Usage: python -m dmgsr.reproduce_2011
"""
from __future__ import annotations

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

OUT = Path("outputs/reproduce_2011")
SEED = 11
MEMBERS = 12


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    arch = np.load("outputs/masked_v9_graph/masked_ensemble_validation.npz")
    months = [str(m) for m in arch["months"]]
    stored_obs = arch["observed"]
    stored_valid = arch["valid"]
    stored_truth = arch["truth"]

    # The archive holds 2010-01..2011-12; Table 1 reports the 2011 half.
    ids_2011 = [i for i, m in enumerate(months) if m.startswith("2011")]
    print(f"months in archive: {len(months)}; 2011 half: {len(ids_2011)}", flush=True)

    x, features, valid, target, weights, keys, lat = prepare_masked_data(DATA_ROOT)
    idx = np.array([keys.index(m) for m in months])
    np.testing.assert_allclose(x[idx].numpy(), stored_truth, atol=1e-4)
    np.testing.assert_allclose(valid[idx].numpy(), stored_valid, atol=1e-6)
    print("stored truth and valid masks match the reloaded data", flush=True)

    ck = torch.load(f"outputs/masked_v9_graph/model_{SEED}.pt",
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

    # `sel` holds the global month indices for the 2011 half; the same array
    # must index the tensors, the keys and the masks, otherwise the sampling
    # silently runs on the wrong months.
    sel = np.array([idx[i] for i in ids_2011])
    sel_keys = [keys[i] for i in sel]
    ii = torch.as_tensor(sel, device=device)
    masks = torch.from_numpy(stored_obs[ids_2011]).to(device)
    with torch.no_grad():
        samples = sample_masked(
            net, td[ii], sel_keys, vd[ii], weather[ii],
            residual[ii] * masks, masks, weights.to(device), float(ck["mean_scale"]),
            scale, base[ii], factor[ii], seed=4242, members=MEMBERS).cpu().numpy()

    hidden = (stored_obs[ids_2011] <= 0) & (stored_valid[ids_2011] > 0)
    y = stored_truth[ids_2011][hidden]
    pooled = np.concatenate([
        samples[i].reshape(MEMBERS, -1)[:, hidden[i].reshape(-1)].T
        for i in range(len(ids_2011))], axis=0)

    fresh = metric(pooled, y)
    # Same masks, archived ensemble, for the direct comparison.
    arch_pool = np.concatenate([
        arch["gsr"][ids_2011][i].reshape(MEMBERS, -1)[:, hidden[i].reshape(-1)].T
        for i in range(len(ids_2011))], axis=0)
    archived = metric(arch_pool, y)

    report = {
        "split": "2011 stored masks",
        "hidden_cells": int(y.size),
        "hidden_fraction_of_valid": float(hidden.sum() / (stored_valid[ids_2011] > 0).sum()),
        "fresh_sampling": fresh,
        "archived_ensemble": archived,
        "manuscript_table1_diffusion": {"rmse": 24.51, "mae": 18.81, "crps": 14.46,
                                        "coverage90": 0.589, "width90": 40.11},
        "conclusion": (
            "fresh sampling reproduces the manuscript" if abs(fresh["rmse"] - 24.51) < 1.5
            else "fresh sampling does NOT reproduce the manuscript headline"),
    }
    (OUT / "reproduce.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    np.savez_compressed(OUT / "fresh_2011.npz", gsr=samples,
                        observed=stored_obs[ids_2011], truth=stored_truth[ids_2011],
                        valid=stored_valid[ids_2011], months=np.array(months)[ids_2011])


if __name__ == "__main__":
    main()
