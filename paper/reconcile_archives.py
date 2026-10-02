"""Definitive reconciliation: archived validation ensemble vs manuscript evidence.

Pooling follows audit_numbers.py exactly (per-month boolean selection on the
[months, members, days, H, W] layout). All arrays are [24, members, 31, 10, 10].
"""
from __future__ import annotations

import numpy as np


def pool(samples, mask):
    out = []
    for i in range(mask.shape[0]):
        sel = np.broadcast_to(mask[i], samples.shape[2:]).reshape(-1)
        if not sel.any():
            continue
        out.append(samples[i].reshape(samples.shape[1], -1)[:, sel].T)
    return np.concatenate(out, axis=0)


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
            "cov": float(np.mean((y >= lo) & (y <= hi))),
            "pred_mean": float(mean.mean()), "truth_mean": float(y.mean())}


F = np.load("paper/evidence/final_ensembles.npz")
A = np.load("outputs/masked_v9_graph/masked_ensemble_validation.npz")
FR = np.load("outputs/reproduce_2011/fresh_2011.npz")

rep = F["missing"].copy()
rep[:12] = False
truth = F["truth"]
y = truth[rep]
print(f"reporting cells: {int(rep.sum())}")
print(f"shapes: evidence {F['Diffusion'].shape}  archived {A['gsr'].shape}  fresh {FR['gsr'].shape}")

print("\n== pooled on the evidence reporting mask ==")
for name in ("Diffusion", "UNet", "Fusion", "Neighbor"):
    print(f"  evidence {name:10s}", {k: round(v, 3) for k, v in metric(pool(F[name][12:], rep[12:]), y).items()})
print(f"  archived masked_v9 ", {k: round(v, 3) for k, v in metric(pool(A['gsr'][12:], rep[12:]), y).items()})
print(f"  fresh (this session)", {k: round(v, 3) for k, v in metric(pool(FR['gsr'], rep[12:]), y).items()})

print("\n== mask identity ==")
arch_hidden = (A["observed"][12:] <= 0) & (A["valid"][12:] > 0)
print("  archived hidden == evidence missing:", bool(np.array_equal(arch_hidden, rep[12:])))
print("  archived truth  == evidence truth  :", bool(np.allclose(A["truth"][12:], truth[12:], atol=1e-4)))
print("  fresh observed  == evidence missing:", bool(np.array_equal(FR["observed"] <= 0, A["observed"][12:] <= 0)))
