"""Audit the trace-based and figure-embedded claims in Manuscript.tex.

Recomputes: Fig. 8 trajectory RMSE/coverage after each chunk, Fig. 3 pooled
RMSE/CRPS, Fig. 4 reliability and width, Fig. 6 monthly RMSE, Fig. 7 fused
block series. Read-only.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EV = ROOT / "paper" / "evidence"

CHECKS: list = []


def check(label, claimed, got, tol=0.02):
    CHECKS.append((label, float(claimed), float(got), tol))


# ------------------------------------------------------------------ Fig. 8
tr = np.load(EV / "diffusion_trace.npz")
stages = tr["stages"]
truth = tr["truth"]
obs = tr["observed"]
valid = tr["valid"]
miss = tr["missing"]
print("trace stages shape:", stages.shape, " truth:", truth.shape, " missing:", int(miss.sum()))

# stages: [n_stages, members, days, H, W] over the fixed case month(s)
rmse_series, cov_series = [], []
for i in range(stages.shape[0]):
    s = stages[i]  # [members, days, H, W]
    mean = s.mean(0)
    rmse_series.append(float(np.sqrt(np.mean((mean[miss] - truth[miss]) ** 2))))
    lo = np.quantile(s, 0.05, axis=0)
    hi = np.quantile(s, 0.95, axis=0)
    cov_series.append(float(np.mean((truth[miss] >= lo[miss]) & (truth[miss] <= hi[miss]))))

print("\n== Fig. 8 trajectory ==")
for i, (r, c) in enumerate(zip(rmse_series, cov_series)):
    print(f"  stage {i:2d}: RMSE {r:8.3f}   coverage90 {c:.4f}")

check("trajectory RMSE after 8 steps = 48.31", 48.31, rmse_series[0])
check("trajectory RMSE after 40 steps = 42.72", 42.72, rmse_series[4])
check("trajectory RMSE after 64 steps = 44.67", 44.67, rmse_series[7])
check("final processing RMSE = 37.70", 37.70, rmse_series[8])
check("coverage before final processing = 60.7%", 60.7, cov_series[7] * 100, 0.1)
check("coverage after final processing = 44.6%", 44.6, cov_series[8] * 100, 0.1)
print(f"  stages: {len(rmse_series)}; last two are the constraint/spatial postprocessing segments")

# ------------------------------------------------------------------ Fig. 7
fm = np.load(EV / "fixed_case_members.npz")
fmiss = fm["missing"]
check("fixed-case values = 112", 112, int(fmiss.sum()), 0)
print("\n== Fig. 7 fixed case ==")
for k in fm.files:
    a = fm[k]
    if hasattr(a, "shape"):
        print(f"  {k:16s} {a.shape}")

# ------------------------------------------------------------------ Fig. 3/4
A = np.load(EV / "final_ensembles.npz")
truth2 = A["truth"]
missing2 = A["missing"]
rep = missing2.copy()
rep[:12] = False


def pool(samples, mask):
    out = []
    for i in range(mask.shape[0]):
        sel = np.broadcast_to(mask[i], samples.shape[2:]).reshape(-1)
        if not sel.any():
            continue
        out.append(samples[i].reshape(samples.shape[1], -1)[:, sel].T)
    return np.concatenate(out, axis=0)


print("\n== Fig. 3 (pooled skill on 547 values) ==")
for k in ("Neighbor", "Laplacian", "Diffusion", "UNet", "UNet_cal", "Fusion"):
    s = pool(A[k], rep)
    y = truth2[rep]
    n = s.shape[1]
    order = np.sort(s, axis=1)
    coeff = (2 * np.arange(1, n + 1) - n - 1)[None, :]
    crps = np.abs(s - y[:, None]).mean(1) - (coeff * order).sum(1) / n**2
    print(f"  {k:10s} RMSE {np.sqrt(np.mean((s.mean(1)-y)**2)):8.3f}  CRPS {crps.mean():8.3f}")

print("\n== Fig. 4 (reliability, nominal 0.1..0.9) ==")
levels = np.arange(0.1, 1.0, 0.1)
for k in ("Diffusion", "UNet", "UNet_cal", "Fusion"):
    s = pool(A[k], rep)
    y = truth2[rep]
    cov = []
    for p in levels:
        lo = np.quantile(s, 0.5 - p / 2, axis=1)
        hi = np.quantile(s, 0.5 + p / 2, axis=1)
        cov.append(float(np.mean((y >= lo) & (y <= hi))))
    print(f"  {k:10s} " + " ".join(f"{c:.3f}" for c in cov))

print("\n== Fig. 6 (monthly RMSE on reporting months) ==")
months = A["months"]
for i in range(12, 24):
    row = []
    for k in ("Diffusion", "UNet", "Fusion"):
        sel = missing2[i].reshape(-1)
        if not sel.any():
            row.append(None)
            continue
        s = A[k][i].reshape(A[k].shape[1], -1)[:, sel].T
        y = truth2[i].reshape(-1)[sel]
        row.append(float(np.sqrt(np.mean((s.mean(1) - y) ** 2))))
    print(f"  {months[i]}  n={int(missing2[i].sum()):3d}  diffusion {row[0]:7.3f}  unet {row[1]:7.3f}  fusion {row[2]:7.3f}")

print("\n== checks ==")
fails = 0
for label, claimed, got, tol in CHECKS:
    ok = abs(claimed - got) <= tol
    fails += 0 if ok else 1
    print(f"  [{'OK ' if ok else 'BAD'}] {label:52s} claimed {claimed:>8.2f}  recomputed {got:>8.2f}")
print(f"\n{len(CHECKS)} checks, {fails} mismatch(es)")
