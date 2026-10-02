"""Audit every recomputable numeric claim in Manuscript.tex against paper/evidence/.

Recomputes each reported quantity from the saved ensemble archives and compares
it with the value printed in the manuscript. Read-only: nothing is written.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EV = ROOT / "paper" / "evidence"

results = json.loads((EV / "results.json").read_text(encoding="utf-8"))
A = np.load(EV / "final_ensembles.npz")
truth = A["truth"]
missing = A["missing"]
months = A["months"]

CHECKS: list = []
NOTES: list = []


def check(label, claimed, got, tol=0.01):
    CHECKS.append((label, float(claimed), float(got), tol))


def note(text):
    NOTES.append(text)


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
    }


def pool(samples, mask):
    """samples [n, members, H, W] -> [cells, members] restricted to true `mask` entries.

    The mask is per month, so it is applied month by month.
    """
    out = []
    for i in range(mask.shape[0]):
        sel = np.broadcast_to(mask[i], samples.shape[2:]).reshape(-1)
        if not sel.any():
            continue
        out.append(samples[i].reshape(samples.shape[1], -1)[:, sel].T)  # [cells, members]
    return np.concatenate(out, axis=0)


cal_mask = missing.copy()
cal_mask[12:] = False
rep_mask = missing.copy()
rep_mask[:12] = False
print(f"months={len(months)}  calibration cells={int(cal_mask.sum())}  reporting cells={int(rep_mask.sum())}")
check("calibration cells = 415", 415, int(cal_mask.sum()), 0)
check("reporting cells = 547", 547, int(rep_mask.sum()), 0)
note(f"months in archive: {months[:2].tolist()} ... {months[-2:].tolist()}")

KEYS = ["Neighbor", "Laplacian", "Diffusion", "UNet", "UNet_cal", "Fusion"]
for k in KEYS:
    note(f"{k:10s} archive shape {A[k].shape}")

rec = {k: metric(pool(A[k], rep_mask), truth[rep_mask]) for k in KEYS}

TABLE = {
    "Neighbour propagation": ("Neighbor", (33.89, 22.86, 22.86)),
    "Laplacian smoothing": ("Laplacian", (24.25, 18.02, 18.02)),
    "Diffusion": ("Diffusion", (24.51, 18.81, 14.46, 58.9, 40.11)),
    "U-Net": ("UNet", (20.84, 15.96, 12.27, 58.5, 33.06)),
    "Calibrated U-Net": ("UNet_cal", (20.84, 15.96, 12.34, 85.4, 66.11)),
    "Fusion": ("Fusion", (19.77, 15.15, 11.24, 93.2, 81.83)),
}
print("\n== Table 1 ==")
for label, (key, c) in TABLE.items():
    r = rec[key]
    got = [r["rmse"], r["mae"], r["crps"]]
    if len(c) == 5:
        got += [r["coverage90"] * 100, r["width90"]]
    fs = ["RMSE", "MAE", "CRPS", "cov%", "width"]
    for i, v in enumerate(got):
        check(f"{label} {fs[i]}", c[i], v, 0.1 if fs[i] == "cov%" else 0.01)
    print(f"  {label:24s} claimed {c}")
    print(f"  {'':24s} recomputed {[round(v, 3) for v in got]}")

fus, ucal, dif, unet = (rec[k] for k in ("Fusion", "UNet_cal", "Diffusion", "UNet"))
check("fusion-minus-unet_cal RMSE = 1.07", 1.07, ucal["rmse"] - fus["rmse"], 0.02)
check("fusion RMSE gain = 5.1%", 5.1, 100 * (ucal["rmse"] - fus["rmse"]) / ucal["rmse"], 0.1)
check("fusion CRPS gain = 8.9%", 8.9, 100 * (ucal["crps"] - fus["crps"]) / ucal["crps"], 0.1)
check("fusion > calibrated width/23.8%", 23.8, 100 * (fus["width90"] / ucal["width90"] - 1), 0.2)
check("raw U-Net CRPS < calibrated (1=yes)", 1, 1 if unet["crps"] < ucal["crps"] else 0, 0)
check("Laplacian RMSE 24.25", 24.25, rec["Laplacian"]["rmse"], 0.01)
check("Diffusion RMSE 24.51", 24.51, rec["Diffusion"]["rmse"], 0.01)
check("U-Net RMSE 20.84", 20.84, rec["UNet"]["rmse"], 0.01)

boot = A["bootstrap"]
check("bootstrap RMSE 2.5% = -2.04", -2.04, float(np.quantile(boot[:, 0], 0.025)), 0.02)
check("bootstrap RMSE 97.5% = -0.36", -0.36, float(np.quantile(boot[:, 0], 0.975)), 0.02)
check("bootstrap CRPS 2.5% = -1.66", -1.66, float(np.quantile(boot[:, 1], 0.025)), 0.02)
check("bootstrap CRPS 97.5% = -0.55", -0.55, float(np.quantile(boot[:, 1], 0.975)), 0.02)

check("fusion w = 0.80", 0.80, float(results["fusion_selection"]["w"]), 0.001)
check("fusion gamma = 4.25", 4.25, float(results["fusion_selection"]["g"]), 0.001)
check("U-Net-only gamma = 2.00", 2.00, float(results["unet_calibration"]["g"]), 0.001)

fus_pool = pool(A["Fusion"], rep_mask)
neg_pct = float((fus_pool < 0).mean()) * 100
check("negative fused members = 0.95%", 0.95, neg_pct, 0.01)
check("fused member values = 10940", 10940, fus_pool.size, 0)

print("\n== fixed case and trace ==")
fc = np.load(EV / "fixed_case_recalibrated.npz")
fm = np.load(EV / "fixed_case_members.npz")
tr = np.load(EV / "diffusion_trace.npz")
note(f"fixed_case_members keys: {list(fm.keys())}")
note(f"diffusion_trace keys: {list(tr.keys())}")
if "missing" in fm:
    check("fixed-case missing values = 112", 112, int(fm["missing"].sum()), 0)

# ---------------------------------------------------------------- helpers
print("\n== checks ==")
fails = 0
for label, claimed, got, tol in CHECKS:
    ok = abs(claimed - got) <= tol
    fails += 0 if ok else 1
    print(f"  [{'OK ' if ok else 'BAD'}] {label:38s} claimed {claimed:>9.4f}  recomputed {got:>9.4f}  tol {tol}")
print(f"\n{len(CHECKS)} numeric checks, {fails} mismatch(es)")
print("\n== notes ==")
for t in NOTES:
    print("  " + t)
