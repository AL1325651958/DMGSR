"""Cross-check per-month RMSE three ways: archive recomputation, the archived
monthly_metrics.csv, and the definition used in paper/build_evidence.py.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EV = ROOT / "paper" / "evidence"
A = np.load(EV / "final_ensembles.npz")
truth = A["truth"]
missing = A["missing"]
months = A["months"]

rows = list(csv.DictReader((EV / "monthly_metrics.csv").open()))
print("csv columns:", list(rows[0].keys()))

# ---------------------------------------------------------------- way 1
# Exact analogue of build_evidence.metric: for one month, samples are indexed
# as s[i][:, miss[i]] with s[i] of shape [members, days, H, W].
def month_rmse_archive(key, name):
    i = list(months).index(key)
    s = A[name][i][:, missing[i]]          # [members, cells]
    y = truth[i][missing[i]]
    return float(np.sqrt(np.mean((s.mean(0) - y) ** 2)))


# ---------------------------------------------------------------- way 2
def month_rmse_bug(key, name):
    """The variant that produced the suspicious numbers."""
    i = list(months).index(key)
    s = A[name][i:i + 1].reshape(1, A[name].shape[1], -1)
    sel = np.broadcast_to(missing[i], A[name].shape[2:]).reshape(-1)
    ss = s[0][:, sel]
    yy = truth[i:i + 1].reshape(-1)[sel]
    return float(np.sqrt(np.mean((ss.mean(0) - yy) ** 2)))


print(f"\n{'month':9s} {'n':>4s} {'way1':>9s} {'way2':>9s} {'csvDiff':>9s} {'csvUNet':>9s} {'csvFus':>9s}")
for i in range(12, 24):
    key = str(months[i])
    n = int(missing[i].sum())
    csvvals = {}
    for m in ("Diffusion", "UNet", "Fusion"):
        v = [r for r in rows if r["method"] == m and r["month"] == key]
        csvvals[m] = float(v[0]["rmse"]) if v else float("nan")
    print(f"{key:9s} {n:4d} {month_rmse_archive(key,'Diffusion'):9.4f} "
          f"{month_rmse_bug(key,'Diffusion'):9.4f} "
          f"{csvvals['Diffusion']:9.4f} {csvvals['UNet']:9.4f} {csvvals['Fusion']:9.4f}")

print("\n== interpretation ==")
print("way2 uses truth[i:i+1].reshape(-1)[sel]; for i=12 that is truth[12:13]")
print("because truth[i] was reassigned inside the loop in the buggy snippet,")
print("so the wrong month was compared against the mask. way1 mirrors the CSV.")
for m in ("Diffusion", "UNet", "Fusion"):
    v = [float(r["rmse"]) for r in rows if r["method"] == m]
    print(f"  csv {m:10s} equal-month mean over 24 months = {np.mean(v):.4f}")
    print(f"  archive {m:10s} equal-month mean over 24 months = "
          f"{np.mean([month_rmse_archive(str(months[i]), m) for i in range(24)]):.4f}")
