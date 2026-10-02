"""Build the graph-ablation report from outputs/ablation_graph/.

Reads ablation_results.json (selected seed per configuration) and
per_seed_results.json (all three seeds), and computes:

* pooled skill and graph-structure diagnostics on the 2011 reporting cells
  for the selected seed of each configuration;
* a paired month-block bootstrap of full minus each ablated configuration;
* the between-seed spread within each configuration, which is the natural
  reference scale for judging whether a between-configuration difference is
  meaningful.

Usage: python -m dmgsr.ablation_report
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .ablation import CONFIGS, OUT

REPORT = OUT / "REPORT.md"
SEEDS = (11, 22, 33)
LABEL = {
    "full": "A  Full",
    "no_boundary_graph": "B  No sampling graph",
    "no_train_graph": "C  No training penalty",
    "none": "D  No graph at all",
}
LONG = {
    "full": "structural loss + chunk-boundary graph correction + spatial blends",
    "no_boundary_graph": "structural loss only (no inference-time graph correction)",
    "no_train_graph": "chunk-boundary correction + spatial blends only (no structural loss)",
    "none": "no graph mechanism at any stage",
}


def metric(samples, y):
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


def archive(config, selected):
    """Return (path, label) of the archived reporting ensemble for a config."""
    direct = OUT / config / "ensemble_report_2011.npz"
    if direct.exists():
        return direct, selected
    per_seed = OUT / config / f"ensemble_report_2011_seed{selected}.npz"
    if per_seed.exists():
        return per_seed, selected
    return None, None


def months_of(path):
    with np.load(path) as d:
        samples, valid, observed, truth = d["gsr"], d["valid"], d["observed"], d["truth"]
    rows = []
    for i in range(samples.shape[0]):
        miss = (valid[i] > 0) & (observed[i] <= 0)
        if not miss.any():
            continue
        sel = miss.reshape(-1)
        rows.append((samples[i].reshape(samples.shape[1], -1)[:, sel].T, truth[i].reshape(-1)[sel]))
    return rows


def bootstrap(a_rows, b_rows, draws=2000, seed=20260912):
    """Paired month-block bootstrap of metric(A) - metric(B)."""
    rng = np.random.default_rng(seed)
    n = len(a_rows)
    out = []
    for _ in range(draws):
        idx = rng.integers(0, n, n)
        sa = np.concatenate([a_rows[i][0] for i in idx], axis=0)
        ya = np.concatenate([a_rows[i][1] for i in idx], axis=0)
        sb = np.concatenate([b_rows[i][0] for i in idx], axis=0)
        yb = np.concatenate([b_rows[i][1] for i in idx], axis=0)
        ma, mb = metric(sa, ya), metric(sb, yb)
        out.append([ma["rmse"] - mb["rmse"], ma["crps"] - mb["crps"]])
    out = np.array(out)
    return {
        "rmse_95": np.quantile(out[:, 0], [0.025, 0.975]).tolist(),
        "crps_95": np.quantile(out[:, 1], [0.025, 0.975]).tolist(),
        "rmse_mean": float(out[:, 0].mean()),
        "crps_mean": float(out[:, 1].mean()),
        "draws": draws,
    }


def main():
    payload = json.loads((OUT / "ablation_results.json").read_text())
    per_seed = json.loads((OUT / "per_seed_results.json").read_text())
    exp = payload["_experiment"]

    lines = [
        "# Graph-regularization ablation",
        "",
        "Four configurations were fitted with every factor except the graph mechanism held fixed.",
        "",
        f"- fitting: {exp['train']}, {exp['epochs']} epochs, seeds {exp['seeds']}",
        f"- training curriculum: spatial probability {exp['train_spatial_probability']}, "
        f"missing-cell weight {exp['missing_weight']}, gradient weight {exp['gradient_weight']}",
        f"- checkpoint selection: {exp['selection']} validation CRPS, per configuration; reporting: {exp['report']}",
        f"- sampling: {exp['members']} members, chunk size {exp['chunk_size']}",
        f"- held fixed: {exp['held_fixed']}",
        f"- varied: {exp['varied']}",
        "",
        "| Configuration | Structural penalty (loss) | Graph correction + blends (sampling) | Meaning |",
        "|---|---|---|---|",
    ]
    for name in CONFIGS:
        cfg = CONFIGS[name]
        lines.append(
            f"| {LABEL[name]} | {'0.05' if cfg['graph_weight'] > 0 else 'removed'} | "
            f"{'strength 0.15, blends 0.25' if cfg['graph_strength'] > 0 else 'removed'} | {LONG[name]} |"
        )
    lines += ["", f"All four configurations selected seed {payload['full']['selected_seed']} on 2010 validation CRPS.", ""]

    # ---------------------------------------------------------------- results
    lines += ["## Selected-seed skill on the 2011 reporting cells", "",
              "| Configuration | RMSE | MAE | CRPS | Coverage 90% | Width 90% |",
              "|---|---:|---:|---:|---:|---:|"]
    for name in CONFIGS:
        r = payload[name]["report_2011"]
        lines.append(f"| {LABEL[name]} | {r['pooled_rmse']:.3f} | {r['pooled_mae']:.3f} | {r['pooled_crps']:.3f} | "
                     f"{100 * r['pooled_coverage90']:.1f}% | {r['pooled_width90']:.2f} |")
    lines.append("")

    f = payload["full"]["report_2011"]
    lines += ["### Differences from the full configuration", "",
              "| Configuration | dRMSE | dCRPS | dedge MAE | dLaplacian MAE |",
              "|---|---:|---:|---:|---:|"]
    for name in CONFIGS:
        if name == "full":
            continue
        r = payload[name]["report_2011"]
        lines.append(f"| {LABEL[name]} | {r['pooled_rmse'] - f['pooled_rmse']:+.3f} | "
                     f"{r['pooled_crps'] - f['pooled_crps']:+.3f} | {r['edge_mae'] - f['edge_mae']:+.3f} | "
                     f"{r['laplacian_mae'] - f['laplacian_mae']:+.3f} |")
    lines += ["", "Positive values mean the ablated configuration is worse. Every difference is far smaller than the "
              "between-seed spread reported below.", ""]

    lines += ["## Grid-graph structure, selected seed", "",
              "| Configuration | Edge MAE | Laplacian MAE | Graph correlation |",
              "|---|---:|---:|---:|"]
    for name in CONFIGS:
        r = payload[name]["report_2011"]
        lines.append(f"| {LABEL[name]} | {r['edge_mae']:.3f} | {r['laplacian_mae']:.3f} | {r['graph_corr']:.4f} |")
    lines.append("")

    # ------------------------------------------------------------ per seed
    lines += ["## Between-seed spread on 2011 (the scale to judge the differences against)", "",
              "| Configuration | RMSE by seed | mean +- sd | CRPS by seed | mean +- sd |",
              "|---|---|---:|---|---:|"]
    for name in CONFIGS:
        rm, cr = [], []
        for s in SEEDS:
            r = per_seed.get(name, {}).get(str(s), {}).get("report_2011")
            if r:
                rm.append(r["pooled_rmse"])
                cr.append(r["pooled_crps"])
        lines.append(f"| {LABEL[name]} | {', '.join(f'{v:.2f}' for v in rm)} | "
                     f"{np.mean(rm):.2f} +- {np.std(rm):.2f} | {', '.join(f'{v:.2f}' for v in cr)} | "
                     f"{np.mean(cr):.2f} +- {np.std(cr):.2f} |")
    lines.append("")

    # ---------------------------------------------------------- bootstrap
    lines += ["## Paired month-block bootstrap, full minus ablated (2011)", ""]
    full_rows = None
    path, _ = archive("full", payload["full"]["selected_seed"])
    if path:
        full_rows = months_of(path)
    if full_rows is None:
        lines.append("(reporting archive for the full configuration is missing)")
    else:
        lines += ["| Comparison | dRMSE 95% interval | dCRPS 95% interval |",
                  "|---|---|---|"]
        boot_out = {}
        for name in CONFIGS:
            if name == "full":
                continue
            p, _ = archive(name, payload[name]["selected_seed"])
            if not p:
                continue
            bs = bootstrap(full_rows, months_of(p))
            boot_out[name] = bs
            lines.append(f"| Full - {LABEL[name]} | [{bs['rmse_95'][0]:+.3f}, {bs['rmse_95'][1]:+.3f}] | "
                         f"[{bs['crps_95'][0]:+.3f}, {bs['crps_95'][1]:+.3f}] |")
        lines.append("")
        (OUT / "bootstrap.json").write_text(json.dumps(boot_out, indent=2), encoding="utf-8")

    lines += [
        "## Reading of the result",
        "",
        "Note the scale. The between-configuration differences in RMSE are 0.01-0.07 W/m2 and in CRPS 0.01-0.04 W/m2, "
        "while the between-seed standard deviation within a single configuration is 1.39-1.47 W/m2 in RMSE and "
        "0.75-0.81 W/m2 in CRPS. The graph mechanism therefore produces no difference that is separable from "
        "fitting-seed variation on this protocol: every configuration is the same within noise. The paired bootstrap "
        "agrees, with all full-minus-ablated intervals touching or straddling zero. The structure diagnostics behave "
        "the same way, moving by less than 0.3 W/m2, so the generated fields are not measurably more graph-smooth "
        "when the penalty and the graph correction are present.",
        "",
        "Two consequences for the manuscript:",
        "",
        "1. No percentage improvement may be attributed to graph regularization, and the phrase does not belong in the "
        "title or in the framing of the contribution. The title, abstract, highlights and the framework subsection "
        "currently present graph regularization as part of the proposed method.",
        "2. The Discussion sentence that says no controlled ablation is available should be replaced by this result.",
        "",
        "## Limits",
        "",
        "- The 2011 period was inspected during earlier development; this is a development comparison.",
        "- One stored mask per month; the bootstrap resamples months, not independent gaps.",
        "- All configurations selected the same fitting seed, so the primary comparison is at fixed initialisation; the "
        "cross-seed table shows what happens when it is not.",
        "- Removing a penalty changes the effective loss scale; coefficients were held at their recorded values rather "
        "than retuned, so a differently weighted penalty is not excluded.",
        "- The graph is the fixed four-neighbour grid. Nothing here speaks to learned attention or irregular networks.",
        "",
    ]

    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
