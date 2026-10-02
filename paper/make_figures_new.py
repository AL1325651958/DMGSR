"""Two new figures carried by this session's experiments.

Fig09  Missingness-regime matrix and the applicability boundary: where the
       diffusion branch wins and where the spatial U-Net does.
Fig10  Sampling-mechanism ladder: the chunked correction, its timing and its
       gain form, against the graph ablation.

Both follow the layout contract of make_figures.py (183-mm width, Arial,
editable vector text, panel-alignment audit).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib as mpl
import numpy as np

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "paper"
OUT = P / "figures"
sys.path.insert(0, "C:/Users/13256/.agents/skills/nature-figure/scripts")
from audit_panel_alignment import require_matplotlib_panel_alignment  # noqa: E402

mpl.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial"], "font.size": 7,
    "axes.titlesize": 8, "axes.labelsize": 7, "xtick.labelsize": 6,
    "ytick.labelsize": 6, "legend.fontsize": 6, "pdf.fonttype": 42,
    "svg.fonttype": "none", "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "savefig.facecolor": "white", "axes.linewidth": 0.6,
})

C = {"diffusion": "#9282AA", "unet": "#C29B69", "accent": "#D55E00", "ink": "#202D42"}
MODES = ["temporal", "spatial", "mixed", "endpoint"]
BANDS = ["1-3", "4-7", "8-14"]


def label(ax, s):
    ax.annotate(s, xy=(0, 1), xytext=(-16, 8), textcoords="offset points",
                xycoords="axes fraction", weight="bold", fontsize=9)


def finish(fig, num, slug):
    stem = f"Fig{num:02d}_{slug}"
    require_matplotlib_panel_alignment(fig, json_out=OUT / f"{stem}.alignment.json", strict=True)
    for ext in ("pdf", "svg"):
        fig.savefig(OUT / f"{stem}.{ext}")
    fig.savefig(OUT / f"{stem}.png", dpi=300)
    plt.close(fig)
    print(f"{stem} done", flush=True)


def load_matrix():
    d = json.loads((ROOT / "outputs" / "missingness_matrix" / "matrix_validation.json").read_text())
    out = {}
    for c in d["cells"]:
        out[(c["mode"], f"{c['band'][0]}-{c['band'][1]}")] = c
    return out, d


def fig9():
    cells, meta = load_matrix()
    f, axes = plt.subplots(1, 4, figsize=(7.2, 2.9),
                           gridspec_kw={"left": .085, "right": .985, "bottom": .26,
                                        "top": .78, "wspace": .86})

    # (a) RMSE difference map, diffusion minus U-Net
    ax = axes[0]
    diff = np.array([[cells[(m, b)]["diffusion"]["rmse"] - cells[(m, b)]["unet"]["rmse"]
                      for b in BANDS] for m in MODES])
    vmax = np.abs(diff).max()
    im = ax.imshow(diff, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set(xticks=range(3), xticklabels=BANDS, yticks=range(4),
           yticklabels=[m.capitalize() for m in MODES],
           xlabel="Gap duration (days)", title="Diffusion $-$ U-Net")
    for i in range(4):
        for j in range(3):
            ax.text(j, i, f"{diff[i, j]:+.1f}", ha="center", va="center", fontsize=6,
                    color="white" if abs(diff[i, j]) > .6 * vmax else C["ink"])
    cb = f.add_axes([.085, .075, .20, .020])
    cb.set_label("<colorbar>")
    f.colorbar(im, cax=cb, orientation="horizontal").set_label("RMSE difference (W/m$^2$)", fontsize=6)

    # (b) RMSE against gap duration
    ax = axes[1]
    h = []
    for m, col in (("temporal", C["diffusion"]), ("spatial", C["unet"])):
        for branch, ls in (("diffusion", "-"), ("unet", "--")):
            y = [cells[(m, b)][branch]["rmse"] for b in BANDS]
            h.append(ax.plot(range(3), y, ls, marker="o", ms=3, color=col,
                             label=f"{m.capitalize()} / {'diffusion' if branch == 'diffusion' else 'U-Net'}")[0])
    ax.set(xticks=range(3), xticklabels=BANDS, xlabel="Gap duration (days)",
           ylabel="RMSE (W/m$^2$)", title="Duration sensitivity", ylim=(20, 56))
    # Top-left keeps the legend clear of both families of curves.
    ax.legend(handles=h, frameon=False, loc="upper left", fontsize=5.5, handlelength=1.4,
              borderaxespad=.1)

    # (c) CRPS by regime
    ax = axes[2]
    x = np.arange(4)
    w = .36
    ax.bar(x - w / 2, [np.mean([cells[(m, b)]["diffusion"]["crps"] for b in BANDS]) for m in MODES],
           w, color=C["diffusion"], label="Diffusion")
    ax.bar(x + w / 2, [np.mean([cells[(m, b)]["unet"]["crps"] for b in BANDS]) for m in MODES],
           w, color=C["unet"], label="U-Net")
    ax.set(xticks=x, xticklabels=["Temp.", "Spat.", "Mixed", "Endpt."],
           ylabel="CRPS (W/m$^2$)", title="CRPS by regime", ylim=(0, 39))
    ax.legend(frameon=False, loc="upper center", fontsize=5.5, ncol=2,
              borderaxespad=.05, columnspacing=1.0, handlelength=1.2)

    # (d) coverage
    ax = axes[3]
    for m, col in (("temporal", C["diffusion"]), ("spatial", C["unet"])):
        ax.plot(range(3), [100 * cells[(m, b)]["diffusion"]["coverage90"] for b in BANDS],
                "-o", ms=3, color=C["diffusion"])
        ax.plot(range(3), [100 * cells[(m, b)]["unet"]["coverage90"] for b in BANDS],
                "--o", ms=3, color=C["unet"])
    ax.axhline(90, ls=":", color=C["ink"], lw=.8)
    ax.text(2.0, 91.5, "nominal 90%", fontsize=5.5, color=C["ink"], ha="right")
    ax.set(xticks=range(3), xticklabels=BANDS, xlabel="Gap duration (days)",
           ylabel="Empirical 90% coverage (%)", ylim=(30, 100), title="Coverage falls short")

    for ax, s in zip(axes, "abcd"):
        label(ax, s)
    finish(f, 9, "regime_boundary")


def fig10():
    lad = json.loads((ROOT / "outputs" / "mechanism_ladder" / "ladder_validation.json").read_text())
    abl = json.loads((ROOT / "outputs" / "ablation_graph" / "ablation_results.json").read_text())
    f = plt.figure(figsize=(7.2, 4.4))
    gs = f.add_gridspec(2, 2, left=.11, right=.98, bottom=.10, top=.92,
                        hspace=.62, wspace=.34, height_ratios=[1, .82])
    axes = [f.add_subplot(gs[0, 0]), f.add_subplot(gs[0, 1]), f.add_subplot(gs[1, :])]

    # (a) chunk size
    ax = axes[0]
    cs = sorted(lad["chunk"], key=int)
    rm = [lad["chunk"][c]["rmse"] for c in cs]
    ax.plot([int(c) for c in cs], rm, "-o", ms=3, color=C["diffusion"])
    ax.axhline(rm[0], ls=":", color=C["ink"], lw=.8)
    ax.set(xscale="log", xticks=[1, 2, 4, 8, 16, 32, 64], xticklabels=cs,
           xlabel="Constraint interval (steps)", ylabel="RMSE (W/m$^2$)",
           title="Chunked feedback has no effect")
    ax.set_ylim(min(rm) - .15, max(rm) + .15)
    ax.text(.5, .12, f"spread {max(rm) - min(rm):.3f} W/m$^2$", transform=ax.transAxes,
            ha="center", fontsize=6, color=C["accent"])

    # (b) projection timing and gain form
    ax = axes[1]
    items = ([("per chunk", lad["projection"]["per_chunk"]),
              ("final only", lad["projection"]["final_only"])] +
             [(f"gain {k}", lad["gain"][k]) for k in ("uniform", "full")])
    vals = [v["rmse"] for _, v in items]
    ax.barh(range(len(items)), vals, color=C["diffusion"], height=.6)
    ax.set(yticks=range(len(items)), yticklabels=[k for k, _ in items],
           xlabel="RMSE (W/m$^2$)", title="Timing and gain form")
    ax.set_xlim(min(vals) - .1, max(vals) + .1)
    for i, v in enumerate(vals):
        ax.text(v + .01, i, f"{v:.3f}", va="center", fontsize=6)
    ax.axvline(lad["chunk"]["8"]["rmse"], ls=":", color=C["ink"], lw=.8)
    ax.invert_yaxis()

    # (c) graph ablation
    ax = axes[2]
    order = ["full", "no_boundary_graph", "no_train_graph", "none"]
    short = {"full": "Full", "no_boundary_graph": "No sampling\ngraph",
             "no_train_graph": "No training\npenalty", "none": "No graph\nat all"}
    vals = [abl[k]["report_2011"]["pooled_rmse"] for k in order]
    ax.bar(range(4), vals, color=["#18858A"] + [C["unet"]] * 3, width=.5)
    ax.set(xticks=range(4),
           xticklabels=["Full\n(both)", "No sampling graph\n(- boundary, - blends)",
                        "No training penalty\n(- structural loss)", "No graph\nat all"],
           ylabel="RMSE (W/m$^2$)", title="Graph ablation on the 2011 reporting cells",
           ylim=(29.68, 29.92))
    for t in ax.get_xticklabels():
        t.set_fontsize(6)
    for i, v in enumerate(vals):
        ax.text(i, v + .005, f"{v:.2f}", ha="center", fontsize=6)

    for ax, s in zip(axes, "abc"):
        label(ax, s)
    finish(f, 10, "mechanism_ladder")


if __name__ == "__main__":
    fig9()
    fig10()
