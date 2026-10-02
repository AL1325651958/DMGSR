"""Controlled graph-regularization ablation for the masked temporal diffusion.

Four configurations are trained with everything held fixed except the two graph
mechanisms: the four-neighbour structural penalty in the loss, and the graph
boundary correction plus spatial boundary blends applied during sampling.

    full              both graph mechanisms present (paper configuration)
    no_boundary_graph training penalty retained, boundary correction disabled
    no_train_graph    training penalty removed, boundary correction retained
    none              neither mechanism

Data, chronological split, mask schedule, seed list, epoch count and checkpoint
selection are identical across configurations. Checkpoints are selected on 2010
validation CRPS and then scored once on 2011. The 2011 period was inspected
during earlier development, so this remains a development comparison.

Usage:
    python -m dmgsr.ablation                       # train, select, report
    python -m dmgsr.ablation --epochs 2 --seeds 11 --configs full  # pilot
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
    graph_laplacian,
    prepare_masked_data,
    random_observation_mask,
    sample_masked,
    summarize_masked_metrics,
    train_masked,
)
from .solar_prior import SolarPrior

OUT = Path("outputs/ablation_graph")

CONFIGS = {
    "full": dict(graph_weight=0.05, graph_strength=0.15, boundary_blend=0.25),
    "no_boundary_graph": dict(graph_weight=0.05, graph_strength=0.0, boundary_blend=0.0),
    "no_train_graph": dict(graph_weight=0.0, graph_strength=0.15, boundary_blend=0.25),
    "none": dict(graph_weight=0.0, graph_strength=0.0, boundary_blend=0.0),
}

DESCRIPTION = {
    "full": "structural loss + chunk-boundary graph correction + spatial blends",
    "no_boundary_graph": "structural loss only; no inference-time graph correction",
    "no_train_graph": "chunk-boundary graph correction + spatial blends only; no structural loss",
    "none": "no graph mechanism at any stage",
}


def graph_metrics(samples, truth, observed, valid):
    """Grid-graph structure diagnostics on the hidden cells and their incident edges."""
    samples = samples.to(truth.device)
    truth = truth.to(truth.device)
    observed = observed.to(truth.device)
    valid = valid.to(truth.device)
    mean = samples.mean(1)
    missing = (valid > 0) & (observed <= 0)

    def laplacian(z):
        shape = z.shape
        q = z.reshape(-1, shape[-2], shape[-1])
        p = torch.nn.functional.pad(q[:, None], (1, 1, 1, 1), mode="replicate")[:, 0]
        out = 4.0 * q - (p[:, 1:-1, :-2] + p[:, 1:-1, 2:] + p[:, :-2, 1:-1] + p[:, 2:, 1:-1])
        return out.reshape(shape)

    errors = []
    for axis in (-1, -2):
        a = [slice(None)] * mean.ndim
        b = [slice(None)] * mean.ndim
        a[axis], b[axis] = slice(1, None), slice(None, -1)
        a, b = tuple(a), tuple(b)
        affected = (missing[a] | missing[b]) & (valid[a] > 0) & (valid[b] > 0)
        errors.append(((mean[a] - mean[b]) - (truth[a] - truth[b]))[affected])
    edge_mae = float(torch.cat(errors).abs().mean())
    lap_mae = float((laplacian(mean) - laplacian(truth)).abs()[missing].mean())
    pred = mean[missing]
    y = truth[missing]
    corr = float(torch.corrcoef(torch.stack((pred, y)))[0, 1]) if pred.numel() > 1 else float("nan")
    return {"edge_mae": edge_mae, "laplacian_mae": lap_mae, "graph_corr": corr}


def load_bundle(root=DATA_ROOT):
    x, daily_features, valid, target, weights, keys, latitudes = prepare_masked_data(root)
    years = np.array([int(key[:4]) for key in keys])
    train = np.flatnonzero(years <= 2009)
    val = np.flatnonzero(years == 2010)
    report = np.flatnonzero(years == 2011)
    return dict(x=x, features=daily_features, valid=valid, target=target, weights=weights,
                keys=keys, lat=latitudes, train=train, val=val, report=report)


@torch.no_grad()
def run_split(name, cfg, bundle, ids, device, seed=11, members=12, chunk_size=8):
    """Sample one split under a configuration and score the hidden cells."""
    checkpoint = torch.load(OUT / name / f"model_{seed}.pt", map_location="cpu", weights_only=False)
    net = MaskedTemporalVelocityNet().to(device)
    net.load_state_dict(checkpoint["model"])
    net.eval()
    physics = SolarPrior(checkpoint["latitude"], checkpoint["template"])
    x, features, valid = bundle["x"], bundle["features"], bundle["valid"]
    target, weights, keys = bundle["target"], bundle["weights"], bundle["keys"]
    valid_device = valid.to(device)
    target_device = target.to(device)
    base, factor = physics.fields(target_device, keys, weights.to(device), valid_device)
    scale = float(checkpoint["residual_scale"])
    residual = ((x.to(device) - base) / factor.clamp_min(1e-4)) * valid_device / scale
    feature_mean = checkpoint["feature_mean"].to(device).to(torch.float32)
    feature_std = checkpoint["feature_std"].to(device).to(torch.float32)
    weather = (
        (features.to(device) - feature_mean[None, None, :, None, None])
        / feature_std[None, None, :, None, None]
    )
    rng = np.random.default_rng(20260911)
    pieces, observed_rows = [], []
    for group in np.array_split(ids, max(1, len(ids) // 6)):
        index = torch.as_tensor(group, device=device)
        observed = random_observation_mask(
            valid[group], rng, 1, 7, 0.35, "mixed"
        ).to(device)
        pieces.append(sample_masked(
            net, target_device[index], [keys[i] for i in group], valid_device[index],
            weather[index], residual[index] * observed, observed, weights.to(device),
            float(checkpoint["mean_scale"]), scale, base[index], factor[index],
            seed=20260911 + int(group[0]), members=members, chunk_size=chunk_size,
            boundary_blend=cfg["boundary_blend"], graph_strength=cfg["graph_strength"],
        ))
        observed_rows.append(observed.cpu())
    samples = torch.cat(pieces).to(device)
    observed = torch.cat(observed_rows).to(device)
    truth = x[ids].to(device)
    valid_ids = valid[ids].to(device)
    summary = summarize_masked_metrics(samples, truth, observed, valid_ids)
    summary.update(graph_metrics(samples, truth, observed, valid_ids))
    return summary, samples.cpu(), observed.cpu()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seeds", type=int, nargs="+", default=[11])
    parser.add_argument("--configs", nargs="+", default=list(CONFIGS))
    parser.add_argument("--root", type=Path, default=DATA_ROOT)
    parser.add_argument("--members", type=int, default=12)
    parser.add_argument("--chunk-size", type=int, default=8)
    # Recorded settings of the evaluated checkpoint
    # (outputs/masked_v9_graph/protocol.json). They differ from the current
    # code defaults, so they are passed explicitly.
    parser.add_argument("--train-spatial-probability", type=float, default=0.8)
    parser.add_argument("--missing-weight", type=float, default=5.0)
    parser.add_argument("--gradient-weight", type=float, default=0.08)
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--skip-report", action="store_true")
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    started = time.time()

    if not args.skip_training:
        for name in args.configs:
            cfg = CONFIGS[name]
            target_dir = OUT / name
            print(f"[train] {name} -> {target_dir}", flush=True)
            train_masked(
                epochs=args.epochs,
                seeds=tuple(args.seeds),
                root=args.root,
                output=target_dir,
                graph_weight=cfg["graph_weight"],
                spatial_probability=args.train_spatial_probability,
                missing_weight=args.missing_weight,
                gradient_weight=args.gradient_weight,
            )
            # Record the inference-side settings next to the checkpoint: they
            # are not part of the training protocol.
            (target_dir / "ablation.json").write_text(json.dumps({
                "config": name, "description": DESCRIPTION[name],
                "graph_weight_training": cfg["graph_weight"],
                "graph_strength_sampling": cfg["graph_strength"],
                "boundary_blend_sampling": cfg["boundary_blend"],
                "train_spatial_probability": args.train_spatial_probability,
                "missing_weight": args.missing_weight,
                "gradient_weight": args.gradient_weight,
                "epochs": args.epochs, "seeds": list(args.seeds),
            }, indent=2), encoding="utf-8")

    bundle = load_bundle(args.root)
    print(f"[data] train={len(bundle['train'])} val(2010)={len(bundle['val'])} report(2011)={len(bundle['report'])}", flush=True)

    results = {
        "_experiment": {
            "train": "2000-2009", "selection": "2010", "report": "2011",
            "note": "2011 was inspected during earlier development; this is a development comparison.",
            "selection_rule": "per configuration, the fitting seed with the lowest 2010 validation CRPS is scored on 2011; "
                              "the rule selects among the three fitting seeds only",
            "epochs": args.epochs, "seeds": list(args.seeds), "members": args.members,
            "chunk_size": args.chunk_size, "device": str(device),
            "train_spatial_probability": args.train_spatial_probability,
            "missing_weight": args.missing_weight,
            "gradient_weight": args.gradient_weight,
            "held_fixed": "data, chronological split, mask schedule, seed list, epoch count, checkpoint selection",
            "varied": "graph-structure loss weight and the two inference-time graph operations",
        }
    }
    per_seed = {}

    for name in args.configs:
        cfg = CONFIGS[name]
        per_seed[name] = {}
        validation_scores = {}
        for seed in args.seeds:
            summary, samples, observed = run_split(
                name, cfg, bundle, bundle["val"], device, seed, args.members, args.chunk_size
            )
            per_seed[name][str(seed)] = {"validation_2010": summary}
            validation_scores[seed] = summary["pooled_crps"]
            np.savez_compressed(
                OUT / name / f"ensemble_validation_2010_seed{seed}.npz",
                gsr=samples.numpy(), truth=bundle["x"][bundle["val"]].numpy(),
                observed=observed.numpy(), valid=bundle["valid"][bundle["val"]].numpy(),
                months=np.array(bundle["keys"])[bundle["val"]],
            )
            print(f"[val] {name} seed={seed} crps={summary['pooled_crps']:.4f} "
                  f"rmse={summary['pooled_rmse']:.4f}", flush=True)

        chosen = min(validation_scores, key=validation_scores.get)
        for seed in args.seeds:
            if seed == chosen:
                continue
            other, s_other, o_other = run_split(
                name, cfg, bundle, bundle["report"], device, seed, args.members, args.chunk_size
            )
            per_seed[name][str(seed)]["report_2011"] = other
            np.savez_compressed(
                OUT / name / f"ensemble_report_2011_seed{seed}.npz",
                gsr=s_other.numpy(), truth=bundle["x"][bundle["report"]].numpy(),
                observed=o_other.numpy(), valid=bundle["valid"][bundle["report"]].numpy(),
                months=np.array(bundle["keys"])[bundle["report"]],
            )
        summary, samples, observed = run_split(
            name, cfg, bundle, bundle["report"], device, chosen, args.members, args.chunk_size
        )
        np.savez_compressed(
            OUT / name / "ensemble_report_2011.npz",
            gsr=samples.numpy(), truth=bundle["x"][bundle["report"]].numpy(),
            observed=observed.numpy(), valid=bundle["valid"][bundle["report"]].numpy(),
            months=np.array(bundle["keys"])[bundle["report"]],
        )
        per_seed[name][str(chosen)]["report_2011"] = summary
        print(f"[report] {name} selected seed={chosen} " + json.dumps(
            {k: round(v, 4) for k, v in summary.items()}), flush=True)
        results[name] = {
            "config": name, "description": DESCRIPTION[name], **cfg,
            "selected_seed": int(chosen),
            "validation_crps_by_seed": {str(k): float(v) for k, v in validation_scores.items()},
            "validation_2010": per_seed[name][str(chosen)]["validation_2010"],
            "report_2011": summary,
        }
        (OUT / "ablation_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
        (OUT / "per_seed_results.json").write_text(json.dumps(per_seed, indent=2), encoding="utf-8")

    (OUT / "ablation_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"[done] {time.time() - started:.1f} s", flush=True)


if __name__ == "__main__":
    main()
