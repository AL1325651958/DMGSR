"""Batch literature discovery for the DMGSR related-work survey.

Runs a list of OpenAlex queries through the nature-academic-search no-MCP
fallback script, keeps relevance-ranked results (citation re-ranking drags in
off-topic mega-cited surveys), deduplicates by DOI/title, and writes a machine
readable corpus plus a compact console listing.

Usage: python paper/literature_survey.py [--year-from 2018] [--per-query 12]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

SKILL = Path(r"C:\Users\13256\.agents\skills\nature-academic-search\scripts\academic_search.py")
OUT = Path(__file__).resolve().parent / "literature"

# Topic buckets. Each query is a retrieval probe, not a judgement.
QUERIES = {
    "solar_recovery": [
        "solar radiation missing data imputation",
        "solar irradiance gap filling",
        "solar radiation quality control gap filling network",
        "estimating missing solar radiation records",
        "solar radiation reconstruction satellite",
        "global horizontal irradiance gap filling",
    ],
    "remote_sensing_infill": [
        "remote sensing image missing data reconstruction deep learning",
        "cloud removal remote sensing deep learning",
        "MODIS gap filling spatiotemporal",
        "land surface remote sensing data gap filling neural network",
        "satellite image inpainting deep learning",
    ],
    "diffusion_generative": [
        "diffusion model time series imputation",
        "conditional diffusion model spatiotemporal data",
        "score-based generative model missing data",
        "diffusion model geoscience remote sensing",
        "generative model probabilistic downscaling climate",
    ],
    "classical_spatial": [
        "spatiotemporal kriging interpolation",
        "geostatistical interpolation solar radiation",
        "graph signal processing regularization",
        "graph neural network spatiotemporal interpolation",
    ],
    "calibration_uq": [
        "ensemble spread calibration probabilistic forecast",
        "deep ensemble uncertainty regression",
        "conformal prediction spatiotemporal",
        "CRPS proper scoring rule evaluation",
    ],
    "unet_comparators": [
        "U-Net geoscience regression",
        "convolutional encoder decoder spatiotemporal gap filling",
    ],
}


def run_query(query: str, limit: int, year_from: int | None):
    cmd = [sys.executable, str(SKILL), query, "--limit", str(limit), "--sort", "relevance_score"]
    if year_from:
        cmd += ["--year-from", str(year_from)]
    env = dict(os.environ, PYTHONIOENCODING="utf-8", OPENALEX_MAILTO=os.environ.get(
        "OPENALEX_MAILTO", "1325651958@qq.com"))
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", env=env)
    if proc.returncode != 0:
        print(f"  ! query failed: {query}\n    {proc.stderr.strip()[:200]}", flush=True)
        return []
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError:
        print(f"  ! non-JSON output for: {query}", flush=True)
        return []


def key_of(rec):
    doi = (rec.get("doi") or "").strip().lower()
    if doi:
        return "doi:" + doi
    return "title:" + " ".join((rec.get("title") or "").lower().split())[:120]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--year-from", type=int, default=2018)
    parser.add_argument("--per-query", type=int, default=12)
    parser.add_argument("--buckets", nargs="*", default=list(QUERIES))
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    corpus: dict[str, dict] = {}
    for bucket in args.buckets:
        print(f"== {bucket}", flush=True)
        for query in QUERIES[bucket]:
            hits = run_query(query, args.per_query, args.year_from)
            print(f"  {len(hits):2d}  {query}", flush=True)
            for rec in hits:
                k = key_of(rec)
                if k in corpus:
                    corpus[k].setdefault("buckets", []).append(bucket)
                    corpus[k].setdefault("queries", []).append(query)
                    continue
                rec["buckets"] = [bucket]
                rec["queries"] = [query]
                corpus[k] = rec

    records = sorted(corpus.values(), key=lambda r: (r.get("year") or 0), reverse=True)
    (OUT / "corpus.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nunique records: {len(records)}  ->  {OUT / 'corpus.json'}")

    # Compact listing for reading
    lines = []
    for r in records:
        au = ", ".join((r.get("authors") or [])[:2])
        lines.append(f"[{r.get('year')}] {r.get('title','')[:110]} | {au} | {r.get('journal','')[:38]} "
                     f"| cit {r.get('cited_by_count',0)} | {'/'.join(r['buckets'])}")
    (OUT / "corpus.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[:40]))


if __name__ == "__main__":
    main()
