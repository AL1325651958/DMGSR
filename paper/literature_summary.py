"""Summarise the literature corpus: per-bucket counts, yearly volume, venues.

Reads paper/literature/corpus.json (produced by literature_survey.py) and writes
paper/literature/summary.json plus a console table. Read-only otherwise.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

LIT = Path(__file__).resolve().parent / "literature"


def main():
    records = json.loads((LIT / "corpus.json").read_text(encoding="utf-8"))
    print(f"records: {len(records)}")

    by_bucket = Counter()
    for r in records:
        for b in set(r.get("buckets") or []):
            by_bucket[b] += 1
    print("\n== records per topic bucket ==")
    for b, n in by_bucket.most_common():
        print(f"  {b:24s} {n}")

    years = Counter(r.get("year") for r in records if r.get("year"))
    print("\n== publication year ==")
    for y in sorted(y for y in years if y and y >= 2015):
        print(f"  {y}  {years[y]:3d}  {'#' * min(years[y], 60)}")

    print("\n== venues (top 15) ==")
    for v, n in Counter((r.get("journal") or "(preprint/none)") for r in records).most_common(15):
        print(f"  {n:3d}  {v[:70]}")

    print("\n== most cited in corpus (top 12) ==")
    for r in sorted(records, key=lambda x: -(x.get("cited_by_count") or 0))[:12]:
        print(f"  {r.get('cited_by_count'):5d}  [{r.get('year')}] {(r.get('title') or '')[:88]}")

    # Journal-only subset: how much of the volume sits in peer-reviewed venues
    with_journal = [r for r in records if r.get("journal")]
    print(f"\nrecords with a named venue: {len(with_journal)} / {len(records)}")

    out = {
        "records": len(records),
        "per_bucket": dict(by_bucket),
        "per_year": {str(k): v for k, v in sorted(years.items()) if k},
        "venues_top": Counter((r.get("journal") or "(preprint/none)") for r in records).most_common(25),
    }
    (LIT / "summary.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {LIT / 'summary.json'}")


if __name__ == "__main__":
    main()
