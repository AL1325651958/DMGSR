"""True yearly publication volume per topic, from OpenAlex.

Two OpenAlex behaviours shape this script:

* The query must sit inside ``filter`` with ``title_and_abstract.search``. The
  bare ``search`` parameter also matches full text and reference lists and
  inflates a count by one to two orders of magnitude.
* ``group_by=publication_year`` is returned newest-first and its *length* is
  bounded by ``per_page``; with ``per_page=1`` only a single (newest) bucket
  comes back. Grouping is therefore not used. Each year is queried separately
  with a count-only request, which is exact and cheap.

Usage: python paper/literature_volume.py [--year-from 2015] [--year-to 2026]
"""
from __future__ import annotations

import argparse
import json
import os
import time
import urllib.parse
import urllib.request
from pathlib import Path

OUT = Path(__file__).resolve().parent / "literature"
MAILTO = os.environ.get("OPENALEX_MAILTO", "1325651958@qq.com")
API = "https://api.openalex.org/works"

TOPICS = {
    "solar radiation missing-data imputation": "solar radiation missing data imputation",
    "solar irradiance gap filling": "solar irradiance gap filling",
    "solar radiation reconstruction": "solar radiation reconstruction",
    "PV power missing-data imputation": "photovoltaic power missing data imputation",
    "remote sensing gap filling": "remote sensing gap filling",
    "satellite cloud removal / inpainting": "cloud removal satellite image deep learning",
    "diffusion for time-series imputation": "diffusion time series imputation",
    "diffusion for spatiotemporal data": "diffusion spatiotemporal imputation",
    "spatiotemporal kriging": "spatiotemporal kriging",
    "graph signal processing": "graph signal processing",
    "ensemble spread calibration": "ensemble spread calibration probabilistic",
    "conformal prediction": "conformal prediction",
}


def count(query: str, year: int, retries: int = 4) -> int | None:
    """Exact count of title/abstract matches published in `year`.

    A per-year window is used because OpenAlex's group_by response is both
    truncated and ordered newest-first, which makes it unusable for a full
    yearly series. HTTP 429 is retried with exponential backoff.
    """
    filt = f"from_publication_date:{year}-01-01,to_publication_date:{year}-12-31,title_and_abstract.search:{query}"
    params = {"filter": filt, "per_page": "1", "mailto": MAILTO}
    url = f"{API}?{urllib.parse.urlencode(params, safe=':,')}"
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                return json.loads(resp.read().decode())["meta"]["count"]
        except urllib.error.HTTPError as exc:
            if exc.code != 429 or attempt == retries - 1:
                raise
            time.sleep(2.0 * (2**attempt))
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--year-from", type=int, default=2015)
    parser.add_argument("--year-to", type=int, default=2026)
    parser.add_argument("--sleep", type=float, default=1.2)
    args = parser.parse_args()

    years = list(range(args.year_from, args.year_to + 1))
    result = {"years": years, "field": "title_and_abstract.search", "topics": {}}
    for label, query in TOPICS.items():
        by_year = {}
        for y in years:
            try:
                by_year[str(y)] = count(query, y)
            except Exception as exc:
                by_year[str(y)] = None
                print(f"  ! {label} {y}: {exc}", flush=True)
            time.sleep(args.sleep)
        total = sum(v for v in by_year.values() if v)
        result["topics"][label] = {"query": query, "total": total, "by_year": by_year}
        print(f"{label:44s} total {total:>5}", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "volume.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    hdr = "topic".ljust(42) + "".join(str(y)[2:].rjust(6) for y in years) + "  total"
    print("\n" + hdr)
    for label, d in result["topics"].items():
        print(label.ljust(42) + "".join(str(d["by_year"].get(str(y)) or 0).rjust(6) for y in years)
              + f"  {d['total']:>5}")
    print(f"\nwrote {OUT / 'volume.json'}")


if __name__ == "__main__":
    main()
