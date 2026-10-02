"""Download daily NASA POWER GSR and weather-feature tiles for masked V6."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import requests
import xarray as xr


API = "https://power.larc.nasa.gov/api/temporal/daily/regional"
DEFAULT_VARIABLES = ("ALLSKY_SFC_SW_DWN", "T2M", "PRECTOTCORR", "CLOUD_AMT")


def edges(lower: float, upper: float, width: float = 10.0) -> list[tuple[float, float]]:
    result = []
    cursor = lower
    while cursor < upper:
        stop = min(cursor + width, upper)
        result.append((cursor, stop))
        cursor = stop
    return result


def coordinate_tag(value: float, positive: str, negative: str) -> str:
    return f"{positive if value >= 0 else negative}{abs(value):g}"


def output_path(root: Path, variable: str, year: int, south: float, north: float, west: float, east: float) -> Path:
    bounds = "_".join(
        [
            coordinate_tag(south, "N", "S"),
            coordinate_tag(north, "N", "S"),
            coordinate_tag(west, "E", "W"),
            coordinate_tag(east, "E", "W"),
        ]
    )
    return root / variable / f"{variable}_{year}_{bounds}.nc"


def valid_file(path: Path, variable: str) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        with xr.open_dataset(path) as dataset:
            return variable in dataset and dataset.sizes.get("time") in (365, 366)
    except Exception:
        return False


def download_file(path: Path, variable: str, params: dict[str, str | float], retries: int = 4) -> None:
    temporary = path.with_suffix(".nc.part")
    for attempt in range(retries):
        try:
            response = requests.get(API, params=params, timeout=300)
            response.raise_for_status()
            temporary.write_bytes(response.content)
            with xr.open_dataset(temporary) as dataset:
                if variable not in dataset:
                    raise ValueError(f"{variable} missing from API response")
            temporary.replace(path)
            return
        except Exception:
            temporary.unlink(missing_ok=True)
            if attempt + 1 == retries:
                raise
            time.sleep(2**attempt)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-year", type=int, required=True)
    parser.add_argument("--end-year", type=int, required=True)
    parser.add_argument("--south", type=float, required=True)
    parser.add_argument("--north", type=float, required=True)
    parser.add_argument("--west", type=float, required=True)
    parser.add_argument("--east", type=float, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/nasa_power/pilot_east_asia_features"))
    parser.add_argument("--variables", nargs="+", default=list(DEFAULT_VARIABLES))
    parser.add_argument("--pause", type=float, default=0.5)
    args = parser.parse_args()
    if not (-90 <= args.south < args.north <= 90):
        raise ValueError("Latitude bounds must satisfy -90 <= south < north <= 90")
    if not (-180 <= args.west < args.east <= 180):
        raise ValueError("Longitude bounds must satisfy -180 <= west < east <= 180")
    if args.start_year > args.end_year:
        raise ValueError("start-year must not exceed end-year")
    args.output.mkdir(parents=True, exist_ok=True)
    jobs = [
        (variable, year, south, north, west, east)
        for variable in args.variables
        for year in range(args.start_year, args.end_year + 1)
        for south, north in edges(args.south, args.north)
        for west, east in edges(args.west, args.east)
    ]
    for number, (variable, year, south, north, west, east) in enumerate(jobs, start=1):
        path = output_path(args.output, variable, year, south, north, west, east)
        path.parent.mkdir(parents=True, exist_ok=True)
        if valid_file(path, variable):
            print(f"[{number}/{len(jobs)}] exists: {path}")
            continue
        params = {
            "latitude-min": south,
            "latitude-max": north,
            "longitude-min": west,
            "longitude-max": east,
            "parameters": variable,
            "community": "RE",
            "start": f"{year}0101",
            "end": f"{year}1231",
            "format": "NetCDF",
            "time-standard": "UTC",
        }
        print(f"[{number}/{len(jobs)}] downloading: {path}")
        download_file(path, variable, params)
        time.sleep(args.pause)


if __name__ == "__main__":
    main()
