"""Daily GSR targets and meteorological conditioning fields for masked V6."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import xarray as xr


GSR = "ALLSKY_SFC_SW_DWN"
FEATURES = ("T2M", "PRECTOTCORR", "CLOUD_AMT")
GSR_KWH_TO_WM2 = 1000.0 / 24.0


@dataclass(frozen=True)
class DailyFeatureBundle:
    gsr: xr.DataArray
    features: xr.DataArray
    feature_names: tuple[str, ...]


def _open_variable(root: Path, variable: str) -> xr.DataArray:
    paths = sorted((root / variable).glob("*.nc"))
    if not paths:
        raise FileNotFoundError(f"No daily {variable} files found under {root / variable}")
    datasets = [xr.open_dataset(path) for path in paths]
    try:
        combined = xr.combine_by_coords(datasets, combine_attrs="drop_conflicts")
        if variable not in combined:
            raise KeyError(f"{variable!r} is missing from the downloaded product")
        result = combined[variable].astype(np.float32).load()
        result = result.transpose("time", "lat", "lon")
        return result
    finally:
        for dataset in datasets:
            dataset.close()


def open_daily_feature_bundle(root: str | Path) -> DailyFeatureBundle:
    root = Path(root)
    gsr = _open_variable(root, GSR) * GSR_KWH_TO_WM2
    arrays = [_open_variable(root, name) for name in FEATURES]
    reference = gsr
    for name, array in zip(FEATURES, arrays):
        if not np.array_equal(array.time.values.astype("datetime64[D]"), reference.time.values.astype("datetime64[D]")):
            raise ValueError(f"Time coordinate mismatch for {name}")
        # POWER serves some variables on a 0.5-degree grid and others on a
        # 1-degree grid. Interpolate the former to the GSR target grid so all
        # conditioning channels are collocated before entering the network.
        same_grid = (
            array.lat.size == reference.lat.size
            and array.lon.size == reference.lon.size
            and np.allclose(array.lat.values, reference.lat.values)
            and np.allclose(array.lon.values, reference.lon.values)
        )
        if not same_grid:
            array = array.interp(lat=reference.lat, lon=reference.lon, method="linear")
            if not np.isfinite(array.values).all():
                raise ValueError(f"Regridding {name} to the target grid produced non-finite values")
        arrays[FEATURES.index(name)] = array
    values = np.stack([array.values for array in arrays], axis=1).astype(np.float32)
    features = xr.DataArray(
        values,
        dims=("time", "feature", "lat", "lon"),
        coords={"time": reference.time, "feature": list(FEATURES), "lat": reference.lat, "lon": reference.lon},
        name="daily_weather_features",
        attrs={
            "source": "NASA POWER daily regional API",
            "features": "T2M (degC), PRECTOTCORR (mm/day), CLOUD_AMT (%)",
            "gsr_conversion": "ALLSKY_SFC_SW_DWN kWh m-2 day-1 * 1000 / 24 = W m-2",
        },
    )
    gsr = gsr.rename("gsr")
    gsr.attrs.update(units="W m-2", source="NASA POWER ALLSKY_SFC_SW_DWN")
    if not np.isfinite(gsr.values).all() or not np.isfinite(features.values).all():
        raise ValueError("Daily target or conditioning fields contain non-finite values")
    if np.any(gsr.values < 0) or np.any(features.sel(feature="CLOUD_AMT").values < 0):
        raise ValueError("Daily GSR/cloud fields contain invalid negative values")
    return DailyFeatureBundle(gsr=gsr, features=features, feature_names=FEATURES)


def write_feature_manifest(bundle: DailyFeatureBundle, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "target": "NASA POWER ALLSKY_SFC_SW_DWN converted to W m-2",
        "conditioning": {
            "T2M": "daily near-surface air temperature, degC",
            "PRECTOTCORR": "daily corrected precipitation, mm/day",
            "CLOUD_AMT": "daily cloud amount, percent",
        },
        "time_start": str(bundle.gsr.time.values[0]),
        "time_end": str(bundle.gsr.time.values[-1]),
        "days": int(bundle.gsr.sizes["time"]),
        "grid": {"lat": int(bundle.gsr.sizes["lat"]), "lon": int(bundle.gsr.sizes["lon"])},
        "note": "NASA POWER is a satellite/reanalysis-derived product, not station measurements.",
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
