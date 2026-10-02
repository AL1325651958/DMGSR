# DMGSR: masked diffusion for daily global solar radiation recovery

This repository is a research prototype for probabilistic recovery of daily
global horizontal irradiance (GSR) from a partially observed field, plus the
LaTeX manuscript draft that reports it.

## Scope

- target: NASA POWER daily `ALLSKY_SFC_SW_DWN`, converted from
  `kW-hr m-2 day-1` to daily-mean `W m-2` (`value * 1000 / 24`);
- conditioning: NASA POWER daily `T2M`, `PRECTOTCORR`, `CLOUD_AMT`, linearly
  regridded from 0.5 to 1 degree where the source grid differs;
- domain: 20--30 N, 100--110 E, a 10x10 one-degree grid, 2000--2014;
- split: train 2000--2009, validation 2010--2011, exploratory follow-up
  2012--2014. The reporting period has been inspected during development and
  is **not** an untouched test set.

NASA POWER is a satellite/reanalysis-derived product, not station
measurements.

## Layout

```
src/dmgsr/numerics.py         area-weighted coarse/expand/project grid operators
src/dmgsr/diffusion_ops.py    64-step velocity schedule + masked monthly projection
src/dmgsr/solar_prior.py      train-only FAO-56 extraterrestrial-radiation prior
src/dmgsr/daily_features.py   NASA POWER loader and feature manifest
src/dmgsr/masked_temporal.py  masked weather-conditioned temporal diffusion (main model)
src/dmgsr/spatial_unet.py     independently trained spatial U-Net comparator
scripts/download_*.py         NASA POWER daily target and weather-feature downloaders
scripts/benchmark_baseline_suite.py  classical/graph point baselines on stored masks
scripts/plot_*.py             figure generation from saved ensembles
paper/                        Elsevier-format draft, evidence and eight figures
outputs/                      frozen experiment artifacts (one report per directory)
tests/                        unit tests for the surviving modules
```

## Install and test

```powershell
$env:PYTHONPATH = "src"
python -m unittest discover -s tests
```

Requires Python 3.10+, PyTorch, NumPy, xarray and netCDF4. `tests` skip the
bundled-data case when `data/nasa_power/pilot_east_asia_features` is absent.

## Model

`MaskedTemporalVelocityNet` sees the noisy latent, four monthly channels, the
daily weather features, the observed residual, the observation mask, the
valid-day mask, the diffusion timestep and a within-month sin/cos position.
Reverse diffusion runs in 8-step chunks over 64 steps. At each chunk boundary
the sampler applies an ensemble-spread-weighted monthly projection with a
solar upper bound, a four-neighbour graph boundary correction and a spatial
boundary blend, then rebuilds a consistent noisy latent before the next chunk.
Observed cells are held at their reference value throughout.

The coarse monthly projection (`diffusion_ops.partial_project`) is the
component that carries the monthly energy requirement. It weights by the
number of *valid* days rather than the 31-day padded length, leaves blocks with
no hidden cell untouched, and never scales a block down when the observations
already meet the monthly requirement. See `tests/test_diffusion_ops.py` for the
exact invariants.

## Run

```powershell
$env:PYTHONPATH = "src"

# download the daily GSR target and weather features
python scripts/download_nasa_power_features.py --start-year 2000 --end-year 2014 `
  --south 20 --north 30 --west 100 --east 110 `
  --output data/nasa_power/pilot_east_asia_features

# train and evaluate the masked diffusion model
python -m dmgsr.masked_temporal --epochs 30 --output outputs/masked_v6
python -m dmgsr.masked_temporal --evaluate --output outputs/masked_v6 `
  --split validation --members 12 --chunk-size 8

# independently train the spatial U-Net comparator
python -m dmgsr.spatial_unet --epochs 10 --output outputs/spatial_unet_v9

# classical and graph baselines on a stored mask
python scripts/benchmark_baseline_suite.py `
  --input outputs/masked_v9_graph/masked_ensemble_validation.npz `
  --output outputs/baseline_suite
```

Training uses a spatial curriculum: the first third of the epochs mixes
temporal and spatial gaps, the remainder uses spatial blocks only.

## Bundled experiments

| Directory | Contents |
|---|---|
| `outputs/masked_v9_graph/` | masked diffusion checkpoints and validation ensemble used by the manuscript |
| `outputs/masked_v9_graph_calibrated/` | validation-only ensemble spread calibration |
| `outputs/spatial_unet_ensemble/` | eight-member U-Net deep ensemble |
| `outputs/graph_retraining_verification/` | matched diffusion vs U-Net verification on a shared mask |
| `outputs/masked_v6/` | missingness sweep over temporal, spatial, mixed and endpoint gaps |
| `outputs/figures/` | rendered comparison and diffusion-evolution figures |

## Manuscript

`paper/Manuscript.tex` is compiled with the bundled `paper/tools/tectonic.exe`.
`paper/build_evidence.py` recomputes the calibration and metrics from the saved
aligned ensembles and writes `paper/evidence/`; `paper/make_figures.py` redraws
the eight figures. `paper/DRAFT_AUDIT.md` records the claim--evidence ledger and
the claims that were removed for lack of support.

Known boundary: the 2010--2014 period was inspected during development, monthly
conditioning uses the complete reference monthly field (so it carries
information about the hidden values), and there is no matched ablation of the
graph regulariser. The draft states these limits explicitly.

## History

The repository previously carried an earlier research line: monthly CMIP6
`rsds` disaggregation, the `reference_paired` daily pilot, the V2--V5 diffusion
lineage, inference-time guided inpainting, and placeholder compact baselines.
That lineage was removed during a cleanup, together with its output archives,
configurations and the modules that only served it (`experiment`, `model`,
`physics`, `data`, `targets`, `train`, `train_daily`, `optimize_diffusion`,
`missing_inpaint`, `temporal_denoiser`, `calibrate_ensemble`,
`transfer_experiment`, `external_features`). The shared numerical primitives
they provided were re-homed in `numerics.py` and `diffusion_ops.py`.

`data/cmip6/` still holds the downloaded GISS-E2-1-G monthly fields in case the
CMIP6 transfer experiment is revived; no current code reads it.

During the same cleanup three defects in the shared projection were fixed: the
monthly energy target used the 31-day padded length, missing cells were blanked
to zero in blocks whose observations already exceeded the target, and
`SolarPrior.fit` could silently produce an all-zero month. Artifacts under
`outputs/` were produced before those fixes and are no longer bit-reproducible
from the current code.
