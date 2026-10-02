# V6 masked weather-conditioned diffusion: validation report

## Experimental protocol

NASA POWER daily ALLSKY_SFC_SW_DWN is used as the GSR target, with daily T2M, PRECTOTCORR, and CLOUD_AMT as weather conditions. The model is trained on 2000–2009, while 2010–2011 is used for validation. Artificial gaps include temporal, spatial, mixed, and endpoint patterns with lengths of 1–3, 4–7, and 8–14 days. Reverse diffusion uses 8-step chunks and a bounded monthly projection.

## Missingness sweep

| Scenario | pooled CRPS | pooled RMSE | pooled 90% coverage | monthly CRPS |
|---|---:|---:|---:|---:|
| temporal_1_3 | 43.29 | 73.47 | 0.380 | 50.33 |
| temporal_4_7 | 28.03 | 46.15 | 0.464 | 28.88 |
| temporal_8_14 | 24.39 | 38.98 | 0.481 | 24.49 |
| spatial_1_3 | 57.38 | 96.40 | 0.357 | 59.20 |
| spatial_4_7 | 32.14 | 52.14 | 0.399 | 31.95 |
| spatial_8_14 | 24.81 | 39.47 | 0.442 | 24.57 |
| mixed_1_3 | 47.08 | 81.20 | 0.386 | 56.13 |
| mixed_4_7 | 29.40 | 48.77 | 0.445 | 29.34 |
| mixed_8_14 | 24.90 | 39.73 | 0.470 | 24.46 |
| endpoint_1_3 | 39.84 | 68.39 | 0.410 | 49.04 |
| endpoint_4_7 | 27.23 | 44.33 | 0.465 | 27.91 |
| endpoint_8_14 | 24.41 | 39.05 | 0.480 | 24.57 |

## Validation spread calibration

For the representative mixed 4–7 day case, the selected spread multiplier is **1.7**. It keeps the ensemble mean unchanged (RMSE 48.82 W/m²; MAE 35.57 W/m²), reduces CRPS to 27.09 W/m², and increases 90% coverage to 0.852. The calibrated interval width is 130.86 W/m².

## Interpretation and limitation

The sweep shows that spatial blocks are harder than temporal gaps, especially for short gaps. Coverage remains heterogeneous across months, so the next paper-scale experiment should use separate calibration by missingness type and radiation regime, then test the frozen procedure on an untouched period or independent region. The calibration coefficient is selected on validation data only and must not be refit on the final test set.

Generated artifacts: `sweep_summary_validation.json`, `calibration_mixed_4_7.json`, and `masked_ensemble_validation.npz`.
