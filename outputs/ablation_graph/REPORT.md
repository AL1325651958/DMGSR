# Graph-regularization ablation

Four configurations were fitted with every factor except the graph mechanism held fixed.

- fitting: 2000-2009, 30 epochs, seeds [11, 22, 33]
- training curriculum: spatial probability 0.8, missing-cell weight 5.0, gradient weight 0.08
- checkpoint selection: 2010 validation CRPS, per configuration; reporting: 2011
- sampling: 12 members, chunk size 8
- held fixed: data, chronological split, mask schedule, seed list, epoch count, checkpoint selection
- varied: graph-structure loss weight and the two inference-time graph operations

| Configuration | Structural penalty (loss) | Graph correction + blends (sampling) | Meaning |
|---|---|---|---|
| A  Full | 0.05 | strength 0.15, blends 0.25 | structural loss + chunk-boundary graph correction + spatial blends |
| B  No sampling graph | 0.05 | removed | structural loss only (no inference-time graph correction) |
| C  No training penalty | removed | strength 0.15, blends 0.25 | chunk-boundary correction + spatial blends only (no structural loss) |
| D  No graph at all | removed | removed | no graph mechanism at any stage |

All four configurations selected seed 22 on 2010 validation CRPS.

## Selected-seed skill on the 2011 reporting cells

| Configuration | RMSE | MAE | CRPS | Coverage 90% | Width 90% |
|---|---:|---:|---:|---:|---:|
| A  Full | 29.756 | 23.138 | 17.561 | 79.1% | 80.80 |
| B  No sampling graph | 29.828 | 23.209 | 17.599 | 79.3% | 81.30 |
| C  No training penalty | 29.748 | 23.137 | 17.555 | 78.8% | 79.74 |
| D  No graph at all | 29.821 | 23.209 | 17.594 | 79.0% | 80.24 |

### Differences from the full configuration

| Configuration | dRMSE | dCRPS | dedge MAE | dLaplacian MAE |
|---|---:|---:|---:|---:|
| B  No sampling graph | +0.072 | +0.039 | +0.079 | +0.173 |
| C  No training penalty | -0.009 | -0.006 | -0.053 | -0.264 |
| D  No graph at all | +0.064 | +0.033 | +0.025 | -0.094 |

Positive values mean the ablated configuration is worse. Every difference is far smaller than the between-seed spread reported below.

## Grid-graph structure, selected seed

| Configuration | Edge MAE | Laplacian MAE | Graph correlation |
|---|---:|---:|---:|
| A  Full | 20.037 | 50.670 | 0.9376 |
| B  No sampling graph | 20.115 | 50.844 | 0.9373 |
| C  No training penalty | 19.984 | 50.406 | 0.9377 |
| D  No graph at all | 20.062 | 50.577 | 0.9373 |

## Between-seed spread on 2011 (the scale to judge the differences against)

| Configuration | RMSE by seed | mean +- sd | CRPS by seed | mean +- sd |
|---|---|---:|---|---:|
| A  Full | 33.00, 29.76, 32.26 | 31.67 +- 1.39 | 19.34, 17.56, 18.87 | 18.59 +- 0.75 |
| B  No sampling graph | 33.11, 29.83, 32.34 | 31.76 +- 1.40 | 19.39, 17.60, 18.91 | 18.63 +- 0.76 |
| C  No training penalty | 33.12, 29.75, 32.49 | 31.79 +- 1.46 | 19.43, 17.55, 19.00 | 18.66 +- 0.80 |
| D  No graph at all | 33.23, 29.82, 32.56 | 31.87 +- 1.47 | 19.48, 17.59, 19.04 | 18.71 +- 0.81 |

## Paired month-block bootstrap, full minus ablated (2011)

| Comparison | dRMSE 95% interval | dCRPS 95% interval |
|---|---|---|
| Full - B  No sampling graph | [-0.314, +0.000] | [-0.133, +0.000] |
| Full - C  No training penalty | [-0.068, +0.079] | [-0.061, +0.047] |
| Full - D  No graph at all | [-0.356, +0.057] | [-0.168, +0.027] |

## Reading of the result

Note the scale. The between-configuration differences in RMSE are 0.01-0.07 W/m2 and in CRPS 0.01-0.04 W/m2, while the between-seed standard deviation within a single configuration is 1.39-1.47 W/m2 in RMSE and 0.75-0.81 W/m2 in CRPS. The graph mechanism therefore produces no difference that is separable from fitting-seed variation on this protocol: every configuration is the same within noise. The paired bootstrap agrees, with all full-minus-ablated intervals touching or straddling zero. The structure diagnostics behave the same way, moving by less than 0.3 W/m2, so the generated fields are not measurably more graph-smooth when the penalty and the graph correction are present.

Two consequences for the manuscript:

1. No percentage improvement may be attributed to graph regularization, and the phrase does not belong in the title or in the framing of the contribution. The title, abstract, highlights and the framework subsection currently present graph regularization as part of the proposed method.
2. The Discussion sentence that says no controlled ablation is available should be replaced by this result.

## Limits

- The 2011 period was inspected during earlier development; this is a development comparison.
- One stored mask per month; the bootstrap resamples months, not independent gaps.
- All configurations selected the same fitting seed, so the primary comparison is at fixed initialisation; the cross-seed table shows what happens when it is not.
- Removing a penalty changes the effective loss scale; coefficients were held at their recorded values rather than retuned, so a differently weighted penalty is not excluded.
- The graph is the fixed four-neighbour grid. Nothing here speaks to learned attention or irregular networks.
