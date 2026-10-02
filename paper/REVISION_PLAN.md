# Revision plan: reframing the manuscript around the evidence

The manuscript's three framing claims do not all survive the experiments run for
it. This plan states what changes and why, section by section. It is a proposal:
nothing in `Manuscript.tex` has been altered yet.

## What the evidence now says

| Claim | Evidence | Verdict |
|---|---|---|
| Graph regularization contributes | Four matched configurations indistinguishable; RMSE 29.75–29.83 vs a 1.4 W/m² between-seed sd (Fig. 10c) | **not supported** |
| The chunked correction feedback matters | Constraint interval 1 vs 64 steps changes RMSE by 0.002 W/m²; projection timing and gain form change it by 0.04 W/m² (Fig. 10a,b) | **not supported** |
| Conditional diffusion is the better reconstructor | Diffusion loses to the U-Net on spatial gaps and wins on temporal, mixed and endpoint gaps; U-Net is almost duration-insensitive (49.7–49.8) while diffusion degrades with duration (32.3→43.6) (Fig. 9) | **conditional, and the condition is the interesting part** |
| Raw neural ensembles are underdispersed | Replicated on new data: coverage 63.0% and 49.5% against a nominal 90% | **supported** |
| Constraining the sampler improves the mean while hurting reliability | Fig. 8: RMSE 44.67→37.70 with coverage 60.7%→44.6% | **supported** |
| Fusion beats a calibrated spatial comparator | 2011: +5.1% RMSE, +8.9% CRPS. 2012–2014 (untouched, pre-registered): +24.0% RMSE, all four decision rules confirmed | **supported, and much stronger on held-out data** |

The three supported items share a theme: **the value and the cost of imposing
constraints during generative sampling, and the boundary between what a
generative branch and a spatial regressor are each good for.** That is the
contribution. Graph regularization and the chunked feedback are not.

## 1. Title

**Now.** Graph-regularized diffusion and ensemble calibration for probabilistic
solar radiation recovery

**Proposed.** Constraint, calibration and applicability boundaries in
probabilistic solar radiation recovery: a controlled study of masked diffusion
against a spatial ensemble

Shorter alternative: *Where masked diffusion helps in solar radiation recovery:
constraint, calibration and regime dependence*

Rationale: removes the unsupported mechanism, names the two things the paper
actually establishes.

## 2. Abstract

Replace the framing sentences. The current abstract opens on graph
regularization and reports the development-period gain as the headline. Proposed
structure:

1. Problem and design (unchanged): synthetic gaps on a 10×10 POWER grid, monthly
   conditioning known from the complete product, chronological split.
2. **New second sentence**: state that the sampling-time constraints are the
   object of study, and that they are evaluated by switching them off.
3. **New result sentence**: on four gap families the two branches have
   complementary skill — diffusion wins on temporal, mixed and endpoint gaps
   (−6 to −17 W/m² RMSE) and loses on spatial blocks (+2 to +9); the U-Net is
   nearly insensitive to gap duration while diffusion degrades with it.
4. **New negative sentence**: neither the graph penalty nor the chunked feedback
   changes any reported metric measurably.
5. Keep the calibration and reliability sentences; they replicated.
6. Add the confirmatory sentence: all four pre-registered decision rules were
   confirmed on 2012–2014, which had not been evaluated.
7. Boundaries: unchanged, plus that the monthly condition leaks by construction.

## 3. Highlights

- Replace "Calendar-aligned fusion reaches 19.77 W/m² RMSE on development
  reporting masks" with the regime-complementarity finding.
- Replace the graph-regularization implication with the negative result.
- Keep the denoising/coverage tension item, and promote it.

## 4. Methods

- **Graph penalties and constraint feedback**: keep the description, add the two
  new switches (`chunk_projection`, `gain_mode`) and state that each mechanism
  is ablated in Results.
- **Correction of implementation inconsistencies**: correct the two coefficient
  errors found in audit: missing-cell weight **4 → 5**, and state the training
  curriculum spatial probability **0.8** (the section currently uses 0.35, which
  is the evaluation-mask value).
- Add one sentence recording that the archived `masked_v9_graph` ensemble
  predates the valid-day fix and that all reported numbers come from the
  regenerated ensembles.

## 5. Results

Proposed order, with the new material first:

1. **Applicability boundary across gap regimes** — new subsection, Fig. 9. The
   12-cell matrix, the duration sensitivity, and the CRPS and coverage panels.
2. **The sampling mechanism does not change the reported metrics** — new
   subsection, Fig. 10a,b. Includes the graph ablation (Fig. 10c) moved here from
   Discussion.
3. **What constraining sampling costs** — promote the existing Fig. 8
   subsections. This is the methodological finding; it currently sits last.
4. **Fusion against a calibrated spatial comparator** — the existing Table 1 and
   Fig. 3–5, demoted, with the 2011 numbers labelled as development-period.
5. **Confirmatory evaluation on 2012–2014** — new subsection. Pre-registered
   decision rules, the four outcomes, and the `w` 0.80→0.40 shift.
6. Monthly heterogeneity and the fixed case — keep as is.

## 6. Discussion

- **Delete** "No controlled graph ablation is available in this record." It is
  now available and negative.
- **Retitle** the subsection "What graph regularization means here" to something
  like "A tested component that did not matter".
- Add a subsection on why the correction sequence is inert: the final projection
  pass re-imposes the monthly energy and the observed values, so trajectories
  that differ mid-sampling converge to the same constrained set. State this as a
  measured property, not a derivation.
- Add the complementary-skill argument: a spatial regressor is the right tool for
  contiguous blocks and the wrong tool for temporal runs, because it never sees
  the time axis.
- Keep the physical-support, monthly-condition and independent-validation
  boundaries; add that the confirmatory run tests replication, not removal of the
  monthly-condition information leak.

## 7. New figures

| Figure | Content | Source |
|---|---|---|
| **Fig. 9** | Regime boundary: (a) diffusion − U-Net RMSE across 4 families × 3 durations, (b) duration sensitivity, (c) CRPS by regime, (d) raw coverage against the nominal level | `outputs/missingness_matrix/matrix_validation.json` |
| **Fig. 10** | Mechanism ladder: (a) constraint interval 1→64 steps, (b) projection timing and gain form, (c) the four-configuration graph ablation | `outputs/mechanism_ladder/ladder_validation.json`, `outputs/ablation_graph/ablation_results.json` |

Both are generated by `paper/make_figures_new.py` and follow the same layout
contract as Figures 1–8.

## 8. Corrections that do not change framing

- Mardani reference: arXiv record → *Communications Earth & Environment* **6**,
  124 (2025), DOI `10.1038/s43247-025-02042-5`.
- Missing-cell weight coefficient 4 → 5.
- Add the training curriculum spatial probability 0.8, and distinguish it from
  the 0.35 evaluation-mask probability.
- Add the solar-specific imputation literature (at least Başakın et al. 2023,
  Girimurugan et al. 2023) and the post-CSDI diffusion-imputation line
  (PriSTI, MTSCI, SaSDim); the current reference list is method-only.
- Do not cite `outputs/graph_retraining_verification/` or
  `outputs/masked_v9_graph_calibrated/` until they are regenerated from the
  current code; see `FINDINGS_2026-10-02.md` §1.

## 9. What this plan deliberately does not do

It does not claim the fusion gain is a general result: one region, one product,
one monthly-conditioning regime. It does not turn the regime matrix into a
general law: 12 cells, one domain, three mask realisations on the calibration
period. And it does not soften the negative results to protect the original
framing — a measured null is a result, and this manuscript is stronger for
reporting three of them than for implying a mechanism it cannot show.
