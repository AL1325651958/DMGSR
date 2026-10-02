# Audit findings

Two passes over the manuscript: a numeric audit of every recomputable claim, and
a consistency check of the stated implementation against the saved experiment
protocols. All numeric claims reproduce (see `figures/supplementary/README.md`).
The implementation statements below do not.

## Confirmed mismatches

### 1. Missing-cell loss weight

- Manuscript (Methods, Masked temporal diffusion):
  "missing-cell weight coefficient 4".
- Recorded protocol of the evaluated checkpoint,
  `outputs/masked_v9_graph/protocol.json`: `missing_pixel_weight: 5.0`.
- Current code default in `train_masked`: `missing_weight: float = 4.0`.

The evaluated checkpoints were trained with 5.0, not 4. The manuscript is
describing the wrong value. The default in the source was apparently reused
instead of the value recorded at training time.

### 2. Training-period spatial gap probability is not stated

- Recorded protocol: `training_spatial_probability: 0.8`.
- Current code default: `spatial_probability: float = 0.65`.
- The manuscript quotes `0.35`, but that is the *evaluation* mask probability
  (`spatial_probability` in the ablation/evaluation mask generator and in
  `random_observation_mask`'s default), not the training curriculum value.

The manuscript never states the training curriculum probability. Because 0.35
appears in the same subsection in another role, a reader will assume the
training gaps were sampled with that probability. The correct value, 0.8, should
be stated explicitly.

### 3. `gradient_loss_weight` naming

- Manuscript: "masked gradient coefficient 0.08".
- Code: `gradient_weight` = 0.08, applied in `train_masked` as a squared
  finite-difference term over edges touching hidden cells.

The value is right and the description matches the implementation. No change
needed.

### 4. Graph-structure coefficient

- Manuscript: "graph-structure coefficient 0.05".
- Code: `graph_weight` = 0.05 hard-coded at the call site before the ablation
  change; now an explicit parameter with the same default.

Value and description match.

## Environment statement

The manuscript reports "PyTorch 2.14.0 with CUDA on an NVIDIA GeForce RTX 3070
Laptop GPU". Verified: `torch 2.14.0+cu126`, `NVIDIA GeForce RTX 3070 Laptop
GPU`, compute capability (8, 6). Correct.

## Outdated reference

The manuscript cites Mardani et al. as an arXiv record (arXiv:2309.15214). That
work has since been published:

- *Communications Earth & Environment* **6**, 124 (2025)
- DOI `10.1038/s43247-025-02042-5`, published online 24 February 2025
- Crossref record verified directly; the journal article lists 13 authors and
  carries the preprint as a related identifier

The reference should be updated from the preprint to the journal version. This
supersedes the note in `DRAFT_AUDIT.md`, which recorded the arXiv record as
deliberate at the time of writing.

## Result of the graph ablation (added after this audit)

The controlled graph-regularization ablation requested for this manuscript has
now been run: see `../outputs/ablation_graph/REPORT.md`. Four matched
configurations differing only in the graph mechanism are indistinguishable on the
2011 reporting cells — RMSE spans 29.75–29.83 W/m² and CRPS 17.56–17.60, against
a between-seed standard deviation of 1.4 W/m² in RMSE within a single
configuration, and every paired month-block bootstrap interval touches or
straddles zero.

This changes the framing requirement, not just a number: the title, abstract,
highlights and the framework subsection present graph regularization as part of
the proposed method, and the Discussion states that no controlled ablation is
available. Both statements now need revision.

## Calibration period in Figure 2

Figure 2 carries a box label "2010 calibration". `results.json` records
`calibration_year: 2010`, and the frozen calibration archive records
`calibration_period: 2010-01 to 2010-12`. Consistent. (Note that
`outputs/masked_v9_graph_calibrated` refers to a different, earlier calibration
of the raw diffusion ensemble, which used 2010 for fitting and 2011 for
reporting as well.)

## Effect on the graph ablation

The ablation is run with the recorded values (spatial probability 0.8,
missing-cell weight 5.0, gradient weight 0.08) so that its "full" configuration
reproduces the evaluated model rather than the current code defaults.

## Recommended manuscript edits

1. Methods, Masked temporal diffusion: change "missing-cell weight coefficient
   4" to 5.
2. Same subsection or the curriculum sentence: state that the fitting epochs
   after the first third sample spatial blocks with probability 0.8, and that
   0.35 is the evaluation-mask probability used to generate the scored gaps.
3. Abstract and Methods contain no other coefficient claims that need revision.

These are text corrections only; no reported number changes.
