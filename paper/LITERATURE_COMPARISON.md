# Related work: volume, positioning and comparison

Scope: recovery and imputation of missing surface solar radiation, together with
the four method families the present study borrows from. Volume figures come from
OpenAlex (`literature/volume.json`), method records from a relevance-ranked
corpus of 276 works (`literature/corpus.json`).

Method note. Counts use OpenAlex `title_and_abstract.search` inside a `filter`
clause, one open-ended window from 2015. The bare `search` parameter also matches
full text and reference lists and inflates a count by one to two orders of
magnitude, so it is not used. `literature/volume.json` records the method and the
per-topic totals. A per-year split is **not** stored: OpenAlex's grouping endpoint
returns truncated, newest-first buckets, and the per-year window queries were
rate-limited (HTTP 429). Re-run `literature_volume.py` to produce one. Counts are
keyword-based and are an order-of-magnitude guide, not a systematic-review yield.

A caution on the corpus. `literature/corpus.json` is a **relevance-ranked method
sample** of 276 records, not a volume census, and its per-year histogram is
therefore not a publication trend: OpenAlex ranks a keyword query by relevance, so
older heavily indexed surveys dominate and recent years look artificially sparse.
Use `volume.json` for magnitude and `corpus.json` for which methods exist.

## 1. How large is the field

The specific problem this study addresses is **small**. Since 2015, works whose
title or abstract mention solar radiation missing-data imputation number in the
tens, not the thousands:

| Topic (title/abstract match, 2015–2025) | Total |
|---|---:|
| solar radiation missing-data imputation | 53 |
| solar irradiance gap filling | 55 |
| PV power missing-data imputation | 72 |
| solar radiation reconstruction | 831 |
| remote sensing gap filling | 891 |
| satellite cloud removal / inpainting | 120 |
| spatiotemporal kriging | 905 |
| diffusion for time-series imputation | 185 |
| diffusion for spatiotemporal data | 43 |
| ensemble spread calibration | 76 |
| graph signal processing | 12 296 |
| conformal prediction | 15 482 |

Three consequences:

1. **Solar-specific imputation is thin.** Roughly 50–120 papers each for the three
   solar/PV phrasings, against ~900 for the generic remote-sensing phrasing. The
   sub-field this manuscript occupies is not crowded.
2. **The method supply comes from elsewhere.** Diffusion imputation and calibration
   machinery is developed mainly on traffic, air quality, medical and general
   time-series benchmarks; solar radiation is a late adopter. Importing a
   well-characterised method into a thin application area is the realistic
   contribution shape here, which is what the manuscript already claims.
3. **The generic terms are not informative.** `graph signal processing` (12 k) and
   `conformal prediction` (15 k) are whole sub-disciplines; a count for them says
   nothing about this paper's positioning. They are listed only to show why the
   narrow rows are the meaningful ones.

## 2. Closest prior work, with the comparison fields the manuscript needs

Ordered by closeness to the present design.

| Work | Task / target | Method | Probabilistic? | Missingness handled | Validation | Difference from this study |
|---|---|---|---|---|---|---|
| Tashiro et al. 2021, CSDI (NeurIPS) | multivariate time-series imputation | conditional score-based diffusion | yes | random point and block missingness | simulated and benchmark series | Not solar; no spatial grid, no physical bound, no monthly aggregate constraint |
| Liu et al. 2023, PriSTI | spatiotemporal imputation | conditional diffusion with spatial-temporal transformer, GCN prior | yes | random and block missingness | traffic / air-quality benchmarks | Reported CRPS is 10–100× larger than RMSE (implausible); no radiative constraints |
| Liu et al. 2024, MTSCI | multivariate series imputation | conditional diffusion, consistent imputation | yes | random missingness | benchmark series | Series-based, no spatial field, no grid-graph penalty |
| Zhang et al. 2024, SaSDim | spatial time-series imputation | self-adaptive noise-scaling diffusion | yes | random missingness | benchmark spatial series | Noise schedule is the object of study; no physical or area constraints |
| Basakın et al. 2023, Energy Convers. Manag. | solar radiation imputation | hybrid differential evolution + learning | no | point/record gaps | station data | Point estimate only; no uncertainty, no spatial field |
| Girimurugan et al. 2023, Int. J. Photoenergy | solar irradiance prediction with missing data | deep learning | no | missing inputs | station/satellite series | Forecasting framing, no imputation of a spatial field, no calibration |
| Fan et al. 2023, PACMMOD | PV data imputation | spatio-temporal denoising **graph autoencoder** | no | spatial-temporal gaps | PV plant fleet | Learned graph over plants, not a fixed image grid; no diffusion, no calibration |
| Zhang et al. 2025, Renewable Energy | PV power imputation | weather- and context-aware hybrid transformer | no | missing PV power | plant data | Transformer point imputation; no generative ensemble |
| Mital et al. 2020, Front. Water | spatiotemporal precipitation | sequential random forests | no | spatial-temporal gaps | gridded precipitation | Non-solar; deterministic; no uncertainty calibration |
| Mardani et al. 2025, *Commun. Earth Environ.* 6, 124 | km-scale downscaling | residual corrective diffusion + UNet mean | yes | not an imputation task | regional reanalysis | Shares the "UNet mean + diffusion residual" idea; different task, no missing-data mask, no calibration study |
| Shuman et al. 2013 | graph signal processing theory | Laplacian / spectral framework | — | — | — | Theory reference; this study applies it on a regular grid and finds no measurable effect |
| Gneiting et al. 2007 (both papers) | scoring rules and calibration | CRPS, calibration/sharpness | — | — | — | Evaluation framework the manuscript adopts |

### What the table makes visible

- **None of the closest works is solar *and* spatial *and* probabilistic *and*
  constrained.** The present study's combination is unoccupied, which supports the
  contribution claim, and also means there is no directly comparable published
  number to quote. Every comparison in the manuscript is therefore internal.
- **The strongest external comparator would be CSDI or PriSTI re-trained on this
  protocol.** Both are time-series methods without a spatial field, so the
  adaptation is non-trivial, but leaving them out means the "conditional
  generative imputation" line of work is cited but not measured against.
- **Graph-based imputation exists in the PV literature** (Fan et al. 2023) but uses
  a *learned* graph over plants. The manuscript correctly does not claim to be the
  first graph-based imputer; it claims a fixed-grid penalty, which its own ablation
  now shows to be inert (see `../outputs/ablation_graph/REPORT.md`).

## 3. Coverage of the manuscript's own reference list

The draft cites 14 works. Mapping them onto the corpus above:

| Cited | Role in the paper | Present in corpus? |
|---|---|---|
| NASA POWER (2 refs) | data provenance | no (grey literature) |
| Allen et al. 1998 | FAO-56 solar geometry | no (report) |
| Ronneberger et al. 2015 | U-Net architecture | partially (U-Net bucket) |
| Lakshminarayanan et al. 2017 | deep ensembles | no (corpus sampled 2018+) |
| Shuman et al. 2013 | graph signal processing | yes |
| Ho et al. 2020, Song et al. 2021, Salimans & Ho 2022 | diffusion foundations | yes (diffusion bucket) |
| Tashiro et al. 2021 (CSDI) | conditional diffusion imputation | yes |
| Mardani et al. 2025 (CorrDiff) | residual corrective diffusion | yes |
| Price et al. 2025 | ML weather forecasting | via downscaling bucket |
| Gneiting & Raftery 2007, Gneiting et al. 2007 | scoring and calibration | yes |
| **Missing** | | |
| — | solar-specific imputation literature | **absent** |
| — | time-series diffusion imputation after CSDI (PriSTI, MTSCI, SaSDim, ImDiffusion) | **absent** |
| — | remote-sensing gap filling / cloud removal | **absent** |
| — | kriging / geostatistical interpolation baselines | cited only as "not full kriging" |
| — | **a solar-specific diffusion forecasting paper that already exists**: Hatanaka et al. 2023, "Diffusion models for high-resolution solar forecasts" (arXiv:2302.00170) | **absent** |

CorrDiff also produced a further own-citation correction: the manuscript cites it as
an arXiv record, but it has since been published — *Communications Earth &
Environment* **6**, 124 (2025), DOI `10.1038/s43247-025-02042-5`, published online
24 February 2025. The reference should be updated to the journal version.

The reference list is a *method* bibliography, not an *application* one. Two edits
follow from that:

1. **Add solar-imputation prior work** (at least Basakın et al. 2023 and
   Girimurugan et al. 2023) so the paper is situated in its own application area
   rather than only in the diffusion literature.
2. **Add the post-CSDI imputation line** (PriSTI, MTSCI, SaSDim) because those are
   the methods a reader will ask to see compared, and because PriSTI's reported
   metric behaviour is itself worth citing as a cautionary note.

## 4. Files

| File | Contents |
|---|---|
| `literature/corpus.json` | 276 relevance-ranked records with DOI, authors, year, venue, citations, abstract |
| `literature/corpus.txt` | one-line-per-record reading list |
| `literature/summary.json` | per-bucket counts, venue counts, per-year counts of the corpus |
| `literature/volume.json` | yearly title/abstract counts per topic (OpenAlex) |
| `literature_survey.py` | rebuilds the corpus |
| `literature_volume.py` | rebuilds the volume table |
| `literature_summary.py` | rebuilds the summary |
