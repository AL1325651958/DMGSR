# Draft evidence and editorial audit

## Claim–evidence ledger

| Claim / figure | Evidence | Boundary |
|---|---|---|
| Domain and chronological split / Fig. 1 | Local POWER loader, train-only mean, archived month keys | Product reference, not station truth |
| Architecture / Fig. 2 | Saved configuration and implementation inspected | Fixed-grid penalties, no learned graph attention |
| Matched aggregate skill / Fig. 3 | `evidence/results.json`, aligned arrays, common missing mask | 547 dependent values in previously inspected 2011 |
| Reliability / Fig. 4 | Empirical quantiles of saved ensembles | 8, 12 and 20 members; no calibration guarantee |
| Calibration / Fig. 5 | `fusion_search.csv`, `unet_search.csv` | 2010-only search; retrospective design overall |
| Monthly variation / Fig. 6 | `monthly_metrics.csv` | Different masks confound seasonal interpretation |
| Fixed July illustration / Fig. 7 | `fixed_case_members.npz`, `fixed_case_recalibrated.npz` | 112 values, excluded from headline pooling |
| Sampling evolution / Fig. 8 | `diffusion_trace.npz` | One realization; chunk output includes corrections |
| Paired uncertainty interval | 2,000 month-block resamples in `build_evidence.py` | Conditional descriptive interval, no selection uncertainty |

## Terminology decisions

- Use **daily mean irradiance (W m−2)** for converted POWER daily irradiation.
- Use **reference**, not observed station ground truth.
- Use **graph-regularized**, not adaptive graph-attention U-Net.
- Use **development reporting period**, not untouched test set.
- Distinguish 12 samples from one diffusion checkpoint and 8 independently fitted U-Net members.
- Use **empirical fused ensemble**, not a rigorously derived joint posterior.
- Distinguish sampling constraints from fused-ensemble physical support.

## Manuscript allocation and removed claims

Introduction: gap recovery, conditional generative methods and the calibrated-comparator question. Data: source provenance, aggregate conditioning and chronological partitions. Methods: implementation, fixed settings, calibration and metrics. Results: pooled skill, reliability, heterogeneity and fixed-case dynamics. Discussion: interpretation, missing ablation, physical support and independent validation. Eight figures belong in the main text; machine-readable ensembles, searches and bootstrap details accompany the draft as evidence files, not invented supplementary experiments.

Removed: unsupported 18% graph-ablation improvement; unmatched six-method superiority percentages; adaptive/wind-aligned graph claims; treating a 112-value case as a full test; broad climate-scenario and operational forecasting claims. Revised the title and abstract around the implemented and evaluated framework.

## Reference verification scope

Fourteen references are supplied. NASA POWER provenance and API documentation, the FAO-56 source, and the listed arXiv method records were inspected. Primary Nature metadata and official Crossref metadata were used for bibliographic checks; the latter are saved in `evidence/reference_metadata.json`. Metadata verification is not a claim of full-text verification. Citations support general background and method attribution, not the new numerical results. The Mardani reference is explicitly an arXiv record; no journal venue is inferred. No quotations or fabricated references were inserted.

## Figure QA

Eight vector PDFs were checked for rendered collisions, clipping, panel alignment and extractable text. All minimum text sizes are 6 pt in the standalone files. Seven collision audits have no warnings. Figure 1 has one automated filled-region-edge warning for “not used here”; final visual inspection shows the timeline note is separated from the bars and does not obscure data. Box-contained labels in Fig. 2 and numeric bar annotations in Fig. 3 are intentional. Individual JSON audit records are in `figures/qa`; strict panel checks are saved beside the figures. The all-figure contact sheet was visually inspected. PNGs are 300 dpi previews; vector PDF/SVG files are the publication-layout masters. These checks do not certify a particular journal's final production requirements.

## Final document checks

The compiled draft has 17 pages including highlights, eight embedded figures and 14 references. Citation keys and figure/equation labels resolve without missing or duplicate keys. No extracted text block crosses a PDF page boundary. Rendered pages were inspected, including every figure and the final reference page. The compiler retains a minor 2.608-pt overfull footer-box warning and a reference-paragraph spacing warning; neither causes visible clipping. Detailed checks and approximate before/after section word counts are in `qa/manuscript_checks.json` and `qa/section_word_counts.json`. Current section counts (including captions and mathematical labels) are Introduction 492, Data 483, Methods 1,391, Results 1,040, Discussion 666 and Conclusions 87; these are editorial estimates, not a journal word-limit certification.
