# C220 Paper 1 — manuscript-aligned code review, 2026-11-09

## Scope and provenance

The `scripts/` directory is a **verbatim preserved copy** of the 25 selected server scripts, including historical alternatives. Files in `publication_aligned_scripts/` are new, explicit modifications; these are **not claimed to be the original figure-generating sources**. Original laptop-created code may not be represented in the archive. The original script inventory remains in `SCRIPT_REVIEW_SELECTION.tsv`.

## Paper 1 biomass correction

Final lab/manuscript input for 50% DSW Experiment 1 is **0.2575 g**. The historical `scripts/03_BIOMASS/plot_C220_DSW_dry_biomass_final.py` used 0.3575 g. `publication_aligned_scripts/plot_C220_DSW_dry_biomass_PAPER1_CORRECTED.py` contains a one-number correction, preserving the original. No other biomass data have been changed.

## Extracellular carbohydrate

The manuscript describes A490–A700 correction, and **only 1:5 and 1:10** dilution readings retained for the final biological-replicate summaries. A historical server script aggregated the undiluted (neat), 1:5, and 1:10 measurements together. `publication_aligned_scripts/plot_crude_eps_per_g_biomass_A700_PAPER1_DILUTIONS.py` is a publication-method-aligned variant: neat values are retained in the raw/QC table but excluded from downstream per-replicate aggregation and statistical tests. The exact manuscript figure output is **not verified** without the original workbook. Other historical EPS scripts are preserved unchanged.

## RNA-seq repository and GEO evidence

GEO series accession stated by the researcher: **GSE353704** — https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE353704 . The public accession page was not retrievable independently through the web tool on 2026-11-09; do not present this note as verification of public release or current NCBI processing.

GEO V7 audit as reported by user on 2026-11-05: 9/9 saved DESeq2 raw-count sample columns match the complete raw input; saved DESeq2 model contains 8,275 filtered genes versus 13,455 complete quantified features. Legacy 13,455-gene normalized matrix is **not exactly interchangeable** with counts(dds, normalized=TRUE) from the original saved DESeq2 model: nine sample-specific normalization diagnostics were flagged for gene-specific differences. The exact 8,275-model export should be preferred for model-supported genes when separately verified. The old Figure 4 script enforces a legacy matrix SHA256 and may need input/provenance-specific adjustment; the original was deliberately not changed.

Upload to GEO is separate from confirming that all processed files and manuscript figures reproduce from deposited records. No reanalysis of original DESeq2 data or direct GEO retrieval was performed when preparing this package.

## Unresolved publication checks

- Compare the corrected biomass panel numerically against the final figure.
- Reproduce the carbohydrate values and statistics from original workbooks (1:5 + 1:10 only); check figure and supplementary values before replacing outputs.
- Check all 11 Figure 4 displayed genes against the 8,275-gene saved model and compare plotted values against the original final figure; do not relabel an older matrix as model-normalized without confirmation.
- Check original figure-producing laptop scripts, if obtainable, before making a historical provenance assertion.
- SEM image-field observations must not be treated as biological culture replicates.
- Confirm GEO series access and nine sample mappings using public record or official depositor correspondence.

No original input data were modified and no GitHub upload was performed.
