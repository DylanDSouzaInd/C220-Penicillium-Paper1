# *Penicillium* sp. C220 — Dead Sea Water response (Paper 1)

Analysis-code provenance for a study of the Dead Sea sediment isolate *Penicillium* sp. C220, comparing fungal biomass, SEM-derived morphology, crude extracellular carbohydrate, Oxford Nanopore RNA-seq and RT-qPCR. The RNA-seq, SEM and carbohydrate validation focus on 0%, 10% and 20% (v/v) Dead Sea Water; RNA-seq includes three biological replicates per condition. A wider gradient was used for biomass measurements.

## Where to start

- [`Figure4_GEO_V7_FINAL/`](Figure4_GEO_V7_FINAL/) — **model-aligned Figure 4 v2 code and evidence**: verified plotting workflow, comparison table, 36 adjusted-p-value audit and preview figure.
- [`documentation/`](documentation/) — original script selection manifest, historical-vs-corrected notes, publication changelog, and outstanding release checks.
- [`historical_scripts/`](historical_scripts/) — **25 historically collected scripts**, retained unchanged as previously uploaded. 
- [`publication_aligned_scripts/`](publication_aligned_scripts/) — the separately labelled biomass and carbohydrate correction variants; historical originals remain unchanged.

## Data and analysis provenance

NCBI Gene Expression Omnibus accession: **[GSE353704](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE353704)** (the accession reported by the investigator; public status and file availability should be confirmed directly).

The GEO V7 audit of the original DESeq2 model reports nine samples, 8,275 model-retained genes and exact raw-count correspondence for those genes; the 13,455-feature legacy normalized matrix is **not** interchangeable with the saved-model normalized counts. See the Figure 4 audit directory for the 12-gene-specific cross-check. Differential-expression p-values are sourced from the original DESeq2 contrast files. No DESeq2 model was refitted for this figure.

## Scientific interpretation and caveats

The analyses address DSW-associated morphology, surface/matrix-related phenotypes and transcription. Crude carbohydrate measurements do not chemically establish purified EPS. Multiple SEM fields must not be treated as independent biological cultures. RT-qPCR provides directional support; the previously reported independent-test findings do not remain significant after BH correction.

## Reproducibility and citation

This is a **provenance and figure-level review archive**, not an independently replayed end-to-end pipeline. Some historical scripts require original input workbooks, annotation assets and server-specific paths. Read [`documentation/PUBLIC_RELEASE_CHECKLIST.md`](documentation/PUBLIC_RELEASE_CHECKLIST.md) before public release. The complete source model and original DESeq2 contrast files are not bundled.

Cite the associated paper when publication information is finalized. No reuse license is granted by default until the repository owner selects and adds one.
