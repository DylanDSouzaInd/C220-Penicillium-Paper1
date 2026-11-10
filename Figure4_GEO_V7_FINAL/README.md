# Figure 4 — GEO V7 saved-model aligned analysis

This directory contains the final **v2 candidate** plotting script, launcher, source-value cross-check table, adjusted-p-value audit and manuscript-style preview image for Paper 1 Figure 4.

**Inputs and provenance:** the original saved DESeq2 model had 8,275 retained genes and nine biological samples. The `01_FIGURE4_108_VALUE_COMPARISON.tsv` records the 12 panel genes × nine samples from the saved-model normalized export and the October legacy matrix. The original DESeq2 pairwise contrast CSVs, not a rerun, supply significance annotations. The older normalized matrix differs for 86/108 values; do not confuse it with the saved-model export.

**Requirements:** `python`, `pandas`, `numpy`, `matplotlib`; original saved-model normalized-count TSV and DESeq2 contrast CSVs. The provided launcher uses `/data/dylan` paths from the original server and should be adapted when running elsewhere.

**Execute on original server:** copy both scripts into `/data/dylan` and run `bash /data/dylan/run_C220_Figure4_FINAL_VERIFIED_v2.sh` in an environment containing the listed Python packages. The script uses read-only sources and refuses to overwrite a nonempty output directory. It independently checks the cross-check values against the 8,275-gene model export and loads the original three DESeq2 contrast CSVs before plotting.

**Reported audit:** the uploaded contrast audit has exactly 36 gene/contrast rows, 18 FDR-significant comparisons and no internal mismatch between `padj`, stars and `bracket` fields. This package does not include the full original model/contrast files or the output verification JSON from your server; these must be retained externally to independently repeat the full verification.

**Visual summary:** dots are individual biological replicates; horizontal black bars are the mean of the three `log10(normalized count + 1)` values in each condition (not the log of the arithmetic mean raw normalized count). Brackets use DESeq2 adjusted p < 0.05. 0%, 10% and 20% are distinct conditions.

**Do not overwrite** the October historical plot, source matrix or manuscript figure until the publication version is formally selected.
