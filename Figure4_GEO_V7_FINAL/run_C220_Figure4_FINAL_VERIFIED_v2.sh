#!/usr/bin/env bash
set -euo pipefail
ROOT=/data/dylan
MODEL="$ROOT/C220_GEO_V7_MODEL_AUDIT_20261105/04_DESeq2_original_fitted_model_normalized_counts_8275.tsv"
COMP="$ROOT/C220_FIGURE4_GEO_V7_FINAL_20261110/01_crosscheck/01_FIGURE4_108_VALUE_COMPARISON.tsv"
DESEQ="$ROOT/01_ONT_PCRcDNARNASeq/salinity_rnaseq_analysis/06_deseq2_v2/results_all_samples"
SCRIPT="$ROOT/C220_Figure4_GEO_V7_FINAL_VERIFIED_v2.py"
OUT="$ROOT/C220_FIGURE4_GEO_V7_FINAL_VERIFIED_v2_20261110"
export MPLBACKEND=Agg
export MPLCONFIGDIR="$ROOT/.matplotlib_cache"
mkdir -p "$MPLCONFIGDIR"
for p in "$MODEL" "$COMP" "$SCRIPT"; do
  [[ -f "$p" ]] || { echo "ERROR Missing $p" >&2; exit 2; }
done
[[ -d "$DESEQ" ]] || { echo "ERROR Missing DESeq2 directory: $DESEQ" >&2; exit 2; }
python -u "$SCRIPT" --comparison "$COMP" --model-counts "$MODEL" \
   --deseq-dir "$DESEQ" --outdir "$OUT"
echo '=== Figure / audit output ==='
ls -lh "$OUT"
echo '=== Verified DESeq2 significance counts ==='
python - "$OUT/Figure4_DESeq2_36_COMPARISON_AUDIT.tsv" <<'PY'
import pandas as pd,sys
d=pd.read_csv(sys.argv[1],sep='\t')
print(d.groupby('contrast',sort=False)['bracket'].sum().to_string())
print('36 tests:',len(d),'brackets:',int(d['bracket'].sum()))
PY
