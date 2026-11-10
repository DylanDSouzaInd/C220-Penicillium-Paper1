# Figure 4 visual refinement v2

Run using the existing GEO V7 108-value crosscheck, original saved DESeq2 normalized-count matrix and original DESeq2 contrast CSVs.

```bash
cd /data/dylan
conda activate ont_dsw_rnaseq
bash run_C220_Figure4_FINAL_VERIFIED_v2.sh
```

Place both `C220_Figure4_GEO_V7_FINAL_VERIFIED_v2.py` and `run_C220_Figure4_FINAL_VERIFIED_v2.sh` in `/data/dylan` first.

Differences from v1:
- Mean bars centered on each 0/10/20% condition, with biological replicate dots on top and symmetric offsets.
- Bold italic *Penicillium* in the main title (STIX mathtext).
- Larger panel titles (14pt), y-axis labels (14pt), and figure title (21pt).
- New output folder (`/data/dylan/C220_FIGURE4_GEO_V7_FINAL_VERIFIED_v2_20261110`); original outputs untouched.

All count and adjusted-p-value checks retained. The server output must be validated against actual DESeq2 contrasts; any local sample render generated from fixtures is **not** valid evidence for significance brackets.
