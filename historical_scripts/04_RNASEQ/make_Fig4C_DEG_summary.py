#!/usr/bin/env python3

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--comp_10_vs_0", required=True)
parser.add_argument("--comp_20_vs_0", required=True)
parser.add_argument("--comp_20_vs_10", required=True)
parser.add_argument("--padj", type=float, default=0.05)
parser.add_argument("--lfc", type=float, default=1.0)
args = parser.parse_args()

OUT = Path("/data/dylan/01_ONT_PCRcDNARNASeq/salinity_rnaseq_analysis/10_final_thesis_paper_outputs/26_RESULT5_FIG4_RNASEQ")
FIGS = OUT / "figures"
TABLES = OUT / "tables"
FIGS.mkdir(parents=True, exist_ok=True)
TABLES.mkdir(parents=True, exist_ok=True)

def read_table(path):
    path = Path(path)
    df = pd.read_csv(path, sep=None, engine="python")
    if "gene_id" not in df.columns:
        first = df.columns[0]
        if first.lower() in ["", "unnamed: 0", "index"]:
            df = df.rename(columns={first: "gene_id"})
        else:
            df = df.reset_index().rename(columns={"index": "gene_id"})
    return df

def find_col(df, candidates):
    lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    raise ValueError(f"Could not find any of {candidates} in columns: {list(df.columns)}")

def count_deg(path, label):
    df = read_table(path)
    gene_col = find_col(df, ["gene_id", "gene", "id"])
    lfc_col = find_col(df, ["log2FoldChange", "log2FC", "logFC"])
    padj_col = find_col(df, ["padj", "FDR", "qvalue", "q_value"])

    df[lfc_col] = pd.to_numeric(df[lfc_col], errors="coerce")
    df[padj_col] = pd.to_numeric(df[padj_col], errors="coerce")

    sig = df.dropna(subset=[lfc_col, padj_col])
    sig = sig[(sig[padj_col] < args.padj) & (sig[lfc_col].abs() >= args.lfc)]

    up = int((sig[lfc_col] > 0).sum())
    down = int((sig[lfc_col] < 0).sum())

    clean = sig[[gene_col, lfc_col, padj_col]].copy()
    clean.columns = ["gene_id", "log2FoldChange", "padj"]
    clean["comparison"] = label
    clean["direction"] = np.where(clean["log2FoldChange"] > 0, "upregulated", "downregulated")

    return {
        "comparison": label,
        "upregulated": up,
        "downregulated": down,
        "total_DEGs": up + down,
        "input_file": str(path),
    }, clean

items = [
    ("10% vs 0%", args.comp_10_vs_0),
    ("20% vs 0%", args.comp_20_vs_0),
    ("20% vs 10%", args.comp_20_vs_10),
]

summary_rows = []
deg_tables = []

for label, path in items:
    row, clean = count_deg(path, label)
    summary_rows.append(row)
    deg_tables.append(clean)

summary = pd.DataFrame(summary_rows)
all_deg = pd.concat(deg_tables, ignore_index=True)

summary.to_csv(TABLES / "Figure_4C_DEG_summary_counts.tsv", sep="\t", index=False)
all_deg.to_csv(TABLES / "Figure_4C_all_significant_DEGs_used.tsv", sep="\t", index=False)

# Plot
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 12,
    "axes.titlesize": 17,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

x = np.arange(len(summary))
width = 0.36

fig, ax = plt.subplots(figsize=(8.8, 5.6))

ax.bar(x - width/2, summary["upregulated"], width, label="Upregulated")
ax.bar(x + width/2, summary["downregulated"], width, label="Downregulated")

ax.set_title("Differential gene expression across salinity conditions", pad=14)
ax.set_ylabel("Number of DE genes")
ax.set_xlabel("Pairwise comparison")
ax.set_xticks(x)
ax.set_xticklabels(summary["comparison"])

max_y = max(summary["upregulated"].max(), summary["downregulated"].max(), 1)
for i, row in summary.iterrows():
    ymax = max(row["upregulated"], row["downregulated"])
    ax.text(i, ymax + max_y * 0.05, f"Total={row['total_DEGs']}",
            ha="center", va="bottom", fontsize=11)

ax.text(
    0.5, 0.98,
    f"FDR < {args.padj}; |log2FC| ≥ {args.lfc}",
    transform=ax.transAxes,
    ha="center",
    va="top",
    fontsize=10.5,
    bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="0.7")
)

ax.legend(frameon=False)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", alpha=0.2)
ax.set_axisbelow(True)

fig.tight_layout()
fig.savefig(FIGS / "Figure_4C_DEG_summary_barplot.png", dpi=300, bbox_inches="tight")
fig.savefig(FIGS / "Figure_4C_DEG_summary_barplot.pdf", bbox_inches="tight")

print(summary.to_string(index=False))
print("Saved:")
print(FIGS / "Figure_4C_DEG_summary_barplot.png")
print(TABLES / "Figure_4C_DEG_summary_counts.tsv")
