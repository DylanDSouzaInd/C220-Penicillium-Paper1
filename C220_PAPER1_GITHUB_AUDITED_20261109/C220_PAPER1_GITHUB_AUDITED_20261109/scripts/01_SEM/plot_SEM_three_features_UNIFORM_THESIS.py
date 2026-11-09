#!/usr/bin/env python3

from pathlib import Path
import itertools
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.stats.multitest import multipletests

BASE = Path(".")
OUT = BASE / "SEM_UNIFORM_THESIS_PLOTS"
OUT.mkdir(exist_ok=True)

# ---------------------------------------
# Find image-level feature table
# ---------------------------------------
candidate_tables = []
for p in [
    BASE / "results_sem_eps",
    BASE / "sem_reanalysis_robust/v2",
    BASE,
]:
    if p.exists():
        candidate_tables.extend(list(p.rglob("*.tsv")))

needed_features = [
    "eps_proxy_index",
    "branchpoints_per_100um2",
    "skel_length_um_per_100um2",
    "canny_fraction",
]

def read_possible_table(path):
    try:
        return pd.read_csv(path, sep="\t")
    except Exception:
        return None

feature_table = None
feature_table_path = None

for f in candidate_tables:
    df = read_possible_table(f)
    if df is None or df.empty:
        continue
    cols = set(df.columns)
    if any(x in cols for x in needed_features):
        group_candidates = [c for c in df.columns if c.lower() in ["group", "condition", "group_label", "dsw", "salinity"]]
        if group_candidates:
            feature_table = df.copy()
            feature_table_path = f
            break

if feature_table is None:
    raise SystemExit(
        "Could not find an image-level table with SEM feature columns. "
        "Run: find results_sem_eps sem_reanalysis_robust -type f -name '*.tsv'"
    )

print(f"[INFO] Using image-level table: {feature_table_path}")

# ---------------------------------------
# Harmonize group labels
# ---------------------------------------
group_col = None
for c in ["group", "condition", "group_label", "DSW", "salinity"]:
    if c in feature_table.columns:
        group_col = c
        break

if group_col is None:
    raise SystemExit("No group/condition column found.")

def norm_group(x):
    s = str(x)
    if "DSW0" in s or s in ["0", "0.0", "0% DSW"]:
        return "0% DSW"
    if "DSW10" in s or s in ["10", "10.0", "10% DSW"]:
        return "10% DSW"
    if "DSW20" in s or s in ["20", "20.0", "20% DSW"]:
        return "20% DSW"
    m = re.search(r"(0|10|20)", s)
    if m:
        return f"{m.group(1)}% DSW"
    return s

feature_table["condition_plot"] = feature_table[group_col].map(norm_group)
feature_table = feature_table[feature_table["condition_plot"].isin(["0% DSW", "10% DSW", "20% DSW"])].copy()

order = ["0% DSW", "10% DSW", "20% DSW"]

# ---------------------------------------
# Feature display settings
# ---------------------------------------
features = {
    "branchpoints_per_100um2": {
        "title": "SEM branchpoint density across salinity",
        "ylabel": "Branchpoint density\n(branchpoints per 100 µm²)",
        "filename": "SEM_branchpoints_UNIFORM_THESIS",
    },
    "skel_length_um_per_100um2": {
        "title": "SEM skeleton length across salinity",
        "ylabel": "Skeleton length\n(µm per 100 µm²)",
        "filename": "SEM_skeleton_length_UNIFORM_THESIS",
    },
    "eps_proxy_index": {
        "title": "SEM-derived EPS proxy index across salinity",
        "ylabel": "EPS proxy index\n(robust z-score composite)",
        "filename": "SEM_EPS_proxy_UNIFORM_THESIS",
    },
    "canny_fraction": {
        "title": "SEM edge/texture fraction across salinity",
        "ylabel": "Canny edge fraction",
        "filename": "SEM_canny_fraction_UNIFORM_THESIS",
    },
}

# keep only features present
features = {k:v for k,v in features.items() if k in feature_table.columns}

# ---------------------------------------
# Styling
# ---------------------------------------
colors = {
    "0% DSW": "#4C78A8",
    "10% DSW": "#F58518",
    "20% DSW": "#54A24B",
}
light_colors = {
    "0% DSW": "#C9D8E8",
    "10% DSW": "#F9DDBF",
    "20% DSW": "#CDE5C9",
}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 12,
    "axes.titlesize": 18,
    "axes.labelsize": 15,
    "xtick.labelsize": 14,
    "ytick.labelsize": 13,
    "legend.fontsize": 10,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

def p_to_stars(p):
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"

def violin_box_scatter(ax, data_by_group, feature):
    positions = np.arange(len(order))

    data_list = [data_by_group[g] for g in order]

    parts = ax.violinplot(
        data_list,
        positions=positions,
        widths=0.72,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )

    for body, g in zip(parts["bodies"], order):
        body.set_facecolor(light_colors[g])
        body.set_edgecolor("none")
        body.set_alpha(0.8)

    # boxplots
    bp = ax.boxplot(
        data_list,
        positions=positions,
        widths=0.32,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color="black", linewidth=2.0),
        boxprops=dict(linewidth=1.2, color="black"),
        whiskerprops=dict(linewidth=1.2, color="black"),
        capprops=dict(linewidth=1.2, color="black"),
    )

    for patch, g in zip(bp["boxes"], order):
        patch.set_facecolor(colors[g])
        patch.set_alpha(0.28)

    # jittered points, deterministic
    rng = np.random.default_rng(20260618)
    for i, g in enumerate(order):
        y = np.asarray(data_by_group[g], dtype=float)
        jitter = rng.normal(0, 0.045, size=len(y))
        ax.scatter(
            np.full(len(y), positions[i]) + jitter,
            y,
            s=48,
            color=colors[g],
            edgecolor="black",
            linewidth=0.45,
            alpha=0.9,
            zorder=3,
        )

    # median bars
    for i, g in enumerate(order):
        med = np.median(data_by_group[g])
        ax.plot([i - 0.19, i + 0.19], [med, med], color="black", linewidth=2.2, zorder=4)

    ax.set_xticks(positions)
    ax.set_xticklabels(order)
    ax.set_xlabel("Condition")
    ax.set_ylabel(features[feature]["ylabel"])
    ax.set_title(features[feature]["title"], pad=16)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.18)
    ax.set_axisbelow(True)

    # n labels, all from the exact plotted data
    ymin, ymax = ax.get_ylim()
    yspan = ymax - ymin
    for i, g in enumerate(order):
        y = np.asarray(data_by_group[g], dtype=float)
        ax.text(
            i,
            np.nanmax(y) + 0.03 * yspan,
            f"n={len(y)}",
            ha="center",
            va="bottom",
            fontsize=12,
        )

def add_stats(ax, data_by_group, feature):
    data_list = [data_by_group[g] for g in order]

    H, p_kw = stats.kruskal(*data_list)

    pairs = []
    raw_p = []
    for i, j in itertools.combinations(range(len(order)), 2):
        u, p = stats.mannwhitneyu(data_list[i], data_list[j], alternative="two-sided")
        pairs.append((i, j, p))
        raw_p.append(p)

    padj = multipletests(raw_p, method="fdr_bh")[1]

    pair_rows = []
    sig_pairs = []
    for (i, j, p), q in zip(pairs, padj):
        stars = p_to_stars(q)
        pair_rows.append({
            "feature": feature,
            "comparison": f"{order[i]} vs {order[j]}",
            "p_uncorrected": p,
            "p_adj_BH_3_pairwise_tests": q,
            "stars": stars,
            "n1": len(data_list[i]),
            "n2": len(data_list[j]),
        })
        if stars != "ns":
            sig_pairs.append((i, j, q, stars))

    ymin, ymax = ax.get_ylim()
    yspan = ymax - ymin

    # more top space
    top_needed = ymax + (0.16 + 0.06 * len(sig_pairs)) * yspan
    ax.set_ylim(ymin, top_needed)
    ymin, ymax = ax.get_ylim()
    yspan = ymax - ymin

    # KW p box in same location/style
    ax.text(
        0.985,
        0.965,
        f"Kruskal–Wallis p = {p_kw:.2e}",
        ha="right",
        va="top",
        transform=ax.transAxes,
        fontsize=11,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="0.7", alpha=0.95),
    )

    # brackets
    base = max([np.nanmax(v) for v in data_list]) + 0.08 * yspan
    step = 0.075 * yspan
    h = 0.018 * yspan

    for k, (i, j, q, stars) in enumerate(sig_pairs):
        y = base + k * step
        ax.plot([i, i, j, j], [y, y + h, y + h, y], color="black", linewidth=1.4)
        ax.text((i + j) / 2, y + h + 0.006 * yspan, stars, ha="center", va="bottom", fontsize=14, weight="bold")

    return H, p_kw, pair_rows

all_stats = []
all_group_summary = []

for feature, meta in features.items():
    df = feature_table[["condition_plot", feature]].copy()
    df[feature] = pd.to_numeric(df[feature], errors="coerce")
    df = df.dropna()

    data_by_group = {}
    for g in order:
        vals = df.loc[df["condition_plot"] == g, feature].astype(float).values
        data_by_group[g] = vals
        all_group_summary.append({
            "feature": feature,
            "group": g,
            "n_plotted": len(vals),
            "median": float(np.median(vals)),
            "mean": float(np.mean(vals)),
            "q1": float(np.percentile(vals, 25)),
            "q3": float(np.percentile(vals, 75)),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
        })

    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    violin_box_scatter(ax, data_by_group, feature)
    H, p_kw, pair_rows = add_stats(ax, data_by_group, feature)

    for r in pair_rows:
        r["kruskal_H"] = H
        r["kruskal_p"] = p_kw
    all_stats.extend(pair_rows)

    fig.tight_layout()
    fig.savefig(OUT / f"{meta['filename']}.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / f"{meta['filename']}.pdf", bbox_inches="tight")
    plt.close(fig)

pd.DataFrame(all_group_summary).to_csv(OUT / "SEM_UNIFORM_group_summary_plotted.tsv", sep="\t", index=False)
pd.DataFrame(all_stats).to_csv(OUT / "SEM_UNIFORM_pairwise_mannwhitney_BH.tsv", sep="\t", index=False)

report = f"""SEM UNIFORM THESIS PLOTS
========================

Input image-level table:
{feature_table_path}

Plotting rule:
No feature-specific outlier removal was applied in this plotting script.
Therefore n values are taken directly from the plotted image-level data.

Expected n:
0% DSW = {sum(feature_table['condition_plot'] == '0% DSW')}
10% DSW = {sum(feature_table['condition_plot'] == '10% DSW')}
20% DSW = {sum(feature_table['condition_plot'] == '20% DSW')}

Statistics:
Kruskal-Wallis across 0%, 10%, and 20% DSW.
Post hoc pairwise Mann-Whitney U tests.
Benjamini-Hochberg FDR correction across the three pairwise tests per feature.

Outputs:
{OUT}
"""
(OUT / "README_SEM_UNIFORM_THESIS_PLOTS.txt").write_text(report)
print(report)
