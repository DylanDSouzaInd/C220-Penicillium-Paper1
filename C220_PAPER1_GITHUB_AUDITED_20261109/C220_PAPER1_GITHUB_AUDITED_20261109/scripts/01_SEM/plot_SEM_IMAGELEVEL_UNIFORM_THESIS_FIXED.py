#!/usr/bin/env python3

from pathlib import Path
import itertools
import re
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.stats.multitest import multipletests

parser = argparse.ArgumentParser()
parser.add_argument("--input", default="", help="Optional explicit image-level TSV")
parser.add_argument("--outdir", default="SEM_IMAGELEVEL_UNIFORM_THESIS_PLOTS")
args = parser.parse_args()

BASE = Path(".")
OUT = Path(args.outdir)
OUT.mkdir(parents=True, exist_ok=True)

FEATURES = {
    "branchpoints_per_100um2": {
        "title": "SEM branchpoint density across salinity",
        "ylabel": "Branchpoint density\n(branchpoints per 100 µm²)",
        "filename": "SEM_branchpoints_IMAGELEVEL_UNIFORM_THESIS",
    },
    "skel_length_um_per_100um2": {
        "title": "SEM skeleton length across salinity",
        "ylabel": "Skeleton length\n(µm per 100 µm²)",
        "filename": "SEM_skeleton_length_IMAGELEVEL_UNIFORM_THESIS",
    },
    "canny_fraction": {
        "title": "SEM edge/texture fraction across salinity",
        "ylabel": "Canny edge fraction",
        "filename": "SEM_canny_fraction_IMAGELEVEL_UNIFORM_THESIS",
    },
    "eps_proxy_index": {
        "title": "SEM-derived EPS/matrix proxy across salinity",
        "ylabel": "EPS/matrix proxy\n(SEM-derived robust z-score composite)",
        "filename": "SEM_EPS_proxy_IMAGELEVEL_UNIFORM_THESIS",
    },
}

ORDER = ["0% DSW", "10% DSW", "20% DSW"]

COLORS = {
    "0% DSW": "#4C78A8",
    "10% DSW": "#F58518",
    "20% DSW": "#54A24B",
}
LIGHT_COLORS = {
    "0% DSW": "#C9D8E8",
    "10% DSW": "#F9DDBF",
    "20% DSW": "#CDE5C9",
}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 18,
    "axes.labelsize": 15,
    "xtick.labelsize": 14,
    "ytick.labelsize": 12,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

def norm_group(x):
    s = str(x)
    if s in ["0", "0.0"] or "DSW0" in s or "0%" in s:
        return "0% DSW"
    if s in ["10", "10.0"] or "DSW10" in s or "10%" in s:
        return "10% DSW"
    if s in ["20", "20.0"] or "DSW20" in s or "20%" in s:
        return "20% DSW"
    return s

def find_group_col(df):
    for c in ["group", "condition", "group_label", "DSW", "salinity"]:
        if c in df.columns:
            return c
    return None

def score_table(path):
    try:
        df = pd.read_csv(path, sep="\t")
    except Exception:
        return None

    present_features = [f for f in FEATURES if f in df.columns]
    if not present_features:
        return None

    group_col = find_group_col(df)
    if group_col is None:
        return None

    tmp = df.copy()
    tmp["condition_plot"] = tmp[group_col].map(norm_group)
    tmp = tmp[tmp["condition_plot"].isin(ORDER)]

    counts = tmp["condition_plot"].value_counts().to_dict()
    if not all(g in counts for g in ORDER):
        return None

    max_n = max(counts.values())

    # HARD REJECTION: this prevents tile-level data like n=2596/4970/5894
    if max_n > 100:
        return {
            "path": path,
            "status": "REJECT_TILE_LEVEL",
            "counts": counts,
            "df": df,
            "features": present_features,
            "group_col": group_col,
            "score": -999,
        }

    # Prefer exact expected image-level counts.
    expected = {"0% DSW": 20, "10% DSW": 21, "20% DSW": 18}
    exact_score = sum(abs(counts[g] - expected[g]) for g in ORDER)

    return {
        "path": path,
        "status": "ACCEPT_IMAGE_LEVEL",
        "counts": counts,
        "df": df,
        "features": present_features,
        "group_col": group_col,
        "score": exact_score,
    }

candidates = []

if args.input:
    hit = score_table(Path(args.input))
    if hit is None:
        raise SystemExit(f"Input file is not a usable SEM image-level feature table: {args.input}")
    candidates.append(hit)
else:
    for root in [BASE / "results_sem_eps", BASE / "sem_reanalysis_robust/v2", BASE]:
        if root.exists():
            for f in root.rglob("*.tsv"):
                hit = score_table(f)
                if hit is not None:
                    candidates.append(hit)

accepted = [c for c in candidates if c["status"] == "ACCEPT_IMAGE_LEVEL"]

candidate_report = []
for c in candidates:
    candidate_report.append({
        "path": str(c["path"]),
        "status": c["status"],
        "n_0_DSW": c["counts"].get("0% DSW", np.nan),
        "n_10_DSW": c["counts"].get("10% DSW", np.nan),
        "n_20_DSW": c["counts"].get("20% DSW", np.nan),
        "features_found": ",".join(c["features"]),
        "score": c["score"],
    })

pd.DataFrame(candidate_report).to_csv(OUT / "SEM_candidate_input_tables_checked.tsv", sep="\t", index=False)

if not accepted:
    raise SystemExit(
        "No image-level SEM feature table found. Tile-level tables were rejected. "
        f"See {OUT/'SEM_candidate_input_tables_checked.tsv'}"
    )

best = sorted(accepted, key=lambda x: x["score"])[0]
df = best["df"].copy()
group_col = best["group_col"]
df["condition_plot"] = df[group_col].map(norm_group)
df = df[df["condition_plot"].isin(ORDER)].copy()

print("[INFO] Using IMAGE-LEVEL table:", best["path"])
print("[INFO] Counts:", {g: int((df["condition_plot"] == g).sum()) for g in ORDER})

def p_to_stars(p):
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"

def fmt_p(p):
    if p == 0 or p < 1e-300:
        return "<1e-300"
    return f"{p:.2e}"

def plot_feature(feature):
    d = df[["condition_plot", feature]].copy()
    d[feature] = pd.to_numeric(d[feature], errors="coerce")
    d = d.dropna()

    data = [d.loc[d["condition_plot"] == g, feature].values.astype(float) for g in ORDER]
    ns = [len(x) for x in data]

    if max(ns) > 100:
        raise SystemExit(f"Refusing to plot {feature}: n values look tile-level: {dict(zip(ORDER, ns))}")

    H, p_kw = stats.kruskal(*data)

    raw_pairs = []
    for i, j in itertools.combinations(range(3), 2):
        U, p = stats.mannwhitneyu(data[i], data[j], alternative="two-sided")
        raw_pairs.append((i, j, U, p))

    adj = multipletests([x[3] for x in raw_pairs], method="fdr_bh")[1]

    pair_rows = []
    sig_pairs = []
    for (i, j, U, p), q in zip(raw_pairs, adj):
        stars = p_to_stars(q)
        pair_rows.append({
            "feature": feature,
            "comparison": f"{ORDER[i]} vs {ORDER[j]}",
            "n1": ns[i],
            "n2": ns[j],
            "mannwhitney_U": U,
            "p_uncorrected": p,
            "p_adj_BH_3_pairwise_tests": q,
            "stars": stars,
            "kruskal_H": H,
            "kruskal_p": p_kw,
        })
        if stars != "ns":
            sig_pairs.append((i, j, stars))

    fig, ax = plt.subplots(figsize=(7.1, 5.2))
    xpos = np.arange(3)

    parts = ax.violinplot(
        data,
        positions=xpos,
        widths=0.72,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )

    for body, g in zip(parts["bodies"], ORDER):
        body.set_facecolor(LIGHT_COLORS[g])
        body.set_edgecolor("none")
        body.set_alpha(0.82)

    bp = ax.boxplot(
        data,
        positions=xpos,
        widths=0.31,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color="black", linewidth=2.0),
        boxprops=dict(linewidth=1.2, color="black"),
        whiskerprops=dict(linewidth=1.2, color="black"),
        capprops=dict(linewidth=1.2, color="black"),
    )

    for patch, g in zip(bp["boxes"], ORDER):
        patch.set_facecolor(COLORS[g])
        patch.set_alpha(0.28)

    rng = np.random.default_rng(20260618)
    for i, g in enumerate(ORDER):
        y = data[i]
        jitter = rng.normal(0, 0.045, size=len(y))
        ax.scatter(
            np.full(len(y), i) + jitter,
            y,
            s=46,
            color=COLORS[g],
            edgecolor="black",
            linewidth=0.45,
            alpha=0.9,
            zorder=3,
        )
        med = np.median(y)
        ax.plot([i - 0.19, i + 0.19], [med, med], color="black", linewidth=2.2, zorder=4)

    ax.set_xticks(xpos)
    ax.set_xticklabels(ORDER)
    ax.set_xlabel("Condition")
    ax.set_ylabel(FEATURES[feature]["ylabel"])
    ax.set_title(FEATURES[feature]["title"], pad=16)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.18)
    ax.set_axisbelow(True)

    ymin, ymax = ax.get_ylim()
    yspan = ymax - ymin

    for i, n in enumerate(ns):
        ax.text(
            i,
            np.nanmax(data[i]) + 0.035 * yspan,
            f"n={n}",
            ha="center",
            va="bottom",
            fontsize=12,
        )

    # Expand ylim before adding p-value/brackets
    ymin, ymax = ax.get_ylim()
    yspan = ymax - ymin
    ax.set_ylim(ymin, ymax + (0.18 + 0.05 * len(sig_pairs)) * yspan)

    ymin, ymax = ax.get_ylim()
    yspan = ymax - ymin

    ax.text(
        0.985,
        0.965,
        f"Kruskal–Wallis p = {fmt_p(p_kw)}",
        ha="right",
        va="top",
        transform=ax.transAxes,
        fontsize=11,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="0.7", alpha=0.95),
    )

    base = max([np.nanmax(x) for x in data]) + 0.09 * yspan
    step = 0.075 * yspan
    h = 0.018 * yspan

    for k, (i, j, stars) in enumerate(sig_pairs):
        y = base + k * step
        ax.plot([i, i, j, j], [y, y + h, y + h, y], color="black", linewidth=1.35)
        ax.text((i + j) / 2, y + h + 0.006 * yspan, stars, ha="center", va="bottom", fontsize=14, weight="bold")

    fig.tight_layout()
    fig.savefig(OUT / f"{FEATURES[feature]['filename']}.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / f"{FEATURES[feature]['filename']}.pdf", bbox_inches="tight")
    plt.close(fig)

    group_rows = []
    for g, vals in zip(ORDER, data):
        group_rows.append({
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

    return group_rows, pair_rows

all_groups = []
all_pairs = []

for feature in FEATURES:
    if feature in df.columns:
        g, p = plot_feature(feature)
        all_groups.extend(g)
        all_pairs.extend(p)

pd.DataFrame(all_groups).to_csv(OUT / "SEM_IMAGELEVEL_UNIFORM_group_summary_plotted.tsv", sep="\t", index=False)
pd.DataFrame(all_pairs).to_csv(OUT / "SEM_IMAGELEVEL_UNIFORM_pairwise_mannwhitney_BH.tsv", sep="\t", index=False)

report = f"""SEM IMAGE-LEVEL UNIFORM THESIS PLOTS
====================================

Input table used:
{best['path']}

Rejected tile-level tables:
Any candidate table with group n > 100 was rejected.

Final image-level n values:
0% DSW = {int((df['condition_plot'] == '0% DSW').sum())}
10% DSW = {int((df['condition_plot'] == '10% DSW').sum())}
20% DSW = {int((df['condition_plot'] == '20% DSW').sum())}

Statistics:
Kruskal-Wallis across 0%, 10%, and 20% DSW.
Post hoc pairwise Mann-Whitney U tests.
Benjamini-Hochberg FDR correction across the three pairwise tests per feature.

Output directory:
{OUT}
"""
(OUT / "README_SEM_IMAGELEVEL_UNIFORM_THESIS_PLOTS.txt").write_text(report)
print(report)
