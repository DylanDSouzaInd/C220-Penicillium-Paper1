#!/usr/bin/env python3
"""
Correct crude EPS-like carbohydrate normalization to dry biomass.

This script uses ALL sample dilution rows (neat, 1:5, 1:10) correctly:
  1. Uses the dilution-corrected total carbohydrate already calculated in the workbook.
  2. Divides EACH dilution-corrected reading by matched dry biomass for that biological tube.
  3. Aggregates the three dilution-derived estimates per biological replicate.
     Default = median, because dilution response was not perfectly linear.
  4. Converts biological-replicate values to a relative ratio against the 0% DSW mean.
  5. Plots the relative crude EPS-like carbohydrate per g dry biomass.

IMPORTANT:
- Do not treat the 9 dilution readings per salinity as 9 independent biological replicates.
  They are 3 biological replicates × 3 dilution-derived technical estimates.
- Statistics are performed on the 3 biological-replicate aggregate values per salinity.

Run:
  python plot_crude_eps_per_g_biomass_corrected.py \
    --xlsx CrudeEPS_repeat_analysis_S0_to_S9_no_S10.xlsx \
    --biomass biomass_template_fill_this.csv \
    --out crude_eps_per_g_biomass_relative.png

The biomass CSV must contain:
  Sample,dry_biomass_g
  0_R1,0.1234
  ...
"""

import argparse
import itertools
import os
import re
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import kruskal, mannwhitneyu


FRACTION_ORDER = {"neat": 0, "1:5": 1, "1:10": 2}
COND_ORDER = ["0% DSW", "10% DSW", "20% DSW"]
COND_COLORS = {"0% DSW": "#79a9d1", "10% DSW": "#f2a64b", "20% DSW": "#70b66b"}
DIL_MARKERS = {"neat": "o", "1:5": "s", "1:10": "^"}


def clean_condition(x):
    s = str(x)
    m = re.search(r"(0|10|20)\s*%", s)
    return f"{m.group(1)}% DSW" if m else s


def p_fmt(p):
    if p < 1e-4:
        return f"{p:.2e}"
    return f"{p:.4f}"


def bh_fdr(pvals):
    pvals = np.asarray(pvals, dtype=float)
    n = len(pvals)
    order = np.argsort(pvals)
    ranked = pvals[order]
    out = np.empty(n, dtype=float)
    prev = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        val = ranked[i] * n / rank
        prev = min(prev, val)
        out[order[i]] = min(prev, 1.0)
    return out


def anova_F(vals, labels):
    vals = np.asarray(vals, dtype=float)
    labels = np.asarray(labels)
    groups = [vals[labels == lab] for lab in sorted(np.unique(labels))]
    grand = vals.mean()
    ss_between = sum(len(g) * (g.mean() - grand) ** 2 for g in groups)
    ss_within = sum(((g - g.mean()) ** 2).sum() for g in groups)
    return (ss_between / (len(groups) - 1)) / (ss_within / (len(vals) - len(groups)))


def exact_permutation_global(vals, labels):
    """Exact permutation ANOVA-like p for 3 groups of n=3 each."""
    vals = np.asarray(vals, dtype=float)
    labels = np.asarray(labels)
    if len(vals) != 9 or any(np.sum(labels == g) != 3 for g in np.unique(labels)):
        return np.nan, np.nan
    obs = anova_F(vals, labels)
    idx = list(range(len(vals)))
    count = 0
    total = 0
    for g0 in itertools.combinations(idx, 3):
        rem = [i for i in idx if i not in g0]
        for g1 in itertools.combinations(rem, 3):
            g2 = [i for i in rem if i not in g1]
            lab = np.empty(len(vals), dtype=int)
            lab[list(g0)] = 0
            lab[list(g1)] = 1
            lab[list(g2)] = 2
            stat = anova_F(vals, lab)
            total += 1
            if stat >= obs - 1e-12:
                count += 1
    return obs, count / total


def exact_contrast_20_vs_lower(vals, conds):
    """Exact permutation p for a priori contrast: 20% vs pooled 0%+10%."""
    vals = np.asarray(vals, dtype=float)
    conds = np.asarray(conds)
    mask_obs = conds == "20% DSW"
    if len(vals) != 9 or mask_obs.sum() != 3:
        return np.nan, np.nan
    obs = vals[mask_obs].mean() - vals[~mask_obs].mean()
    idx = list(range(len(vals)))
    count = 0
    total = 0
    for comb in itertools.combinations(idx, 3):
        mask = np.zeros(len(vals), dtype=bool)
        mask[list(comb)] = True
        stat = vals[mask].mean() - vals[~mask].mean()
        total += 1
        if abs(stat) >= abs(obs) - 1e-12:
            count += 1
    return obs, count / total


def add_bracket(ax, x1, x2, y, h, text):
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], color="black", lw=1.3, clip_on=False)
    ax.text((x1 + x2) / 2, y + h, text, ha="center", va="bottom", fontsize=9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xlsx", required=True, help="Workbook with Samples_Calcs sheet.")
    ap.add_argument("--biomass", required=True, help="CSV with columns Sample,dry_biomass_g.")
    ap.add_argument("--out", default="crude_eps_per_g_biomass_relative.png")
    ap.add_argument("--aggregate", choices=["median", "mean"], default="median",
                    help="How to combine neat, 1:5 and 1:10 dilution-corrected estimates per biological replicate.")
    ap.add_argument("--final-volume-ml", type=float, default=0.7,
                    help="Final dissolved crude EPS volume in mL. Workbook already uses 0.7 mL; kept for documentation.")
    args = ap.parse_args()

    raw = pd.read_excel(args.xlsx, sheet_name="Samples_Calcs")
    raw = raw[raw["Fraction/dilution"].isin(FRACTION_ORDER)].copy()
    raw["Condition"] = raw["Condition"].map(clean_condition)
    raw = raw[raw["Condition"].isin(COND_ORDER)].copy()
    raw["Fraction_order"] = raw["Fraction/dilution"].map(FRACTION_ORDER)

    total_col_candidates = [c for c in raw.columns if "Total carbohydrate" in str(c)]
    if not total_col_candidates:
        sys.exit("ERROR: Could not find total carbohydrate column.")
    total_col = total_col_candidates[0]
    raw["total_ug_glucose_eq"] = pd.to_numeric(raw[total_col], errors="coerce")

    biom = pd.read_csv(args.biomass)
    required = {"Sample", "dry_biomass_g"}
    if not required.issubset(biom.columns):
        sys.exit("ERROR: biomass CSV must contain columns Sample,dry_biomass_g")
    biom["dry_biomass_g"] = pd.to_numeric(biom["dry_biomass_g"], errors="coerce")
    missing = biom[biom["dry_biomass_g"].isna()]
    if not missing.empty:
        print("ERROR: dry_biomass_g is missing for these samples:")
        print(missing[["Sample", "dry_biomass_g"]].to_string(index=False))
        sys.exit(2)

    df = raw.merge(biom[["Sample", "dry_biomass_g"]], on="Sample", how="left")
    if df["dry_biomass_g"].isna().any():
        print("ERROR: no dry biomass supplied for:")
        print(df.loc[df["dry_biomass_g"].isna(), "Sample"].drop_duplicates().to_string(index=False))
        sys.exit(2)

    df["ug_GE_per_g_dry_biomass"] = df["total_ug_glucose_eq"] / df["dry_biomass_g"]
    df = df.sort_values(["Condition", "Sample", "Fraction_order"])

    agg_fun = np.median if args.aggregate == "median" else np.mean
    rep = (
        df.groupby(["Condition", "Sample", "Replicate", "dry_biomass_g"], observed=True)
        .agg(
            bio_rep_ug_GE_per_g=("ug_GE_per_g_dry_biomass", agg_fun),
            mean_ug_GE_per_g=("ug_GE_per_g_dry_biomass", "mean"),
            median_ug_GE_per_g=("ug_GE_per_g_dry_biomass", "median"),
            n_dilution_readings=("ug_GE_per_g_dry_biomass", "size"),
            sd_across_dilutions=("ug_GE_per_g_dry_biomass", "std"),
        )
        .reset_index()
    )

    zero_mean = rep.loc[rep["Condition"] == "0% DSW", "bio_rep_ug_GE_per_g"].mean()
    df["relative_ug_GE_per_g_vs_0pct_mean"] = df["ug_GE_per_g_dry_biomass"] / zero_mean
    rep["relative_bio_rep_vs_0pct_mean"] = rep["bio_rep_ug_GE_per_g"] / zero_mean

    # stats on biological replicate aggregate values only
    vals = rep["relative_bio_rep_vs_0pct_mean"].to_numpy(dtype=float)
    conds = rep["Condition"].astype(str).to_numpy()
    label_map = {c: i for i, c in enumerate(COND_ORDER)}
    labels = np.array([label_map[c] for c in conds])
    groups = [rep.loc[rep["Condition"] == c, "relative_bio_rep_vs_0pct_mean"].to_numpy() for c in COND_ORDER]
    H, kw_p = kruskal(*groups)
    Fobs, perm_p = exact_permutation_global(vals, labels)
    contrast_stat, contrast_p = exact_contrast_20_vs_lower(vals, conds)

    pair_rows = []
    pairs = [("0% DSW", "10% DSW"), ("0% DSW", "20% DSW"), ("10% DSW", "20% DSW")]
    pvals = []
    Us = []
    for a, b in pairs:
        xa = rep.loc[rep["Condition"] == a, "relative_bio_rep_vs_0pct_mean"].to_numpy()
        xb = rep.loc[rep["Condition"] == b, "relative_bio_rep_vs_0pct_mean"].to_numpy()
        U, p = mannwhitneyu(xa, xb, alternative="two-sided", method="exact")
        Us.append(U); pvals.append(p)
    fdr = bh_fdr(pvals)
    for (a, b), U, p, pf in zip(pairs, Us, pvals, fdr):
        pair_rows.append({"comparison": f"{a} vs {b}", "Mann_Whitney_U": U, "p_raw": p, "p_BH_FDR": pf})

    summary = (
        rep.groupby("Condition", observed=True)
        .agg(
            n_biological_reps=("Sample", "size"),
            mean_relative=("relative_bio_rep_vs_0pct_mean", "mean"),
            sd_relative=("relative_bio_rep_vs_0pct_mean", "std"),
            median_relative=("relative_bio_rep_vs_0pct_mean", "median"),
            mean_ug_GE_per_g=("bio_rep_ug_GE_per_g", "mean"),
            sd_ug_GE_per_g=("bio_rep_ug_GE_per_g", "std"),
        )
        .reindex(COND_ORDER)
        .reset_index()
    )
    stats = pd.DataFrame([
        {"test": "Kruskal-Wallis", "statistic": H, "p": kw_p, "note": "on biological replicate aggregates only"},
        {"test": "Exact permutation ANOVA-like", "statistic": Fobs, "p": perm_p, "note": "on biological replicate aggregates only"},
        {"test": "A priori exact permutation contrast", "statistic": contrast_stat, "p": contrast_p, "note": "20% DSW vs pooled 0%+10%; on biological replicate aggregates only"},
    ])

    base, ext = os.path.splitext(args.out)
    if not ext:
        args.out = args.out + ".png"
        base = args.out[:-4]
    out_pdf = base + ".pdf"

    df.to_csv(base + "_all_dilution_corrected_readings.tsv", sep="\t", index=False)
    rep.to_csv(base + "_biological_replicate_aggregates.tsv", sep="\t", index=False)
    summary.to_csv(base + "_condition_summary.tsv", sep="\t", index=False)
    stats.to_csv(base + "_stats.tsv", sep="\t", index=False)
    pd.DataFrame(pair_rows).to_csv(base + "_pairwise_mannwhitney.tsv", sep="\t", index=False)

    # Plot
    fig, ax = plt.subplots(figsize=(7.2, 4.9), dpi=300)
    rng = np.random.default_rng(21)
    x_map = {c: i + 1 for i, c in enumerate(COND_ORDER)}

    # show all dilution-corrected readings as small pale points; shapes indicate dilution
    for cond in COND_ORDER:
        sub = df[df["Condition"] == cond]
        x0 = x_map[cond]
        for frac in ["neat", "1:5", "1:10"]:
            sf = sub[sub["Fraction/dilution"] == frac]
            xs = np.full(len(sf), x0) + rng.uniform(-0.16, 0.16, len(sf))
            ax.scatter(xs, sf["relative_ug_GE_per_g_vs_0pct_mean"],
                       marker=DIL_MARKERS[frac], s=33, alpha=0.28,
                       facecolor=COND_COLORS[cond], edgecolor="#333333", linewidth=0.5,
                       label=frac if cond == COND_ORDER[0] else None, zorder=2)

    # biological replicate aggregate values = stats unit
    rep_values_by_cond = []
    for cond in COND_ORDER:
        vals_c = rep.loc[rep["Condition"] == cond, "relative_bio_rep_vs_0pct_mean"].to_numpy()
        rep_values_by_cond.append(vals_c)

    bp = ax.boxplot(rep_values_by_cond, positions=[1, 2, 3], widths=0.38, patch_artist=True,
                    showfliers=False,
                    medianprops={"color": "black", "linewidth": 1.8},
                    boxprops={"color": "#444444", "linewidth": 1.2},
                    whiskerprops={"color": "#444444", "linewidth": 1.2},
                    capprops={"color": "#444444", "linewidth": 1.2})
    for patch, cond in zip(bp["boxes"], COND_ORDER):
        patch.set_facecolor(COND_COLORS[cond])
        patch.set_alpha(0.40)

    for cond in COND_ORDER:
        sub = rep[rep["Condition"] == cond]
        x0 = x_map[cond]
        xs = np.full(len(sub), x0) + rng.uniform(-0.05, 0.05, len(sub))
        ax.scatter(xs, sub["relative_bio_rep_vs_0pct_mean"], s=85,
                   facecolor=COND_COLORS[cond], edgecolor="black", linewidth=0.9, zorder=4)
        ax.text(x0, sub["relative_bio_rep_vs_0pct_mean"].max() * 1.05, "n=3 bio reps",
                ha="center", va="bottom", fontsize=9)

    ymax = max(df["relative_ug_GE_per_g_vs_0pct_mean"].max(), rep["relative_bio_rep_vs_0pct_mean"].max())
    ymin = min(0, df["relative_ug_GE_per_g_vs_0pct_mean"].min())
    yr = ymax - ymin if ymax > ymin else 1
    ax.set_ylim(ymin - 0.05 * yr, ymax + 0.25 * yr)

    add_bracket(ax, 1, 3, ymax + 0.06 * yr, 0.04 * yr,
                f"A priori contrast: 20% vs 0%+10%, permutation p={p_fmt(contrast_p)}")
    ax.text(0.985, 0.985,
            f"Exact permutation ANOVA-like p={p_fmt(perm_p)}\nKruskal-Wallis p={p_fmt(kw_p)}",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="#bfbfbf", alpha=0.92))

    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(COND_ORDER, fontsize=12)
    ax.set_xlabel("Condition", fontsize=12)
    ax.set_ylabel("Relative crude EPS-like carbohydrate\n(µg glucose equivalents / g dry biomass; 0% mean = 1)", fontsize=11)
    ax.grid(axis="y", alpha=0.20)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    leg = ax.legend(title="Dilution-corrected readings", frameon=False, fontsize=8.5, title_fontsize=8.5, loc="upper left")

    fig.tight_layout()
    fig.savefig(args.out, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")

    print("Wrote:")
    for p in [args.out, out_pdf, base + "_all_dilution_corrected_readings.tsv", base + "_biological_replicate_aggregates.tsv", base + "_condition_summary.tsv", base + "_stats.tsv"]:
        print("  " + p)
    print("\nCondition summary:")
    print(summary.to_string(index=False))
    print("\nStats:")
    print(stats.to_string(index=False))


if __name__ == "__main__":
    main()
