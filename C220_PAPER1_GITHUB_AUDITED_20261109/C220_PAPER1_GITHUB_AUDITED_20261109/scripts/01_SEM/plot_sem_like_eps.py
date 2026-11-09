#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kruskal, mannwhitneyu

# ----------------------------
# Input files
# ----------------------------
FEATURES_FILE = "sem_reanalysis_robust/v2/sem_eps_features_clean_with_outlier_flags.tsv"

# ----------------------------
# Plot settings
# ----------------------------
ORDER = ["DSW0", "DSW10", "DSW20"]
DISPLAY = {
    "DSW0": "0% DSW",
    "DSW10": "10% DSW",
    "DSW20": "20% DSW",
}
COLORS = {
    "DSW0": "#4C78A8",   # blue
    "DSW10": "#F2A65A",  # orange
    "DSW20": "#59A14F",  # green
}

STAR_THRESHOLDS = [
    (0.001, "***"),
    (0.01,  "**"),
    (0.05,  "*"),
]

def p_to_stars(p):
    for thr, stars in STAR_THRESHOLDS:
        if p < thr:
            return stars
    return "ns"

def bh_adjust(pvals):
    pvals = np.asarray(pvals, dtype=float)
    n = len(pvals)
    order = np.argsort(pvals)
    ranked = pvals[order]
    adj = np.empty(n, dtype=float)

    prev = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        val = ranked[i] * n / rank
        prev = min(prev, val)
        adj[i] = min(prev, 1.0)

    out = np.empty(n, dtype=float)
    out[order] = adj
    return out

def add_sig_bracket(ax, x1, x2, y, h, text, lw=1.8, fs=14):
    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y],
            color="black", lw=lw, clip_on=False)
    ax.text((x1 + x2) / 2, y + h + 0.01*(ax.get_ylim()[1] - ax.get_ylim()[0]),
            text, ha="center", va="bottom", fontsize=fs, fontweight="bold")

def make_plot(df, feature, ylabel, title, outfile,
              remove_feature_outliers=False,
              show_all_three_brackets=True,
              ypad_frac=0.24):
    work = df.copy()

    # Optionally remove feature-specific global robust outliers
    if remove_feature_outliers:
        outlier_col = f"{feature}_outlier_absrz_gt3p5"
        if outlier_col in work.columns:
            work = work.loc[~work[outlier_col].fillna(False)].copy()

    # Keep only needed groups
    work = work.loc[work["group_label"].isin(ORDER)].copy()

    # Data by group
    data = [work.loc[work["group_label"] == g, feature].dropna().values for g in ORDER]
    ns = [len(x) for x in data]
    medians = [np.median(x) for x in data]

    # Global stats
    H, p_kw = kruskal(*data)

    # Pairwise MW + BH within this feature
    pairs = [(0, 1), (1, 2), (0, 2)]
    raw_ps = []
    pair_stats = []
    for i, j in pairs:
        x = data[i]
        y = data[j]
        U, p = mannwhitneyu(x, y, alternative="two-sided")
        raw_ps.append(p)
        pair_stats.append((i, j, U, p))
    adj_ps = bh_adjust(raw_ps)

    # Figure
    fig, ax = plt.subplots(figsize=(10.0, 7.3), dpi=200)
    positions = np.array([1, 2, 3], dtype=float)

    # Violin
    v = ax.violinplot(
        data,
        positions=positions,
        widths=0.72,
        showmeans=False,
        showmedians=False,
        showextrema=False
    )
    for body, g in zip(v["bodies"], ORDER):
        body.set_facecolor(COLORS[g])
        body.set_edgecolor("none")
        body.set_alpha(0.28)

    # Boxplot
    bp = ax.boxplot(
        data,
        positions=positions,
        widths=0.18,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color="black", linewidth=2.2),
        whiskerprops=dict(color="black", linewidth=1.4),
        capprops=dict(color="black", linewidth=1.4),
        boxprops=dict(linewidth=1.4, color="black")
    )
    for patch, g in zip(bp["boxes"], ORDER):
        patch.set_facecolor(COLORS[g])
        patch.set_alpha(0.35)
        patch.set_edgecolor("black")

    # Points
    rng = np.random.default_rng(42)
    for pos, g, vals in zip(positions, ORDER, data):
        jitter = rng.uniform(-0.08, 0.08, size=len(vals))
        ax.scatter(
            np.full(len(vals), pos) + jitter,
            vals,
            s=74,
            color=COLORS[g],
            edgecolor="black",
            linewidth=0.45,
            alpha=0.9,
            zorder=3
        )

    # Thick group median bar
    for pos, med in zip(positions, medians):
        ax.hlines(med, pos - 0.19, pos + 0.19,
                  colors="black", linewidth=2.2, zorder=4)

    # Axes and labels
    ax.set_xticks(positions)
    ax.set_xticklabels([DISPLAY[g] for g in ORDER], fontsize=17)
    ax.set_ylabel(ylabel, fontsize=19)
    ax.set_xlabel("Condition", fontsize=19)
    ax.set_title(title, fontsize=23, pad=16)

    # Y limits
    all_vals = np.concatenate(data)
    ymin = np.min(all_vals)
    ymax = np.max(all_vals)
    yr = ymax - ymin
    if yr == 0:
        yr = 1.0

    lower = ymin - 0.08 * yr
    upper = ymax + ypad_frac * yr
    ax.set_ylim(lower, upper)

    # Compute upper whiskers from boxplot artists and place n above them
    upper_whiskers = []
    whiskers = bp["whiskers"]
    # whiskers come in pairs per box: lower, upper
    for i in range(1, len(whiskers), 2):
        upper_y = max(whiskers[i].get_ydata())
        upper_whiskers.append(upper_y)

    for pos, n, uw in zip(positions, ns, upper_whiskers):
        ax.text(pos, uw + 0.018 * yr, f"n={n}",
                ha="center", va="bottom", fontsize=14)

    # Smaller Kruskal-Wallis box
    ax.text(
        0.985, 0.975,
        f"Kruskal–Wallis p = {p_kw:.2e}",
        transform=ax.transAxes,
        ha="right", va="top",
        fontsize=13,
        bbox=dict(boxstyle="round,pad=0.22", facecolor="white",
                  edgecolor="0.7", alpha=0.95)
    )

    # Significance brackets
    stars01 = p_to_stars(adj_ps[0])  # 0 vs 10
    stars12 = p_to_stars(adj_ps[1])  # 10 vs 20
    stars02 = p_to_stars(adj_ps[2])  # 0 vs 20

    def add_sig_bracket(ax, x1, x2, y, h, text, fs=14):
        ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y],
                color="black", lw=1.6, clip_on=False)
        ax.text((x1 + x2) / 2, y + h + 0.004 * (ax.get_ylim()[1] - ax.get_ylim()[0]),
                text, ha="center", va="bottom",
                fontsize=fs, fontweight="bold")

    # Compact bracket stack
    h = 0.018 * yr
    y01 = ymax + 0.040 * yr
    y12 = ymax + 0.085 * yr
    y02 = ymax + 0.145 * yr

    add_sig_bracket(ax, 1, 2, y01, h, stars01, fs=14)
    add_sig_bracket(ax, 2, 3, y12, h, stars12, fs=14)
    if show_all_three_brackets:
        add_sig_bracket(ax, 1, 3, y02, h, stars02, fs=14)

    # Style
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis='y', labelsize=15)
    ax.tick_params(axis='x', labelsize=17)

    # Slightly tighter layout
    fig.subplots_adjust(top=0.87, left=0.13, right=0.98, bottom=0.13)

    fig.savefig(outfile, dpi=300, bbox_inches="tight")
    print(f"Saved: {outfile}")

    # Print stats
    print("\n==================================================")
    print(title)
    print("Feature:", feature)
    print("Kruskal–Wallis H =", H, "p =", p_kw)
    for (i, j, U, p_raw), p_adj in zip(pair_stats, adj_ps):
        print(f"{DISPLAY[ORDER[i]]} vs {DISPLAY[ORDER[j]]}: "
              f"U={U:.3f}, raw p={p_raw:.6g}, BH p={p_adj:.6g}")
    print("Medians:",
          {DISPLAY[g]: float(m) for g, m in zip(ORDER, medians)})

def main():
    df = pd.read_csv(FEATURES_FILE, sep="\t")

    # 1) Branchpoints: keep all images (no global outliers flagged for this feature)
    make_plot(
        df=df,
        feature="branchpoints_per_100um2",
        ylabel="Branchpoint density\n(branchpoints per 100 µm²)",
        title="SEM branchpoint density across salinity",
        outfile="SEM_branchpoints_THESIS_READY.png",
        remove_feature_outliers=False,
        show_all_three_brackets=True,
        ypad_frac=0.26
    )

    # 2) Skeleton length: remove feature-specific global robust outlier
    make_plot(
        df=df,
        feature="skel_length_um_per_100um2",
        ylabel="Skeleton length\n(µm per 100 µm²)",
        title="SEM skeleton length across salinity",
        outfile="SEM_skeleton_length_THESIS_READY.png",
        remove_feature_outliers=True,
        show_all_three_brackets=True,
        ypad_frac=0.28
    )

if __name__ == "__main__":
    main()
