import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

ROOT = Path("/data/dylan/sem_image_analysis")
IN = ROOT / "results_sem_eps"
OUT = ROOT / "sem_reanalysis_robust/v2"
OUT.mkdir(parents=True, exist_ok=True)

features_path = IN / "image_features.tsv"
meta_path = IN / "image_metadata.tsv"

df = pd.read_csv(features_path, sep="\t")
meta = pd.read_csv(meta_path, sep="\t")

# Ensure group labels are clean.
df["group"] = df["group"].astype(str)
df["group_label"] = df["group"].map({"0": "DSW0", "10": "DSW10", "20": "DSW20"}).fillna(df["group"])
df["salinity_numeric"] = pd.to_numeric(df["salinity_numeric"], errors="coerce")

# Useful feature set.
feature_cols = [
    "eps_proxy_index",
    "branchpoints_per_100um2",
    "skel_length_um_per_100um2",
    "canny_fraction",
    "entropy_mean",
    "highpass_std",
    "material_fraction",
    "pore_fraction",
    "edge_mean",
]

feature_cols = [c for c in feature_cols if c in df.columns]

# Add magnification/FOV bins.
df["magnification"] = pd.to_numeric(df["magnification"], errors="coerce")
df["fov_um_x"] = pd.to_numeric(df["fov_um_x"], errors="coerce")
df["fov_um_y"] = pd.to_numeric(df["fov_um_y"], errors="coerce")
df["analyzed_area_um2"] = pd.to_numeric(df["analyzed_area_um2"], errors="coerce")

def mag_bin(x):
    if pd.isna(x):
        return "unknown"
    if x <= 1300:
        return "low_750_1300"
    if x <= 1800:
        return "mid_1400_1800"
    if x <= 2700:
        return "high_2200_2700"
    return "very_high_3000plus"

df["mag_bin"] = df["magnification"].apply(mag_bin)

# Robust outlier flag per feature using MAD within all images.
def robust_z(x):
    x = pd.to_numeric(x, errors="coerce")
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    if mad == 0 or np.isnan(mad):
        return pd.Series(np.zeros(len(x)), index=x.index)
    return 0.6745 * (x - med) / mad

for c in feature_cols:
    df[c + "_robust_z_global"] = robust_z(df[c])
    df[c + "_outlier_absrz_gt3p5"] = df[c + "_robust_z_global"].abs() > 3.5

# Save audit summaries.
df.to_csv(OUT / "sem_eps_features_clean_with_outlier_flags.tsv", sep="\t", index=False)

qc_group = df.groupby("group_label").agg(
    n_images=("image_id", "count"),
    n_replicate_labels=("replicate", "nunique"),
    median_magnification=("magnification", "median"),
    min_magnification=("magnification", "min"),
    max_magnification=("magnification", "max"),
    median_fov_um_x=("fov_um_x", "median"),
    min_fov_um_x=("fov_um_x", "min"),
    max_fov_um_x=("fov_um_x", "max"),
    median_area_um2=("analyzed_area_um2", "median"),
).reset_index()

qc_group.to_csv(OUT / "QC_group_imaging_summary.tsv", sep="\t", index=False)

mag_counts = pd.crosstab(df["group_label"], df["magnification"])
mag_counts.to_csv(OUT / "QC_magnification_counts_by_group.tsv", sep="\t")

magbin_counts = pd.crosstab(df["group_label"], df["mag_bin"])
magbin_counts.to_csv(OUT / "QC_magnification_bin_counts_by_group.tsv", sep="\t")

# Check replicate labels.
rep_counts = pd.crosstab(df["group_label"], df["replicate"])
rep_counts.to_csv(OUT / "QC_replicate_label_counts_by_group.tsv", sep="\t")

# Basic image-level group summaries.
summary_rows = []
for c in feature_cols:
    for g, sub in df.groupby("group_label"):
        vals = pd.to_numeric(sub[c], errors="coerce").dropna()
        summary_rows.append({
            "feature": c,
            "group": g,
            "n_images": len(vals),
            "mean": vals.mean(),
            "median": vals.median(),
            "sd": vals.std(ddof=1),
            "iqr": vals.quantile(0.75) - vals.quantile(0.25),
            "min": vals.min(),
            "max": vals.max(),
            "n_outliers_absrz_gt3p5": int(sub[c + "_outlier_absrz_gt3p5"].sum()) if c + "_outlier_absrz_gt3p5" in sub else 0
        })

summary = pd.DataFrame(summary_rows)
summary.to_csv(OUT / "image_level_group_summary.tsv", sep="\t", index=False)

# Tests: image-level Kruskal and Spearman trend.
test_rows = []
pair_rows = []

for c in feature_cols:
    groups = []
    for g in ["DSW0", "DSW10", "DSW20"]:
        vals = pd.to_numeric(df.loc[df["group_label"] == g, c], errors="coerce").dropna()
        groups.append(vals)

    if all(len(x) >= 2 for x in groups):
        H, p = stats.kruskal(*groups)
    else:
        H, p = np.nan, np.nan

    valid = df[["salinity_numeric", c]].dropna()
    if len(valid) >= 5:
        rho, sp = stats.spearmanr(valid["salinity_numeric"], valid[c])
    else:
        rho, sp = np.nan, np.nan

    test_rows.append({
        "analysis": "image_level_all_images",
        "feature": c,
        "kruskal_H": H,
        "kruskal_p": p,
        "spearman_rho_salinity": rho,
        "spearman_p": sp,
        "n_total_images": int(df[c].notna().sum())
    })

    for a, b in [("DSW0", "DSW10"), ("DSW0", "DSW20"), ("DSW10", "DSW20")]:
        va = pd.to_numeric(df.loc[df["group_label"] == a, c], errors="coerce").dropna()
        vb = pd.to_numeric(df.loc[df["group_label"] == b, c], errors="coerce").dropna()
        if len(va) >= 2 and len(vb) >= 2:
            U, pp = stats.mannwhitneyu(va, vb, alternative="two-sided")
            delta = vb.median() - va.median()
        else:
            U, pp, delta = np.nan, np.nan, np.nan
        pair_rows.append({
            "analysis": "image_level_all_images",
            "feature": c,
            "comparison": f"{b}_minus_{a}",
            "group_a": a,
            "group_b": b,
            "n_a": len(va),
            "n_b": len(vb),
            "median_a": va.median() if len(va) else np.nan,
            "median_b": vb.median() if len(vb) else np.nan,
            "median_difference_b_minus_a": delta,
            "mannwhitney_U": U,
            "p_uncorrected": pp
        })

tests = pd.DataFrame(test_rows)
pairs = pd.DataFrame(pair_rows)

# BH correction helper.
def bh_adjust(pvals):
    pvals = np.asarray(pvals, dtype=float)
    out = np.full_like(pvals, np.nan)
    mask = ~np.isnan(pvals)
    p = pvals[mask]
    n = len(p)
    if n == 0:
        return out
    order = np.argsort(p)
    ranked = p[order]
    adj = ranked * n / (np.arange(n) + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.minimum(adj, 1.0)
    tmp = np.empty(n)
    tmp[order] = adj
    out[mask] = tmp
    return out

tests["kruskal_padj_BH"] = bh_adjust(tests["kruskal_p"].values)
tests["spearman_padj_BH"] = bh_adjust(tests["spearman_p"].values)
pairs["p_adj_BH_within_all_pairs"] = bh_adjust(pairs["p_uncorrected"].values)

tests.to_csv(OUT / "image_level_tests_all_images.tsv", sep="\t", index=False)
pairs.to_csv(OUT / "image_level_pairwise_mannwhitney_all_images.tsv", sep="\t", index=False)

# Outlier sensitivity: remove feature-specific robust outliers and rerun core tests.
sens_rows = []
for c in feature_cols:
    subdf = df[~df[c + "_outlier_absrz_gt3p5"]].copy()

    groups = []
    for g in ["DSW0", "DSW10", "DSW20"]:
        vals = pd.to_numeric(subdf.loc[subdf["group_label"] == g, c], errors="coerce").dropna()
        groups.append(vals)

    if all(len(x) >= 2 for x in groups):
        H, p = stats.kruskal(*groups)
    else:
        H, p = np.nan, np.nan

    valid = subdf[["salinity_numeric", c]].dropna()
    if len(valid) >= 5:
        rho, sp = stats.spearmanr(valid["salinity_numeric"], valid[c])
    else:
        rho, sp = np.nan, np.nan

    medians = {}
    ns = {}
    for g in ["DSW0", "DSW10", "DSW20"]:
        vals = pd.to_numeric(subdf.loc[subdf["group_label"] == g, c], errors="coerce").dropna()
        medians[g] = vals.median() if len(vals) else np.nan
        ns[g] = len(vals)

    sens_rows.append({
        "analysis": "image_level_remove_global_robust_outliers_absrz_gt3p5",
        "feature": c,
        "n_DSW0": ns["DSW0"],
        "n_DSW10": ns["DSW10"],
        "n_DSW20": ns["DSW20"],
        "median_DSW0": medians["DSW0"],
        "median_DSW10": medians["DSW10"],
        "median_DSW20": medians["DSW20"],
        "kruskal_H": H,
        "kruskal_p": p,
        "spearman_rho_salinity": rho,
        "spearman_p": sp,
        "trend_direction_0_to_20": (
            "increase" if medians["DSW20"] > medians["DSW0"] else
            "decrease" if medians["DSW20"] < medians["DSW0"] else
            "flat"
        )
    })

sens = pd.DataFrame(sens_rows)
sens["kruskal_padj_BH"] = bh_adjust(sens["kruskal_p"].values)
sens["spearman_padj_BH"] = bh_adjust(sens["spearman_p"].values)
sens.to_csv(OUT / "outlier_sensitivity_tests.tsv", sep="\t", index=False)

# Magnification-bin sensitivity: only bins represented in all groups.
common_bins = []
for b, sub in df.groupby("mag_bin"):
    counts = sub["group_label"].value_counts()
    if all(counts.get(g, 0) >= 2 for g in ["DSW0", "DSW10", "DSW20"]):
        common_bins.append(b)

bin_rows = []
for b in common_bins:
    subdf = df[df["mag_bin"] == b].copy()
    for c in feature_cols:
        groups = []
        for g in ["DSW0", "DSW10", "DSW20"]:
            vals = pd.to_numeric(subdf.loc[subdf["group_label"] == g, c], errors="coerce").dropna()
            groups.append(vals)
        if all(len(x) >= 2 for x in groups):
            H, p = stats.kruskal(*groups)
        else:
            H, p = np.nan, np.nan
        valid = subdf[["salinity_numeric", c]].dropna()
        if len(valid) >= 5:
            rho, sp = stats.spearmanr(valid["salinity_numeric"], valid[c])
        else:
            rho, sp = np.nan, np.nan

        bin_rows.append({
            "analysis": "within_common_magnification_bin",
            "mag_bin": b,
            "feature": c,
            "n_DSW0": int((subdf["group_label"] == "DSW0").sum()),
            "n_DSW10": int((subdf["group_label"] == "DSW10").sum()),
            "n_DSW20": int((subdf["group_label"] == "DSW20").sum()),
            "kruskal_H": H,
            "kruskal_p": p,
            "spearman_rho_salinity": rho,
            "spearman_p": sp,
            "median_DSW0": subdf.loc[subdf["group_label"] == "DSW0", c].median(),
            "median_DSW10": subdf.loc[subdf["group_label"] == "DSW10", c].median(),
            "median_DSW20": subdf.loc[subdf["group_label"] == "DSW20", c].median(),
        })

bin_tests = pd.DataFrame(bin_rows)
if not bin_tests.empty:
    bin_tests["kruskal_padj_BH"] = bh_adjust(bin_tests["kruskal_p"].values)
    bin_tests["spearman_padj_BH"] = bh_adjust(bin_tests["spearman_p"].values)
bin_tests.to_csv(OUT / "magnification_bin_sensitivity_tests.tsv", sep="\t", index=False)

# Replicate label aggregation. This is only valid if replicate truly means biological replicate.
rep_features = []
if "replicate" in df.columns:
    group_cols = ["group_label", "salinity_numeric", "replicate"]
    agg_cols = {c: "median" for c in feature_cols}
    agg_cols.update({
        "image_id": "count",
        "magnification": "median",
        "fov_um_x": "median",
        "analyzed_area_um2": "median",
    })
    rep = df.groupby(group_cols).agg(agg_cols).reset_index()
    rep = rep.rename(columns={"image_id": "n_images_in_replicate_label"})
    rep.to_csv(OUT / "replicate_label_median_features.tsv", sep="\t", index=False)

    rep_test_rows = []
    for c in feature_cols:
        groups = []
        for g in ["DSW0", "DSW10", "DSW20"]:
            vals = pd.to_numeric(rep.loc[rep["group_label"] == g, c], errors="coerce").dropna()
            groups.append(vals)
        if all(len(x) >= 2 for x in groups):
            H, p = stats.kruskal(*groups)
        else:
            H, p = np.nan, np.nan
        valid = rep[["salinity_numeric", c]].dropna()
        if len(valid) >= 5:
            rho, sp = stats.spearmanr(valid["salinity_numeric"], valid[c])
        else:
            rho, sp = np.nan, np.nan

        rep_test_rows.append({
            "analysis": "replicate_label_median",
            "feature": c,
            "n_replicate_labels_DSW0": len(groups[0]),
            "n_replicate_labels_DSW10": len(groups[1]),
            "n_replicate_labels_DSW20": len(groups[2]),
            "median_DSW0": groups[0].median() if len(groups[0]) else np.nan,
            "median_DSW10": groups[1].median() if len(groups[1]) else np.nan,
            "median_DSW20": groups[2].median() if len(groups[2]) else np.nan,
            "kruskal_H": H,
            "kruskal_p": p,
            "spearman_rho_salinity": rho,
            "spearman_p": sp,
        })

    rep_tests = pd.DataFrame(rep_test_rows)
    rep_tests["kruskal_padj_BH"] = bh_adjust(rep_tests["kruskal_p"].values)
    rep_tests["spearman_padj_BH"] = bh_adjust(rep_tests["spearman_p"].values)
    rep_tests.to_csv(OUT / "replicate_label_tests.tsv", sep="\t", index=False)

# Plot robust subset figures.
def plot_feature(c, data, outname, title_suffix=""):
    fig, ax = plt.subplots(figsize=(7, 5))
    xpos = {"DSW0": 0, "DSW10": 1, "DSW20": 2}
    for g in ["DSW0", "DSW10", "DSW20"]:
        vals = pd.to_numeric(data.loc[data["group_label"] == g, c], errors="coerce").dropna()
        x = np.random.default_rng(1).normal(xpos[g], 0.045, size=len(vals))
        ax.scatter(x, vals, alpha=0.75)
        if len(vals):
            ax.hlines(vals.median(), xpos[g]-0.2, xpos[g]+0.2, linewidth=3)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["0% DSW", "10% DSW", "20% DSW"])
    ax.set_title(f"{c}{title_suffix}")
    ax.set_ylabel(c)
    ax.set_xlabel("Condition")
    fig.tight_layout()
    fig.savefig(OUT / outname, dpi=240)
    plt.close(fig)

for c in ["eps_proxy_index", "branchpoints_per_100um2", "skel_length_um_per_100um2"]:
    if c in feature_cols:
        plot_feature(c, df, f"plot_all_images_{c}.png", " — all images")
        subdf = df[~df[c + "_outlier_absrz_gt3p5"]].copy()
        plot_feature(c, subdf, f"plot_no_global_outliers_{c}.png", " — no global robust outliers")

# Produce final classification table.
class_rows = []
for c in feature_cols:
    all_row = tests[tests["feature"] == c].iloc[0].to_dict()
    sens_row = sens[sens["feature"] == c].iloc[0].to_dict()

    direction_all = summary.pivot(index="feature", columns="group", values="median").loc[c]
    med0 = direction_all.get("DSW0", np.nan)
    med20 = direction_all.get("DSW20", np.nan)

    status = "exploratory"
    reason = []

    if pd.notna(all_row["kruskal_padj_BH"]) and all_row["kruskal_padj_BH"] < 0.05:
        reason.append("image-level Kruskal BH-significant")
    if pd.notna(all_row["spearman_padj_BH"]) and all_row["spearman_padj_BH"] < 0.05:
        reason.append("image-level salinity trend BH-significant")
    if pd.notna(sens_row["kruskal_padj_BH"]) and sens_row["kruskal_padj_BH"] < 0.05:
        reason.append("outlier-sensitivity Kruskal BH-significant")
    if pd.notna(sens_row["spearman_padj_BH"]) and sens_row["spearman_padj_BH"] < 0.05:
        reason.append("outlier-sensitivity salinity trend BH-significant")

    if len(reason) >= 2:
        status = "moderate_image_level_support"
    elif len(reason) == 1:
        status = "weak_image_level_support"
    else:
        status = "visual_or_exploratory_only"

    class_rows.append({
        "feature": c,
        "median_DSW0_all_images": med0,
        "median_DSW20_all_images": med20,
        "direction_DSW20_vs_DSW0": "higher" if med20 > med0 else "lower" if med20 < med0 else "flat",
        "classification_without_confirmed_biorep": status,
        "supporting_reasons": "; ".join(reason),
        "caution": "Image fields are not guaranteed independent biological replicates; use as SEM support unless culture-level replicate mapping is confirmed."
    })

classification = pd.DataFrame(class_rows)
classification.to_csv(OUT / "FINAL_SEM_feature_support_classification.tsv", sep="\t", index=False)

# Print report.
print("\n==== SEM EPS robust reanalysis v2 ====")
print("Input rows:", len(df))
print("\nGroup counts:")
print(df["group_label"].value_counts().to_string())
print("\nQC group imaging summary:")
print(qc_group.to_string(index=False))
print("\nMagnification counts by group:")
print(mag_counts.to_string())
print("\nMagnification-bin counts by group:")
print(magbin_counts.to_string())
print("\nImage-level tests:")
print(tests.to_string(index=False))
print("\nOutlier sensitivity:")
print(sens.to_string(index=False))
print("\nCommon magnification bins represented in all groups with >=2 images/group:", common_bins)
if not bin_tests.empty:
    print("\nMagnification-bin sensitivity:")
    print(bin_tests.to_string(index=False))
print("\nFinal classification:")
print(classification.to_string(index=False))
print("\nWrote outputs to:", OUT)
