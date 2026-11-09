#!/usr/bin/env python3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

data = pd.DataFrame({
    "DSW_percent": [0, 10, 20, 30, 40, 50, 60],
    "Exp_1": [0.1195, 2.2113, 2.5231, 0.9204, 0.5577, 0.3575, 0.0825],
    "Exp_2": [0.6271, 1.8337, 2.2234, 0.9454, 0.4313, 0.4256, 0.0633],
    "Exp_3": [0.7732, 1.9939, 2.5557, 1.1050, 0.4288, 0.4677, 0.1356],
})

long_df = data.melt(
    id_vars="DSW_percent",
    value_vars=["Exp_1", "Exp_2", "Exp_3"],
    var_name="Experiment",
    value_name="Dry_biomass_g"
)

summary = long_df.groupby("DSW_percent")["Dry_biomass_g"].agg(["mean", "std", "count"]).reset_index()

labels = [f"{x}%" for x in data["DSW_percent"]]
x_pos = np.arange(len(labels))
x_map = dict(zip(data["DSW_percent"], x_pos))
long_df["x_pos"] = long_df["DSW_percent"].map(x_map)
summary["x_pos"] = summary["DSW_percent"].map(x_map)

outdir = Path("C220_DSW_biomass_plot")
outdir.mkdir(exist_ok=True)

long_df.drop(columns=["x_pos"]).to_csv(outdir / "C220_DSW_dry_biomass_long_final.csv", index=False)
summary.drop(columns=["x_pos"]).to_csv(outdir / "C220_DSW_dry_biomass_summary_final.csv", index=False)

def make_plot(path_prefix, title=True):
    fig, ax = plt.subplots(figsize=(5.8, 4.0))

    offsets = {"Exp_1": -0.10, "Exp_2": 0.00, "Exp_3": 0.10}
    for exp, offset in offsets.items():
        sub = long_df[long_df["Experiment"] == exp]
        ax.scatter(
            sub["x_pos"] + offset,
            sub["Dry_biomass_g"],
            s=48,
            marker="o",
            edgecolor="black",
            linewidth=0.5,
            label=exp.replace("_", " ")
        )

    ax.errorbar(
        summary["x_pos"],
        summary["mean"],
        yerr=summary["std"],
        fmt="-o",
        capsize=4,
        linewidth=1.6,
        markersize=4.2,
        color="black",
        ecolor="black",
        label="Mean ± SD"
    )

    if title:
        ax.set_title("Endpoint dry biomass after 7 days", pad=10)

    ax.set_xlabel("Dead Sea Water concentration (% v/v in PDB)")
    ax.set_ylabel("Dry biomass (g per 50 mL culture)")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels)
    ax.set_xlim(-0.55, len(labels) - 0.45)
    ax.set_ylim(0, 3.0)
    ax.set_yticks(np.arange(0, 3.1, 0.5))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.text(x_map[60], 0.32, "No visible\ngrowth", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()

    fig.savefig(outdir / f"{path_prefix}.png", dpi=600, bbox_inches="tight")
    fig.savefig(outdir / f"{path_prefix}.pdf", bbox_inches="tight")
    fig.savefig(outdir / f"{path_prefix}.svg", bbox_inches="tight")

make_plot("C220_DSW_dry_biomass_3dots_meanSD_all_percent_labels", title=True)
make_plot("C220_DSW_dry_biomass_3dots_meanSD_all_percent_labels_no_title", title=False)
print(f"Done. Files written to: {outdir.resolve()}")
