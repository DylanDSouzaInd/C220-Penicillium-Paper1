import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path

OUT = Path("/data/dylan/01_ONT_PCRcDNARNASeq/salinity_rnaseq_analysis/10_final_thesis_paper_outputs/26_RESULT5_FIG4_RNASEQ")
FIGS = OUT / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

title = "Experimental design and ONT cDNA RNA-seq processing workflow"

conditions = [
    ("0% DSW", "B1  B2  B3"),
    ("10% DSW", "B1  B2  B3"),
    ("20% DSW", "B1  B2  B3"),
]

workflow_steps = [
    ("Cultivation", "Penicillium sp. C220\n0%, 10%, 20% DSW"),
    ("RNA-seq libraries", "ONT PCR-cDNA\n3 biological replicates\nper condition"),
    ("Read QC", "Barcode FASTQ merge\nNanoPlot / seqkit QC"),
    ("cDNA processing", "Pychopper\nfull-length cDNA\norientation/filtering"),
    ("Genome alignment", "minimap2 mapping\nto project genome"),
    ("Gene counts", "featureCounts\ngene-level matrix"),
    ("Expression analysis", "DESeq2\npairwise Wald tests\nFDR < 0.05; |log2FC| ≥ 1"),
]

footer = (
    "Nine ONT cDNA libraries were analyzed. Differential expression was tested for "
    "10% vs 0%, 20% vs 0%, and 20% vs 10% DSW."
)

fig, ax = plt.subplots(figsize=(17, 6.2))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

# Title
ax.text(
    0.5, 0.96, title,
    ha="center", va="top",
    fontsize=20, fontweight="bold"
)

# Experimental design label
ax.text(
    0.5, 0.84, "Experimental design",
    ha="center", va="center",
    fontsize=14, fontweight="bold"
)

# Condition boxes
cond_y = 0.67
cond_w = 0.20
cond_h = 0.13
cond_xs = [0.17, 0.40, 0.63]
cond_colors = ["#dbeafe", "#ffedd5", "#dcfce7"]

for x, (main, reps), color in zip(cond_xs, conditions, cond_colors):
    patch = FancyBboxPatch(
        (x, cond_y), cond_w, cond_h,
        boxstyle="round,pad=0.02,rounding_size=0.025",
        linewidth=1.6,
        edgecolor="black",
        facecolor=color
    )
    ax.add_patch(patch)

    ax.text(
        x + cond_w / 2, cond_y + cond_h * 0.62,
        main,
        ha="center", va="center",
        fontsize=14, fontweight="bold"
    )
    ax.text(
        x + cond_w / 2, cond_y + cond_h * 0.32,
        reps,
        ha="center", va="center",
        fontsize=12, fontweight="bold"
    )

# Arrow from design to workflow
ax.add_patch(
    FancyArrowPatch(
        (0.50, 0.64), (0.50, 0.56),
        arrowstyle="->",
        mutation_scale=18,
        linewidth=2
    )
)

ax.text(
    0.53, 0.60,
    "9 ONT cDNA libraries",
    ha="left", va="center",
    fontsize=11
)

# Workflow boxes
y = 0.30
w = 0.118
h = 0.17
xs = [0.025, 0.155, 0.285, 0.415, 0.545, 0.675, 0.805]

for i, (x, (head, body)) in enumerate(zip(xs, workflow_steps)):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.018,rounding_size=0.022",
        linewidth=1.4,
        edgecolor="black",
        facecolor="#f8fafc"
    )
    ax.add_patch(patch)

    ax.text(
        x + w / 2, y + h * 0.72,
        head,
        ha="center", va="center",
        fontsize=10.5, fontweight="bold"
    )
    ax.text(
        x + w / 2, y + h * 0.39,
        body,
        ha="center", va="center",
        fontsize=8.7
    )

    if i < len(xs) - 1:
        ax.add_patch(
            FancyArrowPatch(
                (x + w, y + h / 2),
                (xs[i + 1], y + h / 2),
                arrowstyle="->",
                mutation_scale=14,
                linewidth=1.6
            )
        )

# Bottom note
ax.text(
    0.5, 0.105,
    footer,
    ha="center", va="center",
    fontsize=11
)

ax.text(
    0.5, 0.045,
    "Processing route used for gene-level salinity-response testing.",
    ha="center", va="center",
    fontsize=9.5,
    color="0.25"
)

fig.tight_layout()

fig.savefig(
    FIGS / "Figure_4A_ONT_RNAseq_design_workflow_SPECIFIC.png",
    dpi=300,
    bbox_inches="tight"
)
fig.savefig(
    FIGS / "Figure_4A_ONT_RNAseq_design_workflow_SPECIFIC.pdf",
    bbox_inches="tight"
)

print("Saved:")
print(FIGS / "Figure_4A_ONT_RNAseq_design_workflow_SPECIFIC.png")
print(FIGS / "Figure_4A_ONT_RNAseq_design_workflow_SPECIFIC.pdf")
