from pathlib import Path
import sys, os, re, json, math
import pandas as pd
import numpy as np

ROOT = Path(os.environ["ROOT"])
DE_DIR = Path(os.environ["DE_DIR"])
ANN_DIR = Path(os.environ["ANN_DIR"])
BIO_DIR = Path(os.environ["BIO_DIR"])
OUT = Path(os.environ["OUT"])

T10_0 = Path(os.environ["T10_0"])
T20_0 = Path(os.environ["T20_0"])
T20_10 = Path(os.environ["T20_10"])

FDR = 0.05
LFC = 1.0
SAMPLES = [
    "DSW0_B1","DSW0_B2","DSW0_B3",
    "DSW10_B1","DSW10_B2","DSW10_B3",
    "DSW20_B1","DSW20_B2","DSW20_B3",
]

# Paper-1 candidate identities already established in the manuscript/supplement.
# These are used ONLY as cross-references; they do NOT determine DEG membership.
PAPER1 = {
    "DSWSTRG.449":   ("Reference-gene candidate 1", "Reference candidate", True),
    "DSWSTRG.2532":  ("Reference-gene candidate 2", "Reference candidate", True),
    "DSWSTRG.11494": ("Reference-gene candidate 3", "Reference candidate", True),
    "DSWSTRG.12473": ("Putative cell-wall/surface protein", "Cell surface / extracellular remodelling", True),
    "DSWSTRG.6916":  ("AA11 CAZyme / putative extracellular protein", "CAZyme / carbohydrate remodelling", True),
    "DSWSTRG.2225":  ("GH43 glycoside hydrolase", "CAZyme / carbohydrate remodelling", True),
    "DSWSTRG.7271":  ("Glycosyltransferase-like surface glycan candidate", "Cell surface / glycan remodelling", True),
    "DSWSTRG.7303":  ("GH18 chitinase / glycoside hydrolase", "Cell-wall remodelling", True),
    "DSWSTRG.7881":  ("MFS / major facilitator superfamily transporter", "Transport / osmoadaptation", True),
    "DSWSTRG.7160":  ("Amino-acid/leucine/proline permease", "Transport / osmoadaptation", True),
    "DSWSTRG.4764":  ("Laccase/multicopper oxidase-like protein", "Oxidative / surface remodelling", True),
    "DSWSTRG.12325": ("SDR/short-chain dehydrogenase-like oxidoreductase", "Metabolic / redox adjustment", True),
    "DSWSTRG.12648": ("HAD-like / haloacid-dehalogenase type II-like hydrolase", "Metabolic adjustment", True),
    # Retained as RNA-seq-only candidate in the current Paper-1 revision logic.
    "DSWSTRG.9665":  ("Putative O-methyltransferase", "Metabolic adjustment", False),
}


def read_tsv(path):
    try:
        return pd.read_csv(path, sep="\t", dtype=str, low_memory=False)
    except Exception as exc:
        raise RuntimeError(f"Could not read TSV {path}: {exc}")


def find_col(df, preferred, contains=()):
    # exact, case-insensitive
    lower = {str(c).lower(): c for c in df.columns}
    for p in preferred:
        if p.lower() in lower:
            return lower[p.lower()]
    for c in df.columns:
        lc = str(c).lower()
        if any(k.lower() in lc for k in contains):
            return c
    return None


def detect_gene_col(df):
    c = find_col(df, ["gene_id","gene","GeneID","row","id"], ["gene_id"])
    if c is not None:
        return c
    # Prefer a first column dominated by DSWSTRG IDs.
    for c in df.columns[:5]:
        vals = df[c].astype(str).head(200)
        if vals.str.startswith("DSWSTRG.").mean() > 0.5:
            return c
    return df.columns[0]


def standardize(path, prefix):
    df = read_tsv(path)
    gcol = detect_gene_col(df)
    lfc_col = find_col(df, ["log2FoldChange","log2FC","log2_fc"], ["log2fold","log2fc"])
    padj_col = find_col(df, ["padj","FDR","adj.P.Val","adjusted_pvalue"], ["padj","fdr","adjusted"])
    p_col = find_col(df, ["pvalue","p_value","PValue"], ["pvalue","p_value"])
    bm_col = find_col(df, ["baseMean","basemean"], ["basemean"])
    if lfc_col is None or padj_col is None:
        raise RuntimeError(
            f"{path.name}: could not identify log2FoldChange/padj columns.\n"
            f"Columns: {list(df.columns)}"
        )
    out = pd.DataFrame({"gene_id": df[gcol].astype(str)})
    out[f"{prefix}_baseMean"] = pd.to_numeric(df[bm_col], errors="coerce") if bm_col else np.nan
    out[f"{prefix}_log2FC"] = pd.to_numeric(df[lfc_col], errors="coerce")
    out[f"{prefix}_pvalue"] = pd.to_numeric(df[p_col], errors="coerce") if p_col else np.nan
    out[f"{prefix}_padj"] = pd.to_numeric(df[padj_col], errors="coerce")
    out[f"{prefix}_significant"] = (
        out[f"{prefix}_padj"].notna()
        & (out[f"{prefix}_padj"] < FDR)
        & out[f"{prefix}_log2FC"].notna()
        & (out[f"{prefix}_log2FC"].abs() >= LFC)
    )
    out[f"{prefix}_direction"] = np.where(
        out[f"{prefix}_significant"],
        np.where(out[f"{prefix}_log2FC"] > 0, "up", "down"),
        "not_significant"
    )
    # Keep compact genomic context from the annotated result where available.
    context = {}
    context_patterns = [
        "class_code", "annotation_class", "collapsed", "reference", "contig",
        "seqname", "start", "end", "strand", "transcript_id", "ref_gene",
        "ref_transcript"
    ]
    for c in df.columns:
        lc = str(c).lower()
        if any(k in lc for k in context_patterns) and c not in [gcol, lfc_col, padj_col, p_col, bm_col]:
            context[f"{prefix}__{c}"] = df[c].astype(str)
    if context:
        out = pd.concat([out, pd.DataFrame(context)], axis=1)
    return out, df, gcol


r10, raw10, g10 = standardize(T10_0, "DSW10_vs_DSW0")
r20, raw20, g20 = standardize(T20_0, "DSW20_vs_DSW0")
r2010, raw2010, g2010 = standardize(T20_10, "DSW20_vs_DSW10")

sig20 = r20[r20["DSW20_vs_DSW0_significant"]].copy()
sig2010 = r2010[r2010["DSW20_vs_DSW10_significant"]].copy()

# Hard validation against the Paper-1 numbers.
expected = {
    "20v0_total": 35,
    "20v0_up": 35,
    "20v0_down": 0,
    "20v10_total": 19,
    "20v10_up": 18,
    "20v10_down": 1,
}
obs = {
    "20v0_total": len(sig20),
    "20v0_up": int((sig20["DSW20_vs_DSW0_direction"] == "up").sum()),
    "20v0_down": int((sig20["DSW20_vs_DSW0_direction"] == "down").sum()),
    "20v10_total": len(sig2010),
    "20v10_up": int((sig2010["DSW20_vs_DSW10_direction"] == "up").sum()),
    "20v10_down": int((sig2010["DSW20_vs_DSW10_direction"] == "down").sum()),
}
if obs != expected:
    raise RuntimeError(
        "The original tables did NOT reproduce the Paper-1 DEG counts.\n"
        f"Expected: {expected}\nObserved: {obs}\n"
        "STOPPING rather than silently building the wrong Supplementary Table."
    )

ids20 = set(sig20.gene_id)
ids2010 = set(sig2010.gene_id)
union_ids = sorted(ids20 | ids2010)
shared_ids = sorted(ids20 & ids2010)

# Merge only the union of the two 20%-related DEG sets.
merged = pd.DataFrame({"gene_id": union_ids})
for tab in [r10, r20, r2010]:
    statcols = [c for c in tab.columns if "__" not in c]
    merged = merged.merge(tab[statcols], on="gene_id", how="left")


def sig_contrasts(row):
    z=[]
    if bool(row.get("DSW20_vs_DSW0_significant", False)):
        z.append("20% vs 0%")
    if bool(row.get("DSW20_vs_DSW10_significant", False)):
        z.append("20% vs 10%")
    return "; ".join(z)


def response20(row):
    parts=[]
    if bool(row.get("DSW20_vs_DSW0_significant", False)):
        parts.append("increased vs 0%" if row["DSW20_vs_DSW0_log2FC"] > 0 else "decreased vs 0%")
    if bool(row.get("DSW20_vs_DSW10_significant", False)):
        parts.append("increased vs 10%" if row["DSW20_vs_DSW10_log2FC"] > 0 else "decreased vs 10%")
    return "; ".join(parts)

merged["significant_contrast_s"] = merged.apply(sig_contrasts, axis=1)
merged["response_at_20_DSW"] = merged.apply(response20, axis=1)

# Cross-reference Paper-1 candidate panel.
merged["paper1_preselected_candidate"] = merged.gene_id.map(lambda g: "YES" if g in PAPER1 else "NO")
merged["paper1_working_annotation"] = merged.gene_id.map(lambda g: PAPER1[g][0] if g in PAPER1 else "")
merged["paper1_functional_module"] = merged.gene_id.map(lambda g: PAPER1[g][1] if g in PAPER1 else "")
merged["paper1_qPCR_primer_in_S1"] = merged.gene_id.map(lambda g: "YES" if g in PAPER1 and PAPER1[g][2] else "NO")

# ------------------------------------------------------------
# Gather EXISTING annotation evidence; never use it to decide significance.
# ------------------------------------------------------------
annotation_files = []
known = [
    BIO_DIR / "robust_DE_master_table_with_all_blast_annotations.tsv",
    BIO_DIR / "robust_DE_master_table_with_blast_annotations.tsv",
    BIO_DIR / "robust_DE_ion_homeostasis_candidates_ALLANNOT.tsv",
]
for p in known:
    if p.is_file() and p.stat().st_size > 0:
        annotation_files.append(p)

# Also inspect additional manageable TSVs already present in biological_interpretation.
if BIO_DIR.is_dir():
    for p in sorted(BIO_DIR.glob("*.tsv")):
        if p not in annotation_files and p.stat().st_size <= 250_000_000:
            annotation_files.append(p)

ann_records=[]
for path in annotation_files:
    try:
        df = read_tsv(path)
    except Exception:
        continue
    if df.empty:
        continue
    gcol = detect_gene_col(df)
    mask = df[gcol].astype(str).isin(union_ids)
    if not mask.any():
        continue
    # Identify columns likely to contain human-readable annotation evidence.
    ann_cols=[]
    keys = [
        "annotation", "description", "stitle", "product", "protein", "header",
        "blast", "cazy", "dbcan", "pfam", "family", "domain", "target_id",
        "sseqid", "accession", "subject", "gene_name", "symbol"
    ]
    for c in df.columns:
        lc=str(c).lower()
        if c != gcol and any(k in lc for k in keys):
            ann_cols.append(c)
    # If no named annotation columns, still preserve the whole row in the raw evidence output.
    for _, row in df.loc[mask].iterrows():
        gid=str(row[gcol])
        pieces=[]
        for c in ann_cols:
            v=str(row[c]).strip()
            if v and v.lower() not in {"nan","na","none",""}:
                pieces.append(f"{c}={v}")
        ann_records.append({
            "gene_id":gid,
            "source_file":str(path),
            "annotation_evidence":" | ".join(pieces)[:12000],
        })

ann_df = pd.DataFrame(ann_records)
if ann_df.empty:
    ann_df = pd.DataFrame(columns=["gene_id","source_file","annotation_evidence"])

# Collapse evidence per gene while retaining source provenance.
def collapse_evidence(group):
    ev=[]
    sources=[]
    for _, r in group.iterrows():
        if r["annotation_evidence"]:
            ev.append(r["annotation_evidence"])
        sources.append(Path(r["source_file"]).name)
    # de-duplicate, preserve order
    ev=list(dict.fromkeys(ev))
    sources=list(dict.fromkeys(sources))
    return pd.Series({
        "existing_annotation_evidence":" || ".join(ev)[:20000],
        "annotation_source_files":"; ".join(sources),
    })

if not ann_df.empty:
    collapsed = ann_df.groupby("gene_id", sort=False).apply(collapse_evidence).reset_index()
    merged = merged.merge(collapsed, on="gene_id", how="left")
else:
    merged["existing_annotation_evidence"]=""
    merged["annotation_source_files"]=""

merged["existing_annotation_evidence"] = merged["existing_annotation_evidence"].fillna("")
merged["annotation_source_files"] = merged["annotation_source_files"].fillna("")

# Extract a conservative auto-summary from evidence. This is an aid for review, not a new annotation claim.
def auto_annotation(row):
    if row["paper1_working_annotation"]:
        return row["paper1_working_annotation"]
    txt=row["existing_annotation_evidence"]
    if not txt:
        return "Function unresolved from currently indexed annotation tables"
    # Prefer explicit 'description:' text, then protein labels from Dead Sea transcript headers.
    m=re.search(r"description[:=]([^|]{3,180})", txt, flags=re.I)
    if m:
        val=m.group(1).strip(" ;\t")
        if val:
            return val
    # Search for informative protein terms.
    patterns = [
        r"([A-Za-z0-9'()\- /]+(?:transporter|permease|hydrolase|oxidase|oxidoreductase|transferase|synthetase|synthase|kinase|phosphatase|chitinase|glycosidase|dehydrogenase|methyltransferase|endonuclease|exoribonuclease|domain-containing protein))",
    ]
    for pat in patterns:
        mm=re.search(pat, txt, flags=re.I)
        if mm:
            val=re.sub(r"\s+"," ",mm.group(1)).strip(" ;|=")
            if 3 <= len(val) <= 180:
                return val
    return "Protein homolog detected; specific function unresolved"

merged["best_supported_annotation_for_review"] = merged.apply(auto_annotation, axis=1)

# Automated broad category is intentionally conservative.
def category(row):
    if row["paper1_functional_module"]:
        return row["paper1_functional_module"]
    s=(row["best_supported_annotation_for_review"] + " " + row["existing_annotation_evidence"]).lower()
    if any(k in s for k in ["cell wall","chitin","glycosyl","glycoside","cazy","glucan","mannan"]):
        return "Cell-wall / carbohydrate-associated"
    if any(k in s for k in ["transporter","permease","channel","antiporter","atpase"]):
        return "Transport / membrane-associated"
    if any(k in s for k in ["oxidoreductase","dehydrogenase","oxidase","redox","catalase","peroxidase"]):
        return "Redox / metabolic-associated"
    if any(k in s for k in ["stress","heat shock","chaperone","autophagy","vacuol"]):
        return "Stress-associated"
    return "Unresolved / other"

merged["functional_relevance_for_paper1_review"] = merged.apply(category, axis=1)

# ------------------------------------------------------------
# Optional normalized-count discovery for the EXTENDED audit only.
# ------------------------------------------------------------
def try_normalized_counts():
    candidates=[]
    for base in [DE_DIR, ROOT]:
        if not base.is_dir():
            continue
        for pat in ["*normal*count*.tsv","*normalized*.tsv","*normalised*.tsv"]:
            candidates.extend(base.rglob(pat))
    seen=set()
    for p in candidates:
        if p in seen or not p.is_file() or p.stat().st_size > 500_000_000:
            continue
        seen.add(p)
        try:
            df=read_tsv(p)
        except Exception:
            continue
        if not set(SAMPLES).issubset(set(df.columns)):
            continue
        gcol=detect_gene_col(df)
        sub=df[[gcol]+SAMPLES].copy()
        sub[gcol]=sub[gcol].astype(str)
        if len(set(sub[gcol]) & set(union_ids)) < max(5, int(0.5*len(union_ids))):
            continue
        for s in SAMPLES:
            sub[s]=pd.to_numeric(sub[s], errors="coerce")
        sub=sub[sub[gcol].isin(union_ids)].drop_duplicates(gcol)
        sub=sub.rename(columns={gcol:"gene_id"})
        sub["norm_DSW0_mean"]=sub[["DSW0_B1","DSW0_B2","DSW0_B3"]].mean(axis=1)
        sub["norm_DSW10_mean"]=sub[["DSW10_B1","DSW10_B2","DSW10_B3"]].mean(axis=1)
        sub["norm_DSW20_mean"]=sub[["DSW20_B1","DSW20_B2","DSW20_B3"]].mean(axis=1)
        return p, sub
    return None, None

norm_source, norm_df = try_normalized_counts()
if norm_df is not None:
    merged = merged.merge(norm_df, on="gene_id", how="left")

# Single downregulated gene in the 20 vs 10 comparison.
down = merged[
    merged["DSW20_vs_DSW10_significant"].fillna(False)
    & (merged["DSW20_vs_DSW10_log2FC"] < 0)
].copy()
if len(down) != 1:
    raise RuntimeError(f"Expected exactly one downregulated 20% vs 10% gene; observed {len(down)}")

down_gene = down.iloc[0]["gene_id"]
merged["single_downregulated_20_vs_10"] = np.where(merged.gene_id == down_gene, "YES", "NO")

# Sort: single down first; then shared significant genes; then by strongest 20-related adjusted p.
merged["_shared"] = merged["DSW20_vs_DSW0_significant"].fillna(False) & merged["DSW20_vs_DSW10_significant"].fillna(False)
merged["_minpadj"] = merged[["DSW20_vs_DSW0_padj","DSW20_vs_DSW10_padj"]].min(axis=1, skipna=True)
merged = merged.sort_values(
    ["single_downregulated_20_vs_10","_shared","_minpadj"],
    ascending=[False,False,True], na_position="last"
).drop(columns=["_shared","_minpadj"])

# Main paste-ready table: concise enough for the supplement.
main_cols = [
    "gene_id",
    "DSW10_vs_DSW0_log2FC","DSW10_vs_DSW0_padj",
    "DSW20_vs_DSW0_log2FC","DSW20_vs_DSW0_padj",
    "DSW20_vs_DSW10_log2FC","DSW20_vs_DSW10_padj",
    "significant_contrast_s","response_at_20_DSW",
    "best_supported_annotation_for_review",
    "functional_relevance_for_paper1_review",
    "paper1_preselected_candidate","paper1_qPCR_primer_in_S1",
    "single_downregulated_20_vs_10",
]
main = merged[main_cols].copy()
main = main.rename(columns={
    "gene_id":"Gene ID",
    "DSW10_vs_DSW0_log2FC":"10% vs 0% log2FC",
    "DSW10_vs_DSW0_padj":"10% vs 0% adjusted P",
    "DSW20_vs_DSW0_log2FC":"20% vs 0% log2FC",
    "DSW20_vs_DSW0_padj":"20% vs 0% adjusted P",
    "DSW20_vs_DSW10_log2FC":"20% vs 10% log2FC",
    "DSW20_vs_DSW10_padj":"20% vs 10% adjusted P",
    "significant_contrast_s":"Significant contrast(s)",
    "response_at_20_DSW":"Response at 20% DSW",
    "best_supported_annotation_for_review":"Best-supported functional annotation",
    "functional_relevance_for_paper1_review":"Functional relevance to Paper 1",
    "paper1_preselected_candidate":"Preselected Paper-1 candidate",
    "paper1_qPCR_primer_in_S1":"qPCR primer in Table S1",
    "single_downregulated_20_vs_10":"Single decreased 20% vs 10% locus",
})

# Formatting for human readability.
for c in [x for x in main.columns if "log2FC" in x]:
    main[c] = pd.to_numeric(main[c], errors="coerce").round(3)
for c in [x for x in main.columns if "adjusted P" in x]:
    main[c] = pd.to_numeric(main[c], errors="coerce")

OUT.mkdir(parents=True, exist_ok=True)
main.to_csv(OUT / "Supplementary_Table_S2_FOR_PASTE.tsv", sep="\t", index=False, na_rep="NA")
merged.to_csv(OUT / "Supplementary_Table_S2_EXTENDED_AUDIT.tsv", sep="\t", index=False, na_rep="NA")
sig20.to_csv(OUT / "01_DSW20_vs_DSW0_35_DEGs.tsv", sep="\t", index=False, na_rep="NA")
sig2010.to_csv(OUT / "02_DSW20_vs_DSW10_19_DEGs.tsv", sep="\t", index=False, na_rep="NA")
down.to_csv(OUT / "03_SINGLE_DOWNREGULATED_20_vs_DSW10.tsv", sep="\t", index=False, na_rep="NA")
ann_df.to_csv(OUT / "04_existing_annotation_evidence_rows.tsv", sep="\t", index=False, na_rep="NA")

# Also make a markdown version for easy viewing/pasting.
try:
    md = main.to_markdown(index=False)
except Exception:
    md = main.to_csv(sep="\t", index=False)
(OUT / "Supplementary_Table_S2_FOR_PASTE.md").write_text(md + "\n", encoding="utf-8")

# Optional XLSX workbook if openpyxl/xlsxwriter is available.
xlsx_written=False
for engine in ["openpyxl", "xlsxwriter"]:
    try:
        with pd.ExcelWriter(OUT / "Supplementary_Table_S2_FINAL.xlsx", engine=engine) as writer:
            main.to_excel(writer, sheet_name="Table_S2_main", index=False)
            merged.to_excel(writer, sheet_name="Extended_audit", index=False)
            sig20.to_excel(writer, sheet_name="20v0_35_DEGs", index=False)
            sig2010.to_excel(writer, sheet_name="20v10_19_DEGs", index=False)
            down.to_excel(writer, sheet_name="single_down_gene", index=False)
            ann_df.to_excel(writer, sheet_name="annotation_evidence", index=False)
        xlsx_written=True
        break
    except Exception:
        continue

summary = []
summary.append("C220 PAPER 1 — SUPPLEMENTARY TABLE S2 FINAL AUDIT")
summary.append("==================================================")
summary.append("")
summary.append(f"Source 10% vs 0%:  {T10_0}")
summary.append(f"Source 20% vs 0%:  {T20_0}")
summary.append(f"Source 20% vs 10%: {T20_10}")
summary.append("")
summary.append(f"20% vs 0% significant:  {obs['20v0_total']} total = {obs['20v0_up']} up, {obs['20v0_down']} down")
summary.append(f"20% vs 10% significant: {obs['20v10_total']} total = {obs['20v10_up']} up, {obs['20v10_down']} down")
summary.append(f"Shared between the two 20%-related contrasts: {len(shared_ids)}")
summary.append(f"Union for Supplementary Table S2: {len(union_ids)} genes")
summary.append("")
summary.append(f"SINGLE DOWNREGULATED 20% vs 10% GENE: {down_gene}")
summary.append(f"  log2FC = {down.iloc[0]['DSW20_vs_DSW10_log2FC']}")
summary.append(f"  adjusted P = {down.iloc[0]['DSW20_vs_DSW10_padj']}")
summary.append(f"  auto annotation = {down.iloc[0]['best_supported_annotation_for_review']}")
summary.append(f"  Paper-1 preselected candidate = {down.iloc[0]['paper1_preselected_candidate']}")
summary.append("")
summary.append("Shared 20%-contrast genes:")
summary.extend([f"  {g}" for g in shared_ids] or ["  none"])
summary.append("")
if norm_source is not None:
    summary.append(f"Normalized-count source used in extended audit: {norm_source}")
else:
    summary.append("Normalized-count source: not automatically located; not required for the main S2 table.")
summary.append("")
summary.append("IMPORTANT INTERPRETATION BOUNDARY:")
summary.append("- Significance is taken ONLY from the original all-nine-sample DESeq2 contrast tables.")
summary.append("- Existing BLAST/annotation tables are used only to annotate those already-selected genes.")
summary.append("- Paper-1 candidate labels are cross-references and do not determine DEG membership.")
summary.append("- Review 'Best-supported functional annotation' before manuscript submission; automated extraction is conservative but not a substitute for manual inspection of ambiguous hits.")
summary.append("")
summary.append(f"XLSX workbook written: {'YES' if xlsx_written else 'NO'}")
(OUT / "00_FINAL_SUMMARY.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")

# Save a small machine-readable manifest.
manifest = {
    "thresholds": {"padj_lt": FDR, "abs_log2FC_gte": LFC},
    "observed_counts": obs,
    "shared_20_contrasts": shared_ids,
    "union_gene_count": len(union_ids),
    "single_downregulated_20_vs_10_gene": down_gene,
    "normalized_count_source": str(norm_source) if norm_source else None,
    "annotation_files_consulted": [str(p) for p in annotation_files],
}
(OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

print("\n".join(summary))
