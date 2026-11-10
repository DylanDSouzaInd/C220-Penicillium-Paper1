suppressPackageStartupMessages({
  library(DESeq2)
  library(ggplot2)
})

outdir <- "/data/dylan/01_ONT_PCRcDNARNASeq/salinity_rnaseq_analysis/06_deseq2_v2"
dir.create(file.path(outdir, "results_all_samples"), showWarnings=FALSE, recursive=TRUE)
dir.create(file.path(outdir, "figures_all_samples"), showWarnings=FALSE, recursive=TRUE)

counts <- read.delim(file.path(outdir, "gene_counts_matrix.annotation_v2.tsv"), check.names=FALSE)
meta <- read.delim(file.path(outdir, "samples.tsv"), check.names=FALSE)

rownames(counts) <- counts$gene_id
counts$gene_id <- NULL

rownames(meta) <- meta$sample
meta$condition <- factor(meta$condition, levels=c("DSW0","DSW10","DSW20"))
counts <- counts[, rownames(meta)]

dds <- DESeqDataSetFromMatrix(
  countData = round(as.matrix(counts)),
  colData = meta,
  design = ~ condition
)

keep <- rowSums(counts(dds) >= 10) >= 3
dds <- dds[keep,]

dds <- DESeq(dds)

res_10_vs_0 <- results(dds, contrast=c("condition","DSW10","DSW0"))
res_20_vs_0 <- results(dds, contrast=c("condition","DSW20","DSW0"))
res_20_vs_10 <- results(dds, contrast=c("condition","DSW20","DSW10"))

write.csv(as.data.frame(res_10_vs_0[order(res_10_vs_0$padj),]),
          file.path(outdir, "results_all_samples/DESeq2_DSW10_vs_DSW0_all.csv"))

write.csv(as.data.frame(res_20_vs_0[order(res_20_vs_0$padj),]),
          file.path(outdir, "results_all_samples/DESeq2_DSW20_vs_DSW0_all.csv"))

write.csv(as.data.frame(res_20_vs_10[order(res_20_vs_10$padj),]),
          file.path(outdir, "results_all_samples/DESeq2_DSW20_vs_DSW10_all.csv"))

# LRT: any condition effect
dds_lrt <- DESeq(dds, test="LRT", reduced=~1)
res_lrt <- results(dds_lrt)
write.csv(as.data.frame(res_lrt[order(res_lrt$padj),]),
          file.path(outdir, "results_all_samples/DESeq2_salinity_overall_LRT_all.csv"))

summarize_res <- function(res, name) {
  df <- as.data.frame(res)
  up <- sum(!is.na(df$padj) & df$padj < 0.05 & df$log2FoldChange >= 1)
  down <- sum(!is.na(df$padj) & df$padj < 0.05 & df$log2FoldChange <= -1)
  total <- sum(!is.na(df$padj) & df$padj < 0.05)
  data.frame(comparison=name, padj_lt_0.05=total, up_LFC_ge_1=up, down_LFC_le_minus1=down)
}

summary <- rbind(
  summarize_res(res_10_vs_0, "DSW10_vs_DSW0"),
  summarize_res(res_20_vs_0, "DSW20_vs_DSW0"),
  summarize_res(res_20_vs_10, "DSW20_vs_DSW10"),
  data.frame(comparison="salinity_overall_LRT",
             padj_lt_0.05=sum(!is.na(res_lrt$padj) & res_lrt$padj < 0.05),
             up_LFC_ge_1=NA,
             down_LFC_le_minus1=NA)
)

write.table(summary,
            file.path(outdir, "results_all_samples/DESeq2_summary_counts_all.tsv"),
            sep="\t", quote=FALSE, row.names=FALSE)

saveRDS(dds, file.path(outdir, "dds_final_all_samples.rds"))

print(summary)
