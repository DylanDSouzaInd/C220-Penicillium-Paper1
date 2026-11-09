suppressPackageStartupMessages({
  library(DESeq2)
  library(ggplot2)
  library(pheatmap)
})

outdir <- "/data/dylan/01_ONT_PCRcDNARNASeq/salinity_rnaseq_analysis/06_deseq2_v2"
dir.create(file.path(outdir, "figures"), showWarnings=FALSE, recursive=TRUE)
dir.create(file.path(outdir, "tables"), showWarnings=FALSE, recursive=TRUE)
dir.create(file.path(outdir, "results_all_samples"), showWarnings=FALSE, recursive=TRUE)
dir.create(file.path(outdir, "results_no_DSW20_B2"), showWarnings=FALSE, recursive=TRUE)

counts0 <- read.delim(file.path(outdir, "gene_counts_matrix.annotation_v2.tsv"), check.names=FALSE)
meta0 <- read.delim(file.path(outdir, "samples.tsv"), check.names=FALSE)

rownames(counts0) <- counts0$gene_id
counts0$gene_id <- NULL

rownames(meta0) <- meta0$sample
meta0$condition <- factor(meta0$condition, levels=c("DSW0","DSW10","DSW20"))
meta0$salinity <- as.numeric(meta0$salinity)

counts0 <- counts0[, rownames(meta0)]

run_deseq <- function(counts, meta, label, outsubdir) {
  dds <- DESeqDataSetFromMatrix(
    countData = round(as.matrix(counts)),
    colData = meta,
    design = ~ condition
  )

  keep <- rowSums(counts(dds) >= 10) >= 3
  dds <- dds[keep,]

  dds <- DESeq(dds)

  norm_counts <- counts(dds, normalized=TRUE)
  vst_mat <- assay(vst(dds, blind=TRUE))

  write.table(
    data.frame(gene_id=rownames(norm_counts), norm_counts, check.names=FALSE),
    file.path(outdir, "tables", paste0("normalized_counts_", label, ".tsv")),
    sep="\t", quote=FALSE, row.names=FALSE
  )

  write.table(
    data.frame(gene_id=rownames(vst_mat), vst_mat, check.names=FALSE),
    file.path(outdir, "tables", paste0("vst_expression_", label, ".tsv")),
    sep="\t", quote=FALSE, row.names=FALSE
  )

  detected <- data.frame(
    sample = colnames(counts),
    total_counts = colSums(counts),
    detected_ge1 = colSums(counts >= 1),
    detected_ge10 = colSums(counts >= 10),
    condition = meta$condition
  )

  write.table(
    detected,
    file.path(outdir, "tables", paste0("sample_count_depth_detected_features_", label, ".tsv")),
    sep="\t", quote=FALSE, row.names=FALSE
  )

  pdf(file.path(outdir, "figures", paste0("library_size_barplot_", label, ".pdf")), width=8, height=5)
  print(
    ggplot(detected, aes(x=sample, y=total_counts, fill=condition)) +
      geom_col() +
      theme_bw() +
      theme(axis.text.x=element_text(angle=45, hjust=1)) +
      labs(title=paste("Assigned counts:", label), x="Sample", y="Assigned counts")
  )
  dev.off()

  pdf(file.path(outdir, "figures", paste0("detected_features_barplot_", label, ".pdf")), width=8, height=5)
  print(
    ggplot(detected, aes(x=sample, y=detected_ge10, fill=condition)) +
      geom_col() +
      theme_bw() +
      theme(axis.text.x=element_text(angle=45, hjust=1)) +
      labs(title=paste("Detected loci:", label), x="Sample", y="Features with count >= 10")
  )
  dev.off()

  pdf(file.path(outdir, "figures", paste0("PCA_vst_", label, ".pdf")), width=7, height=5.5)
  print(plotPCA(vst(dds, blind=TRUE), intgroup="condition") + theme_bw() + ggtitle(paste("PCA:", label)))
  dev.off()

  sample_dists <- dist(t(vst_mat))
  sample_dist_mat <- as.matrix(sample_dists)
  rownames(sample_dist_mat) <- colnames(vst_mat)
  colnames(sample_dist_mat) <- colnames(vst_mat)

  pdf(file.path(outdir, "figures", paste0("sample_distance_heatmap_", label, ".pdf")), width=7, height=6)
  pheatmap(sample_dist_mat, main=paste("Sample distances:", label))
  dev.off()

  pdf(file.path(outdir, "figures", paste0("dispersion_plot_", label, ".pdf")), width=7, height=5)
  plotDispEsts(dds, main=paste("Dispersion estimates:", label))
  dev.off()

  res_10_vs_0 <- results(dds, contrast=c("condition","DSW10","DSW0"))
  res_20_vs_0 <- results(dds, contrast=c("condition","DSW20","DSW0"))
  res_20_vs_10 <- results(dds, contrast=c("condition","DSW20","DSW10"))

  write.csv(as.data.frame(res_10_vs_0[order(res_10_vs_0$padj),]),
            file.path(outdir, outsubdir, paste0("DESeq2_DSW10_vs_DSW0_", label, ".csv")))
  write.csv(as.data.frame(res_20_vs_0[order(res_20_vs_0$padj),]),
            file.path(outdir, outsubdir, paste0("DESeq2_DSW20_vs_DSW0_", label, ".csv")))
  write.csv(as.data.frame(res_20_vs_10[order(res_20_vs_10$padj),]),
            file.path(outdir, outsubdir, paste0("DESeq2_DSW20_vs_DSW10_", label, ".csv")))

  dds_lrt <- DESeq(dds, test="LRT", reduced=~1)
  res_lrt <- results(dds_lrt)
  write.csv(as.data.frame(res_lrt[order(res_lrt$padj),]),
            file.path(outdir, outsubdir, paste0("DESeq2_salinity_overall_LRT_", label, ".csv")))

  summarize_res <- function(res, name) {
    df <- as.data.frame(res)
    up <- sum(!is.na(df$padj) & df$padj < 0.05 & df$log2FoldChange >= 1)
    down <- sum(!is.na(df$padj) & df$padj < 0.05 & df$log2FoldChange <= -1)
    total <- sum(!is.na(df$padj) & df$padj < 0.05)
    data.frame(comparison=name, padj_lt_0.05=total,
               up_LFC_ge_1=up, down_LFC_le_minus1=down)
  }

  summary <- rbind(
    summarize_res(res_10_vs_0, "DSW10_vs_DSW0"),
    summarize_res(res_20_vs_0, "DSW20_vs_DSW0"),
    summarize_res(res_20_vs_10, "DSW20_vs_DSW10"),
    data.frame(
      comparison="salinity_overall_LRT",
      padj_lt_0.05=sum(!is.na(res_lrt$padj) & res_lrt$padj < 0.05),
      up_LFC_ge_1=NA,
      down_LFC_le_minus1=NA
    )
  )

  write.table(summary,
              file.path(outdir, outsubdir, paste0("DESeq2_summary_counts_", label, ".tsv")),
              sep="\t", quote=FALSE, row.names=FALSE)

  saveRDS(dds, file.path(outdir, "tables", paste0("dds_", label, ".rds")))
  return(summary)
}

summary_all <- run_deseq(counts0, meta0, "all", "results_all_samples")

meta_sens <- meta0[meta0$sample != "DSW20_B2",]
counts_sens <- counts0[, rownames(meta_sens)]
summary_sens <- run_deseq(counts_sens, meta_sens, "no_DSW20_B2", "results_no_DSW20_B2")

summary_all$analysis <- "all_9_samples"
summary_sens$analysis <- "exclude_DSW20_B2"
combined <- rbind(summary_all, summary_sens)

write.table(combined,
            file.path(outdir, "DESeq2_all_vs_no_DSW20_B2_summary.tsv"),
            sep="\t", quote=FALSE, row.names=FALSE)

print(combined)
