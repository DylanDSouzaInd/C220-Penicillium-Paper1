#!/usr/bin/env Rscript
# C220 GEO V7: original nine-sample DESeq2 model provenance + processed data export.
# READS original RDS, complete raw matrix, existing normalized matrix and original DESeq2 CSVs.
# DOES NOT rerun DESeq(), modify originals, or upload anything.
suppressPackageStartupMessages(library(DESeq2))
a <- commandArgs(trailingOnly=TRUE)
if (length(a)!=1) stop('Usage: Rscript C220_GEO_verify_and_export_model_v7.R NEW_OUTPUT_DIRECTORY')
out <- normalizePath(a[[1]], mustWork=FALSE)
if (file.exists(out) || dir.exists(out)) stop('Output directory exists; use a NEW path: ', out)
base <- '/data/dylan/01_ONT_PCRcDNARNASeq/salinity_rnaseq_analysis'
file <- function(s) file.path(base,s)
rds <- file('06_deseq2_v2/dds_final_all_samples.rds')
raw_file <- file('06_deseq2_v2/gene_counts_matrix.annotation_v2.tsv')
legacy_file <- file('10_final_thesis_paper_outputs/57_BGC_FIGURE5_AUTHENTIC_REBUILD/07_BGC_EXPRESSION_DESEQ2_FIGURE5/tables/05_all_genes_DESeq2_normalized_counts.tsv')
stats <- c(
 '10vs0'=file('06_deseq2_v2/results_all_samples/DESeq2_DSW10_vs_DSW0_all.csv'),
 '20vs0'=file('06_deseq2_v2/results_all_samples/DESeq2_DSW20_vs_DSW0_all.csv'),
 '20vs10'=file('06_deseq2_v2/results_all_samples/DESeq2_DSW20_vs_DSW10_all.csv'))
for(p in c(rds,raw_file,legacy_file,unname(stats))) if(!file.exists(p))stop('Missing: ',p)
want <- c('DSW0_B1','DSW0_B2','DSW0_B3','DSW10_B1','DSW10_B2','DSW10_B3','DSW20_B1','DSW20_B2','DSW20_B3')
readmatrix <- function(p) {
 x <- read.delim(p, row.names=1, check.names=FALSE, quote='"', comment.char='', stringsAsFactors=FALSE)
 if(anyDuplicated(rownames(x)) || anyNA(rownames(x))) stop('Duplicated/NA feature IDs: ',p)
 if(!all(want %in% colnames(x))) stop('Missing sample column(s): ',p, '\nCols: ',paste(colnames(x),collapse=','))
 x <- as.matrix(x[,want,drop=FALSE]); suppressWarnings(storage.mode(x)<-'double')
 if(any(!is.finite(x))) stop('Non-numeric / NA input matrix: ',p)
 x
}
raw <- readmatrix(raw_file)
legacy <- readmatrix(legacy_file)
if (nrow(raw)!=13455 || nrow(legacy)!=13455) stop('Expected original 13,455 rows. Found ',nrow(raw),' and ',nrow(legacy))
if(!setequal(rownames(raw),rownames(legacy)))stop('Full raw/normalized gene ID sets differ')
if(any(raw < 0) || any(abs(raw-round(raw))>1e-8))stop('Raw matrix must contain nonnegative integers')
dds <- readRDS(rds)
if(!inherits(dds,'DESeqDataSet'))stop('RDS is not a DESeqDataSet')
if(!identical(colnames(dds),want))stop('RDS sample order or identities changed')
if(!identical(paste(deparse(design(dds)),collapse=''),'~condition'))stop('Unexpected model design: ',deparse(design(dds)))
model_raw <- as.matrix(counts(dds,normalized=FALSE))
model_norm <- as.matrix(counts(dds,normalized=TRUE))
if(nrow(dds)!=8275 || ncol(dds)!=9)stop('Unexpected filtered RDS dimensions')
if(!all(rownames(model_raw)%in%rownames(raw)))stop('RDS contains unknown IDs')
if(!isTRUE(all.equal(unname(model_raw),unname(raw[rownames(model_raw),want]),check.attributes=FALSE,tolerance=0))) stop('BLOCK: RDS raw counts do not match complete raw input')
if(is.null(sizeFactors(dds)) || !is.null(normalizationFactors(dds)))stop('Unexpected normalization type')
expected <- c('0vs10','0vs20','10vs20')
# Detect existing result CSV gene IDs, compare model output where factor levels match.
readresults <- function(p) {
 x <- read.csv(p, row.names=1,check.names=FALSE,stringsAsFactors=FALSE)
 if(!all(c('log2FoldChange','padj') %in% colnames(x)))stop('Unrecognized result columns: ',p)
 if(anyDuplicated(rownames(x))) stop('Duplicate genes in result table')
 x
}
res_report <- list()
levels_info <- levels(colData(dds)$condition)
contrast_candidates <- list('10vs0'=c('DSW10','DSW0'),'20vs0'=c('DSW20','DSW0'),'20vs10'=c('DSW20','DSW10'))
# If labels are numeric or percentages, make explicit; never guess contrast direction.
can_reproduce <- all(c('DSW0','DSW10','DSW20') %in% levels_info)
for(k in names(stats)) {
 saved <- readresults(stats[[k]])
 if(can_reproduce) {
  r <- results(dds,contrast=c('condition',contrast_candidates[[k]]))
  shared <- intersect(rownames(saved),rownames(r))
  if(!length(shared))stop('No shared genes with DESeq2 results: ',k)
  old <- as.numeric(saved[shared,'padj']); new <- as.numeric(r[shared,'padj'])
  good <- is.finite(old)&is.finite(new)
  p_err<-if(any(good)) max(abs(old[good]-new[good])) else NA_real_
  oldfc<-as.numeric(saved[shared,'log2FoldChange']); newfc<-as.numeric(r[shared,'log2FoldChange'])
  fg<-is.finite(oldfc)&is.finite(newfc)
  fc_err<-if(any(fg))max(abs(oldfc[fg]-newfc[fg])) else NA_real_
  state<-if(is.finite(p_err) && is.finite(fc_err) && p_err<1e-8 && fc_err<1e-8) 'EXACT_MODEL_STATS' else 'CHECK_MODEL_RESULTS'
  res_report[[k]]<-data.frame(contrast=k,model_levels=paste(levels_info,collapse=';'),state=state,saved_gene_rows=nrow(saved),shared_rows=length(shared),max_abs_padj_error=p_err,max_abs_log2FC_error=fc_err)
 } else {
  res_report[[k]]<-data.frame(contrast=k,model_levels=paste(levels_info,collapse=';'),state='LEVELS_REQUIRE_MANUAL_CONTRAST_REVIEW',saved_gene_rows=nrow(saved),shared_rows=NA,max_abs_padj_error=NA,max_abs_log2FC_error=NA)
 }
}
# Diagnostic is robust to low-count rounding: compare sufficiently expressed genes (both >= 100).
report <- list(); sampled <- list()
for(s in want) {
 x <- as.numeric(model_norm[,s]); y <- as.numeric(legacy[rownames(model_norm),s]); r <- as.numeric(model_raw[,s]);
 ok <- is.finite(x)&is.finite(y)&x>=100&y>=100
 # high-count multiplicative scaling ratio, and residual after adjusting it
 ratio <- y[ok]/x[ok]
 median_scale <- if(length(ratio)) median(ratio) else NA_real_
 residual <- if(length(ratio))abs(ratio/median_scale-1)else numeric()
 # raw / legacy high-count implied factor for all-feature export
 full_r <- as.numeric(raw[,s]); full_y <- as.numeric(legacy[rownames(raw),s]);
 h <- full_r>=100 & full_y>=100 & is.finite(full_y)
 implied <- full_r[h]/full_y[h]
 factor <- if(length(implied))median(implied) else NA_real_
 model_sf <- unname(sizeFactors(dds)[s])
 classification <- if(length(ratio)<100)'INSUFFICIENT_HIGH_COUNTS' else if(quantile(residual,0.95,na.rm=TRUE) < 0.005)'APPROXIMATE_SAMPLE_SCALE_ONLY' else 'GENE_SPECIFIC_DIFFERENCE_REVIEW'
 report[[s]] <- data.frame(sample=s,shared_model_genes=length(x),high_count_genes=length(ratio),saved_model_size_factor=model_sf,full_export_implied_factor=factor,legacy_over_model_high_count_median_ratio=median_scale,high_count_p95_residual_after_scale=if(length(residual))unname(quantile(residual,.95))else NA_real_,high_count_max_residual_after_scale=if(length(residual))max(residual)else NA_real_,diagnosis=classification)
 # example 30 largest relative residuals for traceability
 if(length(ratio)) {
  idx<-which(ok); order_idx<-idx[order(abs((y[idx]/x[idx])/median_scale-1),decreasing=TRUE)][seq_len(min(30,length(idx)))]
  sampled[[s]]<-data.frame(sample=s,gene_id=rownames(model_norm)[order_idx],model_normalized=x[order_idx],existing_export=y[order_idx],relative_difference_after_median_scale=abs((y[order_idx]/x[order_idx])/median_scale-1))
 }
}
dir.create(out,recursive=TRUE)
write.table(do.call(rbind,report),file.path(out,'01_NORMALIZATION_DIAGNOSTIC.tsv'),sep='\t',quote=FALSE,row.names=FALSE)
write.table(do.call(rbind,res_report),file.path(out,'02_ORIGINAL_DESEQ2_RESULTS_REPRODUCTION.tsv'),sep='\t',quote=FALSE,row.names=FALSE)
write.table(do.call(rbind,sampled),file.path(out,'03_TOP_DIFFERENT_FEATURES.tsv'),sep='\t',quote=FALSE,row.names=FALSE)
# Exactly model-owned numbers, not recomputed/rounded and not expanded into full 13,455 set.
export <- data.frame(gene_id=rownames(model_norm),model_norm,check.names=FALSE)
write.table(export,file.path(out,'04_DESeq2_original_fitted_model_normalized_counts_8275.tsv'),sep='\t',quote=FALSE,row.names=FALSE,col.names=TRUE,na='NA')
write.table(data.frame(sample=want,original_model_size_factor=unname(sizeFactors(dds)[want])),file.path(out,'05_MODEL_SIZE_FACTORS.tsv'),sep='\t',quote=FALSE,row.names=FALSE)
writeLines(c(
 'C220 GEO provenance v7 — no DESeq model refitting.',
 paste('Original input raw matrix:',raw_file),
 paste('Saved DESeq2 object:',rds),
 paste('Existing 13,455-feature Figure4 legacy normalized matrix:',legacy_file),
 'The 13,455-feature raw matrix is the complete quantification; 8,275 features passed the original model filter.',
 'The 8,275-feature normalized export is obtained directly from counts(dds, normalized=TRUE) in the original saved object.',
 'Never replace the original Figure4 13,455-feature matrix without resolving its source provenance.',
 'The v7 report checks if the mismatch is explained by per-sample scaling or needs gene-level investigation.',
 'These files are generated in a separate output directory. No FTP transfer or GEO submission has occurred.'),file.path(out,'README_PROVENANCE.txt'))
cat('V7_COMPLETE ',out,'\n',sep='')
cat('RAW_COUNT_EXACT_MODEL_MATCH 9/9; MODEL_GENES ',nrow(model_raw),' COMPLETE_RAW_GENES ',nrow(raw),'\n',sep='')
for(i in report) cat('NORMALIZATION_DIAGNOSIS ',i$sample,' ',i$diagnosis,' median_ratio ',signif(i$legacy_over_model_high_count_median_ratio,5),' highcount_p95_resid ',signif(i$high_count_p95_residual_after_scale,5),'\n',sep='')
for(i in res_report)cat('DESEQ2_RESULT ',i$contrast,' ',i$state,' max_padj_diff ',i$max_abs_padj_error,'\n',sep='')
cat('READ_ONLY_INPUTS; NO REANALYSIS; NO UPLOAD\n')
