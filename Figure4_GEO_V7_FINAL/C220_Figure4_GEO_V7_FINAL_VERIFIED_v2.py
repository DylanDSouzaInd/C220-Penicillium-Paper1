#!/usr/bin/env python3
"""C220 Paper 1 Figure 4: strict saved-model / original-DESeq2 figure generation.

Requires the GEO V7 saved-model normalized-count matrix, November 10 108-value
crosscheck, and original DESeq2 10v0, 20v0, 20v10 CSVs. Never fits a new model.
Refuses ambiguous/missing statistics or mismatched normalized source values.
Writes a new directory; does not overwrite historical plots.
"""
import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

GENES = [
 ('DSWSTRG.12473','Putative cell-wall/surface'),
 ('DSWSTRG.6916','AA11 CAZyme candidate'),
 ('DSWSTRG.2225','GH43 glycoside hydrolase'),
 ('DSWSTRG.7271','Glycosyltransferase-like'),
 ('DSWSTRG.7303','GH18 chitinase-like'),
 ('DSWSTRG.7881','MFS transporter'),
 ('DSWSTRG.7160','Amino-acid permease'),
 ('DSWSTRG.4764','Laccase/MCO-like'),
 ('DSWSTRG.12325','SDR oxidoreductase-like'),
 ('DSWSTRG.12648','HAD-like hydrolase'),
 ('DSWSTRG.9665','O-methyltransferase-like'),
 ('DSWSTRG.2532','Low-variability candidate'),
]
CONTRASTS=[('DSW10_vs_DSW0','DESeq2_DSW10_vs_DSW0_all.csv',0,1),
           ('DSW20_vs_DSW0','DESeq2_DSW20_vs_DSW0_all.csv',0,2),
           ('DSW20_vs_DSW10','DESeq2_DSW20_vs_DSW10_all.csv',1,2)]
SAMPLES=[f'DSW{c}_B{b}' for c in (0,10,20) for b in (1,2,3)]
COLORS={0:'#4E79A7',10:'#E28B38',20:'#5A956D'}


def digest(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for block in iter(lambda:f.read(1024*1024), b''):
   h.update(block)
 return h.hexdigest()


def identify(df, names):
 return next((c for c in df.columns if c.strip().lower() in names),None)


def gene_key(series):
 return series.astype(str).str.strip().str.replace(r'^(DSWSTRG\.\d+)\.\d+$',r'\1',regex=True)


def load_comparison(path):
 d=pd.read_csv(path,sep='\t',dtype={'gene_id':str,'sample':str})
 needed={'gene_id','sample','geo_v7_model_norm','oct_legacy_norm'}
 if not needed.issubset(d.columns): raise ValueError(f'Comparison missing: {needed-set(d.columns)}')
 expected={(g,s) for g,_ in GENES for s in SAMPLES}
 actual=list(zip(d.gene_id,d['sample']))
 if len(actual)!=108 or len(set(actual))!=108 or set(actual)!=expected:
  raise ValueError('Comparison must contain the exact 12 genes x nine samples, once each')
 for c in ('geo_v7_model_norm','oct_legacy_norm'):
  d[c]=pd.to_numeric(d[c],errors='raise')
  if not np.isfinite(d[c]).all() or (d[c]<0).any(): raise ValueError(f'Invalid counts in {c}')
 d['condition']=d['sample'].str.extract(r'DSW(0|10|20)_B')[0].astype(int)
 d['replicate']=d['sample'].str.extract(r'_B([123])$')[0].astype(int)
 return d


def verify_saved_model(d,model_path):
 m=pd.read_csv(model_path,sep='\t',low_memory=False)
 missing=[s for s in SAMPLES if s not in m.columns]
 if missing: raise ValueError(f'Saved-model matrix missing sample columns: {missing}')
 idcol=identify(m,{'gene','gene_id','geneid','id','unnamed: 0','locus','transcript_id'})
 if idcol is None:
  idcol=m.columns[0]
  if idcol in SAMPLES: raise ValueError('Cannot infer saved-model gene ID column')
 m['__gene']=gene_key(m[idcol])
 if m['__gene'].duplicated().any():raise ValueError('Duplicated gene IDs in model matrix')
 ids=[g for g,_ in GENES]
 missing_genes=set(ids)-set(m['__gene'])
 if missing_genes:raise ValueError(f'Saved-model matrix missing genes: {sorted(missing_genes)}')
 model=m.set_index('__gene').loc[ids,SAMPLES]
 diffs=[]
 for r in d.itertuples(index=False):
  reference=float(model.loc[r.gene_id,r.sample])
  current=float(r.geo_v7_model_norm)
  if not np.isclose(reference,current,rtol=1e-9,atol=1e-7):
   diffs.append((r.gene_id,r.sample,reference,current))
 if diffs:raise ValueError(f'Comparison disagrees with saved DESeq2 normalized counts: first 5 {diffs[:5]}')
 return len(m),len(d)


def parse_stats(folder):
 allrows=[]
 for contrast,filename,x1,x2 in CONTRASTS:
  p=folder/filename
  if not p.is_file():raise FileNotFoundError(f'Original DESeq2 contrast missing: {p}')
  v=pd.read_csv(p,low_memory=False)
  idcol=identify(v,{'gene','gene_id','geneid','id','unnamed: 0','locus'})
  if idcol is None:raise ValueError(f'No explicit gene identifier column in {p}')
  padj=identify(v,{'padj','fdr','qvalue','q_value'})
  lfc=identify(v,{'log2foldchange','log2fc','log2_fold_change'})
  if padj is None or lfc is None: raise ValueError(f'Missing adjusted p or log2FC in {p}')
  v['__gene']=gene_key(v[idcol])
  if v['__gene'].duplicated().any():raise ValueError(f'Duplicate gene ID in {p}')
  v=v.set_index('__gene')
  for gene,_ in GENES:
   if gene not in v.index:raise ValueError(f'{gene} missing in {p}')
   q=pd.to_numeric(pd.Series([v.loc[gene,padj]]),errors='coerce').iloc[0]
   fold=pd.to_numeric(pd.Series([v.loc[gene,lfc]]),errors='coerce').iloc[0]
   if pd.notna(q) and not 0<=q<=1:raise ValueError(f'Invalid padj: {gene} {contrast} {q}')
   if pd.notna(fold) and not np.isfinite(fold):raise ValueError('Nonfinite log2FC')
   label='NA' if pd.isna(q) else '****' if q<1e-4 else '***' if q<1e-3 else '**' if q<.01 else '*' if q<.05 else 'ns'
   allrows.append(dict(gene_id=gene,contrast=contrast,source_csv=filename,padj=q,log2FoldChange=fold,significance=label,bracket=(label not in ('NA','ns'))))
 result=pd.DataFrame(allrows)
 # Known historical checkpoints help flag a wrong contrasts directory or reversed mapping.
 checkpoints={('DSW10_vs_DSW0','DSWSTRG.12473'):5.41e-4,
              ('DSW10_vs_DSW0','DSWSTRG.6916'):5.10e-4,
              ('DSW10_vs_DSW0','DSWSTRG.7303'):4.11e-7,
              ('DSW20_vs_DSW0','DSWSTRG.12473'):6.81e-8,
              ('DSW20_vs_DSW0','DSWSTRG.6916'):2.08e-7,
              ('DSW20_vs_DSW0','DSWSTRG.7303'):.650,
              ('DSW20_vs_DSW10','DSWSTRG.7303'):.003}
 for (contrast,gene),known in checkpoints.items():
  q=result.loc[(result.contrast==contrast)&(result.gene_id==gene),'padj'].iloc[0]
  tolerance=max(.15*known,.005 if known>.02 else 1e-8)
  if pd.isna(q) or abs(q-known)>tolerance:
   raise ValueError(f'Original-DESeq2 checkpoint failed for {contrast} {gene}: observed {q}, checkpoint {known}')
 return result


def make_plot(d,stats,out):
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12.7,'mathtext.fontset':'stix', 'pdf.fonttype':42,'svg.fonttype':'none'})
 fig,axs=plt.subplots(3,4,figsize=(22,16.5),layout='constrained')
 fig.get_layout_engine().set(wspace=.105,hspace=.20,w_pad=.16,h_pad=.18)
 stats_index=stats.set_index(['gene_id','contrast'])
 summaries=[]
 for n,(gene,label) in enumerate(GENES):
  ax=axs.flat[n]
  subset=d[d.gene_id==gene]
  transformed=np.log10(subset.geo_v7_model_norm.to_numpy()+1)
  bottom=float(np.min(transformed));top=float(np.max(transformed));span=max(.75,top-bottom)
  for j,condition in enumerate((0,10,20)):
   s=subset[subset.condition==condition].sort_values('replicate')
   ys=np.log10(s.geo_v7_model_norm.to_numpy()+1)
   # Center both the mean bar and biological replicate observations at the same condition.
   # Bar is behind the points, so an observation at the mean stays visible.
   xs=j+np.array([-.125, 0.0, .125])
   ax.hlines(float(np.mean(ys)),j-.19,j+.19,lw=2.05,color='#172533',zorder=3)
   ax.scatter(xs,ys,s=54,color=COLORS[condition],edgecolors='white',linewidths=.85,zorder=5)
   summaries.append(dict(gene_id=gene,condition=condition,n=3,
                         mean_log10_normalized_plus1=float(np.mean(ys)),
                         sd_log10_normalized_plus1=float(np.std(ys,ddof=1)),
                         mean_normalized_count=float(np.mean(s.geo_v7_model_norm))))
  brackets=[]
  for contrast,filename,a,b in CONTRASTS:
   row=stats_index.loc[(gene,contrast)]
   if row.bracket:brackets.append((a,b,row.significance,float(row.padj)))
  gap=max(.21,.12*span)
  bracket_start=top+max(.22,.13*span)
  for i,(a,b,stars,q) in enumerate(brackets):
   y=bracket_start+i*gap
   ax.plot([a,a,b,b],[y-.045,y,y,y-.045],color='#33465a',lw=.95,clip_on=False)
   ax.text((a+b)/2,y+.018,stars,weight='bold',fontsize=12,ha='center',va='bottom')
  ymax=max(top+max(.50,.26*span),bracket_start+(len(brackets)-1)*gap+max(.25,.15*span) if brackets else top)
  ymin=max(0.,bottom-max(.16,.13*span))
  if bottom<=.05:ymin=-.09
  ax.set_ylim(ymin,ymax)
  ax.set_xlim(-.46,2.44)
  ax.set_title(f'{label}\n{gene}',fontsize=14.0,pad=19,weight='bold',linespacing=1.14)
  ax.set_xticks(range(3),['0%','10%','20%'])
  ax.tick_params(labelsize=12.2)
  ax.spines[['top','right']].set_visible(False)
  ax.grid(axis='y',alpha=.18,linewidth=.5)
  ax.set_axisbelow(True)
  if n%4==0:ax.set_ylabel(r'log$_{10}$(normalized count + 1)',fontsize=14.0,labelpad=10)
  ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
 handles=[Line2D([0],[0],marker='o',markersize=8,color='none',markerfacecolor=COLORS[c],
                 markeredgecolor='white',label=f'{c}% DSW') for c in (0,10,20)]
 fig.suptitle(r'$\mathbfit{Penicillium}$ sp. C220 | RNA-seq salinity-associated candidates',fontsize=21,weight='bold')
 fig.legend(handles=handles,loc='outside lower center',ncol=3,frameon=False,fontsize=13)
 for ext in ('png','pdf','svg'):
  fig.savefig(out/f'Figure4_GEO_V7_FINAL_VERIFIED_v2.{ext}',dpi=300,bbox_inches='tight')
 plt.close(fig)
 return summaries


def main():
 pa=argparse.ArgumentParser(description=__doc__)
 pa.add_argument('--comparison',type=Path,required=True)
 pa.add_argument('--model-counts',type=Path,required=True)
 pa.add_argument('--deseq-dir',type=Path,required=True)
 pa.add_argument('--outdir',type=Path,required=True)
 a=pa.parse_args()
 for p in (a.comparison,a.model_counts):
  if not p.is_file():pa.error(f'Input missing: {p}')
 if not a.deseq_dir.is_dir():pa.error(f'DESeq results folder missing: {a.deseq_dir}')
 if a.outdir.exists() and any(a.outdir.iterdir()):
  pa.error(f'Output directory not empty; refusing to overwrite: {a.outdir}')
 a.outdir.mkdir(parents=True,exist_ok=True)
 print('CHECK_1_COMPARISON',flush=True)
 d=load_comparison(a.comparison)
 print('CHECK_2_SAVED_MODEL',flush=True)
 genes_in_model,n_values=verify_saved_model(d,a.model_counts)
 print('SAVED_MODEL_VALUES_EXACT',n_values,'MODEL_GENES',genes_in_model,flush=True)
 print('CHECK_3_ORIGINAL_DESEQ2_PVALUES',flush=True)
 stats=parse_stats(a.deseq_dir)
 stats.to_csv(a.outdir/'Figure4_DESeq2_36_COMPARISON_AUDIT.tsv',sep='\t',index=False)
 print('DESEQ2_CONTRASTS_VERIFIED',len(stats),'SIGNIFICANT_BRACKETS',int(stats.bracket.sum()),flush=True)
 print('CHECK_4_RENDERING',flush=True)
 summary=make_plot(d,stats,a.outdir)
 pd.DataFrame(summary).to_csv(a.outdir/'Figure4_12_gene_condition_summary.tsv',sep='\t',index=False)
 report={'status':'PASS_GEO_V7_MODEL_AND_ORIGINAL_DESEQ2_CHECKS',
         'n_genes':len(GENES),'n_samples':len(SAMPLES),'n_comparison_values':n_values,
         'n_contrast_gene_tests':len(stats),'n_significance_brackets':int(stats.bracket.sum()),
         'expression_transformation':'log10(saved-model normalized count + 1)',
         'mean_line':'centered horizontal arithmetic mean of the three log10-transformed sample values; replicate points displayed on top' ,
         'significance':'original DESeq2 contrast padj < 0.05; NA is not called significant',
         'sources':{str(p):digest(p) for p in [a.comparison,a.model_counts]+[a.deseq_dir/f for _,f,_,_ in CONTRASTS]},
         'limitations':'Not an independent DESeq2 rerun; archived images require separate visual comparison before manuscript replacement.'}
 (a.outdir/'Figure4_FINAL_VERIFICATION.json').write_text(json.dumps(report,indent=2)+'\n')
 print('C220_FIGURE4_FINAL_VERIFICATION_PASS',a.outdir,flush=True)
 print('MANUSCRIPT_ORIGINALS_UNCHANGED',flush=True)

if __name__=='__main__':main()
