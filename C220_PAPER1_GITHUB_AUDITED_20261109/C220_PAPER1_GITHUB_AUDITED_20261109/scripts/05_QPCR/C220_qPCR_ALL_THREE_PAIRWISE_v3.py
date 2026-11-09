#!/usr/bin/env python3
"""
Penicillium C220 RT-qPCR; 0%, 10%, 20% DSW as THREE independent groups.

Input: final user-curated mean Cq table of three biological samples per condition.
Each biological sample's target Cq is normalized against EF1alpha/DSWSTRG.7219.1
from the SAME sample, without imposing a 0% reference group.

Plots: three separately displayed conditions, all individual biological data,
means +/- SEM (or SD), no connecting trend, all three pairwise brackets:
0 vs 10, 10 vs 20, and 0 vs 20. For visual clarity, log2 normalized
abundances are centered on the mean ACROSS ALL 9 samples for each gene; this
is a constant shift for plotting only and changes no test or fold difference.

Inference: two-sided Welch independent-group t-test of log2 normalized
quantities, Holm adjusted within the three contrasts per gene, and BH-FDR
across all 30 contrasts (10 genes x three comparisons). Not a dose trend.
No assumptions of paired biological batches are made; if B1/B2/B3 are
matched experimental blocks rather than independent cultures, revise tests.
Assumed equal PCR efficiency = 100% due to unavailable per-assay efficiency.

Requirements: Python 3.9+, numpy, scipy, matplotlib.
Run:
  python C220_qPCR_ALL_THREE_PAIRWISE_v3.py --outdir /data/dylan/C220_qPCR_THREE_GROUPS
Optional:
  --errorbar sd         (default sem; true SD always in numeric tables)
  --style bar           (default dot: recommended for n=3 per condition)
  --label-mode pq       (default q: bracket shows global BH q and 'ns'/*)
"""
import argparse
import csv
import math
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

GENES = ['12473','2225','6916','7160','7271','7303','7881','12648','4764','12325']
OTHER_REFERENCE_CANDIDATES = ['449','2532','11494']
FIELDS = ['EF1alpha','449','2532','11494','12473','2225','6916',
          '7160','7271','7303','7881','12648','4764','12325']
# Final submitted Cq values (23 October 2026): do not silently alter samples.
RAW = [
 ('B1', 0, 22.05,27.5,26.5,29.6,25.6,29.5,29.2,29.1,29.1,23.840,31.911,29.683,29.217,30.343),
 ('B1',10, 19.44,28.2,23.2,30.1,25.9,28.8,27.9,23.6,27.9,24.016,28.385,29.026,28.007,27.250),
 ('B1',20, 21.82,23.8,22.8,22.4,22.8,23.0,23.0,24.0,23.1,21.548,25.452,22.965,22.678,23.608),
 ('B2', 0, 21.77,23.4,22.5,22.5,24.0,26.8,23.6,24.5,23.2,23.450,25.030,23.110,23.555,24.393),
 ('B2',10, 20.91,24.9,22.8,27.9,25.6,24.9,23.0,23.6,23.2,21.908,25.606,23.595,22.686,23.830),
 ('B2',20, 21.24,26.1,22.4,22.3,22.6,22.7,22.4,24.3,23.0,21.510,24.547,22.729,22.040,22.764),
 ('B3', 0, 22.11,27.3,25.1,27.7,24.8,26.5,29.0,28.5,28.4,22.507,30.453,27.419,27.623,27.476),
 ('B3',10, 24.92,26.8,27.3,28.2,25.8,24.4,26.8,25.8,27.5,23.592,31.960,28.792,26.173,26.338),
 ('B3',20, 22.19,27.5,25.9,25.5,21.9,25.9,23.8,28.0,28.1,21.513,28.292,28.174,22.901,23.718),
]
CONDITIONS = (0,10,20)
BATCHES = ('B1','B2','B3')
PAIRWISE = ((0,10),(10,20),(0,20))
NAMES = {
 '12473':'Putative cell-wall/surface','2225':'GH43 hydrolase',
 '6916':'AA11 CAZyme','7160':'Amino-acid permease',
 '7271':'Glycosyltransferase-like','7303':'GH18 chitinase',
 '7881':'MFS transporter','12648':'HAD-like hydrolase',
 '4764':'Laccase/MCO-like','12325':'SDR oxidoreductase'
}
COLORS=('#537A9A','#BE804E','#4D8E78')


def save_rows(filepath, rows):
    if not rows:
        return
    with open(filepath, 'w', newline='', encoding='utf-8') as out:
        writer=csv.DictWriter(out, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def adjust_bh(pvalues):
    p=np.array(pvalues,dtype=float)
    if np.any(~np.isfinite(p)):
        raise ValueError('At least one pairwise p-value was nonfinite')
    order=np.argsort(p)
    q=np.minimum.accumulate((p[order]*len(p)/np.arange(1,len(p)+1))[::-1])[::-1]
    output=np.empty(len(p));output[order]=np.clip(q,0,1)
    return output


def adjust_holm(pvalues):
    p=np.array(pvalues,dtype=float)
    order=np.argsort(p)
    adjusted=np.maximum.accumulate((len(p)-np.arange(len(p)))*p[order])
    output=np.empty(len(p));output[order]=np.clip(adjusted,0,1)
    return output


def read_samples():
    d={}
    for b,c,*v in RAW:
        assert len(v)==len(FIELDS), (b,c,len(v))
        assert (b,c) not in d
        d[(b,c)]=dict(zip(FIELDS,map(float,v)))
    assert len(d)==9
    return d


def analyze(samples):
    per_sample=[];means=[];comparisons=[];series={}
    for gene in GENES:
        y={}
        q={}
        for c in CONDITIONS:
            q[c]=np.array([2.0**(samples[(b,c)]['EF1alpha']-samples[(b,c)][gene])
                           for b in BATCHES], dtype=float)
            y[c]=np.log2(q[c])
            for b,amount,value in zip(BATCHES,q[c],y[c]):
                target=samples[(b,c)][gene]
                reference=samples[(b,c)]['EF1alpha']
                per_sample.append(dict(batch=b,dsw_percent=c,gene=gene,
                                       target_cq=target,EF1alpha_cq=reference,
                                       delta_cq=target-reference,
                                       normalized_quantity=amount,
                                       log2_normalized_abundance=value))
        center=np.mean(np.concatenate([y[c] for c in CONDITIONS]))
        series[gene]={c: y[c]-center for c in CONDITIONS}
        for c in CONDITIONS:
            means.append(dict(gene=gene,dsw_percent=c,n_bio=3,
                       mean_normalized_quantity=np.mean(q[c]),
                       mean_log2_normalized_abundance=np.mean(y[c]),
                       sample_SD_log2=np.std(y[c],ddof=1),
                       sample_SEM_log2=np.std(y[c],ddof=1)/math.sqrt(3),
                       mean_display_log2=np.mean(series[gene][c]),
                       mean_fold_vs_0=np.mean(q[c])/np.mean(q[0]),
                       sample_SD_fold_vs_0=np.std(q[c]/np.mean(q[0]),ddof=1)))
        for a,b in PAIRWISE:
            # This test never uses technical replicates as independent n.
            test=ttest_ind(y[a],y[b],equal_var=False,alternative='two-sided')
            comparisons.append(dict(gene=gene,group_a=a,group_b=b,
                  contrast=f'{a}% vs {b}%',p_raw=float(test.pvalue),
                  mean_log2_B_minus_A=float(np.mean(y[b])-np.mean(y[a])),
                  fold_B_over_A=float(np.mean(q[b])/np.mean(q[a])),
                  test='Welch independent two-sided on sample log2 normalized abundance'))
    qvals=adjust_bh([r['p_raw'] for r in comparisons])
    for row, qvalue in zip(comparisons,qvals):
        row['q_BH_global_30tests']=float(qvalue)
        row['significant_BH_q_lt_0_05']='YES' if qvalue<.05 else 'NO'
    for gene in GENES:
        indices=[i for i,r in enumerate(comparisons) if r['gene']==gene]
        holm=adjust_holm([comparisons[i]['p_raw'] for i in indices])
        for index,value in zip(indices,holm):
            comparisons[index]['p_Holm_within_gene_3tests']=float(value)
    return per_sample, means, comparisons, series


def format_p(v):
    return f'{v:.2g}' if v<.001 else f'{v:.3f}'


def stars(v):
    if v < .001: return '***'
    if v < .01: return '**'
    if v < .05: return '*'
    return 'ns'


def bracket(ax,x1,x2,y,height,label):
    ax.plot([x1,x1,x2,x2],[y,y+height,y+height,y],
            color='#1D2A38',linewidth=1.15,clip_on=False,zorder=8)
    ax.text((x1+x2)/2,y+height+0.04,label,ha='center',va='bottom',
            fontsize=8.4,color='#1D2A38',zorder=9)


def one_panel(ax,gene,series,stats,errorbar,style,label_mode):
    values=series[gene]
    means=np.array([values[c].mean() for c in CONDITIONS])
    sd=np.array([values[c].std(ddof=1) for c in CONDITIONS])
    errors=sd if errorbar=='sd' else sd/np.sqrt(3)
    # Each condition is plotted as a distinct group (NOT a trend).
    if style=='bar':
        ax.bar([0,1,2],means,width=.58,alpha=.52,color=COLORS,edgecolor='#405364',
               linewidth=.8,zorder=2)
    offsets=(-.145,0,.145)
    for i,c in enumerate(CONDITIONS):
        for j,val in enumerate(values[c]):
            ax.scatter(i+offsets[j],val,color=COLORS[i],edgecolor='white',
                       linewidth=.9,s=62,zorder=5)
        ax.errorbar(i,means[i],yerr=errors[i],fmt='o',color='#142E42',
                    markerfacecolor='white',markeredgewidth=1.55,
                    markersize=7,linewidth=1.5,capsize=6,capthick=1.6,zorder=6)
        ax.hlines(means[i],i-.22,i+.22,color='#142E42',linewidth=1.3,zorder=4)
    all_y=np.concatenate([values[c] for c in CONDITIONS])
    lo=float(min(np.min(all_y),np.min(means-errors),0))
    hi=float(max(np.max(all_y),np.max(means+errors),0))
    dy=max(hi-lo,2.8)
    # 3 distinct, non-overlapping tiers; 0v10, 10v20 and 0v20.
    base=hi+.16*dy
    tier=.28*dy
    height=.055*dy
    for i,(a,b) in enumerate(PAIRWISE):
        r=stats[(gene,a,b)]
        q=r['q_BH_global_30tests']
        p=r['p_raw']
        core=f'q={format_p(q)} {stars(q)}'
        label=core if label_mode=='q' else f'p={format_p(p)}, {core}'
        bracket(ax,CONDITIONS.index(a),CONDITIONS.index(b),base+i*tier,height,label)
    ax.set_ylim(lo-.15*dy,base+3*tier+height+.12*dy)
    ax.set_xlim(-.42,2.42)
    ax.set_xticks([0,1,2],['0%','10%','20%'])
    ax.set_title(f'{gene}  ·  {NAMES[gene]}',fontsize=11.3,fontweight='bold',loc='left',pad=10)
    ax.spines[['top','right']].set_visible(False)
    ax.grid(axis='y',alpha=.16,linewidth=.7,zorder=0)
    ax.axhline(0,color='#8793A1',linestyle=':',linewidth=.9,zorder=0)
    ax.tick_params(labelsize=9.8)


def make_figure(outdir,stem,genes,series,stats,errorbar,style,label_mode,title):
    nrows=2 if len(genes)==5 else 5
    ncols=3 if len(genes)==5 else 2
    fig,axs=plt.subplots(nrows,ncols,figsize=(18.5,12.0) if len(genes)==5 else (16.2,26.0),dpi=120)
    for i,g in enumerate(genes):
        ax=axs.flat[i]
        one_panel(ax,g,series,stats,errorbar,style,label_mode)
        if (i%ncols)==0:
            ax.set_ylabel('Log₂ EF1α-normalized expression\n(pooled-mean centered)',fontsize=10)
    for i in range(len(genes),nrows*ncols):
        axs.flat[i].axis('off')
    if len(genes)==5:
        blank=axs.flat[-1]
        blank.text(.04,.92,'HOW TO READ',transform=blank.transAxes,fontsize=13,fontweight='bold')
        blank.text(.04,.78,
         '0%, 10% and 20% are separate groups.\n'
         'No monotonic dose-response model.\n\n'
         'Dots = biological samples (n=3/group)\n'
         f'White marker + whisker = mean ± {errorbar.upper()}\n'
         'Brackets = 0–10, 10–20 and 0–20\n'
         'q = BH adjusted over all 30 tests\n'
         'ns = q ≥ 0.05\n\n'
         'Log₂ values centered on pooled nine-sample\n'
         'mean for display only. No group is baseline.',
         transform=blank.transAxes,va='top',linespacing=1.65,fontsize=11)
    fig.suptitle(title,fontsize=18,fontweight='bold',y=.99)
    fig.text(.5,.954,f'All three group-to-group comparisons  •  Mean ± {errorbar.upper()}  •  Biological points, n=3/group',
             ha='center',fontsize=12)
    fig.text(.5,.02,
       'Assumes equal PCR efficiencies; EF1α is a single provisional reference. No technical wells treated as biological replicates.',
       ha='center',fontsize=10,color='#57616D')
    fig.subplots_adjust(left=.09,right=.975,top=.91,bottom=.095 if len(genes)==5 else .052,
                        wspace=.32,hspace=.52 if len(genes)==5 else .43)
    for ext in ('png','svg','pdf'):
        fig.savefig(outdir/f'{stem}.{ext}',dpi=300,bbox_inches='tight')
    plt.close(fig)


def main():
    ap=argparse.ArgumentParser(description='C220 qPCR: independent 0/10/20 groups, ALL pairwise brackets')
    ap.add_argument('--outdir',default='C220_qPCR_ALL_THREE_PAIRWISE')
    ap.add_argument('--errorbar',choices=['sem','sd'],default='sem')
    ap.add_argument('--style',choices=['dot','bar'],default='dot')
    ap.add_argument('--label-mode',choices=['q','pq'],default='q')
    args=ap.parse_args()
    output=Path(args.outdir).expanduser().resolve()
    output.mkdir(parents=True,exist_ok=True)
    samples=read_samples()
    per_sample,means,comparisons,series=analyze(samples)
    stats={(r['gene'],r['group_a'],r['group_b']):r for r in comparisons}
    save_rows(output/'01_final_Cq_inputs.csv',[
        dict(batch=b,dsw_percent=c,**samples[(b,c)])
        for b in BATCHES for c in CONDITIONS])
    save_rows(output/'02_per_sample_normalized.csv',per_sample)
    save_rows(output/'03_group_mean_SD_SEM.csv',means)
    save_rows(output/'04_all_three_pairwise_tests.csv',comparisons)
    main_five=['12473','6916','2225','4764','12325']
    for name,genes in [('MAIN_five_genes_all_pairs',main_five),
                       ('ALL_ten_genes_all_pairs',GENES),
                       ('SUPP_other_five_all_pairs',[g for g in GENES if g not in main_five])]:
        make_figure(output,name,genes,series,stats,args.errorbar,args.style,args.label_mode,
                    'Penicillium sp. C220 | 0%, 10% and 20% DSW: independent group comparisons')
    report=[
      'C220 analysis: three independent groups, not a salinity trend.',
      'Data: 9 finalized biological-sample mean Cq values supplied by user.',
      'EF1alpha/7219.1: one provisional reference per matching biological sample.',
      'log2 normalized expression = Cq(EF1alpha) - Cq(target).',
      'Plotting values are centered by subtracting the mean across ALL NINE samples for each target.',
      'No condition is imposed as the baseline; centering does not affect tests.',
      'All 3 pairs tested for each of 10 genes: 0 vs 10, 10 vs 20, 0 vs 20.',
      'Welch two-sided t-test; Holm within 3 comparisons per gene; BH across 30 tests.',
      f'Figure displays mean +/- {args.errorbar.upper()} and all 3 biological points per condition.',
      'Sample SD, SEM and arithmetic normalized quantities are retained in CSV.',
      'Fold changes are arithmetic mean normalized quantity ratios per MIQE 2.0.',
      'Assumes equal 100% PCR efficiency for target and reference; unverified.',
      'If batches represent linked experimental blocks, independence assumption may be wrong.',
      'Do not infer equivalence from p >= 0.05 or biological significance from ns.',
      'qPCR is supplementary directional validation; RNA-seq DESeq2 supplies primary DE evidence.',
      'https://academic.oup.com/clinchem/article/71/6/634/8119148',
      'https://www.graphpad.com/guides/prism/latest/statistics/stat_multiple-comparisons-without-a.htm',
    ]
    (output/'README_methods.txt').write_text('\n'.join(report)+'\n',encoding='utf-8')
    print('\n'.join(report))
    print('\nPairwise tests (all targets, global FDR corrected across 30):')
    for g in GENES:
        for a,b in PAIRWISE:
            r=stats[(g,a,b)]
            print(f"{g:>5s} {a:>2}vs{b:<2} p={r['p_raw']:.5f}  q30={r['q_BH_global_30tests']:.5f}  FC_B/A={r['fold_B_over_A']:.3f}")
    print('\nOutputs:',output)

if __name__=='__main__':
    main()
