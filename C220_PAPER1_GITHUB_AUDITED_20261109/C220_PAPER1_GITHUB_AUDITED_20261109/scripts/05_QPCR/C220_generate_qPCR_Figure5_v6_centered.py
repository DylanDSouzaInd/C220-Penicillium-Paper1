#!/usr/bin/env python3
"""C220 qPCR Figure 5 v6 (centered replicate dots + sample-matched normalization).

Self-contained, uses ONLY the nine finalized, sample-matched Cq mean values from
23 October 2026. Three biologically distinct DSW conditions (0, 10, 20%).

- Each target is normalized within its biological sample to EF1alpha (7219.1).
- qPCR normalized quantity Q = 2**(EF1alpha_Cq - target_Cq), assuming 100%
  and equal amplification efficiencies (efficiencies have not been measured).
- For visualization only, log2(Q) is centered on each gene's pooled (n=9)
  mean: no DSW condition is privileged as the plotting baseline.
- All n=3 points are centered horizontally on their treatment mean bars. Actual biological SD is retained
  in accompanying data; --show-sd adds SD whiskers, without changing data.
- Welch 2-sided independent-sample tests for all 3 pairs × 10 genes, BH across
  all 30 comparisons, and Holm within each gene in supplementary CSV only.
- RNA-seq DESeq2 contrasts (provided in C220 supplementary Table S2) are used
  in concordance table, NOT to change the qPCR statistical inference.

Usage:
  python C220_generate_qPCR_Figure5_v6_centered.py --outdir /data/dylan/C220_qPCR_FIG5_V6
  python C220_generate_qPCR_Figure5_v6_centered.py --outdir ... --show-sd
Dependencies: numpy, scipy, matplotlib. No requests to external services.
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

FIELDS = ['EF1alpha','449','2532','11494','12473','2225','6916',
          '7160','7271','7303','7881','12648','4764','12325']
# Final corrected biological sample Ct means supplied by the user Oct 23, 2026.
# Do NOT substitute an earlier iteration or exclude B3/10 because it differs.
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
GENES=['12473','2225','6916','7160','7271','7303','7881','12648','4764','12325']
MAIN=['12473','6916','2225','4764','12325']
CONDITIONS=(0,10,20)
BATCHES=('B1','B2','B3')
PAIRS=((0,10),(10,20),(0,20))
NAMES={
 '12473':'Putative cell-wall/surface protein',
 '6916':'AA11 CAZyme candidate',
 '2225':'GH43 glycoside hydrolase',
 '4764':'Laccase/MCO-like candidate',
 '12325':'SDR oxidoreductase-like',
 '7160':'Amino-acid permease',
 '7271':'Glycosyltransferase-like',
 '7303':'GH18 chitinase',
 '7881':'MFS transporter',
 '12648':'HAD-like hydrolase',
}
# RNA-seq values from provided C220 Supplementary Table S2; q = DESeq2 BH.
RNASEQ={
 '12473':{(0,10):(4.86,5.41e-4),(0,20):(6.99,6.81e-8),(10,20):(2.13,1.0)},
 '6916': {(0,10):(7.15,5.10e-4),(0,20):(9.79,2.08e-7),(10,20):(2.64,0.450)},
 '2225': {(0,10):(6.72,0.026),(0,20):(8.96,0.002),(10,20):(2.24,1.0)},
 '4764': {(0,10):(6.19,0.011),(0,20):(9.26,1.93e-5),(10,20):(3.06,0.575)},
 '12325':{(0,10):(3.93,0.044),(0,20):(6.63,7.29e-5),(10,20):(2.70,0.817)},
}

def save_csv(path, rows):
    if not rows: return
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys()))
        w.writeheader();w.writerows(rows)

def bh(p):
    p=np.asarray(p,dtype=float)
    ind=np.argsort(p)
    q=np.minimum.accumulate((p[ind]*len(p)/np.arange(1,len(p)+1))[::-1])[::-1]
    ans=np.empty_like(p);ans[ind]=np.clip(q,0,1)
    return ans

def holm(p):
    p=np.asarray(p,dtype=float)
    ind=np.argsort(p)
    a=np.maximum.accumulate((len(p)-np.arange(len(p)))*p[ind])
    ans=np.empty_like(p);ans[ind]=np.clip(a,0,1)
    return ans

def calculate():
    samples={}
    for batch,cond,*values in RAW:
        assert len(values)==len(FIELDS)
        samples[(batch,cond)]=dict(zip(FIELDS, map(float,values)))
    per_sample=[];summary=[];tests=[]; plotting={};concordance=[]
    for gene in GENES:
        logq={}; qty={}
        for cond in CONDITIONS:
            q=np.array([2.0**(samples[(b,cond)]['EF1alpha']-samples[(b,cond)][gene]) for b in BATCHES])
            qty[cond]=q;logq[cond]=np.log2(q)
            for b,q_i,logq_i in zip(BATCHES,q,logq[cond]):
                per_sample.append({'gene':gene,'batch':b,'DSW_percent':cond,
                                   'target_Cq':samples[(b,cond)][gene],
                                   'EF1alpha_Cq':samples[(b,cond)]['EF1alpha'],
                                   'delta_Cq':samples[(b,cond)][gene]-samples[(b,cond)]['EF1alpha'],
                                   'normalized_Q':q_i,'log2_Q':logq_i})
        center=float(np.mean(np.concatenate(list(logq.values()))))
        plotting[gene]={c:logq[c]-center for c in CONDITIONS}
        for c in CONDITIONS:
            summary.append({'gene':gene,'DSW_percent':c,'n':3,
                           'mean_normalized_Q':np.mean(qty[c]),
                           'SD_normalized_Q':np.std(qty[c],ddof=1),
                           'mean_log2_Q':np.mean(logq[c]),
                           'SD_log2_Q':np.std(logq[c],ddof=1),
                           'SEM_log2_Q':np.std(logq[c],ddof=1)/math.sqrt(3),
                           'fold_vs_0_arithmetic_means':np.mean(qty[c])/np.mean(qty[0])})
        for a,b in PAIRS:
            p=float(ttest_ind(logq[a],logq[b],equal_var=False,alternative='two-sided').pvalue)
            if not np.isfinite(p):raise ValueError((gene,a,b,p))
            tests.append({'gene':gene,'contrast':f'{a}v{b}','a_DSW_percent':a,'b_DSW_percent':b,
                          'Welch_two_sided_p':p,
                          'log2_difference_mean_logq':float(np.mean(logq[b])-np.mean(logq[a])),
                          'qPCR_fold_change_ratio_of_arithmetic_means':float(np.mean(qty[b])/np.mean(qty[a]))})
        if gene in MAIN:
            for a,b in PAIRS:
                qrna=RNASEQ[gene][(a,b)]
                fc=float(np.mean(qty[b])/np.mean(qty[a]))
                concordance.append({'gene':gene,'functional_label':NAMES[gene],
                    'contrast':f'{b}% versus {a}%',
                    'RNAseq_log2FC':qrna[0], 'RNAseq_DESeq2_q':qrna[1],
                    'qPCR_fold_change':fc,
                    'RNAseq_qPCR_direction_agrees':('YES' if ((qrna[0]>0)==(fc>1.0)) else 'NO'),
                    'qPCR_20_all_samples_exceed_0_all_samples':
                        ('YES' if (np.min(qty[20])>np.max(qty[0])) else 'NO') if a==0 and b==20 else ''})
    q_all=bh([r['Welch_two_sided_p'] for r in tests])
    for r,q_i in zip(tests,q_all):
        r['BH_q_all_30_qPCR_tests']=float(q_i)
    for gene in GENES:
        ind=[i for i,r in enumerate(tests) if r['gene']==gene]
        within=holm([tests[i]['Welch_two_sided_p'] for i in ind])
        for i,h in zip(ind,within): tests[i]['Holm_p_within_gene']=float(h)
    return samples,per_sample,summary,tests,plotting,concordance

def draw_panel(ax, gene, plotting, show_sd=False, font_scale=1.0):
    """Single panel: independent salinity groups, raw sample points and biological means."""
    vals = plotting[gene]
    # Match original RNA-seq figure's 0/10/20 colour semantics.
    colors = ('#4C78A8', '#E88634', '#4B9970')
    # Center the three biological points around their OWN treatment mean bar.
    # Only X coordinates are offset (deterministic, no random jitter): Y remains
    # the authentic sample's normalized log2 expression, unchanged.
    # Reuse the same B1/B2/B3 horizontal positions across every group/panel.
    jitter = (-0.115, 0.000, 0.115)
    mean_bar_halfwidth = 0.245
    for i, cond in enumerate(CONDITIONS):
        ys = np.asarray(vals[cond], dtype=float)
        mu = float(ys.mean())
        # Plot the short mean bar BEHIND the points: close-to-mean points may
        # legitimately sit on the line, but their white outlines stay visible.
        ax.hlines(mu, i - mean_bar_halfwidth, i + mean_bar_halfwidth,
                  color='#26323F', lw=2.15, zorder=3)
        for dx, y in zip(jitter, ys):
            ax.scatter(i + dx, y, s=62, color=colors[i], edgecolor='white',
                       lw=1.0, zorder=5)
        if show_sd:
            sd = float(ys.std(ddof=1))
            # SD whisker belongs to the same group; keep it to the right of
            # the mean-bar center so it does not obscure the three dots.
            ax.errorbar(i + mean_bar_halfwidth, mu, yerr=sd, fmt='none',
                        capsize=4, capthick=1, color='#333D49', lw=1,
                        zorder=4)
    ax.axhline(0, color='#677686', lw=0.7, ls=':', alpha=.65, zorder=1)
    ax.set_xticks(range(3), ['0%', '10%', '20%'])
    ax.set_xlim(-0.53, 2.53)
    # Two-line heading mimics the 12-panel RNA-seq Figure 4 style.
    ax.set_title(f'{NAMES[gene]}\nDSWSTRG.{gene}', fontsize=11.4 * font_scale,
                 fontweight='bold', pad=15, linespacing=1.12)
    ax.tick_params(axis='both', labelsize=10.4 * font_scale)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=.13, lw=.6, zorder=0)
    all_y = np.concatenate([vals[c] for c in CONDITIONS])
    lo, hi = float(min(all_y.min(), 0)), float(max(all_y.max(), 0))
    span = max(hi - lo, 1.5)
    ax.set_ylim(lo - .14 * span, hi + .18 * span)


def render_figure(out, stem, genes, plotting, show_sd=False, main=False):
    if main:
        # Three panels above, two centered below: no commentary panel / no empty plot.
        fig = plt.figure(figsize=(17.0, 10.3))
        gs = fig.add_gridspec(2, 6, left=.078, right=.985,
                             top=.835, bottom=.13, wspace=1.10, hspace=.48)
        axes = [fig.add_subplot(gs[0, 0:2]), fig.add_subplot(gs[0, 2:4]),
                fig.add_subplot(gs[0, 4:6]), fig.add_subplot(gs[1, 1:3]),
                fig.add_subplot(gs[1, 3:5])]
    else:
        fig, grid = plt.subplots(5, 2, figsize=(13.3, 20.3))
        axes = list(grid.flat)
        fig.subplots_adjust(left=.10, right=.985, top=.93, bottom=.065,
                            wspace=.25, hspace=.52)
    for idx, gene in enumerate(genes):
        ax = axes[idx]
        draw_panel(ax, gene, plotting, show_sd=show_sd)
        if (idx in (0, 3)) if main else (idx % 2 == 0):
            ax.set_ylabel('log₂(EF1α-normalized expression)\n(centered within gene)',
                          fontsize=10.5, labelpad=6)
    # The main title matches Figure 4: bold+italic genus, rest bold upright.
    fig.suptitle(r'$\mathbfit{Penicillium}$ sp. C220 | Targeted RT-qPCR validation',
                 fontsize=18.0, fontweight='bold', y=.990)
    if main:
        fig.text(.5, .933,
                 'Independent 0%, 10% and 20% DSW conditions  •  n = 3 biological samples per group',
                 ha='center', fontsize=11)
        fig.text(.5, .053,
                 'Dots = individual biological samples (horizontally separated)   |   Black bars = group means' +
                 ('   |   Whiskers = biological SD' if show_sd else ''),
                 ha='center', fontsize=10.8)
        fig.text(.5, .025,
                 'Normalized to matched EF1α (DSWSTRG.7219.1); log₂ values centered within each gene for display.',
                 ha='center', fontsize=9.5, color='#444')
    else:
        fig.text(.5, .956,
                 'Independent 0%, 10% and 20% DSW conditions  •  n = 3/group',
                 ha='center', fontsize=10.5)
        fig.text(.5, .02,
                 'Dots = biological samples centered on each group; bars = means; SD in CSV.',
                 ha='center', fontsize=9)
    for ext in ('png','svg','pdf'):
        fig.savefig(out / f'{stem}.{ext}', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--outdir',default='C220_qPCR_FOCUSED')
    parser.add_argument('--show-sd',action='store_true',help='also show true biological SD whiskers')
    args=parser.parse_args()
    out=Path(args.outdir).expanduser().resolve()
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f'ERROR: output directory is not empty: {out}\nChoose a NEW --outdir to protect previous results.')
    out.mkdir(parents=True,exist_ok=True)
    samples,per_sample,summary,tests,plotting,concordance=calculate()
    save_csv(out/'01_original_final_Cq_means.csv',[
        dict(batch=b,DSW_percent=c,**samples[(b,c)]) for b in BATCHES for c in CONDITIONS])
    save_csv(out/'02_sample_matched_EF1alpha_normalization.csv',per_sample)
    save_csv(out/'03_biological_group_mean_SD_and_SEM.csv',summary)
    save_csv(out/'04_all_30_qPCR_pairwise_tests_full_disclosure.csv',tests)
    save_csv(out/'05_RNAseq_qPCR_three_contrast_direction_agreement.csv',concordance)
    render_figure(out,'Figure5_RTqPCR_five_targets_FINAL',MAIN,plotting,args.show_sd,main=True)
    render_figure(out,'Supplement_RTqPCR_all_ten_targets',GENES,plotting,args.show_sd,main=False)
    notes='''C220 RT-qPCR focused analysis (23 October 2026)\n\n'''+'''Main figure: five targets selected because their RNA-seq differential expression
(20% vs 0%, q<0.05) and annotation support relate to hyphal boundary remodeling.
They were selected from the manuscript's existing candidate panel, not by new qPCR p-values.
All 3 salinity groups and all 3 biological observations per group shown. No trend model.
Q_i = 2**(EF1alpha_Cq_i - target_Cq_i): assumes equal 100% PCR efficiencies.
For each gene, plot log2(Q_i) minus pooled 9-sample mean log2(Q); this affects
neither relative differences nor inference and does not privilege 0% DSW.
A short line behind the centered biological dots marks the mean log2 normalized quantity; --show-sd draws true SD.
Statistical tests remain transparently available: Welch two-sided independent
group comparison of log2(Q), 3 contrasts per 10 genes, BH over 30, Holm within
gene. No qPCR comparison meets q<0.05 over 30 tests. No asterisks in figure.
Original RNA-seq DESeq2 FDR values in concordance CSV are from user-provided
Supplementary Table S2 and must NOT be treated as qPCR significance.
B1/B2/B3 assumed independent within condition; if truly paired experimental
blocks, a block-aware model requires documentation and re-analysis.
The stated biological replicate identities and final Cq means are preserved.
The five-gene directional consistency is descriptive evidence rather than
an independent qPCR significance claim. The EF1alpha reference remains
provisional, with no assay-specific efficiencies measured.
MIQE 2.0: https://academic.oup.com/clinchem/article/71/6/634/8119148
'''
    (out/'README_methods_and_limitations.txt').write_text(notes,encoding='utf-8')
    for gene in MAIN:
        hit=next(r for r in concordance if r['gene']==gene and r['contrast']=='20% versus 0%')
        row=next(r for r in tests if r['gene']==gene and r['contrast']=='0v20')
        print(f"{gene}: 20%/0% qPCR FC={hit['qPCR_fold_change']:.2f}; "
              f"RNAseq log2FC={hit['RNAseq_log2FC']:.2f}, "
              f"RNAseq q={hit['RNAseq_DESeq2_q']:.3g}; "
              f"qPCR Welch p={row['Welch_two_sided_p']:.3g}, q30={row['BH_q_all_30_qPCR_tests']:.3g}")
    print('Output directory:',out)
    print('Five primary gene IDs:', ', '.join(MAIN))

if __name__=='__main__':main()
