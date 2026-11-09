#!/usr/bin/env python3
"""Rebuild C220 Paper 1 RNA-seq Fig 4 from genuine DESeq2 normalized counts.
Safety: never copy pixels; never estimate counts; never overwrite source files.
Accepts a wide DESeq2-normalized gene x sample table, or explicit DESeqDataSet RDS.
"""
import argparse, csv, gzip, json, os, re, subprocess, sys, hashlib
from pathlib import Path
from datetime import datetime

GENES = [
 ('DSWSTRG.12473','Putative cell-wall/surface','Surface / extracellular'),
 ('DSWSTRG.6916','AA11 CAZyme candidate','CAZyme / carbohydrate'),
 ('DSWSTRG.2225','GH43 glycoside hydrolase','CAZyme / carbohydrate'),
 ('DSWSTRG.7271','Glycosyltransferase-like','Surface / glycan'),
 ('DSWSTRG.7303','GH18 chitinase-like','Cell-wall'),
 ('DSWSTRG.7881','MFS transporter','Transport'),
 ('DSWSTRG.7160','Amino-acid permease','Transport'),
 ('DSWSTRG.4764','Laccase/MCO-like','Oxidative / surface'),
 ('DSWSTRG.12325','SDR oxidoreductase-like','Metabolic / redox'),
 ('DSWSTRG.12648','HAD-like hydrolase','Metabolic'),
 ('DSWSTRG.9665','O-methyltransferase-like','Metabolic'),
 ('DSWSTRG.2532','RNA-seq low-variability candidate','Reference candidate'),
]
IDS=[x[0] for x in GENES]
CONDITIONS=(0,10,20)
ROOT=Path('/data/dylan/01_ONT_PCRcDNARNASeq/salinity_rnaseq_analysis')
SKIP_DIR={'.git','conda_envs','node_modules','__pycache__','fastq','bam','bams','alignment','alignments','reads','raw_reads','fastqs','figures','plots','images','genome','genomes'}
EXT=('.tsv','.txt','.csv','.tsv.gz','.csv.gz','.txt.gz')


def source_files(roots, patterns, max_n=200000):
    seen=set(); out=[]
    for root in roots:
        if not root.exists(): continue
        for dirpath, subdirs, files in os.walk(root, followlinks=False):
            subdirs[:]=[d for d in subdirs if d not in SKIP_DIR and not d.startswith('.')]
            for filename in files:
                if len(out)>=max_n: raise RuntimeError('Large source scan (>200,000 files); supply --matrix explicitly')
                low=filename.lower()
                if any(low.endswith(ext) for ext in patterns):
                    p=Path(dirpath)/filename
                    try:
                        key=(p.stat().st_dev,p.stat().st_ino)
                        if key not in seen: seen.add(key);out.append(p)
                    except OSError: continue
    return out


def sample_slot(name):
    """Conservative sample naming. Never guess from bare numeric columns."""
    s=str(name).lower().strip().replace(' ','').replace('%','pct')
    s=re.sub(r'\.(bam|fastq|fq)(\.gz)?$', '',s)
    # Explicit barcode mapping applies solely to original ONT barcode01...barcode09.
    m=re.search(r'(?:barcode|bc|barc)[_\-]?0?([1-9])(?=$|\D)',s)
    if m:
        n=int(m.group(1)); return ((n-1)//3*10, (n-1)%3+1)
    patterns=[
      r'(?:dsw|salinity|salt)[_\-]?(0|10|20)(?:pct)?[_\-]?(?:dsw)?[_\-]?(?:rep|replicate|bio|b|r)[_\-]?([1-4])(?:$|\D)',
      r'(?:rep|replicate|bio|b|r)[_\-]?([1-4])[_\-]?(?:dsw|salinity)[_\-]?(0|10|20)(?:pct)?(?:$|\D)',
      r'(?:^|[_\-])(0|10|20)pct[_\-]?(?:rep|replicate|bio|b|r)[_\-]?([1-4])(?:$|\D)',
      r'(?:^|[_\-])(0|10|20)[_\-](?:rep|replicate|bio|b|r)[_\-]?([1-4])(?:$|\D)',
    ]
    for idx,p in enumerate(patterns):
        m=re.search(p,s)
        if m:
            a,b=int(m.group(1)),int(m.group(2))
            return (b,a) if idx==1 else (a,b)
    return None


def read_delimited(path, **kwargs):
    import pandas as pd
    name=path.name.lower()
    sep='\t' if any(name.endswith(e) for e in ('.tsv','.tsv.gz','.txt','.txt.gz')) else ','
    return pd.read_csv(path,sep=sep,low_memory=False,**kwargs)


def gene_key(v):
    m=re.fullmatch(r'(DSWSTRG\.\d+)',str(v).strip())
    return m.group(1) if m else None


def inspect_matrix(path, manual_map=None):
    import pandas as pd
    try:
        peek=read_delimited(path,nrows=3)
        cols=list(peek.columns)
        gene_cols=[x for x in cols if re.search(r'(^gene(_?id)?$|^geneid$|^id$|^unnamed: 0$|^locus$|^feature$)',str(x),re.I)]
        gene_col=gene_cols[0] if gene_cols else cols[0]
        slots={}; dup=[]
        for col in cols:
            if col==gene_col: continue
            slot=(manual_map or {}).get(str(col)) or sample_slot(col)
            if slot and slot[0] in CONDITIONS and 1<=slot[1]<=3:
                if slot in slots.values():dup.append(str(col))
                slots[str(col)]=slot
        low=str(path).lower()
        name_score=(16 if 'normalized' in low or 'normalised' in low else 0)+(8 if 'deseq' in low else 0)+(5 if 'gene' in low else 0)+(2 if 'final' in low else 0)
        bad= any(x in low for x in ('raw_count','raw.count','featurecount','vst','rlog','tpm','fpkm','cpm','transcript','isoform'))
        # Only score candidates with nine clearly resolved unique sample columns.
        return {'path':str(path),'gene_col':str(gene_col),'columns':cols,'mapped':{k:list(v) for k,v in slots.items()},'duplicate_slots':dup,'nine_samples':len(set(slots.values()))==9 and not dup,'score':name_score+len(slots)*4-(100 if bad else 0),'blocked_reason':'Raw/VST/TPM/transcript-level filename; not automatically accepted' if bad else None}
    except Exception as ex:
        return {'path':str(path),'error':str(ex),'score':-999,'nine_samples':False}


def validate_and_extract(path, manual_map=None, allow_ambiguous=False):
    import numpy as np, pandas as pd
    info=inspect_matrix(path,manual_map)
    if 'error' in info: raise ValueError(f'Cannot read source: {info["error"]}')
    if not info['nine_samples']:
        raise ValueError('Cannot identify exactly 9 sample columns (3 each at 0/10/20% DSW). Write sample_map.csv and use --sample-map; see source_audit.json.')
    if info['blocked_reason'] and not allow_ambiguous:
        raise ValueError(f'Source name suggests a non-normalized or transformed quantity: {info["blocked_reason"]}. Refusing to relabel as DESeq2-normalized count.')
    df=read_delimited(path)
    col=info['gene_col']
    if col not in df.columns: raise ValueError('Missing gene ID column')
    df['__gene']=df[col].map(gene_key)
    raw_ids=set(df['__gene'].dropna())
    missing=sorted(set(IDS)-raw_ids)
    if missing: raise ValueError('Gene IDs absent from selected matrix: '+', '.join(missing))
    mapping={str(k):tuple(v) for k,v in info['mapped'].items()}
    usecols=list(mapping)
    selected=df.loc[df['__gene'].isin(IDS),['__gene']+usecols].copy()
    if selected['__gene'].duplicated().any():raise ValueError('Duplicate gene IDs detected; refusing to combine gene and isoform counts')
    vals=selected[usecols].apply(pd.to_numeric,errors='coerce')
    if vals.isna().any().any():raise ValueError('Missing or nonnumeric values in selected genes')
    if (~np.isfinite(vals.to_numpy())).any() or (vals.to_numpy()<0).any():raise ValueError('Values must be finite and non-negative')
    if np.all(np.mod(vals.to_numpy(),1)==0):
        print('WARNING: all 108 selected values are integers: confirm this is a DESeq2-normalized table, NOT raw featureCounts counts.',file=sys.stderr)
    for g in IDS:
        if (vals.loc[selected['__gene']==g].to_numpy()==0).all(): print('WARNING: all zeros for '+g,file=sys.stderr)
    selected[usecols]=vals
    long=selected.melt(id_vars='__gene',value_vars=usecols,var_name='sample_column',value_name='normalized_count')
    long['salinity']=long['sample_column'].map(lambda c:mapping[c][0]);long['replicate']=long['sample_column'].map(lambda c:mapping[c][1]);long=long.rename(columns={'__gene':'gene'})
    return long,info


def map_from_csv(path):
    if not path:return {}
    out={}
    with open(path,newline='') as f:
        for r in csv.DictReader(f):
            c=r.get('sample_column','').strip()
            if c and r.get('salinity','').strip() and r.get('replicate','').strip():
                k=int(r['salinity'].strip().replace('%','')); n=int(r['replicate'])
                if k not in CONDITIONS or n not in (1,2,3):raise ValueError(f'Invalid sample map row {r}')
                out[c]=(k,n)
    return out


def save_map_template(info,outdir):
    path=outdir/'SAMPLE_MAP_EDIT_IF_NEEDED.csv'
    with open(path,'w',newline='') as f:
        w=csv.writer(f); w.writerow(['sample_column','salinity','replicate'])
        for col in info.get('columns',[]):
            if col==info.get('gene_col'):continue
            slot=info.get('mapped',{}).get(col,[])
            w.writerow([col,*slot] if slot else [col,'',''])
    return path


def get_stats(statspaths):
    """Only use DESeq2 original adjusted p-values with known comparison direction."""
    import pandas as pd
    stats={}
    for comp,path in statspaths.items():
        if not path or not Path(path).exists():continue
        d=read_delimited(Path(path))
        fields=list(d.columns)
        gene_col=next((x for x in fields if re.search(r'^(gene(_id)?|geneid|id|unnamed: 0|locus)$',str(x),re.I)),fields[0]);
        pcol=next((x for x in fields if str(x).lower() in ('padj','adjusted_pvalue','adjusted_p_value','fdr','qvalue','q_value')),None)
        if not pcol:raise ValueError(f'No DESeq2 adjusted p-value column in {path}; cannot annotate.')
        d['__gene']=d[gene_col].map(gene_key)
        d[pcol]=pd.to_numeric(d[pcol],errors='coerce')
        if d['__gene'].duplicated().any():d=d.dropna(subset=['__gene']).drop_duplicates('__gene',keep=False)
        stats[comp]={r['__gene']:float(r[pcol]) for _,r in d.iterrows() if pd.notna(r['__gene']) and pd.notna(r[pcol]) and 0<=r[pcol]<=1}
    return stats


def stat_candidates(roots):
    paths=source_files(roots, EXT)
    ans={}
    for comp, alternatives in {'0vs10':['10_vs_0','10vs0','0_vs_10','0vs10','dsw10_dsw0'],
                                '0vs20':['20_vs_0','20vs0','0_vs_20','0vs20','dsw20_dsw0'],
                                '10vs20':['20_vs_10','20vs10','10_vs_20','10vs20','dsw20_dsw10']}.items():
        ranked=[]
        for p in paths:
            s=p.name.lower().replace('-','_');
            if any(x in s for x in alternatives) and ('deseq' in s or 'deg' in s or 'result' in s or 'diff' in s):
                score=(10 if 'deseq2' in s else 0)+(8 if 'all' in s else 0)+(5 if 'final' in s else 0)
                ranked.append((score,str(p)))
        if ranked:ans[comp]=sorted(ranked,reverse=True)[0][1]
    return ans


def make_plot(long,stats,outdir,source_label):
    # Plot text size only. No alterations to scientific data or statistics.
    ylab_pt = float(os.environ.get('C220_FIG4_YLABEL_PT', '14'))
    if not 8 <= ylab_pt <= 22:
        raise ValueError('C220_FIG4_YLABEL_PT must be between 8 and 22')
    import numpy as np, pandas as pd, matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    colors={0:'#4e79a7',10:'#e28b38',20:'#5a956d'}
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,
                         'axes.labelsize':12,'xtick.labelsize':11.5,
                         'ytick.labelsize':11.5,'pdf.fonttype':42,'svg.fonttype':'none'})
    fig,axes=plt.subplots(3,4,figsize=(19.5,15.0),layout='constrained')
    fig.get_layout_engine().set(wspace=0.10, hspace=0.12, w_pad=0.10, h_pad=0.10)
    rng_jitter={1:-0.10,2:0.0,3:0.10}
    audit=[]
    for idx,(gene,title,module) in enumerate(GENES):
        ax=axes.flat[idx];g=long[long.gene==gene]
        mx=-np.inf;mn=np.inf
        for c in CONDITIONS:
            rows=g.loc[g.salinity==c].sort_values('replicate')
            y=np.log10(rows['normalized_count'].to_numpy(dtype=float)+1)
            x=CONDITIONS.index(c)
            ax.scatter([x+rng_jitter[int(r)] for r in rows.replicate],y,s=51,color=colors[c],alpha=.93,edgecolor='white',linewidth=.65,zorder=4)
            ax.hlines(float(np.mean(y)),x-.22,x+.22,lw=2.0,color='#202934',zorder=5)
            audit.append({'gene':gene,'condition':c,'n':len(rows),'mean_log10_norm_plus1':float(y.mean()),'sd_log10_norm_plus1':float(y.std(ddof=1))})
            mx=max(mx,float(np.max(y)));mn=min(mn,float(np.min(y)))
        ax.set_title(f'{title}\n{gene}',fontsize=12.4,pad=18,weight='bold',linespacing=1.16)
        ax.set_xticks([0,1,2],['0%','10%','20%']); ax.set_xlim(-.48,2.48)
        ax.tick_params(axis='both',which='major',labelsize=11.7,pad=5)
        ax.set_ylabel('log$_{10}$(normalized count + 1)' if idx%4==0 else '',fontsize=ylab_pt,labelpad=10)
        ax.spines[['top','right']].set_visible(False)
        ax.grid(axis='y',linewidth=.45,alpha=.22);ax.set_axisbelow(True)
        span=max(.8,mx-mn)
        ylimtop=mx+max(.52,.31*span)
        # Negative DISPLAY padding only. log10(normalized_count + 1) is never negative.
        # This prevents true 0% RNA-seq points at y=0 from touching/clipping on the spine.
        ybottom=(-0.30 if mn <= 0.12 else mn-max(.24,.16*span))
        ax.set_ylim(ybottom, ylimtop)
        valid_bars=[]
        for key,a,b in [('0vs10',0,1),('0vs20',0,2),('10vs20',1,2)]:
            q=stats.get(key,{}).get(gene)
            if q is not None and q<0.05:
                valid_bars.append((a,b,q))
        for k,(a,b,q) in enumerate(valid_bars):
            yy=mx+max(.22,.13*span)+k*max(.18,.12*span)
            ax.plot([a,a,b,b],[yy-.05,yy,yy,yy-.05],lw=.9,color='#24364d')
            ax.text((a+b)/2,yy+max(.02,.014*span),'****' if q<1e-4 else '***' if q<.001 else '**' if q<.01 else '*',ha='center',va='bottom',fontsize=12.5,fontweight='bold')
            ax.set_ylim(ax.get_ylim()[0],max(ax.get_ylim()[1],yy+max(.31,.21*span)))
        # Keep y=0 tick, but do not imply that negative log-expression data exist.
        if ybottom < 0:
            from matplotlib.ticker import MaxNLocator
            ax.yaxis.set_major_locator(MaxNLocator(nbins=6, min_n_ticks=4))
            import matplotlib.ticker as mticker
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(
                lambda v, pos: '' if v < -1e-9 else f'{v:g}'
            ))
    handles=[Line2D([0],[0],marker='o',color='none',markerfacecolor=colors[c],markeredgecolor='white',markersize=8,label=f'{c}% DSW') for c in CONDITIONS]
    fig.legend(handles=handles,loc='outside lower center',ncol=3,frameon=False,fontsize=12.0)
    fig.suptitle(r'$\mathbfit{Penicillium}$ sp. C220 | RNA-seq salinity-associated candidates',fontsize=18.0,weight='bold')
    fig.supylabel('Original DESeq2-normalized counts (log$_{10}$ scale)',fontsize=ylab_pt)
    for ext in ('png','pdf','svg'):
        fig.savefig(outdir/f'Figure4_RNAseq_11targets_plus_2532_FROM_DATA.{ext}',dpi=350,bbox_inches='tight')
    plt.close(fig)
    pd.DataFrame(audit).to_csv(outdir/'Figure4_data_group_summary.csv',index=False)


def rds_to_csv(rds,outdir):
    """Optional explicit DESeqDataSet only, to avoid arbitrary matrix normalization."""
    rds=Path(rds)
    script=outdir/'export_deseq2_normalized.R'
    script.write_text('''args <- commandArgs(trailingOnly=TRUE)\n'
 'suppressPackageStartupMessages(library(DESeq2))\n'
 'obj <- readRDS(args[1])\n'
 'if (!inherits(obj,"DESeqDataSet")) stop("RDS is not a DESeqDataSet")\n'
 'x <- counts(obj, normalized=TRUE)\n'
 'write.csv(data.frame(gene=rownames(x),as.data.frame(x),check.names=FALSE),args[2],row.names=FALSE,quote=TRUE)\n'''.replace("'\n '",''))
    dest=outdir/'EXPORTED_FROM_DESEQ2_RDS_normalized_counts.csv'
    subprocess.run(['Rscript',str(script),str(rds),str(dest)],check=True)
    return dest



def audit_and_rebuild():
    """Replot ONLY verified original DESeq2 counts + original adjusted P values."""
    import pandas as pd
    import numpy as np
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--previous-run',default='/data/dylan/C220_PAPER1_AUTHENTIC_FIGURE4_20261024_143504')
    ap.add_argument('--root',default=str(ROOT))
    ap.add_argument('--outdir',default=None)
    args=ap.parse_args()
    prev=Path(args.previous_run).resolve()
    old_audit=prev/'SOURCE_AUDIT.json'
    if not old_audit.is_file():raise SystemExit('Missing previous SOURCE_AUDIT.json: '+str(old_audit))
    old=json.loads(old_audit.read_text())
    matrix=Path(old['rna_seq_matrix']).resolve()
    if not matrix.is_file():raise SystemExit('Missing original source matrix: '+str(matrix))
    h=hashlib.sha256(matrix.read_bytes()).hexdigest()
    if h != old.get('matrix_sha256') or h != 'ecb41cb27e5888d8a6357aee2fd1f6f54d45c6e883f5e05f40de9f109f879fb6':
        raise SystemExit('STOP: SHA256 mismatch; original matrix was changed or not the verified source: '+h)
    out=Path(args.outdir).resolve() if args.outdir else Path('/data/dylan/C220_PAPER1_FIG4_DESEQ2_VERIFIED_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
    if out.exists():raise SystemExit('STOP: output directory exists; refusing to overwrite: '+str(out))
    root=Path(args.root).resolve()
    files={
     '0vs10': root/'06_deseq2_v2/results_all_samples/DESeq2_DSW10_vs_DSW0_all.csv',
     '0vs20': root/'06_deseq2_v2/results_all_samples/DESeq2_DSW20_vs_DSW0_all.csv',
     '10vs20':root/'06_deseq2_v2/results_all_samples/DESeq2_DSW20_vs_DSW10_all.csv',
    }
    for k,p in files.items():
        if not p.is_file():raise SystemExit('Missing original DESeq2 results '+k+': '+str(p))
    long,info=validate_and_extract(matrix)
    mapped={str(k):list(v) for k,v in info['mapped'].items()}
    expected_slots={(d,r) for d in (0,10,20) for r in (1,2,3)}
    if len(mapped)!=9 or {tuple(x) for x in mapped.values()} != expected_slots:
        raise SystemExit('Incorrect sample map; refusing figure reconstruction')
    if mapped != old.get('sample_columns'):
        raise SystemExit('STOP: sample mapping differs from previous source audit')
    if any(int(x)!=3 for x in long.groupby(['gene','salinity']).size()):
        raise SystemExit('STOP: expected 3 biological samples per gene and salinity')
    stats=get_stats({k:str(p) for k,p in files.items()})
    if set(stats)!={'0vs10','0vs20','10vs20'}:raise SystemExit('Not all 3 DESeq2 contrasts were loaded')
    # These values originate from the previously uploaded Supplementary Table S2,
    # and are ONLY verification checkpoints; statistics are read from the server files.
    checkpoint={
     '0vs10':{'DSWSTRG.12473':5.41e-4,'DSWSTRG.6916':5.10e-4,'DSWSTRG.2225':.026,
              'DSWSTRG.4764':.011,'DSWSTRG.12325':.044,'DSWSTRG.7303':4.11e-7},
     '0vs20':{'DSWSTRG.12473':6.81e-8,'DSWSTRG.6916':2.08e-7,'DSWSTRG.2225':.002,
              'DSWSTRG.4764':1.93e-5,'DSWSTRG.12325':7.29e-5,'DSWSTRG.7303':.650},
     '10vs20':{'DSWSTRG.12473':1.0,'DSWSTRG.6916':.450,'DSWSTRG.2225':1.0,
               'DSWSTRG.4764':.575,'DSWSTRG.12325':.817,'DSWSTRG.7303':.003},
    }
    for c,genes in checkpoint.items():
        if len(set(IDS)&set(stats[c])) != 12:
            raise SystemExit(f'STOP: {c} only has {len(set(IDS)&set(stats[c]))} of 12 figure gene adjusted p values')
        for gene,expect in genes.items():
            found=stats[c].get(gene)
            if found is None or not np.isfinite(found) or abs(found-expect)>max(.15*expect,.005 if expect>.02 else 1e-8):
                raise SystemExit(f'STOP: {c} {gene} padj={found} differs from supplementary {expect}')
    # Check log2FC orientation. All three selected paths use numerator DSW10 or DSW20.
    # Sign checkpoints explicitly prevent using reversed DESeq2 contrasts.
    signed={
      '0vs10':{'DSWSTRG.12473':1,'DSWSTRG.6916':1,'DSWSTRG.7303':-1},
      '0vs20':{'DSWSTRG.12473':1,'DSWSTRG.6916':1,'DSWSTRG.2225':1,'DSWSTRG.4764':1,'DSWSTRG.12325':1},
      '10vs20':{'DSWSTRG.7303':1},
    }
    for c,p in files.items():
        d=read_delimited(p)
        cols=list(d.columns)
        gene_col=next((x for x in cols if re.search(r'^(gene(_id)?|geneid|id|unnamed: 0|locus)$',str(x),re.I)),cols[0])
        lcol=next((x for x in cols if str(x).lower() in ('log2foldchange','log2fc','log2_fold_change')),None)
        if lcol is None:raise SystemExit('STOP: Cannot verify log2FC orientation in '+str(p))
        d['__gene']=d[gene_col].map(gene_key)
        d[lcol]=pd.to_numeric(d[lcol],errors='coerce')
        for gene,sign in signed[c].items():
            v=d.loc[d['__gene']==gene,lcol]
            if len(v)!=1 or pd.isna(v.iloc[0]) or v.iloc[0]*sign<=0:
                raise SystemExit(f'STOP: orientation/sign mismatch {c} {gene}: '+str(v.tolist()))
    out.mkdir(parents=True)
    long.to_csv(out/'Figure4_EXACT_original_normalized_values.csv',index=False)
    make_plot(long,stats,out,str(matrix))
    rows=[]
    for g in IDS:
        vals=[stats[c].get(g,float('nan')) for c in ('0vs10','0vs20','10vs20')]
        rows.append({'gene':g,'padj_DSW10_vs_DSW0':vals[0],
                     'padj_DSW20_vs_DSW0':vals[1], 'padj_DSW20_vs_DSW10':vals[2],
                     'n_significant_FDR05':sum(float(x)<.05 for x in vals if pd.notna(x))})
    pd.DataFrame(rows).to_csv(out/'Figure4_original_DESeq2_q_values.csv',index=False)
    qa={
     'created':datetime.now().isoformat(),'parent_run':str(prev),
     'rna_seq_matrix':str(matrix),'rna_seq_matrix_sha256':h,
     'mapped_9_samples':mapped,
     'reconstructed_gene_ids':IDS,
     'excluded':['DSWSTRG.449','DSWSTRG.11494'],
     'statistic_sources':{k:{'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),
                            'direction':'DSW10 versus DSW0' if k=='0vs10' else 'DSW20 versus DSW0' if k=='0vs20' else 'DSW20 versus DSW10'} for k,p in files.items()},
     'summary':'Original DESeq2 padj values used for significance brackets. No re-fitting or extra tests.',
     'checkpoints':'6/6 manuscript adjusted-p checkpoints matched for each contrast',
     'plot_axis':'log10(original DESeq2-normalized count + 1)',
    }
    (out/'SOURCE_AUDIT_VERIFIED.json').write_text(json.dumps(qa,indent=2)+'\n')
    print('SUCCESS:',out/'Figure4_RNAseq_11targets_plus_2532_FROM_DATA.png')
    print('Original DESeq2 contrasts verified: 0vs10, 0vs20, 10vs20')
    print('All 18 q-value checkpoints passed; 9 biological RNA-seq samples and matrix SHA256 verified')
    print('q value table:',out/'Figure4_original_DESeq2_q_values.csv')
    print('Source audit:',out/'SOURCE_AUDIT_VERIFIED.json')

if __name__=='__main__':
    audit_and_rebuild()
