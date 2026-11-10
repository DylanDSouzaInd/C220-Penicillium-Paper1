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
    import numpy as np, pandas as pd, matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    colors={0:'#4e79a7',10:'#e28b38',20:'#5a956d'}
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42,'svg.fonttype':'none'})
    fig,axes=plt.subplots(3,4,figsize=(16.5,12.4),layout='constrained')
    rng_jitter={1:-0.10,2:0.0,3:0.10}
    audit=[]
    for idx,(gene,title,module) in enumerate(GENES):
        ax=axes.flat[idx];g=long[long.gene==gene]
        mx=-np.inf;mn=np.inf
        for c in CONDITIONS:
            rows=g.loc[g.salinity==c].sort_values('replicate')
            y=np.log10(rows['normalized_count'].to_numpy(dtype=float)+1)
            x=CONDITIONS.index(c)
            ax.scatter([x+rng_jitter[int(r)] for r in rows.replicate],y,s=35,color=colors[c],alpha=.92,edgecolor='white',linewidth=.55,zorder=4)
            ax.hlines(float(np.mean(y)),x-.21,x+.21,lw=1.8,color='#202934',zorder=5)
            audit.append({'gene':gene,'condition':c,'n':len(rows),'mean_log10_norm_plus1':float(y.mean()),'sd_log10_norm_plus1':float(y.std(ddof=1))})
            mx=max(mx,float(np.max(y)));mn=min(mn,float(np.min(y)))
        ax.set_title(f'{title}\n{gene}',fontsize=9.3,pad=14,weight='bold')
        ax.set_xticks([0,1,2],['0%','10%','20%']); ax.set_xlim(-.48,2.48)
        ax.set_ylabel('log$_{10}$(normalized count + 1)' if idx%4==0 else '')
        ax.spines[['top','right']].set_visible(False)
        ax.grid(axis='y',linewidth=.45,alpha=.22);ax.set_axisbelow(True)
        span=max(.8,mx-mn)
        ylimtop=mx+max(.37,.27*span)
        ax.set_ylim(max(0,mn-max(.20,.16*span)) if mn>=0 else mn-.2,ylimtop)
        valid_bars=[]
        for key,a,b in [('0vs10',0,1),('0vs20',0,2),('10vs20',1,2)]:
            q=stats.get(key,{}).get(gene)
            if q is not None and q<0.05:
                valid_bars.append((a,b,q))
        for k,(a,b,q) in enumerate(valid_bars):
            yy=mx+max(.17,.11*span)+k*max(.12,.10*span)
            ax.plot([a,a,b,b],[yy-.05,yy,yy,yy-.05],lw=.9,color='#24364d')
            ax.text((a+b)/2,yy+max(.02,.014*span),'****' if q<1e-4 else '***' if q<.001 else '**' if q<.01 else '*',ha='center',va='bottom',fontsize=10,fontweight='bold')
            ax.set_ylim(ax.get_ylim()[0],max(ax.get_ylim()[1],yy+max(.24,.20*span)))
    handles=[Line2D([0],[0],marker='o',color='none',markerfacecolor=colors[c],markeredgecolor='white',markersize=8,label=f'{c}% DSW') for c in CONDITIONS]
    fig.legend(handles=handles,loc='outside lower center',ncol=3,frameon=False)
    fig.suptitle('Penicillium sp. C220 | RNA-seq salinity-associated candidates',fontsize=15,weight='bold')
    fig.supylabel('Original DESeq2-normalized counts (log$_{10}$ scale)',fontsize=10)
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


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',default=str(ROOT),help='C220 salinity_rnaseq_analysis root')
    ap.add_argument('--matrix',default=None,help='Explicit original DESeq2-normalized count CSV/TSV')
    ap.add_argument('--dds-rds',default=None,help='Explicit DESeqDataSet RDS; requires Rscript & DESeq2')
    ap.add_argument('--sample-map',default=None,help='Optional CSV with sample_column,salinity,replicate')
    ap.add_argument('--stat-0vs10',default=None);ap.add_argument('--stat-0vs20',default=None);ap.add_argument('--stat-10vs20',default=None)
    ap.add_argument('--outdir',default='/data/dylan/C220_RNASEQ_FIG4_AUTHENTIC_REBUILD')
    ap.add_argument('--scan-only',action='store_true')
    args=ap.parse_args()
    root=Path(args.root).resolve();outdir=Path(args.outdir).resolve();outdir.mkdir(parents=True,exist_ok=True)
    if root==outdir or root in outdir.parents: pass # may be inside root: excluded below
    scan_roots=[root/'10_final_thesis_paper_outputs',root]
    # Record existing figure versions for provenance; their pixels are never inputs.
    figbase=root/'10_final_thesis_paper_outputs'/'39_FINAL_qPCR_TARGET_EXPRESSION_PANEL'/'figures'
    image_files=[]
    if figbase.is_dir():
        for f in figbase.iterdir():
            if f.is_file() and f.suffix.lower() in ('.svg','.pdf','.png'):
                image_files.append((f.stat().st_mtime,str(f)))
    (outdir/'EXISTING_FIGURE_VERSIONS_BY_MTIME.txt').write_text('\n'.join(f'{datetime.fromtimestamp(t).isoformat()}  {f}' for t,f in sorted(image_files,reverse=True))+'\n')
    candidates=[]
    for p in source_files(scan_roots,EXT):
        if outdir==p.parent or outdir in p.parents: continue
        if p.stat().st_size>300*1024*1024:continue
        desc=inspect_matrix(p)
        if desc.get('nine_samples') or any(tag in p.name.lower() for tag in ('normalized','deseq2','vst')):
            candidates.append(desc)
    candidates.sort(key=lambda x:x.get('score',-999),reverse=True)
    (outdir/'SOURCE_DISCOVERY_RANKING.json').write_text(json.dumps(candidates[:150],indent=2))
    (outdir/'SOURCE_DISCOVERY_SHORTLIST.txt').write_text('\n'.join(f'{c.get("score",0):5d}  {c["path"]} | 9 cols={c.get("nine_samples")} | {c.get("blocked_reason") or "candidate"}' for c in candidates[:60])+'\n')
    # Trace original plotting scripts without executing them.
    provenance=[]
    for pp in source_files(scan_roots,('.py','.r','.R','.sh')):
        try:
            if pp.stat().st_size>2_000_000:continue
            contents=pp.read_text(errors='replace')
            if ('Figure_qPCR_validation_targets' in contents or '39_FINAL_qPCR_TARGET_EXPRESSION_PANEL' in contents):
                hits=[(i+1,line[:220]) for i,line in enumerate(contents.splitlines()) if any(t in line.lower() for t in ('normalized','deseq','csv','tsv','figure_qpcr_validation_targets','source'))][:45]
                provenance.append({'script':str(pp),'source_mentions':hits})
        except (OSError,UnicodeError):continue
    (outdir/'ORIGINAL_PLOT_SCRIPT_PROVENANCE.json').write_text(json.dumps(provenance,indent=2))
    if args.scan_only:
        print((outdir/'SOURCE_DISCOVERY_SHORTLIST.txt').read_text());print('Scan only: no figure produced.');return
    if args.matrix and args.dds_rds:raise SystemExit('Supply --matrix OR --dds-rds, not both')
    if args.dds_rds: matrix=rds_to_csv(args.dds_rds,outdir)
    elif args.matrix: matrix=Path(args.matrix).resolve()
    else:
        valid=[c for c in candidates if c.get('nine_samples') and not c.get('blocked_reason') and ('normalized' in c['path'].lower() or 'normalised' in c['path'].lower() or 'deseq' in c['path'].lower())]
        if not valid:
            print('No verified normalized count matrix automatically found. See:',outdir/'SOURCE_DISCOVERY_SHORTLIST.txt')
            raise SystemExit('Supply --matrix /actual/DESeq2_normalized_counts.tsv or --dds-rds /actual/DESeqDataSet.rds; DO NOT plot from raw counts.')
        # Do not choose an ambiguous source if leading candidates tie.
        top=valid[0]
        if len(valid)>1 and valid[1]['score']==top['score']:
            print('Multiple equally plausible source tables; explicit --matrix required.')
            for v in valid[:8]:print(v['score'],v['path'])
            raise SystemExit(2)
        matrix=Path(top['path'])
    map_data=map_from_csv(args.sample_map)
    try:
        long,info=validate_and_extract(matrix,map_data,allow_ambiguous=bool(args.dds_rds))
    except ValueError as exc:
        desc=inspect_matrix(matrix,map_data)
        temp=save_map_template(desc,outdir)
        print('SOURCE VALIDATION FAILED:',exc,'\nSample mapping template:',temp)
        raise SystemExit(3)
    # Additional strict checks: exactly three RNA-seq biological replicates per condition and gene.
    if any(x!=3 for x in long.groupby(['gene','salinity']).size()):raise SystemExit('Incorrect replicate count; refusing plotting')
    long.to_csv(outdir/'Figure4_EXACT_original_normalized_values.csv',index=False)
    found=stat_candidates(scan_roots)
    explicit={'0vs10':args.stat_0vs10,'0vs20':args.stat_0vs20,'10vs20':args.stat_10vs20}
    paths={k:explicit[k] or found.get(k) for k in explicit}
    try:
        stats=get_stats(paths)
    except (ValueError,KeyError) as exc:
        if any(explicit.values()):
            raise  # Explicit statistical inputs must be correct.
        print('WARNING: automatic DESeq2 stats candidate could not be verified:',exc,file=sys.stderr)
        print('No significance brackets will be rendered. Supply explicit --stat-... files when confirmed.',file=sys.stderr)
        stats={}; paths={key:None for key in paths}
    make_plot(long,stats,outdir,str(matrix))
    srcsha=hashlib.sha256(matrix.read_bytes()).hexdigest()
    audit={'generated_at':datetime.now().isoformat(),'rna_seq_matrix':str(matrix),'matrix_sha256':srcsha,'sample_columns':info['mapped'],'genes_included':IDS,'genes_excluded':['DSWSTRG.449','DSWSTRG.11494'],'statistics_sources':paths,'significance_from_original_DESeq2_only':True,'no_stats_means_no_significance_bars':True,'transform':'log10(original DESeq2-normalized count + 1)','no_AI_rendering':True}
    (outdir/'SOURCE_AUDIT.json').write_text(json.dumps(audit,indent=2))
    print('SUCCESS:',outdir/'Figure4_RNAseq_11targets_plus_2532_FROM_DATA.png')
    print('INPUT (SHA256):',srcsha,'\nRNA-seq stats used:',stats.keys())
    print('NOTE: Confirm source paths, sample map and DESeq2 analysis provenance in SOURCE_AUDIT.json before submission.')
if __name__=='__main__':main()
