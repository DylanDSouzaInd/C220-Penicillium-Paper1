#!/usr/bin/env python3
"""Read-only forensic check of C220 Figure 4 data provenance and original DESeq2 statistics.

No source files edited. Never re-estimates statistics. No automatic selection of ambiguous results.
"""
import argparse
import csv
import gzip
import hashlib
import json
import os
import re
from pathlib import Path

CONTRASTS = ('0vs10', '0vs20', '10vs20')
EXPECTED = {
    '0vs10': {'DSWSTRG.12473': 5.41e-4, 'DSWSTRG.6916': 5.10e-4,
               'DSWSTRG.2225': 0.026, 'DSWSTRG.4764': 0.011,
               'DSWSTRG.12325': 0.044, 'DSWSTRG.7303': 4.11e-7},
    '0vs20': {'DSWSTRG.12473': 6.81e-8, 'DSWSTRG.6916': 2.08e-7,
               'DSWSTRG.2225': 0.002, 'DSWSTRG.4764': 1.93e-5,
               'DSWSTRG.12325': 7.29e-5, 'DSWSTRG.7303': 0.650},
    '10vs20': {'DSWSTRG.12473': 1.0, 'DSWSTRG.6916': 0.450,
                'DSWSTRG.2225': 1.0, 'DSWSTRG.4764': 0.575,
                'DSWSTRG.12325': 0.817, 'DSWSTRG.7303': 0.003},
}
TARGETS = [
    'DSWSTRG.12473', 'DSWSTRG.6916', 'DSWSTRG.2225', 'DSWSTRG.7271',
    'DSWSTRG.7303', 'DSWSTRG.7881', 'DSWSTRG.7160', 'DSWSTRG.4764',
    'DSWSTRG.12325', 'DSWSTRG.12648', 'DSWSTRG.9665', 'DSWSTRG.2532'
]
SKIP = {'figures', 'plots', 'images', 'bams', 'bam', 'fastq', 'fastqs', 'reads',
        'genome', 'genomes', '.git', '__pycache__', 'conda_envs'}
P_VALUE_NAMES = {'padj', 'adjusted_pvalue', 'adjusted_p_value', 'fdr', 'qvalue', 'q_value', 'adj_pval', 'adj_p_value'}
EXPR = re.compile(r'DSWSTRG\.\d+')
EXT = ('.csv', '.tsv', '.txt', '.csv.gz', '.tsv.gz', '.txt.gz')


def open_text(path):
    return gzip.open(path, 'rt', encoding='utf-8-sig', errors='replace', newline='') if str(path).endswith('.gz') else open(path, 'r', encoding='utf-8-sig', errors='replace', newline='')


def label_from_filename(path):
    s = str(path).lower().replace('-', '_')
    # These two orientations yield identical contrast p-values; direction must be checked for log2FC.
    codes = []
    for lbl, a, b in (('0vs10', '0', '10'), ('0vs20', '0', '20'), ('10vs20', '10', '20')):
        regexes = [rf'(?<!\d){a}[_]?vs[_]?{b}(?!\d)', rf'(?<!\d){b}[_]?vs[_]?{a}(?!\d)',
                   rf'(?<!\d)dsw{a}[_]?dsw{b}(?!\d)', rf'(?<!\d)dsw{b}[_]?dsw{a}(?!\d)',
                   rf'(?<!\d){a}[_]?v[_]?{b}(?!\d)', rf'(?<!\d){b}[_]?v[_]?{a}(?!\d)']
        if any(re.search(q, s) for q in regexes): codes.append(lbl)
    return codes


def inspect_stats(path):
    try:
        with open_text(path) as f:
            head = f.readline()
            if not head: return None
            if len(head) > 100000: return None
            delim = '\t' if head.count('\t') > head.count(',') else ','
            fieldnames = next(csv.reader([head], delimiter=delim))
            pcol = next((x for x in fieldnames if x.strip().lower() in P_VALUE_NAMES), None)
            if not pcol: return None
            gcol = next((x for x in fieldnames if x.strip().lower() in
                         {'gene', 'gene_id', 'geneid', 'id', 'locus', 'feature', 'unnamed: 0'}), fieldnames[0])
            reader = csv.DictReader(f, fieldnames=fieldnames, delimiter=delim)
            found = {}
            for i, row in enumerate(reader):
                if i > 200000: break
                gene_field = str(row.get(gcol, '')).strip()
                match = EXPR.search(gene_field)
                if not match: continue
                gene = match.group(0)
                if gene not in TARGETS or gene in found: continue
                try:
                    p = float(row[pcol])
                    if 0 <= p <= 1: found[gene] = p
                except (ValueError, TypeError, KeyError): pass
            return {'path': str(path), 'padj_col': pcol, 'gene_col': gcol,
                    'gene_count': len(found), 'values': found, 'contrast_filename': label_from_filename(path)}
    except (OSError, UnicodeError, csv.Error):
        return None


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''): h.update(chunk)
    return h.hexdigest()


def score_candidate(row, contrast):
    matches = 0
    tested = 0
    for gene, expected in EXPECTED[contrast].items():
        found = row['values'].get(gene)
        if found is None: continue
        tested += 1
        # Source manuscript prints rounded values: allow 15% relative tolerance or 0.005 absolute at high q.
        if abs(found - expected) <= max(0.15 * expected, 0.005 if expected > .02 else 1e-8):
            matches += 1
    name_match = contrast in row['contrast_filename']
    score = (matches * 8 + row['gene_count'] * 2 + (8 if name_match else 0)
             - (12 if row['contrast_filename'] and not name_match else 0))
    return score, matches, tested, name_match


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run-dir', required=True)
    ap.add_argument('--root', default='/data/dylan/01_ONT_PCRcDNARNASeq/salinity_rnaseq_analysis')
    ap.add_argument('--max-files', type=int, default=20000)
    args = ap.parse_args()
    out = Path(args.run_dir).resolve()
    audit_file = out/'SOURCE_AUDIT.json'
    if not audit_file.is_file(): raise SystemExit(f'ERROR: missing {audit_file}')
    audit = json.loads(audit_file.read_text())
    src = Path(audit.get('rna_seq_matrix', ''))
    if not src.is_file(): raise SystemExit(f'ERROR: missing input matrix: {src}')
    current_sha = sha256(src)
    orig_sha = audit.get('matrix_sha256')
    if current_sha != orig_sha: raise SystemExit(f'ERROR: matrix SHA256 changed: expected {orig_sha}, got {current_sha}')

    samples = audit.get('sample_columns', {})
    positions = [tuple(v) for v in samples.values()]
    expected_positions = {(dsw, rep) for dsw in (0,10,20) for rep in (1,2,3)}
    sample_ok = len(samples) == 9 and set(positions) == expected_positions
    if not sample_ok: raise SystemExit(f'ERROR: unexpected sample map: {samples}')

    print('=== C220 FIGURE 4 SOURCE VERIFICATION ===')
    print('Input matrix:', src)
    print('SHA256 MATCH:', current_sha)
    print('RNA-seq samples: exactly 9 mapped (3 per treatment)')
    print('Sample mapping:')
    for col, loc in samples.items(): print(f'  {col:42s} -> {loc[0]}% DSW, R{loc[1]}')
    print('Excluded genes:', ', '.join(audit.get('genes_excluded', [])))
    print('Statistical files previously used:', audit.get('statistics_sources', {}))

    root = Path(args.root)
    candidates=[]
    n=0
    for directory, subdirs, files in os.walk(root, followlinks=False):
        subdirs[:] = [x for x in subdirs if x not in SKIP and not x.startswith('.')]
        for name in files:
            if not name.lower().endswith(EXT): continue
            p = Path(directory)/name
            try:
                if p.stat().st_size > 100 * 1024 * 1024: continue
            except OSError: continue
            n += 1
            if n > args.max_files:
                print('WARNING: scan limit reached; repeat with --max-files 50000')
                break
            d = inspect_stats(p)
            if d and d['gene_count']: candidates.append(d)
        if n > args.max_files: break

    report = out/'DESEQ2_STATISTICS_DISCOVERY_REVIEW.tsv'
    with report.open('w', newline='') as f:
        wr=csv.writer(f, delimiter='\t')
        wr.writerow(['contrast','score','manuscript_q_matches','manuscript_q_tested','source_name_match','target_genes_found','path','padj_column'])
        for contrast in CONTRASTS:
            ranked = sorted([(score_candidate(d, contrast), d) for d in candidates], key=lambda x:x[0][0], reverse=True)
            print(f'\n=== {contrast}: original DESeq2 candidate sources ===')
            for j, ((score, matches, tested, name_match), d) in enumerate(ranked[:8],1):
                print(f'{j}. score={score}; manuscript q match={matches}/{tested}; genes={d["gene_count"]}; name_match={name_match}')
                print('   ',d['path'])
            for ((score, matches, tested, name_match), d) in ranked:
                wr.writerow([contrast,score,matches,tested,name_match,d['gene_count'],d['path'],d['padj_col']])
    print('\nRead-only review report:', report)
    print('IMPORTANT: Candidate files are suggestions, not proven provenance. Inspect columns/contrast/model before rerunning the original plotting script with three explicit --stat-* paths.')

if __name__ == '__main__': main()
