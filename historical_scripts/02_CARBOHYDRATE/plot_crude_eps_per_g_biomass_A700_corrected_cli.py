#!/usr/bin/env python3
"""
Correct crude EPS-like carbohydrate / g dry biomass analysis.

Use the RAW Tecan workbook, not the already-calculated workbook, because this script
needs Sheet2/Sheet3 with the A490 plate and the 620/660/700 nm scan.

Example:
  python plot_crude_eps_per_g_biomass_A700_corrected_cli.py \
    --xlsx CrudeEPS_repeat.xlsx \
    --biomass biomass_template_fill_this.csv \
    --out crude_eps_per_g_biomass_relative.png
"""

import argparse
import itertools
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import kruskal
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows

STD_CONC = {1: 0, 2: 25, 3: 50, 4: 100, 5: 150, 6: 200, 7: 300, 8: 400, 9: 600, 10: 800}
SAMPLES = ['0_R1','0_R2','0_R3','10_R1','10_R2','10_R3','20_R1','20_R2','20_R3']
DILUTION_ROW_MAP = {'D': ('neat', 1), 'E': ('1:5', 5), 'F': ('1:10', 10)}
ORDER = ['0% DSW', '10% DSW', '20% DSW']
COLORS = ['#79a9d1', '#f2a64b', '#70b66b']
MARKERS = {'neat': 'o', '1:5': 's', '1:10': '^'}


def parse_args():
    p = argparse.ArgumentParser(description='A700-corrected crude EPS/g biomass analysis from raw Tecan workbook.')
    p.add_argument('--xlsx', required=True, help='Raw Tecan Excel workbook, e.g. CrudeEPS_repeat.xlsx')
    p.add_argument('--biomass', required=True, help='CSV with columns: Sample,dry_biomass_g')
    p.add_argument('--out', default='crude_eps_per_g_biomass_relative.png', help='Output PNG path')
    p.add_argument('--sheet', default='Sheet2', help='Raw Tecan sheet to use; default Sheet2')
    p.add_argument('--ref', default='700', choices=['620','660','700'], help='Reference wavelength for correction; default 700')
    return p.parse_args()


def outbase_from_out(out):
    root, ext = os.path.splitext(out)
    if ext.lower() in ['.png', '.pdf']:
        return root
    return out


def read_biomass(path):
    df = pd.read_csv(path)
    required = {'Sample', 'dry_biomass_g'}
    if not required.issubset(df.columns):
        raise ValueError(f'Biomass CSV must contain columns {required}. Found: {list(df.columns)}')
    df['dry_biomass_g'] = pd.to_numeric(df['dry_biomass_g'], errors='coerce')
    if df['dry_biomass_g'].isna().any():
        bad = df[df['dry_biomass_g'].isna()]
        raise ValueError('Some biomass values are missing or not numeric:\n' + bad.to_string(index=False))
    b = dict(zip(df['Sample'].astype(str), df['dry_biomass_g'].astype(float)))
    missing = [s for s in SAMPLES if s not in b]
    if missing:
        raise ValueError('Biomass CSV is missing samples: ' + ', '.join(missing))
    return b


def parse_raw_tecan_sheet(xlsx, sheet):
    try:
        df = pd.read_excel(xlsx, sheet_name=sheet, header=None)
    except ValueError:
        xl = pd.ExcelFile(xlsx)
        raise ValueError(
            f"Worksheet '{sheet}' not found in {xlsx}. Available sheets: {xl.sheet_names}\n"
            "Use the RAW Tecan file, usually CrudeEPS_repeat.xlsx, not the calculated workbook."
        )

    mat490 = pd.DataFrame(index=list('ABCDEFGH'), columns=list(range(1, 13)), dtype=float)
    for r, idx in zip(list('ABCDEFGH'), range(24, 32)):
        for c in range(1, 13):
            mat490.loc[r, c] = float(df.iloc[idx, c])

    scan = df.iloc[55:151, :4].copy()
    scan.columns = ['well', 'A620', 'A660', 'A700']
    scan = scan.dropna(subset=['well'])
    scan[['A620', 'A660', 'A700']] = scan[['A620', 'A660', 'A700']].astype(float)
    scan = scan.set_index('well')

    # sanity check
    for w in ['A1', 'H6', 'F9']:
        if w not in scan.index:
            raise ValueError(f'Could not find reference wavelength scan for well {w}. Check workbook layout.')
    return mat490, scan


def exact_perm_global(values, labels):
    def fstat(vals, labs):
        groups = [vals[labs == g] for g in sorted(np.unique(labs))]
        grand = vals.mean()
        ssb = sum(len(g) * (g.mean() - grand) ** 2 for g in groups)
        ssw = sum(((g - g.mean()) ** 2).sum() for g in groups)
        return (ssb / (len(groups) - 1)) / (ssw / (len(vals) - len(groups)))

    values = np.asarray(values, float)
    labels = np.asarray(labels)
    obs = fstat(values, labels)
    idx = range(len(values))
    count = total = 0
    for g0 in itertools.combinations(idx, 3):
        rem = [i for i in idx if i not in g0]
        for g1 in itertools.combinations(rem, 3):
            g2 = [i for i in rem if i not in g1]
            lab = np.empty(len(values), dtype=int)
            lab[list(g0)] = 0
            lab[list(g1)] = 1
            lab[list(g2)] = 2
            total += 1
            if fstat(values, lab) >= obs - 1e-12:
                count += 1
    return obs, count / total


def exact_perm_contrast(values, is20):
    values = np.asarray(values, float)
    is20 = np.asarray(is20, bool)
    obs = values[is20].mean() - values[~is20].mean()
    idx = range(len(values))
    count = total = 0
    for comb in itertools.combinations(idx, 3):
        mask = np.zeros(len(values), dtype=bool)
        mask[list(comb)] = True
        stat = values[mask].mean() - values[~mask].mean()
        total += 1
        if abs(stat) >= abs(obs) - 1e-12:
            count += 1
    return obs, count / total


def fmtp(p):
    return f'{p:.2e}' if p < 1e-4 else f'{p:.4f}'


def calculate(xlsx, biomass_csv, sheet='Sheet2', ref='700'):
    biomass = read_biomass(biomass_csv)
    mat490, scan = parse_raw_tecan_sheet(xlsx, sheet)
    ref_col = f'A{ref}'

    # Standards: A-C rows, S0-S9 in columns 1-10.
    std_rows = []
    for row in 'ABC':
        for c, conc in STD_CONC.items():
            well = f'{row}{c}'
            a490 = float(mat490.loc[row, c])
            a620 = float(scan.loc[well, 'A620'])
            a660 = float(scan.loc[well, 'A660'])
            a700 = float(scan.loc[well, 'A700'])
            acorr = a490 - float(scan.loc[well, ref_col])
            std_rows.append([well, row, c, conc, a490, a620, a660, a700, acorr])
    std_df = pd.DataFrame(std_rows, columns=['Well','PlateRow','PlateCol','Std_ug_mL','A490','A620','A660','A700',f'A490_minus_A{ref}'])

    slope, intercept = np.polyfit(std_df['Std_ug_mL'], std_df[f'A490_minus_A{ref}'], 1)
    y = std_df[f'A490_minus_A{ref}'].to_numpy()
    yhat = slope * std_df['Std_ug_mL'].to_numpy() + intercept
    r2 = 1 - ((y-yhat)**2).sum() / ((y-y.mean())**2).sum()
    std_summary = std_df.groupby('Std_ug_mL')[f'A490_minus_A{ref}'].agg(['mean','std']).reset_index()

    # Controls.
    control_rows = []
    for name, wells in [
        ('Water/filter blank', ['H1','H2','H3']),
        ('Reagent blank', ['H4','H5','H6']),
        ('Wash control 0%', ['G1','G2','G3']),
        ('Wash control 10%', ['G4','G5','G6']),
        ('Wash control 20%', ['G7','G8','G9']),
    ]:
        for well in wells:
            r = well[0]
            c = int(well[1:])
            a490 = float(mat490.loc[r, c])
            a620 = float(scan.loc[well, 'A620'])
            a660 = float(scan.loc[well, 'A660'])
            a700 = float(scan.loc[well, 'A700'])
            acorr = a490 - float(scan.loc[well, ref_col])
            control_rows.append([name, well, a490, a620, a660, a700, acorr])
    controls_df = pd.DataFrame(control_rows, columns=['Control','Well','A490','A620','A660','A700',f'A490_minus_A{ref}'])
    control_summary = controls_df.groupby('Control')[f'A490_minus_A{ref}'].agg(['mean','std','min','max']).reset_index()
    water_blank_mean = float(control_summary.loc[control_summary['Control']=='Water/filter blank','mean'].iloc[0])
    reagent_blank_mean = float(control_summary.loc[control_summary['Control']=='Reagent blank','mean'].iloc[0])

    # Samples: D/E/F are neat/1:5/1:10.
    sample_rows = []
    for row, (dil_label, dil_factor) in DILUTION_ROW_MAP.items():
        for c, sample in enumerate(SAMPLES, start=1):
            well = f'{row}{c}'
            a490 = float(mat490.loc[row, c])
            a620 = float(scan.loc[well, 'A620'])
            a660 = float(scan.loc[well, 'A660'])
            a700 = float(scan.loc[well, 'A700'])
            acorr = a490 - float(scan.loc[well, ref_col])
            conc = (acorr - intercept) / slope
            total_ug = max(0, conc) * dil_factor * 0.7  # final dissolved volume = 700 uL = 0.7 mL
            ug_per_g = total_ug / biomass[sample]
            condition = sample.split('_')[0] + '% DSW'
            sample_rows.append([
                sample, condition, dil_label, dil_factor, biomass[sample], well,
                a490, a620, a660, a700, acorr, conc, total_ug, ug_per_g,
                acorr <= water_blank_mean, acorr <= reagent_blank_mean
            ])
    sample_df = pd.DataFrame(sample_rows, columns=[
        'Sample','Condition','Dilution','Dilution_factor','Dry_biomass_g','Well',
        'A490','A620','A660','A700',f'A490_minus_A{ref}','Estimated_conc_ug_mL',
        'Total_crudeEPS_ug_in_700uL','CrudeEPS_ug_per_g_biomass',
        'At_or_below_water_blank_mean','At_or_below_reagent_blank_mean'
    ])

    bioagg = sample_df.groupby(['Sample','Condition','Dry_biomass_g'])['CrudeEPS_ug_per_g_biomass'].agg(['median','mean','min','max']).reset_index()
    bioagg = bioagg.rename(columns={
        'median':'BioRep_median_ug_per_g',
        'mean':'BioRep_mean_ug_per_g',
        'min':'BioRep_min_ug_per_g',
        'max':'BioRep_max_ug_per_g'
    })
    zero_mean = bioagg.loc[bioagg['Condition']=='0% DSW','BioRep_median_ug_per_g'].mean()
    bioagg['Relative_vs_0pct_mean'] = bioagg['BioRep_median_ug_per_g'] / zero_mean
    sample_df = sample_df.merge(bioagg[['Sample','Relative_vs_0pct_mean']], on='Sample', how='left')
    sample_df['Reading_relative_vs_0pct_mean'] = sample_df['CrudeEPS_ug_per_g_biomass'] / zero_mean

    cond_summary = bioagg.groupby('Condition')['BioRep_median_ug_per_g'].agg(['count','mean','std','median','min','max']).reset_index()
    cond_summary['Relative_condition_mean_vs_0pct'] = cond_summary['mean'] / zero_mean

    vals = [bioagg.loc[bioagg['Condition']==cond,'Relative_vs_0pct_mean'].to_numpy() for cond in ORDER]
    kw = kruskal(*vals)
    label_map = {cond: i for i, cond in enumerate(ORDER)}
    allv = bioagg['Relative_vs_0pct_mean'].to_numpy()
    labels = np.array([label_map[c] for c in bioagg['Condition']])
    fobs, perm_p = exact_perm_global(allv, labels)
    obs_contrast, contrast_p = exact_perm_contrast(allv, bioagg['Condition'].eq('20% DSW').to_numpy())
    stats_df = pd.DataFrame({
        'Test': ['Kruskal-Wallis on bio-rep medians', 'Exact permutation ANOVA-like on bio-rep medians', 'A priori contrast: 20% vs pooled 0%+10%'],
        'Statistic': [kw.statistic, fobs, obs_contrast],
        'P_value': [kw.pvalue, perm_p, contrast_p],
        'Scale': ['Relative crude EPS/g dry biomass']*3
    })

    return {
        'std_df': std_df, 'std_summary': std_summary, 'slope': slope, 'intercept': intercept, 'r2': r2,
        'controls_df': controls_df, 'control_summary': control_summary,
        'sample_df': sample_df, 'bioagg': bioagg, 'cond_summary': cond_summary, 'stats_df': stats_df,
        'kw_p': kw.pvalue, 'perm_p': perm_p, 'contrast_p': contrast_p,
        'sheet': sheet, 'ref': ref
    }


def make_plot(res, out_png):
    sample_df = res['sample_df']
    bioagg = res['bioagg']
    stats_df = res['stats_df']
    ref = res['ref']
    kw_p = res['kw_p']
    perm_p = res['perm_p']
    contrast_p = res['contrast_p']

    fig, ax = plt.subplots(figsize=(10.8, 6.4), dpi=300)

    boxvals = [bioagg.loc[bioagg['Condition']==cond,'Relative_vs_0pct_mean'].to_numpy() for cond in ORDER]
    bp = ax.boxplot(
        boxvals, positions=[1,2,3], widths=0.40, patch_artist=True, showfliers=False,
        medianprops={'color':'black','linewidth':1.8},
        boxprops={'linewidth':2.4}, whiskerprops={'linewidth':2.2}, capprops={'linewidth':2.2}
    )
    for patch, color in zip(bp['boxes'], COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.35)
        patch.set_edgecolor('#777777')
    for w in bp['whiskers'] + bp['caps']:
        w.set_color('#777777')

    for i, cond in enumerate(ORDER, start=1):
        sub = sample_df[sample_df['Condition']==cond].copy()
        offsets = {'neat': -0.18, '1:5': 0.0, '1:10': 0.18}
        for dil, mk in MARKERS.items():
            ss = sub[sub['Dilution']==dil]
            xs = np.full(len(ss), i) + offsets[dil] + np.linspace(-0.03, 0.03, len(ss))
            ax.scatter(xs, ss['Reading_relative_vs_0pct_mean'], s=70, marker=mk, color=COLORS[i-1], alpha=0.28,
                       edgecolor='#7f7f7f', linewidth=0.7, zorder=2, label=dil if i==1 else None)
        b = bioagg[bioagg['Condition']==cond]
        xs = np.linspace(i-0.05, i+0.05, len(b))
        ax.scatter(xs, b['Relative_vs_0pct_mean'], s=180, marker='o', color=COLORS[i-1],
                   edgecolor='black', linewidth=1.3, zorder=4)
        ax.text(i, max(b['Relative_vs_0pct_mean'])+0.12, 'n=3 bio reps', ha='center', va='bottom', fontsize=11)

    allplot = np.r_[sample_df['Reading_relative_vs_0pct_mean'].values, bioagg['Relative_vs_0pct_mean'].values]
    ymax = float(np.max(allplot))
    br_y = ymax + 0.18
    h = 0.18
    ax.plot([1,1,3,3], [br_y, br_y+h, br_y+h, br_y], color='black', lw=2.2, clip_on=False)
    ax.text(2, br_y+h+0.02, f'A priori contrast: 20% vs 0%+10%, permutation p={fmtp(contrast_p)}', ha='center', va='bottom', fontsize=11)

    ax.text(0.985, 0.985, f'Exact permutation ANOVA-like p={fmtp(perm_p)}\nKruskal-Wallis p={fmtp(kw_p)}',
            transform=ax.transAxes, ha='right', va='top', fontsize=11,
            bbox=dict(boxstyle='round,pad=0.22', facecolor='white', edgecolor='#bcbcbc', alpha=0.95))
    ax.legend(title=f'A490−A{ref} corrected readings', frameon=False, fontsize=10.5, title_fontsize=12,
              loc='upper left', bbox_to_anchor=(0.01, 0.965))
    ax.set_xticks([1,2,3])
    ax.set_xticklabels(ORDER, fontsize=18)
    ax.set_xlabel('Condition', fontsize=18)
    ax.set_ylabel('Relative crude EPS-like carbohydrate\n(µg glucose equivalents / g dry biomass; 0% mean = 1)', fontsize=18)
    ax.grid(axis='y', alpha=0.18)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='y', labelsize=13)
    ax.set_ylim(0, ymax+0.78)
    fig.tight_layout()

    outbase = outbase_from_out(out_png)
    outdir = os.path.dirname(os.path.abspath(outbase))
    os.makedirs(outdir, exist_ok=True)
    fig.savefig(outbase + '.png', bbox_inches='tight')
    fig.savefig(outbase + '.pdf', bbox_inches='tight')
    plt.close(fig)


def write_outputs(res, out_png):
    outbase = outbase_from_out(out_png)
    outdir = os.path.dirname(os.path.abspath(outbase))
    os.makedirs(outdir, exist_ok=True)

    wb = Workbook()
    ws = wb.active
    ws.title = 'Controls_Summary'
    for r in dataframe_to_rows(res['control_summary'], index=False, header=True): ws.append(r)
    for name, df in [
        ('Controls_All_Wells', res['controls_df']),
        ('Standards_All_Wells', res['std_df']),
        ('Standard_Curve_byConc', res['std_summary']),
        ('Sample_Readings_AllDil', res['sample_df']),
        ('BioRep_Aggregates', res['bioagg']),
        ('Condition_Summary', res['cond_summary']),
        ('Stats', res['stats_df']),
    ]:
        ws = wb.create_sheet(name)
        for r in dataframe_to_rows(df, index=False, header=True): ws.append(r)

    ws = wb.create_sheet('Standard_Curve_Info')
    info = pd.DataFrame({
        'Metric': ['Slope','Intercept','R2','Reference_correction','Primary_sheet_used'],
        'Value': [res['slope'], res['intercept'], res['r2'], f"A490 - A{res['ref']}", res['sheet']]
    })
    for r in dataframe_to_rows(info, index=False, header=True): ws.append(r)
    wb.save(outbase + '.xlsx')

    with open(outbase + '_summary.txt','w') as fh:
        fh.write(f"Primary read used: {res['sheet']}\n")
        fh.write(f"Reference correction: A490 - A{res['ref']}\n")
        fh.write(f"Standard curve slope={res['slope']:.8f}, intercept={res['intercept']:.8f}, R2={res['r2']:.4f}\n\n")
        fh.write('Controls summary:\n')
        fh.write(res['control_summary'].to_string(index=False))
        fh.write('\n\nCondition summary:\n')
        fh.write(res['cond_summary'].to_string(index=False))
        fh.write('\n\nStats:\n')
        fh.write(res['stats_df'].to_string(index=False))
    return [outbase + ext for ext in ['.png','.pdf','.xlsx','_summary.txt']]


def main():
    args = parse_args()
    res = calculate(args.xlsx, args.biomass, sheet=args.sheet, ref=args.ref)
    make_plot(res, args.out)
    paths = write_outputs(res, args.out)
    print('WROTE:')
    for p in paths:
        print('  ' + p)
    print('\nMain results:')
    print(res['cond_summary'].to_string(index=False))
    print('\nStats:')
    print(res['stats_df'].to_string(index=False))

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print('\nERROR:', e, file=sys.stderr)
        sys.exit(1)
