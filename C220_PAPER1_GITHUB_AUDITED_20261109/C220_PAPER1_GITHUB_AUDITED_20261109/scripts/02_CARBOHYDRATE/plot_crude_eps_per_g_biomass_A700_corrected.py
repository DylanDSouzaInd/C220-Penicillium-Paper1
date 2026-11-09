#!/usr/bin/env python3
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import kruskal
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows

XLSX='/data/dylan/01_ONT_PCRcDNARNASeq/salinity_rnaseq_analysis/10_final_thesis_paper_outputs/EPS_biochemical/CrudeEPS_repeat.xlsx'
OUTBASE='/data/data/dylan/01_ONT_PCRcDNARNASeq/salinity_rnaseq_analysis/10_final_thesis_paper_outputs/EPS_biochemical/crude_eps_per_g_biomass_A700_corrected'

biomass = {
    '0_R1':0.27,'0_R2':0.33,'0_R3':0.31,
    '10_R1':0.28,'10_R2':0.29,'10_R3':0.25,
    '20_R1':0.22,'20_R2':0.21,'20_R3':0.22,
}

std_conc = {1:0,2:25,3:50,4:100,5:150,6:200,7:300,8:400,9:600,10:800}
sample_names = ['0_R1','0_R2','0_R3','10_R1','10_R2','10_R3','20_R1','20_R2','20_R3']
dilution_row_map = {'D':('neat',1), 'E':('1:5',5), 'F':('1:10',10)}

def parse_sheet(sheet='Sheet2'):
    df = pd.read_excel(XLSX, sheet_name=sheet, header=None)
    # 490 layout
    mat490 = pd.DataFrame(index=list('ABCDEFGH'), columns=list(range(1,13)), dtype=float)
    for r, idx in zip(list('ABCDEFGH'), range(24,32)):
        for c in range(1,13):
            mat490.loc[r,c] = float(df.iloc[idx,c])
    # scans
    sc = df.iloc[55:151,:4].copy()
    sc.columns = ['well','A620','A660','A700']
    sc = sc.dropna(subset=['well'])
    sc[['A620','A660','A700']] = sc[['A620','A660','A700']].astype(float)
    sc = sc.set_index('well')
    return mat490, sc


def exact_perm_global(values, labels):
    def Fstat(vals, labs):
        groups=[vals[labs==g] for g in sorted(np.unique(labs))]
        grand=vals.mean()
        ssb=sum(len(g)*(g.mean()-grand)**2 for g in groups)
        ssw=sum(((g-g.mean())**2).sum() for g in groups)
        return (ssb/(len(groups)-1))/(ssw/(len(vals)-len(groups)))
    values=np.asarray(values,float); labels=np.asarray(labels)
    obs=Fstat(values,labels)
    idx=range(len(values))
    total=count=0
    for g0 in itertools.combinations(idx,3):
        rem=[i for i in idx if i not in g0]
        for g1 in itertools.combinations(rem,3):
            g2=[i for i in rem if i not in g1]
            lab=np.empty(len(values),dtype=int)
            lab[list(g0)]=0; lab[list(g1)]=1; lab[list(g2)]=2
            total += 1
            if Fstat(values,lab) >= obs - 1e-12:
                count += 1
    return obs, count/total


def exact_perm_contrast(values, is20):
    values=np.asarray(values,float); is20=np.asarray(is20,bool)
    obs=values[is20].mean() - values[~is20].mean()
    idx=range(len(values))
    total=count=0
    for comb in itertools.combinations(idx,3):
        mask=np.zeros(len(values),dtype=bool)
        mask[list(comb)] = True
        stat = values[mask].mean() - values[~mask].mean()
        total += 1
        if abs(stat) >= abs(obs) - 1e-12:
            count += 1
    return obs, count/total


def fmtp(p):
    return f'{p:.2e}' if p<1e-4 else f'{p:.4f}'

mat490, sc = parse_sheet('Sheet2')

# standards
std_rows=[]
for row in 'ABC':
    for c, conc in std_conc.items():
        well=f'{row}{c}'
        a490=float(mat490.loc[row,c]); a620=float(sc.loc[well,'A620']); a660=float(sc.loc[well,'A660']); a700=float(sc.loc[well,'A700'])
        ac=a490-a700
        std_rows.append([well,row,c,conc,a490,a620,a660,a700,ac])
std_df=pd.DataFrame(std_rows, columns=['Well','PlateRow','PlateCol','Std_ug_mL','A490','A620','A660','A700','A490_minus_A700'])
# linear fit including S0-S9 triplicates
slope, intercept = np.polyfit(std_df['Std_ug_mL'], std_df['A490_minus_A700'], 1)
yhat = slope*std_df['Std_ug_mL'] + intercept
y = std_df['A490_minus_A700']
r2 = 1 - ((y-yhat)**2).sum()/((y-y.mean())**2).sum()
std_summary = std_df.groupby('Std_ug_mL')['A490_minus_A700'].agg(['mean','std']).reset_index()

# controls
control_rows=[]
# water/filter blanks H1-H3, reagent H4-H6, wash G1-G9 grouped by salinity
for name,wells in [
    ('Water/filter blank',['H1','H2','H3']),
    ('Reagent blank',['H4','H5','H6']),
    ('Wash control 0%',['G1','G2','G3']),
    ('Wash control 10%',['G4','G5','G6']),
    ('Wash control 20%',['G7','G8','G9']),
]:
    for well in wells:
        r=well[0]; c=int(well[1:])
        a490=float(mat490.loc[r,c]); a620=float(sc.loc[well,'A620']); a660=float(sc.loc[well,'A660']); a700=float(sc.loc[well,'A700'])
        control_rows.append([name,well,a490,a620,a660,a700,a490-a700])
controls_df=pd.DataFrame(control_rows, columns=['Control','Well','A490','A620','A660','A700','A490_minus_A700'])
control_summary = controls_df.groupby('Control')['A490_minus_A700'].agg(['mean','std','min','max']).reset_index()
water_blank_mean = float(control_summary.loc[control_summary['Control']=='Water/filter blank','mean'].iloc[0])
reagent_blank_mean = float(control_summary.loc[control_summary['Control']=='Reagent blank','mean'].iloc[0])

# samples with all dilutions
sample_rows=[]
for row,(dil_label, dil_factor) in dilution_row_map.items():
    for c, sample in enumerate(sample_names, start=1):
        well=f'{row}{c}'
        a490=float(mat490.loc[row,c]); a620=float(sc.loc[well,'A620']); a660=float(sc.loc[well,'A660']); a700=float(sc.loc[well,'A700'])
        ac = a490-a700
        conc = (ac - intercept)/slope
        total_ug = max(0, conc)*dil_factor*0.7
        ug_per_g = total_ug/biomass[sample]
        condition = sample.split('_')[0] + '% DSW'
        sample_rows.append([sample, condition, dil_label, dil_factor, biomass[sample], well, a490,a620,a660,a700,ac, conc, total_ug, ug_per_g,
                            ac <= water_blank_mean, ac <= reagent_blank_mean])
sample_df = pd.DataFrame(sample_rows, columns=['Sample','Condition','Dilution','Dilution_factor','Dry_biomass_g','Well','A490','A620','A660','A700','A490_minus_A700','Estimated_conc_ug_mL','Total_crudeEPS_ug_in_700uL','CrudeEPS_ug_per_g_biomass','At_or_below_water_blank_mean','At_or_below_reagent_blank_mean'])

# Aggregate within each biological replicate using median across dilution-corrected readings
bioagg = sample_df.groupby(['Sample','Condition','Dry_biomass_g'])['CrudeEPS_ug_per_g_biomass'].agg(['median','mean','min','max']).reset_index()
bioagg = bioagg.rename(columns={'median':'BioRep_median_ug_per_g','mean':'BioRep_mean_ug_per_g','min':'BioRep_min_ug_per_g','max':'BioRep_max_ug_per_g'})
zero_mean = bioagg.loc[bioagg['Condition']=='0% DSW','BioRep_median_ug_per_g'].mean()
bioagg['Relative_vs_0pct_mean'] = bioagg['BioRep_median_ug_per_g'] / zero_mean
sample_df = sample_df.merge(bioagg[['Sample','Relative_vs_0pct_mean']], on='Sample', how='left')
sample_df['Reading_relative_vs_0pct_mean'] = sample_df['CrudeEPS_ug_per_g_biomass']/zero_mean

cond_summary = bioagg.groupby('Condition')['BioRep_median_ug_per_g'].agg(['count','mean','std','median','min','max']).reset_index()
cond_summary['Relative_condition_mean_vs_0pct'] = cond_summary['mean']/zero_mean

# stats on biological replicate medians
order=['0% DSW','10% DSW','20% DSW']
vals = [bioagg.loc[bioagg['Condition']==cond,'Relative_vs_0pct_mean'].to_numpy() for cond in order]
kw = kruskal(*vals)
label_map = {cond:i for i,cond in enumerate(order)}
allv = bioagg['Relative_vs_0pct_mean'].to_numpy()
labels = np.array([label_map[c] for c in bioagg['Condition']])
Fobs, perm_p = exact_perm_global(allv, labels)
obs_contrast, contrast_p = exact_perm_contrast(allv, bioagg['Condition'].eq('20% DSW').to_numpy())

# Plot
colors=['#79a9d1','#f2a64b','#70b66b']
markers={'neat':'o','1:5':'s','1:10':'^'}
fig, ax = plt.subplots(figsize=(10.8,6.4), dpi=300)
# boxplots from biological replicate medians
boxvals=[bioagg.loc[bioagg['Condition']==cond,'Relative_vs_0pct_mean'].to_numpy() for cond in order]
bp=ax.boxplot(boxvals, positions=[1,2,3], widths=0.40, patch_artist=True, showfliers=False,
              medianprops={'color':'black','linewidth':1.8}, boxprops={'linewidth':2.4}, whiskerprops={'linewidth':2.2}, capprops={'linewidth':2.2})
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color); patch.set_alpha(0.35); patch.set_edgecolor('#777777')
for w in bp['whiskers']+bp['caps']:
    w.set_color('#777777')
# all dilution-corrected readings
for i, cond in enumerate(order, start=1):
    sub=sample_df[sample_df['Condition']==cond].copy()
    # deterministic x jitter based on sample and dilution
    # x positions around group center by dilution
    offsets={'neat':-0.18,'1:5':0.0,'1:10':0.18}
    for dil, mk in markers.items():
        ss=sub[sub['Dilution']==dil]
        xs=np.full(len(ss),i)+offsets[dil]+np.linspace(-0.03,0.03,len(ss))
        ax.scatter(xs, ss['Reading_relative_vs_0pct_mean'], s=70, marker=mk, color=colors[i-1], alpha=0.28,
                   edgecolor='#7f7f7f', linewidth=0.7, zorder=2, label=dil if i==1 else None)
    # big bio replicate points (medians)
    b=bioagg[bioagg['Condition']==cond]
    xs=np.linspace(i-0.05,i+0.05,len(b))
    ax.scatter(xs, b['Relative_vs_0pct_mean'], s=180, marker='o', color=colors[i-1], edgecolor='black', linewidth=1.3, zorder=4)
    ax.text(i, max(b['Relative_vs_0pct_mean'])+0.12, 'n=3 bio reps', ha='center', va='bottom', fontsize=11)

# significance bracket 20 vs pooled lower salinity contrast
# bracket from x=1 to 3
allplot = np.r_[sample_df['Reading_relative_vs_0pct_mean'].values, bioagg['Relative_vs_0pct_mean'].values]
ymax=float(np.max(allplot)); ymin=float(np.min(allplot))
yrange=ymax-ymin
br_y=ymax+0.18; h=0.18
ax.plot([1,1,3,3],[br_y,br_y+h,br_y+h,br_y], color='black', lw=2.2, clip_on=False)
ax.text(2, br_y+h+0.02, f'A priori contrast: 20% vs 0%+10%, permutation p={fmtp(contrast_p)}', ha='center', va='bottom', fontsize=11)

ax.text(0.985,0.985, f'Exact permutation ANOVA-like p={fmtp(perm_p)}\nKruskal-Wallis p={fmtp(kw.pvalue)}',
        transform=ax.transAxes, ha='right', va='top', fontsize=11,
        bbox=dict(boxstyle='round,pad=0.22', facecolor='white', edgecolor='#bcbcbc', alpha=0.95))
ax.legend(title='A490−A700 corrected readings', frameon=False, fontsize=10.5, title_fontsize=12, loc='upper left', bbox_to_anchor=(0.01,0.965))
ax.set_xticks([1,2,3]); ax.set_xticklabels(order, fontsize=18)
ax.set_xlabel('Condition', fontsize=18)
ax.set_ylabel('Relative crude EPS-like carbohydrate\n(µg glucose equivalents / g dry biomass; 0% mean = 1)', fontsize=18)
ax.grid(axis='y', alpha=0.18)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
ax.tick_params(axis='y', labelsize=13)
ax.set_ylim(0, ymax+0.78)
fig.tight_layout()
fig.savefig(OUTBASE+'.png', bbox_inches='tight')
fig.savefig(OUTBASE+'.pdf', bbox_inches='tight')
plt.close(fig)

# Make workbook
wb=Workbook(); ws=wb.active; ws.title='Controls_Summary'
for r in dataframe_to_rows(control_summary, index=False, header=True): ws.append(r)
ws2=wb.create_sheet('Controls_All_Wells')
for r in dataframe_to_rows(controls_df, index=False, header=True): ws2.append(r)
ws3=wb.create_sheet('Standards_All_Wells')
for r in dataframe_to_rows(std_df, index=False, header=True): ws3.append(r)
ws4=wb.create_sheet('Standard_Curve_Summary')
curve_info=pd.DataFrame({'Metric':['Slope','Intercept','R2','Reference_correction','Primary_sheet_used'],
                         'Value':[slope, intercept, r2, 'A490 - A700', 'Sheet2 (17:36 read)']})
for r in dataframe_to_rows(curve_info, index=False, header=True): ws4.append(r)
for rr in dataframe_to_rows(std_summary, index=False, header=True): ws4.append([]); ws4.append(rr)
ws5=wb.create_sheet('Sample_Readings_AllDil')
for r in dataframe_to_rows(sample_df, index=False, header=True): ws5.append(r)
ws6=wb.create_sheet('BioRep_Aggregates')
for r in dataframe_to_rows(bioagg, index=False, header=True): ws6.append(r)
ws7=wb.create_sheet('Condition_Summary')
for r in dataframe_to_rows(cond_summary, index=False, header=True): ws7.append(r)
ws8=wb.create_sheet('Stats')
stats_df=pd.DataFrame({
    'Test':['Kruskal-Wallis on bio-rep medians','Exact permutation ANOVA-like on bio-rep medians','A priori contrast: 20% vs pooled 0%+10%'],
    'Statistic':[kw.statistic, Fobs, obs_contrast],
    'P_value':[kw.pvalue, perm_p, contrast_p],
    'Scale':['Relative crude EPS/g dry biomass','Relative crude EPS/g dry biomass','Relative crude EPS/g dry biomass']
})
for r in dataframe_to_rows(stats_df, index=False, header=True): ws8.append(r)
out_xlsx=OUTBASE+'.xlsx'
wb.save(out_xlsx)

# Text summary for quick viewing
with open(OUTBASE+'_summary.txt','w') as fh:
    fh.write('Primary read used: Sheet2 (17:36)\n')
    fh.write('Reference correction: A490 - A700\n')
    fh.write(f'Standard curve slope={slope:.8f}, intercept={intercept:.8f}, R2={r2:.4f}\n')
    fh.write(f'Water/filter blank mean corrected absorbance={water_blank_mean:.6f}\n')
    fh.write(f'Reagent blank mean corrected absorbance={reagent_blank_mean:.6f}\n')
    fh.write('\nWash control corrected absorbance means:\n')
    for _,row in control_summary[control_summary.Control.str.contains('Wash')].iterrows():
        fh.write(f"  {row['Control']}: mean={row['mean']:.6f}\n")
    fh.write('\nCondition summary (bio-rep medians, relative vs 0% mean):\n')
    fh.write(cond_summary.to_string(index=False))
    fh.write('\n\nStats:\n')
    fh.write(stats_df.to_string(index=False))

print('WROTE', OUTBASE+'.png', OUTBASE+'.pdf', out_xlsx, OUTBASE+'_summary.txt')
