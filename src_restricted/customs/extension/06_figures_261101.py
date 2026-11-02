# ============================================================
# START DATE   : 2026-11-01
# CODE RUN DATE: 2026-11-01
# TIME TAKEN   : See receipt
# DESCRIPTION  : Figures from aggregate extension results.
# INPUT FILES  : 01_followup_run_v3, 03_seasonal_run, 04_transition_run CSV outputs.
# OUTPUT FILES : PNG/PDF figures and receipt in 06_figures.
# ============================================================
from pathlib import Path
from datetime import datetime
import json
import hashlib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from restricted_paths import SURVEY_DIR, CUSTOMS_DIR, JEV_DIR, OUTPUT_ROOT, out_dir, fresh_out_dir

HERE=Path(__file__).resolve()
BASE=OUTPUT_ROOT / 'customs_extension'
OUT=BASE/'06_figures_v2'
OUT.mkdir(exist_ok=False)
INPUTS=[BASE/'03_seasonal_run/01_seasonal_phases.csv',BASE/'04_transition_run/02_transition_summary.csv',BASE/'01_followup_run_v3/02_estimates.csv']
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'white'})
s=pd.read_csv(INPUTS[0]); t=pd.read_csv(INPUTS[1])
phases=['march_pre','april_post','may_post','july_followthrough']
fig,axs=plt.subplots(1,3,figsize=(13.8,4.7),gridspec_kw={'width_ratios':[1.55,1,1]})
for year,color,style in [(2024,'0.55','--'),(2025,'black','-')]:
    g=s[(s.cohort=='own_prior_year_cohort') & (s.year==year)].set_index('phase').loc[phases]
    values=g.china_weightkg/g.china_weightkg.iloc[0]*100
    axs[0].plot(range(4),values,marker='o',color=color,linestyle=style,label=str(year))
    for j,v in enumerate(values):
        if year==2025: axs[0].annotate(f'{v:.1f}',(j,v),xytext=(0,-15),textcoords='offset points',ha='center',fontsize=9)
axs[0].set_xticks(range(4),['March','April / May','May / June','July / August'])
axs[0].set_ylim(58,142);axs[0].set_ylabel('China-group arrival weight (March = 100)')
axs[0].set_title('A. Decline and recovery in arrival weight',loc='left',fontsize=10)
axs[0].legend(frameon=False,loc='upper left')
for ax,col,title,denom in [(axs[1],'fraction_interrupted','B. No May-window China arrival','Among April-window China importers'),(axs[2],'fraction_resumed_among_interrupted','C. China arrivals resume in July','Among firms without May-window arrivals')]:
    vals=t[col].to_numpy()*100
    ax.bar([0,1],vals,color=['0.65','0.15'],width=.55)
    for x,y in enumerate(vals): ax.text(x,y+1,f'{y:.1f}%',ha='center')
    ax.set_xticks([0,1],['2024','2025']);ax.set_ylim(0,56);ax.set_ylabel('Percent of firms')
    ax.set_title(title,loc='left',fontsize=10)
    ax.set_xlabel(denom,fontsize=8)
fig.suptitle('Recorded arrivals from China: gaps and recovery',x=.07,ha='left',fontsize=14)
fig.text(.07,.035,'2025 cohort: 77,399 prior-year China-group importers; 2024 reference: 72,672. China group includes Hong Kong and Macau.\n2025 windows: March 3-30; April 14-May 11; May 26-June 22; July 21-August 17. Each spans four weeks.\nA recorded-arrival gap is not supplier exit. Historical comparisons do not identify a policy effect.',fontsize=8)
fig.subplots_adjust(left=.07,right=.985,top=.82,bottom=.24,wspace=.48)
for ext in ['png','pdf']:fig.savefig(OUT/f'01_arrival_transitions.{ext}',dpi=180)
plt.close(fig)

e=pd.read_csv(INPUTS[2]); e=e[(e.spec=='complete_120') & (e.adjustment=='none')].set_index('outcome')
fig,ax=plt.subplots(figsize=(8,3.8))
for y,(outcome,label) in enumerate([('prices_post1_1','Later price report (n = 27)'),('invest_done_midpoint62_5','Later investment report (n = 29)')]):
    r=e.loc[outcome]
    ax.errorbar(r.difference,y,xerr=[[r.difference-r.low],[r.high-r.difference]],fmt='o',color='black',capsize=4)
    ax.text(25,y,f'{r.difference:+.2f}\n[{r.low:.2f}, {r.high:.2f}]',va='center',fontsize=9)
ax.axvline(0,color='.65',linewidth=1)
ax.set_yticks([0,1],['Later price report (n = 27)','Later investment report (n = 29)'])
ax.set_xlim(-25,40);ax.set_ylim(-.5,1.5);ax.invert_yaxis()
ax.set_xlabel('Escalating minus softening assignment; percentage-point scale')
ax.set_title('Later cumulative reports among observed returners',loc='left',fontsize=13)
fig.text(.29,.04,'1,697 eligible initial assignments; 61 later responses. Selected-sample means, HC1 95% intervals.\nInvestment uses corrected 62.5 interval midpoint. These are not effects on newly dated actions.',fontsize=8)
fig.subplots_adjust(left=.30,right=.98,top=.82,bottom=.29)
for ext in ['png','pdf']:fig.savefig(OUT/f'02_followup_reports.{ext}',dpi=180)
plt.close(fig)
receipt=dict(timestamp=datetime.now().astimezone().isoformat(),inputs={str(p):sha(p) for p in INPUTS},code={str(HERE):sha(HERE)},
             environment={'matplotlib':matplotlib.__version__,'pandas':pd.__version__,'numpy':np.__version__},outputs={p.name:sha(p) for p in OUT.iterdir()})
(OUT/'receipt_figures_261101.json').write_text(json.dumps(receipt,indent=2))
print('Saved two PNG/PDF figure pairs from aggregate tables.')
