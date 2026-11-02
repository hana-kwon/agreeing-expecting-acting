# ============================================================
# START DATE   : 2026-11-01
# CODE RUN DATE: 2026-11-01
# TIME TAKEN   : See receipt
# DESCRIPTION  : Same-firm China-arrival transitions and first-observed origin persistence.
# INPUT FILES  : customs DUNS-origin-biweek parquet ($RESTRICTED_DATA_DIR/customs).
# OUTPUT FILES : Aggregate transition and persistence tables in 04_transition_run.
# ============================================================
from pathlib import Path
from datetime import datetime
import hashlib
import json
import pandas as pd
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from restricted_paths import SURVEY_DIR, CUSTOMS_DIR, JEV_DIR, OUTPUT_ROOT, out_dir, fresh_out_dir

HERE = Path(__file__).resolve()
INPUT = CUSTOMS_DIR/'smb_trade_duns_biweekly_260512.parquet'
OUT = fresh_out_dir('customs_extension/04_transition_run')
OUT.mkdir(exist_ok=False)
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()
before=sha(INPUT)
F,T='final_dunsno','year_biweek'
d=pd.read_parquet(INPUT,columns=[F,T,'origin','weightkg'])
d=d[d.origin.notna() & d.origin.ne('United States')].copy()
d['china']=d.origin.isin(['China','Hong Kong','Macau','Macao'])
phases={'march':[5,6],'april':[8,9],'may':[11,12],'july':[15,16]}
rows=[]; summaries=[]
# ★ CORE LOGIC: follow the same firm IDs; aggregate recovery alone cannot establish individual return.
for year in [2024,2025]:
    pre=d[d[T].str.startswith(f'{year-1}-')]
    active=pre.groupby(F)[T].nunique()
    ids=pd.Index(active[active.ge(2) & active.index.isin(pre.loc[pre.china,F].unique())].index)
    x=d[d[F].isin(ids)]
    states=pd.DataFrame(index=ids)
    for name,bs in phases.items():
        p=x[x[T].isin([f'{year}-{b:02d}' for b in bs]) & x.china]
        states[name]=states.index.isin(p[F].unique()).astype(int)
    for pattern,g in states.groupby(list(phases)):
        rows.append(dict(year=year,cohort_n=len(ids),march=pattern[0],april=pattern[1],may=pattern[2],july=pattern[3],n=len(g)))
    at_risk=states.april.eq(1)
    interrupted=at_risk & states.may.eq(0)
    resumed=interrupted & states.july.eq(1)
    # First observed in this panel, including all available years and early focal-year records.
    first=x[~x.china].groupby([F,'origin'])[T].min().rename('first_biweek').reset_index()
    entrants=first[first.first_biweek.ge(f'{year}-08') & first.first_biweek.le(f'{year}-12')]
    summer=x[x[T].isin([f'{year}-15',f'{year}-16'])][[F,'origin']].drop_duplicates().assign(repeated=1)
    match=entrants.merge(summer,on=[F,'origin'],how='left',validate='1:1')
    nfirms=entrants[F].nunique()
    repeated_firms=match.loc[match.repeated.eq(1),F].nunique()
    summaries.append(dict(year=year,cohort_n=len(ids),april_china_firms=int(at_risk.sum()),
                          no_may_china_firms=int(interrupted.sum()),resumed_july_china_firms=int(resumed.sum()),
                          fraction_interrupted=float(interrupted.sum()/at_risk.sum()),
                          fraction_resumed_among_interrupted=float(resumed.sum()/interrupted.sum()),
                          first_observed_spring_nonchina_firms=int(nfirms),same_origin_reobserved_july_firms=int(repeated_firms),
                          first_observed_spring_pairs=len(entrants),same_pairs_reobserved_july=int(match.repeated.eq(1).sum()),
                          origin_persistence_fraction=float(repeated_firms/nfirms)))
pd.DataFrame(rows).to_csv(OUT/'01_china_transition_counts.csv',index=False)
pd.DataFrame(summaries).to_csv(OUT/'02_transition_summary.csv',index=False)
assert sha(INPUT)==before
receipt=dict(timestamp=datetime.now().astimezone().isoformat(),inputs={str(INPUT):before},inputs_unchanged=True,
             code={str(HERE):sha(HERE)},environment={'pandas':pd.__version__},outputs={p.name:sha(p) for p in OUT.glob('*.csv')},
             first_available_period=d[T].min(),
             limitations=['No arrivals in one 4-week window does not establish supplier exit or a decision to wait.',
                          'First observed origin is not a new supplier or product, and history is left-censored.',
                          '2024 and 2025 cohorts and history lengths differ; the reference is descriptive, not causal.'])
(OUT/'receipt_transitions_261101.json').write_text(json.dumps(receipt,indent=2))
print(pd.DataFrame(summaries).to_string(index=False))
