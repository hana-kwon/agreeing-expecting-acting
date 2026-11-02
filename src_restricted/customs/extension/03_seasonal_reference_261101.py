# ============================================================
# START DATE   : 2026-11-01
# CODE RUN DATE: 2026-11-01
# TIME TAKEN   : See receipt
# DESCRIPTION  : Prior-year seasonal reference, not an untreated causal control.
# INPUT FILES  : customs DUNS-origin-biweek parquet ($RESTRICTED_DATA_DIR/customs).
# OUTPUT FILES : Aggregate phases and receipt in 03_seasonal_run.
# ============================================================
from pathlib import Path
from datetime import datetime
import hashlib
import json
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from restricted_paths import SURVEY_DIR, CUSTOMS_DIR, JEV_DIR, OUTPUT_ROOT, out_dir, fresh_out_dir

HERE = Path(__file__).resolve()
INPUT = CUSTOMS_DIR / 'smb_trade_duns_biweekly_260512.parquet'
OUT = fresh_out_dir('customs_extension/03_seasonal_run')
OUT.mkdir(exist_ok=False)
def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()
before = sha(INPUT)
F, T = 'final_dunsno', 'year_biweek'
d = pd.read_parquet(INPUT, columns=[F, T, 'origin', 'weightkg'])
d = d[d.origin.notna() & d.origin.ne('United States')].copy()
d['china'] = d.origin.isin(['China', 'Hong Kong', 'Macau', 'Macao'])
cohorts = {}
for year in [2024, 2025]:
    pre = d[d[T].str.startswith(str(year-1)+'-')]
    active = pre.groupby(F)[T].nunique()
    china = pre.loc[pre.china, F].unique()
    cohorts[year] = pd.Index(active[active.ge(2) & active.index.isin(china)].index)
common = cohorts[2024].intersection(cohorts[2025])
phases = {'march_pre': [5, 6], 'april_post': [8, 9], 'may_post': [11, 12], 'july_followthrough': [15, 16]}
rows = []
# ★ CORE LOGIC: repeat the same reference-year rule; do not treat a historical year as randomized control.
for kind in ['own_prior_year_cohort', 'common_cohort_sensitivity']:
    for year in [2024, 2025]:
        ids = cohorts[year] if kind == 'own_prior_year_cohort' else common
        pre = d[d[F].isin(ids) & d[T].str.startswith(str(year-1)+'-')]
        pairs = pre[[F, 'origin']].drop_duplicates().assign(seen_prior=1)
        x = d[d[F].isin(ids) & d[T].str.startswith(str(year)+'-')].merge(pairs, how='left', on=[F, 'origin'], validate='m:1')
        x['new_nonchina'] = x.seen_prior.isna() & ~x.china
        for name, bs in phases.items():
            g = x[x[T].isin([f'{year}-{b:02d}' for b in bs])]
            total = g.groupby(F).weightkg.sum().reindex(ids, fill_value=0)
            ch = g[g.china].groupby(F).weightkg.sum().reindex(ids, fill_value=0)
            new = g[g.new_nonchina][F].nunique()
            rows.append(dict(cohort=kind, year=year, phase=name, cohort_n=len(ids),
                             total_weightkg=total.sum(), china_weightkg=ch.sum(),
                             active_fraction=total.gt(0).mean(), china_active_fraction=ch.gt(0).mean(),
                             china_share_among_active=ch.div(total.where(total.gt(0))).mean(),
                             new_nonchina_fraction=new/len(ids)))
out = pd.DataFrame(rows)
out.to_csv(OUT/'01_seasonal_phases.csv', index=False)
assert sha(INPUT) == before
receipt = dict(timestamp=datetime.now().astimezone().isoformat(), inputs={str(INPUT): before}, inputs_unchanged=True,
               code={str(HERE):sha(HERE)}, environment=dict(pandas=pd.__version__, numpy=np.__version__),
               outputs={'01_seasonal_phases.csv':sha(OUT/'01_seasonal_phases.csv')},
               interpretation='Descriptive seasonal reference. Different prior-year cohorts may differ; common-cohort sensitivity conditions on 2024 activity. Neither is a valid untreated counterfactual by itself.')
(OUT/'receipt_seasonal_reference_261101.json').write_text(json.dumps(receipt,indent=2))
print(out.to_string(index=False))
