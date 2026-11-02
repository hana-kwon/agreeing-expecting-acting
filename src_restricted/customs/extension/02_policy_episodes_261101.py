# ============================================================
# START DATE   : 2026-11-01
# CODE RUN DATE: 2026-11-01
# TIME TAKEN   : See run receipt
# DESCRIPTION  : Fixed pre-policy cohort, arrival dynamics and source persistence.
# INPUT FILES  : customs DUNS-origin-biweek parquet and biweek calendar ($RESTRICTED_DATA_DIR/customs).
# OUTPUT FILES : Aggregate tables only, in 02_customs_run.
# ============================================================
from pathlib import Path
from datetime import datetime
import hashlib
import json
import platform
import numpy as np
import pandas as pd
import pyarrow
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from restricted_paths import SURVEY_DIR, CUSTOMS_DIR, JEV_DIR, OUTPUT_ROOT, out_dir, fresh_out_dir

HERE = Path(__file__).resolve()
L = CUSTOMS_DIR
OUT = fresh_out_dir('customs_extension/02_customs_run_v2')
OUT.mkdir(exist_ok=False)
INPUTS = [L / 'smb_trade_duns_biweekly_260512.parquet', L / 'china_sentiment/smb_china_sentiment_tariff_for_L2_261008.csv']
def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()
before = {str(p): sha(p) for p in INPUTS}
F, T = 'final_dunsno', 'year_biweek'
METRICS = ['weightkg', 'trade_value', 'volumeteu']
CHINA = ['China', 'Hong Kong', 'Macau', 'Macao']
ASIA = ['Japan', 'South Korea', 'Taiwan', 'Vietnam', 'Thailand', 'Malaysia', 'Indonesia', 'Philippines', 'Singapore', 'Cambodia', 'Laos', 'Myanmar', 'Brunei', 'Mongolia', 'North Korea']
d = pd.read_parquet(INPUTS[0], columns=[F, 'origin', T] + METRICS + ['missing_value'])
assert not d.duplicated([F, 'origin', T]).any()
assert d[[F, T]].notna().all().all()
unknown_origin = dict(rows=int(d.origin.isna().sum()), weight_fraction=float(d.loc[d.origin.isna(), 'weightkg'].sum()/d.weightkg.sum()))
d = d.dropna(subset=['origin'])
quality = {m: dict(missing=int(d[m].isna().sum()), negative=int(d[m].lt(0).sum()), zero=int(d[m].eq(0).sum())) for m in METRICS}
assert all(quality[m]['missing'] == 0 and quality[m]['negative'] == 0 for m in METRICS)
assert d.weightkg.gt(0).all()
d = d[d.origin.ne('United States')].copy()
d['region'] = np.select([d.origin.isin(CHINA), d.origin.isin(ASIA), d.origin.isin(['Canada', 'Mexico'])],
                        ['china_policy', 'other_east_se_asia', 'canada_mexico'], default='other_foreign')
cal = pd.read_csv(INPUTS[1], usecols=[T, 'bin_index', 'bin_start', 'bin_end', 'partial_bin'])
assert not cal[T].duplicated().any()
cal['bin_start'] = pd.to_datetime(cal.bin_start)
cal['bin_end'] = pd.to_datetime(cal.bin_end)
assert d[T].isin(cal[T]).all()
pre = d[d[T].str.startswith('2024-')]
active = pre.groupby(F)[T].nunique()
china_firms = pre.loc[pre.region.eq('china_policy'), F].unique()
cohort = pd.Index(active[active.ge(2) & active.index.isin(china_firms)].index)
assert len(cohort) > 0
baseline_pairs = pre.loc[pre[F].isin(cohort), [F, 'origin']].drop_duplicates().assign(observed_2024=1)
bins = [f'2025-{b:02d}' for b in range(2, 19)]
ccal = cal[cal[T].isin(bins)].sort_values('bin_index')
assert len(ccal) == len(bins) and not ccal.partial_bin.astype(bool).any()
assert ((ccal.bin_end-ccal.bin_start).dt.days.eq(13)).all()
d = d[d[F].isin(cohort) & d[T].isin(bins)].copy()
d = d.merge(baseline_pairs, on=[F, 'origin'], how='left', validate='m:1')
d['new_origin_2024_reference'] = d.observed_2024.isna()
d['new_nonchina_origin'] = d.new_origin_2024_reference & d.region.ne('china_policy')

# ★ CORE LOGIC: zero-fill a cohort chosen entirely in 2024; zero means no recorded arrival.
grid = pd.MultiIndex.from_product([cohort, bins], names=[F, T])
panel = d.groupby([F, T])[METRICS].sum().reindex(grid, fill_value=0)
panel['n_origins'] = d.groupby([F, T]).origin.nunique().reindex(grid, fill_value=0)
panel['n_new_nonchina_origins'] = d[d.new_nonchina_origin].groupby([F, T]).origin.nunique().reindex(grid, fill_value=0)
panel['any_arrival'] = panel.weightkg.gt(0).astype(int)
regions = ['china_policy', 'other_east_se_asia', 'canada_mexico', 'other_foreign']
for region in regions:
    g = d[d.region.eq(region)].groupby([F, T])[METRICS].sum().reindex(grid, fill_value=0)
    for m in METRICS:
        panel[f'{region}_{m}'] = g[m]
    panel[f'{region}_any'] = g.weightkg.gt(0).astype(int)
panel['china_share_active'] = panel.china_policy_weightkg.div(panel.weightkg.where(panel.weightkg.gt(0)))
panel['any_new_nonchina_origin'] = panel.n_new_nonchina_origins.gt(0).astype(int)
series = []
for biweek, g in panel.groupby(level=T):
    row = dict(year_biweek=biweek, cohort_n=len(cohort), active_n=int(g.any_arrival.sum()),
               active_fraction=g.any_arrival.mean(), china_active_fraction=g.china_policy_any.mean(),
               mean_china_share_active=g.china_share_active.mean(), mean_origin_count=g.n_origins.mean(),
               new_nonchina_fraction=g.any_new_nonchina_origin.mean())
    for m in METRICS:
        row[f'total_{m}'] = g[m].sum()
        for region in regions:
            row[f'{region}_{m}'] = g[f'{region}_{m}'].sum()
    series.append(row)
pd.DataFrame(series).merge(ccal, on=T, validate='1:1').to_csv(OUT / '01_biweek_dynamics.csv', index=False)

phases = {'march_pre': [5, 6], 'april_post': [8, 9], 'may_post': [11, 12], 'july_followthrough': [15, 16]}
phase_tables = {}
phase_rows = []
for label, bs in phases.items():
    pbins = [f'2025-{b:02d}' for b in bs]
    g = d[d[T].isin(pbins)]
    tab = g.groupby(F)[METRICS].sum().reindex(cohort, fill_value=0)
    tab['n_origins'] = g.groupby(F).origin.nunique().reindex(cohort, fill_value=0)
    tab['new_nonchina_n'] = g[g.new_nonchina_origin].groupby(F).origin.nunique().reindex(cohort, fill_value=0)
    tab['any_arrival'] = tab.weightkg.gt(0).astype(int)
    tab['any_new_nonchina'] = tab.new_nonchina_n.gt(0).astype(int)
    for region in regions:
        rr = g[g.region.eq(region)].groupby(F)[METRICS].sum().reindex(cohort, fill_value=0)
        for m in METRICS:
            tab[f'{region}_{m}'] = rr[m]
        tab[f'{region}_any'] = rr.weightkg.gt(0).astype(int)
    tab['china_share_active'] = tab.china_policy_weightkg.div(tab.weightkg.where(tab.weightkg.gt(0)))
    phase_tables[label] = tab
    row = dict(phase=label, cohort_n=len(cohort), active_n=int(tab.any_arrival.sum()), active_fraction=tab.any_arrival.mean(),
               china_active_fraction=tab.china_policy_any.mean(), mean_china_share_active=tab.china_share_active.mean(),
               mean_origin_count=tab.n_origins.mean(), new_nonchina_fraction=tab.any_new_nonchina.mean())
    for m in METRICS:
        row[f'total_{m}'] = tab[m].sum()
        for region in regions:
            row[f'{region}_{m}'] = tab[f'{region}_{m}'].sum()
    phase_rows.append(row)
pd.DataFrame(phase_rows).to_csv(OUT / '02_phases.csv', index=False)
contrasts = []
for name, a, b in [('april_episode', 'march_pre', 'april_post'), ('may_episode', 'april_post', 'may_post'), ('longer_followthrough', 'march_pre', 'july_followthrough')]:
    left, right = phase_tables[a], phase_tables[b]
    for m in METRICS:
        for prefix in ['', 'china_policy_', 'other_east_se_asia_']:
            col = prefix + m
            contrasts.append(dict(comparison=name, outcome=col, scale='aggregate_pct_change', n=len(cohort),
                                   before=left[col].sum(), after=right[col].sum(), change=100*(right[col].sum()/left[col].sum()-1)))
    for col in ['any_arrival', 'china_policy_any', 'other_east_se_asia_any', 'any_new_nonchina', 'n_origins']:
        contrasts.append(dict(comparison=name, outcome=col, scale='firm_mean_difference', n=len(cohort),
                               before=left[col].mean(), after=right[col].mean(), change=(right[col]-left[col]).mean()))
    both = left.china_share_active.notna() & right.china_share_active.notna()
    contrasts.append(dict(comparison=name, outcome='china_share_active', scale='paired_active_firm_difference', n=int(both.sum()),
                           before=left.loc[both, 'china_share_active'].mean(), after=right.loc[both, 'china_share_active'].mean(),
                           change=(right.loc[both, 'china_share_active']-left.loc[both, 'china_share_active']).mean()))
pd.DataFrame(contrasts).to_csv(OUT / '03_phase_contrasts.csv', index=False)

# ★ IMPORTANT: persistence is an explicitly selected descriptive subgroup, not a causal subgroup effect.
spring_new = d[d[T].isin(['2025-08', '2025-09', '2025-10', '2025-11', '2025-12']) & d.new_nonchina_origin][[F, 'origin']].drop_duplicates()
summer = d[d[T].isin(['2025-15', '2025-16'])][[F, 'origin']].drop_duplicates().assign(seen_summer=1)
repeat = spring_new.merge(summer, on=[F, 'origin'], how='left', validate='1:1')
spring_owners = spring_new[F].nunique()
retained_owners = repeat.loc[repeat.seen_summer.eq(1), F].nunique()
selection = dict(cohort_n=len(cohort), cohort_rule='China-policy origin in 2024; >=2 foreign-arrival biweeks in 2024',
                 spring_new_origin_firms=int(spring_owners), same_new_origin_seen_july_firms=int(retained_owners),
                 same_new_origin_seen_july_fraction=float(retained_owners/spring_owners) if spring_owners else None,
                 spring_new_origin_pairs=len(spring_new), same_pairs_seen_july=int(repeat.seen_summer.eq(1).sum()),
                 value_missing_flag_rows=int(d.missing_value.sum()), quality_known_origin_input=quality, unknown_origin_excluded=unknown_origin)
(OUT / '04_quality_and_persistence.json').write_text(json.dumps(selection, indent=2))
after = {str(p): sha(p) for p in INPUTS}
assert before == after
receipt = dict(timestamp=datetime.now().astimezone().isoformat(), inputs=before, inputs_unchanged=True,
               code={str(HERE): sha(HERE)}, environment=dict(python=platform.python_version(), pandas=pd.__version__, numpy=np.__version__, pyarrow=pyarrow.__version__),
               outputs={p.name: sha(p) for p in OUT.iterdir()},
               limits=['Descriptive fixed-cohort arrivals, not causal effects of statements or policy.',
                       'Origins are not products or supplier identities; no linked survey expectations.',
                       'No unit-level rows written, no external calls, no donor subgroup tests.'])
(OUT / 'receipt_policy_episodes_261101.json').write_text(json.dumps(receipt, indent=2))
print(json.dumps(selection, indent=2))
print(pd.DataFrame(phase_rows).to_string(index=False))
