# ============================================================
# START DATE   : 2026-11-01
# CODE RUN DATE: 2026-11-01
# TIME TAKEN   : See run receipt
# DESCRIPTION  : First verified assignment and subsequent reports, with selection diagnostics.
# INPUT FILES  : survey inputs and alignment mapping ($RESTRICTED_DATA_DIR/survey).
# OUTPUT FILES : Aggregate outputs in analysis/28_longitudinal_event_extension_261101/01_followup_run.
# ============================================================
from pathlib import Path
import hashlib
import json
import platform
from datetime import datetime
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from restricted_paths import SURVEY_DIR, CUSTOMS_DIR, JEV_DIR, OUTPUT_ROOT, out_dir, fresh_out_dir

HERE = Path(__file__).resolve()
IA = SURVEY_DIR
OUT = fresh_out_dir('customs_extension/01_followup_run_v3')
OUT.mkdir(exist_ok=False)
INPUTS = [IA / 'inputs/analysis_all.dta', IA / 'outputs/alignment_for_stata.dta']
def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()
before = {str(p): sha(p) for p in INPUTS}
Y = ['invest_post1_1', 'prices_post1_1']
d = pd.read_stata(INPUTS[0], columns=['responseid', 'uid', 'startdate', 'country', 'three_info', 'finished'] + Y, convert_categoricals=False)
# ★ IMPORTANT: preserve inherited estimates and separately test a documented interval-midpoint correction.
d['invest_done_midpoint62_5'] = d.invest_post1_1.replace({-67.5: -62.5, 67.5: 62.5})
d['invest_decreased'] = d.invest_post1_1.lt(0).astype(float).where(d.invest_post1_1.notna())
d['prices_increased'] = d.prices_post1_1.gt(0).astype(float).where(d.prices_post1_1.notna())
Y += ['invest_done_midpoint62_5', 'invest_decreased', 'prices_increased']
a = pd.read_stata(INPUTS[1], columns=['responseid', 'wave'], convert_categoricals=False)
assert not d.responseid.duplicated().any() and not a.responseid.duplicated().any()
d = d.merge(a, on='responseid', validate='1:1')
d = d[d.country.eq(1)].copy()
d['date'] = pd.to_datetime(d.startdate, errors='raise')
assert d.uid.notna().all() and d.uid.ne('').all()
d = d.sort_values(['uid', 'date', 'responseid'])
end = d.date.max()
initial = d[d.three_info.isin([1, 2]) & d.wave.isin(['W1_copy2', 'W2_apr7', 'W4_may28'])].drop_duplicates('uid')
initial = initial.assign(escalating=lambda x: x.three_info.eq(1).astype(int))
groups = {k: g for k, g in d.groupby('uid', sort=False)}
summary, estimates, bounds, transitions = [], [], [], []
empty_specs = []

# ★ CORE LOGIC: select follow-up before inspecting outcome availability; no post-treatment baseline control.
specs = [('complete_120', 120, 120, False), ('complete_120_other_wave', 120, 120, True),
         ('complete_180', 180, 180, False), ('unequal_opportunity_120', 120, 15, False)]
for label, window, opportunity, other_wave in specs:
    eligible = initial[initial.date.le(end - pd.Timedelta(days=opportunity))].copy()
    rows = []
    for r in eligible.itertuples():
        g = groups[r.uid]
        future = g[g.date.ge(r.date + pd.Timedelta(days=15)) & g.date.le(r.date + pd.Timedelta(days=window))]
        if other_wave:
            future = future[future.wave.ne(r.wave)]
        row = dict(escalating=r.escalating, initial_wave=r.wave, initial_finished=str(r.finished).lower() == 'true', returned=0,
                   same_wave=0, gap_days=np.nan, followup_wave='', followup_assigned=np.nan)
        row.update({y: np.nan for y in Y})
        if len(future):
            f = future.iloc[0]
            row.update(returned=1, same_wave=int(f.wave == r.wave), gap_days=(f.date-r.date).total_seconds()/86400,
                       followup_wave=f.wave, followup_assigned=f.three_info)
            row.update({y: f[y] for y in Y})
        rows.append(row)
    p = pd.DataFrame(rows)
    if p.empty:
        empty_specs.append(label)
        continue
    for arm, g in p.groupby('escalating'):
        summary.append(dict(spec=label, arm=int(arm), eligible=len(g), returned=int(g.returned.sum()),
                            return_rate=g.returned.mean(), same_wave=int(g.same_wave.sum()),
                            initial_finished=int(g.initial_finished.sum()), median_gap=g.gap_days.median(),
                            investment_observed=int(g[Y[0]].notna().sum()), price_observed=int(g[Y[1]].notna().sum())))
    for (arm, w1, w2), g in p[p.returned.eq(1)].groupby(['escalating', 'initial_wave', 'followup_wave']):
        transitions.append(dict(spec=label, arm=int(arm), initial_wave=w1, followup_wave=w2, n=len(g)))
    for y in Y + ['returned']:
        s = p.dropna(subset=[y]).copy()
        if s.escalating.nunique() < 2 or len(s) < 8:
            continue
        for adjusted in [False, True]:
            x = s[['escalating']].astype(float)
            if adjusted:
                x = pd.concat([x, pd.get_dummies(s.initial_wave, prefix='wave', drop_first=True, dtype=float)], axis=1)
            fit = sm.OLS(s[y].astype(float), sm.add_constant(x, has_constant='add')).fit(cov_type='HC1')
            ci = fit.conf_int().loc['escalating']
            estimates.append(dict(spec=label, outcome=y, adjustment='initial_wave' if adjusted else 'none',
                                  n=len(s), n_escalating=int(s.escalating.sum()), n_softening=int(s.escalating.eq(0).sum()),
                                  mean_escalating=s.loc[s.escalating.eq(1), y].mean(), mean_softening=s.loc[s.escalating.eq(0), y].mean(),
                                  difference=fit.params['escalating'], se=fit.bse['escalating'], low=ci.iloc[0], high=ci.iloc[1]))
    for y, sign in [(Y[0], -1), (Y[1], 1)]:
        arm_bounds = {}
        for arm in [0, 1]:
            g = p[p.escalating.eq(arm)]
            observed = g[y].notna()
            successes = int((g.loc[observed, y] * sign > 0).sum())
            missing = int((~observed).sum())
            arm_bounds[arm] = (successes/len(g), (successes+missing)/len(g))
        bounds.append(dict(spec=label, outcome=y, binary='decreased' if sign < 0 else 'increased',
                           difference_lower=arm_bounds[1][0]-arm_bounds[0][1],
                           difference_upper=arm_bounds[1][1]-arm_bounds[0][0],
                           softening_lower=arm_bounds[0][0], softening_upper=arm_bounds[0][1],
                           escalating_lower=arm_bounds[1][0], escalating_upper=arm_bounds[1][1]))
for name, rows in [('01_sample', summary), ('02_estimates', estimates), ('03_bounds', bounds), ('04_wave_transitions', transitions)]:
    pd.DataFrame(rows).to_csv(OUT / f'{name}.csv', index=False)
after = {str(p): sha(p) for p in INPUTS}
assert before == after
receipt = dict(timestamp=datetime.now().astimezone().isoformat(), inputs=before, inputs_unchanged=True,
               code={str(HERE): sha(HERE)}, environment=dict(python=platform.python_version(), pandas=pd.__version__, numpy=np.__version__, statsmodels=statsmodels.__version__),
               last_recorded_response=str(end), unique_initial_assignments=len(initial), empty_specs=empty_specs,
               outputs={p.name: sha(p) for p in OUT.glob('*.csv')},
               limits=['No new data collection or causal identification of behavior since first response.',
                       'Reported differences among returners condition on outcome availability.',
                       'Assignment is not verified exposure. No initial post-treatment outcome controls.',
                       'No firm or respondent rows saved; no model calls; no external delivery.'])
(OUT / 'receipt_followup_261101.json').write_text(json.dumps(receipt, indent=2))
print(pd.DataFrame(summary).to_string(index=False))
print(pd.DataFrame(estimates).query("spec == 'complete_120' and adjustment == 'none'").to_string(index=False))
