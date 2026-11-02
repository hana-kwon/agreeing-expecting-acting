# ============================================================
# START DATE   : 2026-11-01
# CODE RUN DATE: 2026-11-01
# TIME TAKEN   : See receipt
# DESCRIPTION  : Check denominators and lookback windows behind analysis 33 new-origin claims.
# INPUT FILES  : Read-only firm-origin-biweek panel, calendar, saved analysis 33 aggregates.
# OUTPUT FILES : Aggregate diagnostics and receipt only; no firm-level rows.
# ============================================================
from pathlib import Path
from datetime import datetime
import hashlib
import json
import platform
import time
import pandas as pd
import pyarrow
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from restricted_paths import SURVEY_DIR, CUSTOMS_DIR, JEV_DIR, OUTPUT_ROOT, out_dir, fresh_out_dir

HERE = Path(__file__).resolve()
L = CUSTOMS_DIR
INPUTS = [L / 'smb_trade_duns_biweekly_260512.parquet',
          L / 'china_sentiment/smb_china_sentiment_tariff_for_L2_261008.csv',
          OUTPUT_ROOT / 'customs_pause_switch/01_decomposition_261101.csv']
OUT = fresh_out_dir('customs_extension/08_new_origin_diagnostic_v2')
OUT.mkdir(exist_ok=False)
started = time.monotonic()


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


before = {str(p): sha(p) for p in INPUTS}
F, T = 'final_dunsno', 'year_biweek'
CN = ['China', 'Hong Kong', 'Macau', 'Macao']
d = pd.read_parquet(INPUTS[0], columns=[F, T, 'origin'])
d = d[d.origin.notna() & d.origin.ne('United States')].copy()
assert not d.duplicated([F, T, 'origin']).any()
cal = pd.read_csv(INPUTS[1], usecols=[T, 'bin_start', 'bin_end', 'partial_bin'])
cal['bin_start'] = pd.to_datetime(cal.bin_start)
cal['bin_end'] = pd.to_datetime(cal.bin_end)
assert not cal[T].duplicated().any()
cal = cal[cal[T].isin(d[T].unique())].set_index(T)
source = pd.read_csv(INPUTS[2]).set_index(['group', 'year'])
rows = []
for year in [2023, 2024, 2025]:
    prev = d[d[T].str.startswith(f'{year-1}-')]
    activity = prev.groupby(F)[T].nunique()
    china_ids = prev.loc[prev.origin.isin(CN), F].unique()
    ids = activity[activity.ge(2) & activity.index.isin(china_ids)].index
    apr = d[d[F].isin(ids) & d[T].isin([f'{year}-08', f'{year}-09'])]
    risk = pd.Index(apr.loc[apr.origin.isin(CN), F].unique())
    may = d[d[F].isin(risk) & d[T].isin([f'{year}-11', f'{year}-12'])]
    continuing = pd.Index(may.loc[may.origin.isin(CN), F].unique())
    other_ids = pd.Index(may.loc[~may.origin.isin(CN), F].unique()).difference(continuing)
    pairs = may.loc[may[F].isin(other_ids) & ~may.origin.isin(CN), [F, 'origin']].drop_duplicates()
    cutoff = cal.loc[f'{year}-11', 'bin_start']
    start = cutoff - pd.Timedelta(days=365)
    # ★ CORE LOGIC: Reproduce the original definition, then retain observed intervals wholly inside a 365-day lookback.
    # Boundary intervals differ in length, so this does not reconstruct exactly 365 days of daily arrivals.
    old_bins = cal.index[cal.index.str[:4].isin([str(year-2), str(year-1)])]
    rolling_bins = cal.index[(cal.bin_start.ge(start)) & (cal.bin_end.lt(cutoff))]
    current_bins = cal.index[(cal.index.str.startswith(f'{year}-')) & cal.bin_end.lt(cutoff)]
    prior_current = d[d[F].isin(other_ids) & d[T].isin(current_bins)][[F, 'origin']].drop_duplicates()
    for rule, bins in [('original_prior_calendar_years', old_bins), ('trailing_365_contained_intervals', rolling_bins)]:
        history = d[d[F].isin(other_ids) & d[T].isin(bins)][[F, 'origin']].drop_duplicates()
        z = pairs.merge(history.assign(seen=1), how='left', on=[F, 'origin'], validate='1:1')
        new = z[z.seen.isna()][[F, 'origin']]
        new_n = new[F].nunique()
        repeat_current_n = new.merge(prior_current, on=[F, 'origin'], validate='1:1')[F].nunique()
        row = dict(year=year, rule=rule, april_risk_n=len(risk), other_only_n=len(other_ids),
                   new_origin_firms_in_other=new_n, new_share_among_other=new_n/len(other_ids),
                   new_share_among_april_risk=new_n/len(risk),
                   new_flag_but_same_origin_seen_current_year=repeat_current_n,
                   history_bins=len(bins), history_days=int(((cal.loc[bins, 'bin_end']-cal.loc[bins, 'bin_start']).dt.days+1).sum()),
                   history_partial_bins=int(cal.loc[bins, 'partial_bin'].astype(bool).sum()),
                   history_start=str(cal.loc[bins, 'bin_start'].min().date()),
                   history_end=str(cal.loc[bins, 'bin_end'].max().date()), may_start=str(cutoff.date()))
        if rule == 'original_prior_calendar_years':
            ref = source.loc[('china_area', year)]
            assert len(risk) == int(ref.at_risk)
            assert len(other_ids) == round(ref.O * ref.at_risk)
            assert abs(new_n/len(other_ids)-ref.new_origin_among_O) < 1e-12
        rows.append(row)
out = OUT / '01_new_origin_diagnostic_261101.csv'
pd.DataFrame(rows).to_csv(out, index=False)
assert all(sha(p) == before[str(p)] for p in INPUTS)
receipt = dict(timestamp=datetime.now().astimezone().isoformat(), inputs=before,
               code={str(HERE): sha(HERE)}, outputs={out.name: sha(out)},
               environment=dict(python=platform.python_version(), pandas=pd.__version__, pyarrow=pyarrow.__version__),
               elapsed_seconds=time.monotonic()-started,
               comparison='Original China-group April risk, other-only counts and new-origin conditional rates match analysis 33.',
               limitations=['April at-risk group is selected after the policy episode began.',
                            'New means not observed in the stated history window, not first-ever supplier entry.',
                            'No same-product requirement; no causal or statistical difference test.',
                            'Trailing-window definition is an exploratory diagnostic selected after reading results.',
                            'Contained intervals cover 357 or 364 days, and can include partial-year intervals; not an exact common 365-day daily window.'],
               not_done=['No regressions', 'No firm rows saved', 'No shared input edits', 'No proposal edits'])
(OUT / 'receipt_new_origin_diagnostic_261101.json').write_text(json.dumps(receipt, indent=2))
print(pd.DataFrame(rows).to_string(index=False))
