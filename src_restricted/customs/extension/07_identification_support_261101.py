# ============================================================
# START DATE   : 2026-11-01
# CODE RUN DATE: 2026-11-01
# TIME TAKEN   : See receipt
# DESCRIPTION  : Aggregate feasibility counts for within-firm product-origin comparisons.
# INPUT FILES  : Local v2 supplier-HS6-origin biweekly panel, read only.
# OUTPUT FILES : Aggregate JSON and receipt; no firm-level rows are saved.
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
PANEL = CUSTOMS_DIR / 'hs6_panel/smb_duns_supplier_hs6_origin_biweekly_261024.parquet'
OUT = fresh_out_dir('customs_extension/07_identification_support')
OUT.mkdir(exist_ok=False)
start = time.monotonic()


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


initial_hash = sha(PANEL)
d = pd.read_parquet(PANEL, columns=['final_dunsno', 'supplier_id', 'h0', 'origin', 'year_biweek', 'weightkg'],
                    filters=[('year_biweek', '>=', '2024-01'), ('year_biweek', '<=', '2024-27')])
# ★ CORE LOGIC: These are feasibility counts, not an estimation sample or a measure of substitutability.
keep = d.h0.notna() & d.origin.notna() & d.origin.ne('United States') & d.weightkg.gt(0)
x = d.loc[keep].copy()
assert x.h0.astype(str).str.fullmatch(r'\d{6}').all()
assert x.final_dunsno.notna().all()
x['china'] = x.origin.isin(['China', 'Hong Kong', 'Macau', 'Macao'])
g = x.groupby(['final_dunsno', 'h0']).china.agg(['any', 'all'])
ch = g[g['any']]
both = ch[~ch['all']]
r = {
    'status': 'Exploratory feasibility counts, not treatment effects or final regression sample',
    'definition': '2024-01 through 2024-27, nonmissing six-digit H0 and foreign origin, positive weightkg; China includes Hong Kong, Macau and Macao',
    'rows_2024': len(d), 'rows_eligible': len(x),
    'foreign_firm_product_pairs': len(g),
    'china_firm_product_pairs': len(ch),
    'china_firms': ch.index.get_level_values(0).nunique(),
    'china_and_other_origin_firm_product_pairs': len(both),
    'china_and_other_origin_firms': both.index.get_level_values(0).nunique(),
    'china_only_firm_product_pairs': len(ch) - len(both),
    'supplier_id_missing_rows_2024': int(d.supplier_id.isna().sum()),
    'supplier_id_nonmissing_fraction_eligible': float(x.supplier_id.notna().mean()),
    'limitations': ['Other-origin imports do not establish technical substitutability.',
                   'Pre-existing origin networks are not randomly assigned.',
                   'Recorded arrival gaps do not measure deliberate waiting or supplier exit.',
                   'Different from the previous 77,399-firm cohort: no two-biweek activity filter here.']
}
f = OUT / '01_feasibility_counts_261101.json'
f.write_text(json.dumps(r, indent=2))
assert sha(PANEL) == initial_hash
receipt = {
    'timestamp': datetime.now().astimezone().isoformat(),
    'inputs': {str(PANEL): initial_hash}, 'code': {str(HERE): sha(HERE)},
    'outputs': {f.name: sha(f)},
    'environment': {'python': platform.python_version(), 'pandas': pd.__version__, 'pyarrow': pyarrow.__version__},
    'elapsed_seconds': time.monotonic() - start,
    'comparison': '66,137 China-plus-other-origin pairs also independently computed by acceptance_scope_b.',
    'not_done': ['No regressions', 'No firm-level outputs', 'No new policy or supplier measures', 'No stock study', 'No external transmission']
}
(OUT / 'receipt_identification_support_261101.json').write_text(json.dumps(receipt, indent=2))
print(json.dumps(r, indent=2))
