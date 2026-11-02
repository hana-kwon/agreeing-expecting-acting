# ============================================================
# START DATE   : 2026-11-02
# CODE RUN DATE: 2026-11-02
# TIME TAKEN   : ~5 min
# DESCRIPTION  : I. Equivalence tests (TOST, 90% CI inside +-SESOI) for selected null results of this folder, and the receipt JSON
#                (input/output SHA-256, code order, environment, what was not done). SESOI values are a post hoc choice (see 00_spec).
# INPUT FILES  : analysis/54_review_fixes_261102/*.csv (this folder), analysis/35/02_panel_261102.csv
# OUTPUT FILES : analysis/54_review_fixes_261102/10_tost_261102.csv, receipt_review_fixes_261102.json
# ============================================================
import sys, json, hashlib, platform
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
HERE = Path(__file__).resolve(); TOPIC = HERE.parents[2]; A = TOPIC / "analysis/54_review_fixes_261102"
if (A / "10_tost_261102.csv").exists() or (A / "receipt_review_fixes_261102.json").exists(): sys.exit("refuse to overwrite")
rows = []
def tost(label, est, se, sesoi, unit):
    z = stats.norm.ppf(0.95); lo, hi = est - z * se, est + z * se       # 90% CI = two one-sided 5% tests
    p_low = stats.norm.sf((est + sesoi) / se); p_up = stats.norm.cdf((est - sesoi) / se)
    ci95 = (est - 1.96 * se, est + 1.96 * se)
    rows.append(dict(test=label, est=est, se=se, ci95_lo=ci95[0], ci95_hi=ci95[1], sesoi=sesoi, unit=unit, p_tost=max(p_low, p_up), equivalent=bool(lo > -sesoi and hi < sesoi)))
# G (log points per SD of signal), SESOI 0.03
g = pd.read_csv(A / "08_g_coefs_261102.csv"); g = g[(g.term == "S_threats4_cum") & g.spec.isin(["G1_cum_numeric_rate"])]
for _, r in g.iterrows(): tost(f"G {r['sample']} {r.spec}", r.est, r.se_2way, 0.03, "log points per SD signal")
# H primary: SESOI = 0.1 x SD of outcome (y_sd in file)
f = A / "07_h_primary_261102.csv"
if f.exists():
    h = pd.read_csv(f); h = h[h.note.fillna("") == ""]
    for _, r in h.iterrows(): tost(f"H {r.z} {r.outcome} {r.signal}", r.coef, r.se, 0.1 * r.y_sd, "outcome units per SD signal x Z (0.1 SD of outcome)")
# E: SESOI 0.1 SD of Y_S1
f = A / "06_e_sec_fe_261102.csv"
if f.exists():
    pan = pd.read_csv(TOPIC / "analysis/35_sec_tariff_expectations_261102/02_panel_261102.csv")
    e = pd.read_csv(f)
    for _, r in e.iterrows(): tost(f"E {r.y} {r.signal} firmFE={r.firm_fe}", r.beta, r.se, 0.1 * pan[r.y].std(), "Y per SD signal (0.1 SD of Y)")
t = pd.DataFrame(rows); t.to_csv(A / "10_tost_261102.csv", index=False)
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
code = sorted((TOPIC / "code/54_review_fixes_261102").glob("*.*"))
outs = sorted(p for p in A.glob("*") if p.is_file() and not p.name.startswith("receipt"))
inputs = [TOPIC / "analysis/35_sec_tariff_expectations_261102/02_panel_261102.csv", TOPIC / "analysis/36_census_imports_261102/07_c5_rate_path_monthly.csv"]
import subprocess
rec = dict(created="2026-11-02", analysis="54_review_fixes_261102", status="exploratory, cross-check not done",
           inputs_project={str(p.relative_to(TOPIC)): sha(p) for p in inputs if p.exists()},
           inputs_read_only_external=["PRIVATE_DATA_ROOT/SMB_tariff/ideology_analysis/inputs/analysis_all.dta", "…/outputs/alignment_for_stata.dta",
               "…/prep_reg_L2_update/smb_trade_duns_biweekly_260512.parquet", "…/hs6_panel/smb_tariff_h0_origin_month_261029.parquet", "…/hs6_panel/smb_duns_hs6_origin_year_261024.parquet",
               "Jev event_question_option_probabilities.parquet", "data/36_census_imports_261102/hs2ch_*.csv", "PRIVATE_CACHE*.parquet"],
           code_sha256={p.name: sha(p) for p in code}, output_sha256={p.name: sha(p) for p in outs},
           run_order=["01_a_slopes", "02_b_elapsed", "03_c_prep.py", "03_c_es.R small|medium|large", "05_d_prep.py", "05_d_ppml.R", "05b_d_yoy.R", "06_e_sec_fe", "07_h_primary", "08_g_rate_prep", "08_g_country_signal.R", "09_i_slopes_tost", "10_tost_receipt"],
           environment=dict(python=platform.python_version(), system_python="/usr/bin/python3", R=subprocess.run(["Rscript", "--version"], capture_output=True, text=True).stderr.strip() or "R", fixest="see session"),
           not_done=["no re-derivation of raw data", "G not run on full 2024-2026 bins (only the 16 bins with all lags inside the statement panel and coded rates)", "no wild cluster bootstrap by origin in G, only date-shift placebo",
                     "D wild bootstrap is a Kline-Santos score bootstrap p, not a bootstrap SE", "no import_locs exposure control (variable not usable in analysis_all.dta)", "no Grid, paid calls or external sends",
                     "respondent and firm rows not saved in the project (firm cells in PRIVATE_CACHE)"])
(A / "receipt_review_fixes_261102.json").write_text(json.dumps(rec, indent=1))
pd.set_option("display.width", 220, "display.float_format", lambda x: f"{x:.3f}"); print(t.to_string(index=False))
