# ============================================================
# START DATE   : 2026-11-02
# CODE RUN DATE: 2026-11-02
# TIME TAKEN   : ~3 min
# DESCRIPTION  : Receipt for code/44: input and output SHA-256, code order, environment, comparison with code/41 W_A coefficients,
#                what was not done, and a check that no firm-level column is in the project outputs.
# INPUT FILES  : see INPUTS (read only); analysis/44_customs_time_path_261102/*; analysis/41_.../04_main_coefs_261102.csv
# OUTPUT FILES : analysis/44_customs_time_path_261102/receipt_customs_time_path_261102.json
# ============================================================
import hashlib, json, platform, subprocess, sys
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve(); TOPIC = HERE.parents[2]; SHARED = HERE.parents[7] / "PRIVATE_DATA_ROOT"
OUT = TOPIC / "analysis/44_customs_time_path_261102"; CODE = HERE.parent; CACHE = Path.home() / "private_cache/customs40_261102"
rec = OUT / "receipt_customs_time_path_261102.json"
if rec.exists(): sys.exit(f"output exists; refusing to overwrite: {rec}")
L = SHARED / "SMB_tariff/intermediate_data/prep_reg_L2_update"


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""): h.update(b)
    return h.hexdigest()


INPUTS = [L / "smb_trade_duns_biweekly_260512.parquet", L / "china_sentiment/smb_china_sentiment_tariff_for_L2_261008.csv",
          TOPIC / "data/16_tariff_schedule_261029/china_additional_duty_steps_261029.csv",
          CACHE / "cells.parquet", CACHE / "firms.parquet", CACHE / "cells_placebo_261102.parquet",
          TOPIC / "analysis/41_customs_team_spec_261102/01_period_series_261102.csv",
          TOPIC / "analysis/41_customs_team_spec_261102/04_main_coefs_261102.csv"]
inputs = {str(p): (sha(p) if p.exists() else "MISSING") for p in INPUTS}
codes = {p.name: sha(p) for p in sorted(CODE.glob("*")) if p.is_file()}
outs = {p.name: sha(p) for p in sorted(OUT.glob("*")) if p.is_file() and p.name != rec.name}
bad = [p.name for p in OUT.glob("*.csv") if any(c in pd.read_csv(p, nrows=0).columns for c in ["final_dunsno", "firm_id", "duns"])]

# agreement of our P2_0 (windows only) with code/41 W_A on the same sample
m41 = pd.read_csv(TOPIC / "analysis/41_customs_team_spec_261102/04_main_coefs_261102.csv")
m41 = m41[(m41.spec == "W_A_china_windows") & (m41.outcome == "trade_value")][["tercile", "term", "est"]]
m44 = pd.concat([pd.read_csv(OUT / f"03_window_duty_{t}_261102.csv") for t in ["small", "medium", "large"]])
m44 = m44[(m44.spec == "P2_0_windows_only")][["tercile", "term", "est"]]
cmpd = m41.merge(m44, on=["tercile", "term"], suffixes=("_41", "_44"))
cmpd["abs_diff"] = (cmpd.est_41 - cmpd.est_44).abs()


def ver(cmd): return subprocess.run(cmd, capture_output=True, text=True).stdout.strip()


json.dump({
    "title": "code/44 China-area import time path, window model with duty control, window placebo; 2026-11-02",
    "status": "exploratory; not independently cross-checked by a second agent",
    "inputs_sha256": inputs,
    "run_order": ["01_prep_261102.py", "02_eventstudy_261102.R tercile frac (small 1, medium 1, large 0.3)",
                  "03_window_duty_261102.R tercile frac (same)", "03b_window_trend_261102.R (exploratory, added after seeing 2024 path)",
                  "04_placebo_windows_261102.R tercile frac (small 1, medium 0.25, large 0.075)", "05_assemble_figures_261102.py", "06_receipt_261102.py"],
    "environment": {"python": platform.python_version(), "pandas": pd.__version__, "R": ver(["Rscript", "-e", "cat(R.version.string)"]),
                    "R_packages": ver(["Rscript", "-e", "cat(paste(c('fixest','data.table','arrow'), sapply(c('fixest','data.table','arrow'), function(p) as.character(packageVersion(p))), collapse='; '))"])},
    "code_sha256": codes, "outputs_sha256": outs, "firm_level_columns_found_in_project_outputs": bad,
    "firm_level_cache_location": "PRIVATE_CACHE (cells_placebo_261102.parquet added; not in the project)",
    "comparison_with_code41_W_A": {"max_abs_diff_P2_0_vs_41_W_A": float(cmpd.abs_diff.max()), "rows": cmpd.round(6).to_dict("records"),
                                   "note": "P2_0 re-estimates the code/41 window model with the same sample and seed; differences should be ~0."},
    "not_done": ["rebuild of Panjiva or D&B raw data", "Grid or cluster jobs", "paid calls", "external sending", "writing under #Shared_ or private server",
                 "full-sample large tercile (30% firm sample for P1/P2, 7.5% for P3)", "medium placebo uses 25% of firms",
                 "placebo shifts +15..+26 weeks (resolved window too short, not estimable)", "event-study SE clustering by period",
                 "placebo for the trend or duty models"],
}, open(rec, "w"), indent=2)
print(cmpd.round(4).to_string())
