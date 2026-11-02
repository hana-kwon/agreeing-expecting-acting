# ============================================================
# START DATE   : 2026-11-02
# CODE RUN DATE: 2026-11-02
# TIME TAKEN   : ~3 min
# DESCRIPTION  : Receipt for code/41: input and output SHA-256, code order, environment, comparison with the team's
#                Liberation Day result, what was not done, and a check that no firm-level column is in the project outputs.
# INPUT FILES  : see INPUTS below (read only); analysis/41_customs_team_spec_261102/*
# OUTPUT FILES : analysis/41_customs_team_spec_261102/receipt_customs_team_spec_261102.json
# ============================================================
import hashlib, json, platform, subprocess, sys
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve(); TOPIC = HERE.parents[2]; SHARED = HERE.parents[7] / "PRIVATE_DATA_ROOT"
OUT = TOPIC / "analysis/41_customs_team_spec_261102"; CODE = HERE.parent
rec = OUT / "receipt_customs_team_spec_261102.json"
if rec.exists(): sys.exit(f"output exists; refusing to overwrite: {rec}")
L = SHARED / "SMB_tariff/intermediate_data/prep_reg_L2_update"
TPU = Path("PRIVATE_SERVER_PATH")


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""): h.update(b)
    return h.hexdigest()


INPUTS = [L / "smb_trade_duns_biweekly_260512.parquet", L / "smb_duns_matched_firm_attributes_employment_260428.parquet",
          L / "china_sentiment/smb_china_sentiment_tariff_for_L2_261008.csv",
          SHARED / "TEAM_PRIVATE/ideology_share/directional_coding/data/jev_corpus/simple_time_series_v1/event_question_option_probabilities.parquet",
          TOPIC / "data/16_tariff_schedule_261029/china_additional_duty_steps_261029.csv",
          TOPIC / "data/31_tariff_calendar_fix_261101/02_panel_B_extended.csv",
          SHARED / "SMB_tariff/trade_results/Trade_and_Politics_Analysis.md",
          SHARED / "SMB_tariff/code/import_tariff_regressions/dime_regressions_small_med_large_HK_250718_EI.Rmd",
          TPU / "config/threat_resolution_dates.csv", TPU / "02_analysis/threat_resolution_results.md",
          TPU / "to_delete/02_analysis/02_code/estimate_threat_resolution.py",
          TOPIC / "code/27_expectation_action_261101/02_robustness_261101.py"]
inputs = {str(p): (sha(p) if p.exists() else "MISSING") for p in INPUTS}
codes = {p.name: sha(p) for p in sorted(CODE.glob("*")) if p.is_file()}
outs = {p.name: sha(p) for p in sorted(OUT.glob("*")) if p.is_file()}

# no firm-level identifier in any saved table
bad = []
for p in OUT.glob("*.csv"):
    cols = pd.read_csv(p, nrows=0).columns.tolist()
    if any(c in cols for c in ["final_dunsno", "firm_id", "duns"]): bad.append(p.name)

main = pd.read_csv(OUT / "04_main_coefs_261102.csv")
lib = main[(main.spec == "R_lib_full_window") & (main.outcome == "trade_value") & (main.term == "libC")][["tercile", "est", "se_firm"]]
team = {"all": (-0.568, 0.111), "large": (-0.581, 0.111), "medium": (-0.130, 0.077), "small": (-0.531, 0.160)}
cmp = {r.tercile: {"ours_est": r.est, "ours_se": r.se_firm, "team_est": team[r.tercile][0], "team_se": team[r.tercile][1]} for r in lib.itertuples()}


def ver(cmd):
    return subprocess.run(cmd, capture_output=True, text=True).stdout.strip()


env = {"python": platform.python_version(), "pandas": pd.__version__,
       "R": ver(["Rscript", "-e", "cat(R.version.string)"]),
       "R_packages": ver(["Rscript", "-e", "cat(paste(c('fixest','data.table','arrow'), sapply(c('fixest','data.table','arrow'), function(p) as.character(packageVersion(p))), collapse='; '))"])}
json.dump({
    "title": "code/41 customs team specification with leader-statement signal, 2026-11-02",
    "status": "exploratory; not independently cross-checked by a second agent",
    "inputs_sha256": inputs,
    "run_order": ["01_prep_261102.py", "07_prep_extras_261102.py", "02_estimate_261102.R tercile frac (small 1, medium 1, large 0.3)",
                  "06_extras_261102.R tercile frac (small 1, medium 0.5, large 0.1)", "03_placebo_261102.R tercile frac (small 1, medium 0.25, large 0.075)",
                  "08_employment_261102.R", "04_assemble_figures_261102.py", "05_receipt_261102.py"],
    "environment": env, "code_sha256": codes, "outputs_sha256": outs,
    "firm_level_columns_found_in_project_outputs": bad,
    "firm_level_cache_location": "PRIVATE_CACHE (cells, firms, firm_attrs parquet; not in the project)",
    "comparison_with_team_liberation_day_poisson_trade_value": cmp,
    "comparison_note": "Team: single origins, weekly, firm_id^origin + firm_id^year_week, panel from 2022. Ours: six origin groups, two-week periods from 2024-01, China-importing firms, "
                       "event bin dropped. Same numbers are not expected; the table is a sign and order-of-magnitude check only.",
    "not_done": ["rebuild of the Panjiva or D&B raw data", "Grid or cluster jobs", "paid calls", "external sending",
                 "use of the TPU_Small monthly importer x HS6 panel (only its resolution-date csv was used)",
                 "writing anything under PRIVATE_SERVER_PATH (read only)",
                 "full 96-shift placebo (32 shifts used)", "placebo for the employment-bin models and the origin x period FE variants",
                 "election and inauguration event replications", "full-sample large-tercile estimates (30% / 10% firm samples)"],
}, open(rec, "w"), indent=2)
print("receipt written", rec.name, "| firm-level columns found:", bad)
