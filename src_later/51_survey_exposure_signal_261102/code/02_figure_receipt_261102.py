# ============================================================
# START DATE   : 2026-11-02
# CODE RUN DATE: 2026-11-02
# TIME TAKEN   : ~5 min
# DESCRIPTION  : Figure of the 20 pre-specified primary tests (effect in outcome SD, placebo p) and the receipt JSON (hashes) for step 51.
# INPUT FILES  : analysis/51_survey_exposure_signal_261102/10_all_specs_261102.csv
# OUTPUT FILES : analysis/51_survey_exposure_signal_261102/12_fig_primary_261102.png, 12_fig_data_261102.csv, receipt_survey_exposure_261102.json
# ============================================================
import hashlib, json, sys, platform
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TOPIC = Path(__file__).resolve().parents[2]
A = TOPIC / "analysis/51_survey_exposure_signal_261102"
C = TOPIC / "code/51_survey_exposure_signal_261102"
if (A / "receipt_survey_exposure_261102.json").exists():
    sys.exit("receipt exists; refusing to overwrite.")
r = pd.read_csv(A / "10_all_specs_261102.csv")
p = r[(r.primary == 1) & r.note.isna()].copy()
p["b_sd"] = p.coef / p.y_sd
p["lo"] = (p.coef - 1.96 * p.se) / p.y_sd
p["hi"] = (p.coef + 1.96 * p.se) / p.y_sd
p["label"] = p.z.str.split("_").str[0] + " | " + p.outcome.str.replace("_prior", "").str.replace("import_tariff_expect", "imp_tariff_exp").str.replace("knowledge_new2_1", "China_exp_tariff") + " | " + p.kind_ + " 28d" if "kind_" in p else p.z.str.split("_").str[0] + " | " + p.outcome.str.replace("_prior", "") + " | " + p.signal.str.split("_").str[3] + " 28d"
p = p.sort_values(["z", "outcome", "signal"]).reset_index(drop=True)
p[["label", "z", "outcome", "signal", "n", "coef", "se", "b_sd", "lo", "hi", "p_cluster", "placebo_p", "mde_sd"]].to_csv(A / "12_fig_data_261102.csv", index=False)
fig, ax = plt.subplots(figsize=(8.5, 7.5))
y = np.arange(len(p))[::-1]
col = np.where(p.placebo_p < 0.05, "#c0392b", "#5d6d7e")
ax.errorbar(p.b_sd, y, xerr=[p.b_sd - p.lo, p.hi - p.b_sd], fmt="none", ecolor="#aab7b8", capsize=2)
ax.scatter(p.b_sd, y, c=col, zorder=3)
ax.axvline(0, color="k", lw=.8)
ax.set_yticks(y); ax.set_yticklabels(p.label, fontsize=7)
ax.set_xlabel("Effect of 1 SD of the base China-path signal x exposure, in SD of the outcome (95% CI, date+owner clustered)")
ax.set_title("Primary tests: exposure contrasts x China-path signal, response-date FE\nO1 China vs other importers (expect +), O2 China vs non-importers (expect +), O3 other importers vs non-importers (expect 0)\nRed = date-shift placebo p < 0.05", fontsize=8)
fig.tight_layout(); fig.savefig(A / "12_fig_primary_261102.png", dpi=150)


def sha(f): return hashlib.sha256(Path(f).read_bytes()).hexdigest()


SH = Path.home()  # not used; shared inputs resolved from TOPIC
shared = TOPIC.parents[4] / "PRIVATE_DATA_ROOT/SMB_tariff/ideology_analysis"
ins = {str(shared / "inputs/analysis_all.dta"): sha(shared / "inputs/analysis_all.dta"),
       str(shared / "outputs/alignment_for_stata.dta"): sha(shared / "outputs/alignment_for_stata.dta"),
       str(TOPIC / "analysis/49_statement_type_261102/05_daily_series_261102.csv"): sha(TOPIC / "analysis/49_statement_type_261102/05_daily_series_261102.csv"),
       str(TOPIC / "data/16_tariff_schedule_261029/china_additional_duty_steps_261029.csv"): sha(TOPIC / "data/16_tariff_schedule_261029/china_additional_duty_steps_261029.csv")}
import scipy, statsmodels
outs = {f.name: sha(f) for f in sorted(A.glob("*")) if f.is_file()}
code = {f.name: sha(f) for f in sorted(C.glob("*.py"))}
d = pd.read_csv(A / "10_design_261102.csv").iloc[0].to_dict()
json.dump(dict(step="51_survey_exposure_signal_261102", run_date="2026-11-02", inputs_sha256=ins, code_sha256=code,
               run_order=["01_exposure_signal_261102.py", "02_figure_receipt_261102.py"], outputs_sha256=outs,
               env=dict(python=platform.python_version(), pandas=pd.__version__, numpy=np.__version__, scipy=scipy.__version__),
               counts={k: (int(v) if float(v).is_integer() else float(v)) for k, v in d.items()},
               comparison="Reconstructed 28-day level and surprise signals match the stored s28/surprise columns of 05_daily_series exactly (max abs diff 0).",
               not_done=["no respondent rows saved", "no re-coding of statements", "no Grid or paid calls", "no multiple-testing correction", "inputs in #Shared_ read only",
                         "spec file 00_spec_261102.md written before estimation; section 6 correction (non-importer definition) also before estimation"]),
          open(A / "receipt_survey_exposure_261102.json", "w"), indent=1)
print("done")
