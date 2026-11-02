# ============================================================
# START DATE   : 2026-11-02
# CODE RUN DATE: 2026-11-02
# TIME TAKEN   : ~5 min
# DESCRIPTION  : Receipt for step 40: SHA-256 of inputs, code and outputs, environment versions, the check that the
#                A1 "old" specification reproduces 10_slopes_no_agreement_261101.csv, and what was not done.
# INPUT FILES  : see common_261102.py; analysis/27_expectation_action_261101/10_slopes_no_agreement_261101.csv
# OUTPUT FILES : analysis/40_smb_pricing_controls_261102/receipt_smb_pricing_controls_261102.json
# ============================================================
import hashlib
import json
import platform
import sys

import matplotlib
import numpy as np
import pandas as pd
import scipy
import statsmodels

from common_261102 import ANALYSIS, DATA39, HERE, IA, TOPIC, TS, get_out, safe_write


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


out, _ = get_out()
code_dir = HERE.parent
inputs = [IA / "inputs/analysis_all.dta", IA / "outputs/alignment_for_stata.dta",
          TS / "event_question_option_probabilities.parquet",
          TOPIC / "data/16_tariff_schedule_261029/china_additional_duty_steps_261029.csv",
          DATA39 / "Cavallo_Llamas_Vazquez_domestic_imported_trend_line.csv",
          TOPIC / "analysis/27_expectation_action_261101/10_slopes_no_agreement_261101.csv"]
qsf = sorted((IA / "survey_exports").glob("*.qsf"))
outputs = sorted(p for p in out.iterdir() if p.is_file() and not p.name.startswith("receipt_"))

# reproduction of the old specification
new = pd.read_csv(out / "01_a1_controls_261102.csv")
new = new[(new["sample"] == "full") & (new.spec == "old")]
old = pd.read_csv(TOPIC / "analysis/27_expectation_action_261101/10_slopes_no_agreement_261101.csv")
old = old[old.spec == "no_agreement"]
m = new.merge(old, left_on=["wave_sample", "y"], right_on=["sample", "y"], suffixes=("_new", "_old"))
repro = dict(n_compared=len(m), max_abs_coef_diff=float((m.coef_new - m.coef_old).abs().max()),
             max_abs_se_diff=float((m.se_new - m.se_old).abs().max()), n_equal=bool((m.n_new == m.n_old).all()))

receipt = dict(
    topic="small-firm pricing: team controls (A1), weekly price timeline vs Cavallo et al. (A2), events and leader statements (A3)",
    spec="analysis/40_smb_pricing_controls_261102/00_spec_261102.md (written 2026-11-02 03:16 EST before estimation; changes in its  )",
    run_order=["01_a1_controls_261102.py", "02_a2_timeline_261102.py", "03_a3_events_statements_261102.py", "04_receipt_261102.py"],
    command="python code/40_smb_pricing_controls_261102/0N_*.py  (venv PRIVATE_CACHE; no --out = analysis/40_...)",
    environment=dict(python=sys.version.split()[0], platform=platform.platform(), pandas=pd.__version__, numpy=np.__version__,
                     statsmodels=statsmodels.__version__, scipy=scipy.__version__, matplotlib=matplotlib.__version__),
    inputs={str(p.relative_to(TOPIC.parents[4])) if TOPIC.parents[4] in p.parents else str(p): sha(p) for p in inputs},
    qsf_checked={p.name: sha(p) for p in qsf},
    code={p.name: sha(p) for p in sorted(code_dir.glob("*.py"))},
    outputs={p.name: sha(p) for p in outputs},
    comparison_to_baseline=dict(
        a1_old_spec_vs_10_slopes_no_agreement_261101=repro,
        note="A1 'old' = 10_slopes_no_agreement_261101.py specification re-implemented in common/01; W4 and W1-W3 rows compared."),
    not_done=["raw-data rebuild of the survey or team Stata run (the team's do-files were read, not executed; Python re-implements the controls)",
              "respondent-level files saved (none; only weekly/aggregate tables and one figure)",
              "paid calls", "external sending", "re-estimation of Cavallo et al. daily price responses",
              "cross-agent verification (not yet done)"],
    rules=["exploratory, not pre-registration; post-hoc additions listed in 00_spec  ",
           "responses are cross-sections at different dates; 'already done' is cumulative with no stated start date",
           "alignment_for_stata.dta fills missing importer/export/fcomp with 0 and sets miss_*=1; A2/A3 drop miss_importer_dum==1 rows"],
)
safe_write(json.dumps(receipt, indent=2, ensure_ascii=False), out / "receipt_smb_pricing_controls_261102.json")
print(json.dumps(dict(repro=repro, n_outputs=len(outputs)), indent=2))
