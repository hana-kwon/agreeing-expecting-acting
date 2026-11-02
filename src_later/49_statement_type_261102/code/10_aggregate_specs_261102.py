# ============================================================
# START DATE   : 2026-11-02
# CODE RUN DATE: 2026-11-02
# TIME TAKEN   : ~10 min
# DESCRIPTION  : Step 5e. One log of every specification run in 06-09 with estimate, SE, placebo p and the total test count;
#                counts of nominal and placebo hits against chance. A "test" is one row (estimate with its own placebo).
# INPUT FILES  : analysis/49_statement_type_261102/{06_survey_results, retail_*/03_p1_estimates, market_*/02_m{1,2,3},
#                census_c5/07_c5_estimates}
# OUTPUT FILES : analysis/49_statement_type_261102/10_all_specs_log_261102.csv, 10_test_counts_261102.csv
# ============================================================
import importlib.util
from pathlib import Path
import numpy as np, pandas as pd
_s = importlib.util.spec_from_file_location("c", Path(__file__).with_name("common_261102.py"))
C = importlib.util.module_from_spec(_s); _s.loader.exec_module(C)
O = C.OUT
assert not (O / "10_all_specs_log_261102.csv").exists(), "exists; refusing to overwrite"
R = []
s = pd.read_csv(O / "06_survey_results_261102.csv")
for r in s.itertuples():
    R.append(dict(family="survey", variant=r.variant, spec=f"{r.sample}|{r.expectation}|{r.question}|{r.signal}", estimate=r.beta, se=r.se, placebo_p=r.placebo_p, n=r.n))
for d in sorted(O.glob("retail_*")):
    v = d.name.split("_", 1)[1]
    e = pd.read_csv(d / "03_p1_estimates.csv")
    for r in e.itertuples():
        R.append(dict(family="retail", variant=v, spec=f"{r.spec}|{r.contrast}", estimate=r.estimate, se=r.se_boot, placebo_p=r.placebo_p, n=r.n_all))
for d in sorted(O.glob("market_*")):
    v = d.name.split("_", 1)[1]
    for f in ["02_m1_261102.csv", "02_m2_261102.csv", "02_m3_261102.csv"]:
        e = pd.read_csv(d / f)
        for r in e.itertuples():
            # ★ IMPORTANT: ANN_* tests compare legal-announcement days and do not use the statement signal, so every variant repeats them; keep them once (base).
            if r.test.startswith("ANN_") and v != "base":
                continue
            R.append(dict(family="market", variant=v, spec=f"{r.signal}|{r.test}|{r.market}|{r.outcome}", estimate=r.estimate, se=r.se, placebo_p=r.placebo_p, n=r.n_E))
c5 = pd.read_csv(O / "census_c5/07_c5_estimates.csv")
for r in c5[(c5.column == "(3) both") & (c5.term == "z")].itertuples():
    R.append(dict(family="census_c5", variant=r.spec, spec=f"{r.spec}|{r.outcome}|{r.sample}", estimate=r.coef, se=r.se, placebo_p=getattr(r, "placebo_p", np.nan), n=r.N))
A = pd.DataFrame(R)
A["nominal_hit"] = (A.estimate.abs() / A.se > 1.96) & A.se.notna()
A["placebo_hit"] = A.placebo_p < 0.05
A.to_csv(O / "10_all_specs_log_261102.csv", index=False)
T = A.groupby("family").agg(tests=("estimate", "size"), with_estimate=("estimate", lambda x: int(x.notna().sum())), nominal_hits=("nominal_hit", "sum"),
                            with_placebo=("placebo_p", lambda x: int(x.notna().sum())), placebo_hits=("placebo_hit", "sum")).reset_index()
T.loc[len(T)] = ["TOTAL"] + [int(T[c].sum()) for c in T.columns[1:]]
T["expected_placebo_hits_at_5pct"] = (T.with_placebo * 0.05).round(1)
T.to_csv(O / "10_test_counts_261102.csv", index=False)
print(T.to_string())
