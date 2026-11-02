# ============================================================
# START DATE   : 2026-11-02
# CODE RUN DATE: 2026-11-02
# TIME TAKEN   : ~2 min
# DESCRIPTION  : Combines survey and customs test logs into one all-tests table, counts tests, applies the pre-fixed reading
#                rule (placebo p <= 0.10 and sign equals the pre-stated sign) and the expected number of chance passes.
# INPUT FILES  : analysis/43_.../01_survey_tests_261102.csv, 03_customs_<signal>_261102.csv
# OUTPUT FILES : analysis/43_.../01_all_tests_261102.csv, 04_summary_261102.csv
# ============================================================
import sys

import pandas as pd

from common_261102 import OUT

o1, o2 = OUT / "01_all_tests_261102.csv", OUT / "04_summary_261102.csv"
for o in (o1, o2):
    if o.exists():
        sys.exit(f"{o} exists; refusing to overwrite.")
s = pd.read_csv(OUT / "01_survey_tests_261102.csv")
s.insert(0, "data", "survey")
s["spec"] = s["sample"].astype(str) + "|" + s.outcome + "|" + s.aligned.astype(str) + "|" + s.kind
c = pd.concat([pd.read_csv(f) for f in sorted(OUT.glob("03_customs_*_261102.csv"))])
c.insert(0, "data", "customs")
c["sample"] = "DIME R/D firms, 2024-01..2026-03"; c["outcome"] = "trade_value (PPML)"
c["aligned"] = c.spec.str.extract(r"Rep_(\w+)")[0].fillna("none"); c["kind"] = c.spec.str[0].map({"A": "main", "B": "int"})
all_ = pd.concat([s, c], ignore_index=True, sort=False)
all_["sign_ok"] = ((all_.coef * all_.expected_sign) > 0).where(all_.expected_sign.notna())
all_["passes_rule"] = (all_.placebo_p <= 0.10) & (all_.sign_ok == True)
all_["placebo_le_10_any_sign"] = all_.placebo_p <= 0.10
all_.to_csv(o1, index=False)
ran = all_[all_.coef.notna()]
rows = []
for g, df in [("all", ran)] + list(ran.groupby("hyp")) + list(ran.groupby("data")):
    rows.append(dict(group=g, n_tests=len(df), placebo_le_10_any_sign=int(df.placebo_le_10_any_sign.sum()),
                     with_prestated_sign=int(df.expected_sign.notna().sum()), passes_rule=int(df.passes_rule.sum()),
                     chance_expected_at_10pct=round(0.10 * len(df), 1)))
pd.DataFrame(rows).to_csv(o2, index=False)
print(pd.DataFrame(rows).to_string())
print(all_[all_.passes_rule][["data", "hyp", "sample", "outcome", "signal", "aligned", "kind", "coef", "se", "n", "placebo_p"]].round(3).to_string())
