# ============================================================
# START DATE   : 2026-11-01
# CODE RUN DATE: 2026-11-01
# TIME TAKEN   : ~2 min
# DESCRIPTION  : Re-estimates the same-owner expectation slopes (done / planned / 25 percent scenario) without the
#                agreement score, which W4 and W2 measured after the message (post-treatment). Main version used in the proposal.
# INPUT FILES  : same as 06_scenario_compare_261101.py
# OUTPUT FILES : analysis/27_expectation_action_261101/10_slopes_no_agreement_261101.csv
# ============================================================
import csv, sys, warnings
from pathlib import Path
import pandas as pd, statsmodels.api as sm
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from restricted_paths import REPO, SURVEY_DIR, CUSTOMS_DIR, JEV_DIR, OUTPUT_ROOT, out_dir, fresh_out_dir
warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve(); IA = SURVEY_DIR
OUT = out_dir("survey")
if (OUT / "10_slopes_no_agreement_261101.csv").exists(): sys.exit("output exists")
Y = [f"{a}_post{k}_1" for a in ("invest", "prices") for k in (1, 2, 3)]
keep = ["responseid", "uid", "startdate", "country", "three_info", "short_term_prior", "import_tariff_expect", "sales_cost_guess1_1"] + Y
d = pd.read_stata(IA / "inputs/analysis_all.dta", columns=keep, convert_categoricals=False)
d = d[d.country == 1]
for c in keep[4:]: d[c] = pd.to_numeric(d[c], errors="coerce")
a = pd.read_stata(IA / "outputs/alignment_for_stata.dta", convert_categoricals=False)
d = d.merge(a[["responseid", "wave", "industry_code", "emp_2024_bins", "sales_2024_val", "import_loc_post2_1__toward", "import_loc_post2_3__away"]], on="responseid", validate="1:1")
d["date"] = pd.to_datetime(d.startdate, errors="coerce").dt.normalize()
steps = []
with open(REPO / "data/public/china_additional_duty_steps.csv", newline="") as f:
    r = csv.reader(f); next(r)
    for row in r: steps.append((pd.Timestamp(row[0]), float(row[1])))
steps = pd.DataFrame(steps, columns=["eff", "duty_cn"]).sort_values("eff")
d = pd.merge_asof(d.dropna(subset=["date"]).sort_values("date"), steps, left_on="date", right_on="eff", direction="backward")
for c in ["emp_2024_bins", "sales_2024_val"]:
    d["miss_" + c] = d[c].isna().astype(float); d[c] = d[c].fillna(d[c].median())
d["industry_code"] = d.industry_code.fillna(-1)
def fit(df, y, main):
    s = df.dropna(subset=[y] + main).copy()
    X = pd.concat([s[main + ["emp_2024_bins", "sales_2024_val", "miss_emp_2024_bins", "miss_sales_2024_val", "duty_cn"]].astype(float),
                   pd.get_dummies(s.wave.astype(str), prefix="w", drop_first=True).astype(float),
                   pd.get_dummies(s.industry_code.astype(int), prefix="i", drop_first=True).astype(float),
                   pd.get_dummies(s.three_info.fillna(0).astype(int), prefix="m", drop_first=True).astype(float)], axis=1)
    X = sm.add_constant(X.loc[:, X.std() > 0])
    return sm.OLS(s[y].astype(float), X).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(s.uid)[0]}), len(s)
rows = []
for sample, df, E in [("W4", d[d.wave.astype(str).str.contains("W4")], "import_tariff_expect"), ("W1-W3", d[d.wave.astype(str).str.contains("W1|W2|W3")], "short_term_prior")]:
    for y in Y + ["import_loc_post2_1__toward", "import_loc_post2_3__away"]:
        m, n = fit(df, y, [E]); rows.append(dict(sample=sample, y=y, spec="no_agreement", coef=m.params[E], se=m.bse[E], n=n))
        if sample == "W4" and y.startswith("invest_post2"):
            m, n = fit(df, y, [E, "sales_cost_guess1_1"]); rows.append(dict(sample=sample, y=y, spec="no_agreement+expected_sales", coef=m.params[E], se=m.bse[E], n=n))
o = pd.DataFrame(rows); o.to_csv(OUT / "10_slopes_no_agreement_261101.csv", index=False)
pd.set_option("display.width", 160, "display.float_format", lambda x: f"{x:.3f}"); print(o.to_string(index=False))
