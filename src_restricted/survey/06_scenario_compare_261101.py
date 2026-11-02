# ============================================================
# START DATE   : 2026-11-01
# CODE RUN DATE: 2026-11-01
# TIME TAKEN   : ~3 min
# DESCRIPTION  : Compares how expectations relate to three versions of the same plan item (already done,
#                planned next 12 months, would do under a 25% across-the-board tariff) and how the randomized
#                message moves each; counts re-respondents linkable from a message to later executed actions.
#                Added after seeing 01-05 results (Codex 19:19 suggestion); exploratory.
# INPUT FILES  : same as 02_robustness_261101.py
# OUTPUT FILES : analysis/27_expectation_action_261101/06_scenario_compare_261101.csv, 06_rerespondents_261101.csv
# ============================================================
import csv, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd, statsmodels.api as sm
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from restricted_paths import REPO, SURVEY_DIR, CUSTOMS_DIR, JEV_DIR, OUTPUT_ROOT, out_dir, fresh_out_dir
warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve()
IA = SURVEY_DIR; OUT = out_dir("survey")
if (OUT / "06_scenario_compare_261101.csv").exists(): sys.exit("output exists; refusing to overwrite.")
Y = [f"{a}_post{k}_1" for a in ("invest", "prices") for k in (1, 2, 3)]
keep = ["responseid", "uid", "startdate", "country", "three_info", "long_term_prior", "short_term_prior", "import_tariff_expect"] + Y
d = pd.read_stata(IA / "inputs/analysis_all.dta", columns=keep, convert_categoricals=False)
d = d[d.country == 1]
a = pd.read_stata(IA / "outputs/alignment_for_stata.dta", convert_categoricals=False)
d = d.merge(a[["responseid", "wave", "align", "industry_code", "emp_2024_bins", "sales_2024_val"]], on="responseid", validate="1:1")
v2 = pd.read_stata(IA / "trump_alignment_v2/trump_alignment_v2.dta", convert_categoricals=False)
d = d.merge(v2[["responseid", "align_trump"]], on="responseid", how="left", validate="1:1")
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
d["msg_esc"] = (d.three_info == 1).astype(float).where(d.three_info.isin([1, 2]))


def fit(df, y, main, msg_fe=True):
    s = df.dropna(subset=[y] + main).copy()
    parts = [s[main + ["emp_2024_bins", "sales_2024_val", "miss_emp_2024_bins", "miss_sales_2024_val", "duty_cn"]].astype(float),
             pd.get_dummies(s.wave.astype(str), prefix="w", drop_first=True).astype(float),
             pd.get_dummies(s.industry_code.astype(int), prefix="i", drop_first=True).astype(float)]
    if msg_fe: parts.append(pd.get_dummies(s.three_info.fillna(0).astype(int), prefix="m", drop_first=True).astype(float))
    X = pd.concat(parts, axis=1); X = sm.add_constant(X.loc[:, X.std() > 0])
    return sm.OLS(s[y].astype(float), X).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(s.uid)[0]}), len(s)


rows = []
early = d[d.wave.astype(str).str.contains("W1|W2|W3")]; w4 = d[d.wave.astype(str).str.contains("W4")]
for sample, df, E in [("W1-W3", early, "short_term_prior"), ("W1-W3", early, "long_term_prior"), ("W4", w4, "import_tariff_expect")]:
    for y in Y:
        m, n = fit(df, y, [E, "align_trump"])
        rows.append(dict(block="slope_expectation", sample=sample, y=y, x=E, coef=m.params[E], se=m.bse[E], n=n))
        # ★ CORE LOGIC: message reduced form (escalate vs soften) on the same three columns, no message FE
        m, n = fit(df, y, ["msg_esc"], msg_fe=False)
        rows.append(dict(block="message_esc_minus_soft", sample=sample, y=y, x="msg_esc", coef=m.params["msg_esc"], se=m.bse["msg_esc"], n=n))
pd.DataFrame(rows).to_csv(OUT / "06_scenario_compare_261101.csv", index=False)

# re-respondents: a message in an earlier response, an executed-action answer in a later response
full = pd.read_stata(IA / "inputs/analysis_all.dta", columns=["responseid", "uid", "startdate", "country", "three_info", "invest_post1_1", "prices_post1_1"], convert_categoricals=False)
full = full[full.country == 1]; full["date"] = pd.to_datetime(full.startdate, errors="coerce")
msg = full[full.three_info.isin([1, 2])][["uid", "date", "three_info"]].rename(columns={"date": "msg_date"})
rr = []
for y in ["invest_post1_1", "prices_post1_1"]:
    later = full[full[y].notna()][["uid", "date"]].merge(msg, on="uid")
    later = later[later.date > later.msg_date + pd.Timedelta(days=14)]
    rr.append(dict(outcome=y, owners=later.uid.nunique(), pairs=len(later), min_gap_days=int((later.date - later.msg_date).dt.days.min()) if len(later) else None,
                   median_gap_days=float((later.date - later.msg_date).dt.days.median()) if len(later) else None))
pd.DataFrame(rr).to_csv(OUT / "06_rerespondents_261101.csv", index=False)
print(pd.DataFrame(rr).to_string(index=False))
