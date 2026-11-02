# ============================================================
# START DATE   : 2026-11-02
# CODE RUN DATE: 2026-11-02
# TIME TAKEN   : ~10 s
# DESCRIPTION  : Step 8. Reproduces the main owner-survey estimates on the five synthetic copies alone (no restricted data): the
#                effect of the escalation message on the expected-tariff revision, the same-owner slopes of planned price and
#                planned investment changes on the expectation held before the message (without the agreement score), and the
#                same slopes for the 25 percent scenario. Each copy is estimated separately and the five are combined with the
#                rules for multiply imputed synthetic data (Raghunathan et al. 2003). The copies are built from models of the
#                original survey, so matching the original estimates checks this code and the generator; it does not confirm
#                the original result.
# INPUT FILES  : data/synthetic_survey/synthetic_survey_w4_us_v2_1_copy{1..5}.csv, data/public/china_additional_duty_steps.csv
# OUTPUT FILES : <run>/08_synthetic_survey_estimates.csv (combined estimate, combined SE, per-copy values)
# ============================================================
import csv

import pandas as pd
import statsmodels.api as sm

from common import DATA, run_dir

out = run_dir() / "08_synthetic_survey_estimates.csv"
if out.exists():
    raise SystemExit("output exists; refusing to overwrite.")

COPIES = sorted((DATA / "synthetic_survey").glob("synthetic_survey_w4_us_v2_1_copy*.csv"))
assert len(COPIES) == 5, "expected five synthetic copies"

# additional U.S. duty on China in force on the week start (control in every slope regression)
steps = []
with open(DATA / "public/china_additional_duty_steps.csv", newline="") as f:
    r = csv.reader(f)
    next(r)
    for row in r:
        steps.append((pd.Timestamp(row[0]), float(row[1])))
steps = pd.DataFrame(steps, columns=["eff", "duty_cn"]).sort_values("eff")

def prepare(path):
    """One copy -> analysis frame. A blank employee or sales bin means the value was suppressed (rare background combination)."""
    d = pd.read_csv(path)
    d["date"] = pd.to_datetime(d["week"])     # the copies carry the response week (Monday), not an exact date
    d = pd.merge_asof(d.sort_values("date"), steps, left_on="date", right_on="eff", direction="backward").reset_index(drop=True)
    # controls: employee and sales bin codes (median-filled with a missing flag), industry group, message arm
    for c in ["emp_2024_bin", "sales_2024_bin"]:
        d["miss_" + c] = d[c].isna().astype(float)
        d[c] = d[c].fillna(d[c].median())
    d["msg_esc"] = (d.three_info == 1).astype(float).where(d.three_info.isin([1, 2]))
    # expected-tariff revision: answer after the message minus the answer before it (percentage points)
    d["rev_import_exp"] = d.tariff_p_import_exp - d.import_tariff_expect
    d["rev_import_now"] = d.tariff_p_import_now - d.import_tariff_now
    d["rev_fcomp_now"] = d.tariff_p_fcomp_now - d.fcomp_tariff_now
    d["rev_fcomp_exp"] = d.tariff_p_fcomp_exp - d.fcomp_tariff_expect
    return d


def message_effect(df, y):
    """Escalation minus easing on a revision: OLS on the message dummy with HC1 errors (the message is randomized)."""
    s = df.dropna(subset=[y, "msg_esc"])
    m = sm.OLS(s[y].astype(float), sm.add_constant(s[["msg_esc"]])).fit(cov_type="HC1")
    return m.params["msg_esc"], m.bse["msg_esc"], len(s)


def slope(df, y, x):
    """★ KEY FUNCTION: slope of y on the pre-message expectation x with bin, duty, industry and message-arm controls;
    standard errors clustered on cluster_id (a random grouping with the real cluster-size distribution)."""
    s = df.dropna(subset=[y, x]).copy()
    X = pd.concat([s[[x, "emp_2024_bin", "sales_2024_bin", "miss_emp_2024_bin", "miss_sales_2024_bin", "duty_cn"]].astype(float),
                   pd.get_dummies(s.industry_group.astype(int), prefix="i", drop_first=True).astype(float),
                   pd.get_dummies(s.three_info.fillna(0).astype(int), prefix="m", drop_first=True).astype(float)], axis=1)
    X = sm.add_constant(X.loc[:, X.std() > 0])
    m = sm.OLS(s[y].astype(float), X).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(s.cluster_id)[0]})
    return m.params[x], m.bse[x], len(s)


def estimate(d):
    rows = []
    two = d[d.three_info.isin([1, 2])]
    for y in ["rev_import_exp", "rev_import_now", "rev_fcomp_now", "rev_fcomp_exp"]:
        c, s, n = message_effect(two, y)
        rows.append(dict(estimate=f"message_effect_{y}", coef=c, se=s, n=n))
    for k, lab in ((2, "plan"), (3, "scenario25")):
        for a in ("prices", "invest"):
            c, s, n = slope(d, f"{a}_post{k}_1", "import_tariff_expect")
            rows.append(dict(estimate=f"slope_{a}_{lab}_on_expected_tariff", coef=c, se=s, n=n))
    return pd.DataFrame(rows)


per = []
for k, p in enumerate(COPIES, 1):
    e = estimate(prepare(p))
    e["copy"] = k
    per.append(e)
per = pd.concat(per)
res = []
for name, g in per.groupby("estimate", sort=False):
    q, u = g.coef.values, g.se.values ** 2
    m = len(q)
    qb, ub, b = q.mean(), u.mean(), q.var(ddof=1)
    # ★ CORE LOGIC: Raghunathan et al. (2003): T = (1 + 1/m) b - ubar; when that is negative use ubar + b/m
    Tf = (1 + 1 / m) * b - ub
    T, rule = (Tf, "raghunathan") if Tf > 0 else (ub + b / m, "fallback_ubar_plus_b_over_m")
    row = dict(estimate=name, combined_coef=qb, combined_se=T ** 0.5, combined_rule=rule, between_copy_sd=b ** 0.5)
    for k in range(m):
        row[f"coef_copy{k + 1}"] = q[k]
    for k in range(m):
        row[f"se_copy{k + 1}"] = g.se.values[k]
    for k in range(m):
        row[f"n_copy{k + 1}"] = int(g.n.values[k])
    res.append(row)
res = pd.DataFrame(res)
res.to_csv(out, index=False)
print(res[["estimate", "combined_coef", "combined_se", "combined_rule", "between_copy_sd"]].round(3).to_string(index=False))
