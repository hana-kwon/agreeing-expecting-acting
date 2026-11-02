# ============================================================
# START DATE   : 2026-11-01
# CODE RUN DATE: 2026-11-01
# TIME TAKEN   : ~5 min
# DESCRIPTION  : Links Jev-coded leader statements to owners' tariff expectations and to the
#                same owners' plans (L1-L5 in analysis/27_expectation_action_261101/00_spec_261101.md).
# INPUT FILES  : $RESTRICTED_DATA_DIR/survey/inputs/analysis_all.dta,
#                $RESTRICTED_DATA_DIR/survey/inputs/respondent_waves.csv,
#                $RESTRICTED_DATA_DIR/survey/outputs/alignment_for_stata.dta,
#                $RESTRICTED_DATA_DIR/jev_scores/event_question_option_probabilities.parquet,
#                topic1_trump_ai/data/16_tariff_schedule_261029/china_additional_duty_steps_261029.csv
# OUTPUT FILES : analysis/27_expectation_action_261101/01_signals_by_date_261101.csv,
#                01_results_261101.csv, 01_sample_261101.csv (aggregates only; no respondent rows)
# ============================================================
import csv
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from restricted_paths import REPO, SURVEY_DIR, CUSTOMS_DIR, JEV_DIR, OUTPUT_ROOT, out_dir, fresh_out_dir

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve()
IA = SURVEY_DIR
TS = JEV_DIR
OUT = out_dir("survey")
OUT.mkdir(parents=True, exist_ok=True)
if (OUT / "01_results_261101.csv").exists():
    sys.exit("01_results_261101.csv exists; refusing to overwrite. Save a new version instead.")

# ---------- AI signals ----------
ev = pd.read_parquet(TS / "event_question_option_probabilities.parquet")
ev["event_date"] = pd.to_datetime(ev["event_date"])

SIGNALS = {
    "S_path": ("tariff_broad_path", ["higher"], ["lower"]),
    "S_duration": ("tariff_broad_duration", ["permanent"], ["temporary"]),
    "S_unresolved": ("tariff_broad_resolution", ["unresolved_through_2025"], ["resolved_within_month"]),
    "S_path_china": ("tariff_china_path", ["higher"], ["lower"]),
    "S_duration_china": ("tariff_china_duration", ["permanent"], ["temporary"]),
    "S_unresolved_china": ("tariff_china_resolution", ["unresolved_through_2025"], ["resolved_within_month"]),
}


def daily_parts(qid, pos, neg):
    """Event-level numerator and evidence mass by date for one question."""
    x = ev[ev.question_id == qid]
    w = x.pivot_table(index=["event_question_id", "event_date"], columns="option",
                      values="raw_option_probability", aggfunc="sum").fillna(0.0).reset_index()
    absent = x[x.is_absence_option].option.unique()[0]
    w["ev"] = 1.0 - w[absent]
    w["num"] = w[pos].sum(axis=1) - w[neg].sum(axis=1)
    return w.groupby("event_date")[["num", "ev"]].sum()


def window_signal(parts, dates, lead=False, days=28):
    # ★ CORE LOGIC: statements strictly before the response day (no look-ahead); the lead version
    # uses the 28 days strictly after the response day and serves only as a placebo.
    rows = []
    for d in dates:
        if lead:
            a, b = d + pd.Timedelta(days=1), d + pd.Timedelta(days=days)
        else:
            a, b = d - pd.Timedelta(days=days), d - pd.Timedelta(days=1)
        s = parts.loc[(parts.index >= a) & (parts.index <= b)]
        rows.append((d, s.num.sum() / s.ev.sum() if s.ev.sum() > 0 else np.nan, s.ev.sum()))
    return pd.DataFrame(rows, columns=["date", "value", "evidence"]).set_index("date")


# ---------- survey (kept in memory only) ----------
keep = ["responseid", "uid", "startdate", "country", "three_info", "long_term_prior", "short_term_prior",
        "long_term_post", "short_term_post", "import_tariff_expect", "tariff_p_import_exp",
        "invest_post2_1", "prices_post2_1"]
d = pd.read_stata(IA / "inputs/analysis_all.dta", columns=keep, convert_categoricals=False)
d = d[d.country == 1].copy()
a = pd.read_stata(IA / "outputs/alignment_for_stata.dta", convert_categoricals=False)
acols = ["responseid", "wave", "align", "industry_code", "emp_2024_bins", "sales_2024_val",
         "import_loc_post2_1__toward", "import_loc_post2_1__net", "import_loc_post2_3__away",
         "import_loc_post2_3__net", "import_loc_post1_1__toward", "import_loc_post1_3__away"]
d = d.merge(a[acols], on="responseid", how="inner", validate="1:1")
d["date"] = pd.to_datetime(d.startdate, errors="coerce").dt.normalize()
d = d[d.date.notna()].copy()
wave_names = {1: "W0", 2: "W1", 3: "W2", 4: "W3", 5: "W4", 6: "W5"}
print("wave raw values:", sorted(d.wave.dropna().unique())[:10], file=sys.stderr)

# China additional duty in force on the response date (rows have unquoted extra fields, so read by position)
steps = []
with open(REPO / "data/public/china_additional_duty_steps.csv", newline="") as f:
    r = csv.reader(f)
    next(r)
    for row in r:
        steps.append((pd.Timestamp(row[0]), float(row[1])))
steps = pd.DataFrame(steps, columns=["eff", "rate"]).sort_values("eff")
d = pd.merge_asof(d.sort_values("date"), steps, left_on="date", right_on="eff", direction="backward")
d["duty_cn"] = d["rate"]

dates = pd.DatetimeIndex(sorted(d.date.unique()))
sig = pd.DataFrame(index=dates)
for name, (q, pos, neg) in SIGNALS.items():
    parts = daily_parts(q, pos, neg)
    sig[name] = window_signal(parts, dates)["value"]
    sig[name + "_ev"] = window_signal(parts, dates)["evidence"]
    sig[name + "_lead"] = window_signal(parts, dates, lead=True)["value"]
sig.rename_axis("date").reset_index().to_csv(OUT / "01_signals_by_date_261101.csv", index=False)
d = d.merge(sig.reset_index().rename(columns={"index": "date"}), on="date", how="left")
for name in SIGNALS:
    # ★ IMPORTANT: standardized over respondents so coefficients read per 1 SD; lead uses the same scale.
    mu, sd = d[name].mean(), d[name].std()
    d[name + "_z"] = (d[name] - mu) / sd
    d[name + "_lead_z"] = (d[name + "_lead"] - mu) / sd

# ---------- estimation helpers ----------
for c in ["emp_2024_bins", "sales_2024_val"]:
    d["miss_" + c] = d[c].isna().astype(float)
    d[c] = d[c].fillna(d[c].median())
d["industry_code"] = d["industry_code"].fillna(-1)


def design(df, main, msg_fe=False):
    X = df[main + ["emp_2024_bins", "sales_2024_val", "miss_emp_2024_bins", "miss_sales_2024_val", "duty_cn"]].astype(float)
    X = pd.concat([X, pd.get_dummies(df["wave"].astype(str), prefix="w", drop_first=True).astype(float),
                   pd.get_dummies(df["industry_code"].astype(int), prefix="ind", drop_first=True).astype(float)], axis=1)
    if msg_fe:
        X = pd.concat([X, pd.get_dummies(df["three_info"].fillna(0).astype(int), prefix="msg", drop_first=True).astype(float)], axis=1)
    X = X.loc[:, X.std() > 0]
    return sm.add_constant(X)


def ols(df, y, main, cluster, msg_fe=False):
    s = df.dropna(subset=[y] + main).copy()
    if len(s) < 50:
        return None
    X = design(s, main, msg_fe)
    m = sm.OLS(s[y].astype(float), X).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(s[cluster])[0]})
    return m, s


def tsls(df, y, endog, instruments, cluster, exog=()):
    # ★ KEY FUNCTION: 2SLS by hand with cluster-robust second-stage SE; the first-stage F is the
    # cluster-robust joint test on the excluded instruments.
    s = df.dropna(subset=[y, endog] + list(instruments) + list(exog)).copy()
    if len(s) < 50:
        return None
    X0 = design(s, list(exog))
    Z = pd.concat([X0, s[list(instruments)].astype(float)], axis=1)
    Z = Z.loc[:, Z.std() > 0].assign(const=1.0)
    groups = pd.factorize(s[cluster])[0]
    fs = sm.OLS(s[endog].astype(float), Z).fit(cov_type="cluster", cov_kwds={"groups": groups})
    inst_kept = [c for c in instruments if c in Z.columns]
    F = float(fs.f_test(np.eye(len(Z.columns))[[list(Z.columns).index(c) for c in inst_kept]]).fvalue)
    Ehat = fs.fittedvalues
    X2 = X0.copy()
    X2[endog] = Ehat
    b = np.linalg.pinv(X2.values.T @ X2.values) @ (X2.values.T @ s[y].astype(float).values)
    X_actual = X0.copy()
    X_actual[endog] = s[endog].astype(float).values
    resid = s[y].astype(float).values - X_actual.values @ b
    XtXi = np.linalg.pinv(X2.values.T @ X2.values)
    meat = np.zeros((X2.shape[1], X2.shape[1]))
    for g in np.unique(groups):
        idx = groups == g
        u = X2.values[idx].T @ resid[idx]
        meat += np.outer(u, u)
    V = XtXi @ meat @ XtXi
    j = list(X2.columns).index(endog)
    return b[j], float(np.sqrt(V[j, j])), F, len(s), s[cluster].nunique()


res = []


def add(link, sample, y, x, coef, se, n, nclust, extra=""):
    res.append(dict(link=link, sample=sample, y=y, x=x, coef=coef, se=se,
                    ci_low=coef - 1.96 * se, ci_high=coef + 1.96 * se, n=n, clusters=nclust, note=extra))


early = d[d.wave.astype(str).str.contains("W1|W2|W3|copy2|apr7", regex=True)]
w4 = d[d.wave.astype(str).str.contains("W4|may28", regex=True)]
print("rows early", len(early), "w4", len(w4), file=sys.stderr)
pd.DataFrame({
    "sample": ["all_us_merged", "W1-W3", "W4"], "responses": [len(d), len(early), len(w4)],
    "response_dates": [d.date.nunique(), early.date.nunique(), w4.date.nunique()],
    "first_date": [d.date.min(), early.date.min(), w4.date.min()],
    "last_date": [d.date.max(), early.date.max(), w4.date.max()],
}).to_csv(OUT / "01_sample_261101.csv", index=False)

PLANS = ["import_loc_post2_1__toward", "import_loc_post2_1__net", "import_loc_post2_3__away",
         "import_loc_post2_3__net", "invest_post2_1", "prices_post2_1"]
DONE = ["import_loc_post1_1__toward", "import_loc_post1_3__away"]
SZ = ["S_path_z", "S_duration_z", "S_unresolved_z"]
SZC = ["S_path_china_z", "S_duration_china_z", "S_unresolved_china_z"]

# L1: AI signal -> expectation
for sample, df, Es in [("W1-W3", early, ["long_term_prior", "short_term_prior"]), ("W4", w4, ["import_tariff_expect"])]:
    for E in Es:
        for s in SZ + SZC:
            r = ols(df, E, [s], "date")
            if r:
                add("L1", sample, E, s, r[0].params[s], r[0].bse[s], int(r[0].nobs), r[1].date.nunique())
            lead = s.replace("_z", "_lead_z")
            r = ols(df, E, [s, lead], "date")
            if r:
                add("L1_placebo_lead", sample, E, lead, r[0].params[lead], r[0].bse[lead], int(r[0].nobs), r[1].date.nunique())

# L2: expectation -> plan within owner
d["exp_high"] = (d.long_term_prior == 100).astype(float).where(d.long_term_prior.notna())
d["exp_low"] = (d.long_term_prior == -100).astype(float).where(d.long_term_prior.notna())
early = d[d.wave.astype(str).str.contains("W1|W2|W3|copy2|apr7", regex=True)]
w4 = d[d.wave.astype(str).str.contains("W4|may28", regex=True)]
for y in PLANS + DONE:
    for sample, df, E in [("W1-W3", early, "long_term_prior"), ("W1-W3", early, "short_term_prior"), ("W4", w4, "import_tariff_expect")]:
        r = ols(df, y, [E], "uid", msg_fe=True)
        if r:
            add("L2_noalign", sample, y, E, r[0].params[E], r[0].bse[E], int(r[0].nobs), r[1].uid.nunique())
        r = ols(df, y, [E, "align"], "uid", msg_fe=True)
        if r:
            add("L2_align", sample, y, E, r[0].params[E], r[0].bse[E], int(r[0].nobs), r[1].uid.nunique())
            add("L2_align", sample, y, "align", r[0].params["align"], r[0].bse["align"], int(r[0].nobs), r[1].uid.nunique())
    r = ols(early, y, ["exp_high", "exp_low", "align"], "uid", msg_fe=True)
    if r:
        for v in ["exp_high", "exp_low"]:
            add("L2_categories(base=uncertain)", "W1-W3", y, v, r[0].params[v], r[0].bse[v], int(r[0].nobs), r[1].uid.nunique())

# L3: randomized message IV (three_info: 1 escalate, 2 soften, 3 uncertainty)
for df in (d,):
    d["msg_esc"] = (d.three_info == 1).astype(float).where(d.three_info.notna())
    d["msg_unc"] = (d.three_info == 3).astype(float).where(d.three_info.notna())
early = d[d.wave.astype(str).str.contains("W1|W2|W3|copy2|apr7", regex=True)]
w4 = d[d.wave.astype(str).str.contains("W4|may28", regex=True)]
for sample, df, Epost, Eprior in [("W1-W3", early, "long_term_post", "long_term_prior"),
                                  ("W1-W3", early, "short_term_post", "short_term_prior"),
                                  ("W4", w4, "tariff_p_import_exp", "import_tariff_expect")]:
    for y in PLANS:
        r = tsls(df, y, Epost, ["msg_esc", "msg_unc"], "uid", exog=[Eprior])
        if r:
            add("L3_IV_message", sample, y, Epost, r[0], r[1], r[3], r[4], f"first_stage_F={r[2]:.1f}")

# L4: AI signal -> plan (reduced form), with lead placebo
for y in PLANS + DONE:
    for sample, df in [("W1-W3", early), ("W4", w4), ("all", d)]:
        for s in SZ:
            r = ols(df, y, [s], "date")
            if r:
                add("L4", sample, y, s, r[0].params[s], r[0].bse[s], int(r[0].nobs), r[1].date.nunique())
            lead = s.replace("_z", "_lead_z")
            r = ols(df, y, [s, lead], "date")
            if r:
                add("L4_placebo_lead", sample, y, lead, r[0].params[lead], r[0].bse[lead], int(r[0].nobs), r[1].date.nunique())

# L5: AI signal as instrument for expectation
for sample, df, E in [("W1-W3", early, "long_term_prior"), ("W1-W3", early, "short_term_prior"), ("W4", w4, "import_tariff_expect")]:
    for y in PLANS:
        r = tsls(df, y, E, SZ, "date")
        if r:
            add("L5_IV_signal", sample, y, E, r[0], r[1], r[3], r[4], f"first_stage_F={r[2]:.1f}")

pd.DataFrame(res).to_csv(OUT / "01_results_261101.csv", index=False)
print("rows", len(res))
