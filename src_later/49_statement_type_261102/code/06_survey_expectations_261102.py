# ============================================================
# START DATE   : 2026-11-02
# CODE RUN DATE: 2026-11-02
# TIME TAKEN   : ~20 min
# DESCRIPTION  : Step 5a. Survey expectations on the typed 28-day statement signals. Same data preparation, same
#                regression (signal z-score + firm size, sales, missing flags, China duty in force, wave and industry dummies)
#                and same date-shift placebo (+/-5 to 52 weeks, never within +/-4 weeks) as code/27 03_shift_placebo_261101.py,
#                with two additions: date-clustered SE (as in code/27 01) and the surprise signal (28-day minus prior 90-day).
#                Respondent data are read from the team folder (read only); only aggregate coefficients are saved.
# INPUT FILES  : SMB_tariff/ideology_analysis/inputs/analysis_all.dta, outputs/alignment_for_stata.dta (read only);
#                data/16_tariff_schedule_261029/china_additional_duty_steps_261029.csv; variants/evq_<variant>.parquet
# OUTPUT FILES : analysis/49_statement_type_261102/06_survey_results_261102.csv (estimate, SE, placebo p; no respondent rows)
# ============================================================
import csv, importlib.util, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd, statsmodels.api as sm
warnings.filterwarnings("ignore")
_s = importlib.util.spec_from_file_location("c", Path(__file__).with_name("common_261102.py"))
C = importlib.util.module_from_spec(_s); _s.loader.exec_module(C)
OUTF = C.OUT / "06_survey_results_261102.csv"
if OUTF.exists():
    sys.exit("output exists; refusing to overwrite.")
IA = C.DB / "PRIVATE_DATA_ROOT/SMB_tariff/ideology_analysis"
VARIANTS = ["base", "thrint", "decision", "specific", "inforce", "future3", "specific_future3"]
keep = ["responseid", "startdate", "country", "long_term_prior", "short_term_prior", "import_tariff_expect"]
d = pd.read_stata(IA / "inputs/analysis_all.dta", columns=keep, convert_categoricals=False)
d = d[d.country == 1]
a = pd.read_stata(IA / "outputs/alignment_for_stata.dta", convert_categoricals=False)
d = d.merge(a[["responseid", "wave", "industry_code", "emp_2024_bins", "sales_2024_val"]], on="responseid", validate="1:1")
d["date"] = pd.to_datetime(d.startdate, errors="coerce").dt.normalize()
steps = []
with open(C.TOPIC / "data/16_tariff_schedule_261029/china_additional_duty_steps_261029.csv", newline="") as f:
    r = csv.reader(f); next(r)
    for row in r:
        steps.append((pd.Timestamp(row[0]), float(row[1])))
steps = pd.DataFrame(steps, columns=["eff", "duty_cn"]).sort_values("eff")
d = pd.merge_asof(d.dropna(subset=["date"]).sort_values("date"), steps, left_on="date", right_on="eff", direction="backward")
for c in ["emp_2024_bins", "sales_2024_val"]:
    d["miss_" + c] = d[c].isna().astype(float); d[c] = d[c].fillna(d[c].median())
d["industry_code"] = d["industry_code"].fillna(-1)
early = d.wave.astype(str).str.contains("W1|W2|W3"); w4 = d.wave.astype(str).str.contains("W4")
DAYS = pd.date_range("2023-06-01", "2026-12-31")


def parts(variant, q):
    ev = pd.read_parquet(C.VARDIR / f"evq_{variant}.parquet")
    x = ev[ev.question_id == q]
    w = x.pivot_table(index=["event_question_id", "event_date"], columns="option", values="raw_option_probability", aggfunc="sum").fillna(0.0).reset_index()
    for o in ["higher", "lower", "not_evidence"]:
        if o not in w: w[o] = 0.0
    w["ev"] = 1.0 - w["not_evidence"]; w["num"] = w["higher"] - w["lower"]
    g = w.groupby("event_date")[["num", "ev"]].sum().reindex(DAYS).fillna(0.0)
    return g.num.cumsum().values, g.ev.cumsum().values


def signal(cum, dates, shift=0, kind="s28"):
    cn, ce = cum
    def win(t, lo, hi):   # sum over [t+shift-lo, t+shift-hi]  (inclusive calendar days)
        i0 = (t + pd.Timedelta(days=shift - lo) - DAYS[0]).days; i1 = (t + pd.Timedelta(days=shift - hi) - DAYS[0]).days
        if i0 < 1 or i1 >= len(DAYS): return np.nan, np.nan
        return cn[i1] - cn[i0 - 1], ce[i1] - ce[i0 - 1]
    out = {}
    for t in dates:
        n1, e1 = win(t, 28, 1)
        s = n1 / e1 if e1 > 0 else np.nan
        if kind == "surprise":
            n2, e2 = win(t, 118, 29)
            s = s - n2 / e2 if (e2 > 0 and np.isfinite(s)) else np.nan
        out[t] = s
    return pd.Series(out)


def beta(df, E, s):
    tmp = df.assign(sig=df.date.map(s)).dropna(subset=[E, "sig"])
    if len(tmp) < 100 or tmp.sig.std() == 0 or tmp.sig.nunique() < 5:
        return np.nan, np.nan, len(tmp)
    tmp["sig"] = (tmp.sig - tmp.sig.mean()) / tmp.sig.std()
    X = pd.concat([tmp[["sig", "emp_2024_bins", "sales_2024_val", "miss_emp_2024_bins", "miss_sales_2024_val", "duty_cn"]].astype(float),
                   pd.get_dummies(tmp.wave.astype(str), prefix="w", drop_first=True).astype(float),
                   pd.get_dummies(tmp.industry_code.astype(int), prefix="i", drop_first=True).astype(float)], axis=1)
    X = sm.add_constant(X.loc[:, X.std() > 0])
    m = sm.OLS(tmp[E].astype(float), X).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(tmp.date)[0]})
    return float(m.params["sig"]), float(m.bse["sig"]), len(tmp)


SHIFTS = [k for k in range(-364, 365, 7) if abs(k) >= 35]   # ★ CORE LOGIC: same shift grid as code/27 03
rows = []
for sample, mask, E in [("W1-W3", early, "long_term_prior"), ("W1-W3", early, "short_term_prior"), ("W4", w4, "import_tariff_expect")]:
    df = d[mask & d[E].notna()]
    sdates = pd.DatetimeIndex(sorted(df.date.unique()))
    for v in VARIANTS:
        for q, qn in [("tariff_broad_path", "broad"), ("tariff_china_path", "china")]:
            cum = parts(v, q)
            for kind in ["s28", "surprise"]:
                act, se, n = beta(df, E, signal(cum, sdates, 0, kind))
                sh = np.array([beta(df, E, signal(cum, sdates, k, kind))[0] for k in SHIFTS]); sh = sh[~np.isnan(sh)]
                p = float((np.abs(sh) >= abs(act)).mean()) if len(sh) and np.isfinite(act) else np.nan
                rows.append(dict(sample=sample, expectation=E, variant=v, question=qn, signal=kind, beta=act, se=se, n=n,
                                 placebo_p=p, n_shifts=len(sh), shift_sd=float(sh.std()) if len(sh) else np.nan))
                print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(OUTF, index=False)
