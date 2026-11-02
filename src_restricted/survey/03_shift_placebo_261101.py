# ============================================================
# START DATE   : 2026-11-01
# CODE RUN DATE: 2026-11-01
# TIME TAKEN   : ~10 min
# DESCRIPTION  : Date-shift placebo for L1: moves the 28-day Jev statement window by +/-5 to 52 weeks and
#                compares the actual coefficient with the shifted ones (R4 in 00_spec_261101.md). Split from
#                02 after that script stopped on an empty shifted sample.
# INPUT FILES  : same as 01
# OUTPUT FILES : analysis/27_expectation_action_261101/03_shift_placebo_261101.csv (aggregates only)
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
if (OUT / "03_shift_placebo_261101.csv").exists():
    sys.exit("output exists; refusing to overwrite.")

keep = ["responseid", "startdate", "country", "long_term_prior", "short_term_prior", "import_tariff_expect"]
d = pd.read_stata(IA / "inputs/analysis_all.dta", columns=keep, convert_categoricals=False)
d = d[d.country == 1]
a = pd.read_stata(IA / "outputs/alignment_for_stata.dta", convert_categoricals=False)
d = d.merge(a[["responseid", "wave", "industry_code", "emp_2024_bins", "sales_2024_val"]], on="responseid", validate="1:1")
d["date"] = pd.to_datetime(d.startdate, errors="coerce").dt.normalize()
steps = []
with open(REPO / "data/public/china_additional_duty_steps.csv", newline="") as f:
    r = csv.reader(f)
    next(r)
    for row in r:
        steps.append((pd.Timestamp(row[0]), float(row[1])))
steps = pd.DataFrame(steps, columns=["eff", "duty_cn"]).sort_values("eff")
d = pd.merge_asof(d.dropna(subset=["date"]).sort_values("date"), steps, left_on="date", right_on="eff", direction="backward")
for c in ["emp_2024_bins", "sales_2024_val"]:
    d["miss_" + c] = d[c].isna().astype(float)
    d[c] = d[c].fillna(d[c].median())
d["industry_code"] = d["industry_code"].fillna(-1)
early = d.wave.astype(str).str.contains("W1|W2|W3")
w4 = d.wave.astype(str).str.contains("W4")

ev = pd.read_parquet(TS / "event_question_option_probabilities.parquet")
ev["event_date"] = pd.to_datetime(ev["event_date"])
SIG = {"S_path": ("tariff_broad_path", ["higher"], ["lower"]),
       "S_duration": ("tariff_broad_duration", ["permanent"], ["temporary"]),
       "S_unresolved": ("tariff_broad_resolution", ["unresolved_through_2025"], ["resolved_within_month"]),
       "S_path_china": ("tariff_china_path", ["higher"], ["lower"]),
       "S_duration_china": ("tariff_china_duration", ["permanent"], ["temporary"]),
       "S_unresolved_china": ("tariff_china_resolution", ["unresolved_through_2025"], ["resolved_within_month"])}


def parts(q, pos, neg):
    x = ev[ev.question_id == q]
    w = x.pivot_table(index=["event_question_id", "event_date"], columns="option",
                      values="raw_option_probability", aggfunc="sum").fillna(0.0).reset_index()
    w["ev"] = 1.0 - w[x[x.is_absence_option].option.unique()[0]]
    w["num"] = w[pos].sum(axis=1) - w[neg].sum(axis=1)
    return w.groupby("event_date")[["num", "ev"]].sum()


def signal(p, dates, shift=0):
    out = {}
    for t in dates:
        a_, b_ = t + pd.Timedelta(days=shift - 28), t + pd.Timedelta(days=shift - 1)
        s = p.loc[(p.index >= a_) & (p.index <= b_)]
        out[t] = s.num.sum() / s.ev.sum() if s.ev.sum() > 0 else np.nan
    return pd.Series(out)


def beta(df, E, s):
    tmp = df.assign(sig=df.date.map(s)).dropna(subset=[E, "sig"])
    if len(tmp) < 100 or tmp.sig.std() == 0 or tmp.sig.nunique() < 5:
        return np.nan
    tmp["sig"] = (tmp.sig - tmp.sig.mean()) / tmp.sig.std()
    X = pd.concat([tmp[["sig", "emp_2024_bins", "sales_2024_val", "miss_emp_2024_bins", "miss_sales_2024_val", "duty_cn"]].astype(float),
                   pd.get_dummies(tmp.wave.astype(str), prefix="w", drop_first=True).astype(float),
                   pd.get_dummies(tmp.industry_code.astype(int), prefix="i", drop_first=True).astype(float)], axis=1)
    X = sm.add_constant(X.loc[:, X.std() > 0])
    return sm.OLS(tmp[E].astype(float), X).fit().params["sig"]


# ★ CORE LOGIC: shifting keeps the series' autocorrelation but breaks its timing link to responses; the share
# of shifts with |beta| at least the actual |beta| is the p-value. Shifts within +/-4 weeks overlap the real window.
SHIFTS = [k for k in range(-364, 365, 7) if abs(k) >= 35]
rows = []
for sample, mask, E in [("W1-W3", early, "long_term_prior"), ("W1-W3", early, "short_term_prior"), ("W4", w4, "import_tariff_expect")]:
    df = d[mask & d[E].notna()]
    sdates = pd.DatetimeIndex(sorted(df.date.unique()))
    for name, (q, pos, neg) in SIG.items():
        p = parts(q, pos, neg)
        act = beta(df, E, signal(p, sdates, 0))
        sh = np.array([beta(df, E, signal(p, sdates, k)) for k in SHIFTS])
        sh = sh[~np.isnan(sh)]
        rows.append(dict(sample=sample, expectation=E, signal=name, beta_actual=act, shift_mean=sh.mean(), shift_sd=sh.std(),
                         p_two_sided=float((np.abs(sh) >= abs(act)).mean()), n_shifts=len(sh), n=len(df), dates=df.date.nunique()))
        print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(OUT / "03_shift_placebo_261101.csv", index=False)
