# ============================================================
# START DATE   : 2026-11-01
# CODE RUN DATE: 2026-11-01
# TIME TAKEN   : ~10 min
# DESCRIPTION  : Stress-tests the same-owner expectation-plan pattern from 01: current alignment
#                score, executed vs planned actions, importers vs non-importers, party, expected
#                change, and date-shift placebos for the Jev signal (R1-R4 in 00_spec_261101.md).
# INPUT FILES  : same as 01, plus $RESTRICTED_DATA_DIR/survey/trump_alignment_v2/trump_alignment_v2.dta
# OUTPUT FILES : analysis/27_expectation_action_261101/02_robust_261101.csv, 02_shift_placebo_261101.csv
#                (aggregates only; no respondent rows)
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
if (OUT / "02_robust_261101.csv").exists():
    sys.exit("02_robust_261101.csv exists; refusing to overwrite.")

keep = ["responseid", "uid", "startdate", "country", "three_info", "long_term_prior", "short_term_prior",
        "import_tariff_expect", "import_tariff", "invest_post2_1", "prices_post2_1", "invest_post1_1", "prices_post1_1"]
d = pd.read_stata(IA / "inputs/analysis_all.dta", columns=keep, convert_categoricals=False)
d = d[d.country == 1]
a = pd.read_stata(IA / "outputs/alignment_for_stata.dta", convert_categoricals=False)
acols = ["responseid", "wave", "align", "politics_us_group", "importer_dum", "industry_code", "emp_2024_bins", "sales_2024_val",
         "import_loc_post2_1__toward", "import_loc_post2_3__away", "import_loc_post1_1__toward", "import_loc_post1_3__away"]
d = d.merge(a[acols], on="responseid", validate="1:1")
v2 = pd.read_stata(IA / "trump_alignment_v2/trump_alignment_v2.dta", convert_categoricals=False)
d = d.merge(v2[["responseid", "align_trump"]], on="responseid", how="left", validate="1:1")
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
d["party"] = d["politics_us_group"].astype(object).where(d["politics_us_group"].notna(), "missing").astype(str)
d["importer"] = (d.importer_dum > 0).astype(float).where(d.importer_dum.notna())
early = d.wave.astype(str).str.contains("W1|W2|W3")
w4 = d.wave.astype(str).str.contains("W4")


def fit(df, y, main, cluster="uid", party=False):
    s = df.dropna(subset=[y] + main).copy()
    X = s[main + ["emp_2024_bins", "sales_2024_val", "miss_emp_2024_bins", "miss_sales_2024_val", "duty_cn"]].astype(float)
    parts = [X, pd.get_dummies(s.wave.astype(str), prefix="w", drop_first=True).astype(float),
             pd.get_dummies(s.industry_code.astype(int), prefix="i", drop_first=True).astype(float),
             pd.get_dummies(s.three_info.fillna(0).astype(int), prefix="m", drop_first=True).astype(float)]
    if party:
        parts.append(pd.get_dummies(s.party, prefix="p", drop_first=True).astype(float))
    X = pd.concat(parts, axis=1)
    X = sm.add_constant(X.loc[:, X.std() > 0])
    return sm.OLS(s[y].astype(float), X).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(s[cluster])[0]}), s


rows = []


def add(block, sample, y, x, m, n):
    rows.append(dict(block=block, sample=sample, y=y, x=x, coef=m.params[x], se=m.bse[x],
                     ci_low=m.params[x] - 1.96 * m.bse[x], ci_high=m.params[x] + 1.96 * m.bse[x], n=n))


PLAN = ["invest_post2_1", "prices_post2_1", "import_loc_post2_1__toward", "import_loc_post2_3__away"]
DONE = ["invest_post1_1", "prices_post1_1", "import_loc_post1_1__toward", "import_loc_post1_3__away"]
SAMPLES = [("W1-W3", early, "long_term_prior"), ("W1-W3", early, "short_term_prior"), ("W4", w4, "import_tariff_expect")]

for sample, mask, E in SAMPLES:
    df = d[mask]
    for y in PLAN + DONE:
        tag = "planned" if y in PLAN else "done"
        m, s = fit(df, y, [E, "align"])
        add(f"base_{tag}", sample, y, E, m, len(s))
        add(f"base_{tag}", sample, y, "align", m, len(s))
        # R1: current team score, built from three domains with no sourcing, investment, or price items
        if df.dropna(subset=["align_trump", y, E]).shape[0] > 100:
            m, s = fit(df, y, [E, "align_trump"])
            add(f"R1_v2_{tag}", sample, y, E, m, len(s))
            add(f"R1_v2_{tag}", sample, y, "align_trump", m, len(s))
        # party controls, so expectations are not simply partisan optimism
        m, s = fit(df, y, [E, "align"], party=True)
        add(f"party_{tag}", sample, y, E, m, len(s))
        add(f"party_{tag}", sample, y, "align", m, len(s))
    # R3: importers vs non-importers (the tariff environment should bear on input costs mainly for importers)
    if sample == "W1-W3":
        for y in PLAN[:2]:
            for lab, val in [("importers", 1.0), ("non_importers", 0.0)]:
                m, s = fit(df[df.importer == val], y, [E, "align"])
                add(f"R3_{lab}", sample, y, E, m, len(s))

# expected change: hold constant the tariff the owner sees today (W4 only)
for y in PLAN + DONE:
    m, s = fit(d[w4], y, ["import_tariff_expect", "import_tariff", "align"])
    add("expected_change_W4", "W4", y, "import_tariff_expect", m, len(s))
pd.DataFrame(rows).to_csv(OUT / "02_robust_261101.csv", index=False)

# R4: date-shift placebo for L1
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


dates = pd.DatetimeIndex(sorted(d.date.unique()))
# ★ CORE LOGIC: moving the statement window by weeks keeps the autocorrelation of the series but breaks
# the timing link to responses. The share of shifts with |beta| at least the actual |beta| is the p-value.
SHIFTS = [k for k in range(-364, 365, 7) if abs(k) >= 35]
rows = []
for sample, mask, E in SAMPLES:
    df = d[mask]
    sdates = pd.DatetimeIndex(sorted(df.date.unique()))
    for name, (q, pos, neg) in SIG.items():
        p = parts(q, pos, neg)
        betas = {}
        for k in [0] + SHIFTS:
            s = signal(p, sdates, k)
            tmp = df.assign(sig=df.date.map(s))
            tmp["sig"] = (tmp.sig - tmp.sig.mean()) / tmp.sig.std()
            sub = tmp.dropna(subset=[E, "sig"])
            X = pd.concat([sub[["sig", "emp_2024_bins", "sales_2024_val", "miss_emp_2024_bins", "miss_sales_2024_val", "duty_cn"]].astype(float),
                           pd.get_dummies(sub.wave.astype(str), prefix="w", drop_first=True).astype(float),
                           pd.get_dummies(sub.industry_code.astype(int), prefix="i", drop_first=True).astype(float)], axis=1)
            X = sm.add_constant(X.loc[:, X.std() > 0])
            betas[k] = sm.OLS(sub[E].astype(float), X).fit().params["sig"]
        act = betas[0]
        sh = np.array([betas[k] for k in SHIFTS])
        rows.append(dict(sample=sample, expectation=E, signal=name, beta_actual=act, shift_mean=sh.mean(), shift_sd=sh.std(),
                         p_two_sided=float((np.abs(sh) >= abs(act)).mean()), n_shifts=len(sh)))
pd.DataFrame(rows).to_csv(OUT / "02_shift_placebo_261101.csv", index=False)
print("done")
