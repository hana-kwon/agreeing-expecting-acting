# ============================================================
# START DATE   : 2026-11-02
# CODE RUN DATE: 2026-11-02
# TIME TAKEN   : ~15 min
# DESCRIPTION  : Shared loaders for step 40 (small-firm pricing: team controls, price timeline, events and
#                leader statements). Loads the U.S. survey in memory only, builds team controls, the China
#                duty step, week keys, Jev leader-signal functions and the Cavallo et al. weekly index.
# INPUT FILES  :PRIVATE_DATA_ROOT/SMB_tariff/ideology_analysis/inputs/analysis_all.dta,
#                .../outputs/alignment_for_stata.dta (read only),
#                data/16_tariff_schedule_261029/china_additional_duty_steps_261029.csv,
#                data/39_retail_prices_statements_261102/Cavallo_Llamas_Vazquez_domestic_imported_trend_line.csv,
#                Jev event_question_option_probabilities.parquet (read only)
# OUTPUT FILES : none (imported by 01-05). Respondent rows are never written.
# ============================================================
import argparse
import csv
import pathlib
import warnings

import numpy as np
import pandas as pd
import scipy.linalg as sla
import statsmodels.api as sm

warnings.filterwarnings("ignore")
HERE = pathlib.Path(__file__).resolve()
TOPIC = HERE.parents[2]                       # topic1_trump_ai, computed from script location
DBX = HERE.parents[7]                         # project root (one level above @HBS)
SHARED = DBX / "PRIVATE_DATA_ROOT"
IA = SHARED / "SMB_tariff/ideology_analysis"
TS = (SHARED / "TEAM_PRIVATE/ideology_share/directional_coding/data/"
      "jev_corpus/simple_time_series_v1")
DATA39 = TOPIC / "data/39_retail_prices_statements_261102"
ANALYSIS = TOPIC / "analysis/40_smb_pricing_controls_261102"
TEAM_CTRL = ["importer_dum", "export_dum", "fcomp_dum"]
BASE_CTRL = ["emp_2024_bins", "sales_2024_val", "miss_emp_2024_bins", "miss_sales_2024_val", "duty_cn"]


def get_out():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ANALYSIS))
    ap.add_argument("--nshift", type=int, default=0, help="0 = all 96 shifts")
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    return out, a


def safe_write(obj, path):
    path = pathlib.Path(path)
    if path.exists():
        raise SystemExit(f"refuse to overwrite {path}")   # ★ IMPORTANT: scripts never overwrite outputs
    if callable(obj):
        obj(path)
    elif isinstance(obj, str):
        path.write_text(obj)
    else:
        obj.to_csv(path, index=False)


def load_survey():
    """U.S. responses, loaded as 02_robustness_261101.py does, plus the team control dummies (rescaled to 0/1)."""
    keep = ["responseid", "uid", "startdate", "country", "three_info", "import_tariff_expect", "short_term_prior",
            "prices_post1_1", "prices_post2_1", "prices_post3_1", "invest_post1_1", "invest_post2_1", "invest_post3_1"]
    d = pd.read_stata(IA / "inputs/analysis_all.dta", columns=keep, convert_categoricals=False)
    d = d[d.country == 1]
    for c in keep[5:]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    a = pd.read_stata(IA / "outputs/alignment_for_stata.dta", convert_categoricals=False)
    acols = ["responseid", "wave", "politics_us_group", "industry_code", "emp_2024_bins", "sales_2024_val",
             "importer_dum", "export_dum", "fcomp_dum", "miss_importer_dum", "miss_export_dum", "miss_fcomp_dum",
             "import_loc_post2_1__toward", "import_loc_post2_3__away"]
    d = d.merge(a[acols], on="responseid", validate="1:1")
    d["date"] = pd.to_datetime(d.startdate, errors="coerce").dt.normalize()
    d = d.dropna(subset=["date"])
    # ★ CORE LOGIC: the survey stores the dummies as 0/100; they are rescaled to 0/1 so a coefficient reads as
    # "importer minus non-importer". alignment_for_stata.dta fills a missing dummy with 0 and sets miss_*=1,
    # so the importer/non-importer contrast must drop miss_*==1 rows (A2, A3) instead of using the filled 0.
    for c in TEAM_CTRL:
        d[c] = d[c] / 100.0
        d["obs_" + c] = (d["miss_" + c] == 0)
    steps = []
    with open(TOPIC / "data/16_tariff_schedule_261029/china_additional_duty_steps_261029.csv", newline="") as f:
        r = csv.reader(f)
        next(r)
        for row in r:
            steps.append((pd.Timestamp(row[0]), float(row[1])))
    steps = pd.DataFrame(steps, columns=["eff", "duty_cn"]).sort_values("eff")
    d = pd.merge_asof(d.sort_values("date"), steps, left_on="date", right_on="eff", direction="backward")
    for c in ["emp_2024_bins", "sales_2024_val"]:
        d["miss_" + c] = d[c].isna().astype(float)
        d[c] = d[c].fillna(d[c].median())
    d["industry_code"] = d["industry_code"].fillna(-1)
    d["party"] = d["politics_us_group"].astype(object).where(d["politics_us_group"].notna(), "missing").astype(str)
    d["week"] = d.date - pd.to_timedelta(d.date.dt.day_of_week, unit="D")
    return d.reset_index(drop=True)


def dummies(s, prefix):
    return pd.get_dummies(s, prefix=prefix, drop_first=True).astype(float)


def ols(df, y, X, cluster="uid"):
    """OLS with uid-clustered SE; constant columns are dropped (e.g. importer_dum among importer-only items)."""
    X = X.loc[:, X.std() > 0]
    X = sm.add_constant(X, has_constant="add")
    m = sm.OLS(df[y].astype(float), X).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(df[cluster])[0]})
    return m


def drop_collinear(X):
    """Rank-revealing pass (pivoted QR) so redundant dummies are removed before clustering."""
    q, r, piv = sla.qr(X.values, mode="economic", pivoting=True)
    tol = abs(r[0, 0]) * max(X.shape) * np.finfo(float).eps * 10
    rank = int((abs(np.diag(r)) > tol).sum())
    return X.iloc[:, sorted(piv[:rank])]


def prep_importer_sample(d):
    """Importer observed only; shutdown answers (-100) set to missing; four outcomes (pp and any-increase)."""
    s = d[d.miss_importer_dum == 0].copy()
    for k, lab in [(1, "done"), (2, "plan")]:
        y = s[f"prices_post{k}_1"].where(s[f"prices_post{k}_1"] > -100)
        s[f"{lab}_pp"] = y
        s[f"{lab}_any"] = (100 * (y > 0)).where(y.notna())
    return s


# ---- Jev leader signal (same definitions as 02_robustness_261101.py lines 115-141) ----
SIG = {"S_path_china": ("tariff_china_path", ["higher"], ["lower"]),
       "S_path_broad": ("tariff_broad_path", ["higher"], ["lower"])}


def load_events():
    ev = pd.read_parquet(TS / "event_question_option_probabilities.parquet")
    ev["event_date"] = pd.to_datetime(ev["event_date"])
    return ev


def parts(ev, q, pos, neg):
    x = ev[ev.question_id == q]
    w = x.pivot_table(index=["event_question_id", "event_date"], columns="option",
                      values="raw_option_probability", aggfunc="sum").fillna(0.0).reset_index()
    w["ev"] = 1.0 - w[x[x.is_absence_option].option.unique()[0]]
    w["num"] = w[pos].sum(axis=1) - w[neg].sum(axis=1)
    return w.groupby("event_date")[["num", "ev"]].sum()


def signal(p, dates, shift=0):
    # ★ CORE LOGIC: 28-day window ending the day before the response date; shift moves the window to break timing.
    out = {}
    for t in dates:
        a_, b_ = t + pd.Timedelta(days=shift - 28), t + pd.Timedelta(days=shift - 1)
        s = p.loc[(p.index >= a_) & (p.index <= b_)]
        out[t] = s.num.sum() / s.ev.sum() if s.ev.sum() > 0 else np.nan
    return pd.Series(out)


def load_cavallo_weekly():
    """100 x [ln(imported) - ln(domestic)], base = mean over 2025-01-06..2025-02-02, averaged by Monday-start week."""
    b = pd.read_csv(DATA39 / "Cavallo_Llamas_Vazquez_domestic_imported_trend_line.csv")
    b["date"] = pd.to_datetime(b["date"], format="%d%b%Y")
    b = b.set_index("date").sort_index()
    y = 100 * (np.log(b.index_imported) - np.log(b.index_domestic))
    y = y - y.loc["2025-01-06":"2025-02-02"].mean()
    wk = y.groupby(y.index - pd.to_timedelta(y.index.day_of_week, unit="D")).agg(["mean", "count"])
    wk.columns = ["retail_imp_minus_dom", "retail_days"]
    return y, wk
