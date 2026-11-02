# ============================================================
# START DATE   : 2026-11-02
# CODE RUN DATE: 2026-11-02
# TIME TAKEN   : ~5 min
# DESCRIPTION  : E. SEC S1 (expected tariff direction in new filing passages on the 28-day leader signal) with and without firm fixed
#                effects (ticker dummies) next to quarter dummies; date-shift placebo (same 96 shifts) repeated within each specification.
#                parts() and signal() are copied from 35_sec_tariff_expectations_261102/05_analysis_S0_S3.py.
# INPUT FILES  : analysis/35_sec_tariff_expectations_261102/02_panel_261102.csv; Jev event_question_option_probabilities.parquet (read only)
# OUTPUT FILES : analysis/54_review_fixes_261102/06_e_sec_fe_261102.csv
# ============================================================
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd, statsmodels.api as sm
warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve(); TOPIC = HERE.parents[2]
ANA = TOPIC / "analysis/35_sec_tariff_expectations_261102"; OUT = TOPIC / "analysis/54_review_fixes_261102/06_e_sec_fe_261102.csv"
TS = HERE.parents[7] / "PRIVATE_DATA_ROOT/TEAM_PRIVATE/ideology_share/directional_coding/data/jev_corpus/simple_time_series_v1"
if OUT.exists(): sys.exit("refuse to overwrite")
fl = pd.read_csv(ANA / "02_panel_261102.csv", dtype={"accession": str}); fl["filing_date"] = pd.to_datetime(fl.filing_date)
ev = pd.read_parquet(TS / "event_question_option_probabilities.parquet"); ev["event_date"] = pd.to_datetime(ev["event_date"])
SIG = {"S_path": ("tariff_broad_path", ["higher"], ["lower"]), "S_duration": ("tariff_broad_duration", ["permanent"], ["temporary"]),
       "S_unresolved": ("tariff_broad_resolution", ["unresolved_through_2025"], ["resolved_within_month"])}
def parts(q, pos, neg):
    x = ev[ev.question_id == q]
    w = x.pivot_table(index=["event_question_id", "event_date"], columns="option", values="raw_option_probability", aggfunc="sum").fillna(0.0).reset_index()
    w["ev"] = 1.0 - w[x[x.is_absence_option].option.unique()[0]]; w["num"] = w[pos].sum(axis=1) - w[neg].sum(axis=1)
    return w.groupby("event_date")[["num", "ev"]].sum()
def signal(p, dates, shift=0):
    out = {}
    for t in dates:
        a_, b_ = t + pd.Timedelta(days=shift - 28), t + pd.Timedelta(days=shift - 1); s = p.loc[(p.index >= a_) & (p.index <= b_)]
        out[t] = s.num.sum() / s.ev.sum() if s.ev.sum() > 0 else np.nan
    return pd.Series(out)
def beta(df, ycol, sig, fe):
    d = df.assign(sig=df.filing_date.map(sig)).dropna(subset=[ycol, "sig"]).copy(); d["sig"] = (d.sig - d.sig.mean()) / d.sig.std()
    parts_ = [d[["sig"]], pd.get_dummies(d.quarter, prefix="q", drop_first=True).astype(float)]
    if fe: parts_.append(pd.get_dummies(d.ticker, prefix="f", drop_first=True).astype(float))
    X = pd.concat(parts_, axis=1); X = sm.add_constant(X.loc[:, X.std() > 0].astype(float), has_constant="add")
    m = sm.OLS(d[ycol].astype(float), X).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d.ticker)[0]})
    return m.params["sig"], m.bse["sig"], len(d), d.ticker.nunique()
dates = pd.DatetimeIndex(sorted(fl.filing_date.unique())); SH = [k for k in range(-364, 365, 7) if abs(k) >= 35]
rows = []
for ycol in ["Y_S1", "Y_S1_incl_poss"]:
    for name, (q, pos, neg) in SIG.items():
        p = parts(q, pos, neg); sg0 = signal(p, dates, 0); sgs = {k: signal(p, dates, k) for k in SH}
        for fe in (False, True):
            b, se, n, nf = beta(fl, ycol, sg0, fe); sh = np.array([beta(fl, ycol, sgs[k], fe)[0] for k in SH])
            rows.append(dict(y=ycol, signal=name, firm_fe=fe, beta=b, se=se, lo=b - 1.96 * se, hi=b + 1.96 * se, n_filings=n, n_firms=nf, shift_mean=sh.mean(), shift_sd=sh.std(), p_placebo=float((np.abs(sh) >= abs(b)).mean()), n_shifts=len(sh)))
r = pd.DataFrame(rows); r.to_csv(OUT, index=False); pd.set_option("display.width", 200, "display.float_format", lambda x: f"{x:.4f}"); print(r.to_string(index=False))
