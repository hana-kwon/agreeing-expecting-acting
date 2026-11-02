# ============================================================
# START DATE   : 2026-11-02
# CODE RUN DATE: 2026-11-02
# TIME TAKEN   : ~15 min
# DESCRIPTION  : Market response to the AI-coded tariff statements (spec analysis/45_market_response_261102/
#                00_spec_261102.md): M1 escalation days vs other days, M2 acts vs words-only and legal-act
#                announcement days, M3 daily-signal regression, date-shift placebo (+-5..52 weeks), and the
#                2025-04-09 known case. Two-day event window (close of the previous trading day to close of
#                the next trading day) because statements carry a date but no time.
# INPUT FILES  : data/45_market_response_261102/{fred_SP500,fred_VIXCLS,yahoo_*}_261102.csv;
PRIVATE_DATA_ROOT/TEAM_PRIVATE/.../simple_time_series_v1/event_question_option_probabilities.parquet (read only);
#                ~/research_local/orgsci_repos/agreeing_expecting_acting/data/public/tariff_calendar/corrected/02_increase_events_B_extended.csv;
#                data/16_tariff_schedule_261029/china_additional_duty_steps_261029.csv
# OUTPUT FILES : analysis/45_market_response_261102/02_{m1,m2,m3,known_case,signal_days,placebo_draws}_261102.csv
#                (aggregates by trading day only; no firm or respondent rows)
# ============================================================
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve()
TOPIC = HERE.parents[2]
SHARED = HERE.parents[7] / "PRIVATE_DATA_ROOT"
TS = (SHARED / "TEAM_PRIVATE/ideology_share/directional_coding/data/"
      "jev_corpus/simple_time_series_v1/event_question_option_probabilities.parquet")
CAL = Path.home() / "research_local/orgsci_repos/agreeing_expecting_acting/data/public/tariff_calendar/corrected/02_increase_events_B_extended.csv"
CHN = TOPIC / "data/16_tariff_schedule_261029/china_additional_duty_steps_261029.csv"
DATA = TOPIC / "data/45_market_response_261102"
OUT = TOPIC / "analysis/45_market_response_261102"
if (OUT / "02_m1_261102.csv").exists():
    sys.exit("02_m1_261102.csv exists; refusing to overwrite.")
assert TS.exists() and CAL.exists() and CHN.exists()

SAMPLE0, SAMPLE1 = pd.Timestamp("2024-01-02"), pd.Timestamp("2026-09-24")
EXCL = (pd.Timestamp("2025-03-31"), pd.Timestamp("2025-04-30"))  # robustness: drop the April 2025 volatility episode
MAXLAGS = 5
SHIFT_DAYS = [7 * w for w in list(range(-52, -4)) + list(range(5, 53))]  # 96 shifts, +-5..52 weeks
ETFS = ["SPY", "XRT", "FXI", "MCHI"]

# ---------------- prices
px = {}
for tk in ETFS:
    d = pd.read_csv(DATA / f"yahoo_{tk}_261102.csv", parse_dates=["date"]).set_index("date")
    px[tk] = d.adjclose
cal0 = px["SPY"].index
for tk in ETFS:
    assert px[tk].index.equals(cal0), f"{tk} calendar differs from SPY"
vix = pd.read_csv(DATA / "fred_VIXCLS_261102.csv", na_values=["."], parse_dates=["observation_date"]).set_index("observation_date").VIXCLS
spx = pd.read_csv(DATA / "fred_SP500_261102.csv", na_values=["."], parse_dates=["observation_date"]).set_index("observation_date").SP500
T = cal0.values  # master trading calendar (SPY)
N = len(T)
vixa = vix.reindex(cal0)

Y = {}  # (market, outcome) -> array on master calendar, nan where the window is not available
for tk in ETFS:
    lp = np.log(px[tk].values.astype(float))
    r0 = np.full(N, np.nan); r0[1:] = 100 * (lp[1:] - lp[:-1])
    r1 = np.full(N, np.nan); r1[:-1] = r0[1:]
    r2 = np.full(N, np.nan); r2[1:-1] = 100 * (lp[2:] - lp[:-2])
    sd = pd.Series(r0).shift(1).rolling(20, min_periods=15).std().values  # std of daily returns up to t-1
    Y[(tk, "R2")] = r2
    Y[(tk, "abs")] = np.abs(r2)
    Y[(tk, "absz")] = np.abs(r2) / (sd * np.sqrt(2))
    Y[(tk, "R0")] = r0
    Y[(tk, "R1")] = r1
v = vixa.values.astype(float)
dv2 = np.full(N, np.nan); dv2[1:-1] = v[2:] - v[:-2]
Y[("VIX", "dV2")] = dv2
OUTCOMES = list(Y.keys())

sample_idx = np.where((T >= np.datetime64(SAMPLE0)) & (T <= np.datetime64(SAMPLE1)))[0]
sdates = pd.DatetimeIndex(T[sample_idx])
ns = len(sample_idx)
dow = pd.get_dummies(sdates.dayofweek).reindex(columns=range(5), fill_value=0).astype(float).values[:, 1:]  # Tue-Fri dummies
week_id = (sdates.isocalendar().year.astype(int) * 100 + sdates.isocalendar().week.astype(int)).values
keep_all = np.ones(ns, bool)
keep_excl = ~((sdates >= EXCL[0]) & (sdates <= EXCL[1]))

# ---------------- statement signal
ev = pd.read_parquet(TS)
ev["event_date"] = pd.to_datetime(ev["event_date"])


def parts(q):
    # same construction as 27_expectation_action_261101/02_robustness_261101.py parts(), lines 115-141:
    # num = P(higher) - P(lower); ev = 1 - P(not_evidence); summed by statement date
    x = ev[ev.question_id == q]
    w = x.pivot_table(index=["event_question_id", "event_date"], columns="option",
                      values="raw_option_probability", aggfunc="sum").fillna(0.0).reset_index()
    w["ev"] = 1.0 - w[x[x.is_absence_option].option.unique()[0]]
    w["num"] = w["higher"] - w["lower"]
    return w.groupby("event_date")[["num", "ev"]].sum()


PARTS = {"china": parts("tariff_china_path"), "broad": parts("tariff_broad_path")}

# ---------------- rate-change calendars
cb = pd.read_csv(CAL)
cb = cb[(cb.cancelled == False) & (cb.short_lived == False)]  # noqa: E712
BROAD_EFF = pd.DatetimeIndex(sorted(pd.to_datetime(cb.eff).unique()))
BROAD_ANN = pd.DatetimeIndex(sorted(pd.to_datetime(cb.known).unique()))
st = pd.read_csv(CHN, parse_dates=["effective_date", "announcement_date"])
inc = st[st.change_points > 0]
dec = st[st.change_points < 0]
CHN_EFF = pd.DatetimeIndex(sorted(inc.effective_date))
CHN_ANN_INC = pd.DatetimeIndex(sorted(inc.announcement_date.dropna()))
CHN_ANN_DEC = pd.DatetimeIndex(sorted(dec.announcement_date.dropna()))
EFF = {"china": CHN_EFF, "broad": BROAD_EFF}
ANN = {"broad_inc": BROAD_ANN, "china_inc": CHN_ANN_INC, "china_dec": CHN_ANN_DEC}


def to_idx(dates, shift):
    """Trading-day index of the first trading day on/after (date + shift); -1 if outside the calendar."""
    d = (pd.DatetimeIndex(dates) + pd.Timedelta(days=shift)).values
    j = np.searchsorted(T, d, side="left")
    j = np.where((j >= N) | (d < T[0]), -1, j)
    return j


def build(sig, shift, keep):
    """Signal objects for one shift. All masks are over the ns sample trading days."""
    p = PARTS[sig]
    j = to_idx(p.index, shift)
    g = pd.DataFrame({"j": j, "num": p.num.values, "ev": p.ev.values})
    g = g[g.j >= 0].groupby("j")[["num", "ev"]].sum()
    S = np.full(N, np.nan)
    ok = g.ev > 0
    S[g.index[ok]] = (g.num[ok] / g.ev[ok]).values
    S = S[sample_idx]
    evid = ~np.isnan(S) & keep
    E = np.zeros(ns, bool)
    if evid.sum() >= 10:
        thr = np.quantile(S[evid], 0.9)
        E = evid & (S >= thr) & (S > 0)
    # acts: an implemented increase (effective date) within (t, t+30] calendar days; the rate calendar moves
    # with the shifted signal, i.e. rate dates + shift
    effs = (EFF[sig] + pd.Timedelta(days=shift)).values
    acts = np.zeros(ns, bool)
    for i in np.where(E)[0]:
        t0 = sdates[i].to_datetime64()
        acts[i] = bool(((effs > t0) & (effs <= t0 + np.timedelta64(30, "D"))).any())
    words = E & ~acts
    OE = evid & ~E
    z = np.zeros(ns)
    if evid.sum() > 2:
        z[evid] = (S[evid] - S[evid].mean()) / S[evid].std(ddof=1)
    anns = {}
    for k, dts in ANN.items():
        a = np.zeros(ns, bool)
        jj = to_idx(dts, shift)
        pos = {int(x): i for i, x in enumerate(sample_idx)}
        for x in jj[jj >= 0]:
            if int(x) in pos:
                a[pos[int(x)]] = True
        anns[k] = a
    return dict(S=S, evid=evid, E=E, acts=acts & E, words=words, OE=OE, z=z, anns=anns, thr=np.nan if not E.any() else S[E].min())


def mdiff(y, a, b, keep):
    ok = ~np.isnan(y) & keep
    aa, bb = a & ok, b & ok
    if aa.sum() == 0 or bb.sum() == 0:
        return np.nan
    return y[aa].mean() - y[bb].mean()


def m3beta(y, sg, keep, evid_only=False):
    ok = ~np.isnan(y) & keep
    if evid_only:
        ok &= sg["evid"]
        X = np.column_stack([np.ones(ns), sg["z"], dow])
    else:
        X = np.column_stack([np.ones(ns), sg["z"], sg["evid"].astype(float), dow])
    if ok.sum() < 30:
        return np.nan
    b = np.linalg.lstsq(X[ok], y[ok], rcond=None)[0]
    return b[1]


def stats_all(sg, keep):
    """Point statistics used both for the actual data and for each placebo shift."""
    out = {}
    for oc in OUTCOMES:
        y = Y[oc][sample_idx]
        out[("M1_evid", oc)] = mdiff(y, sg["E"], sg["OE"], keep)
        out[("M1_all", oc)] = mdiff(y, sg["E"], ~sg["E"], keep)
        out[("M2_acts", oc)] = mdiff(y, sg["acts"], sg["OE"], keep)
        out[("M2_words", oc)] = mdiff(y, sg["words"], sg["OE"], keep)
        out[("M2_acts_minus_words", oc)] = mdiff(y, sg["acts"], sg["words"], keep)
        for k, a in sg["anns"].items():
            out[(f"ANN_{k}", oc)] = mdiff(y, a & keep, ~a & keep, keep)
        out[("M3", oc)] = m3beta(y, sg, keep)
        out[("M3_evidonly", oc)] = m3beta(y, sg, keep, True)
    return out


def hac_fit(y, X, cols, contrasts, cluster=None):
    """OLS with Newey-West (Bartlett, 5 lags) SEs, or week-clustered SEs; returns {contrast: (est, se, p)}."""
    ok = ~np.isnan(y) & np.isfinite(X).all(axis=1)
    ncol = X.shape[1]
    X = X[ok]; yy = y[ok]
    keepc = [c for c in range(X.shape[1]) if c == 0 or np.ptp(X[:, c]) > 0]
    Xk = X[:, keepc]
    if len(yy) < 30 or Xk.shape[1] < 2:
        return {k: (np.nan, np.nan, np.nan) for k in contrasts}
    if cluster is None:
        m = sm.OLS(yy, Xk).fit(cov_type="HAC", cov_kwds={"maxlags": MAXLAGS})
    else:
        m = sm.OLS(yy, Xk).fit(cov_type="cluster", cov_kwds={"groups": cluster[ok]})
    res = {}
    for name, r in contrasts.items():
        r = np.concatenate([np.array(r, float), np.zeros(ncol - len(r))])
        if (r[[c for c in range(len(r)) if c not in keepc]] != 0).any():
            res[name] = (np.nan, np.nan, np.nan)
            continue
        rr = r[keepc]
        tt = m.t_test(rr)
        res[name] = (float(np.squeeze(tt.effect)), float(np.squeeze(tt.sd)), float(np.squeeze(tt.pvalue)))
    return res


def actual_models(sg, keep):
    rows = []
    kp = keep
    for oc in OUTCOMES:
        y = np.where(kp, Y[oc][sample_idx], np.nan)
        one = np.ones(ns)
        E, OE, A, W = sg["E"].astype(float), sg["OE"].astype(float), sg["acts"].astype(float), sg["words"].astype(float)
        r = hac_fit(y, np.column_stack([one, E, OE]), None, {"M1_evid": [0, 1, -1]})
        rows.append(("M1_evid", oc) + r["M1_evid"])
        r = hac_fit(y, np.column_stack([one, E]), None, {"M1_all": [0, 1]})
        rows.append(("M1_all", oc) + r["M1_all"])
        r = hac_fit(y, np.column_stack([one, A, W, OE]), None,
                    {"M2_acts": [0, 1, 0, -1], "M2_words": [0, 0, 1, -1], "M2_acts_minus_words": [0, 1, -1, 0]})
        for k in r:
            rows.append((k, oc) + r[k])
        for k, a in sg["anns"].items():
            a = (a & kp).astype(float)
            r = hac_fit(y, np.column_stack([one, a]), None, {f"ANN_{k}": [0, 1]})
            rows.append((f"ANN_{k}", oc) + r[f"ANN_{k}"])
        r = hac_fit(y, np.column_stack([one, sg["z"], sg["evid"].astype(float), dow]), None, {"M3": [0, 1]})
        rows.append(("M3", oc) + r["M3"])
        evm = sg["evid"]
        yy = np.where(evm, y, np.nan)
        r = hac_fit(yy, np.column_stack([one, sg["z"], dow]), None, {"M3_evidonly": [0, 1]}, cluster=week_id)
        rows.append(("M3_evidonly", oc) + r["M3_evidonly"])
    return rows


def counts(sg, keep):
    ok = keep
    return dict(n_days=int(ok.sum()), n_evid=int(sg["evid"].sum()), n_E=int(sg["E"].sum()), n_OE=int(sg["OE"].sum()),
                n_acts=int(sg["acts"].sum()), n_words=int(sg["words"].sum()),
                n_ann_broad_inc=int((sg["anns"]["broad_inc"] & ok).sum()),
                n_ann_china_inc=int((sg["anns"]["china_inc"] & ok).sum()),
                n_ann_china_dec=int((sg["anns"]["china_dec"] & ok).sum()),
                thr_S=float(sg["thr"]))


# ---------------- run: actual and placebo
rows, draws, cnt_rows = [], [], []
for variant, keep in [("all", keep_all), ("excl_2025-04", keep_excl)]:
    for sig in ["china", "broad"]:
        sg0 = build(sig, 0, keep)
        cnt_rows.append(dict(variant=variant, signal=sig, **counts(sg0, keep)))
        act = actual_models(sg0, keep)
        pl = {}
        for k in SHIFT_DAYS:
            sg = build(sig, k, keep)
            for key, val in stats_all(sg, keep).items():
                pl.setdefault(key, []).append(val)
                draws.append((variant, sig, key[0], key[1][0], key[1][1], k // 7, val))
        for test, oc, est, se, p in act:
            arr = np.array(pl[(test, oc)], float)
            arr = arr[~np.isnan(arr)]
            rows.append(dict(variant=variant, signal=sig, test=test, market=oc[0], outcome=oc[1], estimate=est, se=se,
                             p_hac=p, placebo_n=len(arr),
                             placebo_mean=arr.mean() if len(arr) else np.nan, placebo_sd=arr.std(ddof=1) if len(arr) > 1 else np.nan,
                             placebo_p=(np.abs(arr) >= abs(est)).mean() if len(arr) and not np.isnan(est) else np.nan))
        # daily signal file (actual, all variant only once)
        if variant == "all":
            df = pd.DataFrame({"date": sdates, "S": sg0["S"], "evid": sg0["evid"], "escalation": sg0["E"],
                               "acts": sg0["acts"], "words_only": sg0["words"],
                               "ann_broad_inc": sg0["anns"]["broad_inc"], "ann_china_inc": sg0["anns"]["china_inc"],
                               "ann_china_dec": sg0["anns"]["china_dec"]})
            df.to_csv(OUT / f"02_signal_days_{sig}_261102.csv", index=False)

res = pd.DataFrame(rows)
res["n_E"] = res.apply(lambda r: next(c["n_E"] for c in cnt_rows if c["variant"] == r.variant and c["signal"] == r.signal), axis=1)
res[res.test.str.startswith("M1")].to_csv(OUT / "02_m1_261102.csv", index=False)
res[res.test.str.startswith("M2") | res.test.str.startswith("ANN")].to_csv(OUT / "02_m2_261102.csv", index=False)
res[res.test.str.startswith("M3")].to_csv(OUT / "02_m3_261102.csv", index=False)
pd.DataFrame(cnt_rows).to_csv(OUT / "02_counts_261102.csv", index=False)
pd.DataFrame(draws, columns=["variant", "signal", "test", "market", "outcome", "shift_weeks", "stat"]).to_csv(
    OUT / "02_placebo_draws_261102.csv", index=False)

# ---------------- known case: 2025-04-09
kc = []
i9 = int(np.where(T == np.datetime64("2025-04-09"))[0][0])
for tk in ETFS:
    for lab, i in [("2025-04-08", i9 - 1), ("2025-04-09", i9), ("2025-04-10", i9 + 1)]:
        kc.append(dict(item=f"{tk} daily return % (adjclose, close-to-close)", day=lab, value=Y[(tk, 'R0')][i]))
    kc.append(dict(item=f"{tk} two-day window return % (Apr 8 close to Apr 10 close)", day="t=2025-04-09", value=Y[(tk, "R2")][i9]))
kc.append(dict(item="FRED SP500 daily % change (check)", day="2025-04-09",
               value=100 * (spx.loc["2025-04-09"] / spx.loc["2025-04-08"] - 1)))
kc.append(dict(item="VIX close change (points)", day="2025-04-09", value=v[i9] - v[i9 - 1]))
kc.append(dict(item="VIX close Apr 8 / Apr 9 / Apr 10", day="", value=f"{v[i9-1]:.2f} / {v[i9]:.2f} / {v[i9+1]:.2f}"))
for sig in ["china", "broad"]:
    sg0 = build(sig, 0, keep_all)
    k = int(np.where(sdates == pd.Timestamp("2025-04-09"))[0][0])
    S = sg0["S"]
    rank = np.nan if np.isnan(S[k]) else float((S[sg0["evid"]] <= S[k]).mean())
    p = PARTS[sig]
    d9 = p.loc[pd.Timestamp("2025-04-09")] if pd.Timestamp("2025-04-09") in p.index else None
    kc.append(dict(item=f"{sig} path: S_t on 2025-04-09", day="", value=S[k]))
    kc.append(dict(item=f"{sig} path: evidence mass (sum ev) on 2025-04-09", day="", value=None if d9 is None else float(d9.ev)))
    kc.append(dict(item=f"{sig} path: percentile of S_t among evidence days", day="", value=rank))
    kc.append(dict(item=f"{sig} path: escalation day? (top decile)", day="", value=bool(sg0["E"][k])))
    kc.append(dict(item=f"{sig} path: acts (increase effective within 30 days)?", day="", value=bool(sg0["acts"][k])))
    kc.append(dict(item=f"{sig} path: threshold S for top decile", day="", value=float(np.quantile(S[sg0["evid"]], 0.9))))
    for a in ["china_inc", "broad_inc", "china_dec"]:
        kc.append(dict(item=f"announcement day of type {a}?", day="2025-04-09", value=bool(sg0["anns"][a][k])))
pd.DataFrame(kc).to_csv(OUT / "02_known_case_261102.csv", index=False)
print("done", len(res), "rows;", len(draws), "placebo draws")
print(pd.DataFrame(cnt_rows).to_string())
