# ============================================================
# START DATE   : 2026-11-02
# CODE RUN DATE: 2026-11-02
# TIME TAKEN   : ~5 min
# DESCRIPTION  : R_r^w = a + b_T Threat_r + b_A Announce_r + u_r over trading days 2025-01-21 to 2026-08-24, one row per reaction day.
#                Windows: previous close to close (all posts), previous close to open and open to close (posts outside trading hours).
#                Comparison days carry no Truth Social rate post; placebo draws the same number of such days. Also |R| and a version without 2025-04-02..04-10.
# INPUT FILES  : repo data/public/tirade-v1.5-rate-statements.csv; data/45_market_response_261102/yahoo_{SPY,FXI}_261102.csv
# OUTPUT FILES : analysis/55_review_reruns_261102/02_market_windows_261102.csv
# ============================================================
import sys
from pathlib import Path
import numpy as np, pandas as pd, statsmodels.api as sm
TOPIC = Path(__file__).resolve().parents[2]; OUT = TOPIC / "analysis/55_review_reruns_261102"
if (OUT / "02_market_windows_261102.csv").exists(): sys.exit("refuse to overwrite")
t = pd.read_csv(Path.home() / "research_local/orgsci_repos/agreeing_expecting_acting/data/public/tirade-v1.5-rate-statements.csv")
t = t[t.platform.eq("Truth Social") & t.time_eastern.notna()].copy()
t["stamp"] = pd.to_datetime(t.date + " " + t.time_eastern)
px = {s: pd.read_csv(TOPIC / f"data/45_market_response_261102/yahoo_{s}_261102.csv", parse_dates=["date"]).dropna(subset=["open", "close"]).sort_values("date").set_index("date") for s in ["SPY", "FXI"]}
days = px["SPY"].index; open_days = set(days)
# ★ CORE LOGIC: a post is matched to the first close it could affect; the window starts at the previous close, so overnight reaction is kept
def react(x):
    d = pd.Timestamp(x.date()); mins = x.hour * 60 + x.minute
    if d not in open_days or mins >= 960: return days[days > d].min(), "off"
    return d, ("off" if mins < 570 else "during")
rr = [react(x) for x in t.stamp]; t["r"] = [a for a, _ in rr]; t["when"] = [b for _, b in rr]
g = t[t.event_type.isin(["Threatened", "Announced"])]
START, END = pd.Timestamp("2025-01-21"), pd.Timestamp("2026-08-24")
EXCL = pd.date_range("2025-04-02", "2025-04-10")
rng = np.random.default_rng(261102); rows = []
for asset in ["SPY", "FXI"]:
    p = px[asset]; prev = p.close.shift(1)
    W = {"close_to_close": 100 * (p.close / prev - 1), "close_to_open": 100 * (p.open / prev - 1), "open_to_close": 100 * (p.close / p.open - 1)}
    for wname, R in W.items():
        ev = g if wname == "close_to_close" else g[g.when.eq("off")]
        other = set(t.r) - set(ev.r)         # days whose only posts are other event types or, in split windows, trading-hour posts: left out of the comparison
        for drop in [False, True]:
            for form in ["signed", "abs"]:
                df = pd.DataFrame({"R": R.abs() if form == "abs" else R}); df = df[(df.index >= START) & (df.index <= END)].dropna()
                df = df[~df.index.isin(other)]
                if drop: df = df[~df.index.isin(EXCL)]
                df["T"] = df.index.isin(ev[ev.event_type.eq("Threatened")].r).astype(float); df["A"] = df.index.isin(ev[ev.event_type.eq("Announced")].r).astype(float); df["TA"] = ((df["T"] + df["A"]) > 0).astype(float)
                m = sm.OLS(df.R, sm.add_constant(df[["T", "A"]])).fit(cov_type="HC1"); m2 = sm.OLS(df.R, sm.add_constant(df[["TA"]])).fit(cov_type="HC1")
                pool = df.R[df.TA == 0].to_numpy(); mu = pool.mean()
                def perm(n, obs):
                    draws = np.array([rng.choice(pool, n, replace=False).mean() for _ in range(5000)]) - mu
                    return float((np.abs(draws) >= abs(obs)).mean())
                for term, mm in [("T", m), ("A", m), ("TA", m2)]:
                    n = int(df[term].sum())
                    rows.append(dict(asset=asset, window=wname, form=form, drop_apr2025=drop, term=term, n_days=n, n_comparison=int((df.TA == 0).sum()), b=mm.params[term], se=mm.bse[term], lo=mm.params[term] - 1.96 * mm.bse[term], hi=mm.params[term] + 1.96 * mm.bse[term], p_hc1=mm.pvalues[term], p_placebo=perm(n, df.R[df[term] == 1].mean() - mu)))
out = pd.DataFrame(rows); out.to_csv(OUT / "02_market_windows_261102.csv", index=False)
pd.set_option("display.width", 200); print(out[(out.form == "signed") & (~out.drop_apr2025)].round(3).to_string(index=False)); print(out[(out.asset == "SPY") & (out.term == "TA")].round(3).to_string(index=False))
