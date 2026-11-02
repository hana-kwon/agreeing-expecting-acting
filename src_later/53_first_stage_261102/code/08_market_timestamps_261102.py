# ============================================================
# START DATE   : 2026-11-02
# CODE RUN DATE: 2026-11-02
# TIME TAKEN   : ~10 min
# DESCRIPTION  : Market test with Truth Social post times (TiRADE v1.5 public file, time_eastern, platform = Truth Social). Posts after 16:00 ET
#                (or on a non-trading day) -> open-to-close return of the NEXT trading day; posts before 09:30 ET -> same-day open-to-close.
#                Posts during trading hours are not used (no intraday prices). SPY and FXI daily open and close from data/45 (yahoo).
#                Placebo: 5,000 random draws of the same number of trading days that carry no TiRADE Truth Social post (seed 261102), p = share |placebo mean| >= |actual|.
#                Groups: Threatened, Announced, Threatened+Announced, and those naming China (FXI). Event type is TiRADE's, not the f3 flag: f3 passages have dates only.
# INPUT FILES  : ~/research_local/orgsci_repos/agreeing_expecting_acting/data/public/tirade-v1.5-rate-statements.csv; data/45_market_response_261102/yahoo_{SPY,FXI}_261102.csv
# OUTPUT FILES : analysis/53_first_stage_261102/08_market_timestamps_261102.csv
# ============================================================
import sys
from pathlib import Path
import numpy as np, pandas as pd
TOPIC = Path(__file__).resolve().parents[2]; OUT = TOPIC / "analysis/53_first_stage_261102"
if (OUT / "08_market_timestamps_261102.csv").exists(): sys.exit("output exists; refusing to overwrite")
t = pd.read_csv(Path.home() / "research_local/orgsci_repos/agreeing_expecting_acting/data/public/tirade-v1.5-rate-statements.csv")
t = t[t.platform.eq("Truth Social") & t.time_eastern.notna()].copy()
t["dt"] = pd.to_datetime(t.date + " " + t.time_eastern); t["china"] = t.countries.fillna("").str.contains("China")
px = {}
for s in ["SPY", "FXI"]:
    d = pd.read_csv(TOPIC / f"data/45_market_response_261102/yahoo_{s}_261102.csv", parse_dates=["date"]).dropna(subset=["open", "close"]).sort_values("date")
    d["oc"] = 100 * (d.close / d.open - 1); px[s] = d.set_index("date")
days = px["SPY"].index
def target_day(dt):
    d = pd.Timestamp(dt.date()); mins = dt.hour * 60 + dt.minute; trading = d in days
    if trading and mins < 570: return d, "pre_open"
    if trading and mins >= 960: return days[days > d].min() if (days > d).any() else pd.NaT, "after_close"
    if not trading: return (days[days > d].min() if (days > d).any() else pd.NaT), "after_close"   # weekend or holiday: next session
    return pd.NaT, "during"
tt = [target_day(x) for x in t.dt]; t["tday"] = [a for a, _ in tt]; t["when"] = [b for _, b in tt]
post_days = set(pd.to_datetime(t.date)); nopost = [d for d in days if d not in post_days]
rng = np.random.default_rng(261102); rows = []
groups = {"Threatened": t.event_type.eq("Threatened"), "Announced": t.event_type.eq("Announced"), "Threatened+Announced": t.event_type.isin(["Threatened", "Announced"]),
          "T+A naming China": t.event_type.isin(["Threatened", "Announced"]) & t.china}
for gname, gm in groups.items():
    for when in ["after_close", "pre_open", "both"]:
        sel = t[gm & (t.when.isin(["after_close", "pre_open"]) if when == "both" else t.when.eq(when)) & t.tday.notna()]
        for s in ["SPY", "FXI"]:
            r = px[s].oc.reindex(sel.tday).dropna(); n = len(r)
            if n < 3: continue
            pl = np.array([px[s].oc.reindex(rng.choice(nopost, n, replace=False)).mean() for _ in range(5000)])
            rows.append(dict(group=gname, timing=when, asset=s, n_posts=n, n_trading_days=int(sel.tday.nunique()), mean_oc_pct=r.mean(), se=r.std(ddof=1) / np.sqrt(n),
                             placebo_mean=pl.mean(), placebo_sd=pl.std(), placebo_p=float((np.abs(pl - pl.mean()) >= abs(r.mean() - pl.mean())).mean()),
                             mde_80_pct=2.8 * pl.std()))
R = pd.DataFrame(rows); R.to_csv(OUT / "08_market_timestamps_261102.csv", index=False)
print(t.when.value_counts().to_dict()); print(R.round(3).to_string())
