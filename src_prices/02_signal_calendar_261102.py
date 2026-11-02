# ============================================================
# START DATE   : 2026-11-02
# CODE RUN DATE: 2026-11-02
# TIME TAKEN   : ~15 min
# DESCRIPTION  : Daily net statement signal (higher minus lower probability mass over evidence mass) for tariff_china_path and
#                tariff_broad_path, escalation days (top decile and top quartile of evidence days), and China increase dates
#                from the corrected calendar (B_extended; cancelled and short_lived rows dropped).
# INPUT FILES  : Jev event_question_option_probabilities.parquet (read only); 02_increase_events_B_extended.csv
# OUTPUT FILES : analysis/39_.../02_daily_signal_{china,broad}.csv, 02_china_events.csv, 02_meta.json
# ============================================================
import json
from common_261102 import *

out, _ = get_out()
D0, D1 = pd.Timestamp("2024-10-02"), pd.Timestamp("2026-03-31")      # analysis window (spec section 3)
ev = pd.read_parquet(JEV); ev["event_date"] = pd.to_datetime(ev["event_date"])


def daily(q):
    """Same parts() as 02_robustness_261101.py lines 115-141, aggregated by day instead of 28-day windows."""
    x = ev[ev.question_id == q]
    absent = x[x.is_absence_option].option.unique(); assert len(absent) == 1
    w = x.pivot_table(index=["event_question_id", "event_date"], columns="option",
                      values="raw_option_probability", aggfunc="sum").fillna(0.0).reset_index()
    w["ev"] = 1.0 - w[absent[0]]                                   # evidence mass of one statement event
    w["num"] = w["higher"] - w["lower"]                            # ★ CORE LOGIC: net direction mass (higher minus lower)
    g = w.groupby("event_date").agg(ev=("ev", "sum"), num=("num", "sum"), n_events=("ev", "size"))
    g["S"] = g.num / g.ev
    return g


# China increase dates (spec section 1)
a = pd.read_csv(CAL / "02_increase_events_B_extended.csv")
a = a[(a.cancelled.astype(str) != "True") & (a.short_lived.astype(str) != "True") & (a.iso3 == "CHN")].copy()
a["eff"] = pd.to_datetime(a["eff"]); a = a.dropna(subset=["eff"])
a["signing"] = pd.to_datetime(a.signing_date).fillna(pd.to_datetime(a.publication_date))
ce = (a.groupby(["eff"]).agg(signing=("signing", "min"), n_rows=("action_id", "size"), new_rate_max=("new_rate", "max"),
                             docs=("action_id", lambda s: ";".join(sorted(set(s))))).reset_index().sort_values("eff"))
safe_write(ce, out / "02_china_events.csv")
effs = ce.eff.tolist()
meta = {"china_effective_dates": [str(e.date()) for e in effs], "window": [str(D0.date()), str(D1.date())]}


def acts_flag(d):
    return any((e > d) and (e <= d + pd.Timedelta(days=30)) for e in effs)   # ★ CORE LOGIC: implemented increase within (d, d+30]


for q, tag in [("tariff_china_path", "china"), ("tariff_broad_path", "broad")]:
    g = daily(q).reset_index().rename(columns={"event_date": "date"})
    g["in_window"] = (g.date >= D0) & (g.date <= D1)
    g["evidence_day"] = g.ev >= 0.5
    base = g[g.in_window & g.evidence_day]
    p90, p75 = np.quantile(base.S, 0.90), np.quantile(base.S, 0.75)    # fixed by spec: percentiles among evidence days in window
    g["esc90"] = g.in_window & g.evidence_day & (g.S >= p90)
    g["esc75"] = g.in_window & g.evidence_day & (g.S >= p75)
    g["act30"] = g.date.map(acts_flag)
    safe_write(g, out / f"02_daily_signal_{tag}.csv")
    meta[tag] = {"n_dates": int(len(g)), "n_in_window": int(g.in_window.sum()), "n_evidence_days_in_window": int(len(base)),
                 "p90": float(p90), "p75": float(p75), "n_esc90": int(g.esc90.sum()), "n_esc75": int(g.esc75.sum()),
                 "n_esc90_act": int((g.esc90 & g.act30).sum()), "n_esc90_words": int((g.esc90 & ~g.act30).sum()),
                 "n_esc75_act": int((g.esc75 & g.act30).sum()), "n_esc75_words": int((g.esc75 & ~g.act30).sum()),
                 "esc90_dates": [str(d.date()) for d in g[g.esc90].date]}
    print(tag, {k: v for k, v in meta[tag].items() if k != "esc90_dates"})
safe_write(json.dumps(meta, indent=2), out / "02_meta.json")
print(ce[["eff", "signing", "new_rate_max"]])
