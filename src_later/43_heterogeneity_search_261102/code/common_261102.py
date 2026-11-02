# ============================================================
# START DATE   : 2026-11-02
# CODE RUN DATE: 2026-11-02
# TIME TAKEN   : ~15 min
# DESCRIPTION  : Shared signal builder for code/43 (Jev event file -> 28-day-window direction/duration signals, incl. the
#                H4 in-force-diluted variants F1-F3) with week-shift support for date-shift placebos.
# INPUT FILES  :PRIVATE_DATA_ROOT/TEAM_PRIVATE/.../simple_time_series_v1/event_question_option_probabilities.parquet (read only)
# OUTPUT FILES : none (imported by 01_survey_tests_261102.py and 02_customs_signals_261102.py)
# ============================================================
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve()
TOPIC = HERE.parents[2]
SHARED = HERE.parents[7] / "PRIVATE_DATA_ROOT"
IA = SHARED / "SMB_tariff/ideology_analysis"
TS = (SHARED / "TEAM_PRIVATE/ideology_share/directional_coding/data/jev_corpus/"
      "simple_time_series_v1/event_question_option_probabilities.parquet")
OUT = TOPIC / "analysis/43_heterogeneity_search_261102"
CACHE = Path.home() / "private_cache/het43_261102"
SHIFTS = [k for k in range(-52, 53) if abs(k) >= 5]          # weeks; 96 shifts

ev = pd.read_parquet(TS)
ev["event_date"] = pd.to_datetime(ev["event_date"])
INFORCE = ["current_or_interim_total", "settled_or_final_total"]


def wide(q):
    x = ev[ev.question_id == q]
    w = x.pivot_table(index=["event_question_id", "event_key", "event_date"], columns="option",
                      values="raw_option_probability", aggfunc="sum").fillna(0.0).reset_index()
    w["ev"] = 1.0 - w[x[x.is_absence_option].option.unique()[0]]
    return w


def inforce_prob(fam):
    # ★ IMPORTANT: no tense field exists in the file. P(rate described is the current/interim or settled total) from the
    # rate_role question is the only available proxy for "describes a rate in force". Events without a rate_role question
    # get 0 (kept as threat/direction statements).
    w = wide(f"tariff_{fam}_rate_role")
    w["pin"] = w[INFORCE].sum(axis=1)
    return w.groupby("event_key").pin.mean()


def daily(w, num, den):
    g = w.assign(num=num, den=den).groupby("event_date")[["num", "den"]].sum()
    idx = pd.date_range("2023-06-01", "2026-12-31")
    return g.reindex(idx).fillna(0.0)


def build_defs():
    D = {}
    for fam in ["china", "broad"]:
        p = wide(f"tariff_{fam}_path")
        d = wide(f"tariff_{fam}_duration")
        dirn = p["higher"] - p["lower"]
        D[f"path_{fam}"] = daily(p, dirn, p["ev"])
        D[f"dur_{fam}"] = daily(d, d["permanent"] - d["temporary"], d["ev"])
        pin = p.event_key.map(inforce_prob(fam)).fillna(0.0)
        w = 1 - pin
        D[f"F1_{fam}"] = daily(p, w * dirn, w * p["ev"])
        keep = (pin <= 0.5).astype(float)
        D[f"F2_{fam}"] = daily(p, keep * dirn, keep * p["ev"])
        D[f"F3_{fam}"] = daily(p, dirn, p["higher"] + p["lower"])
        D[f"_n_events_{fam}"] = (len(p), int((pin > 0.5).sum()), int(p.event_key.isin(inforce_prob(fam).index).sum()))
    return D


DEFS = build_defs()
SIGNALS = [k for k in DEFS if not k.startswith("_")]


def signal_at(name, dates, shift_weeks=0):
    """Share of the window's numerator over denominator, window [t+shift-28, t+shift-1] days (as in code/27 R4)."""
    d = DEFS[name]
    cn, cd = d.num.cumsum(), d.den.cumsum()
    idx = d.index
    out = {}
    for t in pd.DatetimeIndex(dates):
        a_ = t + pd.Timedelta(days=7 * shift_weeks - 28)
        b_ = t + pd.Timedelta(days=7 * shift_weeks - 1)
        if a_ < idx[0] or b_ > idx[-1]:
            out[t] = np.nan
            continue
        n = cn[b_] - (cn[a_ - pd.Timedelta(days=1)] if a_ > idx[0] else 0.0)
        e = cd[b_] - (cd[a_ - pd.Timedelta(days=1)] if a_ > idx[0] else 0.0)
        out[t] = n / e if e > 1e-9 else np.nan
    return pd.Series(out)
