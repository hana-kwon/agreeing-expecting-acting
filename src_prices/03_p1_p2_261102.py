# ============================================================
# START DATE   : 2026-11-02
# CODE RUN DATE: 2026-11-02
# TIME TAKEN   : ~25 min
# DESCRIPTION  : P1 (escalation days vs other days: all / acts / words only, week-block bootstrap SE, HAC SE, 5-52 week shift placebo,
#                announcement-date exclusions X1-X3, secondary variants S1-S4, pre-trend diagnostic), event-time paths, P2 (China increase dates).
# INPUT FILES  : data/39_.../*.csv (Pricing Lab copies), analysis/39_.../02_daily_signal_{china,broad}.csv, 02_china_events.csv
# OUTPUT FILES : analysis/39_.../03_p1_estimates.csv, 03_p1_placebo_shifts.csv, 03_escalation_days.csv, 03_event_paths.csv, 03_p2_dates.csv, 03_meta.json
# ============================================================
import json
import statsmodels.api as sm
from common_261102 import *

out, args = get_out()
# inputs of step 02 live in the same output folder (out)
NBOOT, SEED = args.nboot, 20261102
m = load_prices()
D = pd.date_range("2024-10-02", "2026-03-31"); nD = len(D)
STARRED = pd.to_datetime(["2025-03-04", "2025-04-02", "2025-05-12", "2025-07-07", "2025-07-08", "2025-07-09", "2025-09-29"])
WIDE = STARRED.append(pd.to_datetime(["2025-02-04", "2025-04-10", "2025-06-11"]))


def near(days, dates, k=3):
    return np.array([any(abs((d - x).days) <= k for x in dates) for d in days])


def window_has(days, dates, h):
    return np.array([any((x >= d - pd.Timedelta(days=1)) and (x <= d + pd.Timedelta(days=h)) for x in dates) for d in days])


SAMPLES = {"all": np.ones(nD, bool)}
SAMPLES["X1"] = ~near(D, STARRED)
SAMPLES["X2"] = ~near(D, WIDE)


def outcome(col, h):
    s = m[col]
    if h == "pre":
        return np.array([s.get(d - pd.Timedelta(days=1), np.nan) - s.get(d - pd.Timedelta(days=8), np.nan) for d in D])
    return np.array([s.loc[d + pd.Timedelta(days=h)] - s.loc[d - pd.Timedelta(days=1)] for d in D])


def esc_arrays(tag, rule):
    g = pd.read_csv(out / f"02_daily_signal_{tag}.csv", parse_dates=["date"]).set_index("date")
    e = g[f"esc{rule}"].reindex(D).fillna(False).to_numpy(bool)
    act = g["act30"].reindex(D).fillna(False).to_numpy(bool)
    return e, e & act, e & ~act


def stats(v, Ea, Ec, Ew, mask):
    """Differences in mean change: escalation group minus other days (sample days that are not escalation days)."""
    ok = mask & ~np.isnan(v)
    oth = ok & ~Ea
    mo = v[oth].mean()
    r = {"n_all": int((ok & Ea).sum()), "n_act": int((ok & Ec).sum()), "n_words": int((ok & Ew).sum()), "n_other": int(oth.sum())}
    r["all"] = v[ok & Ea].mean() - mo if r["n_all"] else np.nan
    r["acts"] = v[ok & Ec].mean() - mo if r["n_act"] else np.nan
    r["words"] = v[ok & Ew].mean() - mo if r["n_words"] else np.nan
    r["acts_minus_words"] = r["acts"] - r["words"]
    return r


MON = (D - pd.to_timedelta(D.day_of_week, unit="D"))
WK = pd.factorize(MON)[0]; NW = WK.max() + 1                      # week-of-Monday blocks


def boot(v, Ea, Ec, Ew, mask, seed=SEED):
    ok = mask & ~np.isnan(v)
    groups = {"all": ok & Ea, "acts": ok & Ec, "words": ok & Ew, "oth": ok & ~Ea}
    cnt = {k: np.bincount(WK, weights=x.astype(float), minlength=NW) for k, x in groups.items()}
    sm_ = {k: np.bincount(WK, weights=np.where(x, v, 0.0), minlength=NW) for k, x in groups.items()}
    rng = np.random.default_rng(seed)
    c = rng.multinomial(NW, np.ones(NW) / NW, size=NBOOT).astype(float)    # ★ CORE LOGIC: resample whole weeks, keep all days in a drawn week
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = {k: (c @ sm_[k]) / (c @ cnt[k]) for k in groups}
    res = {"all": mean["all"] - mean["oth"], "acts": mean["acts"] - mean["oth"], "words": mean["words"] - mean["oth"]}
    res["acts_minus_words"] = res["acts"] - res["words"]
    o = {}
    for k, x in res.items():
        fin = np.isfinite(x)
        o[k] = (np.std(x[fin], ddof=1) if fin.sum() > 10 else np.nan,
                np.percentile(x[fin], 2.5) if fin.sum() > 10 else np.nan, np.percentile(x[fin], 97.5) if fin.sum() > 10 else np.nan, int((~fin).sum()))
    return o


def hac(v, Ea, Ec, Ew, mask, h):
    ok = mask & ~np.isnan(v)
    lags = (h if h != "pre" else 7) + 7
    o = {}
    X = sm.add_constant(pd.DataFrame({"E": Ea[ok].astype(float)}))
    r = sm.OLS(v[ok], X).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    o["all"] = float(r.bse["E"])
    if Ec[ok].sum() > 0 and Ew[ok].sum() > 0:
        X = sm.add_constant(pd.DataFrame({"A": Ec[ok].astype(float), "W": Ew[ok].astype(float)}))
        r = sm.OLS(v[ok], X).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
        o["acts"], o["words"] = float(r.bse["A"]), float(r.bse["W"])
        o["acts_minus_words"] = float(r.t_test("A - W = 0").sd[0][0])
    return o


SHIFTS = [7 * w for w in list(range(-52, -4)) + list(range(5, 53))]


def shift_arr(x, k):
    y = np.zeros(nD, bool)
    idx = np.where(x)[0] + k
    idx = idx[(idx >= 0) & (idx < nD)]
    y[idx] = True
    return y


def placebo(v, Ea, Ec, Ew, mask, actual):
    rows = []
    for k in SHIFTS:
        a, c, w = shift_arr(Ea, k), shift_arr(Ec, k), shift_arr(Ew, k)
        ok = mask & ~np.isnan(v)
        n_all = int((a & ok).sum())
        if n_all < 8:
            continue
        s = stats(v, a, c, w, mask)
        for key in ("acts", "words", "acts_minus_words"):
            if s["n_act"] < 3 or s["n_words"] < 3:
                s[key] = np.nan
        rows.append({"shift_days": k, **s})
    pl = pd.DataFrame(rows)
    p = {}
    for key in ("all", "acts", "words", "acts_minus_words"):
        x = pl[key].dropna()
        p[key] = (float((x.abs() >= abs(actual[key])).mean()) if len(x) and np.isfinite(actual[key]) else np.nan, int(len(x)))
    return p, pl


def demean_month(v, mask):
    v2 = v.copy(); mon = D.to_period("M")
    ok = mask & ~np.isnan(v)
    for mm in np.unique(mon):
        sel = (mon == mm) & ok
        if sel.any():
            v2[sel] = v[sel] - v[sel].mean()
    return v2


rows, plrows = [], []


def run(label, stmt, rule, col, h, sample, demean=False):
    Ea, Ec, Ew = esc_arrays(stmt, rule)
    mask = SAMPLES[sample] if sample in SAMPLES else None
    if sample == "X3":
        mask = ~window_has(D, WIDE, 14 if h == "pre" else h)
    v = outcome(col, h)
    if demean:
        v = demean_month(v, mask)
    act = stats(v, Ea, Ec, Ew, mask)
    b = boot(v, Ea, Ec, Ew, mask); hc = hac(v, Ea, Ec, Ew, mask, h)
    p, pl = placebo(v, Ea, Ec, Ew, mask, act)
    pl.insert(0, "spec", label); plrows.append(pl)
    for key in ("all", "acts", "words", "acts_minus_words"):
        rows.append({"spec": label, "stmt": stmt, "esc_rule": rule, "outcome": col, "h": h, "sample": sample, "demean_month": demean,
                     "contrast": key, "estimate": act[key], "n_all": act["n_all"], "n_act": act["n_act"], "n_words": act["n_words"], "n_other": act["n_other"],
                     "se_boot": b[key][0], "ci_lo": b[key][1], "ci_hi": b[key][2], "boot_dropped": b[key][3],
                     "se_hac": hc.get(key, np.nan), "placebo_p": p[key][0], "placebo_valid_shifts": p[key][1]})


# main grid
for h in (7, 14):
    for s in ("all", "X1", "X2", "X3"):
        run(f"MAIN_h{h}_{s}", "china", 90, "y", h, s)
# pre-trend diagnostic (all sample and X1)
for s in ("all", "X1"):
    run(f"PRE7_{s}", "china", 90, "y", "pre", s)
# secondary variants
for h in (7, 14):
    for s in ("all", "X1"):
        run(f"S1_month_h{h}_{s}", "china", 90, "y", h, s, demean=True)
        run(f"S2_imp_dom_h{h}_{s}", "china", 90, "y2", h, s)
        run(f"S3_p75_h{h}_{s}", "china", 75, "y", h, s)
        run(f"S4_broad_h{h}_{s}", "broad", 90, "y", h, s)
est = pd.DataFrame(rows)
safe_write(est, out / "03_p1_estimates.csv")
safe_write(pd.concat(plrows, ignore_index=True), out / "03_p1_placebo_shifts.csv")

# escalation-day table and event paths (primary rule)
Ea, Ec, Ew = esc_arrays("china", 90)
sig = pd.read_csv(out / "02_daily_signal_china.csv", parse_dates=["date"]).set_index("date")
tab = pd.DataFrame({"date": D[Ea], "S": sig.S.reindex(D[Ea]).values, "ev": sig.ev.reindex(D[Ea]).values,
                    "label": np.where(Ec[Ea], "acts", "words"),
                    "d7": outcome("y", 7)[Ea], "d14": outcome("y", 14)[Ea],
                    "near_cavallo_starred_3d": near(D[Ea], STARRED), "near_wide_3d": near(D[Ea], WIDE)})
safe_write(tab, out / "03_escalation_days.csv")
J = list(range(-7, 15))
paths = []
for name, sel in [("acts", Ec), ("words", Ew), ("other", ~Ea)]:
    days = D[sel]
    for j in J:
        # event-time path uses only days whose full window [d-8, d+14] is observed
        vals = [m.y.get(d + pd.Timedelta(days=j), np.nan) - m.y.get(d - pd.Timedelta(days=1), np.nan) for d in days]
        vals = np.array(vals, float)
        paths.append({"group": name, "j": j, "mean": np.nanmean(vals), "n": int(np.isfinite(vals).sum())})
safe_write(pd.DataFrame(paths), out / "03_event_paths.csv")

# P2: China increase dates (merge effective dates within 7 days)
ce = pd.read_csv(out / "02_china_events.csv", parse_dates=["eff", "signing"])
ce = ce[ce.eff <= m.index.max()].sort_values("eff").reset_index(drop=True)
grp, cur = [], 0
for i, r in ce.iterrows():
    if i > 0 and (r.eff - ce.eff[i - 1]).days > 7:
        cur += 1
    grp.append(cur)
ce["g"] = grp
ev = ce.groupby("g").agg(announce=("signing", "min"), eff=("eff", "max"), rows=("eff", "size"), rate_max=("new_rate_max", "max")).reset_index()


def chg(col, a, b):
    if pd.isna(a) or pd.isna(b) or a >= b or a not in m.index or b not in m.index:
        return np.nan
    return m[col].loc[b] - m[col].loc[a]


p2 = []
for _, r in ev.iterrows():
    a, e = r.announce, r.eff
    rec = {"event": f"{a.date()}->{e.date()}", "announce": a, "eff": e, "rate_max": r.rate_max, "gap_days": (e - a).days}
    for col, nm in (("y", "y"), ("y2", "y2")):
        rec[f"{nm}_pre14"] = chg(col, a - pd.Timedelta(days=14), a)
        rec[f"{nm}_ann_to_eff"] = chg(col, a, e)
        rec[f"{nm}_eff_to_p14"] = chg(col, e, e + pd.Timedelta(days=14))
    rec["overlaps_other_event_window"] = bool(((ev.announce - pd.Timedelta(days=14) <= e + pd.Timedelta(days=14)) & (ev.eff + pd.Timedelta(days=14) >= a - pd.Timedelta(days=14)) & (ev.g != r.g)).any())
    p2.append(rec)
safe_write(pd.DataFrame(p2), out / "03_p2_dates.csv")
safe_write(json.dumps({"nboot": NBOOT, "seed": SEED, "n_days": nD, "week_blocks": int(NW), "n_shifts_defined": len(SHIFTS),
                       "starred": [str(x.date()) for x in STARRED], "wide": [str(x.date()) for x in WIDE],
                       "n_sample_days": {k: int(v.sum()) for k, v in SAMPLES.items()}}, indent=2), out / "03_meta.json")
print(est[(est.spec.str.startswith("MAIN"))][["spec", "contrast", "estimate", "n_all", "n_act", "n_words", "se_boot", "se_hac", "placebo_p", "placebo_valid_shifts"]].to_string())
