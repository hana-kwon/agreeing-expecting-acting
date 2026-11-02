# ============================================================
# START DATE   : 2026-11-02
# CODE RUN DATE: 2026-11-02
# TIME TAKEN   : ~1 hour incl. writing
# DESCRIPTION  : C1 front-loading event study (country x month YoY imports around tariff effective months, B_extended calendar) with
#                shifted-date placebo and country bootstrap; C4 statement threats -> next-2-month YoY imports. Exploratory; see 00_spec_261102.md.
# INPUT FILES  : data/36_census_imports_261102/hs2_*.csv; tariff_calendar/corrected/02_increase_events_B_extended.csv; country_list_261101.csv;
#                data/31_tariff_calendar_fix_261101/02_panel_B_extended.csv (aggregate country-week columns only)
# OUTPUT FILES : analysis/36_census_imports_261102/02_c1_*.csv, 02_c4_*.csv, 02_countrymonth_yoy.csv
# ============================================================
import json
import numpy as np, pandas as pd
import common_261102 as cm

OUT = cm.get_out()
RNG = np.random.default_rng(20261102)
KS = list(range(-6, 7)); K0 = KS.index(0)
d = cm.load_hs2()
LAST = max(d.month)
idx = cm.months_index(LAST); n_m = len(idx)
M = cm.wide_country_month(d, idx)
Y = cm.yoy(M)
lo, hi = np.nanpercentile(Y.values, [1, 99])
Yw = Y.clip(lo, hi)                                   # winsorized version for the unweighted robustness run
xw, un = cm.crosswalk(d[["CTY_CODE", "CTY_NAME"]])
ev_rows, all_iso = cm.increase_events(LAST)
codes = list(M.index); cpos = {c: i for i, c in enumerate(codes)}
iso_codes = {i: [c for c in xw.get(i, []) if c in cpos] for i in all_iso | set(ev_rows.iso3)}
ever_treated = set(c for i in all_iso for c in iso_codes.get(i, []))
ctrl = np.array([c not in ever_treated for c in codes])
# ★ CORE LOGIC: control = Census countries that never appear in the B_extended increase table (any date, incl. pending/future).
# Caveat: the 2025-04-05 baseline 10% applied to all countries, but only named countries are in the calendar.

# --- event units: one per calendar month with >=1 effective increase date -------------------------------------------
ev_rows["m"] = ev_rows.eff.dt.to_period("M")
units = []
for m, g in ev_rows.groupby("m"):
    t = sorted(set(c for i in g.iso3 for c in iso_codes.get(i, [])))
    big = g.groupby("document_number").iso3.nunique().idxmax()
    sgn = g[g.document_number == big].signing_date.min()
    units.append(dict(month=m, dates=",".join(sorted(g.eff.dt.strftime("%Y-%m-%d").unique())), n_dates=g.eff.nunique(), n_iso=g.iso3.nunique(),
                      treated=t, gap_m=(m - sgn.to_period("M")).n, sign=str(sgn.date())))
n_dates = ev_rows.eff.nunique()
Mv = M.values; A = Y.values; Aw = Yw.values

def prep(m_i, treated, A_, ex=None):
    """Baseline-adjusted outcomes d (n x K), baseline weights w, masks for one event month index m_i."""
    if m_i - 6 < 0 or m_i >= n_m: return None   # post window may be truncated at the last data month
    cols = [m_i + k for k in KS]
    X = np.full((len(codes), len(KS)), np.nan)
    for j, c in enumerate(cols):
        if 0 <= c < n_m: X[:, j] = A_[:, c]
    base = X[:, 0:3]                                     # k = -6..-4
    ok = np.isfinite(base).all(axis=1)
    b = base.mean(axis=1)
    dd = X - b[:, None]
    w = np.nanmean(Mv[:, m_i - 6:m_i - 3], axis=1)
    tm = np.zeros(len(codes), bool)
    for c in treated:
        if ex is None or c not in ex: tm[cpos[c]] = True
    tm &= ok; cm_ = ctrl & ok
    w = np.where(np.isfinite(w), w, 0.0)
    if tm.sum() < 1 or cm_.sum() < 10: return None   # single-country events (China, Canada) are kept; chosen after seeing that otherwise 4 of 7 event months drop
    return dict(d=dd, w=w, T=tm, C=cm_)

def beta(e, mult, weighted):
    d = e["d"]; fin = np.isfinite(d)
    ww = (e["w"] if weighted else np.ones_like(e["w"])) * mult
    out = np.full(len(KS), np.nan)
    for j in range(len(KS)):
        r = []
        for mask in (e["T"], e["C"]):
            s = mask & fin[:, j]
            den = (ww * s).sum()
            r.append(np.nan if den <= 0 else (ww * s * np.nan_to_num(d[:, j])).sum() / den)
        out[j] = r[0] - r[1]
    return out

def pooled(evs, mult, weighted):
    B = np.array([beta(e, mult, weighted) for e in evs])   # events x K
    bk = np.nanmean(B, axis=0)
    pre = np.nanmean(B[:, K0 - 3:K0], axis=1); post = np.nanmean(B[:, K0:K0 + 3], axis=1)
    return bk, np.nanmean(pre), np.nanmean(post), B

def build(shift=0, ex=None, A_=A, clean=False):
    evs, meta = [], []
    months_t = [u["month"] for u in units]
    for u in units:
        m_i = idx.get_loc(u["month"]) + shift if u["month"] in idx else None
        if m_i is None: continue
        tr = u["treated"]
        if clean:   # drop treated countries that have another event month within +-6 months
            near = set()
            for u2 in units:
                if u2["month"] != u["month"] and abs((u2["month"] - u["month"]).n) <= 6: near |= set(u2["treated"])
            tr = [c for c in tr if c not in near]
        e = prep(m_i, tr, A_, ex)
        if e is not None: evs.append(e); meta.append(u)
    return evs, meta

ones = np.ones(len(codes))
def boot(evs, weighted, B=500):
    res = []
    for _ in range(B):
        mult = RNG.multinomial(len(codes), np.ones(len(codes)) / len(codes)).astype(float)
        bk, pre, post, _ = pooled(evs, mult, weighted)
        res.append(np.r_[bk, pre, post])
    return np.nanstd(np.array(res), axis=0, ddof=1)

SHIFTS = [s for s in list(range(-12, -2)) + list(range(3, 13))]
CHN_CAN_MEX = {"5700", "5820", "5660", "1220", "2010"}
variants = {"main_weighted": dict(weighted=True, A_=A, ex=None, clean=False),
            "unweighted_winsor": dict(weighted=False, A_=Aw, ex=None, clean=False),
            "ex_CHN_CAN_MEX": dict(weighted=True, A_=A, ex=CHN_CAN_MEX, clean=False),
            "clean_treated": dict(weighted=True, A_=A, ex=None, clean=True)}
stats, pooled_rows, plc_rows, ev_rows_out = [], [], [], []
for name, v in variants.items():
    evs, meta = build(0, v["ex"], v["A_"], v["clean"])
    bk, pre, post, Bm = pooled(evs, ones, v["weighted"])
    se = boot(evs, v["weighted"]) if name in ("main_weighted", "unweighted_winsor", "ex_CHN_CAN_MEX") else np.full(len(KS) + 2, np.nan)
    plc = []
    for s in SHIFTS:
        e2, _ = build(s, v["ex"], v["A_"], v["clean"])
        if len(e2) < 3: continue
        b2, pr2, po2, _ = pooled(e2, ones, v["weighted"])
        plc.append((s, len(e2), pr2, po2, *b2))
        plc_rows.append(dict(variant=name, shift=s, n_events=len(e2), PRE=pr2, POST0_2=po2, **{f"k{k}": x for k, x in zip(KS, b2)}))
    P_ = pd.DataFrame(plc, columns=["s", "n", "pre", "post"] + [f"k{k}" for k in KS])
    row = dict(variant=name, n_event_months=len(evs), n_treated_mean=np.mean([e["T"].sum() for e in evs]), n_control=int(np.mean([e["C"].sum() for e in evs])),
               PRE=pre, PRE_se=se[-2], POST0_2=post, POST_se=se[-1], n_placebo=len(P_),
               PRE_plc_mean=P_.pre.mean(), PRE_plc_sd=P_.pre.std(), PRE_p=(P_.pre.abs() >= abs(pre)).mean(),
               POST_plc_mean=P_.post.mean(), POST_plc_sd=P_.post.std(), POST_p=(P_.post.abs() >= abs(post)).mean())
    stats.append(row)
    for j, k in enumerate(KS):
        pooled_rows.append(dict(variant=name, k=k, beta=bk[j], se=se[j], plc_p05=P_[f"k{k}"].quantile(.05), plc_p95=P_[f"k{k}"].quantile(.95)))
    if name == "main_weighted":
        for u, brow in zip(meta, Bm):
            ev_rows_out.append(dict(month=str(u["month"]), dates=u["dates"], n_dates=u["n_dates"], n_iso=u["n_iso"], n_treated_codes=len(u["treated"]),
                                    gap_months=u["gap_m"], sign=u["sign"], PRE=np.nanmean(brow[K0 - 3:K0]), POST0_2=np.nanmean(brow[K0:K0 + 3]),
                                    gapwin=(np.nanmean(brow[K0 - u["gap_m"]:K0]) if 1 <= u["gap_m"] <= 6 else np.nan),
                                    **{f"k{k}": x for k, x in zip(KS, brow)}))
cm.safe_write(pd.DataFrame(stats), OUT / "02_c1_stats.csv")
cm.safe_write(pd.DataFrame(pooled_rows), OUT / "02_c1_pooled_beta.csv")
cm.safe_write(pd.DataFrame(plc_rows), OUT / "02_c1_placebo_shifts.csv")
cm.safe_write(pd.DataFrame(ev_rows_out), OUT / "02_c1_event_months.csv")
out_meta = dict(last_month=str(LAST), n_distinct_effective_dates_used=int(n_dates), n_event_months=len(units), unmapped_iso3=un,
                n_control_countries=int(ctrl.sum()), n_census_countries=len(codes), winsor=[float(lo), float(hi)])
json.dump(out_meta, open(OUT / "02_c1_meta.json", "x"), indent=1)

# --- C4: statements -> imports, iso3 unit level (EU = sum of members) ------------------------------------------------------
units_iso = list(xw.keys())
MU = pd.DataFrame({i: M.loc[[c for c in xw[i] if c in cpos]].sum(axis=0, min_count=1) for i in units_iso}).T
YU = cm.yoy(MU); YUw = YU.clip(lo, hi)
pn = pd.read_csv(cm.PANEL31, usecols=["iso3", "week", "threats4", "mentions", "jev"])
pn["m"] = pd.to_datetime(pn.week).dt.to_period("M")
mm = pn.groupby(["iso3", "m"]).agg(threat=("threats4", lambda s: float((s > 0).any())), n_thr=("threats4", "sum"), mentions=("mentions", "sum"), jev=("jev", "mean")).reset_index()
eff_by_iso = ev_rows.groupby("iso3").eff.apply(list).to_dict()
def impl_after(i, m):
    end = m.end_time
    return any(end < e <= end + pd.Timedelta(days=90) for e in eff_by_iso.get(i, []))
mm["impl90"] = [impl_after(i, m) for i, m in zip(mm.iso3, mm.m)]
rows = []
for r in mm.itertuples():
    if r.iso3 not in YU.index or r.m not in idx: continue
    j = idx.get_loc(r.m)
    if j + 2 >= n_m: continue
    y = YUw.loc[r.iso3].values
    out2 = np.nanmean([y[j + 1], y[j + 2]]) if np.isfinite(y[j + 1:j + 3]).any() else np.nan
    pre = np.nanmean(y[j - 3:j]) if np.isfinite(y[j - 3:j]).any() else np.nan
    rows.append(dict(iso3=r.iso3, m=str(r.m), threat=r.threat, impl90=r.impl90, out=out2, adj=out2 - pre))
C4 = pd.DataFrame(rows)
cm.safe_write(C4, OUT / "02_c4_unit_months.csv")
def c4stat(df, col):
    res = {}
    for g, mask in (("threat_later_implemented", (df.threat == 1) & df.impl90), ("threat_not_implemented", (df.threat == 1) & ~df.impl90)):
        diffs = []
        for m, x in df.groupby("m"):
            a = x[mask.loc[x.index]][col].dropna(); b = x[x.threat == 0][col].dropna()
            if len(a) and len(b): diffs.append(a.mean() - b.mean())
        res[g] = (np.mean(diffs) if diffs else np.nan, len(diffs))
    return res
c4rows = []
isos = C4.iso3.unique()
for col in ("out", "adj"):
    pt = c4stat(C4, col)
    bs = {g: [] for g in pt}
    for _ in range(500):
        pick = RNG.choice(isos, len(isos))
        D = pd.concat([C4[C4.iso3 == i] for i in pick], ignore_index=True)
        bb = c4stat(D, col)
        for g in bb: bs[g].append(bb[g][0])
    for g in pt:
        c4rows.append(dict(outcome=col, group=g, diff_mean=pt[g][0], se=np.nanstd(bs[g], ddof=1), n_months=pt[g][1],
                           n_unit_months=int(((C4.threat == 1) & (C4.impl90 if g == "threat_later_implemented" else ~C4.impl90)).sum())))
cm.safe_write(pd.DataFrame(c4rows), OUT / "02_c4_stats.csv")
cm.safe_write(C4.groupby("threat").agg(n=("out", "size"), out=("out", "mean"), adj=("adj", "mean")).reset_index(), OUT / "02_c4_descriptive.csv")
print("done", LAST, n_dates, len(units), un)
