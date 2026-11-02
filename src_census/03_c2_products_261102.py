# ============================================================
# START DATE   : 2026-11-02
# CODE RUN DATE: 2026-11-02
# TIME TAKEN   : ~30 min incl. writing
# DESCRIPTION  : C2 product tariff events: YoY world imports by HS4 around announcement and effective months (Section 232 steel/aluminum,
#                autos, copper, lumber/furniture, trucks, pharma [not implemented], semiconductors) vs other HS4; shifted-date placebo, product bootstrap.
# INPUT FILES  : data/36_census_imports_261102/hs4w_*.csv (HS4 x month, CTY_CODE='-' world total)
# OUTPUT FILES : analysis/36_census_imports_261102/03_c2_*.csv
# ============================================================
import numpy as np, pandas as pd
import common_261102 as cm

OUT = cm.get_out()
RNG = np.random.default_rng(20261102)
d4 = cm.load_hs4w(); LAST = max(d4.month)
idx = cm.months_index(LAST); n_m = len(idx)
W = d4.pivot_table(index="hs4", columns="month", values="gen", aggfunc="sum")
W.columns = pd.PeriodIndex(W.columns, freq="M"); W = W.reindex(columns=idx)
Y = cm.yoy(W); lo, hi = np.nanpercentile(Y.values, [1, 99]); Yw = Y.clip(lo, hi)
codes = np.array(W.index); Wv = W.values
KS = list(range(-8, 5)); K0 = KS.index(0)

def rng_(a, b): return {f"{c:04d}" for c in range(a, b + 1)}
SETS = {"steel_alu": rng_(7201, 7229) | rng_(7301, 7326) | rng_(7601, 7616),
        "autos": {"8703", "8708"}, "copper": rng_(7401, 7419),
        "lumber_furniture": {"4407", "4408", "4409", "9401", "9403"}, "trucks": {"8701", "8704"},
        "pharma": rng_(3002, 3006), "semis": {"8541", "8542"}}
# ★ IMPORTANT: product sets are HS4 approximations of the Section 232 lists (not the exact annex HTS lists); steel/aluminum derivative
# products outside these HS4s are not in the treated set.
P = pd.Period
EVENTS = [dict(name="steel_alu_25", set="steel_alu", ann=P("2025-02", "M"), eff=P("2025-03", "M"), impl=True),
          dict(name="steel_alu_50", set="steel_alu", ann=P("2025-05", "M"), eff=P("2025-06", "M"), impl=True),
          dict(name="autos", set="autos", ann=P("2025-03", "M"), eff=P("2025-04", "M"), impl=True),
          dict(name="copper", set="copper", ann=P("2025-07", "M"), eff=P("2025-08", "M"), impl=True),
          dict(name="lumber_furniture", set="lumber_furniture", ann=P("2025-09", "M"), eff=P("2025-10", "M"), impl=True),
          dict(name="trucks", set="trucks", ann=P("2025-09", "M"), eff=P("2025-11", "M"), impl=True),
          dict(name="semis", set="semis", ann=P("2025-08", "M"), eff=P("2026-01", "M"), impl=True),
          dict(name="pharma_not_implemented", set="pharma", ann=P("2025-09", "M"), eff=P("2025-10", "M"), impl=False)]
union = set().union(*SETS.values())
EXCL_CH = {"30", "44", "72", "73", "74", "76", "87", "94"}
istreated = lambda s: np.array([c in SETS[s] for c in codes])
excl = np.array([(c in union) or c[:2] in EXCL_CH for c in codes])

def prep(ev, A, shift=0):
    m_i = idx.get_loc(ev["eff"]) + shift
    if m_i - 8 < 0 or m_i >= n_m: return None
    X = np.full((len(codes), len(KS)), np.nan)
    for j, k in enumerate(KS):
        c = m_i + k
        if 0 <= c < n_m: X[:, j] = A[:, c]
    base = X[:, 0:3]; ok = np.isfinite(base).all(axis=1)
    dd = X - base.mean(axis=1)[:, None]
    w = np.nanmean(Wv[:, m_i - 8:m_i - 5], axis=1); w = np.where(np.isfinite(w), w, 0.0)
    tm = istreated(ev["set"]) & ok & (w >= 1e6)
    cmk = (~excl) & ok & (w >= 1e7)
    if tm.sum() < 2 or cmk.sum() < 20: return None
    gap = (ev["eff"] - ev["ann"]).n
    return dict(d=dd, w=w, T=tm, C=cmk, gap=gap)

def beta(e, mult, weighted):
    d = e["d"]; fin = np.isfinite(d); ww = (e["w"] if weighted else np.ones_like(e["w"])) * mult
    out = np.full(len(KS), np.nan)
    for j in range(len(KS)):
        r = []
        for mask in (e["T"], e["C"]):
            s = mask & fin[:, j]; den = (ww * s).sum()
            r.append(np.nan if den <= 0 else (ww * s * np.nan_to_num(d[:, j])).sum() / den)
        out[j] = r[0] - r[1]
    return out

def stats_of(b, gap):
    ann = np.nanmean(b[K0 - gap:K0]) if 1 <= gap <= 8 else np.nan
    return ann, np.nanmean(b[K0:K0 + 2])      # ANN = announcement month ..month before effective; EFF = k0..k1
ones = np.ones(len(codes))
SHIFTS = list(range(-12, -2)) + list(range(3, 13))
rows, brows, prow = [], [], []
cache = {}
for weighted, A, tag in ((True, Y.values, "weighted"), (False, Yw.values, "unweighted_winsor")):
    ev_stats = {}
    for ev in EVENTS:
        e = prep(ev, A)
        if e is None: continue
        b = beta(e, ones, weighted); ann, eff = stats_of(b, e["gap"])
        bs = []
        for _ in range(500):
            mult = RNG.multinomial(len(codes), np.ones(len(codes)) / len(codes)).astype(float)
            bb = beta(e, mult, weighted); a2, f2 = stats_of(bb, e["gap"]); bs.append((a2, f2))
        bs = np.array(bs)
        pl = []
        for s in SHIFTS:
            e2 = prep(ev, A, s)
            if e2 is None: continue
            a2, f2 = stats_of(beta(e2, ones, weighted), e2["gap"]); pl.append((s, a2, f2))
        pl = pd.DataFrame(pl, columns=["s", "ann", "eff"])
        ev_stats[ev["name"]] = (ann, eff)
        rows.append(dict(spec=tag, event=ev["name"], implemented=ev["impl"], ann_month=str(ev["ann"]), eff_month=str(ev["eff"]), gap_months=e["gap"],
                         n_treated=int(e["T"].sum()), n_control=int(e["C"].sum()),
                         ANN=ann, ANN_se=np.nanstd(bs[:, 0], ddof=1), ANN_p_placebo=(pl.ann.abs() >= abs(ann)).mean() if np.isfinite(ann) else np.nan,
                         EFF01=eff, EFF_se=np.nanstd(bs[:, 1], ddof=1), EFF_p_placebo=(pl.eff.abs() >= abs(eff)).mean(), n_placebo=len(pl)))
        for k, x in zip(KS, b): brows.append(dict(spec=tag, event=ev["name"], k=k, beta=x))
        cache[(tag, ev["name"])] = pl
    # pooled over implemented events (mean of event stats), placebo pooled by shift
    impl = [ev["name"] for ev in EVENTS if ev["impl"] and ev["name"] in ev_stats]
    pa = np.nanmean([ev_stats[n][0] for n in impl]); pe = np.nanmean([ev_stats[n][1] for n in impl])
    S = {}
    for n in impl:
        for r in cache[(tag, n)].itertuples(): S.setdefault(r.s, []).append((r.ann, r.eff))
    pls = [(s, np.nanmean([x[0] for x in v]), np.nanmean([x[1] for x in v]), len(v)) for s, v in S.items() if len(v) >= 4]
    pls = pd.DataFrame(pls, columns=["s", "ann", "eff", "n"])
    prow.append(dict(spec=tag, n_events=len(impl), ANN_pooled=pa, EFF01_pooled=pe, n_shifts=len(pls),
                     ANN_p=(pls.ann.abs() >= abs(pa)).mean(), EFF_p=(pls.eff.abs() >= abs(pe)).mean(),
                     ANN_plc_mean=pls.ann.mean(), ANN_plc_sd=pls.ann.std(), EFF_plc_mean=pls.eff.mean(), EFF_plc_sd=pls.eff.std()))
cm.safe_write(pd.DataFrame(rows), OUT / "03_c2_event_stats.csv")
cm.safe_write(pd.DataFrame(brows), OUT / "03_c2_event_study_beta.csv")
cm.safe_write(pd.DataFrame(prow), OUT / "03_c2_pooled.csv")
cm.safe_write(pd.DataFrame(dict(last_month=[str(LAST)], n_hs4=[len(codes)], n_rows=[len(d4)])), OUT / "03_c2_meta.csv")
print(pd.DataFrame(rows).round(3).to_string()); print(pd.DataFrame(prow).round(3).to_string())
