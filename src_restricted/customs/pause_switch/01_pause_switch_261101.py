# ============================================================
# START DATE   : 2026-11-01
# CODE RUN DATE: 2026-11-01
# TIME TAKEN   : ~5 min
# DESCRIPTION  : Follows the same importers through spring 2025: did firms that stopped China-area arrivals
#                arrive from elsewhere, from nowhere, and did they return? Baselines 2023-2024, other origins,
#                heterogeneity, and a sector-level link to survey expectations. Exploratory.
# INPUT FILES  : $RESTRICTED_DATA_DIR/customs/smb_trade_duns_biweekly_260512.parquet,
#                .../smb_firm_features_for_L2_260626.parquet, $RESTRICTED_DATA_DIR/survey/inputs/analysis_all.dta,
#                $RESTRICTED_DATA_DIR/survey/outputs/alignment_for_stata.dta
# OUTPUT FILES : analysis/33_firm_pause_switch_261101/01_*.csv (aggregates only; firm rows stay in memory)
# ============================================================
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd, statsmodels.api as sm
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from restricted_paths import SURVEY_DIR, CUSTOMS_DIR, JEV_DIR, OUTPUT_ROOT, out_dir, fresh_out_dir
warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve()
L = CUSTOMS_DIR; IA = SURVEY_DIR
OUT = out_dir("customs_pause_switch")
if (OUT / "01_decomposition_261101.csv").exists(): sys.exit("output exists; refusing to overwrite.")
F, T = "final_dunsno", "year_biweek"
d = pd.read_parquet(L / "smb_trade_duns_biweekly_260512.parquet", columns=[F, T, "origin", "weightkg", "dnb_sic1"])
d = d[d.origin.notna() & d.origin.ne("United States")].copy()
EU = ["Austria", "Belgium", "Bulgaria", "Croatia", "Cyprus", "Czech Republic", "Czechia", "Denmark", "Estonia", "Finland", "France", "Germany", "Greece", "Hungary", "Ireland", "Italy", "Latvia", "Lithuania", "Luxembourg", "Malta", "Netherlands", "Poland", "Portugal", "Romania", "Slovakia", "Slovenia", "Spain", "Sweden"]
EA = ["Japan", "South Korea", "Taiwan", "Vietnam", "Thailand", "Malaysia", "Indonesia", "Philippines", "Singapore", "Cambodia", "Laos", "Myanmar", "Brunei", "Mongolia", "North Korea"]
CN = ["China", "Hong Kong", "Macau", "Macao"]
GROUPS = {"china_area": CN, "east_asia_ex_china": EA, "eu": EU}
PH = {"april": [8, 9], "may": [11, 12], "july": [15, 16]}
d["year"] = d[T].str[:4].astype(int); d["bin"] = d[T].str[5:].astype(int)
d = d[d.year.between(2021, 2025)]
fs = pd.read_parquet(L / "smb_firm_features_for_L2_260626.parquet", columns=[F, "v2_dime_party_strict"])
party = fs.set_index(F).v2_dime_party_strict.astype(str).str.lower().str[:1]


def firm_panel(group, Y):
    # ★ CORE LOGIC: cohort fixed from the prior year only, so 2025 activity cannot select the cohort.
    pre = d[d.year == Y - 1]
    act = pre.groupby(F)[T].nunique()
    gset = pre.loc[pre.origin.isin(GROUPS[group]), F].unique()
    ids = pd.Index(act[(act >= 2) & act.index.isin(gset)].index)
    x = d[d[F].isin(ids) & (d.year == Y)]
    s = pd.DataFrame(index=ids)
    for name, bs in PH.items():
        pp = x[x.bin.isin(bs)]
        s[name + "_g"] = s.index.isin(pp.loc[pp.origin.isin(GROUPS[group]), F].unique())
        s[name + "_o"] = s.index.isin(pp.loc[~pp.origin.isin(GROUPS[group]), F].unique())
    prev = d[(d[F].isin(ids)) & (d.year.isin([Y - 1, Y - 2]))][[F, "origin"]].drop_duplicates()
    may = x[x.bin.isin(PH["may"]) & ~x.origin.isin(GROUPS[group])][[F, "origin"]].drop_duplicates()
    new = may.merge(prev.assign(seen=1), on=[F, "origin"], how="left")
    s["may_new_origin"] = s.index.isin(new.loc[new.seen.isna(), F].unique())
    s["n_g_bins_prior"] = pre[pre.origin.isin(GROUPS[group])].groupby(F)[T].nunique().reindex(ids).fillna(0)
    s["wt_prior"] = pre.groupby(F).weightkg.sum().reindex(ids).fillna(0)
    s["g_share_prior"] = (pre[pre.origin.isin(GROUPS[group])].groupby(F).weightkg.sum().reindex(ids).fillna(0) / s.wt_prior.replace(0, np.nan)).fillna(0)
    s = s[s.april_g].copy()
    s["state"] = np.where(s.may_g, "C", np.where(s.may_o, "O", "N"))
    s["year"] = Y
    return s


rows = []; panels = {}
for g in GROUPS:
    for Y in [2023, 2024, 2025]:
        s = firm_panel(g, Y); panels[(g, Y)] = s
        r = dict(group=g, year=Y, at_risk=len(s), C=s.state.eq("C").mean(), O=s.state.eq("O").mean(), N=s.state.eq("N").mean(), paused=s.state.ne("C").mean(),
                 paused_count=int(s.state.ne("C").sum()), new_origin_among_O=s.loc[s.state.eq("O"), "may_new_origin"].mean())
        for st in ["O", "N"]:
            z = s[s.state.eq(st)]
            r[f"{st}_return_july"] = z.july_g.mean(); r[f"{st}_other_only_july"] = (~z.july_g & z.july_o).mean(); r[f"{st}_none_july"] = (~z.july_g & ~z.july_o).mean()
        r["paused_return_july"] = s.loc[s.state.ne("C"), "july_g"].mean()
        rows.append(r)
dec = pd.DataFrame(rows); dec.to_csv(OUT / "01_decomposition_261101.csv", index=False)
pd.set_option("display.width", 200, "display.max_columns", 30, "display.float_format", lambda x: f"{x:.3f}")
print(dec.to_string(index=False))

# triple difference against other origins, with firm bootstrap-free analytic SE per group-year proportion
td = []
for g in ["east_asia_ex_china", "eu"]:
    for out in ["paused", "N", "O"]:
        v = {k: dec.set_index(["group", "year"]).loc[k, out] for k in [("china_area", 2025), ("china_area", 2024), ("china_area", 2023), (g, 2025), (g, 2024), (g, 2023)]}
        td.append(dict(control=g, outcome=out, china_2025_minus_base=v[("china_area", 2025)] - (v[("china_area", 2024)] + v[("china_area", 2023)]) / 2,
                       control_2025_minus_base=v[(g, 2025)] - (v[(g, 2024)] + v[(g, 2023)]) / 2))
td = pd.DataFrame(td); td["triple_diff"] = td.china_2025_minus_base - td.control_2025_minus_base
td.to_csv(OUT / "01_triple_diff_261101.csv", index=False); print(td.to_string(index=False))

# heterogeneity
pool = pd.concat([panels[("china_area", Y)].assign(firm=lambda x: x.index) for Y in [2024, 2025]])
pool["party"] = pool.firm.map(party).map({"r": "rep", "d": "dem"}).fillna("none")
pool["freq"] = pd.cut(pool.n_g_bins_prior, [-1, 3, 6, 12, 99], labels=["f1_3", "f4_6", "f7_12", "f13p"]).astype(str)
pool["share"] = pd.cut(pool.g_share_prior, [-.01, .25, .5, .75, 1.01], labels=["s0_25", "s25_50", "s50_75", "s75_100"]).astype(str)
pool["size"] = pd.qcut(pool.wt_prior.rank(method="first"), 4, labels=["w1", "w2", "w3", "w4"]).astype(str)
pool["y25"] = (pool.year == 2025).astype(float)
print("party x year:", pool.groupby(["party", "year"]).size().to_dict(), file=sys.stderr)
for c, base in [("freq", "f13p"), ("share", "s0_25"), ("size", "w4"), ("party", "dem")]:
    pass
hrows = []
for o, y in [("paused", pool.state.ne("C")), ("none_anywhere", pool.state.eq("N")), ("other_origin", pool.state.eq("O")), ("return_july", pool.july_g)]:
    q = pool.assign(y=y.astype(float))
    if o == "return_july": q = q[q.state.ne("C")]
    X = pd.DataFrame({"y25": q.y25})
    for c, base in [("freq", "f13p"), ("share", "s0_25"), ("size", "w4"), ("party", "dem")]:
        dm = pd.get_dummies(q[c], prefix=c).drop(columns=f"{c}_{base}").astype(float)
        X = pd.concat([X, dm, dm.mul(q.y25, axis=0).add_prefix("y25_x_")], axis=1)
    m = sm.OLS(q.y, sm.add_constant(X)).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(q.firm)[0]})
    for v in m.params.index:
        if v.startswith("y25"):
            hrows.append(dict(outcome=o, term=v, coef=m.params[v], se=m.bse[v], n=int(m.nobs)))
h = pd.DataFrame(hrows); h.to_csv(OUT / "01_heterogeneity_261101.csv", index=False); print(h.to_string(index=False))

# sector link
def sector(sic):
    try: s = int(str(sic).strip()[:2])
    except Exception: return np.nan
    if s <= 9: return 1
    if s <= 14: return 2
    if s <= 17: return 4
    if s <= 39: return 5
    if s <= 47: return 7
    if s == 48: return 8
    if s == 49: return 3
    if s <= 51: return 6
    if s == 58 or s == 70: return 17
    if s <= 59: return 20
    if s == 65: return 10
    if s <= 67: return 9
    if s == 73: return 11
    if s in (78, 79, 84): return 16
    if s in (80, 83): return 15
    if s == 82: return 14
    if s in (81, 87): return 11
    if s <= 89: return 18
    return np.nan


sec = d[[F, "dnb_sic1"]].drop_duplicates(F).set_index(F).dnb_sic1.map(sector)
z = pool.assign(sec=pool.firm.map(sec)).dropna(subset=["sec"])
g = z.groupby(["sec", "year"]).agg(n=("firm", "size"), paused=("state", lambda s: s.ne("C").mean()), none=("state", lambda s: s.eq("N").mean()), other=("state", lambda s: s.eq("O").mean())).reset_index().pivot(index="sec", columns="year")
ss = pd.DataFrame({"n_2025": g[("n", 2025)], "d_paused": g[("paused", 2025)] - g[("paused", 2024)], "d_none": g[("none", 2025)] - g[("none", 2024)], "d_other": g[("other", 2025)] - g[("other", 2024)]})
cols = ["responseid", "country", "industry", "long_term_prior", "short_term_prior", "import_tariff_expect"]
sv = pd.read_stata(IA / "inputs/analysis_all.dta", columns=cols, convert_categoricals=False)
sv = sv[sv.country == 1]
for c in cols[2:]: sv[c] = pd.to_numeric(sv[c], errors="coerce")
a = pd.read_stata(IA / "outputs/alignment_for_stata.dta", convert_categoricals=False)[["responseid", "wave", "importer_dum"]]
sv = sv.merge(a, on="responseid"); sv = sv[sv.importer_dum > 50]
w2 = sv[sv.wave.astype(str).str.contains("W2")].groupby("industry").agg(long_e=("long_term_prior", "mean"), short_e=("short_term_prior", "mean"), n_w2=("short_term_prior", "count"))
w4 = sv[sv.wave.astype(str).str.contains("W4")].groupby("industry").agg(tar_e=("import_tariff_expect", "mean"), n_w4=("import_tariff_expect", "count"))
lnk = ss.join(w2).join(w4); lnk.to_csv(OUT / "01_sector_link_261101.csv"); print(lnk.round(3).to_string())
keep = lnk[(lnk.n_2025 >= 200) & (lnk.n_w2.fillna(0) >= 15)]
out = []
for e in ["long_e", "short_e", "tar_e"]:
    for yv in ["d_paused", "d_none", "d_other"]:
        k = keep.dropna(subset=[e, yv])
        if len(k) >= 5:
            out.append(dict(expectation=e, outcome=yv, sectors=len(k), corr=k[e].corr(k[yv]), spearman=k[e].corr(k[yv], method="spearman")))
pd.DataFrame(out).to_csv(OUT / "01_sector_corr_261101.csv", index=False); print(pd.DataFrame(out).to_string(index=False))
