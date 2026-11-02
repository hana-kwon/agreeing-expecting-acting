# ============================================================
# START DATE   : 2026-11-02
# CODE RUN DATE: 2026-11-02
# TIME TAKEN   : ~20 min incl. writing
# DESCRIPTION  : C3 2025 China episode in public data: China-area (CN+HK+MO) imports by HS2, drop from Mar-Apr to May, recovery by Jul-Aug and the
#                latest month, versus the same calendar months of 2023 and 2024, and versus Vietnam, Mexico, other SE Asia, EU27. Descriptive.
# INPUT FILES  : data/36_census_imports_261102/hs2ch_*.csv
# OUTPUT FILES : analysis/36_census_imports_261102/04_c3_*.csv
# ============================================================
import numpy as np, pandas as pd
from scipy import stats as st
import common_261102 as cm

OUT = cm.get_out()
RNG = np.random.default_rng(20261102)
d = cm.load_hs2(); LAST = max(d.month)
idx = cm.months_index(LAST); n_m = len(idx)
nm = d.drop_duplicates("CTY_CODE").set_index("CTY_CODE").CTY_NAME.str.upper()
byname = lambda names: [c for c, n in nm.items() if n in names]
GROUPS = {"china_area": byname({"CHINA", "HONG KONG", "MACAU"}), "vietnam": byname({"VIETNAM"}), "mexico": byname({"MEXICO"}),
          "other_seasia": byname({"THAILAND", "MALAYSIA", "INDONESIA", "SINGAPORE", "PHILIPPINES", "CAMBODIA", "LAOS", "BURMA", "BRUNEI"}),
          "eu27": byname(set(cm.EU27)), "world": list(nm.index)}
G = {}
for g, cs in GROUPS.items():
    x = d[d.CTY_CODE.isin(cs)].groupby(["hs2", "month"]).gen.sum().unstack("month")
    x.columns = pd.PeriodIndex(x.columns, freq="M"); G[g] = x.reindex(columns=idx).fillna(0.0)
P = pd.Period
def win(x, y):
    """Episode windows for year y (rows = HS2 or one total row)."""
    g = lambda m: x[P(f"{y}-{m:02d}", "M")]
    latest = P(f"{y + 1}-08", "M")
    return dict(A=(g(3) + g(4)) / 2, MAY=g(5), JA=(g(7) + g(8)) / 2, LATEST=(x[latest] if latest in idx else np.nan * g(3)))
def metrics(x, y):
    w = win(x, y); out = {}
    out.update(A=w["A"], MAY=w["MAY"], JA=w["JA"], LATEST=w["LATEST"])
    out["DROP"] = 1 - w["MAY"] / w["A"]
    den = (w["A"] - w["MAY"])
    out["RECOV"] = np.where(den > 0, (w["JA"] - w["MAY"]) / den, np.nan) if hasattr(den, "__len__") else ((w["JA"] - w["MAY"]) / den if den > 0 else np.nan)
    out["JA_over_A"] = w["JA"] / w["A"]; out["LATEST_over_A"] = w["LATEST"] / w["A"]; out["MAY_over_A"] = w["MAY"] / w["A"]
    return out
# --- aggregate (sum over HS2) table ---------------------------------------------------------------------------------------
rows = []
for g, x in G.items():
    tot = x.sum(axis=0).to_frame().T
    for y in (2023, 2024, 2025):
        m = {k: float(np.asarray(v).ravel()[0]) for k, v in metrics(tot, y).items()}
        rows.append(dict(group=g, year=y, **m))
agg = pd.DataFrame(rows)
# ★ CORE LOGIC (chosen after seeing results, see 00_spec change log): in normal years May exceeds Mar-Apr (seasonal rise), so the raw DROP/RECOV of the
# spec are negative/undefined in 2023-24. Seasonally adjusted versions divide the 2025 ratios by the mean 2023-24 ratios.
def adj_cols(df, a, b):
    """df has MAY_over_A, JA_over_A, LATEST_over_A for 2025 (a) and 2023/24 (b)."""
    rMAY = a["MAY_over_A"] / b["MAY_over_A"].mean(); rJA = a["JA_over_A"] / b["JA_over_A"].mean()
    return rMAY, rJA
adj = []
for g in agg.group.unique():
    x = agg[agg.group == g].set_index("year")
    rMAY, rJA = adj_cols(agg, x.loc[2025], x.loc[[2023, 2024]])
    adj.append(dict(group=g, rMAY_adj=rMAY, rJA_adj=rJA, DROP_adj=1 - rMAY, RECOV_adj=(rJA - rMAY) / (1 - rMAY) if rMAY < 1 else np.nan,
                    LATEST_over_A_2025=x.loc[2025, "LATEST_over_A"], LATEST_over_A_2023=x.loc[2023, "LATEST_over_A"],
                    LATEST_adj_vs2023=x.loc[2025, "LATEST_over_A"] / x.loc[2023, "LATEST_over_A"]))
cm.safe_write(pd.DataFrame(adj), OUT / "04_c3_aggregate_adjusted.csv")
# share of world imports from the group (same windows)
sh = []
for g in ("china_area", "vietnam", "mexico", "other_seasia", "eu27"):
    tg, tw = G[g].sum(axis=0), G["world"].sum(axis=0)
    for y in (2023, 2024, 2025):
        a = ((tg / tw)[P(f"{y}-03", "M")] + (tg / tw)[P(f"{y}-04", "M")]) / 2
        sh.append(dict(group=g, year=y, share_MarApr=a, share_May=(tg / tw)[P(f"{y}-05", "M")],
                       share_JulAug=((tg / tw)[P(f"{y}-07", "M")] + (tg / tw)[P(f"{y}-08", "M")]) / 2))
cm.safe_write(agg, OUT / "04_c3_aggregate.csv")
cm.safe_write(pd.DataFrame(sh), OUT / "04_c3_shares.csv")
# --- monthly series index for the figure (Mar-Apr avg of the year = 100), May..Dec of each year ---------------------------------
ser = []
for g in ("china_area", "vietnam", "mexico", "other_seasia", "eu27"):
    tot = G[g].sum(axis=0)
    for y in (2023, 2024, 2025):
        A = (tot[P(f"{y}-03", "M")] + tot[P(f"{y}-04", "M")]) / 2
        for m in range(1, 13):
            p = P(f"{y}-{m:02d}", "M")
            if p in idx: ser.append(dict(group=g, year=y, month=m, index=100 * tot[p] / A))
    for p in idx:
        if p.year == 2026: ser.append(dict(group=g, year=2026, month=p.month, index=100 * tot[p] / ((tot[P("2025-03", "M")] + tot[P("2025-04", "M")]) / 2)))
cm.safe_write(pd.DataFrame(ser), OUT / "04_c3_series.csv")
# --- HS2 level --------------------------------------------------------------------------------------------------------------------
hs = {}
for y in (2023, 2024, 2025):
    mc = metrics(G["china_area"], y)
    t = pd.DataFrame({f"{k}_{y}": v for k, v in mc.items()}, index=G["china_area"].index)
    for g in ("vietnam", "mexico", "other_seasia", "eu27"):
        mg = win(G[g], y)
        t[f"{g}_delta_{y}"] = (mg["JA"] - mg["A"]) / mc["A"]                  # change of origin g Mar-Apr -> Jul-Aug as share of China-area Mar-Apr
        t[f"{g}_JAoverA_{y}"] = mg["JA"] / mg["A"]
    hs[y] = t
H = pd.concat(hs.values(), axis=1)
H["others_delta_2025"] = H[[f"{g}_delta_2025" for g in ("vietnam", "mexico", "other_seasia", "eu27")]].sum(axis=1)
for y in (2023, 2024): H[f"others_delta_{y}"] = H[[f"{g}_delta_{y}" for g in ("vietnam", "mexico", "other_seasia", "eu27")]].sum(axis=1)
H["rMAY_adj"] = H.MAY_over_A_2025 / H[["MAY_over_A_2023", "MAY_over_A_2024"]].mean(axis=1)
H["rJA_adj"] = H.JA_over_A_2025 / H[["JA_over_A_2023", "JA_over_A_2024"]].mean(axis=1)
H["DROP_adj"] = 1 - H.rMAY_adj
H["RECOV_adj"] = np.where(H.DROP_adj > 0.05, (H.rJA_adj - H.rMAY_adj) / H.DROP_adj, np.nan).clip(-2, 3)   # clipped: tiny drops blow up the ratio
H["LATEST_adj_vs2023"] = H.LATEST_over_A_2025 / H.LATEST_over_A_2023
H["others_did"] = H.others_delta_2025 - H[["others_delta_2023", "others_delta_2024"]].mean(axis=1)
H = H.reset_index().rename(columns={"hs2": "hs2"})
cm.safe_write(H, OUT / "04_c3_hs2.csv")
# --- cross-section ------------------------------------------------------------------------------------------------------------------
S = H[(H.A_2025 >= 5e7) & np.isfinite(H.DROP_adj)].reset_index(drop=True)
def xs(df, xcol, ycol):
    r, _ = st.spearmanr(df[xcol], df[ycol])
    w = df.A_2025.values; X = np.c_[np.ones(len(df)), df[xcol].values]
    b = np.linalg.lstsq(X * np.sqrt(w)[:, None], df[ycol].values * np.sqrt(w), rcond=None)[0]
    return r, b[1]
rows = []
for xcol, ycol, lab in (("DROP_adj", "RECOV_adj", "adjusted drop (May) vs share of drop recovered by Jul-Aug (drop>5% only)"),
                        ("DROP_adj", "others_did", "adjusted drop vs gain of other origins, 2025 minus 2023-24 (share of China-area base)"),
                        ("DROP_adj", "LATEST_adj_vs2023", "adjusted drop vs 2026-08 level relative to Mar-Apr 2025 (÷ 2024-08/Mar-Apr 2023)")):
    s = S.dropna(subset=[xcol, ycol]); s = s[(s[ycol].abs() < 20)]
    r, b = xs(s, xcol, ycol)
    bs = [xs(s.iloc[RNG.integers(0, len(s), len(s))], xcol, ycol) for _ in range(500)]
    bs = np.array(bs)
    rows.append(dict(test=lab, n_hs2=len(s), spearman=r, spearman_se=bs[:, 0].std(ddof=1), wls_slope=b, slope_se=bs[:, 1].std(ddof=1)))
cm.safe_write(pd.DataFrame(rows), OUT / "04_c3_cross.csv")
print(agg.round(3).to_string()); print(pd.DataFrame(adj).round(3).to_string()); print(pd.DataFrame(sh).round(4).to_string()); print(pd.DataFrame(rows).round(3).to_string()); print({g: len(c) for g, c in GROUPS.items()})
