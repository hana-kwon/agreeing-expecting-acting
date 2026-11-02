# ============================================================
# START DATE   : 2026-11-02
# CODE RUN DATE: 2026-11-02
# TIME TAKEN   : ~10 min
# DESCRIPTION  : D prep. Country x HS2 x month Census imports (2023-01..2025-12) with (i) the headline country rate path of 36/07
#                (read from 07_c5_rate_path_monthly.csv), (ii) a product-level additional rate aggregated to HS2 with 2024 SMB import weights
#                by origin x H0 (all components; and without Section 232). EU = EUN (rates of DEU, weights summed over EU27 members).
# INPUT FILES  : data/36_census_imports_261102/hs2ch_*.csv, analysis/36.../07_c5_rate_path_monthly.csv,
PRIVATE_DATA_ROOT/SMB_tariff/intermediate_data/prep_reg_L2_update/hs6_panel/{smb_tariff_h0_origin_month_261029,smb_duns_hs6_origin_year_261024,origin_name_iso3_crosswalk_261024}
# OUTPUT FILES : PRIVATE_CACHE (public-data aggregates), analysis/54.../05_d_group_duty_261102.csv, 05_d_coverage_261102.csv
# ============================================================
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1].parent / "code/36_census_imports_261102"))
import numpy as np, pandas as pd
import common_261102 as cm
TOPIC = cm.ROOT; OUT = TOPIC / "analysis/54_review_fixes_261102"; CACHE = pathlib.Path.home() / "private_cache/review54_261102"
L = pathlib.Path(__file__).resolve().parents[7] / "PRIVATE_DATA_ROOT/SMB_tariff/intermediate_data/prep_reg_L2_update/hs6_panel"
assert L.exists(), L
for t in [CACHE / "d_panel.csv", OUT / "05_d_group_duty_261102.csv", OUT / "05_d_coverage_261102.csv"]:
    if t.exists(): sys.exit("refuse to overwrite " + str(t))
d = cm.load_hs2(); xw, un = cm.crosswalk(d[["CTY_CODE", "CTY_NAME"]]); key = {c: iso for iso, cs in xw.items() for c in cs}
d = d[d.CTY_CODE.isin(key)].copy(); d["iso"] = d.CTY_CODE.map(key)
d["m"] = pd.PeriodIndex(d.month, freq="M")
d = d[(d.m >= pd.Period("2023-01", "M")) & (d.m <= pd.Period("2025-12", "M"))]
imp = d.groupby(["iso", "hs2", "m"], as_index=False).gen.sum().rename(columns={"gen": "imp"})
# headline rates
hr = pd.read_csv(cm.ANALYSIS / "07_c5_rate_path_monthly.csv"); hr["m"] = pd.PeriodIndex(hr.month, freq="M"); hr = hr.rename(columns={"iso3": "iso", "tau_percent": "tau_head"})[["iso", "m", "tau_head"]]
# product rates
t = pd.read_parquet(L / "smb_tariff_h0_origin_month_261029.parquet")
t["is232"] = t.component.str.startswith("section232")
tot = t.groupby(["h0", "origin_iso3", "month"], as_index=False).rate_pct.sum().rename(columns={"rate_pct": "r_all"})
t232 = t[t.is232].groupby(["h0", "origin_iso3", "month"], as_index=False).rate_pct.sum().rename(columns={"rate_pct": "r_232"})
tot = tot.merge(t232, on=["h0", "origin_iso3", "month"], how="left").fillna({"r_232": 0.0}); tot["r_no232"] = tot.r_all - tot.r_232
# weights: 2024 SMB import value by origin iso x H0
w = pd.read_parquet(L / "smb_duns_hs6_origin_year_261024.parquet", columns=["h0", "origin", "year", "trade_value"])
w = w[(w.year == 2024) & w.h0.notna()]
cw = pd.read_csv(L / "origin_name_iso3_crosswalk_261024.csv")[["origin", "iso3"]]
w = w.merge(cw, on="origin", how="inner").groupby(["iso3", "h0"], as_index=False).trade_value.sum()
EU = [k for k in xw["EUN"]]   # census codes; need iso3 of members for weights
eu_iso = ["AUT","BEL","BGR","HRV","CYP","CZE","DNK","EST","FIN","FRA","DEU","GRC","HUN","IRL","ITA","LVA","LTU","LUX","MLT","NLD","POL","PRT","ROU","SVK","SVN","ESP","SWE"]
weu = w[w.iso3.isin(eu_iso)].groupby("h0", as_index=False).trade_value.sum().assign(iso3="EUN")
w = pd.concat([w[~w.iso3.isin(eu_iso)], weu])
w["hs2"] = w.h0.str[:2]
# rates for EUN = DEU rows
deu = tot[tot.origin_iso3 == "DEU"].copy(); deu["origin_iso3"] = "EUN"; tot = pd.concat([tot, deu])
months = pd.date_range("2024-01-01", "2025-12-01", freq="MS")
isos = sorted(set(imp.iso) & set(w.iso3))
rows = []
ww = w[w.iso3.isin(isos)].copy()
for mth in months:
    r = tot[tot.month == mth]
    # full product list per iso from weights; missing rate row = 0 within coverage (dictionary)
    x = ww.merge(r[["h0", "origin_iso3", "r_all", "r_no232"]], left_on=["h0", "iso3"], right_on=["h0", "origin_iso3"], how="left").fillna({"r_all": 0.0, "r_no232": 0.0})
    x["a"] = x.r_all * x.trade_value; x["b"] = x.r_no232 * x.trade_value
    g = x.groupby(["iso3", "hs2"], as_index=False).agg(a=("a", "sum"), b=("b", "sum"), w=("trade_value", "sum"))
    g["tau_prod"] = g.a / g.w; g["tau_prod_no232"] = g.b / g.w; g["m"] = pd.Period(mth, "M"); rows.append(g[["iso3", "hs2", "m", "tau_prod", "tau_prod_no232", "w"]])
pr = pd.concat(rows).rename(columns={"iso3": "iso"})
pan = imp.merge(hr, on=["iso", "m"], how="left").merge(pr, on=["iso", "hs2", "m"], how="left")
pan["mi"] = (pan.m.dt.year - 2023) * 12 + pan.m.dt.month - 1
pan = pan.drop(columns="m")
cov = dict(cells_all=int(pan.groupby(["iso", "hs2"]).ngroups), cells_with_product_weights=int(pan[pan.tau_prod.notna()].groupby(["iso", "hs2"]).ngroups),
           share_import_value_2025_with_weights=float(pan[(pan.mi >= 24) & pan.tau_prod.notna()].imp.sum() / pan[pan.mi >= 24].imp.sum()), countries=len(isos))
pd.DataFrame([cov]).to_csv(OUT / "05_d_coverage_261102.csv", index=False); print(cov)
pan.to_csv(CACHE / "d_panel.csv", index=False)
# group-level mean rates for C3 table: import-weighted by origin x month (2024 SMB weights)
gl = {"china": ["CHN", "HKG"], "vietnam": ["VNM"], "mexico": ["MEX"], "canada": ["CAN"], "eu": ["EUN"], "east_asia_other": ["JPN", "KOR", "TWN", "THA", "MYS", "IDN", "PHL", "SGP"]}
z = []
for gname, isl in gl.items():
    for mth in months:
        r = tot[(tot.month == mth) & tot.origin_iso3.isin(isl)]
        x = w[w.iso3.isin(isl)].merge(r[["h0", "origin_iso3", "r_all"]], left_on=["h0", "iso3"], right_on=["h0", "origin_iso3"], how="left").fillna({"r_all": 0.0})
        z.append(dict(group=gname, month=str(mth.date())[:7], mean_added_rate=(x.r_all * x.trade_value).sum() / x.trade_value.sum()))
pd.DataFrame(z).pivot(index="month", columns="group", values="mean_added_rate").round(2).to_csv(OUT / "05_d_group_duty_261102.csv")
print(pd.DataFrame(z).pivot(index="month", columns="group", values="mean_added_rate").round(1).iloc[[0,12,14,16,18,20,22,23]])
