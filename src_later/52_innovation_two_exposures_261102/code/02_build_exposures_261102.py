# ============================================================
# START DATE   : 2026-11-02
# CODE RUN DATE: 2026-11-02
# TIME TAKEN   : ~10 min
# DESCRIPTION  : Build (i) HS6 x origin tariff INCREASE 2025 vs 2024 excluding Section 301, (ii) firm input shock IN and old-301
#                control from 2024 firm HS6 imports (private DUNS and Compustat gvkey), (iii) SIC4 output protection OUT from Census
#                2024 HS6 x country imports and the Pierce-Schott HS10-SIC concordance.
# INPUT FILES  : SMB_tariff hs6_panel (tariff h0 file, DUNS HS6 file, origin crosswalk), D&B attributes 260428,
#                PRIVATE_CACHE*.csv, ps/hs_sic_naics_imports_89_117_20180927.csv,
#                PRIVATE_SERVER_PATH{importer_hs6_origin_2024,importer_parent_link}.parquet (read only)
# OUTPUT FILES : cache firm_exposure.parquet, gvkey_exposure.parquet (firm level, local only); analysis/52_*/02_sic4_out.csv,
#                02_build_diagnostics.json (aggregates only)
# ============================================================
import sys, json
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[2]
ANA = ROOT / "analysis/52_innovation_two_exposures_261102"
CA = Path.home() / "private_cache/innov52_261102"
D = Path.home() / "PRIVATE_DATA_ROOT/SMB_tariff/intermediate_data/prep_reg_L2_update"
VOL = Path("PRIVATE_SERVER_PATH")
outs = [CA/"firm_exposure.parquet", CA/"gvkey_exposure.parquet", ANA/"02_sic4_out.csv", ANA/"02_build_diagnostics.json"]
for f in outs:
    if f.exists(): sys.exit(f"refusing to overwrite {f}")
diag = {}
CN = {"CHN", "HKG", "MAC"}

# ---- tariff increase (h0, iso3)
t = pd.read_parquet(D/"hs6_panel/smb_tariff_h0_origin_month_261029.parquet")
t = t[t.component != "section301"].copy() if False else t
t["yr"] = t.month.dt.year
inc_comp = t[(t.component != "section301") & t.yr.isin([2024, 2025])]
# ★ CORE LOGIC: missing product-origin-month rows inside a component's coverage are zero, so mean over 12 months = sum / 12.
m = inc_comp.groupby(["h0", "origin_iso3", "yr"]).rate_pct.sum().unstack("yr").fillna(0) / 12
m["dtau"] = m[2025] - m[2024]
dt = m[["dtau"]].reset_index().rename(columns={"origin_iso3": "iso3"})
s301 = t[(t.component == "section301") & (t.yr == 2024)].groupby(["h0", "origin_iso3"]).rate_pct.sum().div(12).rename("tau301").reset_index().rename(columns={"origin_iso3": "iso3"})
tt = dt.merge(s301, on=["h0", "iso3"], how="outer").fillna(0)
diag["dtau_mean_nonzero_rows"] = float(tt.dtau[tt.dtau != 0].mean()); diag["n_tariff_pairs"] = len(tt)

# ---- hs6 -> h0 map from the firm import file (value-dominant h0)
h = pd.read_parquet(D/"hs6_panel/smb_duns_hs6_origin_year_261024.parquet")
cw = pd.read_csv(D/"hs6_panel/origin_name_iso3_crosswalk_261024.csv")
h = h.merge(cw[["origin", "iso3"]], on="origin", how="left")
h["iso3"] = h.iso3.fillna("X:" + h.origin.fillna("NA"))
mp = h[h.h0.notna() & h.hs6.notna()].groupby(["hs6", "h0"]).trade_value.sum().reset_index().sort_values("trade_value", ascending=False).drop_duplicates("hs6")[["hs6", "h0"]]
h0set = set(tt.h0)

# ---- firm input shock, 2024 weights (same product keys as step 46: h0 mapped rows)
f24 = h[(h.year == 2024) & h.h0.notna() & (h.trade_value > 0)]
w = f24.groupby(["final_dunsno", "h0", "iso3"]).trade_value.sum().reset_index().merge(tt, on=["h0", "iso3"], how="left").fillna({"dtau": 0, "tau301": 0})
w["a"] = w.dtau * w.trade_value; w["b"] = w.tau301 * w.trade_value
g = w.groupby("final_dunsno")[["a", "b", "trade_value"]].sum()
fe = pd.DataFrame({"id": g.index, "IN": (g.a / g.trade_value).values, "O301": (g.b / g.trade_value).values, "imp24_h0": g.trade_value.values})
at = pd.read_parquet(D/"smb_duns_matched_firm_attributes_employment_260428.parquet")
at = at[at.YEAR == 2024].drop_duplicates("final_dunsno")
at["emp"] = at.EMPLOYEESALLSITES.where(at.EMPLOYEESALLSITES > 0, at.EMPLOYEESTHISSITE)
at["sic4"] = at.SIC1.astype("Int64")
fe = fe.merge(at[["final_dunsno", "emp", "sic4"]].rename(columns={"final_dunsno": "id"}), on="id", how="left")
diag["n_firms_IN"] = len(fe); diag["IN_mean"] = float(fe.IN.mean()); diag["IN_sd"] = float(fe.IN.std()); diag["O301_mean"] = float(fe.O301.mean())
diag["corr_IN_O301"] = float(fe[["IN", "O301"]].corr().iloc[0, 1])

# ---- gvkey input shock (Compustat), capped value as in step 48
hv = pd.read_parquet(VOL/"importer_hs6_origin_2024.parquet", columns=["importer_id", "hs6", "origin_iso3", "value_2024_capped"])
lk = pd.read_parquet(VOL/"importer_parent_link.parquet", columns=["importer_id", "gvkey", "included", "sic"])
lk = lk[lk.included == True].dropna(subset=["gvkey"]).drop_duplicates("importer_id")
hv = hv.merge(lk, on="importer_id").merge(mp, on="hs6", how="left")
hv["h0"] = hv.h0.fillna(hv.hs6.where(hv.hs6.isin(h0set)))
hv = hv.dropna(subset=["h0"]).rename(columns={"origin_iso3": "iso3"}).merge(tt, on=["h0", "iso3"], how="left").fillna({"dtau": 0, "tau301": 0})
hv["a"] = hv.dtau * hv.value_2024_capped; hv["b"] = hv.tau301 * hv.value_2024_capped
gg = hv.groupby("gvkey").agg(a=("a", "sum"), b=("b", "sum"), v=("value_2024_capped", "sum"), sic=("sic", "first"))
ge = pd.DataFrame({"gvkey": gg.index, "IN": (gg.a / gg.v).values, "O301": (gg.b / gg.v).values, "sic4": pd.to_numeric(gg.sic, errors="coerce").values})
diag["n_gvkey_IN"] = len(ge)

# ---- Census 2024 HS6 x country -> SIC4 output protection
cen = pd.concat([pd.read_csv(f, dtype=str) for f in sorted((CA/"census_hs6").glob("hs6_*.csv"))], ignore_index=True)
cen = cen.loc[:, ~cen.columns.duplicated()]
cen = cen[cen.CTY_CODE.str.match(r"^[1-9][0-9]{3}$")].copy()
cen["v"] = cen.GEN_VAL_YR.astype(float); cen["hs6"] = cen.I_COMMODITY.str[:6]
diag["census_total_2024_usd_countries"] = float(cen.v.sum()); diag["census_rows"] = len(cen)
names = cen.drop_duplicates("CTY_CODE")[["CTY_CODE", "CTY_NAME"]]
cwn = cw.assign(nm=cw.origin.str.upper())[["nm", "iso3"]].drop_duplicates("nm")
OVR = {"KOREA, SOUTH": "KOR", "TAIWAN": "TWN", "RUSSIA": "RUS", "VIETNAM": "VNM", "CHINA": "CHN", "HONG KONG": "HKG", "MACAU": "MAC", "BURMA": "MMR",
       "CZECH REPUBLIC": "CZE", "SLOVAKIA": "SVK", "TURKEY": "TUR", "UNITED KINGDOM": "GBR", "UNITED ARAB EMIRATES": "ARE", "IVORY COAST": "CIV", "MACEDONIA": "MKD",
       "CONGO, REPUBLIC OF THE": "COG", "CONGO, DEMOCRATIC REPUBLIC OF THE": "COD", "BOLIVIA": "BOL", "VENEZUELA": "VEN", "LAOS": "LAO", "IRAN": "IRN", "SYRIA": "SYR",
       "MOLDOVA": "MDA", "TANZANIA": "TZA", "BRUNEI": "BRN", "CAPE VERDE": "CPV", "SOUTH KOREA": "KOR"}
names["iso3"] = names.CTY_NAME.str.upper().map(OVR)
names = names.merge(cwn, left_on=names.CTY_NAME.str.upper(), right_on="nm", how="left", suffixes=("", "_cw"))
names["iso3"] = names.iso3.fillna(names.iso3_cw)
cen = cen.merge(names[["CTY_CODE", "iso3"]], on="CTY_CODE", how="left")
diag["census_value_share_country_mapped"] = float(cen.v[cen.iso3.notna()].sum() / cen.v.sum())
diag["census_unmapped_top"] = cen[cen.iso3.isna()].groupby("CTY_NAME").v.sum().sort_values(ascending=False).head(8).round(0).to_dict()
cen = cen.merge(mp, on="hs6", how="left")
cen["h0"] = cen.h0.fillna(cen.hs6.where(cen.hs6.isin(h0set)))
diag["census_value_share_h0_mapped"] = float(cen.v[cen.h0.notna()].sum() / cen.v.sum())
cen["iso3"] = cen.iso3.fillna("X")
cen = cen.merge(tt, on=["h0", "iso3"], how="left").fillna({"dtau": 0, "tau301": 0})
cen["a"] = cen.dtau * cen.v
hs = cen.groupby("hs6").agg(a=("a", "sum"), v=("v", "sum"))
hs["dtau"] = hs.a / hs.v
# Pierce-Schott: latest year, HS10 -> SIC4; HS6 split across SIC4 by HS10-line counts
ps = pd.read_csv(CA/"ps/hs_sic_naics_imports_89_117_20180927.csv", sep="\t", dtype=str)
ps = ps[ps.year == str(ps.year.astype(int).max())].copy()
diag["ps_year"] = int(ps.year.iloc[0])
ps["hs6"] = ps.commodity.str[:6]; ps["sic4"] = ps.sic.str.replace('"', "")
ps = ps[ps.sic4.str.match(r"^\d{4}$")]
sh = ps.groupby(["hs6", "sic4"]).size().rename("n").reset_index()
sh["share"] = sh.n / sh.groupby("hs6").n.transform("sum")
x = hs.reset_index().merge(sh, on="hs6", how="left")
diag["census_value_share_hs6_in_concordance"] = float(x.v[x.sic4.notna()].sum() / x.v.sum())
x = x.dropna(subset=["sic4"]); x["vw"] = x.v * x.share
o = x.groupby("sic4").apply(lambda d: pd.Series({"OUT": (d.a * d.share).sum() / d.vw.sum(), "imp_value_2024": d.vw.sum(), "n_hs6": d.hs6.nunique()})).reset_index()
o["sic4"] = o.sic4.astype(int)
o.to_csv(ANA/"02_sic4_out.csv", index=False)
diag["n_sic4_out"] = len(o); diag["OUT_mean"] = float(o.OUT.mean())
fe = fe.merge(o[["sic4", "OUT", "imp_value_2024"]], on="sic4", how="left")
ge = ge.merge(o[["sic4", "OUT"]], on="sic4", how="left")
fe.to_parquet(CA/"firm_exposure.parquet"); ge.to_parquet(CA/"gvkey_exposure.parquet")
mf = fe[(fe.sic4 // 100).between(20, 39)]
diag["mfg_firms"] = len(mf); diag["mfg_firms_with_OUT"] = int(mf.OUT.notna().sum())
diag["corr_IN_OUT_mfg"] = float(mf[["IN", "OUT"]].corr().iloc[0, 1])
diag["OUT_sd_between_sic4_mfg"] = float(mf.OUT.std())
json.dump(diag, open(ANA/"02_build_diagnostics.json", "w"), indent=1, default=str)
print(json.dumps(diag, indent=1, default=str))
