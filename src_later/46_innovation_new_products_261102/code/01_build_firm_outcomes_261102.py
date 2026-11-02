# ============================================================
# START DATE   : 2026-11-02
# CODE RUN DATE: 2026-11-02
# TIME TAKEN   : ~3 min run
# DESCRIPTION  : Build firm-year new-product / new-pair / new-origin outcomes (2023-2026) from the DUNS x HS6 x origin x year
#                customs panel, plus 2024-weighted 2025 tariff exposure (total, 232, IEEPA, 301) and firm covariates.
# INPUT FILES  : SMB_tariff/.../hs6_panel/{smb_duns_hs6_origin_year_261024.parquet, smb_tariff_h0_origin_month_261029.parquet,
#                origin_name_iso3_crosswalk_261024.csv}; ../smb_duns_matched_firm_attributes_employment_260428.parquet;
#                ../smb_firm_features_for_L2_260626.parquet  (all read only)
# OUTPUT FILES : PRIVATE_CACHE (firm-year outcomes, local only),
#                firm_cov_261102.parquet (firm covariates + exposure, local only)
# ============================================================
import sys, os
from pathlib import Path
import numpy as np, pandas as pd

D = Path.home()/"PRIVATE_DATA_ROOT/SMB_tariff/intermediate_data/prep_reg_L2_update"
OUT = Path.home()/"private_cache/innov46_261102"
OUT.mkdir(parents=True, exist_ok=True)
f_long, f_cov = OUT/"firm_long_261102.parquet", OUT/"firm_cov_261102.parquet"
for f in (f_long, f_cov):
    if f.exists(): sys.exit(f"refusing to overwrite {f}")

CN = {"CHN", "HKG", "MAC"}
h = pd.read_parquet(D/"hs6_panel/smb_duns_hs6_origin_year_261024.parquet")
cw = pd.read_csv(D/"hs6_panel/origin_name_iso3_crosswalk_261024.csv")[["origin", "iso3"]]
h = h.merge(cw, on="origin", how="left")
h["iso3"] = h["iso3"].fillna("X:" + h["origin"].fillna("NA"))
h = h[h.trade_value > 0].copy()

# ★ CORE LOGIC: product key = H0 code when mapped, else raw hs6 flagged "r:". A product counts as new only if neither its key
# nor its raw hs6 code was imported earlier by the same firm, so edition changes (hs6 -> h0 remaps) are not counted as new products.
h["coded"] = h.hs6.notna() & h.hs_status.isin(["ok", "not_in_hs2022"])
h["key"] = np.where(h.h0.notna(), h.h0, "r:" + h.hs6.fillna(""))
h["ch"] = h.hs6.str[:2]
h["cn"] = h.iso3.isin(CN)
h = h.rename(columns={"final_dunsno": "id"})

# --- firm-year totals & any-import flags (history existence uses ALL rows incl. uncoded)
anyimp = h.groupby(["id", "year"]).trade_value.sum().unstack().fillna(0) > 0
tot_all = h.groupby(["id", "year"]).trade_value.sum()

c = h[h.coded].copy()
c["pair"] = c.key + "|" + c.iso3
c["rawpair"] = c.hs6 + "|" + c.iso3

def first_year(df, col):
    return df.groupby(["id", col]).year.transform("min")

c["fy_key"] = first_year(c, "key"); c["fy_raw"] = first_year(c, "hs6")
c["fy_pair"] = first_year(c, "pair"); c["fy_rawpair"] = first_year(c, "rawpair")
c["fy_orig"] = first_year(c, "iso3"); c["fy_ch"] = first_year(c, "ch")

def newflag(df, fy_a, fy_b, group_cols):
    """product-level newness: first seen in year t under its key AND no raw-code precedent among that unit's rows."""
    a = df[fy_a] == df.year
    mb = df.groupby(group_cols + ["year"])[fy_b].transform("min")
    return a & (mb == df.year)

c["new_key"] = newflag(c, "fy_key", "fy_raw", ["id", "key"])
c["new_pair"] = newflag(c, "fy_pair", "fy_rawpair", ["id", "pair"])
c["new_orig"] = c.fy_orig == c.year
c["new_ch"] = c.fy_ch == c.year
v = c.trade_value
for nm in ["new_key", "new_pair", "new_orig", "new_ch"]:
    c[nm + "_v"] = np.where(c[nm], v, 0.0)
c["newkey_cn_v"] = np.where(c.new_key & c.cn, v, 0.0)
c["newkey_nc_v"] = np.where(c.new_key & ~c.cn, v, 0.0)

g = c.groupby(["id", "year"])
res = pd.DataFrame({
    "coded_val": g.trade_value.sum(),
    "n_hs6": g.key.nunique(),
    "new_vs_num": g.new_key_v.sum(), "newpair_vs_num": g.new_pair_v.sum(),
    "newch_vs_num": g.new_ch_v.sum(),
    "new_vs_CN_num": g.newkey_cn_v.sum(), "new_vs_nonCN_num": g.newkey_nc_v.sum(),
})
# product-level counts
pk = c.groupby(["id", "year", "key"]).agg(new=("new_key", "max"), cn=("cn", "max"), nc=("cn", lambda s: (~s).any())).reset_index()
pk["new_cn"] = pk.new & pk.cn; pk["new_nc"] = pk.new & pk.nc
gk = pk.groupby(["id", "year"])
res["new_n"] = gk.new.sum(); res["new_n_CN"] = gk.new_cn.sum(); res["new_n_nonCN"] = gk.new_nc.sum()
pp = c.groupby(["id", "year", "pair"]).new_pair.max().reset_index().groupby(["id", "year"]).new_pair.sum()
res["newpair_n"] = pp
oo = c.groupby(["id", "year", "iso3"]).new_orig.max().reset_index().groupby(["id", "year"]).new_orig.sum()
res["neworig_n"] = oo
cc = c.groupby(["id", "year", "ch"]).new_ch.max().reset_index().groupby(["id", "year"]).new_ch.sum()
res["newch_n"] = cc
res = res.reset_index()

# dropped products: keys in t-1 absent in t
pset = pk[["id", "year", "key"]].copy()
prev = pset.copy(); prev["year"] += 1
mm = prev.merge(pset, on=["id", "year", "key"], how="left", indicator=True)
dropn = (mm._merge == "left_only").groupby([mm.id, mm.year]).sum().rename("drop_n").reset_index()
res = res.merge(dropn, on=["id", "year"], how="left")
res["drop_n"] = res.drop_n.fillna(0)

# history existence: any import (all rows) in some year 2022..t-1
hist = {}
for t in range(2023, 2027):
    cols = [y for y in range(2022, t)]
    hist[t] = anyimp[cols].any(axis=1)
res = res[res.year >= 2023].copy()
res["has_hist"] = [bool(hist[t].get(i, False)) for i, t in zip(res.id, res.year)]
res = res[res.year >= 2023].copy()
for o in ["new_vs", "newpair_vs", "newch_vs", "new_vs_CN", "new_vs_nonCN"]:
    res[o] = res[o + "_num"] / res.coded_val
res["new_any"] = (res.new_n > 0).astype(float)
res["new_frac"] = res.new_n / res.n_hs6
res.drop(columns=[c_ for c_ in res.columns if c_.endswith("_num")], inplace=True)
res.to_parquet(f_long)
print("long", res.shape, res.groupby("year").agg(n=("id", "size"), hist=("has_hist", "sum")))

# ---------------- covariates and exposure (2024 weights) ----------------
f24 = c[c.year == 2024]
# ★ IMPORTANT: exposure uses only 2024 value with a mapped h0 code (denominator), tariff rows missing inside coverage = 0.
t = pd.read_parquet(D/"hs6_panel/smb_tariff_h0_origin_month_261029.parquet")
t25 = t[(t.month >= "2025-01-01") & (t.month <= "2025-12-01")].copy()
grp = {"section301": "r301", "china_ieepa_s122": "ieepa", "reciprocal_ieepa": "ieepa", "other_ieepa": "ieepa"}
t25["g"] = t25.component.map(grp).fillna("r232")
tt = t25.groupby(["h0", "origin_iso3", "g"]).rate_pct.sum().div(12).unstack("g").fillna(0).reset_index()
tt["tot"] = tt.r232 + tt.ieepa + tt.r301
w = f24[f24.h0.notna()].groupby(["id", "h0", "iso3"]).trade_value.sum().reset_index()
w = w.merge(tt, left_on=["h0", "iso3"], right_on=["h0", "origin_iso3"], how="left")
for k in ["r232", "ieepa", "r301", "tot"]:
    w[k] = w[k].fillna(0) * w.trade_value
ex = w.groupby("id")[["r232", "ieepa", "r301", "tot", "trade_value"]].sum()
for k, nm in [("tot", "exp_tot"), ("r232", "exp_232"), ("ieepa", "exp_ieepa"), ("r301", "exp_301")]:
    ex[nm] = ex[k] / ex.trade_value
ex = ex[["exp_tot", "exp_232", "exp_ieepa", "exp_301"]]

a24 = h[h.year == 2024].groupby("id").agg(imp24=("trade_value", "sum"), rec24=("n_records", "sum"),
                                         cn24=("trade_value", lambda s: 0.0))
a24["cn_share24"] = h[(h.year == 2024) & h.cn].groupby("id").trade_value.sum().reindex(a24.index).fillna(0) / a24.imp24
a24["cn24"] = a24.cn_share24
a24["n_orig24"] = h[h.year == 2024].groupby("id").iso3.nunique()
a24["n_hs6_24"] = f24.groupby("id").key.nunique()
a24["hist_n_pre24"] = c[c.year < 2024].groupby("id").key.nunique()
a24["years_imp_22_24"] = anyimp[[2022, 2023, 2024]].sum(axis=1)
cov = a24.join(ex, how="left")
cov["imp_2022"] = anyimp[2022]; cov["imp_2023"] = anyimp[2023]; cov["imp_2024"] = anyimp[2024]; cov["imp_2025"] = anyimp[2025]
cov = cov.reindex(anyimp.index).reset_index()
at = pd.read_parquet(D/"smb_duns_matched_firm_attributes_employment_260428.parquet")
at = at[at.YEAR == 2024].drop_duplicates("final_dunsno")
at["emp"] = at.EMPLOYEESALLSITES.where(at.EMPLOYEESALLSITES > 0, at.EMPLOYEESTHISSITE)
at["sic2"] = np.floor(at.SIC1 / 100)
cov = cov.merge(at[["final_dunsno", "emp", "sic2", "SLS"]].rename(columns={"final_dunsno": "id"}), on="id", how="left")
ff = pd.read_parquet(D/"smb_firm_features_for_L2_260626.parquet", columns=["final_dunsno", "dime_party"]).rename(columns={"final_dunsno": "id"})
cov = cov.merge(ff, on="id", how="left")
cov.to_parquet(f_cov)
print("cov", cov.shape); print(cov[["exp_tot", "exp_232", "exp_ieepa", "exp_301", "cn_share24", "emp", "sic2"]].describe().T)
print(cov.imp_2024.sum(), cov.exp_tot.notna().sum(), cov.sic2.notna().sum(), cov.emp.notna().sum(), cov.dime_party.value_counts())
