# ============================================================
# START DATE   : 2026-11-02
# CODE RUN DATE: 2026-11-02
# TIME TAKEN   : ~5 min
# DESCRIPTION  : G prep. Monthly additional tariff rate by origin (sum of components of the product-level file, 2024 SMB import weights by
#                origin x H0; EU = DEU rates with weights summed over EU27; CHN origin includes Hong Kong/Macau codes' rate = CHN rows) mapped
#                to the bins of analysis 50 at lags k = 0 and 8 weeks. Replaces the in-force indicator / step rate of 50.
# INPUT FILES  : hs6_panel tariff + weights files (read only), analysis/50.../01_origin_map_261102.csv, 03_bin_calendar_261102.csv (54)
# OUTPUT FILES : PRIVATE_CACHE (public-rule aggregates by iso3 x bin)
# ============================================================
import sys
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve(); TOPIC = HERE.parents[2]
L = HERE.parents[7] / "PRIVATE_DATA_ROOT/SMB_tariff/intermediate_data/prep_reg_L2_update/hs6_panel"
CACHE = Path.home() / "private_cache/review54_261102"; OUTF = CACHE / "rate_prod.parquet"
if OUTF.exists(): sys.exit("refuse to overwrite")
om = pd.read_csv(TOPIC / "analysis/50_customs_country_signal_261102/01_origin_map_261102.csv").dropna(subset=["iso3"])
isos = sorted(set(om.iso3))
cal = pd.read_csv(TOPIC / "analysis/54_review_fixes_261102/03_bin_calendar_261102.csv", parse_dates=["bin_start"]); cal = cal[cal.bin_index >= 53]
t = pd.read_parquet(L / "smb_tariff_h0_origin_month_261029.parquet")
tot = t.groupby(["h0", "origin_iso3", "month"], as_index=False).rate_pct.sum()
w = pd.read_parquet(L / "smb_duns_hs6_origin_year_261024.parquet", columns=["h0", "origin", "year", "trade_value"]); w = w[(w.year == 2024) & w.h0.notna()]
cw = pd.read_csv(L / "origin_name_iso3_crosswalk_261024.csv")[["origin", "iso3"]]
w = w.merge(cw, on="origin").groupby(["iso3", "h0"], as_index=False).trade_value.sum()
eu = ["AUT","BEL","BGR","HRV","CYP","CZE","DNK","EST","FIN","FRA","DEU","GRC","HUN","IRL","ITA","LVA","LTU","LUX","MLT","NLD","POL","PRT","ROU","SVK","SVN","ESP","SWE"]
weu = w[w.iso3.isin(eu)].groupby("h0", as_index=False).trade_value.sum().assign(iso3="EUN"); w = pd.concat([w[~w.iso3.isin(eu)], weu])
deu = tot[tot.origin_iso3 == "DEU"].assign(origin_iso3="EUN"); tot = pd.concat([tot, deu])
mon = {}
for iso in isos:
    wi = w[w.iso3 == iso]
    if wi.empty: continue
    for m in pd.date_range("2024-01-01", "2025-12-01", freq="MS"):
        r = tot[(tot.month == m) & (tot.origin_iso3 == iso)][["h0", "rate_pct"]]
        x = wi.merge(r, on="h0", how="left").fillna({"rate_pct": 0.0})
        mon[(iso, m)] = (x.rate_pct * x.trade_value).sum() / x.trade_value.sum()
rows = []
for iso in isos:
    for _, b in cal.iterrows():
        rec = dict(iso3=iso, bin_index=b.bin_index)
        for k in (0, 8):
            d = b.bin_start - pd.Timedelta(days=7 * k); key = (iso, pd.Timestamp(d.year, d.month, 1))
            rec[f"Rprod_k{k}"] = mon.get(key, np.nan) / 10.0
        rows.append(rec)
r = pd.DataFrame(rows); r.to_parquet(OUTF)
print(r.groupby("iso3").Rprod_k0.agg(["count", "mean", "max"]).round(2).T.to_string()); print("isos without weights:", [i for i in isos if (i, pd.Timestamp("2025-06-01")) not in mon])
