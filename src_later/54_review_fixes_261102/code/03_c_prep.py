# ============================================================
# START DATE   : 2026-11-02
# CODE RUN DATE: 2026-11-02
# TIME TAKEN   : ~10 min
# DESCRIPTION  : C prep. Cohort = firms with any import in 2022 or 2023; terciles from pre_trade_total with the code/41 cutoffs;
#                capped random sample (seed 261102, max 4000 per tercile, firms with >=2 origin groups in 2022-23);
#                firm x 7 origin-group x two-week cells for bins 1-111 (bin 79 dropped). Also writes origin-group mean duty table.
# INPUT FILES  :PRIVATE_DATA_ROOT/SMB_tariff/intermediate_data/prep_reg_L2_update/smb_trade_duns_biweekly_260512.parquet (read only),
#                .../china_sentiment/smb_china_sentiment_tariff_for_L2_261008.csv, .../hs6_panel/* (duty table)
# OUTPUT FILES : PRIVATE_CACHE (firm level, NOT in project), analysis/54.../03_cohort_counts_261102.csv,
#                03_group_duty_261102.csv, 03_bin_calendar_261102.csv
# ============================================================
import sys
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve(); TOPIC = HERE.parents[2]; SHARED = HERE.parents[7] / "PRIVATE_DATA_ROOT"
L = SHARED / "SMB_tariff/intermediate_data/prep_reg_L2_update"; CACHE = Path.home() / "private_cache/review54_261102"; OUT = TOPIC / "analysis/54_review_fixes_261102"
for t in [CACHE / "cells7.parquet", OUT / "03_cohort_counts_261102.csv", OUT / "03_group_duty_261102.csv", OUT / "03_bin_calendar_261102.csv"]:
    if t.exists(): sys.exit(f"refuse to overwrite {t}")
CN = ["China", "Hong Kong", "Macau", "Macao"]
EA = ["Japan", "South Korea", "Taiwan", "Thailand", "Malaysia", "Indonesia", "Philippines", "Singapore", "Cambodia", "Laos", "Myanmar", "Brunei", "Brunei Darussalam", "Mongolia", "North Korea"]
EU = ["Austria","Belgium","Bulgaria","Croatia","Cyprus","Czech Republic","Czechia","Denmark","Estonia","Finland","France","Germany","Greece","Hungary","Ireland","Italy","Latvia","Lithuania","Luxembourg","Malta","Netherlands","Poland","Portugal","Romania","Slovakia","Slovenia","Spain","Sweden"]
GRP = {**{o: "china" for o in CN}, **{o: "east_asia_other" for o in EA}, **{o: "eu" for o in EU}, "Vietnam": "vietnam", "Mexico": "mexico", "Canada": "canada"}
cal = pd.read_csv(L / "china_sentiment/smb_china_sentiment_tariff_for_L2_261008.csv", usecols=["year_biweek", "bin_index", "bin_start", "bin_end", "partial_bin"])
cal[~cal.partial_bin].to_csv(OUT / "03_bin_calendar_261102.csv", index=False)
p = pd.read_parquet(L / "smb_trade_duns_biweekly_260512.parquet", columns=["final_dunsno", "origin", "year_biweek", "trade_value", "weightkg", "volumeteu", "pre_trade_total"])
p = p[p.origin.notna() & p.origin.ne("United States")].copy()
p = p.merge(cal[["year_biweek", "bin_index", "partial_bin"]], on="year_biweek", how="left"); p = p[~p.partial_bin.fillna(False)]
p["grp"] = p.origin.map(GRP).fillna("rest")
pre = p[p.bin_index <= 52]
pre = pre[pre.trade_value > 0]
coh = pre.groupby("final_dunsno").agg(ng=("grp", "nunique")).reset_index()
tt = p.groupby("final_dunsno").pre_trade_total.first()
coh["pre_trade_total"] = coh.final_dunsno.map(tt)
coh["tercile"] = np.where(coh.pre_trade_total < 77400, "small", np.where(coh.pre_trade_total < 1390500, "medium", "large"))
rows = []; keep = []
rng = np.random.default_rng(261102)
for t, g in coh.groupby("tercile"):
    e = g[g.ng >= 2]; n = min(4000, len(e)); s = e.iloc[rng.permutation(len(e))[:n]]
    rows.append(dict(tercile=t, cohort_firms=len(g), with_2plus_groups=len(e), sampled=n)); keep.append(s)
pd.DataFrame(rows).to_csv(OUT / "03_cohort_counts_261102.csv", index=False)
keep = pd.concat(keep)[["final_dunsno", "tercile"]]
q = p.merge(keep, on="final_dunsno")
c = q.groupby(["final_dunsno", "grp", "bin_index", "tercile"], as_index=False).agg(trade_value=("trade_value", "sum"))
c.to_parquet(CACHE / "cells7.parquet"); print(rows, len(c))
# group mean duty table (2024 import-weighted) from the product tariff file
w = pd.read_parquet(L / "hs6_panel/smb_duns_hs6_origin_year_261024.parquet", columns=["h0", "origin", "year", "trade_value"])
w = w[(w.year == 2024) & w.h0.notna()]
cw = pd.read_csv(L / "hs6_panel/origin_name_iso3_crosswalk_261024.csv"); print(cw.columns.tolist())
