# ============================================================
# START DATE   : 2026-11-02
# CODE RUN DATE: 2026-11-02
# TIME TAKEN   : ~3 min
# DESCRIPTION  : Firm attributes for the extra specifications (2024 attributes only): number of distinct import origins,
#                number of non-China origins, China value share, "had a non-China origin in 2024", D&B 2024 employment bins,
#                2-digit SIC. Firm rows stay in PRIVATE_CACHE only; only counts are written to the project.
# INPUT FILES  : smb_trade_duns_biweekly_260512.parquet, smb_duns_matched_firm_attributes_employment_260428.parquet (read only),
#                PRIVATE_CACHE
# OUTPUT FILES : PRIVATE_CACHE; analysis/41_.../07_extras_counts_261102.csv
# ============================================================
import sys
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve(); TOPIC = HERE.parents[2]; SHARED = HERE.parents[7] / "PRIVATE_DATA_ROOT"
L = SHARED / "SMB_tariff/intermediate_data/prep_reg_L2_update"
CACHE = Path.home() / "private_cache/customs40_261102"; OUT = TOPIC / "analysis/41_customs_team_spec_261102"
for t in [CACHE / "firm_attrs.parquet", OUT / "07_extras_counts_261102.csv"]:
    if t.exists(): sys.exit(f"output exists; refusing to overwrite: {t}")
CN = ["China", "Hong Kong", "Macau", "Macao"]
firms = pd.read_parquet(CACHE / "firms.parquet")
p = pd.read_parquet(L / "smb_trade_duns_biweekly_260512.parquet", columns=["final_dunsno", "origin", "year_biweek", "trade_value", "weightkg", "dnb_sic1"])
sic = p.groupby("final_dunsno").dnb_sic1.first()
p = p[p.final_dunsno.isin(firms.final_dunsno) & p.year_biweek.str.startswith("2024") & p.origin.notna() & p.origin.ne("United States") & (p.weightkg > 0)]
p["cn"] = p.origin.isin(CN)
# ★ CORE LOGIC: "alternatives already in place" is measured from 2024 arrivals only (before the 2025 statements);
# a firm with no 2024 arrivals at all gets NaN, not 0, because "no alternative" cannot be told from "no information".
g = p.groupby("final_dunsno")
a = pd.DataFrame({"n_origins_2024": g.origin.nunique(),
                  "n_nonchina_2024": p[~p.cn].groupby("final_dunsno").origin.nunique(),
                  "val_total": g.trade_value.sum(), "val_cn": p[p.cn].groupby("final_dunsno").trade_value.sum()})
a["n_nonchina_2024"] = a.n_nonchina_2024.fillna(0); a["val_cn"] = a.val_cn.fillna(0)
a["china_share_2024"] = np.where(a.val_total > 0, a.val_cn / a.val_total, np.nan)
a["has_alt"] = (a.n_nonchina_2024 >= 1).astype(float)
f = firms[["final_dunsno", "tercile"]].merge(a[["n_origins_2024", "n_nonchina_2024", "china_share_2024", "has_alt"]], left_on="final_dunsno", right_index=True, how="left")
e = pd.read_parquet(L / "smb_duns_matched_firm_attributes_employment_260428.parquet", columns=["final_dunsno", "YEAR", "EMPLOYEESALLSITES"])
e = e[e.YEAR == 2024].drop_duplicates("final_dunsno").set_index("final_dunsno").EMPLOYEESALLSITES
f["emp2024"] = f.final_dunsno.map(e)
f["emp_bin"] = pd.cut(f.emp2024.where(f.emp2024 > 0), [0, 19, 49, 249, 1e12], labels=["e1_19", "e20_49", "e50_249", "e250p"]).astype(object)
f["sic2"] = f.final_dunsno.map(sic).map(lambda v: np.nan if pd.isna(v) else int(str(int(v)).zfill(4)[:2]))
f.to_parquet(CACHE / "firm_attrs.parquet")
cnt = {"sample_firms": len(f), "with_2024_arrivals": int(f.has_alt.notna().sum()), "has_alt_1": int((f.has_alt == 1).sum()), "has_alt_0": int((f.has_alt == 0).sum()),
       "emp2024_matched": int(f.emp2024.notna().sum()), "emp2024_positive": int((f.emp2024 > 0).sum()), "sic2_available": int(f.sic2.notna().sum())}
for k, v in f.emp_bin.value_counts(dropna=False).items(): cnt[f"emp_bin_{k}"] = int(v)
for t, d in f.groupby("tercile"):
    cnt[f"{t}_share_has_alt_among_known"] = float(d.has_alt.mean()); cnt[f"{t}_median_n_origins_2024"] = float(d.n_origins_2024.median())
pd.Series(cnt).to_frame("value").to_csv(OUT / "07_extras_counts_261102.csv"); print(pd.Series(cnt).to_string())
