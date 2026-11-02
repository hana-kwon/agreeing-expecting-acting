# ============================================================
# START DATE   : 2026-11-02
# CODE RUN DATE: 2026-11-02
# TIME TAKEN   : ~5 min
# DESCRIPTION  : Build the firm sample (1:1 DUNS-rcid links), 2024 China import share, SIC2, employment bin.
# INPUT FILES  : SMB_tariff/intermediate_data/prep_reg_L2_update/{revelio_smb/revelio_rcid_duns_bridge_full_261024,
#                smb_trade_duns_biweekly_260512, smb_firm_features_for_L2_260626,
#                smb_duns_matched_firm_attributes_employment_260428}.parquet (read-only)
# OUTPUT FILES : PRIVATE_CACHE (firm-level, cache only)
# ============================================================
import os, sys, pandas as pd
D = ("PRIVATE_DATA_ROOT/SMB_tariff/"
     "intermediate_data/prep_reg_L2_update/")
OUT = os.path.expanduser("PRIVATE_CACHE")
if os.path.exists(OUT): sys.exit("refuse to overwrite " + OUT)
b = pd.read_parquet(D + "revelio_smb/revelio_rcid_duns_bridge_full_261024.parquet")
n0 = len(b)
# ★ CORE LOGIC: strict 1:1 link, panel DUNS only, no export-only shippers (fixed in spec)
b = b[b.in_panel_260512 & ~b.export_shipper_only & (b.rcid_n_duns == 1) & (b.duns_n_rcid == 1)].copy()
b["strict_link"] = b.link_methods.str.fullmatch("strict_name_state")
print("bridge pairs", n0, "-> sample", len(b), "strict-only", b.strict_link.sum())
t = pd.read_parquet(D + "smb_trade_duns_biweekly_260512.parquet", columns=["final_dunsno", "origin", "year_biweek", "trade_value"])
t = t[t.year_biweek.str[:4] == "2024"]
print("china-like origins", [o for o in t.origin.dropna().unique() if "hina" in o or "Hong" in o]); t = t.dropna(subset=["origin"])
tot = t.groupby("final_dunsno").trade_value.sum().rename("imp2024")
chn = t[t.origin == "China"].groupby("final_dunsno").trade_value.sum().rename("chn2024")
x = pd.concat([tot, chn], axis=1).fillna({"chn2024": 0}); x["china_share_2024"] = x.chn2024 / x.imp2024
f = pd.read_parquet(D + "smb_firm_features_for_L2_260626.parquet", columns=["final_dunsno", "china_share_pre_2022_2024", "dnb_sic1"])
e = pd.read_parquet(D + "smb_duns_matched_firm_attributes_employment_260428.parquet", columns=["YEAR", "final_dunsno", "EMPLOYEESALLSITES"])
e = e[e.YEAR == 2024][["final_dunsno", "EMPLOYEESALLSITES"]].drop_duplicates("final_dunsno")
m = b.merge(x.reset_index(), on="final_dunsno", how="left").merge(f, on="final_dunsno", how="left").merge(e, on="final_dunsno", how="left")
m["sic2"] = (m.dnb_sic1 // 100)
m["emp_bin"] = pd.cut(m.EMPLOYEESALLSITES, [0, 19, 49, 249, 1e12], labels=["1-19", "20-49", "50-249", "250+"])
print(m[["china_share_2024", "china_share_pre_2022_2024", "sic2", "EMPLOYEESALLSITES"]].notna().sum())
print(m[["china_share_2024", "china_share_pre_2022_2024"]].corr()); print(m.emp_bin.value_counts(dropna=False))
m.to_parquet(OUT)
