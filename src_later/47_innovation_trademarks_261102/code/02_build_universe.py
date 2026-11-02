# ============================================================
# START DATE   : 2026-11-02
# CODE RUN DATE: 2026-11-02
# TIME TAKEN   : ~5 min
# DESCRIPTION  : Build importer universe (DUNS) with 2024 import shares by region (China, Asia ex-China, Europe,
#                North America, Latin America/Caribbean, ROW), employment bin, SIC2, state, and candidate names
#                (D&B name all years, concordance name, Panjiva sample name) for trademark matching.
# INPUT FILES  : #Shared_ SMB_tariff prep_reg_L2_update: smb_trade_duns_biweekly_260512.parquet,
#                smb_duns_matched_firm_attributes_employment_260428.parquet, smb_panjiva_duns_concordance_final_260512.parquet.gzip
# OUTPUT FILES : PRIVATE_CACHE (firm-level, names: local only)
#                PRIVATE_CACHE
# ============================================================
import os, pandas as pd, numpy as np
S = "PRIVATE_DATA_ROOT/SMB_tariff/intermediate_data/prep_reg_L2_update/"
C = os.path.expanduser("PRIVATE_CACHE")
for f in ["universe_names.parquet", "universe_attrs.parquet"]:
    assert not os.path.exists(C + f), "refuse to overwrite " + f

t = pd.read_parquet(S + "smb_trade_duns_biweekly_260512.parquet",
                    columns=["final_dunsno", "origin", "year_biweek", "trade_value", "sample_firm_name", "final_companyname", "final_state", "dnb_sic1"])
t = t[t.origin.notna()]
names = pd.concat([t[["final_dunsno", "sample_firm_name"]].rename(columns={"sample_firm_name": "name"}),
                   t[["final_dunsno", "final_companyname"]].rename(columns={"final_companyname": "name"})]).dropna().drop_duplicates()
st = t.groupby("final_dunsno").agg(state=("final_state", "first"), sic1=("dnb_sic1", "first")).reset_index()

asia = set("""Afghanistan Bangladesh Bhutan Brunei Darussalam Cambodia Hong Kong India Indonesia Japan Kazakhstan Kyrgyzstan Laos Macau Malaysia Maldives Mongolia Myanmar Nepal Pakistan Philippines Singapore South Korea Sri Lanka Taiwan Tajikistan Thailand Turkmenistan Uzbekistan Vietnam East Timor""".split())
asia |= {"Brunei Darussalam", "Hong Kong", "South Korea", "Sri Lanka", "East Timor"}
europe = set("""Albania Andorra Austria Belarus Belgium Bosnia and Herzegovina Bulgaria Croatia Cyprus Czech Republic Denmark Estonia Finland France Germany Greece Hungary Iceland Ireland Italy Latvia Liechtenstein Lithuania Luxembourg Malta Moldova, Republic of Monaco Montenegro Netherlands Northern Ireland Norway Poland Portugal Romania Russia San Marino Scotland Serbia Slovakia Slovenia Spain Sweden Switzerland Ukraine United Kingdom Wales Macedonia Turkey Gibraltar Guernsey Jersey Isle of Man Faroe Islands""".split()) | {"Bosnia and Herzegovina", "Czech Republic", "Moldova, Republic of", "Northern Ireland", "United Kingdom", "Faroe Islands", "Isle of Man", "San Marino"}
latam = set("""Argentina Aruba Bahamas Barbados Belize Bolivia Brazil Chile Colombia Costa Rica Cuba Curacao Dominica Dominican Republic Ecuador El Salvador Guatemala Guyana Haiti Honduras Jamaica Nicaragua Panama Paraguay Peru Suriname Uruguay Venezuela Trinidad and Tobago Cayman Islands Bermuda Anguilla Grenada Montserrat""".split()) | {"Costa Rica", "Dominican Republic", "El Salvador", "Trinidad and Tobago", "Cayman Islands", "Antigua And Barbuda", "Saint Lucia", "Saint Kitts and Nevis", "Turks and Caicos Islands", "Virgin Islands (British)", "Saint Vincent and the Grenadines"}
na = {"Canada", "Mexico"}
def region(o):
    if o == "China": return "china"
    if o in asia: return "asia_ex_china"
    if o in europe: return "europe"
    if o in na: return "north_america"
    if o in latam: return "latam_carib"
    return "row"   # includes Middle East, Africa, Oceania, United States (re-imports)

t["yr"] = t.year_biweek.str[:4]
t24 = t[t.yr == "2024"].copy()
t24["region"] = t24.origin.map(region)
sh = t24.pivot_table(index="final_dunsno", columns="region", values="trade_value", aggfunc="sum", fill_value=0)
for c in ["china", "asia_ex_china", "europe", "north_america", "latam_carib", "row"]:
    if c not in sh: sh[c] = 0
tot = sh.sum(axis=1)
imp24 = tot.rename("imports_2024").reset_index()
sh = sh.div(tot, axis=0).add_suffix("_share24").reset_index()
sh = sh.merge(imp24)
sh = sh[sh.imports_2024 > 0]
# also all-years total (to count firms w/o 2024 imports)
n_all = t.final_dunsno.nunique()

emp = pd.read_parquet(S + "smb_duns_matched_firm_attributes_employment_260428.parquet",
                      columns=["YEAR", "final_dunsno", "COMPANYNAME", "STATE", "EMPLOYEESALLSITES", "EMPLOYEESTHISSITE", "SIC1"])
# ★ IMPORTANT: employment taken from the D&B 2024 snapshot (fallback: latest year <=2025) so size is pre-shock and not post-treatment.
e = emp.sort_values("YEAR")
e24 = e[e.YEAR == 2024].drop_duplicates("final_dunsno", keep="last")
elast = e[e.YEAR <= 2025].drop_duplicates("final_dunsno", keep="last")
e1 = pd.concat([e24, elast[~elast.final_dunsno.isin(e24.final_dunsno)]])[["final_dunsno", "EMPLOYEESALLSITES", "SIC1", "STATE"]]
names2 = emp[["final_dunsno", "COMPANYNAME", "STATE"]].rename(columns={"COMPANYNAME": "name", "STATE": "state_dnb"}).dropna(subset=["name"]).drop_duplicates()
con = pd.read_parquet(S + "smb_panjiva_duns_concordance_final_260512.parquet.gzip", columns=["final_dunsno", "final_companyname", "final_state"])
names3 = con.rename(columns={"final_companyname": "name", "final_state": "state_dnb"}).dropna(subset=["name"]).drop_duplicates()

u = sh.merge(e1, on="final_dunsno", how="left").merge(st, on="final_dunsno", how="left")
u["state"] = u.state.fillna(u.STATE)
u["sic1c"] = u.SIC1.fillna(u.sic1)
u["sic2"] = (u.sic1c // 100).astype("Int64")
emp_ = u.EMPLOYEESALLSITES
u["emp_bin"] = pd.cut(emp_, [0, 19, 49, 249, 1e12], labels=["1-19", "20-49", "50-249", "250+"])
u = u[["final_dunsno", "state", "sic2", "emp_bin", "EMPLOYEESALLSITES", "imports_2024"] + [c for c in u.columns if c.endswith("_share24")]]
print("universe DUNS with 2024 imports>0:", len(u), "of", n_all, "in trade panel")
print(u.emp_bin.value_counts(dropna=False).to_string()); print(u.state.isna().sum(), "missing state;", u.sic2.isna().sum(), "missing sic2")
nm = pd.concat([names.assign(state_src="panjiva_row"), names2.assign(state_src="dnb"), names3.assign(state_src="concordance")])
nm = nm[nm.final_dunsno.isin(u.final_dunsno)]
nm = nm.merge(u[["final_dunsno", "state"]], on="final_dunsno")
nm.to_parquet(C + "universe_names.parquet"); u.to_parquet(C + "universe_attrs.parquet")
print("names rows", len(nm))
