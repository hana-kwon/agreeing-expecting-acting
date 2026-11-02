# ============================================================
# START DATE   : 2026-11-02
# CODE RUN DATE: 2026-11-02
# TIME TAKEN   : ~10 min
# DESCRIPTION  : Assign each posting to a function (ENG, SCI, SOFT, PRODDEV, PROC, SALES) by the fixed rules in
#                00_spec_261102.md; aggregate to rcid x month x function counts and DUNS x quarter panel.
# INPUT FILES  : PRIVATE_CACHE{2023,2024,2025}.parquet, firm_sample.parquet
# OUTPUT FILES : cache: rcid_month_function_counts.parquet, panel_duns_quarter.csv
#                analysis: 05_function_counts_by_year.csv (aggregate)
# ============================================================
import os, sys, re, pandas as pd, numpy as np
C = os.path.expanduser("PRIVATE_CACHE")
A = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../analysis/48_innovation_rd_hiring_261102/")) + "/"
for f in (C + "rcid_month_function_counts.parquet", C + "panel_duns_quarter.csv", A + "05_function_counts_by_year_261102.csv"):
    if os.path.exists(f): sys.exit("refuse to overwrite " + f)
PROD = re.compile(r"product development|product design|new product|r&d|research and development|prototype|industrial design")
def fn(onet, role):
    onet = onet if isinstance(onet, str) else ""; role = role if isinstance(role, str) else ""
    s = onet[:7]; p2 = s[:2]; p4 = s[:4]
    # ★ CORE LOGIC: fixed priority order from the spec; one function per posting
    if s[:4] in ("17-2", "17-3") or s == "11-9041": return "ENG"
    if p4 in ("19-1", "19-2", "19-4") or s in ("11-9121", "15-1221"): return "SCI"
    if p4 == "15-1" and s[:5] == "15-12" or s == "15-2051": return "SOFT"
    if s == "27-1021" or (role and PROD.search(role)): return "PRODDEV"
    if s in ("13-1021", "13-1022", "13-1023", "11-3061"): return "PROC"
    if p2 == "41" or s == "11-2022": return "SALES"
    return "OTHER"
parts = []
for y in (2023, 2024, 2025):
    d = pd.read_parquet(C + f"postings_bridge_{y}.parquet")
    n0 = len(d); d = d.drop_duplicates("job_id"); print(y, n0, "->", len(d), "after job_id dedup")
    # classify on unique (onet, role) pairs for speed
    u = d[["onet_code", "role_k1500_v2"]].drop_duplicates()
    u["func"] = [fn(a, b) for a, b in zip(u.onet_code, u.role_k1500_v2)]
    d = d.merge(u, on=["onet_code", "role_k1500_v2"], how="left")
    d["month"] = pd.to_datetime(d.post_date).dt.to_period("M").astype(str)
    d["src_year"] = y
    parts.append(d.groupby(["rcid", "month", "func", "src_year"]).size().rename("n").reset_index())
g = pd.concat(parts)
g = g[g.month.str[:4].astype(int) >= 2023]
g.to_parquet(C + "rcid_month_function_counts.parquet")
yr = g.assign(year=g.month.str[:4]).groupby(["year", "func"]).n.sum().unstack().fillna(0).astype(int)
yr["TOTAL"] = yr.sum(axis=1); yr.to_csv(A + "05_function_counts_by_year_261102.csv"); print(yr)
g["q"] = pd.PeriodIndex(g.month, freq="M").asfreq("Q").astype(str)
w = g.groupby(["rcid", "q", "func"]).n.sum().unstack(fill_value=0).reset_index()
for c in ("ENG", "SCI", "SOFT", "PRODDEV", "PROC", "SALES", "OTHER"):
    if c not in w: w[c] = 0
w["N"] = w[["ENG", "SCI", "SOFT", "PRODDEV", "PROC", "SALES", "OTHER"]].sum(axis=1)
w["RD"] = w.ENG + w.SCI + w.PRODDEV
w[["rcid", "q", "RD", "N"]].to_csv(C + "rcid_quarter_rd.csv", index=False)  # all bridge rcids, used for the SEC-firm branch (Part C)
s = pd.read_parquet(C + "firm_sample.parquet")
p = s.merge(w, on="rcid")
p.to_csv(C + "panel_duns_quarter.csv", index=False); print(p.shape, p.final_dunsno.nunique())
