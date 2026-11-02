# ============================================================
# START DATE   : 2026-11-02
# CODE RUN DATE: 2026-11-02
# TIME TAKEN   : ~10 min
# DESCRIPTION  : Match importer DUNS to trademark owners by standardized name + state, build firm x quarter panel
#                of new trademark applications (total, any, ITU 1(b), use 1(a), NICE groups, SOU/registration),
#                and write match diagnostics (counts only) + a 50-pair precision sample (local only).
# INPUT FILES  : PRIVATE_CACHE*.parquet, universe_names.parquet, universe_attrs.parquet
# OUTPUT FILES : PRIVATE_CACHE (firm-level, local only)
#                PRIVATE_CACHE (names, local only)
#                P/analysis/47_innovation_trademarks_261102/03_match_diagnostics.json (counts only)
# ============================================================
import os, re, glob, json, pandas as pd, numpy as np
TAG = os.environ.get("TAG", "")
C = os.path.expanduser("PRIVATE_CACHE")
P = "HOME"
for f in [C + "panel_firm_quarter" + TAG + ".parquet", P + "03_match_diagnostics" + TAG + ".json"]:
    assert not os.path.exists(f), "refuse to overwrite " + f

SUFFIX = {"INC", "INCORPORATED", "LLC", "LLP", "LP", "LTD", "LIMITED", "CORP", "CORPORATION", "CO", "COMPANY", "PLLC", "PC", "THE", "DBA", "USA", "US", "OF", "AMERICA"}
def norm(s):
    s = str(s).upper().replace("&", " AND ")
    s = re.sub(r"[^A-Z0-9 ]", " ", s)
    tok = s.split()
    # ★ CORE LOGIC: strip legal-form / generic tokens (INC, LLC, CORP, THE...) everywhere, keep distinctive tokens;
    # key then collapsed (no spaces) so 'SUN SHINE' == 'SUNSHINE'. USA/US/AMERICA dropped only if something else remains.
    core = [t for t in tok if t not in SUFFIX]
    if not core: core = [t for t in tok if t not in {"INC", "LLC", "CORP", "CO", "LTD", "THE"}]
    return "".join(core)

tm = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(C + "parsed/*.parquet"))], ignore_index=True)
n_raw = len(tm)
# later files (daily) override annual rows for the same serial: keep last
tm = tm.drop_duplicates("serial", keep="last")
tm = tm[(tm.filing >= "20220101")]
tm["fdate"] = pd.to_datetime(tm.filing, format="%Y%m%d", errors="coerce")
tm = tm[tm.fdate.notna()]
tm["q"] = tm.fdate.dt.year * 10 + (tm.fdate.dt.month - 1) // 3 + 1
tm_us = tm[(tm.country.fillna("US") == "US") & tm.state.notna() & tm.owner.notna()].copy()
tm_us["key"] = tm_us.owner.map(norm) + "|" + tm_us.state.str.upper()

nm = pd.read_parquet(C + "universe_names.parquet")
nm["key"] = nm.name.map(norm) + "|" + nm.state.astype(str).str.upper()
nm = nm[nm.key.str.split("|").str[0].str.len() >= 4].drop_duplicates(["final_dunsno", "key"])
kcount = nm.groupby("key").final_dunsno.nunique()
amb = kcount[kcount > 1].index
nm_u = nm[~nm.key.isin(amb)]
m = tm_us.merge(nm_u[["key", "final_dunsno"]].drop_duplicates(), on="key", how="inner").drop_duplicates(["serial", "final_dunsno"])

# classes -> groups
G = {"apparel_textile": {18, 22, 23, 24, 25, 26}, "electronics_machinery": {7, 8, 9, 11, 12}, "household_consumer": {14, 15, 16, 20, 21, 27, 28},
     "chem_health_food": {1, 2, 3, 4, 5, 29, 30, 31, 32, 33, 34}, "other_goods": {6, 10, 13, 17, 19}, "services": set(range(35, 46))}
def cls(s):
    return {int(x) for x in str(s).split(",") if x.strip().isdigit()}
cs = m.classes.map(cls)
for g, st in G.items(): m["g_" + g] = cs.map(lambda c: int(bool(c & st)))
m["g_any_goods"] = cs.map(lambda c: int(any(1 <= x <= 34 for x in c)))
m["itu"] = (m.itu_filed == "T").astype(int)
m["use"] = (m.use_filed == "T").astype(int)
m["sou"] = m.sou_date.notna().astype(int)
m["reg"] = (m.reg_no.fillna("0").str.strip().str.lstrip("0") != "").astype(int)
m["n"] = 1
vcols = ["n", "itu", "use", "sou", "reg", "g_any_goods"] + ["g_" + g for g in G]
agg = m.groupby(["final_dunsno", "q"])[vcols].sum().reset_index()

u = pd.read_parquet(C + "universe_attrs.parquet")
qs = sorted({y * 10 + k for y in range(2022, 2027) for k in range(1, 5)})
last_q = int(tm.fdate.max().year * 10 + (tm.fdate.max().month - 1) // 3 + 1)
json.dump({"last_filing_date_in_data": str(tm.fdate.max().date())}, open(C + "last_filing.json", "w"))
panel = pd.MultiIndex.from_product([u.final_dunsno, qs], names=["final_dunsno", "q"]).to_frame(index=False)
panel = panel.merge(agg, how="left", on=["final_dunsno", "q"]).fillna({c: 0 for c in vcols}).merge(u, on="final_dunsno")
panel.to_parquet(C + "panel_firm_quarter" + TAG + ".parquet")

ever = agg.groupby("final_dunsno").n.sum()
u["matched_any"] = u.final_dunsno.isin(ever.index)
by_size = u.groupby(u.emp_bin.astype(str)).matched_any.agg(["size", "sum", "mean"]).reset_index().values.tolist()
diag = dict(trademark_rows_parsed_2021plus=int(n_raw), unique_serials_2022plus=int(len(tm)), us_owner_with_state=int(len(tm_us)),
            universe_duns=int(len(u)), name_keys=int(nm.key.nunique()), ambiguous_keys_dropped=int(len(amb)),
            matched_application_rows=int(len(m)), duns_with_any_match=int(u.matched_any.sum()), match_rate_duns=float(u.matched_any.mean()),
            match_by_empbin_bin_n_matched_rate=by_size,
            quarters=qs, last_filing=str(tm.fdate.max().date()),
            apps_by_quarter={int(k): int(v) for k, v in m.groupby("q").n.sum().items()},
            us_apps_by_quarter_all={int(k): int(v) for k, v in tm_us.groupby("q").size().items()})
json.dump(diag, open(P + "03_match_diagnostics" + TAG + ".json", "w"), indent=1)

# precision sample (local only, names)
s = m.sample(50, random_state=47)[["owner", "state", "final_dunsno", "key"]].merge(nm_u[["final_dunsno", "key", "name"]].drop_duplicates("key"), on=["final_dunsno", "key"], how="left")
s.to_csv(C + "match_sample50_local" + TAG + ".csv", index=False)
print(json.dumps(diag, indent=1)[:3000])
