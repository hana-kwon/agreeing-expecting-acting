# ============================================================
# START DATE   : 2026-11-02
# CODE RUN DATE: 2026-11-02
# TIME TAKEN   : ~5 min
# DESCRIPTION  : Are absent country x HS2 x month rows zero trade? Rebuilds the analysis 54 D panel on a full grid and compares summed HS2 values with the Census all-commodity total per country-month.
# INPUT FILES  : code/54_review_fixes_261102/05_d_prep.py logic; Census API imports/hs (public)
# OUTPUT FILES : analysis/56_final_checks_261102/03_census_missing_261102.csv, data/56_final_checks_261102/census_country_totals_261102.csv; private cache d_panel.csv
# ============================================================
import pathlib, json, re, sys, urllib.request, urllib.parse
KEY_FILE = pathlib.Path.home() / ".config/census/api_key.txt"
import pandas as pd, numpy as np
HERE = pathlib.Path(__file__).resolve(); T = HERE.parents[2]; SRC = HERE.parents[1] / "54_review_fixes_261102/05_d_prep.py"
# ★ IMPORTANT: the panel with absent rows marked was built by running 05_d_prep.py with one added grid step (see the git history note in the receipt); cache only
sys.path.insert(0, str(HERE.parents[1] / "36_census_imports_261102")); import common_261102 as cm
_d = cm.load_hs2(); _xw, _un = cm.crosswalk(_d[["CTY_CODE", "CTY_NAME"]]); key = {c: iso for iso, cs in _xw.items() for c in cs}
pan = pd.read_csv(pathlib.Path.home() / "private_cache/final56_261102/d_panel.csv", keep_default_na=False, na_values=[""], dtype={"hs2": str})
w = pan[(pan.mi >= 12) & pan.tau_prod.notna() & pan.tau_head.notna()]
d = _d
d = d[d.CTY_CODE.isin(key)]; d = d[(d.month >= "2024-01") & (d.month <= "2025-12")]
hs = d.groupby(["CTY_CODE", "month"]).gen.sum().rename("hs2_sum").reset_index()
fd = T / "data/56_final_checks_261102/census_country_totals_261102.csv"
if not fd.exists():
    KEY = KEY_FILE.read_text().strip(); q = urllib.parse.urlencode({"get": "CTY_CODE,GEN_VAL_MO", "I_COMMODITY": "-", "time": "from 2024-01 to 2025-12", "key": KEY})
    r = json.load(urllib.request.urlopen("https://api.census.gov/data/timeseries/intltrade/imports/hs?" + q, timeout=120))
    pd.DataFrame(r[1:], columns=r[0]).to_csv(fd, index=False)
t = pd.read_csv(fd, dtype=str); t["tot"] = pd.to_numeric(t.GEN_VAL_MO); t = t.rename(columns={"time": "month"})[["CTY_CODE", "month", "tot"]]
m = hs.merge(t, on=["CTY_CODE", "month"], how="inner"); m["ratio"] = m.hs2_sum / m.tot.replace(0, np.nan)
rec = w.recorded == 1; cellsize = w.groupby(["iso", "hs2"]).imp.transform("mean")
out = dict(cells=w.groupby(["iso", "hs2"]).ngroups, cell_months=len(w), absent_cell_months=int((~rec).sum()), recorded_zero=int((rec & (w.imp == 0)).sum()),
           share_cells_with_any_absent=float(w.assign(a=~rec).groupby(["iso", "hs2"]).a.max().mean()),
           share_2025_value_in_cells_with_any_absent=float(w[(w.mi >= 24) & w.groupby(["iso", "hs2"]).recorded.transform("min").eq(0)].imp.sum() / w[w.mi >= 24].imp.sum()),
           median_mean_monthly_value_cells_with_absent=float(cellsize[~rec].median()), median_mean_monthly_value_recorded=float(cellsize[rec].median()),
           country_months_compared=len(m), share_hs2_sum_equals_total_within_0_1pct=float(((m.ratio - 1).abs() < 0.001).mean()), min_ratio=float(m.ratio.min()), max_ratio=float(m.ratio.max()), value_ratio_overall=float(m.hs2_sum.sum() / m.tot.sum()))
pd.DataFrame([out]).T.to_csv(T / "analysis/56_final_checks_261102/03_census_missing_261102.csv", header=["value"]); print(pd.DataFrame([out]).T.to_string())
