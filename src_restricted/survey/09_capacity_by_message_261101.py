# ============================================================
# START DATE   : 2026-11-01
# CODE RUN DATE: 2026-11-01
# TIME TAKEN   : ~2 min
# DESCRIPTION  : Does the randomized message change plans more where owners saw domestic substitution as
#                feasible before the message? Exploratory; not a test of deliberate waiting (46 P3).
# INPUT FILES  : same as 02_robustness_261101.py
# OUTPUT FILES : analysis/27_expectation_action_261101/09_capacity_by_message_261101.csv
# ============================================================
import sys, warnings
from pathlib import Path
import pandas as pd, statsmodels.api as sm
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from restricted_paths import SURVEY_DIR, CUSTOMS_DIR, JEV_DIR, OUTPUT_ROOT, out_dir, fresh_out_dir
warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve(); IA = SURVEY_DIR
OUT = out_dir("survey")
if (OUT / "09_capacity_by_message_261101.csv").exists(): sys.exit("output exists")
c = ["responseid", "uid", "country", "three_info", "exposure_treat", "reshore_capacity_us", "invest_post2_1", "prices_post2_1", "source_usa_1_2"]
d = pd.read_stata(IA / "inputs/analysis_all.dta", columns=c, convert_categoricals=False)
d = d[d.country == 1]
for v in c[3:]: d[v] = pd.to_numeric(d[v], errors="coerce")
a = pd.read_stata(IA / "outputs/alignment_for_stata.dta", convert_categoricals=False)
d = d.merge(a[["responseid", "wave", "industry_code", "emp_2024_bins", "import_loc_post2_1__toward", "import_loc_post2_3__away"]], on="responseid", validate="1:1")
# ★ IMPORTANT: only the exposure arm answered this item before the message; the rest answered elsewhere or not at all.
d = d[(d.exposure_treat == 1) & d.three_info.isin([1, 2]) & d.reshore_capacity_us.notna()].copy()
d["esc"] = (d.three_info == 1).astype(float)
d["cap"] = (d.reshore_capacity_us - d.reshore_capacity_us.mean()) / d.reshore_capacity_us.std()
d["esc_cap"] = d.esc * d.cap
rows = []
for y in ["invest_post2_1", "prices_post2_1", "import_loc_post2_1__toward", "import_loc_post2_3__away", "source_usa_1_2"]:
    s = d.dropna(subset=[y])
    if len(s) < 40: continue
    X = pd.concat([s[["esc", "cap", "esc_cap"]], pd.get_dummies(s.wave.astype(str), prefix="w", drop_first=True).astype(float)], axis=1)
    m = sm.OLS(s[y].astype(float), sm.add_constant(X.loc[:, X.std() > 0])).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(s.uid)[0]})
    rows.append(dict(y=y, esc=m.params["esc"], esc_se=m.bse["esc"], cap=m.params["cap"], cap_se=m.bse["cap"], esc_x_cap=m.params["esc_cap"], esc_x_cap_se=m.bse["esc_cap"], n=len(s)))
o = pd.DataFrame(rows); o.to_csv(OUT / "09_capacity_by_message_261101.csv", index=False)
pd.set_option("display.width", 170, "display.float_format", lambda x: f"{x:.3f}"); print(o.to_string(index=False))
