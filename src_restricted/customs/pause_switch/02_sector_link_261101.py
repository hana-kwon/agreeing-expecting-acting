# ============================================================
# START DATE   : 2026-11-01
# CODE RUN DATE: 2026-11-01
# TIME TAKEN   : ~2 min
# DESCRIPTION  : Sector-level link between importers' survey expectations and the customs pause rate (T6). Split
#                from 01 because the survey industry field holds labels, so the numeric coercion there emptied it.
# INPUT FILES  : analysis/33_firm_pause_switch_261101/01_sector_link_261101.csv; analysis_all.dta; alignment_for_stata.dta
# OUTPUT FILES : analysis/33_firm_pause_switch_261101/02_sector_link_261101.csv, 02_sector_corr_261101.csv
# ============================================================
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from restricted_paths import SURVEY_DIR, CUSTOMS_DIR, JEV_DIR, OUTPUT_ROOT, out_dir, fresh_out_dir
warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve(); IA = SURVEY_DIR
OUT = out_dir("customs_pause_switch")
if (OUT / "02_sector_corr_261101.csv").exists(): sys.exit("output exists")
ss = pd.read_csv(OUT / "01_sector_link_261101.csv", index_col="sec")[["n_2025", "d_paused", "d_none", "d_other"]]
LAB = {"agriculture": 1, "mining": 2, "utilities": 3, "construction": 4, "manufacturing": 5, "wholesale": 6, "transportation": 7, "information": 8, "finance": 9, "real estate": 10, "professional": 11, "management": 12, "administrative": 13, "educational": 14, "health": 15, "arts": 16, "accommodation": 17, "other services": 18, "retail": 20, "marketing": 21}
sv = pd.read_stata(IA / "inputs/analysis_all.dta", columns=["responseid", "country", "industry", "long_term_prior", "short_term_prior", "import_tariff_expect"], convert_categoricals=False)
sv = sv[sv.country == 1]
print(sv.industry.astype(str).str[:30].value_counts().head(4).to_dict(), file=sys.stderr)
def code(x):
    t = str(x).lower()
    for k, v in LAB.items():
        if k in t: return v
    try: return int(float(t))
    except Exception: return np.nan
sv["sec"] = sv.industry.map(code)
for c in ["long_term_prior", "short_term_prior", "import_tariff_expect"]: sv[c] = pd.to_numeric(sv[c], errors="coerce")
a = pd.read_stata(IA / "outputs/alignment_for_stata.dta", convert_categoricals=False)[["responseid", "wave", "importer_dum"]]
sv = sv.merge(a, on="responseid"); sv = sv[sv.importer_dum > 0]
w2 = sv[sv.wave.astype(str).str.contains("W2")].groupby("sec").agg(long_e=("long_term_prior", "mean"), short_e=("short_term_prior", "mean"), n_w2=("short_term_prior", "count"))
w4 = sv[sv.wave.astype(str).str.contains("W4")].groupby("sec").agg(tar_e=("import_tariff_expect", "mean"), n_w4=("import_tariff_expect", "count"))
l = ss.join(w2).join(w4); l.to_csv(OUT / "02_sector_link_261101.csv")
pd.set_option("display.width", 170, "display.float_format", lambda x: f"{x:.3f}"); print(l.to_string())
keep = l[l.n_2025 >= 200]
r = []
for e, nv in [("long_e", "n_w2"), ("short_e", "n_w2"), ("tar_e", "n_w4")]:
    k = keep[keep[nv].fillna(0) >= 15]
    for y in ["d_paused", "d_none", "d_other"]:
        r.append(dict(expectation=e, outcome=y, sectors=len(k), pearson=k[e].corr(k[y]), spearman=k[e].corr(k[y], method="spearman")))
r = pd.DataFrame(r); r.to_csv(OUT / "02_sector_corr_261101.csv", index=False); print(r.to_string(index=False))
