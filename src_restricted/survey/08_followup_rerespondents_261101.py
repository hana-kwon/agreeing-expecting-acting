# ============================================================
# START DATE   : 2026-11-01
# CODE RUN DATE: 2026-11-01
# TIME TAKEN   : ~3 min
# DESCRIPTION  : Pilot of the follow-up design with existing re-respondents: does the message an owner got
#                at an earlier response predict executed actions and expectations at a later response?
#                Exploratory and small; spec line added to 00_spec_261101.md before running.
# INPUT FILES  : same as 02_robustness_261101.py
# OUTPUT FILES : analysis/27_expectation_action_261101/08_followup_261101.csv (aggregates only)
# ============================================================
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd, statsmodels.api as sm
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from restricted_paths import SURVEY_DIR, CUSTOMS_DIR, JEV_DIR, OUTPUT_ROOT, out_dir, fresh_out_dir
warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve()
IA = SURVEY_DIR; OUT = out_dir("survey")
if (OUT / "08_followup_261101.csv").exists(): sys.exit("output exists; refusing to overwrite.")
Y = ["invest_post1_1", "prices_post1_1", "invest_post2_1", "prices_post2_1", "long_term_prior", "short_term_prior", "import_tariff_expect"]
d = pd.read_stata(IA / "inputs/analysis_all.dta", columns=["responseid", "uid", "startdate", "country", "three_info", "treat"] + Y, convert_categoricals=False)
d = d[d.country == 1].copy()
for c in Y + ["three_info", "treat"]:
    d[c] = pd.to_numeric(d[c], errors="coerce")
a = pd.read_stata(IA / "outputs/alignment_for_stata.dta", convert_categoricals=False)
d = d.merge(a[["responseid", "wave", "import_loc_post1_1__toward", "import_loc_post1_3__away", "import_loc_post2_1__toward", "import_loc_post2_3__away"]], on="responseid", validate="1:1")
d["date"] = pd.to_datetime(d.startdate, errors="coerce")
d = d.dropna(subset=["date"]).sort_values(["uid", "date"])
print("owners with 2+ responses:", int((d.groupby("uid").size() >= 2).sum()), file=sys.stderr)
first = d[d.three_info.isin([1, 2])].assign(msg_esc=lambda x: (x.three_info == 1).astype(float))
later = d[["uid", "date", "responseid", "wave"] + Y + ["import_loc_post1_1__toward", "import_loc_post1_3__away", "import_loc_post2_1__toward", "import_loc_post2_3__away"]]
# ★ CORE LOGIC: pair the earliest message response with the first later response at least 14 days after it;
# the same outcome at the earlier response is the baseline. A later response may carry its own message, recorded as such.
pairs = []
for u, g in first.groupby("uid"):
    t = g.iloc[0]
    l = later[(later.uid == u) & (later.date > t.date + pd.Timedelta(days=14))]
    if len(l):
        r = l.iloc[0]
        row = {"uid": u, "msg_esc": t.msg_esc, "gap": (r.date - t.date).days, "wave1": t.wave, "wave2": r.wave}
        for y in Y + ["import_loc_post1_1__toward", "import_loc_post1_3__away", "import_loc_post2_1__toward", "import_loc_post2_3__away"]:
            row[y + "_t2"] = r[y]; row[y + "_t1"] = t[y]
        pairs.append(row)
p = pd.DataFrame(pairs)
print("pairs:", len(p), "median gap:", p.gap.median(), "by arm:", p.msg_esc.value_counts().to_dict(), file=sys.stderr)
print(pd.crosstab(p.wave1.astype(str), p.wave2.astype(str)), file=sys.stderr)
rows = []
for y in Y + ["import_loc_post1_1__toward", "import_loc_post1_3__away", "import_loc_post2_1__toward", "import_loc_post2_3__away"]:
    s = p.dropna(subset=[y + "_t2"])
    if len(s) < 25: continue
    m0 = sm.OLS(s[y + "_t2"].astype(float), sm.add_constant(s[["msg_esc"]].astype(float))).fit(cov_type="HC1")
    b = s.dropna(subset=[y + "_t1"])
    if len(b) >= 25:
        m1 = sm.OLS(b[y + "_t2"].astype(float), sm.add_constant(b[["msg_esc", y + "_t1"]].astype(float))).fit(cov_type="HC1")
        rows.append(dict(y=y, coef=m0.params["msg_esc"], se=m0.bse["msg_esc"], n=len(s), coef_baseline=m1.params["msg_esc"], se_baseline=m1.bse["msg_esc"], n_baseline=len(b)))
    else:
        rows.append(dict(y=y, coef=m0.params["msg_esc"], se=m0.bse["msg_esc"], n=len(s)))
out = pd.DataFrame(rows); out.to_csv(OUT / "08_followup_261101.csv", index=False)
pd.set_option("display.width", 170, "display.float_format", lambda x: f"{x:.3f}")
print(out.to_string(index=False))
