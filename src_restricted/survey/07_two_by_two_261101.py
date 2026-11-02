# ============================================================
# START DATE   : 2026-11-01
# CODE RUN DATE: 2026-11-01
# TIME TAKEN   : ~5 min
# DESCRIPTION  : (A) message x business-impact-check 2x2, (B) message effect by pre-measured exposure,
#                (C) alignment components with expectations, (D) expectation -> expected cost -> plan chain.
#                Exploratory; spec in analysis/27_expectation_action_261101/00_spec_261101.md.
# INPUT FILES  : same as 02_robustness_261101.py
# OUTPUT FILES : analysis/27_expectation_action_261101/07_two_by_two_261101.csv, 07_cells_261101.csv
# ============================================================
import csv, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd, statsmodels.api as sm
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from restricted_paths import REPO, SURVEY_DIR, CUSTOMS_DIR, JEV_DIR, OUTPUT_ROOT, out_dir, fresh_out_dir
warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve()
IA = SURVEY_DIR; OUT = out_dir("survey")
if (OUT / "07_two_by_two_261101.csv").exists(): sys.exit("output exists; refusing to overwrite.")
keep = ["responseid", "uid", "startdate", "country", "three_info", "treat", "exposure_treat", "long_term_prior", "short_term_prior",
        "long_term_post", "short_term_post", "import_tariff_expect", "tariff_p_import_exp", "invest_post2_1", "prices_post2_1",
        "invest_post3_1", "prices_post3_1", "source_usa_1_2", "source_china_1_2", "sales_cost_guess1_1", "sales_cost_guess2_1",
        "sales_cost_post_new1_1", "sales_cost_post_new2_1", "sales_cost_post_251_1", "sales_cost_post_252_1", "cost_post2_1", "sales_post2_1"]
d = pd.read_stata(IA / "inputs/analysis_all.dta", columns=keep, convert_categoricals=False)
d = d[d.country == 1]
for c in keep[5:]:
    d[c] = pd.to_numeric(d[c], errors="coerce")  # some columns arrive as strings
a = pd.read_stata(IA / "outputs/alignment_for_stata.dta", convert_categoricals=False)
d = d.merge(a[["responseid", "wave", "align", "align_normative", "align_belief", "align_expectation", "importer_dum", "fcomp_dum", "export_dum",
               "industry_code", "emp_2024_bins", "sales_2024_val", "import_loc_post2_1__toward", "import_loc_post2_3__away"]], on="responseid", validate="1:1")
v2 = pd.read_stata(IA / "trump_alignment_v2/trump_alignment_v2.dta", convert_categoricals=False)
d = d.merge(v2[["responseid", "align_trump"]], on="responseid", how="left", validate="1:1")
d["date"] = pd.to_datetime(d.startdate, errors="coerce").dt.normalize()
steps = []
with open(REPO / "data/public/china_additional_duty_steps.csv", newline="") as f:
    r = csv.reader(f); next(r)
    for row in r: steps.append((pd.Timestamp(row[0]), float(row[1])))
steps = pd.DataFrame(steps, columns=["eff", "duty_cn"]).sort_values("eff")
d = pd.merge_asof(d.dropna(subset=["date"]).sort_values("date"), steps, left_on="date", right_on="eff", direction="backward")
for c in ["emp_2024_bins", "sales_2024_val"]:
    d["miss_" + c] = d[c].isna().astype(float); d[c] = d[c].fillna(d[c].median())
d["industry_code"] = d.industry_code.fillna(-1)
d["msg_esc"] = (d.three_info == 1).astype(float).where(d.three_info.isin([1, 2]))
d["expo"] = d.exposure_treat.astype(float)
d["msg_x_expo"] = d.msg_esc * d.expo
d["rev_exp_w4"] = d.tariff_p_import_exp - d.import_tariff_expect
d["rev_long"] = d.long_term_post - d.long_term_prior
d["rev_short"] = d.short_term_post - d.short_term_prior
d["rev_sales"] = d.sales_cost_post_new1_1 - d.sales_cost_guess1_1
d["rev_cost"] = d.sales_cost_post_new2_1 - d.sales_cost_guess2_1
for c in ["importer_dum", "fcomp_dum", "export_dum"]:
    d[c + "_b"] = (d[c] > 50).astype(float).where(d[c].notna())


def fit(df, y, main, cluster="uid", extra_fe=None):
    s = df.dropna(subset=[y] + main).copy()
    if len(s) < 60: return None, 0
    parts = [s[main + ["emp_2024_bins", "sales_2024_val", "miss_emp_2024_bins", "miss_sales_2024_val", "duty_cn"]].astype(float),
             pd.get_dummies(s.wave.astype(str), prefix="w", drop_first=True).astype(float),
             pd.get_dummies(s.industry_code.astype(int), prefix="i", drop_first=True).astype(float)]
    if extra_fe: parts.append(pd.get_dummies(s[extra_fe].fillna(0).astype(int), prefix="m", drop_first=True).astype(float))
    X = pd.concat(parts, axis=1); X = sm.add_constant(X.loc[:, X.std() > 0])
    return sm.OLS(s[y].astype(float), X).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(s[cluster])[0]}), len(s)


rows = []
def add(block, sample, y, x, m, n, note=""):
    if m is None: return
    rows.append(dict(block=block, sample=sample, y=y, x=x, coef=m.params[x], se=m.bse[x], n=n, note=note))

early = d[d.wave.astype(str).str.contains("W1|W2|W3")]; w4 = d[d.wave.astype(str).str.contains("W4")]
two = d[d.three_info.isin([1, 2])]
# balance of the two randomizations
pd.crosstab(two.wave.astype(str), [two.msg_esc, two.expo]).to_csv(OUT / "07_cells_261101.csv")
print(pd.crosstab(two.msg_esc, two.expo, normalize="index").round(3), file=sys.stderr)

# (A) 2x2
A_OUT = {"W4": ["rev_exp_w4", "rev_sales", "rev_cost", "sales_cost_post_251_1", "sales_cost_post_252_1", "invest_post2_1", "prices_post2_1",
                "invest_post3_1", "prices_post3_1", "source_usa_1_2", "source_china_1_2", "import_loc_post2_1__toward", "import_loc_post2_3__away"],
         "W1-W3": ["rev_long", "rev_short", "cost_post2_1", "sales_post2_1", "invest_post2_1", "prices_post2_1", "invest_post3_1", "prices_post3_1",
                   "import_loc_post2_1__toward", "import_loc_post2_3__away"]}
for sample, df in [("W4", w4[w4.three_info.isin([1, 2])]), ("W1-W3", early[early.three_info.isin([1, 2])])]:
    for y in A_OUT[sample]:
        m, n = fit(df, y, ["msg_esc", "expo", "msg_x_expo"])
        for x in ["msg_esc", "expo", "msg_x_expo"]:
            add("A_2x2", sample, y, x, m, n)
        # message effect within each exposure arm
        for lab, val in [("expo0", 0.0), ("expo1", 1.0)]:
            m2, n2 = fit(df[df.expo == val], y, ["msg_esc"])
            add(f"A_msg_within_{lab}", sample, y, "msg_esc", m2, n2)

# (B) message effect by pre-measured exposure
for sample, df in [("W4", w4[w4.three_info.isin([1, 2])]), ("W1-W3", early[early.three_info.isin([1, 2])])]:
    for het in ["importer_dum_b", "fcomp_dum_b", "export_dum_b"]:
        df2 = df.assign(inter=df.msg_esc * df[het])
        for y in ["invest_post2_1", "prices_post2_1", "import_loc_post2_1__toward", "import_loc_post2_3__away"] + (["rev_exp_w4"] if sample == "W4" else ["rev_short"]):
            m, n = fit(df2, y, ["msg_esc", het, "inter"])
            add("B_het", sample, y, f"msg_esc x {het}", m, n) if m is None else rows.append(dict(block="B_het", sample=sample, y=y, x=f"msg_esc x {het}", coef=m.params["inter"], se=m.bse["inter"], n=n, note=f"msg_main={m.params['msg_esc']:.3f}({m.bse['msg_esc']:.3f})"))

# (C) alignment components (old scores) with expectation
for sample, df, E in [("W4", w4, "import_tariff_expect"), ("W1-W3", early, "short_term_prior")]:
    for y in ["invest_post2_1", "prices_post2_1", "import_loc_post2_1__toward", "import_loc_post2_3__away"]:
        m, n = fit(df, y, [E, "align_normative", "align_belief", "align_expectation"], extra_fe="three_info")
        for x in [E, "align_normative", "align_belief", "align_expectation"]:
            add("C_components", sample, y, x, m, n)

# (D) chain in W4: expectation -> prior expected cost/sales change -> plan (descriptive)
for y_mid in ["sales_cost_guess2_1", "sales_cost_guess1_1"]:
    m, n = fit(w4, y_mid, ["import_tariff_expect", "align_trump"], extra_fe="three_info")
    add("D_chain_E_to_mid", "W4", y_mid, "import_tariff_expect", m, n)
for y in ["invest_post2_1", "prices_post2_1", "import_loc_post2_1__toward", "import_loc_post2_3__away"]:
    m, n = fit(w4, y, ["import_tariff_expect", "align_trump"], extra_fe="three_info")
    add("D_chain_E_only", "W4", y, "import_tariff_expect", m, n)
    m, n = fit(w4, y, ["import_tariff_expect", "sales_cost_guess2_1", "sales_cost_guess1_1", "align_trump"], extra_fe="three_info")
    for x in ["import_tariff_expect", "sales_cost_guess2_1", "sales_cost_guess1_1"]:
        add("D_chain_with_mid", "W4", y, x, m, n)
pd.DataFrame(rows).to_csv(OUT / "07_two_by_two_261101.csv", index=False)
print("rows", len(rows))
