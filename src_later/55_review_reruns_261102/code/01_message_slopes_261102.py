# ============================================================
# START DATE   : 2026-11-02
# CODE RUN DATE: 2026-11-02
# TIME TAKEN   : ~5 min
# DESCRIPTION  : Reruns agreed after the 06:20 review. F: message effects on expectation and on plans in the same outcome samples,
#                centered message x check-block term, response check on all 879 assigned, shutdown split of investment.
#                A: W4 between-owner slopes with perceived current tariff; A3a prior traits, A3b adds expected sales/cost; A4 same-sample pair.
#                Respondent rows stay in memory; only aggregates are written.
# INPUT FILES  : 40_smb_pricing_controls_261102/common_261102.py; analysis_all.dta (read only)
# OUTPUT FILES : analysis/55_review_reruns_261102/01_f_*.csv, 01_a_slopes_261102.csv
# ============================================================
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "40_smb_pricing_controls_261102"))
import numpy as np, pandas as pd
import common_261102 as cm
OUT = cm.TOPIC / "analysis/55_review_reruns_261102"
for f in ["01_f_response_261102.csv", "01_f_paired_261102.csv", "01_a_slopes_261102.csv"]:
    if (OUT / f).exists(): sys.exit("refuse to overwrite " + f)
d = cm.load_survey()
ex = pd.read_stata(cm.IA / "inputs/analysis_all.dta", columns=["responseid", "import_tariff", "tariff_p_import_exp", "exposure_treat", "sales_cost_done1_1", "sales_cost_done2_1",
                   "sales_cost_guess1_1", "sales_cost_guess2_1"], convert_categoricals=False)
for c in ex.columns[1:]: ex[c] = pd.to_numeric(ex[c], errors="coerce")
d = d.merge(ex, on="responseid", how="left", validate="1:1")
w4 = d[d.wave.astype(str).str.contains("W4")].copy()
w4["now"] = w4.import_tariff; w4["E"] = w4.import_tariff_expect
# ★ CORE LOGIC: investment is the team's score, where the shutdown choice is coded -100; it is split, never silently dropped
for k in (2, 3):
    y = f"invest_post{k}_1"
    w4[f"shut{k}"] = (w4[y] == -100).astype(float).where(w4[y].notna())
    w4[f"adj{k}"] = w4[y].where(w4[y] > -100)
def X_of(s, cols):
    X = pd.concat([s[list(cols) + cm.BASE_CTRL].astype(float), cm.dummies(s.wave.astype(str), "w"), cm.dummies(s.industry_code.astype(int), "i")], axis=1)
    return X.loc[:, ~X.columns.duplicated()]
# ---------------- F ----------------
a = w4[w4.E.notna() & w4.three_info.isin([1, 2])].copy()      # all 879 assigned with the pre-message expectation
a["M"] = (a.three_info == 1).astype(float); a["B"] = a.exposure_treat.astype(float); a["Bc"] = a.B - a.B.mean(); a["MBc"] = a.M * a.Bc
a["post"] = a.tariff_p_import_exp
RHS = ["M", "B", "MBc", "E"]
Y = ["prices_post2_1", "invest_post2_1", "shut2", "adj2", "prices_post3_1", "invest_post3_1", "shut3", "adj3"]
rows = []
for y in ["post"] + Y:
    t = a.copy(); t["resp"] = t[y].notna().astype(float)
    m = cm.ols(t, "resp", X_of(t, RHS))
    rows.append(dict(y=y, n=len(t), share_M1=t[t.M == 1].resp.mean(), share_M0=t[t.M == 0].resp.mean(), d=m.params["M"], se=m.bse["M"], p=m.pvalues["M"]))
pd.DataFrame(rows).to_csv(OUT / "01_f_response_261102.csv", index=False); print(pd.DataFrame(rows).round(3).to_string(index=False))
rows = []
def est(s, y, sample, y_label=None, rhs=RHS):
    m = cm.ols(s, y, X_of(s, rhs))
    return dict(sample=sample, y=y_label or y, n=len(s), n_M1=int(s.M.sum()), M=m.params["M"], se_M=m.bse["M"], lo=m.params["M"] - 1.96 * m.bse["M"], hi=m.params["M"] + 1.96 * m.bse["M"],
                MxB=m.params["MBc"], se_MxB=m.bse["MBc"], B=m.params["B"], se_B=m.bse["B"], F_M=(m.params["M"] / m.bse["M"]) ** 2)
com = a[a.post.notna()]
rows.append(est(com, "post", "all_with_post"))
for y in Y:
    s = com[com[y].notna()]
    rows.append(est(s, "post", y)); rows.append(est(s, y, y))
    rows.append(est(a[a[y].notna()], y, y + "|prior_only", y))     # same outcome without requiring a post-message expectation
f = pd.DataFrame(rows); f.to_csv(OUT / "01_f_paired_261102.csv", index=False)
pd.set_option("display.width", 220); print(f[["sample", "y", "n", "M", "se_M", "lo", "hi", "MxB", "se_MxB", "F_M"]].round(3).to_string(index=False))
# ---------------- A ----------------
base = w4.dropna(subset=["E", "now"]).copy()
PAST = ["sales_cost_done1_1", "sales_cost_done2_1"]; GUESS = ["sales_cost_guess1_1", "sales_cost_guess2_1"]
for c in PAST + GUESS:
    base["miss_" + c] = base[c].isna().astype(float); base[c] = base[c].fillna(0)
A3a = ["fcomp_dum", "export_dum", "miss_fcomp_dum", "miss_export_dum"] + PAST + ["miss_" + c for c in PAST]
A3b = A3a + GUESS + ["miss_" + c for c in GUESS]
done_of = {"prices_post2_1": "prices_post1_1", "prices_post3_1": "prices_post1_1", "invest_post2_1": "invest_post1_1", "invest_post3_1": "invest_post1_1"}
def fit(df, y, extra, spec):
    s = df.dropna(subset=[y]).copy()
    X = pd.concat([s[["E", "now"] + list(extra) + cm.BASE_CTRL].astype(float), cm.dummies(s.wave.astype(str), "w"), cm.dummies(s.industry_code.astype(int), "i"), cm.dummies(s.three_info.fillna(0).astype(int), "m")], axis=1)
    X = X.loc[:, ~X.columns.duplicated()]
    m = cm.ols(s, y, X)
    t = m.t_test("E + now = 0")     # ★ IMPORTANT: g+h=0 is the test that only the gap (expected minus current) matters
    return dict(y=y, spec=spec, n=len(s), g_E=m.params["E"], se_g=m.bse["E"], h_now=m.params["now"], se_h=m.bse["now"], g_plus_h=float(np.squeeze(t.effect)), se_sum=float(np.squeeze(t.sd)), p_sum=float(np.squeeze(t.pvalue)))
rows = []
for y in Y:
    rows.append(fit(base, y, [], "A1")); rows.append(fit(base, y, A3a, "A3a_prior_traits")); rows.append(fit(base, y, A3b, "A3b_adds_expected_sales_cost_channel"))
    if y in done_of:
        b2 = base.copy(); b2["done"] = b2[done_of[y]].where(b2[done_of[y]] > -100); b2 = b2[b2.done.notna()]
        rows.append(fit(b2, y, [], "A4_A1_same_sample")); rows.append(fit(b2, y, ["done"], "A4_A1_plus_done"))
r = pd.DataFrame(rows); r.to_csv(OUT / "01_a_slopes_261102.csv", index=False); print(r.round(3).to_string(index=False))
print("counts", dict(w4=len(w4), assigned=len(a), base=len(base), done_obs=int(w4.sales_cost_done1_1.notna().sum()), guess_obs=int(w4.sales_cost_guess1_1.notna().sum())))
