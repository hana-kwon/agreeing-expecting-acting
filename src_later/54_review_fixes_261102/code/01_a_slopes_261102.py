# ============================================================
# START DATE   : 2026-11-02
# CODE RUN DATE: 2026-11-02
# TIME TAKEN   : ~15 min
# DESCRIPTION  : A. W4 between-owner slopes of plans on expected year-end tariff with perceived-current-tariff controls (A0-A5), and
#                F. randomized message first stage / reduced form on a common sample (2x2: message x business-impact check).
#                Respondent rows stay in memory; only aggregate coefficients are written.
# INPUT FILES  : see 40_smb_pricing_controls_261102/common_261102.py; analysis_all.dta (read only)
# OUTPUT FILES : analysis/54_review_fixes_261102/01_a_slopes_261102.csv, 01_f_message_261102.csv
# ============================================================
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "40_smb_pricing_controls_261102"))
import numpy as np, pandas as pd, statsmodels.api as sm
import common_261102 as cm
OUT = cm.TOPIC / "analysis/54_review_fixes_261102"
for f in ["01_a_slopes_261102.csv", "01_f_message_261102.csv"]:
    if (OUT / f).exists(): sys.exit("refuse to overwrite " + f)
d = cm.load_survey()
ex = pd.read_stata(cm.IA / "inputs/analysis_all.dta", columns=["responseid", "import_tariff", "tariff_p_import_exp", "exposure_treat", "sales_cost_done1_1", "sales_cost_done2_1",
                   "sales_cost_guess1_1", "sales_cost_guess2_1", "passthrough_exp_1"], convert_categoricals=False)
for c in ex.columns[1:]: ex[c] = pd.to_numeric(ex[c], errors="coerce")
d = d.merge(ex, on="responseid", how="left", validate="1:1")
w4 = d[d.wave.astype(str).str.contains("W4")].copy()
w4["now"] = w4.import_tariff; w4["E"] = w4.import_tariff_expect; w4["delta"] = w4.E - w4.now
PRE = ["sales_cost_done1_1", "sales_cost_done2_1", "sales_cost_guess1_1", "sales_cost_guess2_1"]
for c in PRE:
    w4["miss_" + c] = w4[c].isna().astype(float); w4[c] = w4[c].fillna(0)
EXPO = ["fcomp_dum", "export_dum", "miss_fcomp_dum", "miss_export_dum"] + PRE + ["miss_" + c for c in PRE]
done_of = {"prices_post2_1": "prices_post1_1", "prices_post3_1": "prices_post1_1", "invest_post2_1": "invest_post1_1", "invest_post3_1": "invest_post1_1"}
def fit(df, y, regs, extra=(), spec=""):
    cols = list(regs) + list(extra)
    s = df.dropna(subset=[y] + list(regs) + [c for c in extra if c in ("done",)]).copy()
    X = pd.concat([s[cols + cm.BASE_CTRL].astype(float), cm.dummies(s.wave.astype(str), "w"), cm.dummies(s.industry_code.astype(int), "i"),
                   cm.dummies(s.three_info.fillna(0).astype(int), "m")], axis=1)
    X = X.loc[:, ~X.columns.duplicated()]
    m = cm.ols(s, y, X)
    r = dict(y=y, spec=spec, n=len(s), regressor=regs[0])
    r["coef"] = m.params[regs[0]]; r["se"] = m.bse[regs[0]]; r["lo"] = r["coef"] - 1.96 * r["se"]; r["hi"] = r["coef"] + 1.96 * r["se"]
    if len(regs) > 1: r["coef2"] = m.params[regs[1]]; r["se2"] = m.bse[regs[1]]
    return r
Y = [f"{a}_post{k}_1" for a in ("prices", "invest") for k in (1, 2, 3)] + ["import_loc_post2_1__toward", "import_loc_post2_3__away"]
rows = []
base = w4.dropna(subset=["E", "now"])        # common sample: expectation and perceived-current both observed
rows_n = dict(w4=len(w4), expect_obs=int(w4.E.notna().sum()), now_obs=int(w4.now.notna().sum()), both=len(base))
for y in Y:
    rows.append(fit(w4, y, ["E"], spec="A0_expect_only_all_expect_obs"))
    rows.append(fit(base, y, ["E"], spec="A0c_expect_only_common"))
    rows.append(fit(base, y, ["E", "now"], spec="A1_expect+now"))
    rows.append(fit(base, y, ["delta"], spec="A2_expect_minus_now"))
    rows.append(fit(base, y, ["delta", "now"], spec="A2b_delta+now"))
    rows.append(fit(base, y, ["E", "now"], extra=EXPO, spec="A3_A1+pre_message_exposure"))
    if y in done_of:
        b2 = base.copy(); b2["done"] = b2[done_of[y]].where(b2[done_of[y]] > -100)
        rows.append(fit(b2, y, ["E"], extra=["done"], spec="A4_expect+done"))
        rows.append(fit(b2, y, ["E", "now"], extra=EXPO + ["done"], spec="A5_A1+A3+done"))
a = pd.DataFrame(rows); a["note"] = ""
a.loc[:, "sample_counts"] = str(rows_n)
a.to_csv(OUT / "01_a_slopes_261102.csv", index=False)
pd.set_option("display.width", 200, "display.float_format", lambda x: f"{x:.3f}")
print(rows_n); print(a[a.y.isin(["prices_post2_1", "invest_post2_1", "prices_post1_1", "invest_post1_1", "prices_post3_1", "invest_post3_1"])][["y", "spec", "coef", "se", "n"]].to_string(index=False))
print("pass-through (W4, post-message) nonmissing:", int(w4.passthrough_exp_1.notna().sum()))
# ---------------- F ----------------
w = w4[w4.three_info.isin([1, 2])].copy()
w["msg"] = (w.three_info == 1).astype(float); w["expo"] = w.exposure_treat.astype(float)
w["post"] = w.tariff_p_import_exp; w["prior"] = w.E
rows = []
com = w.dropna(subset=["post", "prior"] )
def ols2(s, y, rhs, label, extra=()):
    X = pd.concat([s[rhs + ["expo"] + list(extra) + cm.BASE_CTRL].astype(float), cm.dummies(s.wave.astype(str), "w"), cm.dummies(s.industry_code.astype(int), "i")], axis=1)
    X = X.loc[:, ~X.columns.duplicated()]
    m = cm.ols(s, y, X)
    r = dict(sample=label, y=y, n=len(s), term=rhs[0], coef=m.params[rhs[0]], se=m.bse[rhs[0]]); r["F_firststage"] = (r["coef"] / r["se"]) ** 2 if y == "post" else np.nan
    if "msg" in rhs and "expo" in X: pass
    return r
for y in ["post"]:
    rows.append(ols2(com, y, ["msg", "prior"], "common(post&prior obs)"))
# 2x2 interaction
com["msg_x_expo"] = com.msg * com.expo
rows.append(ols2(com, "post", ["msg", "prior"], "common 2x2 with interaction", extra=["msg_x_expo"]))
rows[-1]["term"] = "msg (expo=0 main)"
r2 = ols2(com, "post", ["msg_x_expo", "prior"], "common 2x2 with interaction", extra=["msg"]); r2["term"] = "msg x expo"; rows.append(r2)
for y in ["prices_post2_1", "invest_post2_1", "prices_post3_1", "invest_post3_1"]:
    s = com.dropna(subset=[y])
    rows.append(ols2(s, y, ["msg", "prior"], "common reduced form (plans on msg, prior)"))
    rows.append(ols2(s, y, ["msg"], "common reduced form (no prior)"))
f = pd.DataFrame(rows); f["lo"] = f.coef - 1.96 * f.se; f["hi"] = f.coef + 1.96 * f.se
f["label"] = "msg=1 is the 'escalation' arm (three_info==1); OLS slopes elsewhere are between-owner associations in one response, not within-owner changes"
f.to_csv(OUT / "01_f_message_261102.csv", index=False); print(f[["sample", "y", "term", "coef", "se", "lo", "hi", "n", "F_firststage"]].to_string(index=False))
