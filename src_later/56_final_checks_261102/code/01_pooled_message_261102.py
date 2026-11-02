# ============================================================
# START DATE   : 2026-11-02
# CODE RUN DATE: 2026-11-02
# TIME TAKEN   : ~5 min
# DESCRIPTION  : Message effect on plans pooled across Waves 1-4, escalating (three_info 1) against softening (2); the uncertainty arm, W0 and W5 are left out.
#                y = d M + wave FE + industry + size + sales + China duty + u, owner-clustered. By wave and pooled; all owners and importers.
# INPUT FILES  : 40_smb_pricing_controls_261102/common_261102.py; analysis_all.dta (read only)
# OUTPUT FILES : analysis/56_final_checks_261102/01_pooled_message_261102.csv
# ============================================================
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "40_smb_pricing_controls_261102"))
import numpy as np, pandas as pd
import common_261102 as cm
OUT = cm.TOPIC / "analysis/56_final_checks_261102" / "01_pooled_message_261102.csv"
if OUT.exists(): sys.exit("refuse to overwrite")
d = cm.load_survey(); d["w"] = d.wave.astype(str)
d = d[d.w.str.match(r"W[1-4]") & d.three_info.isin([1, 2])].copy(); d["M"] = (d.three_info == 1).astype(float)
for k in (1, 2, 3):
    y = f"invest_post{k}_1"; d[f"shut{k}"] = (d[y] == -100).astype(float).where(d[y].notna()); d[f"adj{k}"] = d[y].where(d[y] > -100)
d["ind"] = d.industry_code.fillna(-1).astype(int)
Y = {"prices_post2_1": "price_planned", "invest_post2_1": "invest_planned", "shut2": "shutdown_planned", "adj2": "invest_planned_ex_shutdown",
     "prices_post3_1": "price_25pct", "invest_post3_1": "invest_25pct", "prices_post1_1": "price_done", "invest_post1_1": "invest_done"}
def est(s, y, lab, pop):
    s = s[s[y].notna()]
    X = pd.concat([s[["M"] + cm.BASE_CTRL].astype(float), cm.dummies(s.w, "w"), cm.dummies(s.ind, "i")], axis=1); X = X.loc[:, ~X.columns.duplicated()]
    m = cm.ols(s, y, X); b, se = m.params["M"], m.bse["M"]
    return dict(population=pop, sample=lab, outcome=Y[y], n=len(s), n_owners=s.uid.nunique(), n_M1=int(s.M.sum()), d=b, se=se, lo=b - 1.96 * se, hi=b + 1.96 * se, p=m.pvalues["M"], mde80=2.8 * se)
rows = []
for pop, g in [("all", d), ("importers", d[(d.importer_dum == 1) & (d.miss_importer_dum == 0)])]:
    for y in Y:
        rows.append(est(g, y, "pooled_W1_W4", pop))
        for w in sorted(g.w.unique()):
            if g[g.w == w][y].notna().sum() > 30: rows.append(est(g[g.w == w], y, w, pop))
r = pd.DataFrame(rows); r.to_csv(OUT, index=False); pd.set_option("display.width", 200)
print(r[r["sample"].eq("pooled_W1_W4")].round(3).to_string(index=False)); print(r[r.outcome.isin(["price_planned", "invest_planned"]) & r.population.eq("all")].round(3)[["sample", "outcome", "n", "d", "se"]].to_string(index=False))
