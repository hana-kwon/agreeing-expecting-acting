# ============================================================
# START DATE   : 2026-11-02
# CODE RUN DATE: 2026-11-02
# TIME TAKEN   : ~3 min
# DESCRIPTION  : y_iw = wave-version FE + tau_w M + kappa_w B + eta_w M x Bc + X + u, owner-clustered; pooled effect = sum (n_w/N) tau_w with its delta-method SE.
#                This is the average escalating-minus-softening message difference, not an effect per point of expectation. Wave 4 text versions are assigned by response date (07-09, 07-18).
# INPUT FILES  : 40_smb_pricing_controls_261102/common_261102.py; analysis_all.dta (read only)
# OUTPUT FILES : analysis/56_final_checks_261102/01b_pooled_by_wave_261102.csv
# ============================================================
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "40_smb_pricing_controls_261102"))
import numpy as np, pandas as pd
import common_261102 as cm
OUT = cm.TOPIC / "analysis/56_final_checks_261102/01b_pooled_by_wave_261102.csv"
if OUT.exists(): sys.exit("refuse to overwrite")
d = cm.load_survey(); ex = pd.read_stata(cm.IA / "inputs/analysis_all.dta", columns=["responseid", "exposure_treat"], convert_categoricals=False); ex["exposure_treat"] = pd.to_numeric(ex.exposure_treat, errors="coerce")
d = d.merge(ex, on="responseid", how="left", validate="1:1"); d["w"] = d.wave.astype(str)
d = d[d.w.str.match(r"W[1-4]") & d.three_info.isin([1, 2])].copy(); d["M"] = (d.three_info == 1).astype(float); d["ind"] = d.industry_code.fillna(-1).astype(int)
print(d.groupby("w").exposure_treat.agg(["size", "count", "mean"]).round(3).to_string())
d["cellv"] = d.w
m4 = d.w.str.startswith("W4"); d.loc[m4 & (d.date >= "2025-07-09"), "cellv"] = "W4_v2"; d.loc[m4 & (d.date >= "2025-07-18"), "cellv"] = "W4_v3"
d["adj2"] = d.invest_post2_1.where(d.invest_post2_1 > -100)
rows = []
for pop, g0 in [("all", d), ("importers", d[(d.importer_dum == 1) & (d.miss_importer_dum == 0)])]:
    for y, lab in [("prices_post2_1", "price_planned"), ("invest_post2_1", "invest_planned"), ("adj2", "invest_planned_ex_shutdown")]:
        s = g0[g0[y].notna()].copy(); W = sorted(s.w.unique()); s["B"] = s.exposure_treat.fillna(0).astype(float); s["Bm"] = s.exposure_treat.isna().astype(float); cols = []
        for w in W:
            k = s.w == w; bc = s.B - s.loc[k, "B"].mean()
            s[f"M_{w}"] = s.M * k; s[f"MB_{w}"] = s.M * bc * k; s[f"B_{w}"] = s.B * k; cols += [f"M_{w}", f"MB_{w}", f"B_{w}"]
        X = pd.concat([s[cols + ["Bm"] + cm.BASE_CTRL].astype(float), cm.dummies(s.cellv, "v"), cm.dummies(s.ind, "i")], axis=1); X = X.loc[:, ~X.columns.duplicated()]
        m = cm.ols(s, y, X); nw = np.array([(s.w == w).sum() for w in W], float); wt = nw / nw.sum(); nm = [f"M_{w}" for w in W]
        b = m.params[nm].values; V = m.cov_params().loc[nm, nm].values; est = wt @ b; se = float(np.sqrt(wt @ V @ wt))
        rows.append(dict(population=pop, outcome=lab, row="weighted_average", n=int(nw.sum()), n_owners=s.uid.nunique(), tau=est, se=se, lo=est - 1.96 * se, hi=est + 1.96 * se))
        for w, t, sd, n in zip(W, b, np.sqrt(np.diag(V)), nw): rows.append(dict(population=pop, outcome=lab, row=w, n=int(n), n_owners=np.nan, tau=t, se=sd, lo=t - 1.96 * sd, hi=t + 1.96 * sd))
r = pd.DataFrame(rows); r.to_csv(OUT, index=False); pd.set_option("display.width", 180); print(r.round(3).to_string(index=False))
