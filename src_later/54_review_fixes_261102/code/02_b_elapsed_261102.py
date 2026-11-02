# ============================================================
# START DATE   : 2026-11-02
# CODE RUN DATE: 2026-11-02
# TIME TAKEN   : ~10 min
# DESCRIPTION  : B. Elapsed-time check for the importer gap in already-made price changes (cumulative, no start date):
#                period gaps per elapsed day, importer x late with days since 2025-02-01 (and importer-specific day slope), and
#                within-owner change among re-respondents (importer vs non-importer, per day).
# INPUT FILES  : see 40_smb_pricing_controls_261102/common_261102.py
# OUTPUT FILES : analysis/54_review_fixes_261102/02_b_period_gap_per_day_261102.csv, 02_b_regressions_261102.csv, 02_b_rerespondents_261102.csv
# ============================================================
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "40_smb_pricing_controls_261102"))
import numpy as np, pandas as pd
import common_261102 as cm
OUT = cm.TOPIC / "analysis/54_review_fixes_261102"
for f in ["02_b_period_gap_per_day_261102.csv", "02_b_regressions_261102.csv", "02_b_rerespondents_261102.csv"]:
    if (OUT / f).exists(): sys.exit("refuse to overwrite " + f)
d = cm.load_survey(); s = cm.prep_importer_sample(d)
s["days"] = (s.date - pd.Timestamp("2025-02-01")).dt.days.astype(float)
s["imp"] = s.importer_dum.astype(float)
per = {"p1": ("2025-03-01", "2025-04-02"), "p2": ("2025-04-03", "2025-05-31"), "p3": ("2025-06-01", "2025-08-31")}
s["period"] = np.nan
for k, (a, b) in per.items(): s.loc[(s.date >= a) & (s.date <= b), "period"] = int(k[1])
rows = []
for k in (1, 2, 3):
    g = s[s.period == k].dropna(subset=["done_pp"]); a, b = g[g.imp == 1], g[g.imp == 0]
    gap = a.done_pp.mean() - b.done_pp.mean(); se = np.hypot(a.done_pp.std() / np.sqrt(len(a)), b.done_pp.std() / np.sqrt(len(b)))
    md = g.days.mean(); rows.append(dict(period=k, n_imp=len(a), n_non=len(b), mean_days=md, gap=gap, gap_se=se, gap_per_day=gap / md, gap_per_day_se=se / md))
pd.DataFrame(rows).to_csv(OUT / "02_b_period_gap_per_day_261102.csv", index=False); print(pd.DataFrame(rows).round(3).to_string())
reg = []
def run(df, y, regs, label, extra_dum=()):
    df = df.dropna(subset=[y]).copy()
    X = pd.concat([df[regs + cm.BASE_CTRL + ["export_dum", "fcomp_dum"]].astype(float), cm.dummies(df.industry_code.astype(int), "i")] + [cm.dummies(df[c].astype(str), c) for c in extra_dum], axis=1)
    X = cm.drop_collinear(X.loc[:, X.std() > 0].assign(const=1.0))
    import statsmodels.api as sm
    m = sm.OLS(df[y].astype(float), X).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(df.uid)[0]})
    for t in regs:
        if t in m.params.index: reg.append(dict(spec=label, y=y, term=t, coef=m.params[t], se=m.bse[t], n=len(df)))
s2 = s[s.period.notna()].copy(); s2["late"] = (s2.period == 3).astype(float); s2["mid"] = (s2.period == 2).astype(float)
s2["imp_x_late"] = s2.imp * s2.late; s2["imp_x_mid"] = s2.imp * s2.mid; s2["imp_x_days"] = s2.imp * s2.days
for y in ["done_pp", "done_any"]:
    run(s2, y, ["imp", "imp_x_late", "imp_x_mid", "late", "mid"], "B0_period_dummies_only", ["wave"])
    run(s2, y, ["imp", "imp_x_late", "imp_x_mid", "days"], "B1_days_common_slope+importer_x_period", ["wave"])
    run(s2, y, ["imp", "imp_x_late", "imp_x_mid", "days", "imp_x_days"], "B2_days+importer_specific_slope", ["wave"])
    run(s2, y, ["imp", "days", "imp_x_days"], "B3_linear_only (no period dummies)", ["wave"])
pd.DataFrame(reg).to_csv(OUT / "02_b_regressions_261102.csv", index=False)
pd.set_option("display.width", 200, "display.float_format", lambda x: f"{x:.3f}"); print(pd.DataFrame(reg).to_string(index=False))
# re-respondents: earliest response with importer observed, later response >=14 days after; change in cumulative done_pp
dd = d.copy()
for k, lab in [(1, "done")]:
    y = dd[f"prices_post{k}_1"].where(dd[f"prices_post{k}_1"] > -100); dd["done_pp"] = y
dd = dd.sort_values(["uid", "date"]); pairs = []
for u, g in dd.groupby("uid"):
    if len(g) < 2: continue
    f = g.iloc[0]
    l = g[g.date > f.date + pd.Timedelta(days=14)]
    if len(l) and pd.notna(f.done_pp) and pd.notna(l.iloc[0].done_pp):
        r = l.iloc[0]; pairs.append(dict(uid=u, imp=f.importer_dum if f.miss_importer_dum == 0 else np.nan, d1=f.done_pp, d2=r.done_pp, gapd=(r.date - f.date).days, t1=f.date, t2=r.date))
p = pd.DataFrame(pairs); print("pairs with done at both:", len(p), "importer observed:", int(p.imp.notna().sum()))
p["chg"] = p.d2 - p.d1; p["chg_day"] = p.chg / p.gapd
out = []
import statsmodels.api as sm
q = p.dropna(subset=["imp"])
for y in ["chg", "chg_day"]:
    m = sm.OLS(q[y], sm.add_constant(q[["imp"]])).fit(cov_type="HC1"); out.append(dict(y=y, n=len(q), n_imp=int(q.imp.sum()), mean_non=q[q.imp == 0][y].mean(), mean_imp=q[q.imp == 1][y].mean(), diff=m.params["imp"], se=m.bse["imp"], mean_gap_days=q.gapd.mean()))
    m = sm.OLS(q[y], sm.add_constant(q[["imp", "gapd"]])).fit(cov_type="HC1"); out.append(dict(y=y + "|ctrl_gapdays", n=len(q), n_imp=int(q.imp.sum()), diff=m.params["imp"], se=m.bse["imp"], mean_gap_days=q.gapd.mean()))
pd.DataFrame(out).to_csv(OUT / "02_b_rerespondents_261102.csv", index=False); print(pd.DataFrame(out).to_string(index=False))
