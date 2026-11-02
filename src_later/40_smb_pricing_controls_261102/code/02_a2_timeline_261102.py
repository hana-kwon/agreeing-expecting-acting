# ============================================================
# START DATE   : 2026-11-02
# CODE RUN DATE: 2026-11-02
# TIME TAKEN   : ~15 min
# DESCRIPTION  : A2. Small-firm price timeline by response week (March-August 2025): mean already-made price
#                change and share reporting any increase, importers vs non-importers, 95% CIs; regression version
#                (importer x week dummies + team controls + week and wave FE); next to the Cavallo-Llamas-Vazquez
#                large-retailer index (imported minus domestic) and the leader China-path signal, one 3-panel figure.
# INPUT FILES  : see common_261102.py
# OUTPUT FILES : analysis/40_smb_pricing_controls_261102/02_weekly_cells_261102.csv, 02_weekly_regression_261102.csv,
#                02_period_gap_261102.csv, 02_fig_data_261102.csv, 02_fig_smb_vs_retail_261102.png
#                (aggregates only; no respondent rows)
# ============================================================
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from common_261102 import (SIG, TEAM_CTRL, drop_collinear, dummies, get_out, load_cavallo_weekly, load_events, load_survey, ols,
                           parts, safe_write, signal)

out, args = get_out()
d = load_survey()
MINN = 20                       # figure and regression report only weeks with n >= 20 in BOTH groups
for p in ["02_weekly_cells_261102.csv", "02_weekly_regression_261102.csv", "02_period_gap_261102.csv",
          "02_fig_data_261102.csv", "02_fig_smb_vs_retail_261102.png"]:
    if (out / p).exists():
        raise SystemExit(f"refuse to overwrite {out / p}")

# ★ CORE LOGIC: shutdown answers (-100) are not price changes; they become missing for A2/A3.
# Importer status missing (question not asked) rows are dropped, never filled with 0 as in alignment_for_stata.dta.
s = d[d.miss_importer_dum == 0].copy()
for k, lab in [(1, "done"), (2, "plan")]:
    y = s[f"prices_post{k}_1"].where(s[f"prices_post{k}_1"] > -100)
    s[f"{lab}_pp"] = y
    s[f"{lab}_any"] = (100 * (y > 0)).where(y.notna())
n_shut = {lab: int((d[f"prices_post{k}_1"] == -100).sum()) for k, lab in [(1, "done"), (2, "plan")]}

# ---- weekly cells ----
rows = []
for lab in ["done", "plan"]:
    for (wk, imp), g in s.groupby(["week", "importer_dum"]):
        for kind in ["pp", "any"]:
            v = g[f"{lab}_{kind}"].dropna()
            if len(v) >= 2:
                rows.append(dict(outcome=f"{lab}_{kind}", week=wk, importer=int(imp), n=len(v), mean=v.mean(),
                                 se=v.std(ddof=1) / np.sqrt(len(v))))
cells = pd.DataFrame(rows)
cells["ci_low"] = cells["mean"] - 1.96 * cells.se
cells["ci_high"] = cells["mean"] + 1.96 * cells.se
safe_write(cells, out / "02_weekly_cells_261102.csv")

# ---- regression version: gamma_w = importer minus non-importer in week w ----
reg_rows = []
for lab in ["done", "plan"]:
    for kind in ["pp", "any"]:
        y = f"{lab}_{kind}"
        df = s.dropna(subset=[y]).copy()
        ok_weeks = [w for w, g in df.groupby("week") if (g.groupby("importer_dum").size().reindex([0.0, 1.0]).fillna(0) >= MINN).all()]
        df["wk"] = df.week.dt.strftime("%Y-%m-%d")
        inter = pd.DataFrame({f"imp_x_{w.strftime('%Y-%m-%d')}": ((df.week == w) & (df.importer_dum == 1)).astype(float)
                              for w in sorted(df.week.unique())})
        ctrl = df[["export_dum", "fcomp_dum", "emp_2024_bins", "sales_2024_val", "miss_emp_2024_bins", "miss_sales_2024_val",
                   "miss_export_dum", "miss_fcomp_dum"]].astype(float)
        X = pd.concat([inter, dummies(df.wk, "wk"), dummies(df.wave.astype(str), "w"),
                       dummies(df.industry_code.astype(int), "i"), ctrl], axis=1)
        X = X.loc[:, X.std() > 0]
        keep_inter = [c for c in X.columns if c.startswith("imp_x_")]
        rest = drop_collinear(X.drop(columns=keep_inter).assign(const=1.0))
        X2 = pd.concat([inter[keep_inter], rest], axis=1)
        m = sm.OLS(df[y].astype(float), X2).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(df.uid)[0]})
        for c in keep_inter:
            w = pd.Timestamp(c.replace("imp_x_", ""))
            reg_rows.append(dict(outcome=y, week=w, gamma=m.params[c], se=m.bse[c], n_imp=int(((df.week == w) & (df.importer_dum == 1)).sum()),
                                 n_non=int(((df.week == w) & (df.importer_dum == 0)).sum()), plotted=w in ok_weeks))
reg = pd.DataFrame(reg_rows)
safe_write(reg, out / "02_weekly_regression_261102.csv")

# ---- pooled period gaps (raw means), three periods around the policy dates ----
per = []
bins = [("1_before_Apr2 (Mar 21-Apr 2)", "2025-03-01", "2025-04-02"), ("2_after_Apr2 (Apr 3-May 2)", "2025-04-03", "2025-05-31"),
        ("3_summer (Jun 9-Aug 15)", "2025-06-01", "2025-08-31")]
for lab in ["done", "plan"]:
    for kind in ["pp", "any"]:
        for name, a_, b_ in bins:
            g = s[(s.date >= a_) & (s.date <= b_)]
            r = dict(outcome=f"{lab}_{kind}", period=name)
            for imp in (0, 1):
                v = g[g.importer_dum == imp][f"{lab}_{kind}"].dropna()
                r[f"n_{imp}"], r[f"mean_{imp}"], r[f"se_{imp}"] = len(v), v.mean(), v.std(ddof=1) / np.sqrt(len(v))
            r["gap"] = r["mean_1"] - r["mean_0"]
            r["gap_se"] = np.hypot(r["se_0"], r["se_1"])
            per.append(r)
safe_write(pd.DataFrame(per), out / "02_period_gap_261102.csv")

# ---- figure data: retail weekly + leader signal ----
retail_daily, retail_wk = load_cavallo_weekly()
ev = load_events()
pc = parts(ev, *SIG["S_path_china"])
days = pd.date_range("2025-02-24", "2025-08-17")
sig_daily = signal(pc, days)
cd = cells[cells.outcome == "done_pp"].pivot(index="week", columns="importer", values=["mean", "se", "n"])
cd.columns = [f"{a}_{b}" for a, b in cd.columns]
cd = cd.reset_index().rename(columns={"week": "week"})
cd["gap"] = cd.mean_1 - cd.mean_0
cd["gap_se"] = np.hypot(cd.se_0, cd.se_1)
cd["plotted"] = (cd.n_0 >= MINN) & (cd.n_1 >= MINN)
cd = cd.merge(retail_wk, left_on="week", right_index=True, how="left")
cd = cd.merge(reg[reg.outcome == "done_pp"][["week", "gamma", "se"]].rename(columns={"gamma": "reg_gap", "se": "reg_gap_se"}), on="week", how="left")
safe_write(cd, out / "02_fig_data_261102.csv")

BLUE, VERM, GREY, INK = "#0072B2", "#D55E00", "#6b7280", "#1f2937"
fig, ax = plt.subplots(3, 1, figsize=(9.2, 10.4), sharex=True, gridspec_kw=dict(height_ratios=[1.1, 1.1, 0.8]))
pw = cd[cd.plotted]
mid = lambda w: w + pd.Timedelta(days=3)
for imp, col, lab in [(0, GREY, "Non-importers"), (1, BLUE, "Importers")]:
    ax[0].errorbar(mid(pw.week), pw[f"mean_{imp}"], yerr=1.96 * pw[f"se_{imp}"], color=col, marker="o", ms=5, lw=0, elinewidth=1.5, capsize=2.5, label=lab)
ax[0].set_ylabel("Already-made price change\n(mean, percent)")
ax[0].set_title("(a) Small U.S. firms: price change already made, by response week (n >= 20 per group)", fontsize=10, loc="left", color=INK)
ax[0].legend(frameon=False, loc="upper center", fontsize=9, ncol=2)
ax[1].errorbar(mid(pw.week), pw.gap, yerr=1.96 * pw.gap_se, color=BLUE, marker="o", ms=5, lw=0, elinewidth=1.5, capsize=2.5, label="Small firms: importer minus non-importer (raw, 95% CI)")
pr = pw.dropna(subset=["reg_gap"])
ax[1].plot(mid(pr.week), pr.reg_gap, ls="none", marker="D", mfc="white", mec=BLUE, ms=6, label="Small firms: same gap with team controls and week + wave FE")
ax[1].plot(retail_daily.index, retail_daily.values, color=VERM, lw=1.0, alpha=0.45)
ax[1].plot(mid(retail_wk.index), retail_wk.retail_imp_minus_dom, color=VERM, lw=2.0, label="Five large retailers (Cavallo et al.): imported minus domestic, log points x 100")
ax[1].axhline(0, color=GREY, lw=0.6)
ax[1].set_ylabel("Percentage points\n(retail rebased to Jan 6-Feb 2, 2025)")
ax[1].set_title("(b) Small-firm importer gap next to the large-retailer imported-minus-domestic index", fontsize=10, loc="left", color=INK)
ax[1].legend(frameon=False, loc="upper left", fontsize=8)
ax[2].plot(sig_daily.index, sig_daily.values, color=INK, lw=1.4)
ax[2].axhline(0, color=GREY, lw=0.6)
ax[2].set_ylabel("China tariff-path signal\n(28-day, higher minus lower)")
ax[2].set_title("(c) Leader China-tariff signal: higher minus lower, prior 28 days (Jev coding)", fontsize=10, loc="left", color=INK)
for a in ax:
    for dt_, lab_ in [("2025-03-04", "Mar 4\ntariffs on CA/MX/CN"), ("2025-04-02", "Apr 2\nLiberation Day"), ("2025-05-12", "May 12\nChina pause")]:
        a.axvline(pd.Timestamp(dt_), color=GREY, lw=0.8, ls=":")
    a.spines[["top", "right"]].set_visible(False)
    a.tick_params(labelsize=8)
    a.grid(axis="y", color="#e5e7eb", lw=0.6)
for dt_, lab_ in [("2025-03-04", "Mar 4"), ("2025-04-02", "Apr 2"), ("2025-05-12", "May 12")]:
    ax[2].text(pd.Timestamp(dt_) + pd.Timedelta(days=1), 0.02, lab_, fontsize=8, color=GREY, va="bottom", transform=ax[2].get_xaxis_transform())
ax[2].xaxis.set_major_locator(mdates.MonthLocator())
ax[2].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
ax[2].set_xlim(pd.Timestamp("2025-02-24"), pd.Timestamp("2025-08-17"))
fig.text(0.01, 0.005, "Survey: U.S. SMB tariff waves, cross-sections (response week = Monday start); 'already made' is cumulative, no stated start date.\n"
         "Retail index: HBS Pricing Lab, daily (thin) and weekly mean (thick). Dotted lines: Mar 4, Apr 2, May 12. Exploratory, not causal.", fontsize=7, color=GREY)
fig.tight_layout(rect=(0, 0.03, 1, 1))
fig.savefig(out / "02_fig_smb_vs_retail_261102.png", dpi=170)

pd.set_option("display.width", 220, "display.float_format", lambda x: f"{x:.2f}")
print("shutdown answers set to missing:", n_shut)
print(cd[["week", "n_0", "n_1", "mean_0", "mean_1", "gap", "gap_se", "reg_gap", "reg_gap_se", "retail_imp_minus_dom", "plotted"]].to_string(index=False))
print(pd.DataFrame(per).to_string(index=False))
print(reg[reg.outcome.isin(["done_any", "plan_pp"])].to_string(index=False))
