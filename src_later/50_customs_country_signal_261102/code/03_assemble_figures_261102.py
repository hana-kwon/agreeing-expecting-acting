# ============================================================
# START DATE   : 2026-11-02
# CODE RUN DATE: 2026-11-02
# TIME TAKEN   : ~10 min
# DESCRIPTION  : Combine job outputs into one coefficient table with minimum detectable effects, summarize permutation and date-shift
#                placebos, and draw three figures (lag profile, placebo distribution, heterogeneity).
# INPUT FILES  : analysis/50_customs_country_signal_261102/02_job{A,B,C}_coefs_261102.csv, 02_jobC_placebo_261102.csv
# OUTPUT FILES : 03_all_coefs_mde_261102.csv, 03_placebo_summary_261102.csv, 03_fig_lags_261102.png, 03_fig_placebo_261102.png, 03_fig_hetero_261102.png
# ============================================================
import sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parents[2] / "analysis/50_customs_country_signal_261102"
for f in ["03_all_coefs_mde_261102.csv", "03_placebo_summary_261102.csv", "03_fig_lags_261102.png", "03_fig_placebo_261102.png", "03_fig_hetero_261102.png"]:
    if (OUT / f).exists(): sys.exit(f"output exists; refusing to overwrite: {OUT / f}")
co = pd.concat([pd.read_csv(OUT / f"02_job{j}_coefs_261102.csv") for j in "ABC"], ignore_index=True)
co["lo95"] = co.est - 1.96 * co.se_2way; co["hi95"] = co.est + 1.96 * co.se_2way
co.to_csv(OUT / "03_all_coefs_mde_261102.csv", index=False)
pl = pd.read_csv(OUT / "02_jobC_placebo_261102.csv")
rows = []
for (s, k), g in pl.groupby(["sample", "kind"]):
    a = g.actual.iloc[0]
    rows.append(dict(sample=s, kind=k, n=len(g), actual=a, placebo_mean=g.est.mean(), placebo_sd=g.est.std(), share_abs_ge=(g.est.abs() >= abs(a)).mean(),
                     share_same_sign_ge=((np.sign(g.est) == np.sign(a)) & (g.est.abs() >= abs(a))).mean(), mde_perm_2_8sd=2.8 * g.est.std()))
ps = pd.DataFrame(rows); ps.to_csv(OUT / "03_placebo_summary_261102.csv", index=False); print(ps.round(4).to_string())

# Figure 1: lag profile
a = co[co.spec.str.startswith("A1_lag") & co.term.str.startswith("S_")].copy(); a["k"] = a.spec.str.extract(r"lag(\d+)").astype(int)
fig, ax = plt.subplots(figsize=(7, 4))
for i, (sm, c) in enumerate([("all_10pct", "#1f4e79"), ("exChina_10pct", "#c0504d")]):
    d = a[a["sample"] == sm].sort_values("k"); x = d.k + (i - 0.5) * 0.5
    ax.errorbar(x, d.est, yerr=1.96 * d.se_2way, fmt="o", color=c, capsize=3, label="with China" if i == 0 else "China excluded")
ax.axhline(0, color="gray", lw=.8); ax.set_xlabel("Lag of the country statement signal (weeks before the two-week bin)")
ax.set_ylabel("PPML coefficient per SD of S\n(log points of trade value)"); ax.set_title("Threat signal S(o,t-k): one lag per model, firm x origin and firm x period FE\n10% firm sample, 95% CI two-way (origin, firm)", fontsize=9)
ax.legend(frameon=False); fig.tight_layout(); fig.savefig(OUT / "03_fig_lags_261102.png", dpi=160); plt.close(fig)

# Figure 2: placebo distributions
fig, axs = plt.subplots(1, 2, figsize=(9, 3.6), sharex=False)
for ax, sm in zip(axs, ["all_3pct", "exChina_3pct"]):
    for kind, c in [("perm_country", "#7f7f7f"), ("date_shift", "#e08a1e")]:
        g = pl[(pl["sample"] == sm) & (pl.kind == kind)]
        if len(g): ax.hist(g.est, bins=15, alpha=.55, color=c, label=kind.replace("_", " "))
    g0 = pl[pl["sample"] == sm];
    if len(g0): ax.axvline(g0.actual.iloc[0], color="k", lw=2, label="actual")
    ax.set_title(("with China" if sm == "all_3pct" else "China excluded") + " (3% firms)", fontsize=9); ax.set_xlabel("coefficient on cumulative threat signal")
axs[0].legend(frameon=False, fontsize=8); fig.tight_layout(); fig.savefig(OUT / "03_fig_placebo_261102.png", dpi=160); plt.close(fig)

# Figure 3: heterogeneity
h = co[co.spec.str.startswith("F_") & (co.term == "S_threats4_cum")].copy()
order = ["F_emp_e1_19", "F_emp_e20_49", "F_emp_e50_249", "F_emp_e250p", "F_alt23_0", "F_alt23_1"]
lab = {"F_emp_e1_19": "1-19", "F_emp_e20_49": "20-49", "F_emp_e50_249": "50-249", "F_emp_e250p": "250+", "F_alt23_0": "no non-China\norigin in 2023", "F_alt23_1": "non-China\norigin in 2023"}
fig, ax = plt.subplots(figsize=(7.5, 4))
for i, (sm, c) in enumerate([("all_25pct", "#1f4e79"), ("exChina_25pct", "#c0504d")]):
    d = h[h["sample"] == sm].set_index("spec").reindex(order).dropna(subset=["est"]); xs = [order.index(s) + (i - .5) * .25 for s in d.index]
    ax.errorbar(xs, d.est, yerr=1.96 * d.se_2way, fmt="o", color=c, capsize=3, label="with China" if i == 0 else "China excluded")
ax.set_xticks(range(6)); ax.set_xticklabels([lab[o] for o in order], fontsize=8); ax.axhline(0, color="gray", lw=.8); ax.axvline(3.5, color="lightgray", lw=.8)
ax.set_ylabel("PPML coefficient per SD of cumulative S"); ax.set_title("By D&B 2024 employment (left) and 2023 alternative origin (right); 25% firm sample, 95% CI", fontsize=9)
ax.legend(frameon=False); fig.tight_layout(); fig.savefig(OUT / "03_fig_hetero_261102.png", dpi=160); plt.close(fig)
print(co[co.term.str.startswith("S_")][["spec", "sample", "term", "est", "se_2way", "p_2way", "mde"]].round(4).to_string())
