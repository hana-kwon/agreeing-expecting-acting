# ============================================================
# START DATE   : 2026-11-02
# CODE RUN DATE: 2026-11-02
# TIME TAKEN   : ~10 min
# DESCRIPTION  : Combine the per-tercile coefficient files, compute date-shift placebo summaries (share of shifts with
#                |b(shift)| >= |b(actual)|), and draw the placebo and coefficient figures. Aggregates only.
# INPUT FILES  : analysis/41_.../{02_coefs,06_coefs,03_placebo}_<tercile>_261102.csv, 08_coefs_employment_261102.csv
# OUTPUT FILES : analysis/41_.../04_*.csv, 04_fig_*.png (+ plotted data csv)
# ============================================================
import sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve(); OUT = HERE.parents[2] / "analysis/41_customs_team_spec_261102"
TERC = ["small", "medium", "large"]


def safe(p):
    p = Path(p)
    if p.exists(): sys.exit(f"output exists; refusing to overwrite: {p}")
    return p


cat = lambda pat: pd.concat([pd.read_csv(OUT / pat.format(t=t)) for t in TERC if (OUT / pat.format(t=t)).exists()], ignore_index=True)
main = cat("02_coefs_{t}_261102.csv"); main.to_csv(safe(OUT / "04_main_coefs_261102.csv"), index=False)
ext = cat("06_coefs_{t}_261102.csv"); ext.to_csv(safe(OUT / "04_extras_coefs_261102.csv"), index=False)
# the large tercile placebo was split over two processes (shifts up to 35 and 38-50); parts are concatenated when no single file exists
parts = []
for t in TERC:
    f1 = OUT / f"03_placebo_{t}_261102.csv"
    if f1.exists(): parts.append(pd.read_csv(f1))
    else: parts += [pd.read_csv(p) for p in sorted(OUT.glob(f"03_placebo_{t}_part*_261102.csv"))]
pl = pd.concat(parts, ignore_index=True); pl.to_csv(safe(OUT / "04_placebo_shifts_261102.csv"), index=False)
if (OUT / "08_coefs_employment_261102.csv").exists():
    pd.read_csv(OUT / "08_coefs_employment_261102.csv").to_csv(safe(OUT / "04_employment_coefs_261102.csv"), index=False)

# ★ CORE LOGIC: placebo p-value = share of shifted series whose |b| is at least |b| of the actual series, same subsample, same model.
rows = []
for (t, mod, term), d in pl.groupby(["tercile", "model", "term"]):
    if term in ("dutyC",): continue
    a = d[d.shift_weeks == 0]
    if a.empty: continue
    s = d[d.shift_weeks != 0]
    b0 = a.b.iloc[0]
    rows.append(dict(tercile=t, model=mod, term=term, firm_fraction=a.firm_fraction.iloc[0], b_actual=b0, se_actual=a.se.iloc[0], n_shifts=len(s),
                     share_abs_ge=(s.b.abs() >= abs(b0)).mean(), share_same_sign_and_ge=((np.sign(s.b) == np.sign(b0)) & (s.b.abs() >= abs(b0))).mean(),
                     shift_mean=s.b.mean(), shift_sd=s.b.std(), shift_p05=s.b.quantile(.05), shift_p95=s.b.quantile(.95),
                     z_actual_vs_shifts=(b0 - s.b.mean()) / s.b.std()))
ps = pd.DataFrame(rows); ps.to_csv(safe(OUT / "04_placebo_summary_261102.csv"), index=False)
pd.set_option("display.width", 220, "display.max_columns", 30); print(ps.round(4).to_string(index=False))

# ---- figure 1: placebo distributions of the signal coefficient ----
fig, axs = plt.subplots(1, 3, figsize=(12, 3.6), sharey=False)
for ax, t in zip(axs, TERC):
    d = pl[(pl.tercile == t) & (pl.model == "P_M1") & (pl.term == "sigS")]
    if d.empty: continue
    s = d[d.shift_weeks != 0]; a = d[d.shift_weeks == 0].b.iloc[0]
    ax.hist(s.b, bins=12, color="#9aa7b8", edgecolor="white")
    ax.axvline(a, color="#c0392b", lw=2); ax.axvline(0, color="black", lw=.6)
    p = (s.b.abs() >= abs(a)).mean()
    ax.set_title(f"{t} firms (placebo share {p:.2f}, n={len(s)})", fontsize=10); ax.set_xlabel("China x signal coefficient (log points per SD)")
axs[0].set_ylabel("number of shifted series")
fig.suptitle("Date-shift placebo: actual signal coefficient (red) vs 5-50 week shifts, trade value, Poisson", fontsize=10)
fig.tight_layout(); fig.savefig(safe(OUT / "04_fig_placebo_signal_261102.png"), dpi=160); plt.close(fig)

# ---- figure 2: main coefficients by tercile (signal, duty, post-Liberation Day) ----
sel = [("M1_signal_china_plus_duty", "sigC", "China x signal (per SD)"), ("M1_signal_china_plus_duty", "dutyC", "China x duty (per 10 pp)"),
       ("R_lib_full_window", "libC", "post-Liberation Day x China")]
fig, axs = plt.subplots(1, 3, figsize=(12, 3.4))
for ax, (spec, term, lab) in zip(axs, sel):
    d = main[(main.spec == spec) & (main.term == term) & (main.outcome == "trade_value")].set_index("tercile").reindex(TERC)
    ax.errorbar(range(3), d.est, yerr=1.96 * d.se_firm, fmt="o", color="#1f4e79", capsize=4, label="firm-clustered 95% CI")
    ax.errorbar(np.arange(3) + .12, d.est, yerr=1.96 * d.se_2way, fmt="none", ecolor="#c0392b", capsize=3, label="firm+period 95% CI")
    ax.axhline(0, color="black", lw=.6); ax.set_xticks(range(3)); ax.set_xticklabels(TERC); ax.set_title(lab, fontsize=10)
axs[0].legend(fontsize=7, loc="best"); axs[0].set_ylabel("Poisson coefficient, trade value")
fig.suptitle("Signal, duty and event coefficients by firm-size tercile (large: 30% firm sample)", fontsize=10)
fig.tight_layout(); fig.savefig(safe(OUT / "04_fig_main_coefs_261102.png"), dpi=160); plt.close(fig)

# ---- figure 3: alternatives interaction (difference between firms with and without a 2024 non-China origin) ----
sel = [("X1_alt_signal", "sigC_alt", "China x signal x alt"), ("X1b_alt_signal_grpbinFE", "sigC_alt", "same, origin x period FE"),
       ("L1_alt_lib", "libC_alt", "post-Liberation x China x alt"), ("L1b_alt_lib_grpbinFE", "libC_alt", "same, origin x period FE"),
       ("A1_any_signal_alt_indperiodFE", "sig_alt", "any import: signal x alt (LPM)"), ("A3_any_lib_alt_indperiodFE", "lib_alt", "any import: post-Lib x alt (LPM)")]
fig, axs = plt.subplots(2, 3, figsize=(12, 6))
for ax, (spec, term, lab) in zip(axs.ravel(), sel):
    d = ext[(ext.spec == spec) & (ext.term == term)].set_index("tercile").reindex(TERC)
    ax.errorbar(range(3), d.est, yerr=1.96 * d.se_firm, fmt="o", color="#1f4e79", capsize=4)
    ax.axhline(0, color="black", lw=.6); ax.set_xticks(range(3)); ax.set_xticklabels(TERC); ax.set_title(lab, fontsize=9)
fig.suptitle("Firms with a non-China origin in 2024 minus firms without, by size tercile (firm-clustered 95% CI)", fontsize=10)
fig.tight_layout(); fig.savefig(safe(OUT / "04_fig_alternatives_261102.png"), dpi=160); plt.close(fig)
print("figures written")
