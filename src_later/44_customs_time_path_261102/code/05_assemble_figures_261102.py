# ============================================================
# START DATE   : 2026-11-02
# CODE RUN DATE: 2026-11-02
# TIME TAKEN   : ~15 min
# DESCRIPTION  : Assemble aggregate coefficient tables and figures for code/44: P1 event-study figures (V1 relative to 2024 Q1, V2
#                relative to the same 2024 period) and key-date table, P2 window + duty table, P3 placebo summary and figure.
#                Aggregate coefficients only (no firm-level data).
# INPUT FILES  : analysis/44_customs_time_path_261102/{02_eventstudy,03_window_duty,04_placebo_windows}_<tercile>_261102.csv
# OUTPUT FILES : analysis/44_customs_time_path_261102/05_*.csv, 05_fig_*.png
# ============================================================
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TOPIC = Path(__file__).resolve().parents[2]
OUT = TOPIC / "analysis/44_customs_time_path_261102"
TERC = ["small", "medium", "large"]
LABEL = {"small": "Small firms", "medium": "Medium firms", "large": "Large firms (30% firm sample)"}
BLUE, GRAY, INK = "#2563a8", "#8a8f98", "#1f2328"
DATES = [("2025-02-04", "1", "10%"), ("2025-03-04", "2", "20%"), ("2025-04-02", "3", "30% (Apr 2-5)"), ("2025-04-10", "4", "145%"),
         ("2025-05-14", "5", "30%"), ("2025-11-10", "6", "20%"), ("2026-02-24", "7", "10%")]
targets = [OUT / n for n in ["05_key_dates_261102.csv", "05_p2_table_261102.csv", "05_p3_summary_261102.csv", "05_p3_shifts_261102.csv",
                             "05_trend_check_261102.csv", "05_fig_eventstudy_v1_261102.png", "05_fig_eventstudy_v2_261102.png", "05_fig_placebo_windows_261102.png"]]
for t in targets:
    if t.exists():
        sys.exit(f"output exists; refusing to overwrite: {t}")
plt.rcParams.update({"font.family": "sans-serif", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": GRAY, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK, "axes.titlesize": 10})

# ---------------- P1 ----------------
es = {t: pd.read_csv(OUT / f"02_eventstudy_{t}_261102.csv", parse_dates=["bin_start", "bin_end"]) for t in TERC}


def fig_es(spec, fname, xmin, title, ylab, shade_ref):
    fig, axes = plt.subplots(3, 1, figsize=(9.2, 9.4), sharex=True)
    for ax, t in zip(axes, TERC):
        d = es[t][es[t].spec == spec].sort_values("bin")
        x = d.bin_start + pd.Timedelta(days=7)
        ax.fill_between(x, d.lo, d.hi, color=BLUE, alpha=0.16, lw=0)
        ax.plot(x, d.est, color=BLUE, lw=1.4, marker="o", ms=3)
        ax.axhline(0, color=GRAY, lw=0.8)
        if shade_ref:
            ax.axvspan(pd.Timestamp("2024-01-01"), pd.Timestamp("2024-03-24"), color=GRAY, alpha=0.18, lw=0)
        for ds, num, _ in DATES:
            ax.axvline(pd.Timestamp(ds), color=GRAY, lw=0.8, ls=(0, (3, 3)))
        ax.set_ylabel(ylab)
        r = d.iloc[0]
        ax.set_title(f"{LABEL[t]} (estimation sample: {int(r.n_firms):,} firms, {int(r.nobs):,} firm-origin-period obs)", loc="left", color=INK)
        ax.set_xlim(pd.Timestamp(xmin), pd.Timestamp("2026-04-05"))
    for ds, num, _ in DATES:
        axes[0].text(pd.Timestamp(ds), axes[0].get_ylim()[1], num, ha="center", va="bottom", fontsize=8, color=INK)
    foot = "Vertical lines: " + "; ".join(f"{n} = {d} (China duty {r})" for d, n, r in DATES) + "."
    fig.suptitle(title, x=0.01, ha="left", fontsize=11, color=INK)
    fig.text(0.01, 0.005, foot + " Points: PPML coefficient on China x period (log points); band: 95% CI, firm-clustered SE.\n"
             "FE: firm x origin group and firm x period. Origin groups: China area, other East Asia, Mexico, Canada, EU, rest. Gray bar: reference period (2024 Q1).",
             fontsize=7.2, color=INK, va="bottom", ha="left", wrap=True)
    fig.tight_layout(rect=(0, 0.05, 1, 0.965))
    fig.savefig(OUT / fname, dpi=170); plt.close(fig)


fig_es("V1_ref_2024Q1", "05_fig_eventstudy_v1_261102.png", "2023-12-15",
       "China-area imports relative to the same firm's other origins, by two-week period (reference: 2024 Q1)",
       "log points vs 2024 Q1", True)
fig_es("V2_same_period_2024", "05_fig_eventstudy_v2_261102.png", "2024-12-15",
       "Same, relative to the same two-week period of 2024 (removes seasonality, not trend)", "log points vs same period 2024", False)

# key dates: the bin containing each date and neighbors (-1, 0, +1, +2)
rows = []
for t in TERC:
    for spec in ["V1_ref_2024Q1", "V2_same_period_2024"]:
        d = es[t][es[t].spec == spec].set_index("bin")
        for ds, num, rate in DATES:
            dd = pd.Timestamp(ds)
            hit = es[t][(es[t].spec == "S_rel_bin53") & (es[t].bin_start <= dd) & (es[t].bin_end >= dd)]
            b0 = int(hit.bin.iloc[0])
            for rel in (-1, 0, 1, 2):
                b = b0 + rel
                if b == 79: b += 1 if rel >= 0 else -1
                if b in d.index:
                    r = d.loc[b]
                    rows.append(dict(tercile=t, spec=spec, date=ds, duty=rate, bin_of_date=b0, rel=rel, bin=b, year_biweek=r.year_biweek,
                                     bin_start=r.bin_start.date(), est=r.est, se=r.se, lo=r.lo, hi=r.hi))
kd = pd.DataFrame(rows); kd.to_csv(OUT / "05_key_dates_261102.csv", index=False)

# descriptive trend check (exploratory, added after seeing the 2024 path): linear trend of V1 over bins 53-78 extrapolated to later bins
tr = []
for t in TERC:
    d = es[t][es[t].spec == "V1_ref_2024Q1"].set_index("bin")
    x = d.loc[53:78].index.values.astype(float); y = d.loc[53:78].est.values
    slope, icpt = np.polyfit(x, y, 1)
    # slope in log points per two-week period; also fitted value at bin 78
    for name, lo_b, hi_b in [("frontload bins 80-87", 80, 87), ("threat bins 88-92", 88, 92), ("resolved bins 93-111", 93, 111)]:
        sel = d.loc[lo_b:hi_b]
        fit = icpt + slope * sel.index.values.astype(float)
        tr.append(dict(tercile=t, slope_per_period_2024=slope, mean_est=sel.est.mean(), mean_linear_extrapolation=fit.mean(),
                       mean_deviation_from_trend=(sel.est.values - fit).mean(), window=name,
                       mean_2024Q2_Q4=d.loc[59:78].est.mean(), est_last2024_bins_mean=d.loc[76:78].est.mean()))
pd.DataFrame(tr).to_csv(OUT / "05_trend_check_261102.csv", index=False)

# ---------------- P2 ----------------
p2 = pd.concat([pd.read_csv(OUT / f"03_window_duty_{t}_261102.csv") for t in TERC])
p2.to_csv(OUT / "05_p2_table_261102.csv", index=False)

# ---------------- P3 ----------------
p3 = pd.concat([pd.read_csv(OUT / f"04_placebo_windows_{t}_261102.csv") for t in TERC])
pw = pd.read_csv(OUT / "01_placebo_windows_261102.csv")
summ, shifts = [], []
fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.9), sharey=False)
for ax, t in zip(axes, TERC):
    x = p3[(p3.tercile == t) & (p3.term == "cR - cT")]
    act = x[x.kind == "actual_same_subsample"].iloc[0]
    pl = x[x.kind == "placebo"].merge(pw[["shift_weeks", "n_res", "n_thr", "n_front", "n_ref"]], on="shift_weeks")
    pl0 = pl[pl.shift_weeks == 0]; pln = pl[pl.shift_weeks != 0]
    a = act.est
    summ.append(dict(tercile=t, firm_fraction=act.firm_fraction, actual_contrast_same_subsample=a, actual_se=act.se,
                     n_placebo_shifts=len(pln), share_abs_ge_actual=(pln.est.abs() >= abs(a)).mean(),
                     share_same_sign_ge=((pln.est <= a) if a < 0 else (pln.est >= a)).mean(),
                     n_ge_abs=int((pln.est.abs() >= abs(a)).sum()), n_same_sign_ge=int(((pln.est <= a) if a < 0 else (pln.est >= a)).sum()),
                     placebo_mean=pln.est.mean(), placebo_sd=pln.est.std(), placebo_min=pln.est.min(), placebo_max=pln.est.max(),
                     n_shifts_res12=int((pln.n_res >= 12).sum()), share_abs_ge_actual_res12=(pln[pln.n_res >= 12].est.abs() >= abs(a)).mean(),
                     placebo_mean_neg_shifts=pln[pln.shift_weeks < 0].est.mean(), placebo_mean_pos_shifts=pln[pln.shift_weeks > 0].est.mean(),
                     k0_contrast=float(pl0.est.iloc[0]) if len(pl0) else np.nan, k0_se=float(pl0.se.iloc[0]) if len(pl0) else np.nan))
    shifts.append(pl.assign(tercile=t, actual=a))
    ax.axhline(a, color=BLUE, lw=1.6)
    ax.axhline(0, color=GRAY, lw=0.8)
    ax.scatter(pln.shift_weeks, pln.est, s=14, color=GRAY, zorder=3)
    if len(pl0):
        ax.scatter([0], pl0.est, s=34, color=INK, marker="D", zorder=4)
    ax.text(0.02, 0.04, f"actual {a:.2f}", transform=ax.transAxes, color=BLUE, fontsize=8.5, va="bottom")
    ax.set_title(LABEL[t].replace(" (30% firm sample)", ""), loc="left", color=INK)
    ax.set_xlabel("window shift (weeks from 2024-01-01 start)")
    ax.set_ylabel("resolved minus threat (log points)" if t == "small" else "")
fig.suptitle("Placebo: resolved-minus-threat contrast with the window dates moved in a fake year (2023-2024 sample)", x=0.01, ha="left", fontsize=10.5, color=INK)
fig.text(0.01, 0.005, "Gray dots: shifts of +-8..+-26 weeks that meet the window-length rule; diamond: shift 0 (same calendar in 2024). Blue line: actual contrast (2024-2026 panel, same firm subsample).",
         fontsize=7.2, color=INK, va="bottom")
fig.tight_layout(rect=(0, 0.04, 1, 0.94)); fig.savefig(OUT / "05_fig_placebo_windows_261102.png", dpi=170); plt.close(fig)
pd.DataFrame(summ).to_csv(OUT / "05_p3_summary_261102.csv", index=False)
pd.concat(shifts).to_csv(OUT / "05_p3_shifts_261102.csv", index=False)
pd.set_option("display.width", 250, "display.max_columns", 40)
print(pd.DataFrame(summ).round(3).T.to_string())
print(kd[kd.rel.isin([0])].round(2).to_string(index=False))
print(pd.DataFrame(tr).round(2).to_string(index=False))
