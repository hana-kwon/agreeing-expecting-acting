# ============================================================
# START DATE   : 2026-11-02
# CODE RUN DATE: 2026-11-02
# TIME TAKEN   : ~10 min
# DESCRIPTION  : Figures for step 36: (1) China-area imports index Mar-Apr=100 in 2023/2024/2025; (2) C1 pooled event-study with placebo band.
# INPUT FILES  : analysis/36_census_imports_261102/04_c3_series.csv, 02_c1_pooled_beta.csv
# OUTPUT FILES : analysis/36_census_imports_261102/05_fig1_china_index_261102.png, 05_fig2_c1_eventstudy_261102.png
# ============================================================
import pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import common_261102 as cm
OUT = cm.get_out()
s = pd.read_csv(OUT / "04_c3_series.csv")
fig, ax = plt.subplots(figsize=(7.5, 4.5))
cols = {2023: "#9aa5b1", 2024: "#5b7fa6", 2025: "#c0392b"}
for y in (2023, 2024, 2025):
    x = s[(s.group == "china_area") & (s.year == y) & (s.month.between(1, 12))].sort_values("month")
    ax.plot(x.month, x["index"], marker="o", ms=3, color=cols[y], lw=2 if y == 2025 else 1.3, label=str(y))
x = s[(s.group == "china_area") & (s.year == 2026)].sort_values("month")
ax.plot(x.month, x["index"], marker="o", ms=3, color=cols[2025], lw=1.3, ls="--", label="2026 (vs Mar-Apr 2025)")
ax.axhline(100, color="k", lw=.6); ax.axvspan(2.5, 4.5, color="#eee", zorder=0)
ax.set_xticks(range(1, 13)); ax.set_xlabel("Month of year"); ax.set_ylabel("China + Hong Kong + Macau imports\n(index, same-year Mar-Apr average = 100)")
ax.set_title("U.S. imports from China area fell sharply in May 2025 and had not returned to\nthe usual seasonal pattern by August (Census, all modes, all HS2)", fontsize=10)
ax.legend(frameon=False, fontsize=8); fig.tight_layout()
p1 = OUT / "05_fig1_china_index_261102.png"
assert not p1.exists(); fig.savefig(p1, dpi=160)
c = pd.read_csv(OUT / "02_c1_pooled_beta.csv"); c = c[c.variant == "main_weighted"]
fig, ax = plt.subplots(figsize=(7.5, 4.3))
ax.fill_between(c.k, c.plc_p05, c.plc_p95, color="#ddd", label="shifted-date placebo 5-95%")
ax.errorbar(c.k, c.beta, yerr=1.96 * c.se, fmt="o-", color="#1f4e79", capsize=2, label="actual (95% CI, country bootstrap)")
ax.axhline(0, color="k", lw=.6); ax.axvline(-0.5, color="#c0392b", ls=":")
ax.set_xlabel("Months relative to effective month of a country-specific tariff increase"); ax.set_ylabel("Log YoY imports, affected minus unaffected countries\n(baseline: months -6 to -4)")
ax.set_title("No clear front-loading before effective dates: pre-period estimates sit inside the placebo band\n(7 event months, 13 effective dates; exploratory)", fontsize=10)
ax.legend(frameon=False, fontsize=8); fig.tight_layout()
p2 = OUT / "05_fig2_c1_eventstudy_261102.png"
assert not p2.exists(); fig.savefig(p2, dpi=160)
