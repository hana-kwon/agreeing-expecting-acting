# ============================================================
# START DATE   : 2026-11-02
# CODE RUN DATE: 2026-11-02
# TIME TAKEN   : ~5 min
# DESCRIPTION  : Figure: within-country AUC (weeks 2-12 ahead vs earlier 2-12 weeks) of each statement measure for effective tariff increases (O1), with bootstrap CI.
# INPUT FILES  : analysis/53_first_stage_261102/02_auc_261102.csv
# OUTPUT FILES : analysis/53_first_stage_261102/09_fig_auc_261102.png, 09_fig_data_261102.csv
# ============================================================
import sys
from pathlib import Path
import pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
OUT = Path(__file__).resolve().parents[2] / "analysis/53_first_stage_261102"
if (OUT / "09_fig_auc_261102.png").exists(): sys.exit("output exists; refusing to overwrite")
a = pd.read_csv(OUT / "02_auc_261102.csv"); a = a[(a.outcome == "O1_eff_incr") & a.window.isin(["fut_2_12", "past_2_12"])]
a.to_csv(OUT / "09_fig_data_261102.csv", index=False)
order = list(a[a.window == "fut_2_12"].sort_values("auc_within_country").measure)
fig, ax = plt.subplots(figsize=(8, 7))
for w, col, off in [("fut_2_12", "#1f4e79", 0.12), ("past_2_12", "#b0b0b0", -0.12)]:
    d = a[a.window == w].set_index("measure").loc[order]
    ax.errorbar(d.auc_within_country, [i + off for i in range(len(order))], xerr=[d.auc_within_country - d.wc_lo, d.wc_hi - d.auc_within_country], fmt="o", color=col, ms=4, lw=1,
                label="effective increase in the NEXT 2-12 weeks" if w == "fut_2_12" else "effective increase in the PREVIOUS 2-12 weeks (statement after the event)")
ax.axvline(0.5, color="k", lw=0.8); ax.set_yticks(range(len(order))); ax.set_yticklabels(order, fontsize=8)
ax.set_xlabel("Within-country AUC (0.5 = no information); 95% bin-block bootstrap CI, 300 draws"); ax.legend(fontsize=7, loc="lower right")
ax.set_title("Do statement measures predict tariff increases within a country?\n96 countries x 44 two-week bins, 2025-01 to 2026-09", fontsize=10)
plt.tight_layout(); plt.savefig(OUT / "09_fig_auc_261102.png", dpi=150)
