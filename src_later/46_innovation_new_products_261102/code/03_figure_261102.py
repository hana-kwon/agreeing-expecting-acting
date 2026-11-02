# ============================================================
# START DATE   : 2026-11-02
# CODE RUN DATE: 2026-11-02
# TIME TAKEN   : ~5 min
# DESCRIPTION  : Figure: main (2025-2024) vs pre-trend placebo (2024-2023) exposure coefficients, in SD units of the outcome change.
# INPUT FILES  : P/analysis/46_innovation_new_products_261102/log_specs_261102.csv
# OUTPUT FILES : P/analysis/46_innovation_new_products_261102/fig_main_vs_placebo_261102.png (+ plotted data csv)
# ============================================================
import sys
from pathlib import Path
import pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
P = Path("HOME")
od = P/"analysis/46_innovation_new_products_261102"
out = od/"fig_main_vs_placebo_261102.png"
if out.exists(): sys.exit("refusing to overwrite")
L = pd.read_csv(od/"log_specs_261102.csv")
outs = ["new_vs","new_any","new_n","newpair_vs","newpair_n","neworig_n","drop_n","n_hs6"]
lab = {"new_vs":"New-product value share","new_any":"Any new product","new_n":"# new products","newpair_vs":"New product-origin value share",
       "newpair_n":"# new product-origin pairs","neworig_n":"# new origins","drop_n":"# dropped products","n_hs6":"# HS6 (total)"}
rows = []
fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
for ax, samp in zip(axes, ["A", "B"]):
    for k, (spec, col, off) in enumerate([("main_2025m2024", "#c0392b", 0.15), ("placebo_2024m2023", "#7f8c8d", -0.15)]):
        d = L[(L.spec == spec) & (L["sample"] == samp) & L.outcome.isin(["d_" + o for o in outs])].copy()
        d["o"] = d.outcome.str[2:]; d = d.set_index("o").loc[outs]
        z = d.est_per10pp / d.sd_dep; e = 1.96 * d.se / d.sd_dep
        ypos = [i + off for i in range(len(outs))]
        ax.errorbar(z, ypos, xerr=e, fmt="o", color=col, label={"main_2025m2024": "Main: 2025 minus 2024", "placebo_2024m2023": "Placebo: 2024 minus 2023"}[spec], capsize=2)
        for o in outs: rows.append(dict(sample=samp, spec=spec, outcome=o, n=d.loc[o, "n"], est_per10pp=d.loc[o, "est_per10pp"], se=d.loc[o, "se"], z_sd=z[o]))
    ax.axvline(0, color="k", lw=.6); ax.set_yticks(range(len(outs))); ax.set_yticklabels([lab[o] for o in outs]); ax.invert_yaxis()
    ax.set_title({"A": "A: importers in both years (with history)", "B": "B: balanced importers 2022-2025"}[samp], fontsize=10)
    ax.set_xlabel("Effect of +10pp exposure (SD of outcome change, 95% CI)", fontsize=8)
axes[0].legend(loc="upper left", fontsize=7)
fig.suptitle("Tariff exposure (2024-weighted, 2025 rates) and new-product scope of small importers\nSIC2 and size FE; controls: log 2024 imports, China share; SE by firm", fontsize=10)
plt.tight_layout(); fig.savefig(out, dpi=160)
pd.DataFrame(rows).to_csv(od/"fig_data_261102.csv", index=False)
