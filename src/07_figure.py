# ============================================================
# START DATE   : 2026-10-29
# CODE RUN DATE: 2026-10-31
# TIME TAKEN   : ~1 s
# DESCRIPTION  : Measurement step 7. Three-panel figure: official China duty, share 'higher' in the preceding 28
#                days, check against TiRADE labels.
# INPUT FILES  : data/public/china_additional_duty_steps.csv, <run>/04_window_shares.csv, <run>/05_tirade_benchmark.csv
# OUTPUT FILES : <run>/07_figure.png, <run>/07_figure_data.csv
# ============================================================
# Step 7. One figure, three panels on a shared time axis where relevant:
#   A. the common additional U.S. duty on goods of China (official schedule);
#   B. share of China-tariff evidence saying "higher" in the 28 days before each Monday, original vs revised measure;
#   C. TiRADE benchmark: share of passages the measure scores as a future claim, by TiRADE label.
# Output: <run>/07_figure.png and the plotted data <run>/07_figure_data.csv.
import csv
from datetime import date

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from common import PUBLIC as DATA, run_dir

out = run_dir()
BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df"
steps = sorted((date.fromisoformat(r["effective_date"]), float(r["rate_percent"]))
               for r in csv.DictReader(open(DATA / "china_additional_duty_steps.csv")))
wk = pd.read_csv(out / "04_window_shares.csv")
wk["week"] = pd.to_datetime(wk.origin)
bm = pd.read_csv(out / "05_tirade_benchmark.csv")

fig = plt.figure(figsize=(9, 7.2))
gs = fig.add_gridspec(3, 1, height_ratios=[1, 1.4, 0.9], hspace=0.55)
axA, axB, axC = fig.add_subplot(gs[0]), fig.add_subplot(gs[1]), fig.add_subplot(gs[2])
for ax in (axA, axB, axC):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)

xs = [pd.Timestamp(d) for d, _ in steps] + [pd.Timestamp("2026-10-21")]
ys = [v for _, v in steps] + [steps[-1][1]]
axA.step(xs, ys, where="post", color=INK, linewidth=2)
axA.set_ylabel("Percent", fontsize=8, color=MUTED)
axA.set_title("A. Common additional U.S. duty on goods of China (official schedule)", loc="left", fontsize=9, color=INK)

for ver, col, lab in [("original", BLUE, "Original measure"), ("revised", ORANGE, "Revised: future claims only")]:
    s = wk[wk.version == ver].sort_values("week")
    axB.plot(s.week, s.share_higher, color=col, linewidth=2, marker="o", markersize=4, label=lab)
axB.set_ylim(0, 1.05)
axB.set_ylabel("Share 'higher'", fontsize=8, color=MUTED)
axB.set_title("B. Share of China-tariff evidence saying 'higher' in the preceding 28 days (evidence weight < 3 omitted)",
              loc="left", fontsize=9, color=INK)
axB.legend(frameon=False, fontsize=8, loc="upper right")
for ax in (axA, axB):
    ax.set_xlim(pd.Timestamp("2025-01-01"), pd.Timestamp("2026-10-31"))
    for d, _ in steps[1:]:
        ax.axvline(pd.Timestamp(d), color=GRID, linewidth=0.8, zorder=0)

axC.barh(bm.tirade, bm.share_scored_future, color=[BLUE if t == "future claim" else ORANGE for t in bm.tirade], height=0.5)
for i, r in enumerate(bm.itertuples()):
    axC.text(r.share_scored_future + 0.01, i, f"{r.share_scored_future:.0%} of {r.passages} passages", va="center", fontsize=8, color=INK)
axC.set_xlim(0, 1.15)
axC.set_xlabel("Share of passages the measure scores as a future claim", fontsize=8, color=MUTED)
axC.set_title("C. Check against TiRADE labels on the same passages", loc="left", fontsize=9, color=INK)
fig.savefig(out / "07_figure.png", dpi=200, bbox_inches="tight")
pd.concat([wk.assign(panel="B"), bm.assign(panel="C")], ignore_index=True).to_csv(out / "07_figure_data.csv", index=False)
print("figure written")
