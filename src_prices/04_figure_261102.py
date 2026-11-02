# ============================================================
# START DATE   : 2026-11-02
# CODE RUN DATE: 2026-11-02
# TIME TAKEN   : ~10 min
# DESCRIPTION  : One figure: (a) relative price of China-origin vs US-origin goods with escalation days, Cavallo et al. announcement dates
#                and China effective dates; (b) mean cumulative change in event time after acts-type, words-only and other days.
# INPUT FILES  : data/39_.../*.csv, analysis/39_.../02_daily_signal_china.csv, 02_china_events.csv, 03_event_paths.csv, 03_escalation_days.csv
# OUTPUT FILES : analysis/39_.../04_fig_relative_price_statements_261102.png and 04_fig_data_261102.csv
# ============================================================
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from common_261102 import *

out, _ = get_out()
m = load_prices()
esc = pd.read_csv(out / "03_escalation_days.csv", parse_dates=["date"])
paths = pd.read_csv(out / "03_event_paths.csv")
ce = pd.read_csv(out / "02_china_events.csv", parse_dates=["eff"])
ce = ce[ce.eff <= m.index.max()]
STARRED = pd.to_datetime(["2025-03-04", "2025-04-02", "2025-05-12", "2025-07-07", "2025-07-09", "2025-09-29"])
BLUE, ORANGE, GREY, INK, MUTE = "#2a78d6", "#eb6834", "#8a8984", "#0b0b0b", "#52514e"   # dataviz default palette slots 1-2, neutral
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.edgecolor": MUTE, "axes.labelcolor": INK, "xtick.color": MUTE, "ytick.color": MUTE})
fig, ax = plt.subplots(1, 2, figsize=(13.5, 5.6), gridspec_kw={"width_ratios": [1.45, 1]})
a = ax[0]
y = m.y * 100
a.plot(y.index, y, color=INK, lw=1.4)
for i, d in enumerate(STARRED):
    a.axvline(d, color=GREY, ls="--", lw=1, zorder=0, label="Announcement dates named by Cavallo et al." if i == 0 else None)
for i, e in enumerate(ce.eff):
    a.axvline(e, color=INK, lw=0.9, alpha=0.6, zorder=0, label="China increase takes effect (calendar B_extended)" if i == 0 else None)
ybase = y.min() - 0.25
ac, wd = esc[esc.label == "acts"], esc[esc.label == "words"]
a.scatter(ac.date, [ybase] * len(ac), s=46, color=BLUE, edgecolor="#fcfcfb", lw=1, zorder=3, label=f"Escalation day followed by an increase within 30 days (n={len(ac)})")
a.scatter(wd.date, [ybase] * len(wd), s=46, marker="s", facecolor="none", edgecolor=ORANGE, lw=1.6, zorder=3, label=f"Escalation day, words only (n={len(wd)})")
a.set_ylim(ybase - 0.2, y.max() + 0.35)
a.set_ylabel("ln(China-origin index) minus ln(US-origin index), x100")
a.set_title("(a) Relative retail price and escalation days", loc="left", fontsize=11, color=INK)
a.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m")); a.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
a.grid(axis="y", color="#e6e5e1", lw=0.6); a.spines[["top", "right"]].set_visible(False)
a.legend(loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=2, fontsize=7.8, frameon=False)
b = ax[1]
for g, c, ls, lab in [("acts", BLUE, "-", "followed by an increase"), ("words", ORANGE, "-", "words only"), ("other", GREY, "-", "all other days")]:
    p = paths[paths.group == g]
    n = int(p.n.max())
    b.plot(p.j, p["mean"] * 100, color=c, lw=2.0, label=f"{lab} (n={n})")
b.axvline(0, color=MUTE, lw=0.8, ls=":"); b.axhline(0, color=MUTE, lw=0.6)
b.set_xlabel("Days since statement day (0 = statement day; base = day -1)")
b.set_ylabel("Mean cumulative change in relative price, x100")
b.set_title("(b) Event-time path after escalation days", loc="left", fontsize=11, color=INK)
b.grid(axis="y", color="#e6e5e1", lw=0.6); b.spines[["top", "right"]].set_visible(False)
b.legend(loc="upper left", fontsize=8, frameon=False)
fig.suptitle("Chinese-origin vs US-origin retail prices around leader statements on China tariffs, 2024-10 to 2026-04", x=0.01, ha="left", fontsize=12, color=INK)
fig.text(0.01, 0.005, "Escalation day = top decile of the daily net China-path signal (Jev coding) among evidence days, 2024-10-02 to 2026-03-31. Prices: HBS Pricing Lab "
         "Tariff Tracker (index_usa equals the domestic index). Exploratory; few independent events, overlapping windows.", fontsize=7.2, color=MUTE)
fig.tight_layout(rect=(0, 0.035, 1, 0.95))
safe_write(lambda p: fig.savefig(p, dpi=170), out / "04_fig_relative_price_statements_261102.png")
safe_write(pd.concat([y.rename("y_x100").reset_index().assign(panel="a_y"),
                      paths.assign(panel="b_paths")], ignore_index=True), out / "04_fig_data_261102.csv")
