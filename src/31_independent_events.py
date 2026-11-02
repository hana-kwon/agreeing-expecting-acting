# ============================================================
# START DATE   : 2026-11-01
# CODE RUN DATE: 2026-11-01
# TIME TAKEN   : ~10 min
# DESCRIPTION  : How many independent events sit under each claim? From the panel: positive cells, spells (maximal runs of consecutive
#                positive weeks per unit), calendar episodes (spell start weeks merged when within 4 weeks of each other across units),
#                distinct effective dates and Federal Register documents behind the positive increase cells, and the share of positive
#                cells due to the largest single date. Original and corrected side by side.
# INPUT FILES  : <run>/30_calendar_<variant>/02_panel_<variant>.csv and 02_increase_events_<variant>.csv (written by step 30)
# OUTPUT FILES : <run>/31_independent_events.csv
# ============================================================
import sys
from pathlib import Path
import numpy as np
import pandas as pd
RUN = Path(sys.argv[1]).resolve()
OUTF = RUN / "31_independent_events.csv"
if OUTF.exists(): sys.exit("outputs exist")
GROUPS = {"CAN": ["CAN"], "MEX": ["MEX"], "CHN": ["CHN"], "EU": ["EUN"], "EAX": ["JPN", "KOR", "TWN", "VNM", "THA", "MYS", "IDN", "PHL", "SGP", "KHM", "LAO", "MMR", "BRN"]}
def spells(w):          # w: sorted array of weeks (datetime) with y=1 for one unit -> list of spell start dates
    w = sorted(w); st = []
    for i, x in enumerate(w):
        if i == 0 or (x - w[i - 1]).days > 7: st.append(x)
    return st
def episodes(starts, gap=28):
    s = sorted(starts); n = 0
    for i, x in enumerate(s):
        if i == 0 or (x - s[i - 1]).days > gap: n += 1
    return n
rows = []
for v in ["orig_repro", "A_listed", "B_extended", "C_inferred"]:
    P = pd.read_csv(RUN / f"30_calendar_{v}/02_panel_{v}.csv", parse_dates=["week"]); I = pd.read_csv(RUN / f"30_calendar_{v}/02_increase_events_{v}.csv", parse_dates=["eff"])
    I = I[(~I.short_lived) & (~I.get("cancelled", pd.Series(False, index=I.index)).astype(bool))]
    I = I[(I.eff >= "2025-01-24") & (I.eff <= "2026-10-23")]
    for claim, y, units in [("increase_96_countries", "y", None), ("reduction_pause_96_countries", "y_placebo", None)] + \
                            [(f"increase_group_{g}", "y", ls) for g, ls in GROUPS.items()] + [("increase_China_only", "y", ["CHN"]), ("reduction_pause_China_only", "y_placebo", ["CHN"])]:
        S = P if units is None else P[P.iso3.isin(units)]
        pos = S[S[y] == 1]
        sp = []; per = {}
        for iso, g in pos.groupby("iso3"):
            s = spells(list(g.week)); sp += s; per[iso] = len(s)
        r = dict(variant=v, claim=claim, positive_cells=len(pos), units_with_positive=pos.iso3.nunique(), spells=len(sp), episodes_4wk_gap=episodes(sp),
                 units_with_2plus_spells=sum(1 for x in per.values() if x >= 2))
        if y == "y":
            Ie = I if units is None else I[I.iso3.isin(S.iso3.unique())]
            r.update(distinct_eff_dates=Ie.eff.nunique(), distinct_documents=Ie.document_number.nunique(), country_date_events=len(Ie.drop_duplicates(["iso3", "eff"])),
                     top_date=str(Ie.groupby("eff").iso3.nunique().idxmax().date()) if len(Ie) else "", top_date_share_of_country_date_events=(Ie.drop_duplicates(["iso3", "eff"]).groupby("eff").size().max() / len(Ie.drop_duplicates(["iso3", "eff"]))) if len(Ie) else np.nan)
        rows.append(r)
pd.DataFrame(rows).round(3).to_csv(OUTF, index=False); print(pd.DataFrame(rows).to_string())
