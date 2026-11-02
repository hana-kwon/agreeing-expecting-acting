# ============================================================
# START DATE   : 2026-11-02
# CODE RUN DATE: 2026-11-02
# TIME TAKEN   : ~10 min
# DESCRIPTION  : Does the measure fire when expected? (1) first TiRADE Threatened bin per country vs first-threat / new-information / threat measures;
#                (2) known pauses and decreases (data/31 actions, direction pause_or_suspension/decrease, targeted countries) vs coded direction "lower" passages.
#                Hit rate = share of events with the measure > 0 in the event bin or one bin either side, against the same rate in all other bins of those countries.
# INPUT FILES  : analysis/53_first_stage_261102/01_panel_261102.csv, analysis/49_statement_type_261102/03_codes_261102.csv, data/31 02_actions_B_extended.csv, 02_country_events_B_extended.csv
# OUTPUT FILES : analysis/53_first_stage_261102/03_validation_261102.csv
# ============================================================
import sys, re
from pathlib import Path
import numpy as np, pandas as pd
TOPIC = Path(__file__).resolve().parents[2]; OUT = TOPIC / "analysis/53_first_stage_261102"
if (OUT / "03_validation_261102.csv").exists(): sys.exit("output exists; refusing to overwrite")
P = pd.read_csv(OUT / "01_panel_261102.csv"); NC, NB = P.iso3.nunique(), P.k.nunique()
P = P.sort_values(["iso3", "k"]); isos = sorted(P.iso3.unique()); names = P.groupby("iso3").name.first().to_dict()
def mat(c): return P[c].fillna(0).values.reshape(NC, NB)
BIN0 = pd.Timestamp("2025-01-24")
def kbin(d): return int(np.ceil((d - BIN0).days / 14.0))
rows = []
def hit(measure_mat, events, label, mname):
    h = []; base = []
    for i, k in events:
        c = isos.index(i); lo, hi = max(k - 1, 0), min(k + 1, NB - 1)
        h.append(float((measure_mat[c, lo:hi + 1] > 0).any()))
        other = [j for j in range(NB) if not (lo <= j <= hi)]
        base.append(float((measure_mat[c, other] > 0).mean()))
    rows.append(dict(events=label, measure=mname, n_events=len(events), hit_rate_pm1bin=np.mean(h) if h else np.nan, same_country_other_bins_rate=np.mean(base) if base else np.nan))
# (1) first TiRADE Threatened bin per country (event = first bin where TiRADE threatened date falls in the bin)
thr = mat("O3_tir_threat__conc"); first = [(i, int(np.argmax(thr[c] > 0))) for c, i in enumerate(isos) if (thr[c] > 0).any()]
for m in ["c1_first_threat", "c2_first_threat_or_intent", "f2_n_new_vs_recent", "b2_n_threat", "b1_n_future", "a1_jev_score", "ref_tirade_threats4"]: hit(mat(m), first, "first TiRADE Threatened bin per country", m)
# (2) pauses / decreases: coded direction lower, mapped like 01
codes = pd.read_csv(TOPIC / "analysis/49_statement_type_261102/03_codes_261102.csv", parse_dates=["date"])
EU = {"european union", "eu", "europe", "european nations", "austria", "belgium", "bulgaria", "croatia", "cyprus", "czech republic", "czechia", "denmark", "estonia", "finland", "france", "germany", "greece", "hungary", "ireland", "italy", "latvia", "lithuania", "luxembourg", "malta", "netherlands", "poland", "portugal", "romania", "slovakia", "slovenia", "spain", "sweden"}
norm = lambda s: re.sub(r"\s+", " ", re.sub(r"\(.*?\)", "", s)).strip().lower().removeprefix("the ")
NM = {norm(n): i for i, n in names.items()}; NM.update({"china": "CHN", "uk": "GBR", "turkey": "TUR", "burma": "MMR", "philippines": "PHL", "bahamas": "BHS"}); NM.update({e: "EUN" for e in EU})
lw = np.zeros((NC, NB))
for d, t, dr in zip(codes.date, codes.target.fillna(""), codes.direction):
    if dr != "lower": continue
    for tg in {x.strip().lower() for x in t.split(";")}:
        i = "CHN" if tg == "china" else NM.get(norm(tg.split(":", 1)[1])) if tg.startswith("country:") else None
        k = kbin(d)
        if i and 0 <= k < NB: lw[isos.index(i), k] += 1
A = pd.read_csv(TOPIC / "data/31_tariff_calendar_fix_261101/02_actions_B_extended.csv"); E = pd.read_csv(TOPIC / "data/31_tariff_calendar_fix_261101/02_country_events_B_extended.csv")
ev = E[E.targeted.fillna(False).astype(bool)].merge(A[["action_id", "direction", "effective_date", "signing_date"]], on="action_id")
ev = ev[ev.direction.isin(["pause_or_suspension", "decrease"])].copy(); ev["d"] = pd.to_datetime(ev.signing_date)
pe = sorted({(r.iso3, kbin(r.d)) for r in ev.itertuples() if r.iso3 in isos and pd.notna(r.d) and 0 <= kbin(r.d) < NB})
hit(lw, pe, "known pauses/decreases (signing bin)", "g_n_lower_coded")
hit(mat("f1_n_future_higher"), pe, "known pauses/decreases (signing bin)", "f1_n_future_higher (should NOT systematically fire)")
pd.DataFrame(rows).to_csv(OUT / "03_validation_261102.csv", index=False); print(pd.DataFrame(rows).round(3).to_string())
