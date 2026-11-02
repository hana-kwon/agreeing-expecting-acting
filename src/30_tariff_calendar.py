# ============================================================
# START DATE   : 2026-11-01
# CODE RUN DATE: 2026-11-01
# TIME TAKEN   : under a minute (all four variants)
# DESCRIPTION  : Corrected tariff effective-date calendar and country x Friday panel, built from public Federal
#                Register documents and the actions extracted from them, together with TiRADE. Four variants:
#                  orig_repro  = original logic, no change
#                  A_listed    = the six known errors (FR 2025-23912, Mexico TRK-0309/10/11, CAN/MEX 3/4 pause and partial
#                                reductions, China postal 6/1 cancellation FR 2025-09297, Nicaragua 0->0, incomplete windows)
#                  B_extended  = A + same-kind errors found while checking (Canada TiRADE 8/7 rows, Switzerland 11/14 direction,
#                                UK 232 carve-out, Canada 2026-08-19 postponement)  (the variant used in the proposal)
#                  C_inferred  = B + removal of 2026-15181 10% rows that replace the expiring Section 122 10% (inference)
#                Usage: python src/30_tariff_calendar.py <run_folder>            (all four variants)
#                       python src/30_tariff_calendar.py <run_folder> <variant>  (one variant; used by the line above)
#                The panel written here holds the calendar outcomes only (y, y_5_8, y_9_12, y_incl_short, y_placebo, past4,
#                past12, pending). The score columns of the proposal panel (statement scores, mention counts, TiRADE
#                threat counts) are not part of this public step; scoring the panel needs restricted inputs (see README).
# INPUT FILES  : data/public/tariff_calendar/{01_fr_documents,03_actions,03_country_events,country_list}_261101.*
#                data/public/tirade-v1.5-rate-statements.csv
# OUTPUT FILES : <run>/30_calendar_<variant>/02_{increase_events,panel,actions,country_events,change_rows}_<variant>.csv
# ============================================================
import re, sys
from pathlib import Path
import numpy as np
import pandas as pd

import subprocess
HERE = Path(__file__).resolve(); ROOT = HERE.parents[1]
D6 = ROOT / "data/public/tariff_calendar"
VARIANTS = ("orig_repro", "A_listed", "B_extended", "C_inferred")
RUN = Path(sys.argv[1]).resolve()
if len(sys.argv) == 2:     # ★ IMPORTANT: one subprocess per variant, because the build below uses module-level state
    for v in VARIANTS:
        r = subprocess.run([sys.executable, str(HERE), str(RUN), v], text=True, capture_output=True)
        print(r.stdout, end="")
        if r.returncode:
            sys.exit(r.stderr[-3000:])
    sys.exit(0)
VARIANT = sys.argv[2]; assert VARIANT in VARIANTS
OUT = RUN / f"30_calendar_{VARIANT}"; OUT.mkdir(parents=True, exist_ok=True)
TIER = {"orig_repro": 0, "A_listed": 1, "B_extended": 2, "C_inferred": 3}[VARIANT]
if (OUT / f"02_panel_{VARIANT}.csv").exists(): sys.exit("outputs exist")
DATA_END = pd.Timestamp("2026-10-31")      # last day of the Federal Register search 
TIR = ROOT / "data/public/tirade-v1.5-rate-statements.csv"

A = pd.read_csv(D6 / "03_actions_261101.csv"); E = pd.read_csv(D6 / "03_country_events_261101.csv")
docs = pd.read_parquet(D6 / "01_fr_documents_261101.parquet").set_index("document_number")
tr = pd.read_csv(TIR, dtype=str).fillna("")
E["k"] = E.groupby(["action_id", "iso3"]).cumcount(); E["row_id"] = E.action_id + "|" + E.iso3 + "|" + E.k.astype(str)
url = lambda dn: docs.loc[dn].html_url
LOG = []
def log(cid, tier, row_id, iso, old_date, new_date, old, new, reason, src):
    if TIER >= tier:
        LOG.append(dict(change_id=cid, tier=tier, row_id=row_id, country=iso, date_old=old_date, date_new=new_date, old_value=old,
                        new_value=new, reason=reason, source=src))
A["cancelled_eff"] = pd.NaT
PAUSE_END = set()          # pause rows whose date is the END of the pause (the tariff then starts), not a stop

def drop_events(mask, cid, tier, reason, src):
    for r in E[mask].itertuples():
        a = A[A.action_id == r.action_id].iloc[0]
        log(cid, tier, r.row_id, r.iso3, a.effective_date, "", f"increase {r.prior_rate}->{r.new_rate}", "removed (not an increase)", reason, src)
    return E[~mask] if TIER >= tier else E

if TIER >= 1:
    # ---- 1. FR 2025-23912: initial 0%, increase scheduled for 2027-06-23 ----
    q = "The additional rate of duty under heading 9903.91.05 shall be increased on June 23, 2027."
    assert re.sub(r"\s+", " ", q) in re.sub(r"\s+", " ", docs.loc["2025-23912"].raw_text)
    i = A.index[A.action_id == "2025-23912_00"][0]; row = A.loc[i].copy()
    A.loc[i, "direction"] = "other"; A.loc[i, "direction_note"] = "initial 0% tariff on semiconductors from China (no rate change on 2025-12-23)"
    row["action_id"] = "2025-23912_FUT"; row["direction"] = "increase"; row["effective_date"] = "2027-06-23"; row["date_source"] = "quote_date"
    row["effective_date_quote"] = q; row["direction_note"] = "scheduled increase on 2027-06-23, rate to be announced at least 30 days earlier"
    A = pd.concat([A, row.to_frame().T], ignore_index=True)
    E = pd.concat([E, pd.DataFrame([dict(action_id="2025-23912_FUT", iso3="CHN", name="China", prior_rate=0.0, new_rate=np.nan, targeted=True,
                                        k=0, row_id="2025-23912_FUT|CHN|0")])], ignore_index=True)
    log("C01a", 1, "2025-23912_00|CHN|0", "CHN", "2025-12-23", "", "increase eff 2025-12-23 (new_rate 0)", "not an increase: initial 0%, direction=other",
        "FR 2025-23912 SUMMARY and III: initial tariff level 0 percent on 2025-12-23, increasing on June 23, 2027", url("2025-23912"))
    log("C01b", 1, "2025-23912_FUT|CHN|0", "CHN", "", "2027-06-23", "(absent)", "increase scheduled 2027-06-23 (known from 2025-12-29, counts as pending)",
        "Annex item (ii): rate of duty under heading 9903.91.05 shall be increased on June 23, 2027", url("2025-23912"))
    # ---- 2. Mexico TRK-0309/0310/0311 (EO 14326 text does not name Mexico) ----
    assert "Mexico" not in docs.loc["2025-15010"].raw_text
    E = drop_events((E.action_id == "2025-15010_TIR") & (E.iso3 == "MEX"), "C02", 1,
                    "TiRADE TRK-0309/0310/0311 (2025-07-31) say Mexico keeps paying the same 25% (90-day extension; steel 50% already in force): current rate kept, not an Aug 7 increase; EO 14326 text never names Mexico",
                    f"{url('2025-15010')} (Mexico absent from text); TiRADE row_ids TRK-0309, TRK-0310, TRK-0311")
    # ---- 3. CAN/MEX March 4: pause rows mark the END of the pause; product-level reductions are not the end of the full increase ----
    PAUSE_END |= {"2025-02478_00", "2025-02479_00"}
    for aid, iso in [("2025-02406_00", "CAN"), ("2025-02406_01", "CAN"), ("2025-02407_00", "MEX")]:
        log("C03a", 1, f"{aid}|{iso}|0", iso, "2025-03-04", "2025-03-04", "increase eff 2025-03-04 flagged short_lived (ended same day)",
            "increase eff 2025-03-04 in force (short_lived=False)",
            "EO 14197/14198 (FR 2025-02478, 2025-02479): duty 'paused and will not take effect until March 4, 2025': 3/4 is the start. Later 3/7 orders (FR 2025-03990, 2025-03991) cut potash and exclude USMCA goods only, not the full increase",
            f"{url('2025-02478')} ; {url('2025-02479')} ; {url('2025-03990')} ; {url('2025-03991')}")
    for aid, iso in [("2025-02478_00", "CAN"), ("2025-02479_00", "MEX")]:
        log("C03b", 1, f"{aid}|{iso}|0", iso, "2025-03-04", "2025-02-04", "pause event dated 2025-03-04 and used as a stop",
            "pause event dated 2025-02-04 (the date the duty would have started); not used as a stop", "pause row's 2025-03-04 is the pause end date", url(aid.split("_")[0]))
    # ---- 4. China postal June 1 steps cancelled by EO 14298 (FR 2025-09297) ----
    q = "notwithstanding the increase contemplated effective June 1, 2025"
    assert q in re.sub(r"\s+", " ", docs.loc["2025-09297"].raw_text)
    for aid in ["2025-06027_03", "2025-06378_03", "2025-06462_04"]:
        A.loc[A.action_id == aid, "cancelled_eff"] = pd.Timestamp("2025-05-14")
        for r in E[(E.action_id == aid) & (E.iso3 == "CHN")].itertuples():
            log("C04", 1, r.row_id, "CHN", "2025-06-01", "2025-05-14", f"increase eff 2025-06-01 (new_rate {r.new_rate})",
                "cancelled before taking effect (cancel date 2025-05-14); counted as pending only until then, never as an increase",
                "EO 14298 Sec. 3(b) retains the $100 per postal item 'notwithstanding the increase contemplated effective June 1, 2025' and (c)(ii) deletes the June 1 $200 step",
                url("2025-09297"))
    # ---- 5. Nicaragua 0 -> 0 ----
    A.loc[A.action_id == "2025-23892_00", "direction"] = "other"
    log("C05", 1, "2025-23892_00|NIC|0", "NIC", "2026-01-01", "", "increase eff 2026-01-01 (0 -> 0)", "not an increase: 0% on 2026-01-01 (steps to 10% on 2027-01-01 and 15% on 2028-01-01 kept as scheduled increases)",
        "FR 2025-23892 Annex A(b): 0% for 2026, 10% for 2027, 15% from 2028", url("2025-23892"))
if TIER >= 2:
    # ---- X1. Canada TiRADE rows dated 2025-08-07 (35% Canada took effect 2025-08-01 via FR 2025-14999) ----
    assert "Canada" not in docs.loc["2025-15010"].raw_text
    E = drop_events((E.action_id == "2025-15010_TIR") & (E.iso3 == "CAN"), "X01", 2,
                    "TiRADE TRK-0312/TRK-0320 are the Canada 35% rise 'effective August 1, 2025' (already in the calendar via FR 2025-14999, date 2025-08-01); TRK-0313 is the 40% transshipment rate. EO 14326 text never names Canada",
                    f"{url('2025-15010')} (Canada absent); {url('2025-14999')}; TiRADE row_ids TRK-0312, TRK-0313, TRK-0320")
    # ---- X2. Switzerland/Liechtenstein 2025-11-14: framework cut, not an increase ----
    A.loc[A.action_id == "2025-23316_00", "direction"] = "decrease"
    log("X02a", 2, "2025-23316_00|CHE|0", "CHE", "2025-11-14", "2025-11-14", "increase", "decrease (framework lowers the tariff to the higher of MFN or 15% all-in; TiRADE TRK-0692 had Switzerland at 39% from 2025-08-07)",
        "FR 2025-23316 SUMMARY: parties 'agreed to modification of certain tariff rates'; apply the higher of MFN or 15 percent. Prior 39% is from TiRADE (annex of EO 14326 is an image), so this row rests on FR text plus a secondary source",
        f"{url('2025-23316')}; TiRADE TRK-0692")
    E = drop_events((E.action_id == "2025-23316_00") & (E.iso3 == "LIE"), "X02b", 2,
                    "Liechtenstein was already at 15% (TiRADE TRK-0670) and the notice sets a 15% all-in cap: not an increase; not coded as a decrease either (uncertain)",
                    f"{url('2025-23316')}; TiRADE TRK-0670")
    # ---- X3. UK named only for a reduced-rate carve-out in the 2026-04-06 metals proclamation ----
    E = drop_events(E.action_id.isin(["2026-06960_01", "2026-06960_02"]) & (E.iso3 == "GBR"), "X03", 2,
                    "UK is named only for reduced rates (25% / 15% with UK-origin metal) inside a global action; the UK rate is not raised",
                    f"{url('2026-06960')} paragraphs 9 and 2(b)")
    # ---- X4. Canada 2026-08-19 duties postponed to 2026-08-22 by Proclamation 11056 ----
    PAUSE_END |= {"2026-17294_00"}
    for aid in ["2026-14991_00", "2026-14992_00", "2026-14997_00"]:
        log("X04a", 2, f"{aid}|CAN|0", "CAN", "2026-08-19", "2026-08-22", "increase eff 2026-08-19 flagged short_lived (3-day suspension)",
            "increase eff 2026-08-22 (postponed), short_lived=False",
            "Proclamation 11056 (FR 2026-17294): 'the effective date of the additional ad valorem duties imposed in Proclamations 11046, 11047, and 11048 shall be 12:01 a.m. ... on August 22, 2026'; signed 2026-08-18, before the 08-19 start",
            f"{url('2026-17294')}; {url(aid.split('_')[0])}")
    log("X04b", 2, "2026-17294_00|CAN|0", "CAN", "2026-08-22", "2026-08-19", "pause event dated 2026-08-22 and used as a stop", "pause event dated 2026-08-19; not used as a stop", "pause row date is the new start date", url("2026-17294"))
if TIER >= 3:
    E = drop_events(E.action_id.isin(["2026-15181_00", "2026-15181_01"]), "Y01", 3,
                    "10% rate on 17 economies (and 10% net of MFN for EU, Taiwan) replaces the Section 122 10% surcharge that expired 2026-07-24 (FR 2026-03824: 'through 12:01 a.m. eastern daylight time on July 24, 2026'): rate unchanged or lower. Inference; the 2026-15181 text states no prior rate",
                    f"{url('2026-15181')}; {url('2026-03824')}")

# ---- build events (same logic as original 04) ----
A["eff"] = pd.to_datetime(A.effective_date, errors="coerce")
A["known"] = pd.concat([pd.to_datetime(A.signing_date, errors="coerce"), pd.to_datetime(A.publication_date, errors="coerce")], axis=1).min(axis=1)
ok = A.date_source.isin(["quote_date", "quote_relative", "team_calendar", "TiRADE_annex"]) & A.eff.notna()
EV = E[E.targeted.fillna(False).astype(bool)].merge(A[ok], on="action_id")
EV["tiraide_annex"] = EV.date_source.eq("TiRADE_annex")
inc = EV[EV.direction == "increase"].copy()
pz = E[E.targeted.fillna(False).astype(bool)].merge(A[ok & A.direction.eq("pause_or_suspension")], on="action_id")
inc["postponed_from"] = pd.NaT
for i, r in inc.iterrows():
    # ★ CORE LOGIC: postponement. Original rule needs the same legal basis; pause rows flagged PAUSE_END match on country only
    q = pz[(pz.iso3 == r.iso3) & ((pz.legal_basis == r.legal_basis) | pz.action_id.isin(PAUSE_END)) & (pz.known <= r.eff) & (pz.eff > r.eff)]
    if len(q):
        inc.at[i, "postponed_from"] = r.eff; inc.at[i, "eff"] = q.eff.max()
end_ok = A.direction.eq("end") & ~A.direction_note.fillna("").str.contains("exclusion|exemption", case=False)
stop = A[ok & (A.direction.isin(["pause_or_suspension", "decrease"]) | end_ok)]
if TIER >= 1: stop = stop[~stop.action_id.isin(PAUSE_END)]        # a pause end is not a stop
stop_named = E[E.targeted.fillna(False).astype(bool)].merge(stop, on="action_id")[["iso3", "eff", "legal_basis", "product_scope"]]
excl_ids = E[E.get("excluded", pd.Series(False, index=E.index)).fillna(False).astype(bool)].groupby("iso3").action_id.apply(set).to_dict()
def compat(s_scope, r_scope):
    # ★ CORE LOGIC: a product-level reduction does not end an all-goods increase (listed error 3)
    return True if TIER < 1 else not (str(s_scope).startswith("named") and r_scope == "all goods")
def short_lived(r):
    s = stop_named[(stop_named.iso3 == r.iso3)]
    s = s[np.array([compat(x, r.product_scope) for x in s.product_scope], dtype=bool)]
    s2 = stop[stop.all_countries & (stop.legal_basis == r.legal_basis) & ~stop.action_id.isin(excl_ids.get(r.iso3, set()))]
    s2 = s2[np.array([compat(x, r.product_scope) for x in s2.product_scope], dtype=bool)]
    d = pd.concat([s.eff, s2.eff])
    return bool(((d >= r.eff) & (d < r.eff + pd.Timedelta(days=7))).any())
inc["short_lived"] = inc.apply(short_lived, axis=1) if len(inc) else []
inc["cancelled"] = inc.cancelled_eff.notna()
inc["pending_until"] = inc.cancelled_eff
# placebo date of PAUSE_END rows = the date the postponed duty would have started
pe_date = inc[inc.postponed_from.notna()].groupby("iso3").postponed_from.min().to_dict()
dec_all = EV[EV.direction.isin(["decrease", "pause_or_suspension", "exclusion", "end"])].copy()
if TIER >= 1:
    m = dec_all.action_id.isin(PAUSE_END)
    # 2025 pause rows -> 2025-02-04 (earliest postponed_from for that country before 2026), 2026 row -> 2026-08-19
    for idx in dec_all[m].index:
        iso, aid = dec_all.at[idx, "iso3"], dec_all.at[idx, "action_id"]
        cand = inc[(inc.iso3 == iso) & inc.postponed_from.notna() & ((inc.postponed_from.dt.year == 2026) == aid.startswith("2026"))].postponed_from
        if len(cand): dec_all.at[idx, "eff"] = cand.min()
inc.drop(columns=["k"], errors="ignore").to_csv(OUT / f"02_increase_events_{VARIANT}.csv", index=False)
A.to_csv(OUT / f"02_actions_{VARIANT}.csv", index=False); E.to_csv(OUT / f"02_country_events_{VARIANT}.csv", index=False)

# ---- panel ----
CL = pd.read_csv(D6 / "country_list_261101.csv")   # 96 countries x 88 Fridays (2025-01-24 to 2026-09-25) = the proposal panel
WEEKS = pd.date_range("2025-01-24", "2026-09-25", freq="W-FRI")
assert len(WEEKS) == 88
names = CL.set_index("iso3").name.to_dict()
rows = []
for iso in sorted(CL.iso3.unique()):
    inc_all = inc[inc.iso3 == iso]
    inc_c = inc_all[~inc_all.cancelled]
    dec_c = dec_all[dec_all.iso3 == iso]
    for t in WEEKS:
        def fut(df, lo, hi, keep_short=False):
            d = df if keep_short else df[~df.short_lived]
            return int(((d.eff > t + pd.Timedelta(days=lo)) & (d.eff <= t + pd.Timedelta(days=hi))).any())
        def win(v, hi):   # ★ CORE LOGIC (listed error 6): outcome window past the end of the data is missing, not zero
            return np.nan if (TIER >= 1 and t + pd.Timedelta(days=hi) > DATA_END) else v
        pend = ((inc_all.known <= t) & (inc_all.eff > t) & (inc_all.pending_until.isna() | (t < inc_all.pending_until))).any() if TIER >= 1 \
            else ((inc_all.known <= t) & (inc_all.eff > t)).any()
        rows.append(dict(iso3=iso, name=names[iso], week=t.date().isoformat(), y=fut(inc_c, 0, 28),
                         y_5_8=win(fut(inc_c, 28, 56), 56), y_9_12=win(fut(inc_c, 56, 84), 84),
                         y_incl_short=fut(inc_c, 0, 28, keep_short=True),
                         y_placebo=int(((dec_c.eff > t) & (dec_c.eff <= t + pd.Timedelta(days=28))).any()),
                         past4=int(((inc_c.eff > t - pd.Timedelta(days=28)) & (inc_c.eff <= t)).any()),
                         past12=int(((inc_c.eff > t - pd.Timedelta(days=84)) & (inc_c.eff <= t)).any()), pending=int(pend)))
C = pd.DataFrame(rows)
P = C[["iso3", "name", "week", "y", "y_5_8", "y_9_12", "y_incl_short", "y_placebo", "past4", "past12", "pending"]]
P.to_csv(OUT / f"02_panel_{VARIANT}.csv", index=False)
if TIER >= 1:
    n58 = int(P.y_5_8.isna().sum()); n912 = int(P.y_9_12.isna().sum())
    log("C06", 1, "panel y_5_8 / y_9_12", "ALL", "", "", "windows past 2026-10-31 coded 0", f"coded missing: y_5_8 {n58} cells, y_9_12 {n912} cells", "outcome window (t+28,t+56] and (t+56,t+84] runs past the end of the Federal Register data (2026-10-31); y 1-4 weeks is complete", "00_spec section 1; status.md Codex review")
pd.DataFrame(LOG).to_csv(OUT / f"02_change_rows_{VARIANT}.csv", index=False)
print(VARIANT, "inc events", len(inc), "| short_lived", int(inc.short_lived.sum()), "| cancelled", int(inc.cancelled.sum()), "| y cells", int(P.y.sum()),
      "| y_5_8 NaN", int(P.y_5_8.isna().sum()), "| y_9_12 NaN", int(P.y_9_12.isna().sum()), "| change rows", len(LOG))
