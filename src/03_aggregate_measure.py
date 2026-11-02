# ============================================================
# START DATE   : 2026-10-29
# CODE RUN DATE: 2026-10-31
# TIME TAKEN   : ~1 s
# DESCRIPTION  : Measurement step 3. From passage scores to a dated measure (one passage per event, evidence
#                threshold 0.10, probability-weighted weekly shares; original and revised versions).
# INPUT FILES  : <run>/02_scores.csv, data/public/public_tariff_statements.csv
# OUTPUT FILES : <run>/03_event_measure.csv, <run>/03_weekly_shares.csv
# ============================================================
# Step 3. From passage scores to a dated measure.
#   Event rule: one passage per event (a post, or one presidential document) and question, the one most likely
#   to address the question; it enters if P(evidence) >= 0.10.
#   Original weights: P(evidence) x option probabilities normalized over higher/lower/same.
#   Revised weights: the same, multiplied by the future share P(future) / (P(future) + P(in force or past)), so
#   descriptions of rates already in force or of past actions carry little weight. (The share was chosen after a
#   check against TiRADE labels showed that P(future) alone marks most in-force statements as future.)
# Outputs: <run>/03_event_measure.csv and <run>/03_weekly_shares.csv (weeks with evidence weight < 3 are blank).
import pandas as pd

from common import PUBLIC as DATA, EVIDENCE_MIN, MIN_MASS, run_dir

out = run_dir()
st = pd.read_csv(DATA / "public_tariff_statements.csv")[["statement_id", "event_id", "date"]]
sc = pd.read_csv(out / "02_scores.csv").merge(st, on="statement_id")
sc = sc.sort_values(["question_id", "event_id", "p_evidence", "statement_id"], ascending=[True, True, False, True])
ev = sc.drop_duplicates(["question_id", "event_id"]).copy()
ev = ev[ev.p_evidence >= EVIDENCE_MIN].copy()
den = ev[["p_higher", "p_lower", "p_same"]].sum(axis=1).clip(lower=1e-9)
ev["future_share"] = ev.p_future / (ev.p_future + ev.p_in_force_or_past).clip(lower=1e-9)
for k in ["higher", "lower", "same"]:
    ev[f"w_{k}"] = ev.p_evidence * ev[f"p_{k}"] / den
    ev[f"wrev_{k}"] = ev[f"w_{k}"] * ev.future_share
ev["mass"] = ev.p_evidence
ev["mass_rev"] = ev.p_evidence * ev.future_share
ev.to_csv(out / "03_event_measure.csv", index=False)

ev["week"] = pd.to_datetime(ev.date).dt.to_period("W-SUN").dt.start_time.dt.date
rows = []
for (q, wk), g in ev.groupby(["question_id", "week"]):
    for ver, pre, m in [("original", "w_", "mass"), ("revised", "wrev_", "mass_rev")]:
        H, L, S, M = g[pre + "higher"].sum(), g[pre + "lower"].sum(), g[pre + "same"].sum(), g[m].sum()
        rows.append(dict(question_id=q, week=wk, version=ver, events=len(g), mass=round(M, 3),
                         share_higher=round(H / (H + L + S), 3) if M >= MIN_MASS else None,
                         share_lower=round(L / (H + L + S), 3) if M >= MIN_MASS else None))
pd.DataFrame(rows).to_csv(out / "03_weekly_shares.csv", index=False)
print(ev.groupby("question_id").size().to_dict(), "events with evidence")
