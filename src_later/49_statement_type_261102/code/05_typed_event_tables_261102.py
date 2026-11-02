# ============================================================
# START DATE   : 2026-11-02
# CODE RUN DATE: 2026-11-02
# TIME TAKEN   : ~15 min
# DESCRIPTION  : Step 4. Typed versions of the Jev event-question table. The released rule picks, per source event and
#                question, the passage with the highest evidence probability (score_id breaks ties). Here the same rule is
#                applied after a type filter, so each variant has the same schema as event_question_option_probabilities.parquet
#                and the earlier scripts (27, 39, 45) run on it unchanged. 'base' (no filter) must reproduce the release.
#                Also writes the daily aggregate series (28-day and surprise) by variant.
# INPUT FILES  : 03_codes_261102.csv, Jev cache, passage_occurrences.parquet, event_question_positions.parquet,
#                event_question_option_probabilities.parquet (read only)
# OUTPUT FILES : PRIVATE_CACHE<variant>.parquet (event-level, no text);
#                analysis/49_statement_type_261102/05_daily_series_261102.csv, 05_base_reproduction_261102.json,
#                05_variant_counts_261102.csv
# ============================================================
import importlib.util, json
from pathlib import Path
import numpy as np, pandas as pd
_s = importlib.util.spec_from_file_location("c", Path(__file__).with_name("common_261102.py"))
C = importlib.util.module_from_spec(_s); _s.loader.exec_module(C)
VD = C.VARDIR; VD.mkdir(parents=True, exist_ok=True)
assert not (C.OUT / "05_daily_series_261102.csv").exists(), "exists; refusing to overwrite"
K = pd.read_csv(C.OUT / "03_codes_261102.csv")
K["names_rate"] = K.names_rate.fillna(False).astype(bool); K["names_date"] = K.names_date.fillna(False).astype(bool)
# ★ CORE LOGIC: variant definitions fixed in spec section 4 (pre-coding); spec_fut3 is a post-hoc addition (  ).
THR_INT = {"threat_or_possible_future", "stated_intention_future"}
VAR = {
    "base": lambda d: np.ones(len(d), bool),
    "thrint": lambda d: d.statement_type.isin(THR_INT).values,
    "decision": lambda d: (d.statement_type == "decision_announcement").values,
    "specific": lambda d: (d.names_rate | d.names_date).values,
    "inforce": lambda d: (d.statement_type == "in_force_description").values,
    "future3": lambda d: d.statement_type.isin(C.FUTURE3).values,
    "specific_future3": lambda d: ((d.names_rate | d.names_date) & d.statement_type.isin(C.FUTURE3)).values,   #
}
J = pd.read_parquet(C.JEV_CACHE)
Q = {"tariff_china_path": "china", "tariff_broad_path": "broad"}
probs = {}
for q in Q:
    for sid, s in zip(J.score_id, J[q + "__probs"]):
        probs[(q, sid)] = json.loads(s)
O = pd.read_parquet(C.OCC, columns=["score_id", "source_family", "source_name", "event_id"])
O["event_key"] = O.source_family + "::" + O.source_name + "::" + O.event_id
O = O.drop_duplicates(["score_id", "event_key"])
Pos = pd.read_parquet(C.CORPUS / "simple_time_series_v1/event_question_positions.parquet",
                      columns=["event_question_id", "event_key", "event_date", "question_id", "selected_score_id", "raw_evidence_probability"])
Pos = Pos[Pos.question_id.isin(Q)]
EV = pd.read_parquet(C.EVQ)
EV = EV[EV.question_id.isin(Q)]
opt_meta = EV.drop_duplicates(["question_id", "option"])[["question_id", "option", "is_absence_option"]]
rows_cnt, series = [], []
for name, fn in VAR.items():
    keep_ids = set(K.score_id[fn(K) & K.statement_type.notna()])
    recs = []
    for q in Q:
        pq = Pos[Pos.question_id == q]
        cand = O[O.score_id.isin(keep_ids) & O.event_key.isin(set(pq.event_key))]
        cand = cand.assign(evp=[1 - probs[(q, s)].get("not_evidence", 0) for s in cand.score_id])
        cand = cand[cand.evp >= 0.10 - 1e-9]
        cand = cand.sort_values(["event_key", "evp", "score_id"], ascending=[True, False, True])   # max evidence; smaller score_id breaks ties
        best = cand.drop_duplicates("event_key")
        b = best.merge(pq[["event_key", "event_question_id", "event_date", "selected_score_id", "raw_evidence_probability"]], on="event_key")
        if name == "base":
            full = pq.merge(best[["event_key", "score_id"]], on="event_key", how="left")
            same = (full.score_id == full.selected_score_id)
            ev_equal = np.isclose(b.evp, b.raw_evidence_probability, atol=0.011)
            json.dump(dict(question=q, release_events=len(pq), base_events=len(b), same_selected_passage=int(same.sum()),
                           share_same=float(same.mean()), evidence_prob_matches_release_within_0p011=float(ev_equal.mean())),
                      open(C.OUT / f"05_base_reproduction_{Q[q]}_261102.json", "x"))
        for r in b.itertuples():
            for opt, p in probs[(q, r.score_id)].items():
                recs.append((r.event_question_id, pd.Timestamp(r.event_date), q, opt, p))
    V = pd.DataFrame(recs, columns=["event_question_id", "event_date", "question_id", "option", "raw_option_probability"])
    V = V.merge(opt_meta, on=["question_id", "option"], how="left")
    V.to_parquet(VD / f"evq_{name}.parquet", index=False)
    for q in Q:
        x = V[V.question_id == q]
        w = x.pivot_table(index=["event_question_id", "event_date"], columns="option", values="raw_option_probability", aggfunc="sum").fillna(0.0).reset_index()
        w["ev"] = 1.0 - w["not_evidence"]; w["num"] = w["higher"] - w["lower"]
        g = w.groupby("event_date")[["num", "ev"]].sum()
        g["n_events"] = w.groupby("event_date").size()
        g = g.reset_index().rename(columns={"event_date": "date"}); g["variant"] = name; g["question"] = Q[q]
        series.append(g)
        rows_cnt.append(dict(variant=name, question=Q[q], events=len(w), days=len(g), sum_ev=round(float(w.ev.sum()), 2)))
S = pd.concat(series)
# ★ CORE LOGIC: 28-day signal = sum(num)/sum(ev) over [t-28, t-1] (same window as code/27 03 signal()); prior-90 = same ratio over [t-118, t-29];
# surprise = 28-day minus prior-90 ratio (spec section 4 clarification, fixed before running).
out = []
for (v, qn), g in S.groupby(["variant", "question"]):
    g = g.set_index("date").reindex(pd.date_range("2024-01-01", "2026-10-02"))
    num, ev = g.num.fillna(0), g.ev.fillna(0)
    n28, e28 = num.shift(1).rolling(28, min_periods=1).sum(), ev.shift(1).rolling(28, min_periods=1).sum()
    n90, e90 = num.shift(29).rolling(90, min_periods=1).sum(), ev.shift(29).rolling(90, min_periods=1).sum()
    r = pd.DataFrame({"date": g.index, "variant": v, "question": qn, "num": num.values, "ev": ev.values, "n_events": g.n_events.fillna(0).values,
                      "daily_signal": np.where(ev > 0, num / ev.where(ev > 0), np.nan),
                      "s28": np.where(e28 > 0, n28 / e28.where(e28 > 0), np.nan), "s90prior": np.where(e90 > 0, n90 / e90.where(e90 > 0), np.nan)})
    r["surprise"] = r.s28 - r.s90prior
    out.append(r)
pd.concat(out).to_csv(C.OUT / "05_daily_series_261102.csv", index=False)
pd.DataFrame(rows_cnt).to_csv(C.OUT / "05_variant_counts_261102.csv", index=False)
print(pd.DataFrame(rows_cnt).to_string())
