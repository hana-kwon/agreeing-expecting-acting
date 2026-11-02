# ============================================================
# START DATE   : 2026-11-02
# CODE RUN DATE: 2026-11-02
# TIME TAKEN   : ~15 min
# DESCRIPTION  : Step 3. Validate the typology against TiRADE labels on the 115 labeled passage-question pairs
#                (45 threatened/announced, 70 in force; same construction as code/20 07) and write a 60-passage blank
#                check sheet. Prior-method counts are recomputed from code/20 outputs.
# INPUT FILES  : analysis/20_tirade_validation_261030/{01_tirade_links,04_nli_scores,05b_claude_haiku_answers}_261030.csv,
#                03_codes_261102.csv, Jev cache, scoring_passages.parquet (text only for non-Factbase sheet rows)
# OUTPUT FILES : 04_validation_pairs_261102.csv (IDs+codes), 04_validation_summary_261102.csv, 04_check_sheet_60_261102.csv,
#                PRIVATE_CACHE (outside the project folder, all text)
# ============================================================
import importlib.util, json
from pathlib import Path
import numpy as np, pandas as pd
_s = importlib.util.spec_from_file_location("c", Path(__file__).with_name("common_261102.py"))
C = importlib.util.module_from_spec(_s); _s.loader.exec_module(C)
T20 = C.TOPIC / "analysis/20_tirade_validation_261030"
for n in ["04_validation_pairs_261102.csv", "04_validation_summary_261102.csv", "04_check_sheet_60_261102.csv"]:
    assert not (C.OUT / n).exists(), n
L = pd.read_csv(T20 / "01_tirade_links_261030.csv"); L = L[L.excluded_reason.isna()]
y = L[L.linked & L.jev_q.notna()]
P = y.groupby(["score_id", "jev_q"]).agg(jev_top=("jev_top", "first"), groups=("tirade_group", list)).reset_index()
def lab(gs):
    g = set(gs); f = bool(g & {"future", "mixed"}); i = bool(g & {"in_force", "mixed"})
    return "future_and_in_force" if f and i else "future_only" if f else "in_force_only" if i else "no_rate_only"
P["label"] = P.groups.map(lab); P = P[P.label != "no_rate_only"].copy()
P["ref"] = P.label.map(lambda s: "in_force_only" if s == "in_force_only" else "future")   # same collapse as code/20 07
N = pd.read_csv(T20 / "04_nli_scores_261030.csv")[["score_id", "jev_q", "nli_label"]]
H = pd.read_csv(T20 / "05b_claude_haiku_answers_261030.csv")[["score_id", "jev_q", "c_timing"]]
P = P.merge(N, on=["score_id", "jev_q"], how="left").merge(H, on=["score_id", "jev_q"], how="left")
K = pd.read_csv(C.OUT / "03_codes_261102.csv")
P = P.merge(K[["score_id", "statement_type", "all_types", "names_rate", "names_date", "direction", "target"]], on="score_id", how="left")
P["orig_jev"] = P.jev_top.isin(["higher", "lower", "same"])
P["nli"] = P.nli_label.eq("future_claim")
P["haiku_prev"] = P.c_timing.isin(["future", "both"])
P["new_a_future3"] = P.statement_type.isin(C.FUTURE3)
P["new_b_threat_intent"] = P.statement_type.isin(["threat_or_possible_future", "stated_intention_future"])
P["new_c_any_future3"] = P.all_types.fillna("").map(lambda s: bool(set(s.split(";")) & C.FUTURE3))
P["new_d_not_inforce_primary"] = ~P.statement_type.isin(["in_force_description", "past_action", "not_about_tariffs", "opinion_or_justification"]) & P.statement_type.notna()
nf, ni = int((P.ref == "future").sum()), int((P.ref == "in_force_only").sum())
rows = []
for m in ["orig_jev", "nli", "haiku_prev", "new_a_future3", "new_b_threat_intent", "new_c_any_future3"]:
    ok = P[m].notna() & (P[m].astype(object) != None)
    d = P[P[m].notna()]
    fut, inf = d[d.ref == "future"], d[d.ref == "in_force_only"]
    rows.append(dict(measure=m, n=len(d), n_future=len(fut), n_inforce=len(inf), future_detected=int(fut[m].astype(bool).sum()),
                     false_future_in_force=int(inf[m].astype(bool).sum()),
                     accuracy=round(float((d[m].astype(bool) == (d.ref == "future")).mean()), 3)))
Sm = pd.DataFrame(rows)
Sm.to_csv(C.OUT / "04_validation_summary_261102.csv", index=False)
# type distribution by TiRADE reference
pd.crosstab(P.ref, P.statement_type.fillna("uncoded")).to_csv(C.OUT / "04_validation_type_by_ref_261102.csv")
P[["score_id", "jev_q", "label", "ref", "jev_top", "statement_type", "all_types", "names_rate", "names_date", "direction", "target", "nli_label", "c_timing"]].to_csv(C.OUT / "04_validation_pairs_261102.csv", index=False)
print(Sm.to_string()); print(pd.crosstab(P.ref, P.statement_type.fillna("uncoded")))
# ---- 60-passage blank check sheet, seeded, stratified over Haiku primary types among selected non-validation passages
S = K[(K.sel_china | K.sel_broad) & K.statement_type.notna()].copy()
parts = []
rng = np.random.default_rng(C.SEED)
per = {"in_force_description": 10, "past_action": 6, "opinion_or_justification": 8, "threat_or_possible_future": 12,
       "stated_intention_future": 10, "decision_announcement": 10, "not_about_tariffs": 4}
for t, n in per.items():
    g = S[S.statement_type == t]
    parts.append(g.sample(min(n, len(g)), random_state=int(rng.integers(1e9))))
Sh = pd.concat(parts).sample(frac=1, random_state=C.SEED).reset_index(drop=True)
Sh.insert(0, "sheet_id", range(1, len(Sh) + 1))
txt = pd.read_parquet(C.SCORING)[["score_id", "passage"]]
full = Sh.merge(txt, on="score_id")
for c in ["human_statement_type", "human_names_rate", "human_names_date", "human_note"]:
    full[c] = ""
C.LOCAL.mkdir(parents=True, exist_ok=True)
full.drop(columns=["haiku_hidden"], errors="ignore").to_csv(C.LOCAL / "04_check_sheet_60_with_text.csv", index=False)
pub = full.copy()
pub["passage"] = np.where(pub.source_name == "factbase", "[Factbase: read by score_id in the corpus]", pub.passage)   # ★ no Factbase text in Dropbox
pub.drop(columns=["statement_type", "all_types", "target", "names_rate", "names_date", "direction"]).to_csv(C.OUT / "04_check_sheet_60_261102.csv", index=False)
Sh[["sheet_id", "score_id", "statement_type", "all_types", "target", "names_rate", "names_date", "direction"]].to_csv(C.LOCAL / "04_check_sheet_60_haiku_key.csv", index=False)
print("sheet", len(full), full.source_name.value_counts().to_dict())
