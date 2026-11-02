# ============================================================
# START DATE   : 2026-11-02
# CODE RUN DATE: 2026-11-02
# TIME TAKEN   : ~3 min
# DESCRIPTION  : Step 1. Select passages with Jev evidence probability (1 - P(not_evidence)) >= 0.10 on
#                tariff_china_path or tariff_broad_path, plus the TiRADE validation passages; fix the batch order
#                by seed before any coding. Stores IDs only.
# INPUT FILES  : Jev cache (code/20), passage_set_261030.parquet, scoring_passages.parquet
# OUTPUT FILES : analysis/49_statement_type_261102/01_selection_261102.csv (IDs, flags, batch id; no text)
# ============================================================
import importlib.util, json
from pathlib import Path
import numpy as np, pandas as pd
_s = importlib.util.spec_from_file_location("c", Path(__file__).with_name("common_261102.py"))
C = importlib.util.module_from_spec(_s); _s.loader.exec_module(C)
f = C.OUT / "01_selection_261102.csv"
assert not f.exists(), "selection exists; refusing to overwrite"
c = pd.read_parquet(C.JEV_CACHE)
for q in ["tariff_china_path", "tariff_broad_path"]:
    c[q + "_ev"] = c[q + "__probs"].map(lambda s: 1 - json.loads(s).get("not_evidence", 0))
    c[q + "_hi"] = c[q + "__probs"].map(lambda s: json.loads(s).get("higher", 0))
c["sel_china"] = c.tariff_china_path_ev >= 0.10
c["sel_broad"] = c.tariff_broad_path_ev >= 0.10
S = c[c.sel_china | c.sel_broad][["score_id", "sel_china", "sel_broad"]].copy()
P = pd.read_parquet(C.PSET)
val = set(P[P.in_A].score_id)
S["in_validation"] = S.score_id.isin(val)
extra = sorted(val - set(S.score_id))
S = pd.concat([S, pd.DataFrame({"score_id": extra, "sel_china": False, "sel_broad": False, "in_validation": True})], ignore_index=True)
S["extra_validation_only"] = S.score_id.isin(extra)
S = S.sample(frac=1, random_state=C.SEED).reset_index(drop=True)   # seeded order, fixed before coding
S["batch"] = np.arange(len(S)) // C.BATCH
C.OUT.mkdir(parents=True, exist_ok=True)
S.to_csv(f, index=False)
print(dict(china=int(S.sel_china.sum()), broad=int(S.sel_broad.sum()), union=int((S.sel_china | S.sel_broad).sum()),
           validation_passages=len(val), extra_validation_only=len(extra), total=len(S), batches=int(S.batch.max()) + 1))
