# ============================================================
# START DATE   : 2026-11-02
# CODE RUN DATE: 2026-11-02
# TIME TAKEN   : ~10 min
# DESCRIPTION  : Assemble coded passages (IDs + codes only), validate field values, write the receipt-ready call log
#                summary and the type counts (Step 2 output; Step 3 uses it).
# INPUT FILES  : PRIVATE_CACHE{codes,calls}.jsonl, 01_selection_261102.csv, passage_occurrences.parquet
# OUTPUT FILES : analysis/49_statement_type_261102/03_codes_261102.csv (IDs and codes, no text),
#                03_call_summary_261102.json, 03_counts_by_type_261102.csv
# ============================================================
import importlib.util, json
from pathlib import Path
import pandas as pd
_s = importlib.util.spec_from_file_location("c", Path(__file__).with_name("common_261102.py"))
C = importlib.util.module_from_spec(_s); _s.loader.exec_module(C)
f = C.OUT / "03_codes_261102.csv"
assert not f.exists(), "exists; refusing to overwrite"
rows = [json.loads(l) for l in open(C.LOCAL / "codes.jsonl")]
D = pd.DataFrame(rows)
assert D.score_id.is_unique, D.score_id.duplicated().sum()
S = pd.read_csv(C.OUT / "01_selection_261102.csv")
M = S.merge(D.drop(columns="batch"), on="score_id", how="left")
meta = pd.read_parquet(C.SCORING)[["score_id", "date", "source_name"]]
M = M.merge(meta, on="score_id")
M["type_valid"] = M.statement_type.isin(C.TYPES)
M["dir_valid"] = M.direction.isin(C.DIRS)
M["all_types"] = M.all_types.map(lambda x: ";".join(x) if isinstance(x, list) else "")
M["target"] = M.target.map(lambda x: ";".join(map(str, x)) if isinstance(x, list) else str(x))
M["names_rate"] = M.names_rate.astype("boolean"); M["names_date"] = M.names_date.astype("boolean")
calls = [json.loads(l) for l in open(C.LOCAL / "calls.jsonl")]
ok = [c for c in calls if c["ok"]]; bad = [c for c in calls if not c["ok"]]
tok = {k: sum(c["usage"][C.MODEL].get(k, 0) for c in calls if c.get("usage")) for k in
       ["inputTokens", "outputTokens", "cacheReadInputTokens", "cacheCreationInputTokens", "thinkingTokens"]}
summ = dict(model=C.MODEL, selected=len(S), coded=int(M.statement_type.notna().sum()), uncoded=int(M.statement_type.isna().sum()),
            invalid_type=int((~M.type_valid & M.statement_type.notna()).sum()), invalid_direction=int((~M.dir_valid & M.statement_type.notna()).sum()),
            calls_total=len(calls), calls_ok=len(ok), calls_failed_logged=len(bad), failed_errors=[c["err"] for c in bad][:10],
            batches=int(S.batch.nunique()), cost_usd=round(sum(c.get("cost") or 0 for c in calls), 4), tokens=tok,
            models_seen=sorted({m for c in calls for m in c.get("models", [])}), seconds_sum=round(sum(c.get("seconds") or 0 for c in calls)))
keep = ["score_id", "date", "source_name", "sel_china", "sel_broad", "in_validation", "extra_validation_only", "batch", "statement_type",
        "all_types", "target", "names_rate", "names_date", "direction", "type_valid", "dir_valid"]
M[keep].to_csv(f, index=False)
json.dump(summ, open(C.OUT / "03_call_summary_261102.json", "x"), indent=1)
a = M[M.sel_china | M.sel_broad]
cnt = pd.concat([a.statement_type.value_counts().rename("all_selected"), a[a.sel_china].statement_type.value_counts().rename("china_sel"),
                 a[a.sel_broad].statement_type.value_counts().rename("broad_sel")], axis=1).fillna(0).astype(int)
cnt.to_csv(C.OUT / "03_counts_by_type_261102.csv")
print(json.dumps(summ, indent=1)); print(cnt)
