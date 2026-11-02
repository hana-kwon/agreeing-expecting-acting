# ============================================================
# START DATE   : 2026-11-02
# CODE RUN DATE: 2026-11-02
# TIME TAKEN   : ~5 min
# DESCRIPTION  : Receipt for step 49: inputs and outputs with SHA-256, code order, environment, model, tokens, cost, comparison to
#                the earlier measures, and what was not done.
# INPUT FILES  : all of 49 outputs; 03_call_summary_261102.json; 04_validation_summary_261102.csv
# OUTPUT FILES : analysis/49_statement_type_261102/receipt_statement_type_261102.json
# ============================================================
import hashlib, importlib.util, json, platform, sys
from pathlib import Path
import pandas as pd
_s = importlib.util.spec_from_file_location("c", Path(__file__).with_name("common_261102.py"))
C = importlib.util.module_from_spec(_s); _s.loader.exec_module(C)
out = C.OUT / "receipt_statement_type_261102.json"
assert not out.exists(), "exists; refusing to overwrite"
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
import numpy, pandas, statsmodels
files = sorted(p for p in C.OUT.rglob("*") if p.is_file() and "placebo_draws" not in p.name)
summ = json.load(open(C.OUT / "03_call_summary_261102.json"))
val = pd.read_csv(C.OUT / "04_validation_summary_261102.csv").to_dict("records")
rec = dict(
    step="49_statement_type_261102", run_date="2026-11-02",
    model=summ["models_seen"], calls=summ["calls_total"], calls_ok=summ["calls_ok"], calls_failed_logged=summ["calls_failed_logged"],
    tokens=summ["tokens"], cost_usd=summ["cost_usd"], cost_cap_usd=12,
    extra_untracked_test_calls="2 (effort-low test, about USD 0.01; not in calls.jsonl)",
    inputs={str(p.relative_to(C.DB)): sha(p) for p in [C.SCORING, C.OCC, C.EVQ, C.CORPUS / "typesafe_choice_v3_results/results.jsonl",
                                                      C.CORPUS / "simple_time_series_v1/event_question_positions.parquet", C.JEV_CACHE.resolve()] if p.is_relative_to(C.DB)},
    local_inputs={"jev_cache": sha(C.JEV_CACHE), "raw_codes": sha(C.LOCAL / "codes.jsonl"), "raw_calls": sha(C.LOCAL / "calls.jsonl")},
    code_order=sorted(p.name for p in Path(__file__).parent.glob("[0-9]*.py")) + ["common_261102.py"],
    reused_unchanged_code=["code/39 02,03 (JEV path swapped; empty-placebo guard)", "code/45 02 (table path, output folder, one sample variant)",
                           "code/36 07 (signal block and spec list replaced; singular-matrix guard)", "code/27 03 logic re-implemented in 06 (not exec'd)"],
    environment=dict(python=sys.version.split()[0], pandas=pandas.__version__, numpy=numpy.__version__, statsmodels=statsmodels.__version__, platform=platform.platform(),
                     cli="claude-code 2.1.314 native binary, flags -p --tools '' --strict-mcp-config --no-session-persistence --output-format json"),
    validation_vs_earlier=val,
    not_done=["no human reading of the 60-passage sheet yet", "no raw-data rebuild", "no Grid or paid calls beyond the Haiku calls above",
              "no external sending", "Factbase text not stored in the project folder (IDs and codes only)", "respondent data read from the team folder, never saved",
              "April-2025-exclusion variant of code/45 not rerun", "no cross-agent verification"],
    outputs={str(p.relative_to(C.TOPIC)): sha(p) for p in files},
)
json.dump(rec, open(out, "x"), indent=1, ensure_ascii=False)
print("receipt written", len(files), "files")
