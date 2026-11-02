# ============================================================
# START DATE   : 2026-11-02
# CODE RUN DATE: 2026-11-02
# TIME TAKEN   : ~5 min
# DESCRIPTION  : Write receipt JSON: inputs and outputs with SHA-256, code order, package versions, model IDs, tokens, cost, what was not done.
# INPUT FILES  : data/35.../*, analysis/35.../*, code/35.../*
# OUTPUT FILES : analysis/35.../receipt_sec_tariff_expectations_261102.json
# ============================================================
import hashlib, json, sys, platform
from pathlib import Path
import pandas as pd, numpy as np, statsmodels, pyarrow
ROOT = Path(__file__).resolve().parents[2]
D, A, C = ROOT / "data/35_sec_tariff_expectations_261102", ROOT / "analysis/35_sec_tariff_expectations_261102", ROOT / "code/35_sec_tariff_expectations_261102"
OUT = A / "receipt_sec_tariff_expectations_261102.json"
if OUT.exists(): sys.exit("exists; refusing")
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
calls = pd.read_csv(D / "04_llm_calls_261102.csv"); trial = pd.read_csv(D / "04_trial_calls_261102.csv")
ext = "data/jev/event_question_option_probabilities.parquet (repository copy)"
rec = dict(
    inputs_external=[dict(path="Wikipedia List of S&P 500 companies (retrieval date in 01_sp500_raw_261102.csv)", sha256=sha(D / "01_sp500_raw_261102.csv")),
                     dict(path="SEC EDGAR filings (URLs, per-filing raw SHA-256, download date in 02_filings_manifest_261102.csv)", sha256=sha(D / "02_filings_manifest_261102.csv")),
                     dict(path="data/jev/event_question_option_probabilities.parquet", sha256=hashlib.sha256(Path(ext).read_bytes()).hexdigest())],
    code_order=sorted(p.name for p in C.glob("*.py")),
    environment=dict(python=platform.python_version(), pandas=pd.__version__, numpy=np.__version__, statsmodels=statsmodels.__version__, pyarrow=pyarrow.__version__),
    llm=dict(model_requested="claude-haiku-5-5", models_reported=sorted(set(",".join(calls.models.astype(str)).split(","))), cli="claude-code 2.1.314 native binary, isolated (-p, --tools '', --strict-mcp-config, CLAUDE_CODE_DISABLE_CLAUDE_MDS=1)",
             calls=int(len(calls)), failed_calls=int((~calls.ok).sum()), retried_calls=int((calls.attempts > 1).sum()), input_tokens=int(calls.in_tok.sum()), output_tokens=int(calls.out_tok.sum()),
             cost_usd_full_run=float(calls.cost_usd.sum()), cost_usd_trial=float(trial.cost_usd.sum()), cost_usd_total=float(calls.cost_usd.sum() + trial.cost_usd.sum())),
    outputs={p.name: sha(p) for p in sorted(list(A.glob("*.csv")) + list(A.glob("*.png")) + list(A.glob("*.md"))) },
    data_outputs={p.name: sha(p) for p in sorted(D.glob("0[1-4]_*"))},
    not_done=["Raw filing HTML not retained (local disk full); only tariff paragraphs and per-filing SHA-256 of the raw download are kept",
              "No human validation of codes (check sheet blank); agent spot-check of 20 passages is not human validation",
              "Exhibits, 8-K and earnings-call transcripts not used", "No causal identification", "No comparison with the SMB survey units",
              "No external model cross-check (Codex unavailable)"])
json.dump(rec, open(OUT, "w"), indent=1)
print("receipt written")
