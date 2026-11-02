# ============================================================
# START DATE   : 2026-11-02
# CODE RUN DATE: 2026-11-02
# TIME TAKEN   : ~3 min
# DESCRIPTION  : Receipt (inputs, code, outputs with SHA-256, environment, what was not done) for analysis/53.
# INPUT FILES  : see list below
# OUTPUT FILES : analysis/53_first_stage_261102/receipt_first_stage_261102.json
# ============================================================
import hashlib, json, platform, subprocess, sys
from pathlib import Path
TOPIC = Path(__file__).resolve().parents[2]; A = TOPIC / "analysis/53_first_stage_261102"; C = TOPIC / "code/53_first_stage_261102"
o = A / "receipt_first_stage_261102.json"
if o.exists(): sys.exit("output exists; refusing to overwrite")
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
H = Path.home()
inputs = [TOPIC / "data/31_tariff_calendar_fix_261101/02_panel_B_extended.csv", TOPIC / "data/31_tariff_calendar_fix_261101/02_increase_events_B_extended.csv",
          TOPIC / "analysis/49_statement_type_261102/03_codes_261102.csv", H / "research_local/orgsci_ai_frontiers/topic6_digital_twin/ideology_share_copies_261031/results.jsonl",
          H / "research_local/orgsci_repos/agreeing_expecting_acting/data/public/tirade-v1.5-rate-statements.csv", TOPIC / "data/45_market_response_261102/yahoo_SPY_261102.csv", TOPIC / "data/45_market_response_261102/yahoo_FXI_261102.csv"]
outs = sorted(p for p in A.rglob("*") if p.is_file() and p.name != o.name and "superseded" not in str(p))
rv = subprocess.run(["/usr/local/bin/Rscript", "-e", "cat(R.version.string, as.character(packageVersion('fixest')))"], capture_output=True, text=True).stdout.strip().splitlines()[-1]
import pandas, numpy
json.dump(dict(date="2026-11-02", spec="analysis/53_first_stage_261102/00_spec_261102.md (fixed 06:01 EST; change log inside)",
               inputs={str(p): sha(p) for p in inputs}, code_order=[p.name for p in sorted(C.iterdir()) if p.suffix in (".py", ".R")],
               environment=dict(python=platform.python_version(), pandas=pandas.__version__, numpy=numpy.__version__, R=rv),
               outputs={str(p.relative_to(A)): sha(p) for p in outs},
               comparison_to_baseline="AUC machinery reproduces the logic of analysis/31 (week-block bootstrap) on a two-week panel; not a numerical replication of analysis/31 (different bins, measures).",
               not_done=["NYT attention counts (about 2,100 calls at 5 per minute; skipped)", "human reading of codes", "cross verification", "intraday market prices", "paid or external calls", "external sending"],
               firm_level_data="PRIVATE_CACHE and customs53_261102 only (not in the repo)"), open(o, "x"), indent=1)
print("receipt written")
