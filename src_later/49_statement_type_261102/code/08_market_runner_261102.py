# ============================================================
# START DATE   : 2026-11-02
# CODE RUN DATE: 2026-11-02
# TIME TAKEN   : ~10 min writing, ~5 min per variant run
# DESCRIPTION  : Step 5d. Runs code/45 02_market_tests_261102.py with three text substitutions only: the statement table
#                (typed variant), the output folder, and the sample variants (only the full sample, to bound run time; the
#                April 2025 exclusion variant of the original is not rerun).
#                Usage: python 08_market_runner_261102.py <variant>
# INPUT FILES  : variants/evq_<variant>.parquet; data/45_market_response_261102 (prices); code/45 script text
# OUTPUT FILES : analysis/49_statement_type_261102/market_<variant>/02_*.csv (aggregates by trading day)
# ============================================================
import os, sys
from pathlib import Path
here = Path(__file__).resolve(); topic = here.parents[2]
v = sys.argv[1]
src_path = topic / "code/45_market_response_261102/02_market_tests_261102.py"
src = src_path.read_text()
BOTH = len(sys.argv) > 2 and sys.argv[2] == "both"   #   : also rerun the April-2025-exclusion sample after seeing the single-day driver in the decision variant
def sub(a, b):
    global src
    assert src.count(a) == 1, a
    src = src.replace(a, b)
sub('ev = pd.read_parquet(TS)', f'ev = pd.read_parquet(Path({os.environ.get("S49_VAR", str(Path.home() / "private_cache/trump_261102_49/variants"))!r}) / "evq_{v}.parquet")')
sub('OUT = TOPIC / "analysis/45_market_response_261102"', f'OUT = Path({os.environ.get("S49_OUT", str(topic / "analysis/49_statement_type_261102"))!r}) / "market{'x' if BOTH else ''}_{v}"; OUT.mkdir(parents=True, exist_ok=True)')
if not BOTH:
    sub('for variant, keep in [("all", keep_all), ("excl_2025-04", keep_excl)]:', 'for variant, keep in [("all", keep_all)]:')
sub('assert TS.exists() and CAL.exists() and CHN.exists()', 'assert CAL.exists() and CHN.exists()')
exec(compile(src, str(here), "exec"), {"__file__": str(here), "__name__": "__main__"})   # __file__ in code/49 keeps the parents[] path arithmetic
