# ============================================================
# START DATE   : 2026-11-02
# CODE RUN DATE: 2026-11-02
# TIME TAKEN   : ~10 min writing, ~10 min per variant run
# DESCRIPTION  : Step 5c. Runs the unchanged code/39 scripts 02 (daily signal, escalation days) and 03 (P1/P2 tests: bootstrap
#                and HAC SE, 5-52 week shift placebo) with the Jev event table replaced by a typed variant.
#                Usage: python 07_retail_runner_261102.py <variant>
# INPUT FILES  : variants/evq_<variant>.parquet; data/39_retail_prices_statements_261102 (price copies); code/39 scripts
# OUTPUT FILES : analysis/49_statement_type_261102/retail_<variant>/ (same files as analysis/39; aggregates only)
# ============================================================
import sys
from pathlib import Path
here = Path(__file__).resolve(); topic = here.parents[2]
v = sys.argv[1]
sys.path.insert(0, str(topic / "code/39_retail_prices_statements_261102"))
import common_261102 as cm
import os
cm.JEV = Path(os.environ.get("S49_VAR", Path.home() / "private_cache/trump_261102_49/variants")) / f"evq_{v}.parquet"   # ★ CORE LOGIC: only the statement table changes
out = Path(os.environ.get("S49_OUT", topic / "analysis/49_statement_type_261102")) / f"retail_{v}"
d39 = topic / "code/39_retail_prices_statements_261102"
for s in ["02_signal_calendar_261102.py", "03_p1_p2_261102.py"]:
    sys.argv = [s, "--out", str(out)]
    src = (d39 / s).read_text()
    if s.startswith("03"):
        # ★ IMPORTANT: guard only. Thin typed variants can have no placebo shift with enough escalation days; the original code then fails on an empty table.
        a = "    pl = pd.DataFrame(rows)\n"
        assert src.count(a) == 1
        src = src.replace(a, '    pl = pd.DataFrame(rows, columns=["shift_days", "n_all", "n_act", "n_words", "n_other", "all", "acts", "words", "acts_minus_words"])\n')
    exec(compile(src, str(d39 / s), "exec"), {"__file__": str(d39 / s), "__name__": "__main__"})
