# ============================================================
# START DATE   : 2026-11-02
# CODE RUN DATE: 2026-11-02
# TIME TAKEN   : ~5 min
# DESCRIPTION  : Receipt (JSON) for step 39: input and output SHA-256, code, environment, comparison to baseline checks, what was not done.
# INPUT FILES  : all inputs and outputs listed in the receipt (hashed)
# OUTPUT FILES : analysis/39_.../receipt_retail_prices_statements_261102.json
# ============================================================
import json, hashlib, sys, platform, datetime
import statsmodels, matplotlib
from common_261102 import *

out, _ = get_out()
CODE = pathlib.Path(__file__).resolve().parent


def sha(p):
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


inputs = [DATA / f for f in FILES] + [DATA / "00_download_log_261102.json", JEV, CAL / "02_increase_events_B_extended.csv"]
rec = {
    "topic": "retail prices of China-origin goods vs leader tariff statements (words versus acts), exploratory",
    "receipt_date": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
    "status": "exploratory; not preregistered; spec fixed before estimation (00_spec_261102.md, 2026-11-02 02:58 EST); no cross-agent verification yet",
    "inputs": {str(p.name): {"path": str(p), "sha256": sha(p)} for p in inputs},
    "code_order": ["01_download_261102.py", "02_signal_calendar_261102.py", "03_p1_p2_261102.py", "04_figure_261102.py", "05_receipt_261102.py"],
    "code": {f: sha(CODE / f) for f in ["common_261102.py", "01_download_261102.py", "02_signal_calendar_261102.py", "03_p1_p2_261102.py", "04_figure_261102.py", "05_receipt_261102.py"]},
    "environment": {"python": sys.version.split()[0], "platform": platform.platform(), "pandas": pd.__version__, "numpy": np.__version__,
                    "statsmodels": statsmodels.__version__, "matplotlib": matplotlib.__version__, "venv": "~/claude_local_cache/trump_261101/venv"},
    "options": {"bootstrap_reps": 2000, "seed": 20261102, "placebo_shifts_weeks": "+-5..52", "hac_lags": "h+7"},
    "outputs": {p.name: sha(p) for p in sorted(out.iterdir()) if p.is_file() and not p.name.startswith("receipt_")},
    "comparison_to_baseline": {
        "recompute_check": "MAIN_h7_all 'all' and 'acts' differences recomputed independently with pandas: -0.0000715 and +0.0023555, equal to the code output to 6 decimals",
        "dev_vs_final_run": "02 and 03 output CSVs (8 files) from the development run and the final run have identical SHA-256 (same code logic and seed)",
        "downloaded_vs_scratchpad_copy": "both Pricing Lab csv files have the same SHA-256 as the earlier scratchpad download",
        "prior_paper": "Cavallo, Llamas and Vazquez (2026) report announcement-date responses; this analysis does not reproduce their numbers (different outcome: China-origin minus US-origin relative price from the public index)",
    },
    "not_done": ["cross-agent (Codex) verification", "rebuild of the price indices from product data", "paid calls", "external sending",
                 "product-level or retailer-level analysis", "tariff-rate controls", "causal identification claim"],
    "redistribution_note": "Pricing Lab csv files carry only 'Copyright HBS Pricing Lab'; copies stay in the project data folder, only the download script goes into any shared package.",
}
safe_write(json.dumps(rec, indent=2), out / "receipt_retail_prices_statements_261102.json")
print("receipt written")
