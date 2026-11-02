# ============================================================
# START DATE   : 2026-11-02
# CODE RUN DATE: 2026-11-02
# TIME TAKEN   : ~5 min
# DESCRIPTION  : Writes the receipt (SHA-256 of inputs, code, outputs; versions; what was not done) for step 36.
# INPUT FILES  : data/36_census_imports_261102/*, calendar files, 02_panel_B_extended.csv, code/36_*/*.py, analysis/36_*/*
# OUTPUT FILES : analysis/36_census_imports_261102/receipt_census_imports_261102.json
# ============================================================
import hashlib, json, sys, platform, pathlib
import numpy as np, pandas as pd, scipy, matplotlib
import common_261102 as cm
out = cm.ANALYSIS / "receipt_census_imports_261102.json"
assert not out.exists(), "refuse to overwrite"
h = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
home = str(pathlib.Path.home())
rel = lambda p: str(p).replace(home, "~")
inputs = {rel(p): h(p) for p in sorted(cm.DATA.glob("*"))}
for p in [cm.CAL / "corrected" / "02_increase_events_B_extended.csv", cm.CAL / "country_list_261101.csv", cm.PANEL31]:
    inputs[rel(p)] = h(p)
code = {p.name: h(p) for p in sorted(cm.ROOT.joinpath("code/36_census_imports_261102").glob("*.py"))}
outs = {p.name: h(p) for p in sorted(cm.ANALYSIS.glob("*")) if p.is_file() and p.name != out.name}
rows = {"hs2ch_rows": int(sum(len(pd.read_csv(p, dtype=str)) for p in cm.DATA.glob("hs2ch_*.csv"))),
        "hs4w_rows": int(sum(len(pd.read_csv(p, dtype=str)) for p in cm.DATA.glob("hs4w_*.csv")))}
json.dump(dict(date="2026-11-02", census_key="read from a local key file", inputs_sha256=inputs, code_order=list(code), code_sha256=code,
               versions=dict(python=sys.version.split()[0], pandas=pd.__version__, numpy=np.__version__, scipy=scipy.__version__, matplotlib=matplotlib.__version__, platform=platform.platform()),
               row_counts=rows, outputs_sha256=outs,
               comparison="see 05_results_261102.md section 6 (public vs licensed customs results)",
               not_done=["firm-level analysis", "country x HS4 (API timed out; HS4 is world total only)", "tariff-rate controls", "exact Section 232 annex HTS lists (HS4 approximations)",
                         "cross-check of Census totals against published Census release tables", "re-estimation of licensed customs results", "paid calls", "external sending",
                         "independent (Codex) cross-check: not done"],
               notes="Three monthly HS2 files from an aborted first download attempt were moved out of data/ (scratchpad) and are not used. Spec change log lists choices made after seeing results."), open(out, "x"), indent=1)
