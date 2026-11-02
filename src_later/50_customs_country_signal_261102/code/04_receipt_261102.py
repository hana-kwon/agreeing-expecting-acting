# ============================================================
# START DATE   : 2026-11-02
# CODE RUN DATE: 2026-11-02
# TIME TAKEN   : ~3 min
# DESCRIPTION  : Write the receipt JSON (input and output SHA-256, code order, environment, what was not done) for analysis/50.
# INPUT FILES  : inputs and outputs listed below
# OUTPUT FILES : analysis/50_customs_country_signal_261102/receipt_customs_country_signal_261102.json
# ============================================================
import hashlib, json, subprocess, sys, platform
from pathlib import Path
HERE = Path(__file__).resolve(); TOPIC = HERE.parents[2]; OUT = TOPIC / "analysis/50_customs_country_signal_261102"
SH = HERE.parents[7] / "PRIVATE_DATA_ROOT"
rec = OUT / "receipt_customs_country_signal_261102.json"
if rec.exists(): sys.exit(f"output exists; refusing to overwrite: {rec}")
h = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
inputs = [SH / "SMB_tariff/intermediate_data/prep_reg_L2_update/smb_trade_duns_biweekly_260512.parquet",
          SH / "SMB_tariff/intermediate_data/prep_reg_L2_update/china_sentiment/smb_china_sentiment_tariff_for_L2_261008.csv",
          TOPIC / "data/31_tariff_calendar_fix_261101/02_panel_B_extended.csv", TOPIC / "data/31_tariff_calendar_fix_261101/02_increase_events_B_extended.csv",
          TOPIC / "data/16_tariff_schedule_261029/china_additional_duty_steps_261029.csv", TOPIC / "analysis/49_statement_type_261102/03_codes_261102.csv",
          Path.home() / "private_cache/customs40_261102/firm_attrs.parquet"]
code = sorted((TOPIC / "code/50_customs_country_signal_261102").glob("*"))
outs = sorted(p for p in OUT.iterdir() if p.suffix in (".csv", ".png", ".md") and p.name != rec.name)
rv = subprocess.run(["/usr/local/bin/Rscript", "-e", "cat(R.version.string, as.character(packageVersion('fixest')), as.character(packageVersion('data.table')))"], capture_output=True, text=True).stdout
json.dump(dict(task="customs country-specific signal re-specification (code/50)", date="2026-11-02", status="exploratory, not cross-verified",
               inputs={p.name: h(p) for p in inputs}, code_order=[c.name for c in code if c.is_file()], code_sha256={c.name: h(c) for c in code if c.is_file()},
               environment=dict(python=platform.python_version(), R_fixest_datatable=rv),
               outputs={p.name: h(p) for p in outs},
               comparison_to_baseline="code/41 and code/44 used one China signal x China dummy; not re-run here. Numbers are not comparable one-to-one (different FE, origins, signal).",
               not_done=["raw data rebuild", "weight (kg) outcome", "full-sample (100% firms) estimation: 10%/25%/3% seeded firm samples used for compute",
                         "199 permutation draws (60 run)", "product-scope-specific rates (232, USMCA)", "Other-origin and Panama regressions", "paid or external calls", "external sending"],
               firm_level_data="PRIVATE_CACHE only"), open(rec, "w"), indent=1)
print("receipt written")
