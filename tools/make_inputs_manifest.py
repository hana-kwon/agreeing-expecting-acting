# ============================================================
# START DATE   : 2026-10-31
# CODE RUN DATE: 2026-10-31
# TIME TAKEN   : ~1 s
# DESCRIPTION  : Writes data/inputs_manifest.json: SHA-256 and row count of every shareable input file.
# INPUT FILES  : data/public/** (including data/public/tariff_calendar/), data/cache/*, config/*.json
# OUTPUT FILES : data/inputs_manifest.json
# ============================================================
import hashlib
import json
from pathlib import Path

import pandas as pd


def n_rows(f):
    try:
        return len(pd.read_csv(f))
    except pd.errors.EmptyDataError:   # the orig_repro variant has no change rows
        return 0

ROOT = Path(__file__).resolve().parents[1]
files = {}
for f in sorted(list((ROOT / "config").glob("*.json")) + [p for d in ["public", "cache"] for p in (ROOT / "data" / d).rglob("*") if p.is_file()]):
    b = f.read_bytes()
    rows = (n_rows(f) if f.suffix == ".csv" else len(pd.read_parquet(f)) if f.suffix == ".parquet"
            else (b.count(b"\n") if f.suffix == ".jsonl" else None))
    files[str(f.relative_to(ROOT))] = {"sha256": hashlib.sha256(b).hexdigest(), "rows": rows, "bytes": len(b)}
(ROOT / "data/inputs_manifest.json").write_text(json.dumps({"files": files}, indent=1))
print(len(files), "files")
