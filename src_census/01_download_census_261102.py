# ============================================================
# START DATE   : 2026-11-02
# CODE RUN DATE: 2026-11-02
# TIME TAKEN   : ~10 min
# DESCRIPTION  : Download U.S. Census monthly imports (country x HS2; HS4 world totals), 2023-01 to latest month.
# INPUT FILES  : ~/.config/census/api_key.txt (key never printed or stored)
# OUTPUT FILES : data/36_census_imports_261102/hs2ch_CC.csv (one HS2 chapter, all months), hs4w_YYYY-MM.csv, 00_download_log_261102.json
# ============================================================
import sys, json, time, datetime, pathlib, requests
from concurrent.futures import ThreadPoolExecutor
ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "36_census_imports_261102"
LOG = OUT / "00_download_log_261102.json"
if LOG.exists():
    sys.exit("refuse to overwrite existing download log")  # ★ IMPORTANT: no overwrite
KEY = open(pathlib.Path.home() / ".config/census/api_key.txt").read().strip()
URL = "https://api.census.gov/data/timeseries/intltrade/imports/hs"
GET = "CTY_CODE,CTY_NAME,I_COMMODITY,GEN_VAL_MO,CON_VAL_MO"
months = [f"{y}-{m:02d}" for y in (2023, 2024, 2025, 2026) for m in range(1, 13)]
months = [m for m in months if m <= "2026-12"]

def fetch(args):
    lvl, t = args
    f = OUT / (f"hs2ch_{t}.csv" if lvl == "HS2" else f"hs4w_{t}.csv")
    if f.exists() and f.stat().st_size > 0:
        return (lvl, t, "exists", None)
    for attempt in range(4):
        try:
            # ★ CORE LOGIC: full country x HS4 per month timed out in a test (too large for the API), so HS4 is pulled as
            # world totals only (CTY_CODE='-'); country x HS2 is pulled in full.
            # ★ CORE LOGIC (2nd change): a full-month country x HS2 request was far too slow (>2 min each, parallel stalls), so HS2 is pulled
            # one chapter at a time over the whole period (t = chapter code here).
            q = {"get": GET, "COMM_LVL": "HS4" if lvl == "HS4W" else lvl, "time": t if lvl == "HS4W" else f"from {months[0]} to {months[-1]}", "key": KEY}
            if lvl == "HS2":
                q["I_COMMODITY"] = t
            if lvl == "HS4W":
                q["CTY_CODE"] = "-"
            r = requests.get(URL, params=q, timeout=900)
            if r.status_code == 204:
                return (lvl, t, "no_data_204", 0)
            if r.status_code == 200:
                d = r.json()
                import csv
                with open(f, "w", newline="") as fh:
                    csv.writer(fh).writerows(d)
                return (lvl, t, "ok", len(d) - 1)
            err = r.status_code
        except Exception as e:
            err = type(e).__name__
        time.sleep(5 * (attempt + 1))
    return (lvl, t, f"failed_{err}", None)

jobs = [("HS4W", t) for t in months] + [("HS2", f"{c:02d}") for c in range(1, 100)]
with ThreadPoolExecutor(8) as ex:
    res = list(ex.map(fetch, jobs))
json.dump({"endpoint": URL, "query_get": GET, "COMM_LVL": ["HS2 (all countries)", "HS4 (world total only, CTY_CODE=-)"], "download_date": str(datetime.date.today()),
           "results": res}, open(LOG, "w"), indent=1)
print(sum(1 for r in res if r[2] == "ok"), "ok;", [r for r in res if r[2] not in ("ok", "exists")])
