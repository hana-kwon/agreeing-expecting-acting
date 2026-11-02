# ============================================================
# START DATE   : 2026-11-02
# CODE RUN DATE: 2026-11-02
# TIME TAKEN   : ~10 min
# DESCRIPTION  : Download public U.S. Census general imports 2024 (calendar-year total GEN_VAL_YR at 2024-12) by HS6 x country,
#                HS4-prefix requests for the 11 chapters whose HS2 request failed with HTTP 500 in 01. Public aggregate data (no firm data).
# INPUT FILES  : Census International Trade API imports/hs endpoint; key in ~/.config/census/api_key.txt (never printed)
# OUTPUT FILES : PRIVATE_CACHE, download_log_p4.json
# ============================================================
import sys, json, time, csv, datetime, pathlib, requests
from concurrent.futures import ThreadPoolExecutor
OUT = pathlib.Path.home() / "private_cache/innov52_261102/census_hs6"
OUT.mkdir(parents=True, exist_ok=True)
LOG = OUT / "download_log_p4.json"
if LOG.exists():
    sys.exit("refuse to overwrite existing download log")  # never overwrite
KEY = open(pathlib.Path.home() / ".config/census/api_key.txt").read().strip()
URL = "https://api.census.gov/data/timeseries/intltrade/imports/hs"

def fetch(ch):
    f = OUT / f"hs6_p4_{ch}.csv"
    if f.exists() and f.stat().st_size > 0:
        return (ch, "exists", None)
    err = None
    for a in range(4):
        try:
            # ★ CORE LOGIC: wildcard on I_COMMODITY with COMM_LVL=HS6 returns every HS6 line in the chapter by country;
            # GEN_VAL_YR at time=2024-12 is the calendar-2024 cumulative general-import value.
            r = requests.get(URL, params={"get": "CTY_CODE,CTY_NAME,I_COMMODITY,GEN_VAL_YR", "COMM_LVL": "HS6",
                                          "I_COMMODITY": f"{ch}*", "time": "2024-12", "key": KEY}, timeout=900)
            if r.status_code == 204:
                return (ch, "no_data_204", 0)
            if r.status_code == 200:
                d = r.json()
                with open(f, "w", newline="") as fh:
                    csv.writer(fh).writerows(d)
                return (ch, "ok", len(d) - 1)
            err = r.status_code
        except Exception as e:
            err = type(e).__name__
        time.sleep(5 * (a + 1))
    return (ch, f"failed_{err}", None)

with ThreadPoolExecutor(8) as ex:
    res = list(ex.map(fetch, [f"{ch}{k:02d}" for ch in ["29","39","40","61","62","73","84","85","87","90","94"] for k in range(1,100)]))
json.dump({"endpoint": URL, "get": "CTY_CODE,CTY_NAME,I_COMMODITY,GEN_VAL_YR", "COMM_LVL": "HS6", "time": "2024-12",
           "download_date": str(datetime.date.today()), "results": res}, open(LOG, "w"), indent=1)
print(sum(1 for r in res if r[1] == "ok"), "ok;", [r for r in res if r[1] not in ("ok", "exists")])
