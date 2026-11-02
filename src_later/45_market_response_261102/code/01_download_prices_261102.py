# ============================================================
# START DATE   : 2026-11-02
# CODE RUN DATE: 2026-11-02
# TIME TAKEN   : ~3 min
# DESCRIPTION  : Downloads public daily price series for the market-response test (spec 00_spec_261102.md):
#                FRED SP500 and VIXCLS; ETFs SPY, XRT, FXI, MCHI. Stooq is tried first for the ETFs and is
#                logged as failed when it returns the JavaScript bot-check page (it is not circumvented);
#                Yahoo Finance chart API is the fallback. Records URL, download time, and SHA-256.
# INPUT FILES  : none (public URLs listed in sources_261102.json)
# OUTPUT FILES : data/45_market_response_261102/fred_SP500_261102.csv, fred_VIXCLS_261102.csv,
#                yahoo_{SPY,XRT,FXI,MCHI}_261102.csv, sources_261102.json
# ============================================================
import datetime as dt
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve()
TOPIC = HERE.parents[2]
OUT = TOPIC / "data/45_market_response_261102"
SRC = OUT / "sources_261102.json"
if SRC.exists():
    sys.exit("sources_261102.json exists; refusing to overwrite.")
OUT.mkdir(parents=True, exist_ok=True)

UA = {"User-Agent": "Mozilla/5.0"}
START, END = "2023-12-01", "2026-11-01"
log = []


def get(url, timeout=60):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read(), r.status


def sha(b):
    return hashlib.sha256(b).hexdigest()


def stamp():
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ---- FRED: S&P 500 index and VIX close
for sid in ["SP500", "VIXCLS"]:
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}&cosd={START}&coed={END}"
    b, st = get(url)
    f = OUT / f"fred_{sid}_261102.csv"
    f.write_bytes(b)
    d = pd.read_csv(f)
    log.append({"series": sid, "source": "FRED", "url": url, "downloaded": stamp(), "http": st,
                "rows": len(d), "first": str(d.iloc[0, 0]), "last": str(d.iloc[-1, 0]), "file": f.name,
                "sha256": sha(b), "used_for": "VIX close (VIXCLS); S&P 500 index check (SP500)"})

# ---- ETFs: Stooq first, Yahoo fallback
p1 = int(dt.datetime(2023, 12, 1, tzinfo=dt.timezone.utc).timestamp())
p2 = int(dt.datetime(2026, 11, 1, tzinfo=dt.timezone.utc).timestamp())
for tk in ["SPY", "XRT", "FXI", "MCHI"]:
    surl = f"https://stooq.com/q/d/l/?s={tk.lower()}.us&i=d"
    try:
        b, st = get(surl, 30)
        ok = b[:4] == b"Date" or b[:5] == b"Date,"
        log.append({"series": tk, "source": "Stooq", "url": surl, "downloaded": stamp(), "http": st,
                    "ok": bool(ok), "note": "CSV" if ok else "JavaScript bot-check HTML page, not CSV; not circumvented",
                    "sha256": sha(b)})
    except Exception as e:
        log.append({"series": tk, "source": "Stooq", "url": surl, "downloaded": stamp(), "ok": False, "note": repr(e)})
    yurl = (f"https://query1.finance.yahoo.com/v8/finance/chart/{tk}?period1={p1}&period2={p2}"
            "&interval=1d&events=div%2Csplit")
    b, st = get(yurl)
    j = json.loads(b)
    res = j["chart"]["result"][0]
    ts = pd.to_datetime(res["timestamp"], unit="s", utc=True).tz_convert("America/New_York").tz_localize(None).normalize()
    q = res["indicators"]["quote"][0]
    d = pd.DataFrame({"date": ts.strftime("%Y-%m-%d"), "open": q["open"], "high": q["high"], "low": q["low"],
                      "close": q["close"], "adjclose": res["indicators"]["adjclose"][0]["adjclose"],
                      "volume": q["volume"]})
    f = OUT / f"yahoo_{tk}_261102.csv"
    d.to_csv(f, index=False)
    log.append({"series": tk, "source": "Yahoo Finance chart API (fallback)", "url": yurl, "downloaded": stamp(),
                "http": st, "rows": len(d), "first": d.date.iloc[0], "last": d.date.iloc[-1],
                "n_missing_adjclose": int(d.adjclose.isna().sum()), "file": f.name, "sha256": sha(f.read_bytes()),
                "note": "adjclose used for returns; raw JSON not saved, CSV is the parsed copy"})

SRC.write_text(json.dumps({"download_date": stamp(), "window": [START, END], "log": log}, indent=1))
print(json.dumps(log, indent=1)[:3000])
