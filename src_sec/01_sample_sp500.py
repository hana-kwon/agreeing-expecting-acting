# ============================================================
# START DATE   : 2026-11-02
# CODE RUN DATE: 2026-11-02
# TIME TAKEN   : ~10 min
# DESCRIPTION  : Fetch Wikipedia S&P 500 constituents table, keep 4 GICS sectors, map ticker -> CIK with SEC company_tickers.json.
# INPUT FILES  : https://en.wikipedia.org/wiki/List_of_S%26P_500_companies ; https://www.sec.gov/files/company_tickers.json (network)
# OUTPUT FILES : data/35_sec_tariff_expectations_261102/01_sp500_raw_261102.csv ; 01_sample_firms_261102.csv
# ============================================================
import os
import io, sys, json, datetime, urllib.request
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "35_sec_tariff_expectations_261102"
OUT_RAW, OUT = DATA / "01_sp500_raw_261102.csv", DATA / "01_sample_firms_261102.csv"
SECTORS = ["Consumer Discretionary", "Consumer Staples", "Industrials", "Information Technology"]
if OUT.exists() or OUT_RAW.exists(): sys.exit("output exists; refusing to overwrite")

def fetch(url, ua):
    return urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": ua}), timeout=60).read()

html = fetch("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", "Mozilla/5.0 (academic research script)").decode("utf8")
t = pd.read_html(io.StringIO(html))[0]       # first table = current constituents
t["retrieved"] = datetime.date.today().isoformat()
t.to_csv(OUT_RAW, index=False)
t = t.rename(columns={"Symbol": "ticker_wiki", "Security": "name", "GICS Sector": "sector", "CIK": "cik_wiki"})
t = t[t.sector.isin(SECTORS)].copy()
# SEC request: the approved User-Agent is used for SEC only
tk = json.loads(fetch("https://www.sec.gov/files/company_tickers.json", os.environ["SEC_USER_AGENT"]))
cik_of = {v["ticker"].upper(): (int(v["cik_str"]), v["title"]) for v in tk.values()}
t["ticker"] = t.ticker_wiki.str.replace(".", "-", regex=False).str.upper()
# ★ CORE LOGIC: primary map = company_tickers.json; Wikipedia CIK only as fallback and as a cross-check
t["cik_sec"] = t.ticker.map(lambda x: cik_of.get(x, (None, None))[0])
t["sec_title"] = t.ticker.map(lambda x: cik_of.get(x, (None, None))[1])
t["cik"] = t.cik_sec.fillna(t.cik_wiki).astype(int)
t["cik_source"] = t.cik_sec.notna().map({True: "company_tickers", False: "wikipedia_fallback"})
t["cik_mismatch"] = (t.cik_sec.notna() & (t.cik_sec != t.cik_wiki)).astype(int)
t.to_csv(OUT, index=False)
print(len(t), t.sector.value_counts().to_dict(), "fallback", (t.cik_source != "company_tickers").sum(), "mismatch", t.cik_mismatch.sum())
