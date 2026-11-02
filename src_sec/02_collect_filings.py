# ============================================================
# START DATE   : 2026-11-02
# CODE RUN DATE: 2026-11-02
# TIME TAKEN   : ~20 min
# DESCRIPTION  : Download 10-K/10-Q primary documents (2023-07-01..2026-10-31) for sample firms from SEC EDGAR; manifest with URL, SHA-256, download date.
# INPUT FILES  : data/35.../01_sample_firms_261102.csv ; data.sec.gov submissions JSON (network)
# OUTPUT FILES : ~/claude_local_cache/sec35_261102/passages/<ticker>/<accession>.json.gz (tariff paragraphs; raw HTML not retained because disk was full) ; data/35.../02_filings_manifest_261102.csv
# ============================================================
import os
import csv, gzip, hashlib, json, sys, threading, time, datetime, urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import pandas as pd
from edgar_text import tariff_paragraphs

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "35_sec_tariff_expectations_261102"
RAW = Path.home() / "claude_local_cache" / "sec35_261102" / "raw"      # transient; deleted after parsing (disk full)
PASS = Path.home() / "claude_local_cache" / "sec35_261102" / "passages"
MANIFEST = DATA / "02_filings_manifest_261102.csv"
FORMS = {"10-K", "10-Q"}
START, END = "2023-07-01", "2026-10-31"       # before 2024-07-01 = baseline (REPEAT detection only)
UA = os.environ["SEC_USER_AGENT"]                 # approved for SEC requests only
lock, last = threading.Lock(), [0.0]

def get(url):
    # ★ IMPORTANT: global limiter of ~6 requests/second across threads (SEC limit is 10/s)
    for attempt in range(6):
        with lock:
            wait = 0.17 - (time.time() - last[0])
            if wait > 0: time.sleep(wait)
            last[0] = time.time()
        req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept-Encoding": "identity"})
        try:
            with urllib.request.urlopen(req, timeout=90) as r: return r.read()
        except urllib.error.HTTPError as e:
            if e.code in (429, 503, 500): time.sleep(5 * (attempt + 1)); continue
            raise
        except Exception: time.sleep(3 * (attempt + 1))
    raise RuntimeError("failed: " + url)

def one_firm(r):
    t, cik = r["ticker"], int(r["cik"]); rows = []
    sub = json.loads(get(f"https://data.sec.gov/submissions/CIK{cik:010d}.json"))
    blks = [sub["filings"]["recent"]] + [json.loads(get("https://data.sec.gov/submissions/" + f["name"])) for f in sub["filings"].get("files", [])]
    for b in blks:
        for i, form in enumerate(b["form"]):
            d = b["filingDate"][i]
            if form not in FORMS or not (START <= d <= END): continue
            acc, doc = b["accessionNumber"][i], b["primaryDocument"][i]
            url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc.replace('-','')}/{doc}"
            pout = PASS / t / f"{acc}.json.gz"; pout.parent.mkdir(parents=True, exist_ok=True)
            tmp = RAW / t / f"{acc}.htm.gz"
            raw = None
            if tmp.exists():                       # leftover from the first (disk-full) run; re-verify, re-download if unreadable
                try: raw = gzip.open(tmp, "rb").read()
                except Exception: raw = None
            if raw is None:
                try: raw = get(url)
                except Exception as e: print("FAIL", t, acc, e, flush=True); continue
            ps = tariff_paragraphs(raw)
            for _try in range(30):                 # the shared disk was intermittently full: retry the small write
                try:
                    with gzip.open(pout, "wt") as f: json.dump(ps, f)
                    break
                except OSError: time.sleep(10)
            else: raise RuntimeError("cannot write " + str(pout))
            if tmp.exists(): tmp.unlink()
            rows.append(dict(ticker=t, cik=cik, sector=r["sector"], form=form, filing_date=d, report_date=b["reportDate"][i], accession=acc, url=url,
                             sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw), n_tariff_paragraphs=len(ps), downloaded=datetime.date.today().isoformat(),
                             baseline=int(d < "2024-07-01")))
    print(t, len(rows), flush=True)
    return rows

def main():
    if MANIFEST.exists(): sys.exit("manifest exists; refusing to overwrite")
    firms = pd.read_csv(DATA / "01_sample_firms_261102.csv").to_dict("records")
    with ThreadPoolExecutor(3) as ex: res = list(ex.map(one_firm, firms))
    rows = sorted([x for r in res for x in r], key=lambda r: (r["ticker"], r["filing_date"], r["accession"]))
    with open(MANIFEST, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print("total", len(rows))

if __name__ == "__main__": main()
