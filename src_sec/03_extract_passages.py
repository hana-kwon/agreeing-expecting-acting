# ============================================================
# START DATE   : 2026-11-02
# CODE RUN DATE: 2026-11-02
# TIME TAKEN   : ~25 min
# DESCRIPTION  : Strip HTML, split into paragraphs, keep paragraphs containing "tariff", mark NEW vs REPEAT per firm (word 4-gram Jaccard >= 0.75 vs the firm's earlier NEW passages, baseline filings included).
# INPUT FILES  : data/35.../02_filings_manifest_261102.csv ; ~/claude_local_cache/sec35_261102/passages/<ticker>/<accession>.json.gz
# OUTPUT FILES : data/35.../03_passages_all_261102.csv ; data/35.../03_filing_counts_261102.csv
# ============================================================
import csv, gzip, hashlib, json, re, sys
from pathlib import Path
from collections import defaultdict
from multiprocessing import Pool
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "35_sec_tariff_expectations_261102"
PASS = Path.home() / "claude_local_cache" / "sec35_261102" / "passages"
OUT, OUT_COUNTS = DATA / "03_passages_all_261102.csv", DATA / "03_filing_counts_261102.csv"
SIM_THRESHOLD = 0.75                              # ★ CORE LOGIC: same as the semiconductor pilot
def work(r):
    # passages were extracted at download time (disk was full; raw HTML not retained, see 02)
    return (r["ticker"], r["accession"], json.load(gzip.open(PASS / r["ticker"] / f"{r['accession']}.json.gz", "rt")))

def norm(s): return re.sub(r"[^a-z0-9 ]", "", s.lower().replace("-", " "))
def shingles(s):
    w = norm(s).split()
    return {tuple(w)} if len(w) < 4 else {tuple(w[i:i + 4]) for i in range(len(w) - 3)}
def jac(a, b):
    u = len(a | b); return len(a & b) / u if u else 0.0

def main():
    if OUT.exists(): sys.exit("output exists; refusing to overwrite")
    man = pd.read_csv(DATA / "02_filings_manifest_261102.csv", dtype={"accession": str}).to_dict("records")
    with Pool(10) as pool: res = pool.map(work, man, chunksize=4)
    ps_of = {(t, a): ps for t, a, ps in res}
    by_t = defaultdict(list)
    for r in man: by_t[r["ticker"]].append(r)
    rows, counts = [], []
    for t, fl in by_t.items():
        fl.sort(key=lambda r: (r["filing_date"], r["accession"]))
        kept = []
        for r in fl:
            ps = ps_of[(t, r["accession"])]
            for i, p in enumerate(ps):
                sh = shingles(p)
                best = max((jac(sh, k) for k in kept), default=0.0)
                status = "NEW" if best < SIM_THRESHOLD else "REPEAT"
                if status == "NEW": kept.append(sh)
                rows.append(dict(ticker=t, sector=r["sector"], form=r["form"], filing_date=r["filing_date"], accession=r["accession"], baseline=r["baseline"],
                                 passage_id=f"{t}_{r['accession']}_{i:03d}", status=status, max_sim=round(best, 3),
                                 sha256=hashlib.sha256(p.encode()).hexdigest(), n_chars=len(p), text=p))
            counts.append(dict(ticker=t, accession=r["accession"], filing_date=r["filing_date"], form=r["form"], baseline=r["baseline"], n_tariff_passages=len(ps)))
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    pd.DataFrame(counts).to_csv(OUT_COUNTS, index=False)
    d = pd.DataFrame(rows); a = d[d.baseline == 0]
    print("passages", len(d), "analysis-window", len(a), a.status.value_counts().to_dict(), "firms with >=1 NEW", a[a.status == "NEW"].ticker.nunique())

if __name__ == "__main__": main()
