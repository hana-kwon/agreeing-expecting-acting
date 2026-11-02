# ============================================================
# START DATE   : 2026-11-02
# CODE RUN DATE: 2026-11-02
# TIME TAKEN   : ~3 min
# DESCRIPTION  : Download the two public HBS Pricing Lab Tariff Tracker CSVs used in step 39 and log URL, date, size, SHA-256.
#                Files are NOT to be redistributed (site shows only "Copyright HBS Pricing Lab"); copy stays in the project data folder.
# INPUT FILES  : https://www.pricinglab.org/files/Cavallo_Llamas_Vazquez_countries.csv and ..._domestic_imported_trend_line.csv
# OUTPUT FILES : data/39_retail_prices_statements_261102/<the two csv> and 00_download_log_261102.json
# ============================================================
import json, hashlib, datetime, urllib.request
from common_261102 import *

log = DATA / "00_download_log_261102.json"
if log.exists() or any((DATA / f).exists() for f in FILES):
    raise SystemExit("refuse to overwrite: data already downloaded")   # ★ IMPORTANT
rec = {"download_date": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
       "note": "Two fixed URLs on pricinglab.org (a Harvard research site). No domain list was crawled, so the adult/gambling filter step does not apply.",
       "files": []}
for f in FILES:
    url = BASE_URL + f
    req = urllib.request.Request(url, headers={"User-Agent": "research-download (academic, single file fetch)"})
    with urllib.request.urlopen(req, timeout=60) as r:
        body = r.read(); hdr = dict(r.headers); code = r.status
    (DATA / f).write_bytes(body)
    rec["files"].append({"file": f, "url": url, "http_status": code, "bytes": len(body),
                         "sha256": hashlib.sha256(body).hexdigest(),
                         "last_modified_header": hdr.get("Last-Modified"), "etag": hdr.get("ETag")})
    print(f, code, len(body))
log.write_text(json.dumps(rec, indent=2))
