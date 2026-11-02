# ============================================================
# START DATE   : 2026-11-02
# CODE RUN DATE: 2026-11-02
# TIME TAKEN   : ~10 min parse (see log)
# DESCRIPTION  : Stream-parse USPTO trademark full-text XML zips (annual backfile parts and daily files)
#                into one slim parquet per zip: serial, filing date, status, owner 1 name/state/country/
#                entity type, NICE classes. Refuses to overwrite existing outputs.
# INPUT FILES  : PRIVATE_CACHE*.zip  (TRTYRAP annual parts, TRTDXFAP daily)
# OUTPUT FILES : PRIVATE_CACHE<zipname>.parquet
# ============================================================
import sys, zipfile, re, os, io
import pandas as pd

RAW = os.path.expanduser("PRIVATE_CACHE")
OUT = os.path.expanduser("PRIVATE_CACHE")

# ★ CORE LOGIC: line-based streaming parse. Only the FIRST case-file-owner is kept (primary applicant);
# filing-date is the first <filing-date> in the case-file-header; NICE classes come from <international-code>
# inside <classification> blocks (zero-padded 3 digits; 001-045).
def parse_zip(path):
    rows = []
    cur = None
    in_owner = False; owner_seen = False; in_cls = False
    with zipfile.ZipFile(path) as z:
        for name in z.namelist():
            if not name.endswith(".xml"): continue
            with z.open(name) as fh:
                for raw in io.TextIOWrapper(fh, encoding="utf-8", errors="replace"):
                    s = raw.strip()
                    if s.startswith("<case-file>"):
                        cur = dict(serial=None, filing=None, status=None, mark_draw=None, owner=None, state=None,
                                   country=None, entity=None, classes=[], tm_in=None, svc_in=None,
                                   itu_filed=None, use_filed=None, amended_to_use=None, reg_no=None, reg_date=None, sou_date=None)
                        in_owner = False; owner_seen = False
                    elif cur is None:
                        continue
                    elif s.startswith("<serial-number>") and cur["serial"] is None:
                        cur["serial"] = s[15:-16]
                    elif s.startswith("<filing-date>") and cur["filing"] is None:
                        cur["filing"] = s[13:-14]
                    elif s.startswith("<status-code>") and cur["status"] is None:
                        cur["status"] = s[13:-14]
                    elif s.startswith("<registration-number>") and cur["reg_no"] is None:
                        cur["reg_no"] = s[21:-22]
                    elif s.startswith("<registration-date>") and cur["reg_date"] is None:
                        cur["reg_date"] = s[19:-20]
                    # ★ IMPORTANT: filing basis. intent-to-use-in = filed under Sec. 1(b); filed-as-use-application-in = Sec. 1(a).
                    # Commercialization step = amended-to-use flag, a 'statement of use' event date, or registration (right-censored).
                    elif s.startswith("<intent-to-use-in>"):
                        cur["itu_filed"] = s[18:-19]
                    elif s.startswith("<filed-as-use-application-in>"):
                        cur["use_filed"] = s[29:-30]
                    elif s.startswith("<amended-to-use-application-in>"):
                        cur["amended_to_use"] = s[31:-32]
                    elif s.startswith("<description-text>") and "STATEMENT OF USE" in s.upper():
                        cur["_sou_pending"] = True
                    elif s.startswith("<date>") and cur.get("_sou_pending"):
                        d = s[6:-7]
                        if cur["sou_date"] is None or d < cur["sou_date"]: cur["sou_date"] = d
                        cur["_sou_pending"] = False
                    elif s.startswith("<trademark-in>"):
                        cur["tm_in"] = s[14:-15]
                    elif s.startswith("<service-mark-in>"):
                        cur["svc_in"] = s[17:-18]
                    elif s.startswith("<international-code>"):
                        cur["classes"].append(s[20:-21])
                    elif s.startswith("<case-file-owner>"):
                        in_owner = not owner_seen
                    elif s.startswith("</case-file-owner>"):
                        if in_owner: owner_seen = True
                        in_owner = False
                    elif in_owner:
                        if s.startswith("<party-name>"): cur["owner"] = s[12:-13]
                        elif s.startswith("<state>"): cur["state"] = s[7:-8]
                        elif s.startswith("<country>") and cur["country"] is None: cur["country"] = s[9:-10]
                        elif s.startswith("<legal-entity-type-code>"): cur["entity"] = s[24:-25]
                    elif s.startswith("</case-file>"):
                        cur.pop("_sou_pending", None)
                        cur["classes"] = ",".join(sorted(set(cur["classes"])))
                        rows.append(cur); cur = None
    return pd.DataFrame(rows)

if __name__ == "__main__":
    for zn in sys.argv[1:]:
        out = os.path.join(OUT, os.path.basename(zn).replace(".zip", ".parquet"))
        if os.path.exists(out):
            print("SKIP (exists, refuse to overwrite):", out); continue
        df = parse_zip(os.path.join(RAW, zn))
        # keep only 2021+ filings (panel starts 2022Q1; 2021 retained for sanity checks)
        df = df[df.filing.fillna("0") >= "20210101"]
        df.to_parquet(out)
        print(zn, len(df), df.filing.min() if len(df) else None, df.filing.max() if len(df) else None, flush=True)
