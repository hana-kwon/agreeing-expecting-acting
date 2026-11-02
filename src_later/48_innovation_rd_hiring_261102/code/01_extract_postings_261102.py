# ============================================================
# START DATE   : 2026-11-02
# CODE RUN DATE: 2026-11-02
# TIME TAKEN   : (filled in run log)
# DESCRIPTION  : Extract job-level Revelio Cosmos postings of bridged companies
#                (columns rcid, job_id, post_date, onet_code, role_k1500_v2 only)
#                into the local cache. No function coding or estimation here.
# INPUT FILES  : PRIVATE_SERVER_PATH{2023,2024,2025}.parquet (read-only)
#                .../SMB_tariff/intermediate_data/prep_reg_L2_update/revelio_smb/revelio_rcid_duns_bridge_full_261024.parquet (read-only)
# OUTPUT FILES : PRIVATE_CACHE{year}.parquet (firm-level cache, not in the project folder)
# ============================================================
import os, sys, time
import pyarrow as pa, pyarrow.parquet as pq, pyarrow.compute as pc
import pandas as pd
from concurrent.futures import ProcessPoolExecutor

SRC = "PRIVATE_SERVER_PATH"
BR = ("PRIVATE_DATA_ROOT/SMB_tariff/"
      "intermediate_data/prep_reg_L2_update/revelio_smb/revelio_rcid_duns_bridge_full_261024.parquet")
CACHE = os.path.expanduser("PRIVATE_CACHE")
COLS = ["rcid", "job_id", "post_date", "onet_code", "role_k1500_v2"]
RC = None

def init():
    global RC
    b = pd.read_parquet(BR, columns=["rcid"])
    RC = pa.array(b.rcid.unique().astype("int64"))

def work(args):
    year, rg = args
    f = pq.ParquetFile(f"{SRC}year_{year}.parquet")
    t = f.read_row_group(rg, columns=COLS)
    # ★ CORE LOGIC: keep only bridged companies; rcid is float-like in the raw file, cast to int64 first
    r = pc.cast(t["rcid"], pa.int64(), safe=False)
    m = pc.is_in(r, value_set=RC)
    t = t.filter(m)
    return t

if __name__ == "__main__":
    year = int(sys.argv[1])
    out = f"{CACHE}postings_bridge_{year}.parquet"
    if os.path.exists(out):
        sys.exit(f"refuse to overwrite {out}")
    n = pq.ParquetFile(f"{SRC}year_{year}.parquet").metadata.num_row_groups
    t0 = time.time(); parts = []
    with ProcessPoolExecutor(int(sys.argv[2]) if len(sys.argv)>2 else 4, initializer=init) as ex:
        for i, t in enumerate(ex.map(work, [(year, g) for g in range(n)], chunksize=4)):
            parts.append(t)
            if i % 100 == 0:
                print(year, i, n, round(time.time() - t0), sum(len(p) for p in parts), flush=True)
    tab = pa.concat_tables(parts)
    pq.write_table(tab, out + ".tmp", compression="zstd")
    os.rename(out + ".tmp", out)
    print("done", year, len(tab), round(time.time() - t0), flush=True)
