# ============================================================
# START DATE   : 2026-11-02
# CODE RUN DATE: 2026-11-02
# TIME TAKEN   : ~2 min
# DESCRIPTION  : The volume's compustat_fundq.parquet lacks xrdq. Pull gvkey, datadate, xrdq, saleq, atq,
#                sic from WRDS comp.fundq (read-only SQL, same filters as the volume's SQL).
# INPUT FILES  : WRDS comp.fundq, comp.company
# OUTPUT FILES : PRIVATE_CACHE
# ============================================================
import os, sys, psycopg2, pandas as pd
OUT = os.path.expanduser("PRIVATE_CACHE")
if os.path.exists(OUT): sys.exit("refuse to overwrite " + OUT)
c = psycopg2.connect(host="wrds-pgdata.wharton.upenn.edu", port=9737, dbname="wrds", user="hangu0526", sslmode="require")
q = """SELECT gvkey, datadate, fyearq, fqtr, datacqtr, fyr, curcdq, conm, saleq, xrdq, atq, rdq
FROM comp.fundq WHERE indfmt='INDL' AND datafmt='STD' AND popsrc='D' AND consol='C'
AND datadate >= '2021-10-01' AND datadate <= '2026-09-30'"""
df = pd.read_sql(q, c); df.to_csv(OUT, index=False); print(df.shape, df.xrdq.notna().sum())
