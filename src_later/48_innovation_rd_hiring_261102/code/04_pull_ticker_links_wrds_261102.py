# ============================================================
# START DATE   : 2026-11-02
# CODE RUN DATE: 2026-11-02
# TIME TAKEN   : ~5 min
# DESCRIPTION  : Ticker -> gvkey (comp.company) and gvkey -> Revelio rcid (revelio.company_mapping) for SEC-panel tickers.
# INPUT FILES  : analysis/35_sec_tariff_expectations_261102/02_panel_261102.csv; WRDS comp.company, revelio.company_mapping
# OUTPUT FILES : PRIVATE_CACHE, gvkey_rcid.csv
# ============================================================
import os, sys, psycopg2, pandas as pd
C = os.path.expanduser("PRIVATE_CACHE")
P = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
for f in ("ticker_gvkey.csv", "gvkey_rcid.csv"):
    if os.path.exists(C + f): sys.exit("refuse to overwrite " + f)
tk = pd.read_csv(P + "/analysis/35_sec_tariff_expectations_261102/02_panel_261102.csv", usecols=["ticker", "cik"]).drop_duplicates()
c = psycopg2.connect(host="wrds-pgdata.wharton.upenn.edu", port=9737, dbname="wrds", user="hangu0526", sslmode="require")
cur = c.cursor()
ciks = ",".join(repr(str(int(x)).zfill(10)) for x in tk.cik.dropna().unique())
# ★ CORE LOGIC: link on SEC CIK (not ticker; comp.company has no tic column here)
cur.execute(f"SELECT gvkey, conm, cik, sic FROM comp.company WHERE cik IN ({ciks})")
g = pd.DataFrame(cur.fetchall(), columns=["gvkey", "conm", "cik", "sic"]); g.to_csv(C + "ticker_gvkey.csv", index=False); print(g.shape)
cur.execute(f"SELECT rcid, gvkey, company, ticker, cik, hq_country FROM revelio.company_mapping WHERE cik IN ({ciks}) OR gvkey IN ({','.join(repr(x) for x in g.gvkey.unique())})")
r = pd.DataFrame(cur.fetchall(), columns=["rcid", "gvkey", "company", "ticker", "cik", "hq_country"]); r.to_csv(C + "gvkey_rcid.csv", index=False); print(r.shape)
