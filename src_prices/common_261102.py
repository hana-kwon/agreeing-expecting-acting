# ============================================================
# START DATE   : 2026-11-02
# CODE RUN DATE: 2026-11-02
# TIME TAKEN   : ~10 min
# DESCRIPTION  : Shared paths, loaders and safe writer for step 39 (retail price indices vs leader statements).
# INPUT FILES  : data/39_retail_prices_statements_261102/*.csv (Pricing Lab copies), Jev parquet (read-only, #Shared_),
#                tariff calendar B_extended (agreeing_expecting_acting repo copy)
# OUTPUT FILES : none (imported by 02-05 scripts)
# ============================================================
import pathlib, argparse
import numpy as np, pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]          # topic1_trump_ai (computed from script location)
DATA = ROOT / "data" / "39_retail_prices_statements_261102"
ANALYSIS = ROOT / "analysis" / "39_retail_prices_statements_261102"
REPO = pathlib.Path(__file__).resolve().parents[1]           # repository root
CAL = REPO / "data/public/tariff_calendar/corrected"
JEV = REPO / "data/jev/event_question_option_probabilities.parquet"   # repository copy of the Jev event scores
BASE_URL = "https://www.pricinglab.org/files/"
FILES = ["Cavallo_Llamas_Vazquez_countries.csv", "Cavallo_Llamas_Vazquez_domestic_imported_trend_line.csv"]


def get_out():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ANALYSIS))
    ap.add_argument("--nboot", type=int, default=2000)
    a = ap.parse_args()
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    return out, a


def safe_write(obj, path):
    path = pathlib.Path(path)
    if path.exists():
        raise SystemExit(f"refuse to overwrite {path}")   # ★ IMPORTANT: scripts never overwrite outputs
    if callable(obj):
        obj(path)
    elif isinstance(obj, (str, bytes)):
        path.write_text(obj) if isinstance(obj, str) else path.write_bytes(obj)
    else:
        obj.to_csv(path, index=False)


def load_prices():
    a = pd.read_csv(DATA / FILES[0]); b = pd.read_csv(DATA / FILES[1])
    for d in (a, b):
        d["date"] = pd.to_datetime(d["date"], format="%d%b%Y")
    m = a.merge(b[["date", "index_imported", "index_domestic"]], on="date", validate="1:1").set_index("date").sort_index()
    # ★ IMPORTANT: index_usa is identical to index_domestic (checked in the spec); "relative price" = China-origin vs US-origin goods.
    assert (m.index_usa - m.index_domestic).abs().max() == 0.0
    m["y"] = np.log(m.index_china) - np.log(m.index_usa)
    m["y2"] = np.log(m.index_imported) - np.log(m.index_domestic)
    return m
