# ============================================================
# START DATE   : 2026-11-02
# CODE RUN DATE: 2026-11-02
# TIME TAKEN   : ~15 min
# DESCRIPTION  : Shared loaders for step 36 (Census imports): HS2/HS4 files, country crosswalk, tariff calendar events, safe writer.
# INPUT FILES  : data/36_census_imports_261102/hs2_*.csv, hs4w_*.csv; tariff calendar repo copy (B_extended increase events, country list)
# OUTPUT FILES : none (imported by 02-05 scripts)
# ============================================================
import pathlib, argparse
import numpy as np, pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "36_census_imports_261102"
ANALYSIS = ROOT / "analysis" / "36_census_imports_261102"
REPO = pathlib.Path.home() / "research_local/orgsci_repos/agreeing_expecting_acting/data/public"
CAL = REPO / "tariff_calendar"
PANEL31 = ROOT / "data" / "31_tariff_calendar_fix_261101" / "02_panel_B_extended.csv"
FIRST, LAST = "2023-01", None  # LAST is read from the files

def get_out():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ANALYSIS))
    a = ap.parse_args()
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    return out

def safe_write(df_or_fn, path):
    path = pathlib.Path(path)
    if path.exists():
        raise SystemExit(f"refuse to overwrite {path}")  # ★ IMPORTANT: scripts never overwrite outputs
    if callable(df_or_fn):
        df_or_fn(path)
    else:
        df_or_fn.to_csv(path, index=False)

def load_hs2():
    fs = sorted(DATA.glob("hs2ch_*.csv"))
    d = pd.concat([pd.read_csv(f, dtype=str) for f in fs], ignore_index=True)
    d = d[d.CTY_CODE.str.match(r"^[1-9][0-9]{3}$")].copy()   # countries only (drop 0xxx groups, 1XXX regions, '-' total)
    d["gen"] = d.GEN_VAL_MO.astype(float); d["con"] = d.CON_VAL_MO.astype(float)
    d["hs2"] = d.I_COMMODITY; d["month"] = d["time"]
    return d[["month", "CTY_CODE", "CTY_NAME", "hs2", "gen", "con"]]

def load_hs4w():
    fs = sorted(DATA.glob("hs4w_*.csv"))
    d = pd.concat([pd.read_csv(f, dtype=str).assign(month=f.stem[5:]) for f in fs], ignore_index=True)
    d = d[d.CTY_CODE == "-"].copy()
    d["gen"] = d.GEN_VAL_MO.astype(float)
    d["hs4"] = d.I_COMMODITY
    return d[["month", "hs4", "gen"]]

EU27 = ["AUSTRIA","BELGIUM","BULGARIA","CROATIA","CYPRUS","CZECH REPUBLIC","DENMARK","ESTONIA","FINLAND","FRANCE","GERMANY","GREECE",
        "HUNGARY","IRELAND","ITALY","LATVIA","LITHUANIA","LUXEMBOURG","MALTA","NETHERLANDS","POLAND","PORTUGAL","ROMANIA","SLOVAKIA","SLOVENIA","SPAIN","SWEDEN"]
OVERRIDE = {"BHS": "BAHAMAS", "CHN": "CHINA", "CIV": "COTE D'IVOIRE", "FLK": "FALKLAND ISLANDS (ISLAS MALVINAS)", "GAB": "GABON",
            "KOR": "KOREA, SOUTH", "MMR": "BURMA", "NRU": "NAOERO", "PHL": "PHILIPPINES", "TUR": "TURKEY"}

def crosswalk(cty):
    """iso3 -> list of Census CTY_CODEs. EUN expands to the EU27 members. Returns (dict, unmapped list)."""
    names = cty.drop_duplicates().assign(u=lambda x: x.CTY_NAME.str.upper()).set_index("u").CTY_CODE.to_dict()
    cl = pd.read_csv(CAL / "country_list_261101.csv")
    m, un = {}, []
    for _, r in cl.iterrows():
        if r.iso3 == "EUN":
            m["EUN"] = [names[n] for n in EU27 if n in names]; continue
        nm = OVERRIDE.get(r.iso3, r["name"].upper())
        if nm in names: m[r.iso3] = [names[nm]]
        else: un.append((r.iso3, r["name"]))
    return m, un

def increase_events(last_month):
    a = pd.read_csv(CAL / "corrected" / "02_increase_events_B_extended.csv")
    a = a[(a.cancelled.astype(str) != "True") & (a.short_lived.astype(str) != "True")].copy()
    a["eff"] = pd.to_datetime(a["eff"])
    all_iso = set(a.iso3)   # any country that ever has an increase row (incl. pending/future) is excluded from controls
    a = a[a.eff <= pd.Period(last_month, "M").end_time]
    a["signing_date"] = pd.to_datetime(a.signing_date).fillna(pd.to_datetime(a.publication_date))   # fall back to publication date if signing date missing
    return a, all_iso

def months_index(last_month):
    return pd.period_range(FIRST, last_month, freq="M")

def wide_country_month(d, idx):
    m = d.groupby(["CTY_CODE", "month"]).gen.sum().unstack("month")
    m.columns = pd.PeriodIndex(m.columns, freq="M")
    return m.reindex(columns=idx)

def yoy(M):
    L = np.log(M.where(M > 0))
    return L - L.shift(12, axis=1)
