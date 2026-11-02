# ============================================================
# START DATE   : 2026-11-02
# CODE RUN DATE: 2026-11-02
# TIME TAKEN   : ~25 min writing
# DESCRIPTION  : Step 5b. Census C5 (country x HS2 x month YoY log imports on the tariff rate in force and the lagged statement
#                signal; country x HS2 and HS2 x month FE, country-clustered SE, circular date-shift placebo) run with the
#                unchanged code/36 07_c5_rate_signal_261102.py, after substituting (i) the country-month signal block and (ii) the
#                list of specifications. New country-month signals come from the typed passages whose target names a country
#                (China -> CHN; EU members and "Europe/EU" -> EUN): indicator = at least one typed statement naming the country in
#                the month. Caveat: passages were selected for China/broad path evidence, so country mentions outside that
#                evidence are not in the signal.
# INPUT FILES  : 03_codes_261102.csv; country_list_261101.csv; code/36 script text and its inputs (Census copies, tariff calendar)
# OUTPUT FILES : analysis/49_statement_type_261102/census_c5/07_c5_*.csv, 07_c5_meta.json
# ============================================================
import os, re, sys
from pathlib import Path
here = Path(__file__).resolve(); topic = here.parents[2]
src_path = topic / "code/36_census_imports_261102/07_c5_rate_signal_261102.py"
sys.path.insert(0, str(topic / "code/36_census_imports_261102"))
src = src_path.read_text()
a = src.index("pn = pd.read_csv(cm.PANEL31"); b = src.index("def sigmat")
NEW = '''
import importlib.util as _iu
_sp = _iu.spec_from_file_location("c49", HERE49 / "common_261102.py"); C49 = _iu.module_from_spec(_sp); _sp.loader.exec_module(C49)
K = pd.read_csv(C49.OUT / "03_codes_261102.csv"); K = K[K.statement_type.notna()].copy()
K["names_rate"] = K.names_rate.fillna(False).astype(bool); K["names_date"] = K.names_date.fillna(False).astype(bool)
CL = pd.read_csv(cm.CAL / "country_list_261101.csv")
EU_MEMBERS = r"france|german|spain|italy|netherlands|ireland|poland|belgium|austria|sweden|denmark|finland|portugal|greece"
def to_iso(t):
    t = str(t).lower()
    if t == "china": return "CHN"
    if not t.startswith("country:"): return None
    n = t.split(":", 1)[1]
    if re.search(r"^(eu|europe|european union)$|" + EU_MEMBERS, n): return "EUN"
    for r_ in CL.itertuples():
        if re.search(r_.regex, n): return r_.iso3
    return None
rows_ = []
for r_ in K.itertuples():
    for t in str(r_.target).split(";"):
        iso = to_iso(t)
        if iso: rows_.append((iso, pd.Period(r_.date, "M"), r_.statement_type, r_.names_rate or r_.names_date))
T_ = pd.DataFrame(rows_, columns=["iso3", "m", "stype", "spec"]).drop_duplicates()
THR_INT_ = {"threat_or_possible_future", "stated_intention_future"}
F3_ = {"threat_or_possible_future", "stated_intention_future", "decision_announcement"}
DEF = {"any": lambda d: d.stype.notna(), "thrint": lambda d: d.stype.isin(THR_INT_), "decision": lambda d: d.stype == "decision_announcement",
       "specific": lambda d: d.spec, "inforce": lambda d: d.stype == "in_force_description", "future3": lambda d: d.stype.isin(F3_)}
g = pd.MultiIndex.from_product([sorted(CL.iso3), WIN], names=["iso3", "m"]).to_frame(index=False)
for nm, fn in DEF.items():
    hit = T_[fn(T_)].groupby(["iso3", "m"]).size().rename(nm).reset_index()
    g = g.merge(hit, on=["iso3", "m"], how="left"); g[nm] = (g[nm].fillna(0) > 0).astype(float)
SIGCOLS = {nm: nm for nm in DEF}
'''
src = src[:a] + NEW + src[b:]
c0 = src.index('for s in ("thr", "jev", "men"): run('); c1 = src.index("cm.safe_write(pd.DataFrame(est_rows)")
RUNS = '''FAILED = []
def rs(name, *a, **k):
    try:
        run(name, *a, **k)
    except np.linalg.LinAlgError:      # ★ IMPORTANT: guard only. A typed signal with no variation in the estimation sample makes X'X singular.
        est_rows[:] = [e for e in est_rows if e["spec"] != name]; FAILED.append(name); print("singular, no variation:", name, flush=True)
for s in DEF: rs(f"A_{s}", P96, sig=s)
for s in ("thrint", "decision", "future3", "inforce"): rs(f"B1_ex_CHN_{s}", P96, sig=s, drop=("CHN",))
print("FAILED", FAILED)
'''
src = src[:c0] + RUNS + src[c1:]
def sub(x, y):
    global src
    assert src.count(x) == 1, x
    src = src.replace(x, y)
sub("OUT = cm.get_out()", 'OUT = Path(os.environ.get("S49_OUT", topic / "analysis/49_statement_type_261102")) / "census_c5"; OUT.mkdir(parents=True, exist_ok=True)')
sub('g[["iso3", "m", "thr", "jev", "lmen", "impl"]]', 'g[["iso3", "m"] + list(DEF)]')
sub("import json, time", "import json, time, re, os\nfrom pathlib import Path")
exec(compile(src, str(here), "exec"), {"__file__": str(here), "__name__": "__main__", "HERE49": here.parent, "topic": topic})
