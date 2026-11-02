# ============================================================
# START DATE   : 2026-11-02
# CODE RUN DATE: 2026-11-02
# TIME TAKEN   : ~10 min
# DESCRIPTION  : Census C5 (code/36 07_c5_rate_signal_261102.py, text unchanged except the signal block, the run list and OUT) with the chosen
#                measure f3: monthly count of higher-direction future claims naming the country before an increase was in force there,
#                signal lagged one month as in C5, standardized. Same circular date-shift placebo of the original script.
# INPUT FILES  : analysis/53_first_stage_261102/05_f3_passages_261102.csv; everything code/36 07 reads
# OUTPUT FILES : analysis/53_first_stage_261102/census_c5_f3/07_c5_*.csv
# ============================================================
import os, sys
from pathlib import Path
here = Path(__file__).resolve(); topic = here.parents[2]
sys.path.insert(0, str(topic / "code/36_census_imports_261102"))
src = (topic / "code/36_census_imports_261102/07_c5_rate_signal_261102.py").read_text()
a = src.index("pn = pd.read_csv(cm.PANEL31"); b = src.index("def sigmat")
NEW = '''
F3 = pd.read_csv(OUT53 / "05_f3_passages_261102.csv", parse_dates=["date"]); F3["m"] = F3.date.dt.to_period("M")
CL = pd.read_csv(cm.CAL / "country_list_261101.csv")
g = pd.MultiIndex.from_product([sorted(CL.iso3), WIN], names=["iso3", "m"]).to_frame(index=False)
g = g.merge(F3.groupby(["iso3", "m"]).size().rename("f3").reset_index(), on=["iso3", "m"], how="left"); g["f3"] = g.f3.fillna(0.0)
g["f3"] = (g.f3 - g.f3.mean()) / g.f3.std()
SIGCOLS = {"f3": "f3"}
'''
src = src[:a] + NEW + src[b:]
c0 = src.index('for s in ("thr", "jev", "men"): run('); c1 = src.index("cm.safe_write(pd.DataFrame(est_rows)")
src = src[:c0] + 'run("A_f3", P96, sig="f3")\nrun("B1_ex_CHN_f3", P96, sig="f3", drop=("CHN",))\n' + src[c1:]
def sub(x, y):
    global src
    assert src.count(x) == 1, x
    src = src.replace(x, y)
sub("OUT = cm.get_out()", 'OUT = OUT53 / "census_c5_f3"; OUT.mkdir(parents=True, exist_ok=True)')
sub('g[["iso3", "m", "thr", "jev", "lmen", "impl"]]', 'g[["iso3", "m", "f3"]]')
exec(compile(src, str(here), "exec"), {"__file__": str(here), "__name__": "__main__", "OUT53": topic / "analysis/53_first_stage_261102"})
