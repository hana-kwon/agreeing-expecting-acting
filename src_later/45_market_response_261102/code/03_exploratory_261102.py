# ============================================================
# START DATE   : 2026-11-02
# CODE RUN DATE: 2026-11-02
# TIME TAKEN   : ~6 min
# DESCRIPTION  : EXPLORATORY additions made after seeing the 02 results (not in the 00_spec): (a) the lower tail of the
#                daily signal (bottom decile, "de-escalation days"), because the 2025-04-09 pause day was not flagged
#                as an escalation day; (b) the in-office sample (trading days from 2025-01-20), because 14 of the 24
#                China-path escalation days fall in the 2024 campaign. Same event window, HAC, and date-shift placebo.
#                Reuses the definitions of 02_market_tests_261102.py by executing its preamble (before the run section).
# INPUT FILES  : same as 02_market_tests_261102.py
# OUTPUT FILES : analysis/45_market_response_261102/03_exploratory_261102.csv
# ============================================================
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
SRC = (HERE.parent / "02_market_tests_261102.py").read_text()
pre = SRC.split("# ---------------- run: actual and placebo")[0]
# the preamble's refuse-overwrite guard belongs to script 02's outputs; 03 has its own guard below
guard = 'if (OUT / "02_m1_261102.csv").exists():\n    sys.exit("02_m1_261102.csv exists; refusing to overwrite.")\n'
assert guard in pre
pre = pre.replace(guard, "")
pre = pre.replace("HERE = Path(__file__).resolve()", f"HERE = Path({str(HERE)!r})")
exec(compile(pre, "02_preamble", "exec"))

if (OUT / "03_exploratory_261102.csv").exists():
    sys.exit("03_exploratory_261102.csv exists; refusing to overwrite.")

INAUG = pd.Timestamp("2025-01-20")
keep_inaug = np.asarray(sdates >= INAUG)


def build_tail(sig, shift, keep, tail):
    sg = build(sig, shift, keep)
    if tail == "top":
        return sg
    S, evid = sg["S"], sg["evid"]
    E = np.zeros(ns, bool)
    if evid.sum() >= 10:
        thr = np.quantile(S[evid], 0.1)
        E = evid & (S <= thr) & (S < 0)
    sg = dict(sg)
    sg.update(E=E, OE=evid & ~E, acts=np.zeros(ns, bool), words=np.zeros(ns, bool),
              thr=np.nan if not E.any() else S[E].max())
    return sg


rows = []
for sample, keep in [("all", keep_all), ("in_office_from_2025-01-20", keep_inaug)]:
    for tail in ["top", "bottom"]:
        if sample == "all" and tail == "top":
            continue  # already in 02
        for sig in ["china", "broad"]:
            sg0 = build_tail(sig, 0, keep, tail)
            pl = {}
            for k in SHIFT_DAYS:
                sg = build_tail(sig, k, keep, tail)
                for oc in OUTCOMES:
                    y = Y[oc][sample_idx]
                    pl.setdefault(("M1", oc), []).append(mdiff(y, sg["E"], sg["OE"], keep))
                    pl.setdefault(("M3", oc), []).append(m3beta(y, sg, keep))
            for oc in OUTCOMES:
                y = np.where(keep, Y[oc][sample_idx], np.nan)
                one = np.ones(ns)
                r1 = hac_fit(y, np.column_stack([one, sg0["E"].astype(float), sg0["OE"].astype(float)]), None, {"c": [0, 1, -1]})["c"]
                tests = [("M1_evid", r1)]
                if tail == "top":
                    r3 = hac_fit(y, np.column_stack([one, sg0["z"], sg0["evid"].astype(float), dow]), None, {"c": [0, 1]})["c"]
                    tests.append(("M3", r3))
                for name, (est, se, p) in tests:
                    arr = np.array(pl[("M1" if name == "M1_evid" else "M3", oc)], float)
                    arr = arr[~np.isnan(arr)]
                    rows.append(dict(sample=sample, tail=tail, signal=sig, test=name, market=oc[0], outcome=oc[1],
                                     n_E=int(sg0["E"].sum()), n_OE=int(sg0["OE"].sum()), threshold_S=float(sg0["thr"]),
                                     estimate=est, se=se, p_hac=p, placebo_n=len(arr),
                                     placebo_p=(np.abs(arr) >= abs(est)).mean() if len(arr) and not np.isnan(est) else np.nan))
pd.DataFrame(rows).to_csv(OUT / "03_exploratory_261102.csv", index=False)
print("done", len(rows))
