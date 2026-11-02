# ============================================================
# START DATE   : 2026-11-02
# CODE RUN DATE: 2026-11-02
# TIME TAKEN   : ~5 min
# DESCRIPTION  : Future-higher claims before a country's first increase (f3), counted as passages (original) and as statement days (the unit of the downstream tests).
#                Within-country and within-bin AUC for effective increases 2-12 weeks ahead, with independent-bin and 6-bin moving-block bootstraps, and by half of the bins.
# INPUT FILES  : analysis/53_first_stage_261102/01_panel_261102.csv, 05_f3_passages_261102.csv
# OUTPUT FILES : analysis/56_final_checks_261102/02_auc_days_261102.csv
# ============================================================
import sys
from pathlib import Path
import numpy as np, pandas as pd
T = Path(__file__).resolve().parents[2]; O = T / "analysis/56_final_checks_261102/02_auc_days_261102.csv"
if O.exists(): sys.exit("refuse to overwrite")
P = pd.read_csv(T / "analysis/53_first_stage_261102/01_panel_261102.csv", parse_dates=["bin_end"]).sort_values(["iso3", "k"])
F = pd.read_csv(T / "analysis/53_first_stage_261102/05_f3_passages_261102.csv", parse_dates=["date"]).drop_duplicates()
t0 = pd.Timestamp("2025-01-24"); F["k"] = np.ceil((F.date - t0).dt.days / 14.0)
days = F.groupby(["iso3", "k"]).size().rename("f3_days").reset_index(); P = P.merge(days, on=["iso3", "k"], how="left"); P["f3_days"] = P.f3_days.fillna(0)
pcol = [c for c in P.columns if c.startswith("f3_")and c != "f3_days"][0]; print("passage column", pcol, "cells differing", int((P[pcol].fillna(0) != P.f3_days).sum()), "sum", P[pcol].sum(), P.f3_days.sum())
nC, nB = P.iso3.nunique(), P.k.nunique(); y = P["O1_eff_incr__fut_2_12"].values.reshape(nC, nB)
def within(x, y, axis):
    v = ~np.isnan(y); pos = ((y == 1) & v).astype(float); neg = ((y == 0) & v).astype(float)
    if axis == "b": x, pos, neg = x.T, pos.T, neg.T
    C = (x[:, :, None] > x[:, None, :]).astype(np.float32) + 0.5 * (x[:, :, None] == x[:, None, :])
    tot = np.einsum("cj,cl->", pos, neg); return np.einsum("cjl,cj,cl->", C, pos, neg) / tot if tot else np.nan
rng = np.random.default_rng(261102); R = 500
iid = [rng.integers(0, nB, nB) for _ in range(R)]
L = 6; block = [np.concatenate([np.arange(s, s + L) % nB for s in rng.integers(0, nB, int(np.ceil(nB / L)))])[:nB] for _ in range(R)]   # ★ IMPORTANT: outcome windows overlap adjacent bins, so resampling single bins understates dependence
rows = []
for name, col in [("passages", pcol), ("statement_days", "f3_days")]:
    x = P[col].fillna(0).values.reshape(nC, nB)
    for ax, lab in [("c", "within_country"), ("b", "within_bin")]:
        pt = within(x, y, ax); r = dict(unit=name, auc=lab, estimate=pt, nonzero_cells=int((x != 0).sum()), positive_cells=int(np.nansum(y)))
        for bn, bs in [("iid_bins", iid), ("block6", block)]:
            q = np.nanpercentile([within(x[:, b], y[:, b], ax) for b in bs], [2.5, 97.5]); r[bn + "_lo"], r[bn + "_hi"] = q
        h = nB // 2; r["first_half_bins"] = within(x[:, :h], y[:, :h], ax); r["second_half_bins"] = within(x[:, h:], y[:, h:], ax); rows.append(r)
pd.DataFrame(rows).to_csv(O, index=False); print(pd.DataFrame(rows).round(3).to_string(index=False))
