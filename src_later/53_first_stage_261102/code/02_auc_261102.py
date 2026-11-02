# ============================================================
# START DATE   : 2026-11-02
# CODE RUN DATE: 2026-11-02
# TIME TAKEN   : ~10 min
# DESCRIPTION  : Pooled, within-country and within-bin AUC of each statement measure against each policy outcome window, with a
#                bin-block bootstrap (300 draws, seed 261102). Spec: analysis/53 00_spec_261102.md. Also independent event counts.
# INPUT FILES  : analysis/53_first_stage_261102/01_panel_261102.csv
# OUTPUT FILES : analysis/53_first_stage_261102/02_auc_261102.csv, 02_event_counts_261102.csv
# ============================================================
import sys
from pathlib import Path
import numpy as np, pandas as pd

OUT = Path(__file__).resolve().parents[2] / "analysis/53_first_stage_261102"
tg = [OUT / "02_auc_261102.csv", OUT / "02_event_counts_261102.csv"]
for t in tg:
    if t.exists(): sys.exit(f"output exists; refusing to overwrite: {t}")
P = pd.read_csv(OUT / "01_panel_261102.csv")
NEW = ["a1_jev_score", "a2_jev_share", "b1_n_future", "b2_n_threat", "b3_n_intent", "c1_first_threat", "c2_first_threat_or_intent", "d1_n_specific", "d2_n_dated", "e1_n_notinforce", "e2_n_notinforce_higher", "f1_n_future_higher", "f2_n_new_vs_recent", "f3_n_higher_no_policy_yet"]
REF = ["ref_mentions28", "ref_jev28", "ref_tirade_threats4"]
MS = NEW + REF
OUTC = ["O1_eff_incr", "O2_signing", "O3_tir_threat", "O3_tir_announce", "O3_tir_implement", "O3_tir_any"]
WINS = ["fut_2_12", "fut_2_4", "fut_5_8", "fut_9_12", "imm_0_2", "conc", "past_2_12"]
P = P.sort_values(["iso3", "k"]).reset_index(drop=True)
isos = sorted(P.iso3.unique()); nC, nB = len(isos), P.k.nunique()
X = np.stack([P[m].fillna(0).values.reshape(nC, nB) for m in MS])                     # [m,c,b]
def Y(o, w): return P[f"{o}__{w}"].values.reshape(nC, nB)                              # may hold NaN
def pooled(x, y):
    v = ~np.isnan(y); xs, ys = x[v], y[v].astype(int); n1 = ys.sum(); n0 = len(ys) - n1
    if n1 == 0 or n0 == 0: return np.nan
    r = pd.Series(xs).rank().values; return (r[ys == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
def within(Xr, y, axis):
    """axis='c': pairs of bins within the same country; axis='b': pairs of countries within the same bin. Ties 0.5."""
    v = ~np.isnan(y); pos = ((y == 1) & v).astype(float); neg = ((y == 0) & v).astype(float)
    if axis == "b": Xr = np.swapaxes(Xr, 1, 2); pos, neg = pos.T, neg.T
    C = (Xr[:, :, :, None] > Xr[:, :, None, :]).astype(np.float32) + 0.5 * (Xr[:, :, :, None] == Xr[:, :, None, :])
    conc = np.einsum("mcjl,cj,cl->m", C, pos, neg); tot = np.einsum("cj,cl->", pos, neg)
    return conc / tot if tot > 0 else np.full(Xr.shape[0], np.nan), tot
rng = np.random.default_rng(261102); REPS = 300
boots = [rng.integers(0, nB, nB) for _ in range(REPS)]
CI_WINS = {o: ["fut_2_12", "past_2_12"] for o in OUTC}
rows = []
for o in OUTC:
    for w in WINS:
        y = Y(o, w)
        if np.isnan(y).all(): continue
        wc, npc = within(X, y, "c"); wb, npb = within(X, y, "b")
        pl = [pooled(X[i].ravel(), y.ravel()) for i in range(len(MS))]
        ci = w in CI_WINS[o]
        if ci:
            bc = np.zeros((REPS, len(MS))); bb = np.zeros((REPS, len(MS))); bp = np.zeros((REPS, len(MS)))
            for r, b in enumerate(boots):
                bc[r], _ = within(X[:, :, b], y[:, b], "c"); bb[r], _ = within(X[:, :, b], y[:, b], "b")
                bp[r] = [pooled(X[i][:, b].ravel(), y[:, b].ravel()) for i in range(len(MS))]
            q = lambda a: np.nanpercentile(a, [2.5, 97.5], axis=0)
            cc, cb, cp = q(bc), q(bb), q(bp)
        for i, m in enumerate(MS):
            rows.append(dict(outcome=o, window=w, measure=m, n_cells=int((~np.isnan(y)).sum()), n_pos=int(np.nansum(y)),
                             auc_pooled=pl[i], auc_pooled_lo=cp[0, i] if ci else np.nan, auc_pooled_hi=cp[1, i] if ci else np.nan,
                             auc_within_country=wc[i], wc_lo=cc[0, i] if ci else np.nan, wc_hi=cc[1, i] if ci else np.nan, wc_pairs=npc,
                             auc_within_bin=wb[i], wb_lo=cb[0, i] if ci else np.nan, wb_hi=cb[1, i] if ci else np.nan, wb_pairs=npb,
                             nonzero_cells=int((X[i] != 0).sum())))
        print(o, w, "done", flush=True)
pd.DataFrame(rows).to_csv(OUT / "02_auc_261102.csv", index=False)
# independent event counts (event dates, not country-bins) for window fut_2_12 and all
ec = []
for o in OUTC:
    y = P[f"{o}__fut_2_12"]
    ec.append(dict(outcome=o, positive_country_bins=int(y.sum()), countries_with_event=int(P[y == 1].iso3.nunique()),
                   countries_with_any_event_in_window_set=int(P.groupby("iso3")[[f"{o}__{w}" for w in WINS]].max().max(axis=1).sum()),
                   bins_with_event_any_country=int(P[y == 1].k.nunique())))
pd.DataFrame(ec).to_csv(OUT / "02_event_counts_261102.csv", index=False)
