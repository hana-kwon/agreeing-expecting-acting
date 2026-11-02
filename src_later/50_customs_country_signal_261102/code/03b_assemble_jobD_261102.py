# ============================================================
# START DATE   : 2026-11-02
# CODE RUN DATE: 2026-11-02
# TIME TAKEN   : ~2 min
# DESCRIPTION  : Summarize job D (placebos on the sample where all 16-week lags are inside the statement panel;   ).
# INPUT FILES  : analysis/50_customs_country_signal_261102/02_jobD_placebo_261102.csv, 02_jobD_coefs_261102.csv
# OUTPUT FILES : 03b_placebo_summary_lagsInPanel_261102.csv
# ============================================================
import sys
from pathlib import Path
import numpy as np, pandas as pd
OUT = Path(__file__).resolve().parents[2] / "analysis/50_customs_country_signal_261102"
o = OUT / "03b_placebo_summary_lagsInPanel_261102.csv"
if o.exists(): sys.exit(f"output exists; refusing to overwrite: {o}")
pl = pd.read_csv(OUT / "02_jobD_placebo_261102.csv"); co = pd.read_csv(OUT / "02_jobD_coefs_261102.csv")
rows = []
for (s, k), g in pl.groupby(["sample", "kind"]):
    a = g.actual.iloc[0]
    rows.append(dict(sample=s, kind=k, n=len(g), actual=a, placebo_mean=g.est.mean(), placebo_sd=g.est.std(), share_abs_ge=(g.est.abs() >= abs(a)).mean(), mde_perm_2_8sd=2.8 * g.est.std()))
r = pd.DataFrame(rows); r.to_csv(o, index=False); print(r.round(4).to_string()); print(co[co.term.str.startswith("S_")][["spec", "sample", "est", "se_2way", "p_2way", "mde", "nobs"]].round(4).to_string())
