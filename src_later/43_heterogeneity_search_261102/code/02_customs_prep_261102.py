# ============================================================
# START DATE   : 2026-11-02
# CODE RUN DATE: 2026-11-02
# TIME TAKEN   : ~5 min
# DESCRIPTION  : Customs inputs for H3/H4/H5: (a) bin-level z-scored signals for every signal definition and 96 week shifts
#                (aggregate series), (b) DIME R/D flags for firms in the code/41 cohort (firm level, cache only).
# INPUT FILES  : analysis/41_customs_team_spec_261102/01_period_series_261102.csv (bin calendar),
#                PRIVATE_CACHE (written by code/41, read only),
PRIVATE_DATA_ROOT/SMB_tariff/intermediate_data/prep_reg_L2_update/smb_firm_features_for_L2_260626.parquet (read only)
# OUTPUT FILES : analysis/43_.../02_bin_signals_261102.csv (aggregate), PRIVATE_CACHE (firm level)
# ============================================================
import sys

import pandas as pd

from common_261102 import CACHE, OUT, SHARED, SHIFTS, SIGNALS, TOPIC, signal_at

CACHE.mkdir(parents=True, exist_ok=True)
t1, t2 = OUT / "02_bin_signals_261102.csv", CACHE / "dime_firms.parquet"
for t in (t1, t2):
    if t.exists():
        sys.exit(f"{t} exists; refusing to overwrite.")
per = pd.read_csv(TOPIC / "analysis/41_customs_team_spec_261102/01_period_series_261102.csv", parse_dates=["bin_start"])
rows = []
for s in SIGNALS:
    for k in [0] + SHIFTS:
        v = signal_at(s, per.bin_start, k)
        z = ((v - v.mean()) / v.std()).values
        for b, raw, zz in zip(per.bin_index, v.values, z):
            rows.append(dict(signal=s, shift_weeks=k, bin_index=b, raw=raw, z=zz))
pd.DataFrame(rows).to_csv(t1, index=False)
f = pd.read_parquet(SHARED / "SMB_tariff/intermediate_data/prep_reg_L2_update/smb_firm_features_for_L2_260626.parquet",
                    columns=["final_dunsno", "v2_dime_party_strict", "v2_dime_party_broad"])
f["final_dunsno"] = f.final_dunsno.astype("int64")
from pathlib import Path  # noqa: E402
c = pd.read_parquet(Path.home() / "private_cache/customs40_261102/firms.parquet")[["final_dunsno"]]
m = c.merge(f, on="final_dunsno", how="inner")
m["rep_strict"] = m.v2_dime_party_strict.map({"R": 1.0, "D": 0.0})
m["rep_broad"] = m.v2_dime_party_broad.map({"R": 1.0, "D": 0.0})
m = m[m.rep_strict.notna() | m.rep_broad.notna()][["final_dunsno", "rep_strict", "rep_broad"]]
m.to_parquet(t2)
print(len(m), m.rep_strict.value_counts(dropna=False).to_dict(), m.rep_broad.value_counts(dropna=False).to_dict())
