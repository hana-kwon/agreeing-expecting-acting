# ============================================================
# START DATE   : 2026-11-02
# CODE RUN DATE: 2026-11-02
# TIME TAKEN   : ~5 min
# DESCRIPTION  : Prepare (a) the period table for 2023-01..2026-03 with the China additional duty in force at arrival and
#                lagged one and two periods, (b) the fake-year window table for the placebo, and (c) the firm x origin-group x
#                two-week cells for 2023-2024 (placebo sample = 2023-2024; same firm cohort and size terciles as code/41).
#                Firm rows stay in PRIVATE_CACHE only.
# INPUT FILES  :PRIVATE_DATA_ROOT/SMB_tariff/intermediate_data/prep_reg_L2_update/smb_trade_duns_biweekly_260512.parquet (read only),
#                .../china_sentiment/smb_china_sentiment_tariff_for_L2_261008.csv (bin calendar),
#                data/16_tariff_schedule_261029/china_additional_duty_steps_261029.csv,
#                PRIVATE_CACHE{cells,firms}.parquet (from code/41/01_prep_261102.py)
# OUTPUT FILES : PRIVATE_CACHE (firm level, NOT in project),
#                analysis/44_customs_time_path_261102/01_period_table_261102.csv, 01_placebo_windows_261102.csv,
#                01_placebo_sample_counts_261102.csv
# ============================================================
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve()
TOPIC = HERE.parents[2]
SHARED = HERE.parents[7] / "PRIVATE_DATA_ROOT"
L = SHARED / "SMB_tariff/intermediate_data/prep_reg_L2_update"
DUTY = TOPIC / "data/16_tariff_schedule_261029/china_additional_duty_steps_261029.csv"
CACHE = Path.home() / "private_cache/customs40_261102"
OUT = TOPIC / "analysis/44_customs_time_path_261102"
targets = [CACHE / "cells_placebo_261102.parquet", OUT / "01_period_table_261102.csv", OUT / "01_placebo_windows_261102.csv",
           OUT / "01_placebo_sample_counts_261102.csv"]
for t in targets:
    if t.exists():
        sys.exit(f"output exists; refusing to overwrite: {t}")
assert str(SHARED).count("#Shared_") == 1 and SHARED.exists()

# group mapping: identical to code/41/01_prep_261102.py (copied, not imported, because that file runs on import)
CN = ["China", "Hong Kong", "Macau", "Macao"]
EA = ["Japan", "South Korea", "Taiwan", "Vietnam", "Thailand", "Malaysia", "Indonesia", "Philippines", "Singapore",
      "Cambodia", "Laos", "Myanmar", "Brunei", "Brunei Darussalam", "Mongolia", "North Korea"]
EU = ["Austria", "Belgium", "Bulgaria", "Croatia", "Cyprus", "Czech Republic", "Czechia", "Denmark", "Estonia", "Finland",
      "France", "Germany", "Greece", "Hungary", "Ireland", "Italy", "Latvia", "Lithuania", "Luxembourg", "Malta",
      "Netherlands", "Poland", "Portugal", "Romania", "Slovakia", "Slovenia", "Spain", "Sweden"]
GRP = {**{o: "china" for o in CN}, **{o: "east_asia" for o in EA}, **{o: "eu" for o in EU}, "Mexico": "mexico", "Canada": "canada"}

cal = pd.read_csv(L / "china_sentiment/smb_china_sentiment_tariff_for_L2_261008.csv",
                  usecols=["year_biweek", "bin_index", "bin_start", "bin_end", "partial_bin"])
cal["bin_start"] = pd.to_datetime(cal.bin_start); cal["bin_end"] = pd.to_datetime(cal.bin_end)
# bins 27 (2023-01) ... 111 (2026-03-29); partial bins (79 = 2024-12-30/31 stub, 112) are dropped
bins = cal[cal.bin_index.between(27, 111) & ~cal.partial_bin].copy().reset_index(drop=True)
assert len(bins) == 84, len(bins)
bins["mid"] = bins.bin_start + pd.Timedelta(days=7)
bins["year"] = bins.year_biweek.str[:4].astype(int)
bins["poy"] = bins.year_biweek.str[5:].astype(int)

# ★ IMPORTANT: Duty is the day-weighted mean China additional duty in force over the two-week period (percent / 10). Lag L uses the
# same-length calendar window moved back 14*L days, so lag 1 is the previous two-week period even where bin lengths differ.
# Before 2025-01-20 the file defines the series as 0 (pre-2025 regime not counted).
days = pd.date_range("2022-10-01", "2026-04-30")
steps = pd.read_csv(DUTY, usecols=["effective_date", "rate_percent"]); steps["effective_date"] = pd.to_datetime(steps.effective_date)
rate = pd.Series(0.0, index=days)
for _, r in steps.sort_values("effective_date").iterrows():
    rate[rate.index >= r.effective_date] = r.rate_percent
for lag in (0, 1, 2):
    sh = pd.Timedelta(days=14 * lag)
    bins[f"duty10_lag{lag}"] = [rate[(rate.index >= a - sh) & (rate.index <= b - sh)].mean() / 10 for a, b in zip(bins.bin_start, bins.bin_end)]
bins.to_csv(OUT / "01_period_table_261102.csv", index=False)

# ---- fake-year windows for the placebo (P3). Actual rule (code/41/02_estimate): frontload = months 1-4, threat = months 5-6,
# resolved = month 7 through month 15 after the frontload start (2025-07..2026-03); bin month = month of the midpoint.
# Placebo: frontload start S_k = 2024-01-01 + 7k days (2024 = fake year, 2023 and unwindowed periods are the reference).
rows = []
ks = [0] + [k for k in range(-26, 27) if abs(k) >= 8]
pb = bins[bins.bin_index <= 78]
for k in ks:
    S = pd.Timestamp("2024-01-01") + pd.Timedelta(days=7 * k)
    e1, e2, e3 = S + pd.DateOffset(months=4), S + pd.DateOffset(months=6), S + pd.DateOffset(months=15)
    inF = (pb.mid >= S) & (pb.mid < e1); inT = (pb.mid >= e1) & (pb.mid < e2); inR = (pb.mid >= e2) & (pb.mid < e3)
    rows.append(dict(shift_weeks=k, S=S.date(), thr_start=e1.date(), res_start=e2.date(), res_end=e3.date(),
                     n_front=int(inF.sum()), n_thr=int(inT.sum()), n_res=int(inR.sum()), n_ref=int((~(inF | inT | inR)).sum()),
                     first_bin_F=int(pb.bin_index[inF].min()) if inF.any() else -1))
pw = pd.DataFrame(rows)
# estimable: resolved >= 6 bins, threat >= 3 bins, frontload >= 3 bins, reference >= 6 bins (fixed in the spec before estimation)
pw["estimable"] = (pw.n_res >= 6) & (pw.n_thr >= 3) & (pw.n_front >= 3) & (pw.n_ref >= 6)
pw.to_csv(OUT / "01_placebo_windows_261102.csv", index=False)

# ---- firm x group x bin cells for 2023 (bins 27..52), appended to the 2024 cells (bins 53..78) from code/41
firms = pd.read_parquet(CACHE / "firms.parquet")
p = pd.read_parquet(L / "smb_trade_duns_biweekly_260512.parquet",
                    columns=["final_dunsno", "origin", "year_biweek", "trade_value", "weightkg", "volumeteu"],
                    filters=[("year_biweek", ">=", "2023-01"), ("year_biweek", "<=", "2023-26")])
p = p[p.final_dunsno.isin(firms.final_dunsno) & p.origin.notna() & p.origin.ne("United States")].copy()
p["grp"] = p.origin.map(GRP).fillna("rest")
p = p.merge(cal[["year_biweek", "bin_index"]], on="year_biweek", how="left")
assert p.bin_index.between(27, 52).all()
c23 = p.groupby(["final_dunsno", "grp", "bin_index"], as_index=False).agg(
    trade_value=("trade_value", "sum"), weightkg=("weightkg", "sum"), volumeteu=("volumeteu", "sum"))
c24 = pd.read_parquet(CACHE / "cells.parquet"); c24 = c24[c24.bin_index <= 78]
cp = pd.concat([c23, c24], ignore_index=True)
assert cp.duplicated(["final_dunsno", "grp", "bin_index"]).sum() == 0
cp.to_parquet(CACHE / "cells_placebo_261102.parquet")
cnt = pd.Series({"firms_cohort": len(firms), "cells_2023": len(c23), "cells_2024": len(c24), "cells_placebo_total": len(cp),
                 "firms_with_2023_cells": c23.final_dunsno.nunique(), "bins_placebo_sample": cp.bin_index.nunique()})
cnt.to_frame("value").to_csv(OUT / "01_placebo_sample_counts_261102.csv")
pd.set_option("display.width", 220, "display.max_columns", 30)
print(cnt.to_string()); print(pw.to_string(index=False))
print(bins[["bin_index", "year_biweek", "bin_start", "duty10_lag0", "duty10_lag1", "duty10_lag2"]].iloc[50:].round(2).to_string(index=False))
