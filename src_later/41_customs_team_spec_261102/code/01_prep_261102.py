# ============================================================
# START DATE   : 2026-11-02
# CODE RUN DATE: 2026-11-02
# TIME TAKEN   : ~10 min
# DESCRIPTION  : Prepare the firm x origin-group x two-week panel (2024-01 onward, firms with any China-area arrival
#                in 2023-2024) and the period-level series (China duty in force, Jev China-path and broad-path
#                signal incl. week shifts, explicit threat counts). Firm rows stay in PRIVATE_CACHE only.
# INPUT FILES  :PRIVATE_DATA_ROOT/SMB_tariff/intermediate_data/prep_reg_L2_update/smb_trade_duns_biweekly_260512.parquet,
#                .../china_sentiment/smb_china_sentiment_tariff_for_L2_261008.csv (bin calendar),
PRIVATE_DATA_ROOT/TEAM_PRIVATE/ideology_share/directional_coding/data/jev_corpus/
#                  simple_time_series_v1/event_question_option_probabilities.parquet,
#                data/16_tariff_schedule_261029/china_additional_duty_steps_261029.csv,
#                data/31_tariff_calendar_fix_261101/02_panel_B_extended.csv
# OUTPUT FILES : PRIVATE_CACHE{cells,firms}.parquet (firm level, NOT in project)
#                analysis/41_customs_team_spec_261102/01_period_series_261102.csv, 01_signal_shifts_261102.csv,
#                01_sample_counts_261102.csv, 01_origin_groups_261102.csv
# ============================================================
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve()
TOPIC = HERE.parents[2]
SHARED = HERE.parents[7] / "PRIVATE_DATA_ROOT"
L = SHARED / "SMB_tariff/intermediate_data/prep_reg_L2_update"
TS = (SHARED / "TEAM_PRIVATE/ideology_share/directional_coding/data/jev_corpus/"
      "simple_time_series_v1/event_question_option_probabilities.parquet")
DUTY = TOPIC / "data/16_tariff_schedule_261029/china_additional_duty_steps_261029.csv"
THR = TOPIC / "data/31_tariff_calendar_fix_261101/02_panel_B_extended.csv"
CACHE = Path.home() / "private_cache/customs40_261102"
OUT = TOPIC / "analysis/41_customs_team_spec_261102"
CACHE.mkdir(parents=True, exist_ok=True)
targets = [CACHE / "cells.parquet", CACHE / "firms.parquet", OUT / "01_period_series_261102.csv",
           OUT / "01_signal_shifts_261102.csv", OUT / "01_sample_counts_261102.csv", OUT / "01_origin_groups_261102.csv"]
for t in targets:
    if t.exists():
        sys.exit(f"output exists; refusing to overwrite: {t}")

BIN_LO, BIN_HI = 53, 111          # 2024-01 ... 2026-06 (ends 2026-03-29); bin 112 is partial and excluded
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
# bin 79 (2024-12-30 and 2024-12-31) is a two-day stub at the year end and is dropped; the last bin is complete
bins = cal[cal.bin_index.between(BIN_LO, BIN_HI) & ~cal.partial_bin].copy()
assert len(bins) == BIN_HI - BIN_LO + 1 - 1 and bins.bin_index.max() == BIN_HI

# ---------------- firm x origin panel ----------------
p = pd.read_parquet(L / "smb_trade_duns_biweekly_260512.parquet",
                    columns=["final_dunsno", "origin", "year_biweek", "trade_value", "weightkg", "volumeteu", "pre_trade_total"])
p = p[p.origin.notna() & p.origin.ne("United States")].copy()
p["grp"] = p.origin.map(GRP).fillna("rest")
p = p.merge(cal[["year_biweek", "bin_index"]], on="year_biweek", how="left")
n_all_firms = p.final_dunsno.nunique()

# ★ CORE LOGIC: cohort fixed from 2023-2024 China-area arrivals only (before the signal series matter), so later
# behavior cannot select firms. 2023-01 ... 2024-26 = bin_index 27..104? computed from labels below, not hard-coded.
lab = p.year_biweek.str[:4].astype(int)
cn_pre = p[(lab.isin([2023, 2024])) & (p.grp == "china") & (p.weightkg > 0)].final_dunsno.unique()
n_cn_pre = len(cn_pre)
pre_tot = p.groupby("final_dunsno").pre_trade_total.first()
firms = pd.DataFrame({"final_dunsno": cn_pre})
firms["pre_trade_total"] = firms.final_dunsno.map(pre_tot)
n_unknown = int((firms.pre_trade_total <= 0).sum())
firms = firms[firms.pre_trade_total > 0].copy()
# firm size = terciles of pre-period trade (2022-01 ... 2024-06, the team's definition), cut within this sample
firms["tercile"] = pd.qcut(firms.pre_trade_total.rank(method="first"), 3, labels=["small", "medium", "large"]).astype(str)
cuts = firms.groupby("tercile").pre_trade_total.agg(["count", "min", "max"])
x = p[p.final_dunsno.isin(firms.final_dunsno) & p.bin_index.isin(bins.bin_index)]
cells = x.groupby(["final_dunsno", "grp", "bin_index"], as_index=False).agg(
    trade_value=("trade_value", "sum"), weightkg=("weightkg", "sum"), volumeteu=("volumeteu", "sum"))
cells.to_parquet(CACHE / "cells.parquet"); firms.to_parquet(CACHE / "firms.parquet")

# origin-group audit (names only, aggregates only)
og = p[p.bin_index.isin(bins.bin_index)].groupby(["grp", "origin"]).trade_value.sum().reset_index()
og["share_of_value"] = og.trade_value / og.trade_value.sum()
og = og.sort_values(["grp", "trade_value"], ascending=[True, False]); og.to_csv(OUT / "01_origin_groups_261102.csv", index=False)

# ---------------- period-level series ----------------
bins["mid"] = bins.bin_start + pd.Timedelta(days=7)
days = pd.date_range("2023-06-01", "2026-04-30")
steps = pd.read_csv(DUTY, usecols=["effective_date", "rate_percent"]); steps["effective_date"] = pd.to_datetime(steps.effective_date)
rate = pd.Series(0.0, index=days)
for _, r in steps.sort_values("effective_date").iterrows():
    rate[rate.index >= r.effective_date] = r.rate_percent
# ★ IMPORTANT: Duty_t is the day-weighted mean China additional duty in force during the two-week period (percent / 10).
bins["duty10"] = [rate[(rate.index >= a) & (rate.index <= b)].mean() / 10 for a, b in zip(bins.bin_start, bins.bin_end)]

thr = pd.read_csv(THR, usecols=["iso3", "week", "threats4"]); thr = thr[thr.iso3 == "CHN"].copy(); thr["week"] = pd.to_datetime(thr.week)
thr = thr.sort_values("week")
# threats4 at Friday t = explicit threat rows for the country in (t-28d, t] (topic6 code 04_build_panel line 120);
# use the last Friday on or before the day before the period starts, so the count precedes the period.
bins["threat4"] = [float(thr.loc[thr.week <= s - pd.Timedelta(days=1), "threats4"].iloc[-1]) if (thr.week <= s - pd.Timedelta(days=1)).any()
                   and thr.week.min() <= s - pd.Timedelta(days=1) else np.nan for s in bins.bin_start]

ev = pd.read_parquet(TS); ev["event_date"] = pd.to_datetime(ev["event_date"])
SIG = {"china": ("tariff_china_path", ["higher"], ["lower"]), "broad": ("tariff_broad_path", ["higher"], ["lower"])}


def parts(q, pos, neg):
    # same construction as code/27_expectation_action_261101/02_robustness_261101.py lines 115-141
    x = ev[ev.question_id == q]
    w = x.pivot_table(index=["event_question_id", "event_date"], columns="option", values="raw_option_probability",
                      aggfunc="sum").fillna(0.0).reset_index()
    w["ev"] = 1.0 - w[x[x.is_absence_option].option.unique()[0]]
    w["num"] = w[pos].sum(axis=1) - w[neg].sum(axis=1)
    return w.groupby("event_date")[["num", "ev"]].sum()


def signal(pp, dates, shift_days=0):
    out = {}
    for t in dates:
        a_, b_ = t + pd.Timedelta(days=shift_days - 28), t + pd.Timedelta(days=shift_days - 1)
        s = pp.loc[(pp.index >= a_) & (pp.index <= b_)]
        out[t] = s.num.sum() / s.ev.sum() if s.ev.sum() > 0 else np.nan
    return pd.Series(out)


# ★ CORE LOGIC: signal for period starting s = direction share of the leader's statements in the 28 days before s
# (strictly earlier than the period). A shift of k weeks moves that window by 7k days; the series keeps its
# autocorrelation but loses its timing link to arrivals (placebo).
SHIFT_WEEKS = [0] + [k for k in range(-52, 53) if abs(k) >= 5]
rows = []
for name, (q, pos, neg) in SIG.items():
    pp = parts(q, pos, neg)
    for k in SHIFT_WEEKS:
        s = signal(pp, bins.bin_start, 7 * k)
        z = (s - s.mean()) / s.std()
        for b, raw, zz in zip(bins.bin_index, s.values, z.values):
            rows.append(dict(signal=name, shift_weeks=k, bin_index=b, raw=raw, z=zz))
sh = pd.DataFrame(rows); sh.to_csv(OUT / "01_signal_shifts_261102.csv", index=False)
for name in SIG:
    s0 = sh[(sh.signal == name) & (sh.shift_weeks == 0)].set_index("bin_index")
    bins["sig_" + name] = bins.bin_index.map(s0.z); bins["sig_" + name + "_raw"] = bins.bin_index.map(s0.raw)
bins["threat4_z"] = (bins.threat4 - bins.threat4.mean()) / bins.threat4.std()
bins.to_csv(OUT / "01_period_series_261102.csv", index=False)

cnt = {"firms_in_panel_file": n_all_firms, "firms_with_china_area_arrival_2023_2024": n_cn_pre,
       "dropped_unknown_size_pre_trade_total_0": n_unknown, "firms_in_sample": len(firms),
       "positive_cells_firm_group_bin": len(cells), "bins": len(bins),
       "bins_signal_china_defined": int(bins.sig_china.notna().sum()), "bins_signal_broad_defined": int(bins.sig_broad.notna().sum()),
       "bins_threat_defined": int(bins.threat4.notna().sum())}
for t, r in cuts.iterrows():
    cnt[f"tercile_{t}_firms"] = int(r["count"]); cnt[f"tercile_{t}_pre_trade_min"] = float(r["min"]); cnt[f"tercile_{t}_pre_trade_max"] = float(r["max"])
for g, n in cells.groupby("grp").size().items(): cnt[f"positive_cells_{g}"] = int(n)
pd.Series(cnt).to_frame("value").to_csv(OUT / "01_sample_counts_261102.csv")
pd.set_option("display.width", 220, "display.max_columns", 30)
print(pd.Series(cnt).to_string())
print(bins[["bin_index", "bin_start", "duty10", "threat4", "sig_china", "sig_broad"]].round(2).to_string(index=False))
print(og.groupby("grp").apply(lambda g: g.head(6)[["origin", "share_of_value"]].round(3).values.tolist()).to_string())
