# ============================================================
# START DATE   : 2026-11-02
# CODE RUN DATE: 2026-11-02
# TIME TAKEN   : ~15 min
# DESCRIPTION  : Build the firm x origin x two-week panel (30 largest origins + Other, firms active in 2023, bins 2024-01..2026-03)
#                and the country x bin series of statement signals (lags 2..16 weeks) and rates in force. Firm rows stay in PRIVATE_CACHE
# INPUT FILES  :PRIVATE_DATA_ROOT/SMB_tariff/intermediate_data/prep_reg_L2_update/smb_trade_duns_biweekly_260512.parquet,
#                .../china_sentiment/smb_china_sentiment_tariff_for_L2_261008.csv, data/31_tariff_calendar_fix_261101/02_panel_B_extended.csv,
#                02_increase_events_B_extended.csv, data/16_tariff_schedule_261029/china_additional_duty_steps_261029.csv,
#                analysis/49_statement_type_261102/03_codes_261102.csv, PRIVATE_CACHE
# OUTPUT FILES : PRIVATE_CACHE{cells,firms,sig_long}.parquet ; analysis/50_*/01_*.csv (aggregates only)
# ============================================================
import sys
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve(); TOPIC = HERE.parents[2]
SHARED = HERE.parents[7] / "PRIVATE_DATA_ROOT"
L = SHARED / "SMB_tariff/intermediate_data/prep_reg_L2_update"
CACHE = Path.home() / "private_cache/customs50_261102"; OUT = TOPIC / "analysis/50_customs_country_signal_261102"
CACHE.mkdir(parents=True, exist_ok=True)
for t in [CACHE / "cells.parquet", CACHE / "firms.parquet", CACHE / "sig_long.parquet", OUT / "01_sample_counts_261102.csv", OUT / "01_origin_map_261102.csv"]:
    if t.exists(): sys.exit(f"output exists; refusing to overwrite: {t}")

BIN_LO, BIN_HI = 53, 111
cal = pd.read_csv(L / "china_sentiment/smb_china_sentiment_tariff_for_L2_261008.csv", usecols=["year_biweek", "bin_index", "bin_start", "bin_end", "partial_bin"])
cal["bin_start"] = pd.to_datetime(cal.bin_start); cal["bin_end"] = pd.to_datetime(cal.bin_end)
bins = cal[cal.bin_index.between(BIN_LO, BIN_HI) & ~cal.partial_bin].copy()   # drops stub bin 79 (2024-12-30/31)

p = pd.read_parquet(L / "smb_trade_duns_biweekly_260512.parquet", columns=["final_dunsno", "origin", "year_biweek", "trade_value"])
p = p[p.origin.notna() & p.origin.ne("United States")].copy()
p["origin"] = p.origin.replace({"Hong Kong": "China", "Macau": "China", "Macao": "China"})
p = p.merge(cal[["year_biweek", "bin_index"]], on="year_biweek", how="left")
yr = p.year_biweek.str[:4].astype(int)
# ★ CORE LOGIC: cohort = firms with a positive arrival in 2023 only (selection never uses 2024+ behavior or China status).
act23 = p[(yr == 2023) & (p.trade_value > 0)]
firms = pd.DataFrame({"final_dunsno": act23.final_dunsno.unique()})
firms["n_orig23"] = firms.final_dunsno.map(act23.groupby("final_dunsno").origin.nunique())
firms["alt23"] = firms.final_dunsno.map(act23.assign(nc=act23.origin.ne("China")).groupby("final_dunsno").nc.max()).astype(float)
firms["china23"] = firms.final_dunsno.isin(act23[act23.origin == "China"].final_dunsno).astype(float)
at = pd.read_parquet(Path.home() / "private_cache/customs40_261102/firm_attrs.parquet")[["final_dunsno", "emp2024", "emp_bin"]]
firms = firms.merge(at, on="final_dunsno", how="left")
x = p[p.final_dunsno.isin(firms.final_dunsno) & p.bin_index.isin(bins.bin_index) & (p.trade_value >= 0)]
tot = x.groupby("origin").trade_value.sum().sort_values(ascending=False)
top = [o for o in tot.index[:30]]
x = x.assign(origin=np.where(x.origin.isin(top), x.origin, "Other"))
cells = x.groupby(["final_dunsno", "origin", "bin_index"], as_index=False).trade_value.sum()
cells.to_parquet(CACHE / "cells.parquet"); firms.to_parquet(CACHE / "firms.parquet")

# ---- origin -> jurisdiction (iso3 in the statement panel) ----
EU = ["Austria", "Belgium", "Bulgaria", "Croatia", "Cyprus", "Czech Republic", "Czechia", "Denmark", "Estonia", "Finland", "France", "Germany", "Greece",
      "Hungary", "Ireland", "Italy", "Latvia", "Lithuania", "Luxembourg", "Malta", "Netherlands", "Poland", "Portugal", "Romania", "Slovakia", "Slovenia", "Spain", "Sweden"]
pan = pd.read_csv(TOPIC / "data/31_tariff_calendar_fix_261101/02_panel_B_extended.csv", parse_dates=["week"])
nm = pan[["iso3", "name"]].drop_duplicates().set_index("name").iso3.to_dict()
ALIAS = {"China": "CHN", "Turkey": "TUR", "Philippines": "PHL", "Burma": "MMR", "Myanmar": "MMR", "Korea, South": "KOR", "Brunei Darussalam": "BRN"}
def iso(o):
    if o in EU: return "EUN"
    return ALIAS.get(o) or nm.get(o)
omap = pd.DataFrame({"origin": top + ["Other"], "trade_value_2024_26": [tot[o] for o in top] + [np.nan]})
omap["iso3"] = omap.origin.map(lambda o: iso(o) if o != "Other" else None)
omap.to_csv(OUT / "01_origin_map_261102.csv", index=False)
print(omap.to_string())

# ---- statement signals, weekly Fridays ----
fr = pd.date_range("2024-09-01", "2026-04-30", freq="W-FRI")   # extends before the panel; pre-panel weeks are 0 by assumption
isos = sorted(set(omap.iso3.dropna()))
base = pd.MultiIndex.from_product([isos, fr], names=["iso3", "week"]).to_frame(index=False)
base = base.merge(pan[["iso3", "week", "mentions", "jev", "threats4"]], on=["iso3", "week"], how="left")
base[["mentions", "jev", "threats4"]] = base[["mentions", "jev", "threats4"]].fillna(0.0)
# typed passages with a country target, trailing 28 days (analysis/49)
cd = pd.read_csv(TOPIC / "analysis/49_statement_type_261102/03_codes_261102.csv", parse_dates=["date"])
cd = cd[cd.statement_type.isin(["threat_or_possible_future", "stated_intention_future"])]
tmap = {"china": "CHN", "country:china": "CHN", "country:european union": "EUN", "country:eu": "EUN", "country:europe": "EUN", "country:france": "EUN",
        "country:germany": "EUN", "country:spain": "EUN", "country:italy": "EUN", "country:ireland": "EUN"}
for o, i3 in zip(omap.origin, omap.iso3):
    if i3 and o not in EU and o != "China": tmap["country:" + o.lower()] = i3
rows = []
for d, t in zip(cd.date, cd.target.fillna("")):
    for tg in set(s.strip().lower() for s in t.split(";")):
        if tg in tmap: rows.append((tmap[tg], d))
ty = pd.DataFrame(rows, columns=["iso3", "date"])
def typed(i3, w):
    s = ty[(ty.iso3 == i3) & (ty.date > w - pd.Timedelta(days=28)) & (ty.date <= w)]
    return float(len(s))
base["typed4"] = [typed(i, w) for i, w in zip(base.iso3, base.week)]

# followed within 90 days by an all-goods increase (country-level effective dates)
ev = pd.read_csv(TOPIC / "data/31_tariff_calendar_fix_261101/02_increase_events_B_extended.csv", parse_dates=["eff"])
steps = pd.read_csv(TOPIC / "data/16_tariff_schedule_261029/china_additional_duty_steps_261029.csv", parse_dates=["effective_date"])
ag = ev[(ev["product_scope"] == "all goods") & (ev.legal_basis == "IEEPA") & ~ev.cancelled.astype(bool) & ~ev.short_lived.astype(bool)]
inc = {i: sorted(g.eff.unique()) for i, g in ag.groupby("iso3")}
inc["CHN"] = sorted(steps[steps.change_points > 0].effective_date.unique())
def followed(i3, w):
    return float(any((e > w) and (e <= w + pd.Timedelta(days=90)) for e in inc.get(i3, [])))
base["fol"] = [followed(i, w) for i, w in zip(base.iso3, base.week)]
base["thr_f"] = base.threats4 * base.fol; base["thr_u"] = base.threats4 * (1 - base.fol)

# rate in force (daily step functions)
days = pd.date_range("2024-01-01", "2026-04-30")
def rate_series(i3):
    r = pd.Series(0.0, index=days); ind = pd.Series(0.0, index=days)
    if i3 == "CHN":
        for _, s in steps.sort_values("effective_date").iterrows():
            r[r.index >= s.effective_date] = s.rate_percent
        ind = (r > 0).astype(float)
    else:
        g = ag[ag.iso3 == i3].sort_values("eff")
        for _, s in g.iterrows():
            ind[ind.index >= s.eff] = 1.0
            r[r.index >= s.eff] = s.new_rate if pd.notna(s.new_rate) else np.nan
    # Section 122 surcharge after IEEPA termination: approximation for all origins
    r[r.index >= "2026-02-24"] = 10.0 if i3 != "CHN" else r[r.index >= "2026-02-24"]; ind[ind.index >= "2026-02-24"] = 1.0
    return r / 10.0, ind
RS = {i: rate_series(i) for i in isos}

K = [0, 2, 4, 6, 8, 12, 16]
rows = []
for i in isos:
    b = base[base.iso3 == i].set_index("week")
    r, ind = RS[i]
    for _, bb in bins.iterrows():
        rec = dict(iso3=i, bin_index=bb.bin_index)
        for k in K:
            ref = bb.bin_start - pd.Timedelta(days=7 * k + 1)
            w = b.index[b.index <= ref]
            wk = w[-1] if len(w) else None
            for v in ["mentions", "jev", "threats4", "typed4", "thr_f", "thr_u"]:
                rec[f"{v}_k{k}"] = float(b.loc[wk, v]) if wk is not None else 0.0
            dd = bb.bin_start - pd.Timedelta(days=7 * k)   # R measured at the bin start shifted back k weeks
            dd = max(dd, days[0])
            rec[f"Rnum_k{k}"] = r.loc[dd]; rec[f"Rind_k{k}"] = ind.loc[dd]
        # pre-panel flag: any lag reaching before the statement panel starts
        rec["lag16_in_panel"] = float(bb.bin_start - pd.Timedelta(days=7 * 16 + 1) >= pd.Timestamp("2025-01-24"))
        rows.append(rec)
sig = pd.DataFrame(rows)
sig.to_parquet(CACHE / "sig_long.parquet")
cnt = {"firms_active_2023": len(firms), "firms_with_emp": int(firms.emp2024.notna().sum()), "cells": len(cells), "origins": len(top) + 1,
       "iso3_origins_mapped": len(isos), "bins": len(bins), "unmapped_origins": ",".join(omap[omap.iso3.isna() & (omap.origin != "Other")].origin)}
pd.Series(cnt).to_frame("value").to_csv(OUT / "01_sample_counts_261102.csv"); print(cnt)
