# ============================================================
# START DATE   : 2026-11-01
# CODE RUN DATE: 2026-11-01
# TIME TAKEN   : ~5 min
# DESCRIPTION  : Tests whether imports fall from regions the leader says to leave (Jev weekly sourcing
#                positions), and whether Republican-donor importers respond more (C1, C2 in 00_spec_261101.md).
# INPUT FILES  : $RESTRICTED_DATA_DIR/customs/smb_trade_duns_biweekly_260512.parquet,
#                .../smb_firm_features_for_L2_260626.parquet, .../china_sentiment/smb_china_sentiment_tariff_for_L2_261008.csv,
#                $RESTRICTED_DATA_DIR/survey/trump_alignment_v2/trump_sourcing_positions_weekly.csv,
#                $RESTRICTED_DATA_DIR/customs/hs6_panel/smb_tariff_h0_origin_month_261026.parquet
# OUTPUT FILES : analysis/27_expectation_action_261101/04_customs_results_261101.csv,
#                04_region_biweek_261101.csv (aggregates only; firm rows stay in memory)
# ============================================================
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from restricted_paths import SURVEY_DIR, CUSTOMS_DIR, JEV_DIR, OUTPUT_ROOT, out_dir, fresh_out_dir

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve()
L = CUSTOMS_DIR
OUT = out_dir("survey")
if (OUT / "04_customs_results_261101.csv").exists():
    sys.exit("output exists; refusing to overwrite.")

EU = ["Austria", "Belgium", "Bulgaria", "Croatia", "Cyprus", "Czech Republic", "Czechia", "Denmark", "Estonia", "Finland", "France",
      "Germany", "Greece", "Hungary", "Ireland", "Italy", "Latvia", "Lithuania", "Luxembourg", "Malta", "Netherlands", "Poland",
      "Portugal", "Romania", "Slovakia", "Slovenia", "Spain", "Sweden"]
EA = ["Japan", "South Korea", "Taiwan", "Hong Kong", "Vietnam", "Thailand", "Malaysia", "Indonesia", "Philippines", "Singapore",
      "Cambodia", "Laos", "Myanmar", "Brunei", "Mongolia", "Macau", "Macao", "North Korea"]
ISO_EU = "AUT BEL BGR HRV CYP CZE DNK EST FIN FRA DEU GRC HUN IRL ITA LVA LTU LUX MLT NLD POL PRT ROU SVK SVN ESP SWE".split()
ISO_EA = "JPN KOR TWN HKG VNM THA MYS IDN PHL SGP KHM LAO MMR BRN MNG MAC PRK".split()


def region(o):
    if o == "China": return "china"
    if o == "Canada": return "canada"
    if o == "Mexico": return "mexico"
    if o in EU: return "eu"
    if o in EA: return "east_asia_ex_china"
    if o == "United States": return None
    return "other_foreign"


def region_iso(o):
    if o == "CHN": return "china"
    if o == "CAN": return "canada"
    if o == "MEX": return "mexico"
    if o in ISO_EU: return "eu"
    if o in ISO_EA: return "east_asia_ex_china"
    if o == "USA": return None
    return "other_foreign"


cal = pd.read_csv(L / "china_sentiment/smb_china_sentiment_tariff_for_L2_261008.csv", usecols=["year_biweek", "bin_index", "bin_start", "bin_end", "partial_bin"])
cal["bin_start"] = pd.to_datetime(cal.bin_start)
cal["bin_end"] = pd.to_datetime(cal.bin_end)
cal = cal.sort_values("bin_index").reset_index(drop=True)
print("partial bins:", cal[cal.partial_bin == True].year_biweek.tolist(), file=sys.stderr)

# Jev weekly positions -> biweek means
w = pd.read_csv(SURVEY_DIR / "trump_alignment_v2/trump_sourcing_positions_weekly.csv", parse_dates=["week_end"])
REG = ["canada", "mexico", "east_asia_ex_china", "china", "eu", "other_foreign"]
rows = []
for _, b in cal.iterrows():
    s = w[(w.week_end >= b.bin_start) & (w.week_end <= b.bin_end)]
    for r_ in REG:
        rows.append((b.bin_index, r_, s[f"trump_away_{r_}"].mean()))
A = pd.DataFrame(rows, columns=["bin_index", "region", "A"])

# imports (memory only)
p = pd.read_parquet(L / "smb_trade_duns_biweekly_260512.parquet", columns=["final_dunsno", "origin", "year_biweek", "trade_value"])
p["region"] = p.origin.map(region)
p = p.dropna(subset=["region"]).merge(cal[["year_biweek", "bin_index", "partial_bin"]], on="year_biweek")
p = p[p.partial_bin != True]
f = pd.read_parquet(L / "smb_firm_features_for_L2_260626.parquet", columns=["final_dunsno", "v2_dime_party_strict", "dime_party", "dime_covered"])
print(f.v2_dime_party_strict.value_counts(dropna=False).head(6).to_dict(), f.dime_party.value_counts(dropna=False).head(6).to_dict(), file=sys.stderr)
p = p.merge(f, on="final_dunsno", how="left")
tot = p.trade_value.sum()
print("value share with party label:", p[p.v2_dime_party_strict.notna()].trade_value.sum() / tot, file=sys.stderr)

# tariffs in force: region x month, simple mean over h0 and origins of summed components
t = pd.read_parquet(CUSTOMS_DIR / "hs6_panel/smb_tariff_h0_origin_month_261026.parquet", columns=["h0", "origin_iso3", "month", "component", "rate_pct"])
t = t.groupby(["h0", "origin_iso3", "month"], observed=True).rate_pct.sum().reset_index()
t["region"] = t.origin_iso3.astype(str).map(region_iso)
t = t.dropna(subset=["region"]).groupby(["region", "month"]).rate_pct.mean().reset_index()
t["month"] = pd.to_datetime(t.month.astype(str)).dt.to_period("M")
cal["month"] = cal.bin_start.dt.to_period("M")
tar = cal[["bin_index", "month"]].merge(t, on="month", how="left")[["bin_index", "region", "rate_pct"]]


def yoy(df, keys):
    df = df.sort_values(keys + ["bin_index"]).copy()
    full = df.set_index(keys + ["bin_index"]).ly
    prev = df.assign(bin_index=df.bin_index + 26).set_index(keys + ["bin_index"]).ly.rename("ly_prev")
    x = pd.concat([full, prev], axis=1).dropna().reset_index()
    x["y"] = x.ly - x.ly_prev
    return x


g = p.groupby(["region", "bin_index"]).trade_value.sum().reset_index()
g["ly"] = np.log(g.trade_value)
c1 = yoy(g[["region", "bin_index", "ly"]], ["region"]).merge(tar, on=["region", "bin_index"], how="left")
c1["rate_pct"] = c1.rate_pct.fillna(0.0)
c1["rate_prev"] = c1.merge(tar.assign(bin_index=tar.bin_index + 26), on=["region", "bin_index"], how="left", suffixes=("", "_p")).rate_pct_p.fillna(0.0).values
c1["d_rate"] = c1.rate_pct - c1.rate_prev
g.merge(A, on=["region", "bin_index"], how="left").to_csv(OUT / "04_region_biweek_261101.csv", index=False)


def attach(df, k):
    # ★ CORE LOGIC: signal from the preceding bin (k = 1 is the real lag); other k values are date-shift placebos.
    a = A.assign(bin_index=A.bin_index + k)
    return df.merge(a, on=["region", "bin_index"], how="left").dropna(subset=["A"])


def est_c1(df, k, regions):
    s = attach(df[df.region.isin(regions)], k)
    X = pd.concat([s[["A", "d_rate"]].astype(float), pd.get_dummies(s.bin_index, prefix="t", drop_first=True).astype(float)], axis=1)
    return sm.OLS(s.y.astype(float), sm.add_constant(X)).fit(), s


def est_c2(df, k, regions):
    s = attach(df[df.region.isin(regions)], k)
    s["rep_A"] = s.rep * s.A
    # within region x time and party x time, by partialling out dummies
    X = pd.concat([s[["rep_A"]].astype(float),
                   pd.get_dummies(s.region + "_" + s.bin_index.astype(str), prefix="rt", drop_first=True, sparse=False).astype(float),
                   pd.get_dummies(s.bin_index, prefix="t", drop_first=True).astype(float).mul(s.rep.values, axis=0)], axis=1)
    return sm.OLS(s.y.astype(float), sm.add_constant(X.loc[:, X.std() > 0])).fit(), s


party = p[p.v2_dime_party_strict.notna()].copy()
vals = party.v2_dime_party_strict.astype(str).str.lower()
party["rep"] = np.where(vals.str.startswith("r"), 1.0, np.where(vals.str.startswith("d"), 0.0, np.nan))
party = party.dropna(subset=["rep"])
h = party.groupby(["rep", "region", "bin_index"]).trade_value.sum().reset_index()
h = h[h.trade_value > 0]
h["ly"] = np.log(h.trade_value)
c2 = yoy(h[["rep", "region", "bin_index", "ly"]], ["rep", "region"])
print("firms R/D:", party.groupby("rep").final_dunsno.nunique().to_dict(), "value share R/D of all:", (party.groupby("rep").trade_value.sum() / tot).round(4).to_dict(), file=sys.stderr)

SHIFTS = [k for k in range(-26, 27) if abs(k - 1) >= 3]
out = []
for label, regions in [("all_regions", REG), ("no_canada_mexico", ["east_asia_ex_china", "china", "eu", "other_foreign"])]:
    for test, data, fn, var in [("C1", c1, est_c1, "A"), ("C2", c2, est_c2, "rep_A")]:
        m, s = fn(data, 1, regions)
        act = m.params[var]
        sh = []
        for k in SHIFTS:
            try:
                mm, ss = fn(data, k, regions)
                if len(ss) > 40:
                    sh.append(mm.params[var])
            except Exception:
                pass
        sh = np.array(sh)
        out.append(dict(test=test, regions=label, coef=act, ols_se=m.bse[var], n=int(m.nobs), bins=s.bin_index.nunique(),
                        first_bin=s.bin_index.min(), last_bin=s.bin_index.max(), shift_mean=sh.mean(), shift_sd=sh.std(),
                        shift_p=float((np.abs(sh) >= abs(act)).mean()), shift_p_negative=float((sh <= act).mean()), n_shifts=len(sh)))
        print(out[-1], file=sys.stderr, flush=True)
        if test == "C1":
            out[-1]["d_rate_coef"] = m.params["d_rate"]
pd.DataFrame(out).to_csv(OUT / "04_customs_results_261101.csv", index=False)
