# ============================================================
# START DATE   : 2026-11-02
# CODE RUN DATE: 2026-11-02
# TIME TAKEN   : ~10 min
# DESCRIPTION  : Chosen measure f3 (higher-direction future claims naming a country when no increase was yet in force there; spec choice 06:11 EST)
#                as passage list and as customs signal: per country x customs two-week bin, count of f3 passages in the 14 days ending
#                7k days + 1 before the bin start (k = 0,2,4,6,8,12,16). Windows that start before the first coded passage date are NaN (not zero).
# INPUT FILES  : analysis/49_statement_type_261102/03_codes_261102.csv; data/31 02_panel_B_extended.csv, 02_increase_events_B_extended.csv;
#                PRIVATE_CACHE (rate controls only);PRIVATE_DATA_ROOT SMB_tariff calendar csv (read only)
# OUTPUT FILES : analysis/53_first_stage_261102/05_f3_passages_261102.csv (country, date, aggregates only); PRIVATE_CACHE
# ============================================================
import re, sys
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve(); TOPIC = HERE.parents[2]; OUT = TOPIC / "analysis/53_first_stage_261102"
L = HERE.parents[7] / "PRIVATE_DATA_ROOT/SMB_tariff/intermediate_data/prep_reg_L2_update"
CACHE = Path.home() / "private_cache/customs53_261102"; CACHE.mkdir(parents=True, exist_ok=True)
for t in [OUT / "05_f3_passages_261102.csv", CACHE / "sig_f3.parquet"]:
    if t.exists(): sys.exit(f"output exists; refusing to overwrite: {t}")
pan = pd.read_csv(TOPIC / "data/31_tariff_calendar_fix_261101/02_panel_B_extended.csv"); names = pan.groupby("iso3").name.first().to_dict()
EU = {"european union", "eu", "europe", "european nations", "austria", "belgium", "bulgaria", "croatia", "cyprus", "czech republic", "czechia", "denmark", "estonia", "finland", "france", "germany", "greece", "hungary", "ireland", "italy", "latvia", "lithuania", "luxembourg", "malta", "netherlands", "poland", "portugal", "romania", "slovakia", "slovenia", "spain", "sweden"}
norm = lambda s: re.sub(r"\s+", " ", re.sub(r"\(.*?\)", "", s)).strip().lower().removeprefix("the ")
NM = {norm(n): i for i, n in names.items()}; NM.update({"china": "CHN", "uk": "GBR", "turkey": "TUR", "burma": "MMR", "philippines": "PHL", "bahamas": "BHS", "south korea": "KOR", "vietnam": "VNM"}); NM.update({e: "EUN" for e in EU})
c = pd.read_csv(TOPIC / "analysis/49_statement_type_261102/03_codes_261102.csv", parse_dates=["date"])
c = c[c.statement_type.isin(["threat_or_possible_future", "stated_intention_future"]) & c.direction.eq("higher")]
rows = []
for d, t in zip(c.date, c.target.fillna("")):
    for tg in {x.strip().lower() for x in t.split(";")}:
        i = "CHN" if tg == "china" else NM.get(norm(tg.split(":", 1)[1])) if tg.startswith("country:") else None
        if i: rows.append((i, d, d))
inc = pd.read_csv(TOPIC / "data/31_tariff_calendar_fix_261101/02_increase_events_B_extended.csv", parse_dates=["eff"])
ok = inc[~inc.cancelled.astype(bool) & ~inc.short_lived.astype(bool)]; fe = ok.groupby("iso3").eff.min().to_dict()
F = pd.DataFrame(rows, columns=["iso3", "date", "d2"]).drop(columns="d2")
F = F[[d < fe.get(i, pd.Timestamp("2100-01-01")) for i, d in zip(F.iso3, F.date)]].drop_duplicates().sort_values(["iso3", "date"])
F.to_csv(OUT / "05_f3_passages_261102.csv", index=False)
FIRST = c.date.min()
cal = pd.read_csv(L / "china_sentiment/smb_china_sentiment_tariff_for_L2_261008.csv", usecols=["bin_index", "bin_start", "partial_bin"]); cal["bin_start"] = pd.to_datetime(cal.bin_start)
cal = cal[cal.bin_index.between(53, 111) & ~cal.partial_bin]
sl = pd.read_parquet(Path.home() / "private_cache/customs50_261102/sig_long.parquet")
isos = sorted(sl.iso3.unique()); K = [0, 2, 4, 6, 8, 12, 16]; recs = []
for i in isos:
    d = F[F.iso3 == i].date.values
    for r in cal.itertuples():
        rec = dict(iso3=i, bin_index=r.bin_index)
        for k in K:
            ref = r.bin_start - pd.Timedelta(days=7 * k + 1); lo = ref - pd.Timedelta(days=14)
            rec[f"f3_k{k}"] = np.nan if lo < FIRST else float(((d > np.datetime64(lo)) & (d <= np.datetime64(ref))).sum())
        recs.append(rec)
S = pd.DataFrame(recs); S["lag16_observed"] = S[[f"f3_k{k}" for k in K]].notna().all(axis=1).astype(float)
S = S.merge(sl[["iso3", "bin_index"] + [c_ for c_ in sl.columns if c_.startswith(("Rnum", "Rind"))]], on=["iso3", "bin_index"])
S.to_parquet(CACHE / "sig_f3.parquet")
print("f3 passages", len(F), "countries", F.iso3.nunique(), F.groupby("iso3").size().sort_values(ascending=False).head(8).to_dict())
print("bins with all lags observed", int(S.drop_duplicates("bin_index").lag16_observed.sum()), "of", S.bin_index.nunique(), "| nonzero cells (k8)", int((S.f3_k8 > 0).sum()))
