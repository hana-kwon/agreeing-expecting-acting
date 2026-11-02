# ============================================================
# START DATE   : 2026-11-02
# CODE RUN DATE: 2026-11-02
# TIME TAKEN   : ~15 min
# DESCRIPTION  : Country x two-week-bin panel for the first stage (spec analysis/53 00_spec_261102.md, fixed 06:01 EST):
#                statement measures (a)-(e), policy outcomes O1 (effective increase), O2 (signing), O3 (TiRADE entries), window flags.
# INPUT FILES  : data/31_tariff_calendar_fix_261101/02_panel_B_extended.csv, 02_increase_events_B_extended.csv;
#                analysis/49_statement_type_261102/03_codes_261102.csv; Jev results + passages in ~/research_local/orgsci_ai_frontiers/topic6_digital_twin/ideology_share_copies_261031;
#                ~/research_local/orgsci_repos/agreeing_expecting_acting/data/public/tirade-v1.5-rate-statements.csv
# OUTPUT FILES : analysis/53_first_stage_261102/01_panel_261102.csv (country x bin aggregates), 01_build_log_261102.json
# ============================================================
import json, re, sys
from pathlib import Path
import numpy as np, pandas as pd

TOPIC = Path(__file__).resolve().parents[2]
OUT = TOPIC / "analysis/53_first_stage_261102"
COPIES = Path.home() / "research_local/orgsci_ai_frontiers/topic6_digital_twin/ideology_share_copies_261031"
TIRADE = Path.home() / "research_local/orgsci_repos/agreeing_expecting_acting/data/public/tirade-v1.5-rate-statements.csv"
tgt = [OUT / "01_panel_261102.csv", OUT / "01_build_log_261102.json"]
for t in tgt:
    if t.exists(): sys.exit(f"output exists; refusing to overwrite: {t}")

pan = pd.read_csv(TOPIC / "data/31_tariff_calendar_fix_261101/02_panel_B_extended.csv")
names = pan.groupby("iso3").name.first().to_dict(); ISOS = sorted(names)
BINS = pd.DatetimeIndex([pd.Timestamp("2025-01-24") + pd.Timedelta(days=14 * k) for k in range(44)])
D_END_CAL, D_END_TIR = pd.Timestamp("2026-10-31"), pd.Timestamp("2026-10-09")
LOG = {}

# ---------- country name -> iso3 ----------
EU_NAMES = ["european union", "eu", "europe", "european nations", "austria", "belgium", "bulgaria", "croatia", "cyprus", "czech republic", "czechia", "denmark", "estonia",
            "finland", "france", "germany", "greece", "hungary", "ireland", "italy", "latvia", "lithuania", "luxembourg", "malta", "netherlands", "poland", "portugal",
            "romania", "slovakia", "slovenia", "spain", "sweden"]
def norm(s): return re.sub(r"\s+", " ", re.sub(r"\(.*?\)", "", s)).strip().lower().removeprefix("the ")
NM = {norm(n): i for i, n in names.items()}
ALIAS = {"china": "CHN", "people's republic of china": "CHN", "uk": "GBR", "britain": "GBR", "turkey": "TUR", "burma": "MMR", "myanmar": "MMR", "philippines": "PHL",
         "bahamas": "BHS", "korea": "KOR", "south korea": "KOR", "gabon": "GAB", "congo": "COD", "falkland islands": "FLK", "vietnam": "VNM", "t[uuml]rkiye": "TUR", "türkiye": "TUR"}
NM.update(ALIAS); NM.update({e: "EUN" for e in EU_NAMES})
def to_iso(s):
    return NM.get(norm(s))

# ---------- statements: Jev ----------
DEMONYM = {"CHN": r"china|chinese|beijing|\bxi\b|hong kong", "CAN": r"canada|canadian", "MEX": r"mexico|mexican",
           "EUN": r"european union|\beu\b|europe|germany|german\b|france|french|italy|spain|netherlands|ireland|belgium|poland|sweden|denmark|austria",
           "JPN": r"japan|japanese", "KOR": r"south korea|\bkorea\b|korean", "TWN": r"taiwan", "VNM": r"vietnam", "IND": r"\bindia\b|\bindian\b", "BRA": r"brazil",
           "CHE": r"switzerland|swiss", "GBR": r"united kingdom|britain|british|\buk\b|england", "RUS": r"russia", "TUR": r"turkey\b|türkiye", "ZAF": r"south africa"}
def rx(i):
    if i in DEMONYM: return DEMONYM[i]
    return r"\b" + re.escape(re.sub(r"\(.*?\)", "", names[i]).strip().lower()) + r"\b"
J = {}
with open(COPIES / "results.jsonl") as f:
    for line in f:
        r = json.loads(line); a = r["answers"]; v = []
        for q in ("tariff_broad_path", "tariff_china_path"):
            p = a.get(q, {}).get("probabilities", {})
            v += [p.get("higher", np.nan), p.get("lower", np.nan), 1 - p["not_evidence"] if "not_evidence" in p else np.nan]
        J[r["score_id"]] = v
ps = pd.read_parquet(COPIES / "scoring_passages.parquet", columns=["score_id", "date", "passage"])
ps["date"] = pd.to_datetime(ps.date); low = ps.passage.str.lower()
jj = pd.DataFrame.from_dict(J, orient="index", columns=["bh", "bl", "be", "ch", "cl", "ce"]); ps = ps.join(jj, on="score_id")
codes = pd.read_csv(TOPIC / "analysis/49_statement_type_261102/03_codes_261102.csv", parse_dates=["date"])
ps["inforce"] = ps.score_id.isin(codes[codes.statement_type == "in_force_description"].score_id)
LOG["passages"] = len(ps); LOG["passages_with_jev"] = int(ps.bh.notna().sum()); LOG["coded_passages"] = len(codes)

def bin_of(d):   # bin index k with end t_k; bin covers (t-14d, t]
    k = np.ceil((d - BINS[0]).dt.days / 14.0)
    return k  # k=0 -> bin 0 holds (t0-14, t0]
rows = []
jev_acc = {}
for i in ISOS:
    m = low.str.contains(rx(i), regex=True); s = ps[m & ps.bh.notna() | m & ps.ch.notna()].copy()
    if len(s) == 0: continue
    h, l, e = (s.ch, s.cl, s.ce) if i == "CHN" else (s.bh, s.bl, s.be)
    s = s.assign(h=h, l=l, e=e, k=bin_of(s.date))
    s = s[(s.k >= 0) & (s.k <= 43)]
    for ex in (False, True):
        t = s[~s.inforce] if ex else s
        g = t.groupby("k").agg(score=("h", "sum"), net=("h", lambda x: 0), n=("h", "size"))
        g["net"] = (t.h - t.l).groupby(t.k).sum(); g["ev"] = t.e.groupby(t.k).sum()
        jev_acc[(i, ex)] = g
# NOTE: Jev-score-excluding-in-force (old e2) dropped: Jev direction after a type filter is not reused
# ---------- statements: typed (Haiku) ----------
codes["iso_list"] = codes.target.fillna("").apply(lambda t: sorted({("CHN" if tg == "china" else to_iso(tg.split(":", 1)[1])) for tg in (x.strip().lower() for x in t.split(";")) if tg == "china" or (tg.startswith("country:") and to_iso(tg.split(":", 1)[1]))}))
unm = sorted({tg.split(":", 1)[1] for t in codes.target.fillna("") for tg in (x.strip().lower() for x in t.split(";")) if tg.startswith("country:") and not to_iso(tg.split(":", 1)[1])})
LOG["target_countries_unmapped"] = unm
cx = codes.explode("iso_list").dropna(subset=["iso_list"]).rename(columns={"iso_list": "iso3"})
cx["k"] = bin_of(cx.date)
cx["threat"] = cx.statement_type.eq("threat_or_possible_future"); cx["intent"] = cx.statement_type.eq("stated_intention_future")
cx["fut"] = cx.threat | cx.intent; cx["spec_rd"] = cx.fut & (cx.names_rate.astype(bool) | cx.names_date.astype(bool)); cx["spec_d"] = cx.fut & cx.names_date.astype(bool)
cx["notinf"] = ~cx.statement_type.eq("in_force_description")
# ★ IMPORTANT (coordinator addition 06:1x): direction/target/type come from the Haiku codes, never from Jev direction scores after a type filter.
cx["higher"] = cx.direction.eq("higher"); cx["fut_higher"] = cx.fut & cx.higher; cx["notinf_higher"] = cx.notinf & cx.higher
cx = cx.sort_values("date")
# new information, not repetition: higher-future claim with no future claim naming the country in the 28 days before it
def new_vs_recent(df):
    out = []
    for i, g in df[df.fut].groupby("iso3"):
        d = g.date.values; h = g.higher.values
        for j in range(len(d)):
            if h[j] and not ((d[:j] > d[j] - np.timedelta64(28, "D")).any()): out.append((i, pd.Timestamp(d[j])))
    return pd.DataFrame(out, columns=["iso3", "date"])
nr = new_vs_recent(cx); nr["k"] = bin_of(nr.date)
# first-time threat: no threat (or threat/intention) passage naming the country in the 90 days before this passage's date (history from 2024-01-05)
def first_flags(df, mask):
    out = []
    for i, g in df[mask].groupby("iso3"):
        d = np.sort(g.date.values)
        for j, x in enumerate(d):
            prev = d[:j]; ok = not ((prev > x - np.timedelta64(90, "D")).any())
            if ok: out.append((i, pd.Timestamp(x)))
    return pd.DataFrame(out, columns=["iso3", "date"])
ft1 = first_flags(cx, cx.threat); ft2 = first_flags(cx, cx.fut)
for f in (ft1, ft2): f["k"] = bin_of(f.date)
# one passage may be coded for the same date twice; flag dedup by (iso3,k)
# ---------- outcomes ----------
inc = pd.read_csv(TOPIC / "data/31_tariff_calendar_fix_261101/02_increase_events_B_extended.csv", parse_dates=["eff", "signing_date", "publication_date"])
inc_ok = inc[~inc.cancelled.astype(bool) & ~inc.short_lived.astype(bool)]
# departure from policy in force: higher-future claim about a country with no effective increase yet in force on the passage date
first_eff = inc_ok.groupby("iso3").eff.min().to_dict()
cx["nopolicy"] = [d < first_eff.get(i, pd.Timestamp("2100-01-01")) for i, d in zip(cx.iso3, cx.date)]
cx["fut_higher_nopol"] = cx.fut_higher & cx.nopolicy
O1 = inc_ok[["iso3", "eff"]].rename(columns={"eff": "d"}).dropna()
O2 = inc[["iso3", "signing_date"]].rename(columns={"signing_date": "d"}).dropna()
tr = pd.read_csv(TIRADE); tr["d"] = pd.to_datetime(tr.date)
tr["types"] = tr.event_type.fillna("").apply(lambda s: {t for t in ["Threatened", "Announced", "Implemented"] if t in s})
dropped = 0; trr = []
for _, r in tr.iterrows():
    isos = {to_iso(c) for c in str(r.countries).split(";")} - {None}
    if not isos: dropped += 1
    for i in isos:
        for t in r.types: trr.append((i, r.d, t))
LOG["tirade_rows"] = len(tr); LOG["tirade_rows_no_country"] = dropped
TR = pd.DataFrame(trr, columns=["iso3", "d", "type"]).drop_duplicates()
OUTC = {"O1_eff_incr": O1, "O2_signing": O2, "O3_tir_threat": TR[TR.type == "Threatened"][["iso3", "d"]], "O3_tir_announce": TR[TR.type == "Announced"][["iso3", "d"]],
        "O3_tir_implement": TR[TR.type == "Implemented"][["iso3", "d"]], "O3_tir_any": TR[["iso3", "d"]].drop_duplicates()}
LIM = {"O1_eff_incr": D_END_CAL, "O2_signing": D_END_CAL};
WIN = {"fut_2_12": (14, 84), "fut_2_4": (14, 28), "fut_5_8": (28, 56), "fut_9_12": (56, 84), "imm_0_2": (0, 14), "conc": (-14, 0), "past_2_12": (-84, -14)}
for o in OUTC: LIM.setdefault(o, D_END_TIR)
ev_by = {o: {i: np.sort(g.d.values) for i, g in df.groupby("iso3")} for o, df in OUTC.items()}
rows = []
for i in ISOS:
    for kk, t in enumerate(BINS):
        r = dict(iso3=i, name=names[i], k=kk, bin_end=t.date().isoformat())
        for o in OUTC:
            ev = ev_by[o].get(i, np.array([], dtype="datetime64[ns]"))
            for w, (lo, hi) in WIN.items():
                miss = (hi > 0) and (t + pd.Timedelta(days=hi) > LIM[o])
                r[f"{o}__{w}"] = np.nan if miss else float(((ev > np.datetime64(t + pd.Timedelta(days=lo))) & (ev <= np.datetime64(t + pd.Timedelta(days=hi)))).any())
        # pending: increase signed/published before t, not yet effective
        iv = inc_ok[(inc_ok.iso3 == i)]; kn = iv[["signing_date", "publication_date"]].min(axis=1)
        r["pending"] = float(((kn <= t) & (iv.eff > t)).any())
        rows.append(r)
P = pd.DataFrame(rows)
# measures
def getj(i, ex, k, col):
    g = jev_acc.get((i, ex)); return float(g.loc[k, col]) if g is not None and k in g.index else 0.0
cnt = lambda df, i, k: int(((df.iso3 == i) & (df.k == k)).sum())
M = []
for i, k in zip(P.iso3, P.k):
    sc, ev_, net = getj(i, False, k, "score"), getj(i, False, k, "ev"), getj(i, False, k, "net")
    M.append(dict(a1_jev_score=sc, a2_jev_share=(net / ev_ if ev_ > 0 else 0.0), f1_n_future_higher=cnt(cx[cx.fut_higher], i, k), f2_n_new_vs_recent=cnt(nr, i, k), f3_n_higher_no_policy_yet=cnt(cx[cx.fut_higher_nopol], i, k),
                  e2_n_notinforce_higher=cnt(cx[cx.notinf_higher], i, k),
                  b1_n_future=cnt(cx[cx.fut], i, k), b2_n_threat=cnt(cx[cx.threat], i, k), b3_n_intent=cnt(cx[cx.intent], i, k),
                  c1_first_threat=float(cnt(ft1, i, k) > 0), c2_first_threat_or_intent=float(cnt(ft2, i, k) > 0),
                  d1_n_specific=cnt(cx[cx.spec_rd], i, k), d2_n_dated=cnt(cx[cx.spec_d], i, k), e1_n_notinforce=cnt(cx[cx.notinf], i, k)))
P = pd.concat([P, pd.DataFrame(M)], axis=1)
# reference: earlier 28-day panel measures at the bin end week (Friday = bin end)
P = P.merge(pan[["iso3", "week", "mentions", "jev", "threats4"]].rename(columns={"week": "bin_end", "mentions": "ref_mentions28", "jev": "ref_jev28", "threats4": "ref_tirade_threats4"}), on=["iso3", "bin_end"], how="left")
P.to_csv(OUT / "01_panel_261102.csv", index=False)
LOG.update(cells=len(P), countries=P.iso3.nunique(), bins=P.k.nunique(), unmapped_tirade_country_tokens="see tirade_rows_no_country",
           measure_nonzero_bins={c: int((P[c] != 0).sum()) for c in M[0]},
           outcome_positive_bins={f"{o}__fut_2_12": int(P[f"{o}__fut_2_12"].sum()) for o in OUTC})
(OUT / "01_build_log_261102.json").write_text(json.dumps(LOG, indent=1, default=str)); print(json.dumps(LOG, indent=1, default=str))
