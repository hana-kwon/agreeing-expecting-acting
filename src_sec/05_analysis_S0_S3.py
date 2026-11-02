# ============================================================
# START DATE   : 2026-11-02
# CODE RUN DATE: 2026-11-02
# TIME TAKEN   : ~40 min
# DESCRIPTION  : Build firm-filing panel from coded NEW tariff passages; S0 describe, S1 leader-signal regressions + date-shift placebo, S2 expectation -> new action, S3 by-quarter shares, S4 quote match + check sheet.
# INPUT FILES  : analysis/35.../01_passage_codes_261102.csv ; data/35.../02_filings_manifest_261102.csv, 03_filing_counts_261102.csv, 03_passages_all_261102.csv, 04_firm_order_261102.csv ;
#                data/jev/event_question_option_probabilities.parquet (repository copy of the Jev event scores)
# OUTPUT FILES : analysis/35.../02_panel_261102.csv, 03_S0_*.csv, 03_S1_*.csv, 03_S2_*.csv, 03_S3_*.csv, 03_S3_timing_261102.png, 04_check_sheet_261102.csv, 04_check_sheet_key_261102.csv
# ============================================================
import sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve()
TOPIC = HERE.parents[2]
DATA, ANA = TOPIC / "data/35_sec_tariff_expectations_261102", TOPIC / "analysis/35_sec_tariff_expectations_261102"
TS = HERE.parents[1] / "data/jev"  # repository copy of the Jev event scores
if (ANA / "02_panel_261102.csv").exists(): sys.exit("outputs exist; refusing to overwrite")
ACTIONS = ["price_increase", "sourcing_shift_away_china", "sourcing_shift_to_us_or_nearshore", "supplier_diversification_other",
           "inventory_pull_forward", "investment_delay_or_pause", "investment_increase_us", "wait_for_clarity", "cost_mitigation_other"]
RANK = {"none": 0, "planned": 1, "underway": 2, "completed": 3}

# ---------------- panel ----------------
codes = pd.read_csv(ANA / "01_passage_codes_261102.csv", dtype={"accession": str})
man = pd.read_csv(DATA / "02_filings_manifest_261102.csv", dtype={"accession": str})
cnt = pd.read_csv(DATA / "03_filing_counts_261102.csv", dtype={"accession": str})
order = pd.read_csv(DATA / "04_firm_order_261102.csv")
keep = set(order[order.included].ticker)
fl = man[(man.baseline == 0) & man.ticker.isin(keep)].merge(cnt[["accession", "n_tariff_passages"]], on="accession", how="left")
fl["filing_date"] = pd.to_datetime(fl.filing_date)
fl["quarter"] = fl.filing_date.dt.to_period("Q").astype(str)
fl = fl.sort_values(["ticker", "filing_date", "accession"]).reset_index(drop=True)

c = codes.copy()
c["future"] = c.tense.isin(["future", "both"])
c["firm_cond"] = c.certainty.isin(["explicit_expectation", "conditional"])
c["f_high_ec"] = c.future & c.firm_cond & (c.direction == "higher")
c["f_low_ec"] = c.future & c.firm_cond & (c.direction == "lower")
c["f_high_any"] = c.future & (c.direction == "higher")
c["f_low_any"] = c.future & (c.direction == "lower")
c["poss_only"] = c.future & (c.certainty == "possibility_only")
c["lasting"] = c.duration == "lasting_stated"
c["expo"] = c.exposure_only == "yes"
for a in ACTIONS: c["r_" + a] = c["act_" + a].map(RANK)
g = c.groupby("accession")
agg = pd.DataFrame({"n_new": g.size(), "n_future": g.future.sum(), "exp_high": g.f_high_ec.sum(), "exp_low": g.f_low_ec.sum(),
                    "high_any": g.f_high_any.sum(), "low_any": g.f_low_any.sum(), "n_poss": g.poss_only.sum(),
                    "last_n": g.lasting.sum(), "expo_n": g.expo.sum()})
for a in ACTIONS: agg["r_" + a] = g["r_" + a].max()
fl = fl.merge(agg, left_on="accession", right_index=True, how="left")
for col in agg.columns: fl[col] = fl[col].fillna(0)
fl["no_new"] = (fl.n_new == 0).astype(int)
fl["n_all"] = fl.n_tariff_passages.fillna(0)
fl["ln_n_all"] = np.log1p(fl.n_all)
fl["Expo"] = np.where(fl.n_new > 0, fl.expo_n / fl.n_new.clip(lower=1), 0.0)
fl["Hexp"] = (fl.exp_high >= 1).astype(int)
fl["Last"] = (fl.last_n >= 1).astype(int)
fl["Y_S1"] = np.where(fl.n_future >= 1, (fl.exp_high - fl.exp_low) / fl.n_future.replace(0, np.nan), np.nan)
fl["Y_S1_incl_poss"] = np.where(fl.n_future >= 1, (fl.high_any - fl.low_any) / fl.n_future.replace(0, np.nan), np.nan)
fl["any_act_t"] = (fl[["r_" + a for a in ACTIONS]].max(axis=1) > 0).astype(int)
fl.to_csv(ANA / "02_panel_261102.csv", index=False)

# ---------------- S0 ----------------
s0 = []
s0.append(dict(item="firms_in_coded_subset", value=len(keep)))
s0.append(dict(item="firms_sample_total", value=len(order)))
s0.append(dict(item="filings_analysis_window_in_subset", value=len(fl)))
s0.append(dict(item="filings_with_>=1_tariff_passage", value=int((fl.n_all > 0).sum())))
s0.append(dict(item="filings_with_>=1_NEW_coded_passage", value=int((fl.n_new > 0).sum())))
s0.append(dict(item="NEW_passages_coded", value=len(c)))
s0.append(dict(item="future_or_both_passages", value=int(c.future.sum())))
s0.append(dict(item="future_possibility_only", value=int(c.poss_only.sum())))
s0.append(dict(item="future_possibility_only_share", value=float(c.poss_only.sum() / c.future.sum())))
pd.DataFrame(s0).to_csv(ANA / "03_S0_counts_261102.csv", index=False)
pd.crosstab(c.tense, c.certainty, margins=True).to_csv(ANA / "03_S0_tense_by_certainty_261102.csv")
pd.crosstab(c[c.future].certainty, c[c.future].direction, margins=True).to_csv(ANA / "03_S0_future_certainty_by_direction_261102.csv")
pd.crosstab(c.duration, c.future, margins=True).to_csv(ANA / "03_S0_duration_261102.csv")
pd.DataFrame({a: c["act_" + a].value_counts() for a in ACTIONS}).fillna(0).astype(int).to_csv(ANA / "03_S0_action_status_counts_261102.csv")
print(pd.DataFrame(s0).to_string())

# ---------------- S1 ----------------
ev = pd.read_parquet(TS / "event_question_option_probabilities.parquet")
ev["event_date"] = pd.to_datetime(ev["event_date"])
SIG = {"S_path": ("tariff_broad_path", ["higher"], ["lower"]),
       "S_duration": ("tariff_broad_duration", ["permanent"], ["temporary"]),
       "S_unresolved": ("tariff_broad_resolution", ["unresolved_through_2025"], ["resolved_within_month"])}

# ★ CORE LOGIC: `parts` and `signal` copied from 27_expectation_action_261101/02_robustness_261101.py lines 115-141 (only SIG dict is limited to the broad questions)
def parts(q, pos, neg):
    x = ev[ev.question_id == q]
    w = x.pivot_table(index=["event_question_id", "event_date"], columns="option", values="raw_option_probability", aggfunc="sum").fillna(0.0).reset_index()
    w["ev"] = 1.0 - w[x[x.is_absence_option].option.unique()[0]]
    w["num"] = w[pos].sum(axis=1) - w[neg].sum(axis=1)
    return w.groupby("event_date")[["num", "ev"]].sum()

def signal(p, dates, shift=0):
    out = {}
    for t in dates:
        a_, b_ = t + pd.Timedelta(days=shift - 28), t + pd.Timedelta(days=shift - 1)
        s = p.loc[(p.index >= a_) & (p.index <= b_)]
        out[t] = s.num.sum() / s.ev.sum() if s.ev.sum() > 0 else np.nan
    return pd.Series(out)

def cl_ols(y, X, groups):
    X = sm.add_constant(X.loc[:, X.std() > 0].astype(float), has_constant="add")
    return sm.OLS(y.astype(float), X).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(groups)[0]})

def s1_beta(df, ycol, sig):
    d = df.assign(sig=df.filing_date.map(sig)).dropna(subset=[ycol, "sig"]).copy()
    d["sig"] = (d.sig - d.sig.mean()) / d.sig.std()
    X = pd.concat([d[["sig"]], pd.get_dummies(d.quarter, prefix="q", drop_first=True).astype(float)], axis=1)
    return cl_ols(d[ycol], X, d.ticker), len(d), d.ticker.nunique()

dates = pd.DatetimeIndex(sorted(fl.filing_date.unique()))
SHIFTS = [k for k in range(-364, 365, 7) if abs(k) >= 35]
rows1 = []
for ycol in ["Y_S1", "Y_S1_incl_poss"]:
    for name, (q, pos, neg) in SIG.items():
        p = parts(q, pos, neg)
        m, n, nf = s1_beta(fl, ycol, signal(p, dates, 0))
        sh = np.array([s1_beta(fl, ycol, signal(p, dates, k))[0].params["sig"] for k in SHIFTS])
        act = m.params["sig"]
        rows1.append(dict(y=ycol, signal=name, beta=act, se=m.bse["sig"], p=m.pvalues["sig"], n_filings=n, n_firms=nf,
                          shift_mean=sh.mean(), shift_sd=sh.std(), p_placebo=float((np.abs(sh) >= abs(act)).mean()), n_shifts=len(sh)))
s1 = pd.DataFrame(rows1); s1.to_csv(ANA / "03_S1_leader_signal_261102.csv", index=False)
print(s1.to_string())

# ---------------- S2 ----------------
def new_report_frames(d):
    """For each firm, flag action types first reported (any status) in filing j given no report in earlier filings; return t+1 pair table."""
    out = []
    for t, f in d.groupby("ticker", sort=False):
        f = f.sort_values(["filing_date", "accession"]).reset_index(drop=True)
        seen = {a: False for a in ACTIONS}
        prev_status = None
        rec = []
        for j in range(len(f)):
            cur = {a: int(f.loc[j, "r_" + a]) for a in ACTIONS}
            newt = {a: (cur[a] > 0 and not seen[a]) for a in ACTIONS}
            rec.append((newt, cur))
            for a in ACTIONS:
                if cur[a] > 0: seen[a] = True
        for j in range(len(f) - 1):
            newt, cur = rec[j + 1]
            r = f.loc[j].to_dict()
            nxt = f.loc[j + 1]
            r.update(next_acc=nxt.accession, next_date=nxt.filing_date, next_quarter=nxt.quarter, next_Hexp=nxt.Hexp, next_ln_n_all=nxt.ln_n_all,
                     next_Expo=nxt.Expo, next_no_new=nxt.no_new)
            for a in ACTIONS: r["new_" + a] = int(newt[a])
            r["new_any"] = int(any(newt.values()))
            r["new_planned"] = int(any(newt[a] and cur[a] == 1 for a in ACTIONS))
            r["new_executed"] = int(any(newt[a] and cur[a] >= 2 for a in ACTIONS))
            out.append(r)
    return pd.DataFrame(out)

pairs = new_report_frames(fl)
pairs["sq"] = pairs.sector + "_" + pairs.quarter
pairs.to_csv(ANA / "03_S2_pairs_261102.csv", index=False)

def fit_lpm(d, y, xs, fe="sq"):
    d = d.dropna(subset=[y] + xs)
    X = pd.concat([d[xs], pd.get_dummies(d[fe], prefix="fe", drop_first=True).astype(float)], axis=1)
    return cl_ols(d[y], X, d.ticker), d

BASE = ["ln_n_all"]; CTRL = ["ln_n_all", "Expo", "no_new"]; FULL = CTRL + ["Hexp", "Last"]
rows2, cmp_rows = [], []
for y in ["new_any", "new_planned", "new_executed"] + ["new_" + a for a in ACTIONS]:
    if pairs[y].sum() < 5: rows2.append(dict(y=y, n_pairs=len(pairs), n_events=int(pairs[y].sum()), note="<5 events, not estimated")); continue
    ma, d = fit_lpm(pairs, y, BASE); mb, _ = fit_lpm(pairs, y, CTRL); mc, _ = fit_lpm(pairs, y, FULL)
    wald = mc.wald_test("Hexp = 0, Last = 0", scalar=True)
    rows2.append(dict(y=y, n_pairs=len(d), n_firms=d.ticker.nunique(), n_events=int(d[y].sum()), base_rate=d[y].mean(),
                      b_Hexp=mc.params["Hexp"], se_Hexp=mc.bse["Hexp"], p_Hexp=mc.pvalues["Hexp"],
                      b_Last=mc.params["Last"], se_Last=mc.bse["Last"], p_Last=mc.pvalues["Last"],
                      b_ln_n_all=mc.params["ln_n_all"], se_ln_n_all=mc.bse["ln_n_all"], b_Expo=mc.params["Expo"], se_Expo=mc.bse["Expo"],
                      r2_a=ma.rsquared, r2_b=mb.rsquared, r2_c=mc.rsquared, r2_gain_c_vs_b=mc.rsquared - mb.rsquared,
                      p_joint_Hexp_Last=float(wald.pvalue)))
    # mention-count-only model coefficient, for the comparison table
    cmp_rows.append(dict(y=y, b_ln_n_all_M_a=ma.params["ln_n_all"], se=ma.bse["ln_n_all"], p=ma.pvalues["ln_n_all"]))
s2 = pd.DataFrame(rows2); s2.to_csv(ANA / "03_S2_expectation_to_new_action_261102.csv", index=False)
pd.DataFrame(cmp_rows).to_csv(ANA / "03_S2_mention_count_only_261102.csv", index=False)
print(s2[["y", "n_pairs", "n_events", "base_rate", "b_Hexp", "se_Hexp", "p_Hexp", "b_Last", "se_Last", "p_Last", "r2_gain_c_vs_b", "p_joint_Hexp_Last"]].to_string())

# reverse ordering (descriptive): expectation in t+1 on any action reported in t
rv = pairs.copy(); rv["sq_next"] = rv.sector + "_" + rv.next_quarter
rrows = []
for xs_lab, xs in [("any_act_t", ["any_act_t"])]:
    m, d = fit_lpm(rv.assign(any_act_t=rv.any_act_t), "next_Hexp", xs + ["next_ln_n_all", "next_Expo", "next_no_new"], fe="sq_next")
    rrows.append(dict(y="next_Hexp", x="any_act_t", b=m.params["any_act_t"], se=m.bse["any_act_t"], p=m.pvalues["any_act_t"], n=len(d), base=d.next_Hexp.mean()))
# the same regression on Hexp at t (same-filing), for contrast of ordering
m, d = fit_lpm(rv, "next_Hexp", ["Hexp", "next_ln_n_all", "next_Expo", "next_no_new"], fe="sq_next")
rrows.append(dict(y="next_Hexp", x="Hexp_t (persistence of expectation)", b=m.params["Hexp"], se=m.bse["Hexp"], p=m.pvalues["Hexp"], n=len(d), base=d.next_Hexp.mean()))
pd.DataFrame(rrows).to_csv(ANA / "03_S2_reverse_descriptive_261102.csv", index=False)
print(pd.DataFrame(rrows).to_string())

# ---------------- S3 ----------------
byq = fl.groupby("quarter").agg(filings=("accession", "size"), filings_with_new=("n_new", lambda s: int((s > 0).sum())), n_new_passages=("n_new", "sum"))
for a in ACTIONS:
    byq["share_" + a] = fl.groupby("quarter").apply(lambda d: float((d["r_" + a] > 0).mean()))
byq["share_any_action"] = fl.groupby("quarter").any_act_t.mean()
cq = c.merge(fl[["accession", "quarter"]], on="accession")
byq["future_claims"] = cq.groupby("quarter").future.sum()
byq["future_possibility_only_share"] = cq[cq.future].groupby("quarter").poss_only.mean()
byq["future_explicit_cond_share"] = cq[cq.future].groupby("quarter").firm_cond.mean()
byq.to_csv(ANA / "03_S3_by_quarter_261102.csv")
fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
sel = ["price_increase", "sourcing_shift_away_china", "sourcing_shift_to_us_or_nearshore", "supplier_diversification_other", "wait_for_clarity", "investment_delay_or_pause"]
for a in sel: ax[0].plot(byq.index, byq["share_" + a], marker="o", label=a)
ax[0].set_title("Share of firm-filings reporting each action (NEW passages)"); ax[0].set_ylabel("share of all filings in quarter"); ax[0].tick_params(axis="x", rotation=60); ax[0].legend(fontsize=7)
ax[1].plot(byq.index, byq.future_possibility_only_share, marker="o", label="possibility only")
ax[1].plot(byq.index, byq.future_explicit_cond_share, marker="s", label="explicit or conditional")
ax[1].set_title("Composition of future tariff claims (NEW passages)"); ax[1].set_ylabel("share of future claims"); ax[1].tick_params(axis="x", rotation=60); ax[1].legend(fontsize=8)
fig.suptitle(f"S&P 500 subset, {len(keep)} firms; filing-date quarter; exploratory", fontsize=9); fig.tight_layout(); fig.savefig(ANA / "03_S3_timing_261102.png", dpi=130)

# ---------------- S4 ----------------
q = pd.DataFrame({"quote_verbatim_rate": [c.quote_verbatim.mean()], "n_passages": [len(c)], "n_nonempty_quotes": [int((c.quote != "").sum())],
                  "rate_among_nonempty": [c[c.quote != ""].quote_verbatim.mean()]})
q.to_csv(ANA / "04_S4_quote_match_261102.csv", index=False); print(q.to_string())
raw = pd.read_csv(DATA / "03_passages_all_261102.csv", dtype={"accession": str}, usecols=["passage_id", "text", "ticker"])
cc = c.merge(raw[["passage_id", "text"]], on="passage_id")
cc["any_act"] = cc[["act_" + a for a in ACTIONS]].ne("none").any(axis=1)
supply = ["act_sourcing_shift_away_china", "act_sourcing_shift_to_us_or_nearshore", "act_supplier_diversification_other"]
other = ["act_investment_delay_or_pause", "act_investment_increase_us", "act_inventory_pull_forward", "act_wait_for_clarity", "act_cost_mitigation_other"]
strata = {"past_no_action": cc[(cc.tense == "in_force_or_past") & ~cc.any_act], "future_no_action": cc[cc.future & ~cc.any_act],
          "price_increase": cc[cc.act_price_increase != "none"], "supply_actions": cc[cc[supply].ne("none").any(axis=1)], "other_actions": cc[cc[other].ne("none").any(axis=1)]}
parts_ = []
for k, dfk in strata.items():
    parts_.append(dfk.sample(min(12, len(dfk)), random_state=20261102).assign(stratum=k))
ck = pd.concat(parts_).drop_duplicates("passage_id").sample(frac=1, random_state=20261102).reset_index(drop=True)
ck["check_id"] = [f"C{i+1:02d}" for i in range(len(ck))]
ans = ["h_tense", "h_certainty", "h_direction", "h_duration", "h_exposure_only", "h_actions_and_status", "h_notes"]
sheet = ck[["check_id", "ticker", "filing_date", "text"]].copy()
for a in ans: sheet[a] = ""
sheet.to_csv(ANA / "04_check_sheet_261102.csv", index=False)
ck[["check_id", "passage_id", "stratum", "tense", "certainty", "direction", "duration", "exposure_only", "quote"] + ["act_" + a for a in ACTIONS]].to_csv(ANA / "04_check_sheet_key_261102.csv", index=False)
print("done; check sheet", len(ck))
