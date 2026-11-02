# ============================================================
# START DATE   : 2026-11-02
# CODE RUN DATE: 2026-11-02
# TIME TAKEN   : ~15 min
# DESCRIPTION  : A3. (i) Team-style event difference: price outcome on importer_dum x post-April-2 (Liberation Day,
#                the team's after_42) with team controls and wave FE (and a variant with the team's date-bin FE
#                start_group); post-March-4 is checked and reported as undefined. (ii) importer_dum x leader
#                China-path signal (28 days before response, standardized) with a date-shift placebo (5-52 weeks).
#                Outcomes: already-made and planned price change (percent) and any-increase share. uid-clustered.
# INPUT FILES  : see common_261102.py
# OUTPUT FILES : analysis/40_smb_pricing_controls_261102/03_event_diff_261102.csv, 03_signal_261102.csv,
#                03_signal_placebo_261102.csv (aggregates only; no respondent rows)
# ============================================================
import numpy as np
import pandas as pd
import statsmodels.api as sm

from common_261102 import (SIG, drop_collinear, dummies, get_out, load_events, load_survey, parts, prep_importer_sample,
                           safe_write, signal)

out, args = get_out()
for p in ["03_event_diff_261102.csv", "03_signal_261102.csv", "03_signal_placebo_261102.csv"]:
    if (out / p).exists():
        raise SystemExit(f"refuse to overwrite {out / p}")
d = load_survey()
s = prep_importer_sample(d)
OUTCOMES = ["done_pp", "done_any", "plan_pp", "plan_any"]
s["imp"] = s.importer_dum
s["post_apr2"] = (s.date > pd.Timestamp("2025-04-02")).astype(float)       # team's after_42
s["post_mar4"] = (s.date > pd.Timestamp("2025-03-04")).astype(float)
print("post_mar4: min", s.post_mar4.min(), "max", s.post_mar4.max(), "(constant 1 = undefined); earliest response", s.date.min().date())
# ★ CORE LOGIC: the team's start_group date bins from all_waves_new.do lines 230-239 (Stata day numbers);
# days in the gaps (2025-06-20, 07-25, 08-08) belong to no bin and drop out of the start_group variant.
sd = (s.date - pd.Timestamp("1960-01-01")).dt.days
edges = [(23821, 23825, 1), (23825, 23830, 2), (23830, 23840, 3), (23840, 23845, 4), (23845, 23855, 5),
         (23855, 23906, 6), (23906, 23912, 7), (23913, 23947, 8), (23948, 23961, 9), (23962, 10 ** 9, 10)]
s["start_group"] = np.nan
for lo, hi, g in edges:
    s.loc[(sd >= lo) & (sd < hi), "start_group"] = g

CTRL = ["imp", "export_dum", "fcomp_dum", "miss_export_dum", "miss_fcomp_dum", "emp_2024_bins", "sales_2024_val",
        "miss_emp_2024_bins", "miss_sales_2024_val"]


def design(df, extra, fe="wave"):
    parts_ = [df[CTRL].astype(float), extra, dummies(df.industry_code.astype(int), "i")]
    parts_.append(dummies(df.wave.astype(str), "w") if fe == "wave" else dummies(df.start_group.astype(int), "g"))
    X = pd.concat(parts_, axis=1)
    X = X.loc[:, X.std() > 0]
    return drop_collinear(X.assign(const=1.0))


def fit(df, y, extra_cols, fe="wave", cluster=True):
    X = design(df, df[extra_cols].astype(float), fe)
    mod = sm.OLS(df[y].astype(float), X)
    return mod.fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(df.uid)[0]}) if cluster else mod.fit()


# ---------------- (i) events ----------------
rows = []
for fe in ["wave", "start_group"]:
    base = s.dropna(subset=["start_group"]) if fe == "start_group" else s
    for y in OUTCOMES:
        df = base.dropna(subset=[y]).copy()
        df["imp_x_post"] = df.imp * df.post_apr2
        m = fit(df, y, ["post_apr2", "imp_x_post"], fe)
        rows.append(dict(block="i_post_apr2", fe=fe, outcome=y, term="imp_x_post_apr2", coef=m.params["imp_x_post"], se=m.bse["imp_x_post"],
                         p=m.pvalues["imp_x_post"], n=len(df), n_imp=int(df.imp.sum()), n_post=int(df.post_apr2.sum()),
                         # post main effect is not separately identified with wave FE (collinear), so it is reported for the start_group variant only
                         post_main=m.params.get("post_apr2", np.nan) if fe == "start_group" else np.nan,
                         post_main_se=m.bse.get("post_apr2", np.nan) if fe == "start_group" else np.nan,
                         imp_main=m.params["imp"], imp_main_se=m.bse["imp"]))
ev_tab = pd.DataFrame(rows)
safe_write(ev_tab, out / "03_event_diff_261102.csv")

# ---------------- (ii) leader signal ----------------
ev = load_events()
SHIFTS = [k for k in range(-364, 365, 7) if abs(k) >= 35]            # 96 shifts, same as 02_robustness_261101.py
if args.nshift:
    SHIFTS = SHIFTS[:: max(1, len(SHIFTS) // args.nshift)]
dates = pd.DatetimeIndex(sorted(s.date.unique()))


def interaction(df, y, sigmap, cluster):
    """Standardize the signal on the estimation sample, regress y on imp x z, z, team controls, wave FE."""
    t = df.assign(sig=df.date.map(sigmap)).dropna(subset=[y, "sig"]).copy()
    if t.sig.std() == 0 or len(t) < 200:
        return None
    t["z"] = (t.sig - t.sig.mean()) / t.sig.std()
    t["imp_x_z"] = t.imp * t.z
    m = fit(t, y, ["z", "imp_x_z"], "wave", cluster=cluster)
    return m, len(t)


res, plc = [], []
for sname, (q, pos, neg) in SIG.items():
    p_ = parts(ev, q, pos, neg)
    smap0 = signal(p_, dates, 0)
    for y in OUTCOMES:
        df = s.dropna(subset=[y])
        m0, n0 = interaction(df, y, smap0, True)
        b0, se0 = m0.params["imp_x_z"], m0.bse["imp_x_z"]
        sh = []
        for k in SHIFTS:
            r = interaction(df, y, signal(p_, dates, k), False)
            if r is not None:
                sh.append(r[0].params["imp_x_z"])
                plc.append(dict(signal=sname, outcome=y, shift_days=k, coef=sh[-1], n=r[1]))
        sh = np.array(sh)
        res.append(dict(signal=sname, outcome=y, coef_imp_x_signal=b0, se=se0, p_cluster=m0.pvalues["imp_x_z"], n=n0,
                        main_signal=m0.params["z"], main_signal_se=m0.bse["z"], imp_main=m0.params["imp"],
                        placebo_mean=sh.mean(), placebo_sd=sh.std(), placebo_p=float((np.abs(sh) >= abs(b0)).mean()), n_shifts=len(sh)))
res = pd.DataFrame(res)
safe_write(res, out / "03_signal_261102.csv")
safe_write(pd.DataFrame(plc), out / "03_signal_placebo_261102.csv")
pd.set_option("display.width", 240, "display.float_format", lambda x: f"{x:.3f}")
print(ev_tab[["fe", "outcome", "coef", "se", "p", "n", "n_imp", "n_post", "post_main", "imp_main"]].to_string(index=False))
print(res.to_string(index=False))
