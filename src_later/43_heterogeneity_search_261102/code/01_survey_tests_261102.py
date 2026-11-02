# ============================================================
# START DATE   : 2026-11-02
# CODE RUN DATE: 2026-11-02
# TIME TAKEN   : ~10 min
# DESCRIPTION  : H1, H2, H4 (survey part), H5 (survey part) of 00_spec_261102.md: signal x alignment (and signal main effect)
#                on expectations and plans, with 96-shift date placebos. Every specification run is logged.
# INPUT FILES  :PRIVATE_DATA_ROOT/SMB_tariff/ideology_analysis/{inputs/analysis_all.dta, outputs/alignment_for_stata.dta,
#                trump_alignment_v2/trump_alignment_v2.dta} (read only), Jev event file via common_261102.py,
#                data/16_tariff_schedule_261029/china_additional_duty_steps_261029.csv
# OUTPUT FILES : analysis/43_heterogeneity_search_261102/01_survey_tests_261102.csv (aggregates; no respondent rows)
# ============================================================
import csv
import sys
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm

from common_261102 import IA, OUT, SHIFTS, SIGNALS, TOPIC, signal_at

warnings.filterwarnings("ignore")
outf = OUT / "01_survey_tests_261102.csv"
if outf.exists():
    sys.exit(f"{outf} exists; refusing to overwrite.")

keep = ["responseid", "uid", "startdate", "country", "three_info", "long_term_prior", "short_term_prior",
        "import_tariff_expect", "invest_post2_1", "prices_post2_1"]
d = pd.read_stata(IA / "inputs/analysis_all.dta", columns=keep, convert_categoricals=False)
d = d[d.country == 1]
a = pd.read_stata(IA / "outputs/alignment_for_stata.dta", convert_categoricals=False)
acols = ["responseid", "wave", "align", "politics_us_group", "importer_dum", "export_dum", "fcomp_dum", "industry_code",
         "emp_2024_bins", "sales_2024_val", "import_loc_post2_1__toward", "import_loc_post2_3__away"]
d = d.merge(a[acols], on="responseid", validate="1:1")
v2 = pd.read_stata(IA / "trump_alignment_v2/trump_alignment_v2.dta", convert_categoricals=False)
d = d.merge(v2[["responseid", "align_trump"]], on="responseid", how="left", validate="1:1")
d["date"] = pd.to_datetime(d.startdate, errors="coerce").dt.normalize()
steps = pd.read_csv(TOPIC / "data/16_tariff_schedule_261029/china_additional_duty_steps_261029.csv")
steps = steps.iloc[:, :2]; steps.columns = ["eff", "duty_cn"]; steps["eff"] = pd.to_datetime(steps.eff)
d = pd.merge_asof(d.dropna(subset=["date"]).sort_values("date"), steps.sort_values("eff"), left_on="date", right_on="eff", direction="backward")
for c in ["emp_2024_bins", "sales_2024_val"]:
    d["miss_" + c] = d[c].isna().astype(float)
    d[c] = d[c].fillna(d[c].median())
for c in ["importer_dum", "export_dum", "fcomp_dum"]:
    d["miss_" + c] = d[c].isna().astype(float)
    d[c] = d[c].fillna(0.0)
d["industry_code"] = d["industry_code"].fillna(-1)
d["rep"] = d.politics_us_group.map({"Republican": 1.0, "Democrat": 0.0})
early = d.wave.astype(str).str.contains("W1|W2|W3")
w4 = d.wave.astype(str).str.contains("W4")
SAMP = {"W1-W3": d[early], "W4": d[w4]}
CTRL = ["emp_2024_bins", "sales_2024_val", "miss_emp_2024_bins", "miss_sales_2024_val", "duty_cn",
        "importer_dum", "export_dum", "fcomp_dum", "miss_importer_dum", "miss_export_dum", "miss_fcomp_dum"]
PLAN = ["prices_post2_1", "invest_post2_1", "import_loc_post2_1__toward", "import_loc_post2_3__away"]
EXP = {"W1-W3": ["long_term_prior", "short_term_prior"], "W4": ["import_tariff_expect"]}

# signal matrices by date (cache of all shifts) -------------------------------------------------
alldates = pd.DatetimeIndex(sorted(d.date.unique()))
SIGM = {}
for s in SIGNALS:
    SIGM[s] = {k: signal_at(s, alldates, k) for k in [0] + SHIFTS}


def design(df, y, aligned):
    """Rows with y and alignment, control block (without signal terms)."""
    if aligned == "rep":
        df = df[df.rep.notna()]
    cols = [y] + ([aligned] if aligned not in ("rep", None) else (["rep"] if aligned == "rep" else []))
    s = df.dropna(subset=cols).copy()
    parts = [s[CTRL].astype(float),
             pd.get_dummies(s.wave.astype(str), prefix="w", drop_first=True).astype(float),
             pd.get_dummies(s.industry_code.astype(int), prefix="i", drop_first=True).astype(float),
             pd.get_dummies(s.three_info.fillna(0).astype(int), prefix="m", drop_first=True).astype(float)]
    if aligned == "rep":
        a_ = s.rep.astype(float)
    elif aligned is None:
        a_ = None
    else:
        a_ = (s[aligned] - s[aligned].mean()) / s[aligned].std()
    if a_ is not None:
        parts.append(a_.rename("A").to_frame())
    X = pd.concat(parts, axis=1)
    X = X.loc[:, X.std() > 0]
    return s, X, a_


def est(s, X, a_, y, sig, kind, cluster=None):
    """OLS of y on [sig, sig*A (if interaction), X]; returns coef of interest, se (cluster uid), n, slopes."""
    z = (sig - sig.mean()) / sig.std()
    cols = {"sig": z}
    if kind == "int":
        cols["sigA"] = z * a_
    ok = z.notna()
    Z = pd.DataFrame(cols)[ok]
    XX = sm.add_constant(pd.concat([Z, X[ok]], axis=1), has_constant="add")
    yy = s.loc[ok, y].astype(float)
    if cluster is None:
        b = np.linalg.lstsq(XX.values, yy.values, rcond=None)[0]
        return dict(zip(XX.columns, b))[("sigA" if kind == "int" else "sig")], None, int(ok.sum())
    m = sm.OLS(yy, XX).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(s.loc[ok, "uid"])[0]})
    key = "sigA" if kind == "int" else "sig"
    return m.params[key], m.bse[key], int(ok.sum()), m, key


rows = []
log = []


def run(h, sample, y, signal, aligned, kind, sign):
    df = SAMP[sample]
    s, X, a_ = design(df, y, aligned if kind == "int" else None)
    if len(s) < 200:
        rows.append(dict(hyp=h, sample=sample, outcome=y, signal=signal, aligned=aligned, kind=kind, n=len(s), note="n<200, not run"))
        return
    S0 = s.date.map(SIGM[signal][0])
    if S0.notna().sum() < 200:
        rows.append(dict(hyp=h, sample=sample, outcome=y, signal=signal, aligned=aligned, kind=kind, n=int(S0.notna().sum()), note="signal defined for <200 rows, not run"))
        return
    b, se, n, m, key = est(s, X, a_, y, S0, kind, cluster=True)
    extra = {}
    if kind == "int":
        extra["slope_aligned_or_high"] = m.params["sig"] + m.params["sigA"]
        extra["slope_unaligned_or_mean"] = m.params["sig"]
    sh = []
    for k in SHIFTS:
        Sk = s.date.map(SIGM[signal][k])
        if Sk.notna().sum() < 100:
            continue
        sh.append(est(s, X, a_, y, Sk, kind)[0])
    sh = np.array(sh)
    rows.append(dict(hyp=h, sample=sample, outcome=y, signal=signal, aligned=aligned if kind == "int" else "none", kind=kind,
                     expected_sign=sign, coef=b, se=se, n=n, n_resp_clusters=s.uid.nunique(), p_cluster=m.pvalues[key],
                     placebo_p=float((np.abs(sh) >= abs(b)).mean()), n_shifts=len(sh), shift_mean=sh.mean(), shift_sd=sh.std(),
                     sign_ok=int(np.sign(b) == sign), **extra, note=""))


PATH = ["path_china", "path_broad"]
ALIGN = ["align", "rep"]
# H1 + H4 (survey expectation test with rebuilt signals) ------------------------------------------
for sample, ys in EXP.items():
    for y in ys:
        for sig in PATH:
            for al in ALIGN + ["align_trump"]:
                run("H1", sample, y, sig, al, "int", +1)
        for fam in ["china", "broad"]:
            for pre in ["F1", "F2", "F3"]:
                for al in ALIGN:
                    run("H4", sample, y, f"{pre}_{fam}", al, "int", +1)
# H2: plans ----------------------------------------------------------------------------------------
for sample in SAMP:
    for y in PLAN:
        for sig in PATH:
            for al in ALIGN + ["align_trump"]:
                run("H2", sample, y, sig, al, "int", +1)
# H5: duration main effects (+ path comparators), and duration x alignment -------------------------
H5Y = [("W1-W3", "long_term_prior"), ("W4", "import_tariff_expect")] + [(sm_, y) for sm_ in SAMP for y in PLAN[2:]]
for sample, y in H5Y:
    for fam in ["china", "broad"]:
        run("H5", sample, y, f"dur_{fam}", None, "main", +1)
        run("H5_cmp_path", sample, y, f"path_{fam}", None, "main", +1)
        for al in ALIGN:
            run("H5", sample, y, f"dur_{fam}", al, "int", +1)
R = pd.DataFrame(rows)
R.to_csv(outf, index=False)
pd.set_option("display.width", 250, "display.max_columns", 30, "display.max_rows", 400)
print(R[["hyp", "sample", "outcome", "signal", "aligned", "kind", "coef", "se", "n", "placebo_p", "note"]].round(3).to_string())
print("rows", len(R), "run", R.coef.notna().sum())
