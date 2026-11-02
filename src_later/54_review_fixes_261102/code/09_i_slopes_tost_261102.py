# ============================================================
# START DATE   : 2026-11-02
# CODE RUN DATE: 2026-11-02
# TIME TAKEN   : ~10 min
# DESCRIPTION  : I. (1) Group-specific slopes (Republican and Democratic; +1 SD and -1 SD alignment) with SEs for the survey tests of
#                analysis 43 that passed the rule, using the exact design of 43/01_survey_tests (its loading/design code is exec'ed from that
#                file up to the test loop, minus the overwrite guard). (2) Correct chance benchmark for the 172-test summary: the pass rule is
#                placebo p <= 0.10 AND the pre-set sign, so the per-test chance rate is about 0.10 x 0.5; exact binomial tail for 18 passes.
#                (3) Equivalence (TOST) for selected null results against a smallest effect of interest recorded in 00_spec (post hoc choice).
# INPUT FILES  : code/43 survey tests file (read, exec'ed in part), analysis/43/01_all_tests_261102.csv, 54 outputs
# OUTPUT FILES : analysis/54_review_fixes_261102/09_i_group_slopes_261102.csv, 09_i_chance_benchmark_261102.csv
# ============================================================
import sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
HERE = Path(__file__).resolve(); TOPIC = HERE.parents[2]; OUT = TOPIC / "analysis/54_review_fixes_261102"
for f in ["09_i_group_slopes_261102.csv", "09_i_chance_benchmark_261102.csv"]:
    if (OUT / f).exists(): sys.exit("refuse to overwrite " + f)
src43 = TOPIC / "code/43_heterogeneity_search_261102"
sys.path.insert(0, str(src43))
txt = (src43 / "01_survey_tests_261102.py").read_text()
head = txt[:txt.index("rows = []\nlog = []")]
head = head.replace('if outf.exists():\n    sys.exit(f"{outf} exists; refusing to overwrite.")', "")
ns = {"__file__": str(src43 / "01_survey_tests_261102.py")}
exec(compile(head, "43head", "exec"), ns)
import statsmodels.api as sm
at = pd.read_csv(TOPIC / "analysis/43_heterogeneity_search_261102/01_all_tests_261102.csv")
passed = at[at.passes_rule == True]
rows = []
for _, r in passed.iterrows():
    if r.data != "survey" or r.kind != "int": continue
    df = ns["SAMP"][r["sample"]]; s, X, a_ = ns["design"](df, r.outcome, r.aligned)
    sig = s.date.map(ns["SIGM"][r.signal][0]); z = (sig - sig.mean()) / sig.std(); ok = z.notna()
    XX = sm.add_constant(pd.concat([pd.DataFrame({"sig": z, "sigA": z * a_})[ok], X[ok]], axis=1), has_constant="add")
    m = sm.OLS(s.loc[ok, r.outcome].astype(float), XX).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(s.loc[ok, "uid"])[0]})
    C = m.cov_params(); b0, b1 = m.params["sig"], m.params["sigA"]
    def lin(w0, w1):
        v = w0 * w0 * C.loc["sig", "sig"] + 2 * w0 * w1 * C.loc["sig", "sigA"] + w1 * w1 * C.loc["sigA", "sigA"]
        return w0 * b0 + w1 * b1, float(np.sqrt(v))
    if r.aligned == "rep":
        lo, hi, lab = (0.0, 1.0, ("Democrat", "Republican"))
    else:
        lo, hi, lab = (-1.0, 1.0, ("align -1SD", "align +1SD"))
    sl, sh_ = lin(1, lo), lin(1, hi)
    rows.append(dict(sample=r["sample"], outcome=r.outcome, signal=r.signal, aligned=r.aligned, interaction=b1, interaction_se=float(np.sqrt(C.loc["sigA", "sigA"])),
                     group_low=lab[0], slope_low=sl[0], se_low=sl[1], group_high=lab[1], slope_high=sh_[0], se_high=sh_[1], n=int(ok.sum()), placebo_p=r.placebo_p))
g = pd.DataFrame(rows); g.to_csv(OUT / "09_i_group_slopes_261102.csv", index=False)
pd.set_option("display.width", 220, "display.float_format", lambda x: f"{x:.2f}"); print(g.to_string(index=False))
# chance benchmark
N = int((at.placebo_p.notna() & at.data.notna()).sum()); n_pass = int(at.passes_rule.sum()); n_any = int(at.placebo_le_10_any_sign.sum())
pc = 0.10 * 0.5
bench = dict(n_tests=N, n_pass_rule=n_pass, n_placebo_le_10_any_sign=n_any, chance_rate_per_test_rule=pc, expected_pass_by_chance=N * pc,
             binomial_tail_P_ge_observed_independent=float(stats.binom.sf(n_pass - 1, N, pc)),
             expected_any_sign=N * 0.10, binomial_tail_any_sign=float(stats.binom.sf(n_any - 1, N, 0.10)),
             note="independence assumed; tests are correlated variants of few signals, so the tail probability is optimistic (too small)")
pd.DataFrame([bench]).to_csv(OUT / "09_i_chance_benchmark_261102.csv", index=False); print(bench)
