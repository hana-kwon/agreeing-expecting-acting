# ============================================================
# START DATE   : 2026-11-02
# CODE RUN DATE: 2026-11-02
# TIME TAKEN   : ~10 min
# DESCRIPTION  : A1. Re-runs the W4 same-owner expectation slopes of 10_slopes_no_agreement_261101.py (planned,
#                done, 25% scenario; price and investment; planned sourcing) adding the team controls
#                importer_dum, export_dum, fcomp_dum (V1) and party (V2). Reports old vs new side by side, and a
#                complete-case sensitivity (old and V1 on the same sample, as the team's reghdfe drops missing).
# INPUT FILES  : see common_261102.py
# OUTPUT FILES : analysis/40_smb_pricing_controls_261102/01_a1_controls_261102.csv (aggregates only)
# ============================================================
import pandas as pd

from common_261102 import (BASE_CTRL, TEAM_CTRL, dummies, get_out, load_survey, ols, safe_write)

out, args = get_out()
d = load_survey()
w4 = d[d.wave.astype(str).str.contains("W4")].copy()
early = d[d.wave.astype(str).str.contains("W1|W2|W3")].copy()
E = "import_tariff_expect"
Y = [f"{a}_post{k}_1" for a in ("prices", "invest") for k in (1, 2, 3)] + \
    ["import_loc_post2_1__toward", "import_loc_post2_3__away"]
MISS = ["miss_" + c for c in TEAM_CTRL]


def fit(df, y, spec, E):
    s = df.dropna(subset=[y, E]).copy()
    cols = [E] + BASE_CTRL
    if spec in ("V1", "V2"):
        cols += TEAM_CTRL + MISS
    parts = [s[cols].astype(float), dummies(s.wave.astype(str), "w"), dummies(s.industry_code.astype(int), "i"),
             dummies(s.three_info.fillna(0).astype(int), "m")]
    if spec == "V2":
        parts.append(dummies(s.party, "p"))
    X = pd.concat(parts, axis=1)
    m = ols(s, y, X)
    row = dict(y=y, spec=spec, n=len(s), coef=m.params[E], se=m.bse[E])
    for c in TEAM_CTRL:     # coefficient on each added control (per 1 = dummy 0 -> 1)
        row["b_" + c] = m.params.get(c, float("nan"))
        row["se_" + c] = m.bse.get(c, float("nan"))
    row["dummy_const_dropped"] = [c for c in TEAM_CTRL if c not in m.params.index]
    return row


rows = []
# ★ IMPORTANT: W4 import_tariff_expect is asked only of importers, so importer_dum is constant (=1) in the W4
# estimation sample and drops out. W1-W3 (short_term_prior, asked of everyone) is added so that importer_dum varies;
# it is a post-hoc addition recorded in 00_spec   (  ).
BLOCKS = [("W4", "full", w4, "import_tariff_expect"), ("W4", "complete_case", w4[(w4[MISS].sum(axis=1) == 0)], "import_tariff_expect"),
          ("W1-W3", "full", early, "short_term_prior"), ("W1-W3", "complete_case", early[(early[MISS].sum(axis=1) == 0)], "short_term_prior")]
for wave_sample, sample_name, df, Ev in BLOCKS:
    for y in Y:
        specs = ["old", "V1"] if sample_name == "complete_case" else ["old", "V1", "V2"]
        for sp in specs:
            r = fit(df, y, sp, Ev)
            r["sample"] = sample_name
            r["wave_sample"] = wave_sample
            r["expectation"] = Ev
            rows.append(r)
res = pd.DataFrame(rows)
res["dummy_const_dropped"] = res.dummy_const_dropped.apply(lambda x: ",".join(x))
safe_write(res, out / "01_a1_controls_261102.csv")
pd.set_option("display.width", 220, "display.float_format", lambda x: f"{x:.3f}")
print(res[["wave_sample", "sample", "y", "spec", "coef", "se", "n", "dummy_const_dropped"]].to_string(index=False))
print(res[res.spec == "V1"][["wave_sample", "sample", "y", "b_importer_dum", "se_importer_dum", "b_export_dum", "se_export_dum", "b_fcomp_dum", "se_fcomp_dum"]].to_string(index=False))
print("W4 n:", len(w4), "importer observed:", int((w4.miss_importer_dum == 0).sum()), "complete-case:", int((w4[MISS].sum(axis=1) == 0).sum()))
