# ============================================================
# START DATE   : 2026-10-29
# CODE RUN DATE: 2026-10-31
# TIME TAKEN   : ~1 s
# DESCRIPTION  : Measurement step 6. Synthetic owner survey with the structure of the real instrument (answer scale,
#                waves, randomized information, agreement groups) and arbitrary parameters; demonstrates that the owner
#                comparison code runs. It says nothing about real owners.
# INPUT FILES  : none (fixed seed)
# OUTPUT FILES : <run>/06_synthetic_respondents.csv, <run>/06_synthetic_results.csv
# ============================================================
# Step 6. Synthetic owner survey. It keeps the structure of the real instrument (answer scale, waves, a randomized
# information treatment, groups by agreement with the leader's positions) so the comparison code runs end to end
# without respondent records. All values are simulated from fixed, arbitrary parameters; they are not estimates
# from any real survey and say nothing about real owners.
# Outputs: <run>/06_synthetic_respondents.csv, <run>/06_synthetic_results.csv.
import numpy as np
import pandas as pd

from common import run_dir

out = run_dir()
rng = np.random.default_rng(20261030)
SCALE = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 80, 100]  # "100% or more" is the top category
WAVES = ["W1", "W2", "W3", "W4", "W5", "W6"]
N_PER_WAVE = 400


def snap(x):
    return SCALE[int(np.argmin([abs(x - s) for s in SCALE]))]


rows = []
for w in WAVES:
    for i in range(N_PER_WAVE):
        agree = rng.uniform()                       # share of other questions answered as the leader does
        group = "higher agreement" if agree >= 0.5 else "lower agreement"
        current = snap(rng.normal(25, 12))          # perceived average tariff today
        expect_pre = snap(current + rng.normal(5 if group == "higher agreement" else 8, 10))
        treated = int(rng.uniform() < 0.5)          # randomized policy information (W4 design)
        shift = treated * rng.normal(4, 6)
        expect_post = snap(expect_pre + shift + rng.normal(0, 4)) if w == "W4" else None
        rows.append(dict(respondent=f"S{w}{i:04d}", wave=w, agreement=round(agree, 3), group=group, current_tariff=current,
                         expected_year_end_pre=expect_pre, treated=treated if w == "W4" else None, expected_year_end_post=expect_post))
R = pd.DataFrame(rows)
R.to_csv(out / "06_synthetic_respondents.csv", index=False)

res = []
for (w, g), s in R.groupby(["wave", "group"]):
    res.append(dict(analysis="share expecting a year-end tariff above today's", wave=w, group=g, n=len(s),
                    estimate=round(float((s.expected_year_end_pre > s.current_tariff).mean()), 3), se=None))
w4 = R[R.wave == "W4"].assign(belief_update=lambda d: d.expected_year_end_post - d.expected_year_end_pre)
for g, s in w4.groupby("group"):
    a, b = s.loc[s.treated == 1, "belief_update"], s.loc[s.treated == 0, "belief_update"]
    res.append(dict(analysis="information effect on the year-end expectation (points)", wave="W4", group=g, n=len(s),
                    estimate=round(float(a.mean() - b.mean()), 2), se=round(float(np.sqrt(a.var() / len(a) + b.var() / len(b))), 2)))
pd.DataFrame(res).to_csv(out / "06_synthetic_results.csv", index=False)
print(pd.DataFrame(res).tail(4).to_string(index=False))
