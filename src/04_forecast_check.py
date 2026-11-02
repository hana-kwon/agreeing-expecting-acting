# ============================================================
# START DATE   : 2026-10-29
# CODE RUN DATE: 2026-10-31
# TIME TAKEN   : ~1 s
# DESCRIPTION  : Measurement step 4. Does the measured China tariff path anticipate the official duty path 30, 60 and
#                90 days later? Baseline: no change.
# INPUT FILES  : <run>/03_event_measure.csv, data/public/china_additional_duty_steps.csv
# OUTPUT FILES : <run>/04_forecast_windows.csv, <run>/04_forecast_metrics.csv, <run>/04_window_shares.csv
# ============================================================
# Step 4. Does the measured China tariff path anticipate the official duty path?
# Every Monday from 2025-01-20, statements from the preceding 28 days (not the same day) give a call: the
# direction with the largest weight, if the window's evidence weight is at least 3. The call is scored against
# the change in the common additional duty on goods of China 30, 60 and 90 days later. Baseline: no change.
# Outputs: <run>/04_forecast_windows.csv, <run>/04_forecast_metrics.csv, <run>/04_window_shares.csv (28-day share 'higher' at each origin).
import csv
from datetime import date, timedelta

import pandas as pd

from common import CHECK_END, PUBLIC as DATA, FIRST_ORIGIN, HORIZONS, MIN_MASS, WINDOW_DAYS, duty_on, run_dir, sign

out = run_dir()
steps = sorted((date.fromisoformat(r["effective_date"]), float(r["rate_percent"]))
               for r in csv.DictReader(open(DATA / "china_additional_duty_steps.csv")))
ev = pd.read_csv(out / "03_event_measure.csv")
ev = ev[ev.question_id == "china_path"].copy()
ev["d"] = pd.to_datetime(ev.date).dt.date
last = pd.to_datetime(pd.read_csv(DATA / "public_tariff_statements.csv").date).max().date()
end = date.fromisoformat(CHECK_END)

rows, shares, t = [], [], date.fromisoformat(FIRST_ORIGIN)
while t <= last + timedelta(days=1):
    sub = ev[(ev.d >= t - timedelta(days=WINDOW_DAYS)) & (ev.d <= t - timedelta(days=1))]
    calls = {}
    for ver, pre, m in [("original", "w_", "mass"), ("revised", "wrev_", "mass_rev")]:
        v = {"up": sub[pre + "higher"].sum(), "down": sub[pre + "lower"].sum(), "same": sub[pre + "same"].sum()}
        best = max(v.values())
        win = [k for k, x in v.items() if abs(x - best) < 1e-12]
        ok = sub[m].sum() >= MIN_MASS and best > 0
        calls[ver] = win[0] if (ok and len(win) == 1) else "none"
        shares.append(dict(origin=t.isoformat(), version=ver, mass=round(float(sub[m].sum()), 3),
                           share_higher=round(v["up"] / sum(v.values()), 3) if ok else None))
    for h in HORIZONS:
        tgt = t + timedelta(days=h)
        if tgt > end:
            continue
        rows.append(dict(origin=t.isoformat(), horizon=h, realized=sign(duty_on(steps, tgt) - duty_on(steps, t)),
                         call_original=calls["original"], call_revised=calls["revised"]))
    t += timedelta(days=7)
W = pd.DataFrame(rows)
W.to_csv(out / "04_forecast_windows.csv", index=False)
pd.DataFrame(shares).to_csv(out / "04_window_shares.csv", index=False)
mets = []
for ver in ["original", "revised"]:
    c = W[W[f"call_{ver}"] != "none"]
    mets.append(dict(version=ver, windows=len(W), calls=len(c),
                     accuracy=round(float((c[f"call_{ver}"] == c.realized).mean()), 3) if len(c) else None,
                     no_change_accuracy_same_windows=round(float((c.realized == "same").mean()), 3) if len(c) else None,
                     share_calls_up=round(float((c[f"call_{ver}"] == "up").mean()), 3) if len(c) else None))
M = pd.DataFrame(mets)
M.to_csv(out / "04_forecast_metrics.csv", index=False)
print(M.to_string(index=False))
