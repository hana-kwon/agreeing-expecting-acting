# ============================================================
# START DATE   : 2026-10-29
# CODE RUN DATE: 2026-10-31
# TIME TAKEN   : ~1 s
# DESCRIPTION  : Measurement step 1. Select the passages each question is asked of: statements that mention a tariff
#                and, for the China question, China.
# INPUT FILES  : data/public/public_tariff_statements.csv, config/questions.json
# OUTPUT FILES : <run>/01_selected.csv
# ============================================================
# Step 1. Select the passages each question is asked of: statements that mention a tariff and, for the
# China question, China. Output: <run>/01_selected.csv (question_id, statement_id).
import re

import pandas as pd

from common import PUBLIC as DATA, QUESTIONS, run_dir

out = run_dir()
st = pd.read_csv(DATA / "public_tariff_statements.csv")
rows = []
for q, spec in QUESTIONS.items():
    rx = re.compile(spec["scope_regex"], re.I) if spec["scope_regex"] else None
    sel = st[st.text.map(lambda t: rx is None or bool(rx.search(str(t))))]
    rows.append(pd.DataFrame({"question_id": q, "statement_id": sel.statement_id}))
S = pd.concat(rows, ignore_index=True)
S.to_csv(out / "01_selected.csv", index=False)
print(S.question_id.value_counts().to_dict())
