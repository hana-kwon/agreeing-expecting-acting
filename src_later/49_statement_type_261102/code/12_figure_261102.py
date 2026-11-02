# ============================================================
# START DATE   : 2026-11-02
# CODE RUN DATE: 2026-11-02
# TIME TAKEN   : ~10 min
# DESCRIPTION  : Figure for the results note: (a) passages by statement type, (b) TiRADE check: in-force passages called future
#                (false future) and future claims detected, by measure. Data behind the figure saved next to it.
# INPUT FILES  : 03_counts_by_type_261102.csv, 04_validation_summary_261102.csv
# OUTPUT FILES : analysis/49_statement_type_261102/12_fig_types_validation_261102.png, 12_fig_data_261102.csv
# ============================================================
import importlib.util
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
_s = importlib.util.spec_from_file_location("c", Path(__file__).with_name("common_261102.py"))
C = importlib.util.module_from_spec(_s); _s.loader.exec_module(C)
f = C.OUT / "12_fig_types_validation_261102.png"; assert not f.exists(), "exists; refusing to overwrite"
cnt = pd.read_csv(C.OUT / "03_counts_by_type_261102.csv", index_col=0).sort_values("all_selected")
v = pd.read_csv(C.OUT / "04_validation_summary_261102.csv")
lab = {"orig_jev": "Original Jev", "nli": "Public NLI classifier", "haiku_prev": "Earlier Haiku (10-30)", "new_a_future3": "Typology: any future type",
       "new_b_threat_intent": "Typology: threat/intention only", "new_c_any_future3": "Typology: any future in all_types"}
v["label"] = v.measure.map(lab)
fig, ax = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [1, 1.15]})
ax[0].barh(cnt.index, cnt.all_selected, color="#4c6a92")
for i, n in enumerate(cnt.all_selected): ax[0].text(n + 8, i, str(n), va="center", fontsize=9)
ax[0].set_title("(a) 3,097 passages with Jev evidence >= 0.10 (China or broad path)\nby primary statement type (Haiku 5.5)", fontsize=10)
ax[0].set_xlabel("passages")
y = range(len(v)); ax[1].barh([i + 0.2 for i in y], v.false_future_in_force / 70, height=0.38, color="#c0504d", label="in-force passages called future (of 70)")
ax[1].barh([i - 0.2 for i in y], v.future_detected / 45, height=0.38, color="#4c6a92", label="threatened/announced detected (of 45)")
ax[1].set_yticks(list(y)); ax[1].set_yticklabels(v.label, fontsize=9); ax[1].invert_yaxis(); ax[1].set_xlim(0, 1)
ax[1].set_xlabel("share"); ax[1].legend(fontsize=8, loc="lower right")
ax[1].set_title("(b) Check against TiRADE labels: 115 passage-question pairs", fontsize=10)
plt.tight_layout(); plt.savefig(f, dpi=150)
pd.concat([cnt.assign(panel="a_counts").reset_index().rename(columns={"index": "statement_type"}), v.assign(panel="b_validation")]).to_csv(C.OUT / "12_fig_data_261102.csv", index=False)
