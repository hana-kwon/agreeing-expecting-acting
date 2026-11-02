# ============================================================
# START DATE   : 2026-10-29
# CODE RUN DATE: 2026-10-31
# TIME TAKEN   : ~1 s from cache; ~40 min with --rerun-nli on Apple MPS (longer on CPU)
# DESCRIPTION  : Measurement step 2. Entailment probabilities of a public NLI classifier for each selected passage
#                and hypothesis (evidence gate, answer options, timing). By default copies the cached scores of the
#                reference run after checking that they cover exactly the selected passages; --rerun-nli recomputes
#                them.
# INPUT FILES  : <run>/01_selected.csv, data/public/public_tariff_statements.csv, data/cache/measure_nli_scores_deberta.csv
# OUTPUT FILES : <run>/02_scores.csv
# ============================================================
# Step 2. Score each selected passage with a public NLI zero-shot classifier. For every hypothesis in
# config/questions.json (evidence, answer options, timing) the score is the highest entailment probability over
# the passage's tariff windows. Output: <run>/02_scores.csv.
import shutil
import sys
import time

import pandas as pd

from common import CACHE, NLI_MODEL, PUBLIC as DATA, QUESTIONS, options, run_dir, windows

out = run_dir()
sel = pd.read_csv(out / "01_selected.csv")

if not options().get("rerun_nli"):
    # ★ IMPORTANT: the cache is used only if it holds exactly the passage-question pairs selected in step 1
    cached = pd.read_csv(CACHE / "measure_nli_scores_deberta.csv")
    a = set(zip(cached.question_id, cached.statement_id))
    b = set(zip(sel.question_id, sel.statement_id))
    if a != b:
        sys.exit(f"cached NLI scores do not match the selected passages ({len(a ^ b)} pairs differ); rerun with --rerun-nli")
    shutil.copyfile(CACHE / "measure_nli_scores_deberta.csv", out / "02_scores.csv")
    print(f"used cached scores for {len(cached)} passage-questions ({NLI_MODEL}, reference run)")
    sys.exit(0)

import torch  # noqa: E402  (only needed for a rerun)
from transformers import AutoModelForSequenceClassification, AutoTokenizer  # noqa: E402

st = pd.read_csv(DATA / "public_tariff_statements.csv").set_index("statement_id")
dev = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
tok = AutoTokenizer.from_pretrained(NLI_MODEL)
mdl = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL).to(dev).eval()
ENT = [i for i, l in mdl.config.id2label.items() if l.lower().startswith("entail")][0]


def entail(texts, hyp):
    ps = []
    for j in range(0, len(texts), 8):
        b = texts[j:j + 8]
        enc = tok(b, [hyp] * len(b), truncation="only_first", max_length=320, padding=True, return_tensors="pt").to(dev)
        with torch.no_grad():
            ps += torch.softmax(mdl(**enc).logits.float(), -1)[:, ENT].cpu().tolist()
    return max(ps)


t0, rows = time.time(), []
for i, r in enumerate(sel.itertuples()):
    spec = QUESTIONS[r.question_id]
    ws = windows(st.loc[r.statement_id, "text"], spec["scope_regex"])
    rec = dict(question_id=r.question_id, statement_id=r.statement_id, n_windows=len(ws), p_evidence=entail(ws, spec["evidence"]))
    for k, h in spec["options"].items():
        rec[f"p_{k}"] = entail(ws, h)
    for k, h in spec["timing"].items():
        rec[f"p_{k}"] = entail(ws, h)
    rows.append(rec)
    if i % 200 == 0:
        print(f"{i}/{len(sel)} {time.time() - t0:.0f}s", flush=True)
pd.DataFrame(rows).round(4).to_csv(out / "02_scores.csv", index=False)
print(f"scored {len(rows)} passage-questions on {dev} in {time.time() - t0:.0f}s with {NLI_MODEL}")
