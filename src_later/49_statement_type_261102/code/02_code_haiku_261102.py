# ============================================================
# START DATE   : 2026-11-02
# CODE RUN DATE: 2026-11-02
# TIME TAKEN   : ~50 min run
# DESCRIPTION  : Step 2. Code each selected passage with claude-haiku-5-5 through the isolated Claude CLI
#                (10 passages per call, 4 parallel). Failures (limit message, bad JSON, wrong count) are kept in the
#                log and retried once. Resumable: finished batches are skipped; raw results stay outside the project folder.
# INPUT FILES  : 01_selection_261102.csv, scoring_passages.parquet, passage_occurrences.parquet (read only)
# OUTPUT FILES : PRIVATE_CACHE, calls.jsonl (codes, tokens, cost; no text)
# ============================================================
import importlib.util, json, os, re, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import pandas as pd
_s = importlib.util.spec_from_file_location("c", Path(__file__).with_name("common_261102.py"))
C = importlib.util.module_from_spec(_s); _s.loader.exec_module(C)

C.LOCAL.mkdir(parents=True, exist_ok=True); C.SCRATCH.mkdir(parents=True, exist_ok=True)
assert not any(C.SCRATCH.iterdir()), "scratch cwd must be empty"
CODES, CALLS = C.LOCAL / "codes.jsonl", C.LOCAL / "calls.jsonl"
S = pd.read_csv(C.OUT / "01_selection_261102.csv")
sp = pd.read_parquet(C.SCORING)[["score_id", "date", "passage", "context_before"]]
S = S.merge(sp, on="score_id", how="left", validate="1:1")
assert S.passage.notna().all()
done = set()
if CALLS.exists():
    for l in open(CALLS):
        r = json.loads(l)
        if r["ok"]:
            done.add(r["batch"])
ENV = dict(os.environ, CLAUDE_CODE_DISABLE_CLAUDE_MDS="1")
LIMIT = re.compile(r"hit your (session )?limit|usage limit|rate limit", re.I)


def call(b):
    part = S[S.batch == b]
    prompt = C.build_prompt(part.to_dict("records"))
    t0 = time.time()
    rec = dict(batch=int(b), n=len(part), ok=False, err=None)
    try:
        r = subprocess.run([str(C.CLI), "-p", "--model", C.MODEL, "--tools", "", "--strict-mcp-config",
                            "--no-session-persistence", "--output-format", "json"],
                           input=prompt, capture_output=True, text=True, cwd=C.SCRATCH, env=ENV, timeout=420)
        j = json.loads(r.stdout)
        mu = j.get("modelUsage", {})
        rec.update(cost=j.get("total_cost_usd"), models=list(mu), usage=mu, seconds=round(time.time() - t0, 1))
        txt = j.get("result", "")
        if j.get("is_error") or LIMIT.search(txt):
            rec["err"] = "limit_or_error: " + txt[:120]
        elif list(mu) != [C.MODEL]:
            rec["err"] = f"unexpected model {list(mu)}"
        else:
            m = re.search(r"\[.*\]", txt, re.S)
            arr = json.loads(m.group(0)) if m else []
            if len(arr) != len(part):
                rec["err"] = f"count {len(arr)} != {len(part)}"
            else:
                ids = part.score_id.tolist()
                with open(CODES, "a") as f:
                    for a in arr:
                        k = int(a["item"]) - 1
                        f.write(json.dumps(dict(score_id=ids[k], batch=int(b), **{x: a.get(x) for x in
                                ["statement_type", "all_types", "target", "names_rate", "names_date", "direction"]})) + "\n")
                rec["ok"] = True
    except Exception as e:
        rec["err"] = f"{type(e).__name__}: {str(e)[:150]}"
        rec["seconds"] = round(time.time() - t0, 1)
    with open(CALLS, "a") as f:
        f.write(json.dumps(rec) + "\n")
    return rec


def run(batches, label):
    n_ok = 0
    with ThreadPoolExecutor(4) as ex:
        for i, rec in enumerate(ex.map(call, batches), 1):
            n_ok += rec["ok"]
            if i % 10 == 0 or not rec["ok"]:
                print(label, i, "/", len(batches), "ok" if rec["ok"] else rec["err"], flush=True)
    return n_ok


todo = [int(b) for b in sorted(S.batch.unique()) if b not in done]
print("batches to run", len(todo), flush=True)
run(todo, "pass1")
# ★ IMPORTANT: one retry pass only; failures stay in calls.jsonl.
done = {json.loads(l)["batch"] for l in open(CALLS) if json.loads(l)["ok"]}
retry = [b for b in todo if b not in done]
print("retry", len(retry), flush=True)
if retry:
    run(retry, "retry")
done = {json.loads(l)["batch"] for l in open(CALLS) if json.loads(l)["ok"]}
print("final failed batches", [b for b in S.batch.unique() if b not in done], flush=True)
