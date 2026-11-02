# ============================================================
# START DATE   : 2026-11-02
# CODE RUN DATE: 2026-11-02
# TIME TAKEN   : ~60 min (trial + full run)
# DESCRIPTION  : Code NEW tariff passages (analysis window) with claude-haiku-5-5 via isolated Claude CLI. Modes: trial (60 passages, cost estimate), decide (firm subset if over cap), run (full coding).
# INPUT FILES  : data/35.../03_passages_all_261102.csv ; data/35.../01_sample_firms_261102.csv
# OUTPUT FILES : data/35.../04_trial_*, 04_budget_decision_261102.json, 04_firm_order_261102.csv, 04_llm_raw_261102.jsonl, 04_llm_calls_261102.csv ; analysis/35.../01_passage_codes_261102.csv
# ============================================================
import tempfile
import csv, json, os, re, subprocess, sys, tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "35_sec_tariff_expectations_261102"
ANA = ROOT / "analysis" / "35_sec_tariff_expectations_261102"
SCRATCH = Path(tempfile.gettempdir()) / "runs35"
CLAUDE = Path.home() / ".vscode/extensions/anthropic.claude-code-2.1.314-darwin-arm64/resources/native-binary/claude"
MODEL = "claude-haiku-5-5"
SEED, CAP, BATCH, WORKERS = 20261102, 15.0, 6, 10
WIN_BEFORE, WIN_AFTER, MAX_CHARS = 1000, 2000, 3000

ACTIONS = ["price_increase", "sourcing_shift_away_china", "sourcing_shift_to_us_or_nearshore", "supplier_diversification_other",
           "inventory_pull_forward", "investment_delay_or_pause", "investment_increase_us", "wait_for_clarity", "cost_mitigation_other"]
SCHEME = {"tense": ["in_force_or_past", "future", "both", "none"],
          "certainty": ["explicit_expectation", "conditional", "possibility_only", "not_future"],
          "direction": ["higher", "unchanged", "lower", "mixed_unclear", "not_stated"],
          "duration": ["lasting_stated", "temporary_stated", "uncertain_stated", "no_duration_stated"],
          "exposure_only": ["yes", "no"]}
STATUS = ["planned", "underway", "completed", "none"]

# ★ CORE LOGIC: coding instruction fixed before any model run (see 00_spec_261102.md section 2). Changes go in the spec's change log.
INSTR = """You are coding one paragraph at a time from SEC filings (10-K/10-Q) of US public companies. Each paragraph mentions tariffs. "the Company" is the filer. Code only what the paragraph text says. Do not use outside knowledge.
For EACH paragraph return one JSON object with exactly these keys:
- id: the paragraph id given
- tense: "in_force_or_past" (describes tariffs already in force, or past events/costs), "future" (makes a claim about tariffs or their effects not yet in force), "both", or "none" (tariffs mentioned without any time reference, e.g. a definition)
- certainty: for the FUTURE part only. "explicit_expectation" (the Company states what it expects, e.g. "we expect tariffs to remain / increase"), "conditional" (an if/then or scenario statement, e.g. "if tariffs increase, our costs would be higher"), "possibility_only" (only that something could/may/might happen, e.g. "tariffs could increase", risk-factor language), or "not_future" (tense is in_force_or_past or none). A statement that tariffs "may" or "could" change is possibility_only, NOT an expectation.
- direction: direction of the EXPECTED tariff level or burden in the future claim: "higher", "unchanged", "lower", "mixed_unclear" (change expected or possible but direction unclear or both ways), or "not_stated" (no future claim about direction)
- duration: "lasting_stated" (says tariffs or their effects will persist / be long-term / permanent), "temporary_stated" (says they are short-lived or reversible), "uncertain_stated" (explicitly says how long they will last is uncertain), or "no_duration_stated" (does not address duration; use this when the paragraph is silent)
- exposure_only: "yes" if the paragraph only describes tariffs in force or their cost impact, without any choice or action by the Company; otherwise "no"
- actions: an object with exactly these nine keys, each set to "planned", "underway", "completed", or "none": price_increase (raising prices or adding surcharges), sourcing_shift_away_china (moving sourcing or production out of China), sourcing_shift_to_us_or_nearshore (moving sourcing or production to the US, Mexico, Canada or nearby), supplier_diversification_other (adding suppliers or other countries not covered above), inventory_pull_forward (building or accelerating inventory purchases ahead of tariffs), investment_delay_or_pause (delaying, pausing or cutting investment or hiring because of tariffs or uncertainty), investment_increase_us (increasing US investment or capacity), wait_for_clarity (explicitly deferring decisions until policy is clearer), cost_mitigation_other (other cost-saving or mitigation steps). Use "planned" for stated intentions or steps being considered with a concrete named action, "underway" for in-progress, "completed" for done. Generic statements that the Company "may" take steps are "none" unless a concrete action is named.
- quote: the shortest verbatim span copied exactly from the paragraph that best supports the codes (empty string if nothing to quote)
Return ONLY a JSON array with one object per paragraph in the same order, no prose, no code fences."""

SUFFIX = re.compile(r"[,.\s]+(inc|incorporated|corp|corporation|co|company|ltd|limited|plc|llc|lp|holdings?|group|n\.v|s\.a|ag|sa|nv|class [a-c]|new|the)\.?\s*$", re.I)
STOP_FIRST = {"American", "General", "United", "First", "International", "National", "Global", "Western", "Southern", "Northern", "Eastern",
              "Advanced", "Applied", "Texas", "Union", "Pool", "Digital", "Dollar", "Home", "Smith", "Ross", "Best", "Core", "Fast", "Level", "Live"}

def mask_fn(name, sec_title, ticker):
    # ★ IMPORTANT: firm names (case-insensitive), ticker and first word of the name (case-sensitive, if not a generic word) -> "the Company".
    # Brand names beyond these cannot be fully detected (limit recorded in the results note).
    full, first = set(), set()
    for n in {name, sec_title or ""}:
        n = n.strip()
        if not n: continue
        c = n
        for _ in range(3): c = SUFFIX.sub("", c)
        for v in {n, c, n.replace(".", ""), c.replace(".", ""), c.replace("&", "and")}:
            if len(v) >= 2: full.add(v); full.add(v.replace("'", "\u2019"))
        fw = c.split()[0] if c.split() else ""
        if len(fw) >= 4 and fw not in STOP_FIRST and fw[0].isupper(): first.add(fw); first.add(fw.replace("'", "\u2019"))
    def rx(vs, flags):
        vs = sorted(vs, key=len, reverse=True)
        return re.compile(r"(?<![A-Za-z0-9])(" + "|".join(re.escape(v) for v in vs) + r")(?![A-Za-z0-9])", flags) if vs else None
    rxs = [rx(full, re.I), rx(first, 0)]
    if len(ticker) >= 3: rxs.append(re.compile(r"(?<![A-Za-z])" + re.escape(ticker) + r"(?![A-Za-z])"))
    def f(text):
        k = 0
        for r_ in rxs:
            if r_ is not None:
                text, n = r_.subn("the Company", text); k += n
        text = re.sub(r"(the Company)(\s+the Company)+", "the Company", text)
        return text, k
    return f

def window(text):
    if len(text) <= MAX_CHARS: return text
    i = re.search(r"tariff", text, re.I).start()
    s = max(0, i - WIN_BEFORE); return text[s:s + WIN_BEFORE + WIN_AFTER]

def prep(rows, firms):
    mk = {r.ticker: mask_fn(r.name, r.sec_title, r.ticker) for r in firms.itertuples()}
    out = []
    for r in rows.to_dict("records"):
        sent, k = mk[r["ticker"]](window(r["text"]))
        r["sent_text"], r["n_masked"], r["n_chars_sent"] = sent, k, len(sent); out.append(r)
    return out

def build_prompt(batch):
    items = [dict(id=f"p{j+1}", filing_date=r["filing_date"], text=r["sent_text"]) for j, r in enumerate(batch)]
    return INSTR + "\n\nPARAGRAPHS:\n" + json.dumps(items, ensure_ascii=False)

def validate(batch, arr):
    if not isinstance(arr, list) or [a.get("id") for a in arr] != [f"p{j+1}" for j in range(len(batch))]: raise ValueError("ids mismatch")
    for a in arr:
        for k, v in SCHEME.items():
            if a.get(k) not in v: raise ValueError(f"bad {k}: {a.get(k)}")
        if not isinstance(a.get("actions"), dict) or set(a["actions"]) != set(ACTIONS) or any(x not in STATUS for x in a["actions"].values()): raise ValueError("bad actions")
        if not isinstance(a.get("quote"), str): raise ValueError("no quote")
    return arr

def call(args):
    idx, batch, raw_path = args
    prompt = build_prompt(batch)
    res = dict(call=idx, n=len(batch), attempts=0, ok=False, models="", cost_usd=0.0, in_tok=0, out_tok=0, err="")
    codes = None
    for attempt in (1, 2):
        res["attempts"] = attempt
        run = Path(tempfile.mkdtemp(prefix=f"r{idx:05d}_{attempt}_", dir=SCRATCH))   # empty cwd per call
        env = dict(os.environ, CLAUDE_CODE_DISABLE_CLAUDE_MDS="1")
        try:
            p = subprocess.run([str(CLAUDE), "-p", "--model", MODEL, "--tools", "", "--strict-mcp-config", "--no-session-persistence", "--output-format", "json"],
                               input=prompt, capture_output=True, text=True, cwd=run, env=env, timeout=400)
            j = json.loads(p.stdout)
            u = j.get("usage", {})
            res["cost_usd"] += j.get("total_cost_usd") or 0
            res["in_tok"] += sum(u.get(k, 0) for k in ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens"))
            res["out_tok"] += u.get("output_tokens", 0)
            res["models"] = ",".join(j.get("modelUsage", {}).keys())
            with open(raw_path, "a") as f: f.write(json.dumps(dict(call=idx, attempt=attempt, ids=[r["passage_id"] for r in batch], response=j), ensure_ascii=False) + "\n")
            txt = (j.get("result") or "").strip()
            if txt.startswith("```"): txt = txt.strip("`").split("\n", 1)[1].rsplit("```", 1)[0]
            if MODEL not in res["models"] or "session limit" in txt.lower() or "hit your" in txt.lower(): raise RuntimeError("limit/model: " + txt[:100])
            codes = validate(batch, json.loads(txt)); res["ok"] = True; res["err"] = ""
            break
        except Exception as e:
            res["err"] = repr(e)[:300]
    return res, codes

def run_batches(items, raw_path):
    SCRATCH.mkdir(parents=True, exist_ok=True)
    batches = [(i, items[k:k + BATCH], raw_path) for i, k in enumerate(range(0, len(items), BATCH))]
    with ThreadPoolExecutor(WORKERS) as ex: results = list(ex.map(call, batches))
    out = []
    for (i, batch, _), (res, codes) in zip(batches, results):
        for j, a in enumerate(codes or []):
            r = batch[j]
            row = dict(ticker=r["ticker"], sector=r["sector"], form=r["form"], filing_date=r["filing_date"], accession=r["accession"], passage_id=r["passage_id"],
                       sha256=r["sha256"], n_masked=r["n_masked"], n_chars_sent=r["n_chars_sent"], tense=a["tense"], certainty=a["certainty"], direction=a["direction"],
                       duration=a["duration"], exposure_only=a["exposure_only"], quote=a["quote"], quote_verbatim=int(a["quote"] == "" or a["quote"] in r["sent_text"]))
            for k in ACTIONS: row["act_" + k] = a["actions"][k]
            out.append(row)
    return out, [r for r, _ in results]

def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

def main():
    mode = sys.argv[1]
    firms = pd.read_csv(DATA / "01_sample_firms_261102.csv")
    allp = pd.read_csv(DATA / "03_passages_all_261102.csv", dtype={"accession": str})
    new = allp[(allp.baseline == 0) & (allp.status == "NEW")].reset_index(drop=True)
    if mode == "trial":
        out_t = ANA / "00_trial_codes_261102.csv"
        if out_t.exists(): sys.exit("exists; refusing")
        samp = new.sample(60, random_state=SEED)
        items = prep(samp, firms)
        out, calls = run_batches(items, DATA / "04_trial_raw_261102.jsonl")
        write_csv(out_t, out); write_csv(DATA / "04_trial_calls_261102.csv", calls)
        cost = sum(c["cost_usd"] for c in calls)
        json.dump(dict(n=60, cost=cost, cost_per_passage=cost / 60, failed_calls=sum(not c["ok"] for c in calls), mean_chars_sent=float(np.mean([i["n_chars_sent"] for i in items]))), open(DATA / "04_trial_summary_261102.json", "w"), indent=1)
        print(open(DATA / "04_trial_summary_261102.json").read())
    elif mode == "decide":
        out_d = DATA / "04_budget_decision_261102.json"
        if out_d.exists(): sys.exit("exists; refusing")
        tr = json.load(open(DATA / "04_trial_summary_261102.json"))
        pp = tr["cost_per_passage"]
        rng = np.random.default_rng(SEED)
        order = list(firms.ticker.iloc[rng.permutation(len(firms))])
        cnt = new.groupby("ticker").size()
        total = int(cnt.sum()); proj_all = total * pp * 1.10
        budget = CAP - tr["cost"]
        K, cum = 0, 0
        for t in order:
            c = int(cnt.get(t, 0))
            if (cum + c) * pp * 1.10 > budget: break
            cum += c; K += 1
        if proj_all <= budget: K, cum = len(order), total
        keep = order[:K]
        pd.DataFrame(dict(ticker=order, rank=range(1, len(order) + 1), included=[t in set(keep) for t in order])).to_csv(DATA / "04_firm_order_261102.csv", index=False)
        json.dump(dict(new_passages_total=total, cost_per_passage_trial=pp, projected_all_with_10pct_margin=proj_all, trial_cost=tr["cost"], cap=CAP,
                       subset_needed=proj_all > budget, firms_total=len(order), firms_kept=K, passages_kept=cum, projected_kept=cum * pp * 1.10), open(out_d, "w"), indent=1)
        print(open(out_d).read())
    elif mode == "run":
        out_c = ANA / "01_passage_codes_261102.csv"
        if out_c.exists(): sys.exit("exists; refusing")
        o = pd.read_csv(DATA / "04_firm_order_261102.csv")
        keep = set(o[o.included].ticker)
        todo = new[new.ticker.isin(keep)].sort_values(["ticker", "filing_date", "passage_id"])
        items = prep(todo, firms)
        print("to code", len(items), flush=True)
        out, calls = run_batches(items, DATA / "04_llm_raw_261102.jsonl")
        write_csv(out_c, out); write_csv(DATA / "04_llm_calls_261102.csv", calls)
        print("coded", len(out), "calls", len(calls), "failed", sum(not c["ok"] for c in calls), "cost", round(sum(c["cost_usd"] for c in calls), 3))

if __name__ == "__main__": main()
