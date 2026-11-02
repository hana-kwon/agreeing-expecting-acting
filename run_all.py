# ============================================================
# START DATE   : 2026-10-29
# CODE RUN DATE: 2026-10-31
# TIME TAKEN   : under a minute for the default demo (cached model outputs) on a laptop CPU
# DESCRIPTION  : Runs the public-data pipeline in order: measurement of the leader's statements (steps 01-07) and the
#                corrected Federal Register tariff calendar (30-31). Refuses to write into a non-empty run folder.
#                Writes receipt.json with the SHA-256 of every input and output, package versions, options and the time of each step.
# INPUT FILES  : config/*.json, data/public/*, data/cache/*
# OUTPUT FILES : <run>/* and <run>/receipt.json
# ============================================================
"""Run the pipeline end to end.

    python run_all.py outputs/run_<name>                       # default: statement measurement + corrected tariff calendar
    python run_all.py outputs/run_<name> --rerun-nli           # recompute the NLI scores (about 40 minutes on MPS)

The analyses of restricted data (survey, customs) are not run here: see src_restricted/ and README.md.
"""
import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parent
# Default run: measurement of the leader's statements and the corrected tariff calendar, all on public data.
STEPS = ["01_select_passages.py", "02_score_passages.py", "03_aggregate_measure.py", "04_forecast_check.py",
         "05_tirade_benchmark.py", "06_synthetic_survey.py", "07_figure.py",
         "30_tariff_calendar.py", "31_independent_events.py"]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run")
    ap.add_argument("--rerun-nli", action="store_true")
    a = ap.parse_args()
    run = Path(a.run).resolve()
    # ★ IMPORTANT: a new run never mixes with an old one
    if run.exists() and any(run.iterdir()):
        raise SystemExit(f"{run} is not empty; choose a new run folder")
    run.mkdir(parents=True, exist_ok=True)
    opts = {"rerun_nli": a.rerun_nli}
    (run / "run_options.json").write_text(json.dumps(opts, indent=1))
    rec = {"started": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "python": platform.python_version(), "platform": platform.platform(),
           "options": opts, "packages": {}, "inputs": {}, "steps": [], "outputs": {}}
    for pkg in ["pandas", "numpy", "statsmodels", "matplotlib", "torch", "transformers"]:
        try:
            rec["packages"][pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            rec["packages"][pkg] = None
    base = ROOT
    steps = list(STEPS)
    for f in sorted(list((base / "config").glob("*.json")) + list((base / "data").rglob("*.*"))):
        if "raw" not in f.relative_to(base).parts and f.is_file() and f.suffix != ".md":
            rec["inputs"][str(f.relative_to(base))] = sha(f)
    t_all = time.time()
    for s in steps:
        t0 = time.time()
        r = subprocess.run([sys.executable, str(base / "src" / s), str(run)], cwd=base / "src", capture_output=True, text=True)
        rec["steps"].append({"step": s, "seconds": round(time.time() - t0, 1), "returncode": r.returncode, "stdout_tail": r.stdout[-1500:]})
        print(f"[{s}] {time.time() - t0:.0f}s\n{r.stdout[-1500:]}", flush=True)
        if r.returncode:
            print(r.stderr[-3000:])
            break
    for f in sorted(run.rglob("*")):
        if f.is_file() and f.name != "receipt.json":
            rec["outputs"][str(f.relative_to(run))] = sha(f)
    rec["finished"] = time.strftime("%Y-%m-%d %H:%M:%S %Z")
    rec["seconds_total"] = round(time.time() - t_all, 1)
    (run / "receipt.json").write_text(json.dumps(rec, indent=1))
    print(f"done in {rec['seconds_total']:.0f}s; receipt: {run / 'receipt.json'}")
    if any(s["returncode"] for s in rec["steps"]):
        sys.exit(1)


if __name__ == "__main__":
    main()
