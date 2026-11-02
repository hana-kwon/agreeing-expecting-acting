# ============================================================
# START DATE   : 2026-10-30
# CODE RUN DATE: 2026-10-31
# TIME TAKEN   : < 1 s (imported by every step)
# DESCRIPTION  : Shared paths, run options and helpers. Every step reads from data/ and config/ and from
#                earlier steps in the same run folder, and writes only inside that run folder.
# INPUT FILES  : config/questions.json, <run>/run_options.json (written by run_all.py)
# OUTPUT FILES : none
# ============================================================
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
PUBLIC = DATA / "public"
CACHE = DATA / "cache"
RAW = DATA / "raw"
CONFIG = json.loads((ROOT / "config/questions.json").read_text())
QUESTIONS = {k: v for k, v in CONFIG.items() if not k.startswith("_")}
NLI_MODEL = "MoritzLaurer/deberta-v3-large-zeroshot-v2.0"

TARIFF_RX = re.compile(r"tariff|duty|duties|percent|%", re.I)
SENT_RX = re.compile(r"(?<=[.!?])\s+")

# Measure rules (the same as the full statement measure in the proposal):
EVIDENCE_MIN = 0.10      # a passage enters if P(evidence) >= 0.10
MIN_MASS = 3.0           # a 28-day window needs evidence weight >= 3 to make a call
WINDOW_DAYS = 28
HORIZONS = (30, 60, 90)
FIRST_ORIGIN = "2025-01-20"
CHECK_END = "2026-10-21"  # last date of the checked duty schedule

N_BOOT = 2000             # week-block bootstrap draws for AUC intervals


def run_dir():
    """The run folder is passed as the first argument by run_all.py."""
    if len(sys.argv) < 2:
        raise SystemExit("usage: python src/<step>.py <run_folder>")
    return Path(sys.argv[1]).resolve()


def options():
    """Run options chosen in run_all.py (cache use, optional reruns)."""
    p = run_dir() / "run_options.json"
    return json.loads(p.read_text()) if p.exists() else {}


def windows(text, scope_regex):
    """Three-sentence windows that mention a tariff (and the scope, if any). NLI models read short texts, so a
    long speech passage is split and each hypothesis keeps its highest entailment over the windows."""
    sents = [s for s in SENT_RX.split(str(text)) if s.strip()]
    ws = [" ".join(sents[i:i + 3]) for i in range(0, max(1, len(sents) - 2))] or [str(text)]
    srx = re.compile(scope_regex, re.I) if scope_regex else None
    keep = [w for w in ws if TARIFF_RX.search(w) and (srx is None or srx.search(w))]
    if not keep:
        keep = [w for w in ws if TARIFF_RX.search(w)]
    return keep or ws[:1]


def duty_on(steps, d):
    """Common additional U.S. duty on non-exempt goods of China in force on date d (percent)."""
    v = 0.0
    for sd, x in steps:
        if sd <= d:
            v = x
    return v


def sign(x):
    return "up" if x > 1e-9 else "down" if x < -1e-9 else "same"


def auc(score, y):
    """★ KEY FUNCTION: Mann-Whitney AUC with ties counted as half (0.5 = chance)."""
    s, y = np.asarray(score, float), np.asarray(y, int)
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    return (np.sum(pos[:, None] > neg[None, :]) + 0.5 * np.sum(pos[:, None] == neg[None, :])) / (len(pos) * len(neg))


def week_draws(df, n=N_BOOT, seed=7):
    """★ IMPORTANT: resample whole weeks (all country groups of a week together), since outcomes in the same
    week share news. Returns a list of resampled data frames."""
    rng = np.random.default_rng(seed)
    weeks = df["week"].unique()
    by = {w: x for w, x in df.groupby("week")}
    return [__import__("pandas").concat([by[w] for w in rng.choice(weeks, len(weeks))]) for _ in range(n)]
