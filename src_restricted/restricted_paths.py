# ============================================================
# START DATE   : 2026-11-01
# CODE RUN DATE: not run in this repository (needs restricted data)
# TIME TAKEN   : n/a
# DESCRIPTION  : Single place where the restricted-data scripts find their inputs and write their outputs.
#                Set RESTRICTED_DATA_DIR to the folder that holds the restricted inputs (layout in README.md).
#                Set RESTRICTED_OUTPUT_DIR to choose where results go (default: outputs/restricted/ in this repository).
# INPUT FILES  : none (paths only)
# OUTPUT FILES : none
# ============================================================
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _root():
    v = os.environ.get("RESTRICTED_DATA_DIR")
    if not v:
        sys.exit("RESTRICTED_DATA_DIR is not set. These analyses need the restricted survey and customs data, which are "
                 "not in the repository. Set RESTRICTED_DATA_DIR to the folder that holds them (layout: see README.md).")
    p = Path(v).expanduser().resolve()
    if not p.is_dir():
        sys.exit(f"RESTRICTED_DATA_DIR={p} is not a folder")
    return p


RESTRICTED_ROOT = _root()
SURVEY_DIR = RESTRICTED_ROOT / "survey"        # inputs/analysis_all.dta, outputs/alignment_for_stata.dta, trump_alignment_v2/...
CUSTOMS_DIR = RESTRICTED_ROOT / "customs"      # smb_trade_duns_biweekly_260512.parquet, china_sentiment/..., hs6_panel/...
JEV_DIR = RESTRICTED_ROOT / "jev_scores"       # event_question_option_probabilities.parquet (proprietary classifier scores)
OUTPUT_ROOT = Path(os.environ.get("RESTRICTED_OUTPUT_DIR", REPO / "outputs/restricted")).expanduser().resolve()


def out_dir(name):
    """Output folder for one analysis; created if missing (outputs/ is ignored by git)."""
    p = OUTPUT_ROOT / name
    p.mkdir(parents=True, exist_ok=True)
    return p


def fresh_out_dir(name):
    """Output folder that must not exist yet (the script itself creates it, so a rerun never overwrites results)."""
    p = OUTPUT_ROOT / name
    p.parent.mkdir(parents=True, exist_ok=True)
    return p
