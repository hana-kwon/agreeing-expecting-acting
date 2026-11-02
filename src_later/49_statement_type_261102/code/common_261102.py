# ============================================================
# START DATE   : 2026-11-02
# CODE RUN DATE: 2026-11-02
# TIME TAKEN   : ~5 min
# DESCRIPTION  : Shared paths, fixed coding fields and prompt for the statement typology (49).
# INPUT FILES  : team corpus files (read only), Jev cache from code/20 (outside the project folder)
# OUTPUT FILES : none
# ============================================================
from pathlib import Path

HERE = Path(__file__).resolve()
TOPIC = HERE.parents[2]
DB = next(p for p in HERE.parents if p.name == "the author")
TEAM = DB / "PRIVATE_DATA_ROOT/TEAM_PRIVATE"
CORPUS = TEAM / "ideology_share/directional_coding/data/jev_corpus"
SCORING = CORPUS / "scoring_passages.parquet"
OCC = CORPUS / "passage_occurrences.parquet"
EVQ = CORPUS / "simple_time_series_v1/event_question_option_probabilities.parquet"
JEV_CACHE = Path.home() / "research_local/orgsci_ai_frontiers/topic1_tirade_261030/jev_tariff_answers_261030.parquet"
PSET = Path.home() / "research_local/orgsci_ai_frontiers/topic1_tirade_261030/passage_set_261030.parquet"
import os
OUT = Path(os.environ.get("S49_OUT", TOPIC / "analysis/49_statement_type_261102"))   # S49_* env vars are only for dry runs on partial codes
VARDIR = Path(os.environ.get("S49_VAR", Path.home() / "private_cache/trump_261102_49/variants"))
LOCAL = Path.home() / "private_cache/trump_261102_49"      # raw CLI output, outside the project folder
SCRATCH = Path.home() / "private_cache/scratch_49"          # empty cwd for the CLI
CLI = Path.home() / ".vscode/extensions/anthropic.claude-code-2.1.314-darwin-arm64/resources/native-binary/claude"
MODEL = "claude-haiku-5-5"
SEED = 20261102
BATCH = 10

# ★ IMPORTANT: coding fields fixed in code before any coding call (spec 00_spec_261102.md).
TYPES = ["in_force_description", "past_action", "opinion_or_justification", "threat_or_possible_future",
         "stated_intention_future", "decision_announcement", "not_about_tariffs"]
FUTURE3 = {"threat_or_possible_future", "stated_intention_future", "decision_announcement"}
DIRS = ["higher", "lower", "same", "unclear"]
FIELDS_TEXT = """statement_type = in_force_description / past_action / opinion_or_justification / threat_or_possible_future / stated_intention_future / decision_announcement / not_about_tariffs; target (china, named other country, all, product, none); specificity flags: names_rate, names_date; direction (higher, lower, same, unclear)."""

SYSTEM = """You code statements by a political leader (Donald Trump) about import tariffs. Each item is a passage from a speech, press exchange, or social media post, with its date. Code what the leader says in the passage about U.S. import tariffs (duties on imported goods). Judge by the time and force of the statement relative to the passage date.

Definitions of statement_type (choose the ONE type that is most directly about the future path of tariff rates; if none is about the future, choose the dominant type):
- in_force_description: describes a tariff rate or policy that is currently in effect, or is being put into effect now (for example "we have a 25% tariff on steel", "tariffs are in place on China").
- past_action: describes a tariff action taken in the past (for example first-term tariffs in 2018-2019, "I put tariffs on China and they paid").
- opinion_or_justification: opinion about tariffs, praise, argument for why tariffs work or who pays, claims about revenue or effects, with no statement of a specific coming action.
- threat_or_possible_future: a conditional threat or possibility ("if they do not, we will put a tariff", "we could do 100%", "maybe we will").
- stated_intention_future: a general intention or plan without a decision ("we will be putting tariffs on pharmaceuticals", "tariffs are coming", "I am going to raise tariffs").
- decision_announcement: announces a decision already taken or an order being issued with a rate or start ("I am announcing 25% tariffs effective Friday", "I signed an order today").
- not_about_tariffs: the passage is not about import tariffs or duties.
Also give all_types: the list of every statement_type that appears in the passage about tariffs.

Other fields:
- target: a list with one or more of: "china", "all" (all or most countries / across the board), "product" (named product or sector), "none", or "country:<name>" for another named country. Include every target the tariff statements concern.
- names_rate: true if the passage states a numeric tariff rate or percentage.
- names_date: true if the passage states a date or specific time at which a tariff takes or would take effect (or a deadline).
- direction: the direction of the tariff rate the statement points to: higher, lower, same, or unclear.
"""

def build_prompt(items):
    blocks = []
    for i, r in enumerate(items, 1):
        ctx = str(r.get("context_before") or "").strip()
        blocks.append(f"### Item {i}\nDate: {r['date']}\n" + (f"Context before (question asked): {ctx[:400]}\n" if ctx else "")
                      + f"Passage:\n{str(r['passage'])[:3600]}")
    return (SYSTEM + "\nReturn ONLY a JSON array with one object per item, in order, each with keys: "
            '"item" (number), "statement_type", "all_types" (list), "target" (list), "names_rate" (true/false), '
            '"names_date" (true/false), "direction". No other text.\n\n' + "\n\n".join(blocks))
