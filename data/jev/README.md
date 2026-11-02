# Jev scores for the full statement record

Answer probabilities of Jev (TypeSafe AI, model `jev-1.13.0`, specification `typesafe-choice-spec-v3`) for 44,607 passages from 12,142 speeches, interviews, remarks, and posts by Donald Trump (January 2024 to October 2026), scored against questions drawn from the owner survey. These are the scores behind the Jev results in the proposal.

| File | Contents |
|---|---|
| `passage_scores.parquet` | One row per passage, question, and answer option: `score_id`, `question_id`, `option`, `probability`, `is_model_choice`, `model`, `spec_version` (6,021,945 rows) |
| `passages.parquet` | One row per occurrence of a scored passage: source, date, document type, URL, speaker status, SHA-256 of the passage, length, and the passage text (45,738 rows) |
| `event_question_option_probabilities.parquet` | Event-level option probabilities used for the dated measures (52,570 rows) |
| `manifest.json` | SHA-256 and row counts of the files and of the source files they were built from |

Passage text is included for the Compilation of Presidential Documents (govinfo.gov, U.S. government works) and Truth Social posts (public archive, CC0-1.0). Transcripts from the Factbase archive (rollcall.com/factbase) are not redistributed: for those rows the text is blank and the date, URL, and SHA-256 identify the passage. Question identifiers of the form `I<hash>_direction` are survey items; `tariff_*` questions ask about the stated path, duration, resolution, role, and ultimate level of tariffs on all goods and on goods of China.
