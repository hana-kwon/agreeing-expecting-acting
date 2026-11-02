# Data

Everything in `public/` and `cache/` can be redistributed. Nothing in this folder is respondent-level survey data (`synthetic_survey/` holds generated rows only), customs data, output of the proprietary classifier, or text from sources that do not permit redistribution. `inputs_manifest.json` records the SHA-256 and row count of every file in `public/` and `cache/` and of `config/*.json`.

## public/

| File | Source and license |
|---|---|
| `tirade-v1.5-rate-statements.csv` | Kalemli-Özcan, Ş., C. Soylu, and M. A. Yildirim (2026). TiRADE: A Database of Tariff Rates in Announcements, Declarations, and Events, v1.5. Global Linkages Lab, Brown University. https://www.globallinkages.org/datasets/tirade/v1.5 (downloaded 2026-10-16; file unchanged). Free to use and redistribute with citation. 749 rows. |
| `public_tariff_statements.csv` | Statements by Donald Trump from 2025-01-20 to 2026-09-24 that mention tariffs: 321 Truth Social posts from the public archive at github.com/stiles/trump-truth-social-archive (CC0-1.0), continued at ix.cnn.io/data/truth-social/truth_archive.json, and 994 passages of his own words from the Compilation of Presidential Documents (govinfo.gov; U.S. government works), split into chunks of about 2,600 characters. |
| `china_additional_duty_steps.csv` | The common additional U.S. duty on non-exempt goods of China, with each change, its instrument and Federal Register document. |
| `tariff_calendar/01_fr_documents_261101.parquet` | 239 Federal Register documents (executive orders, proclamations, USTR notices) on tariff actions (signed 2025-01-19 to 2026-10-23, published 2025-01-23 to 2026-10-30; the search ran through 2026-10-31), with metadata and the document text. Federal Register, federalregister.gov (U.S. government works). |
| `tariff_calendar/03_actions_261101.csv`, `03_country_events_261101.csv` | The tariff actions (one row per action, with the quoted effective-date and rate text) and their country events, extracted from those documents; the quoted passages that fix each date and rate are kept in the file. Public text, derived by this project. |
| `tariff_calendar/country_list_261101.csv` | The 96 countries of the panel (ISO3 code, name, matching expression). |
| `tariff_calendar/corrected/` | The corrected calendar for the four variants (`orig_repro`, `A_listed`, `B_extended`, `C_inferred`): actions, country events, increase events, change rows of each correction, and `01_change_log_261101.csv`. Written by `src/30_tariff_calendar.py` and identical to its output. These files contain no classifier scores: the panels of the working project carry the proprietary classifier's scores and were not copied; step 30 writes calendar-only panels into each run folder. |


## synthetic_survey/

`synthetic_survey_w4_us_v2_1_copy1.csv` to `copy5.csv` are five statistical synthetic copies of the wave-4 U.S. owner survey (generated rows only, coarsened and locally suppressed background traits, no identifiers, exact dates or free text), `codebook.csv` lists its variables, and its README describes how it was made, what it can and cannot show, and the disclosure steps. It is not respondent-level survey data; the original stays private.

## cache/

| File | What it is |
|---|---|
| `measure_nli_scores_deberta.csv` | Entailment probabilities of `MoritzLaurer/deberta-v3-large-zeroshot-v2.0` (MIT license) for the 1,628 passage-question pairs of measurement step 2 (reference run of 2026-10-30). Used unless `--rerun-nli`. |

## raw/ (restricted, not distributed; ignored by git)

The default run needs nothing from `raw/`. Restricted inputs for the survey and customs analyses are read from the folder named by the environment variable `RESTRICTED_DATA_DIR`, not from this repository (layout in `src_restricted/README.md`). The owner survey (six waves, March to September 2025, fielded with a business-networking platform for small firms) and the customs records cannot be shared. The scores of the proprietary classifier that scored the full record of 44,607 passages are not distributed either. Aggregate results from these sources are in `reference/restricted_aggregates/`.
