# Reading Politics, Making Strategy: research artifact

Research artifact for the proposal "Reading Politics, Making Strategy". The proposal measures what a political leader said about future tariffs, what small-business owners expect and plan, and what importers then do. This repository holds the parts that can be shared and shows how the restricted parts are run.

## What the artifact does

1. **Statement measurement (public data).** Selects the passages of a leader's public statements that mention tariffs, scores them on survey-style questions with a public classifier, aggregates the scores into a dated measure, and checks the measure against the official duty schedule on China and against the labels in TiRADE, a public database of tariff announcements.
2. **Corrected tariff calendar (public data).** Rebuilds the effective dates of U.S. tariff increases, and a country-by-Friday panel of whether an increase takes effect in the next 28 days, from Federal Register documents and TiRADE. Four variants show what each correction changes. The proposal uses variant `B_extended`.
3. **Analysis of the owner survey and of customs records (restricted data).** The scripts are included in `src_restricted/`. The data are not. Aggregate results from these scripts are in `reference/restricted_aggregates/`.
4. **Synthetic copies of the owner survey** (`data/synthetic_survey/`, five copies), released with the data owner's permission to share synthetic copies only. `src/08_synthetic_survey_estimates.py` runs the main survey estimates on the copies and combines them, without restricted data. Reproducing an estimate on them checks the code and the generator; it does not confirm the original result (see `data/synthetic_survey/README.md`).

Every run writes into a new folder together with `receipt.json`, which holds the SHA-256 of every input and output, package versions, options and the time of each step. A run refuses to write into a folder that already holds files.

## Quick start (public data, under two minutes on a laptop CPU)

```bash
python -m pip install -r requirements.txt     # numpy, pandas, scipy, statsmodels, patsy, matplotlib, pyarrow suffice
python run_all.py outputs/run_demo
```

The default run uses the cached classifier scores in `data/cache/`; the cache is used only if the inputs it answered are unchanged. `python run_all.py outputs/run_nli --rerun-nli` recomputes all 1,628 classifier scores (about 40 minutes on Apple silicon; needs `torch` and `transformers`).

## What runs on public data

| Step | Script | Input | Output |
|---|---|---|---|
| 1 to 7 | `src/01_select_passages.py` to `src/07_figure.py` | `data/public/public_tariff_statements.csv` (321 Truth Social posts and 994 passages of the Compilation of Presidential Documents that mention tariffs), `config/questions.json`, TiRADE, the China duty schedule, cached classifier scores | Weekly measure, forecast check against the duty schedule, comparison with TiRADE labels, a demonstration on synthetic respondents with arbitrary parameters (step 6), figure |
| 8 | `src/08_synthetic_survey_estimates.py` | the five synthetic copies in `data/synthetic_survey/`, the China duty schedule | `08_synthetic_survey_estimates.csv` (combined estimate, combined SE and per-copy values): message effects on the expected-tariff revision, same-owner slopes of planned price and investment on the expectation held before the message, and the 25 percent scenario slopes, estimated on the synthetic copy |
| 30 | `src/30_tariff_calendar.py` | Federal Register documents, the actions extracted from them, TiRADE, a list of 96 countries | `30_calendar_<variant>/` with actions, country events, increase events, the change rows of each correction, and the calendar-only panel (88 Fridays, 2025-01-24 to 2026-09-25) |
| 31 | `src/31_independent_events.py` | Output of step 30 | Positive cells, spells, calendar episodes and distinct effective dates under each claim, by variant |

Results of the reference run are in `reference/` (`receipt.json` records the run: Python 3.14.6, pandas 3.1.0, numpy 2.5.4, statsmodels 0.15.0, 57 seconds, all nine steps returned 0). Step 08 was added after that run; a new run of all ten steps (with the five synthetic copies) returned 0, and the outputs of steps 01 to 07, 30 and 31 are byte-identical to the reference outputs in `receipt.json`. The four measurement tables of steps 4 to 7 (`04_forecast_metrics`, `05_tirade_benchmark`, `06_synthetic_results`, `07_figure_data`) are byte-identical to the earlier reference run made with the pinned versions in `requirements.txt`. The actions, country events, increase events and change rows of step 30 and the output of step 31 are byte-identical to the project's working files; the calendar columns of the panel are equal to them. Two facts to keep in mind when reading them:

- The measurement uses only public statements that mention tariffs and a public classifier (Laurer et al. 2023). The proposal's numbers on the full record of 44,607 passages come from a proprietary classifier (Jev), so the two sets of numbers differ. Jev's answer probabilities for the full record are released in `data/jev/` (see its README). `reference/measurement/jev_agreement_summary.json` is an aggregate of the agreement between the two classifiers on 128 China passages, computed where the proprietary scores are available.
- Under variant `B_extended`, 566 of 8,448 country-weeks are followed by an increase within 28 days; they rest on 13 distinct effective dates and 17 Federal Register documents (`reference/tariff_calendar/31_independent_events.csv`).

The corrected calendar is in `data/public/tariff_calendar/corrected/` for all four variants, with the change log (`01_change_log_261101.csv`). `orig_repro` reproduces the original logic and has no change rows; `A_listed` applies six listed corrections; `B_extended` adds same-kind errors found while checking; `C_inferred` adds an inference (removal of 10 percent rows that replace the expiring Section 122 duty) that is not used in the proposal.

## What needs restricted data

| Part | Scripts | Restricted inputs | Aggregate results already in the repository |
|---|---|---|---|
| Owner survey: expectations, agreement, plans, message effects, follow-up | `src_restricted/survey/` | The survey files (Jev scores are in `data/jev/`) | `reference/restricted_aggregates/survey/` |
| Customs records: firm pauses and switches, policy episodes, seasonal reference | `src_restricted/customs/` | Seaborne customs records by importer, origin and two-week period | `reference/restricted_aggregates/customs/` |

The survey was fielded with a business-networking platform for small firms and the customs records come from a separate licensed source. Neither can be shared. To run the scripts, set one environment variable that points to a folder holding the inputs:

```bash
export RESTRICTED_DATA_DIR=/path/to/restricted        # required; the scripts stop with an error if it is unset
export RESTRICTED_OUTPUT_DIR=/path/to/results         # optional; default is outputs/restricted/
python src_restricted/survey/10_slopes_no_agreement_261101.py
python src_restricted/customs/pause_switch/01_pause_switch_261101.py
```

The folder layout and the order of the scripts are in `src_restricted/README.md`. The scripts in `src_restricted/` were adapted from the project's working code to read their inputs through this variable; they could not be executed in the repository because the data are absent, so they have been checked for syntax and for the error message when the variable is unset, not rerun end to end.

## What is synthetic or pending

- `src/06_synthetic_survey.py` simulates respondents with arbitrary parameters so that the comparison code runs. It says nothing about real owners.
- `data/synthetic_survey/` holds five statistical synthetic copies of the owner survey (generated from models of the original, so they contain no real respondent). They reproduce the original estimates only as far as the generator learned them from the original; some differ noticeably (see its README).
- The planned experiment (verbatim dated statements, with and without the leader's name, against a control that shows current official rates) is not fielded; `src_restricted/customs/extension/05_experiment_dryrun_261101.py` is a power calculation on synthetic allocations.

## Layout

```
run_all.py            entry point for the public-data pipeline (writes <run>/receipt.json)
config/               questions for the statement measure
src/                  steps 01-07 (statement measurement), 08 (estimates on the synthetic survey), 30-31 (corrected calendar)
src_restricted/       scripts for the survey and customs data (restricted inputs, one environment variable)
tools/                input manifest
data/                 public/ (statements, TiRADE, duty schedule, tariff_calendar/), cache/, synthetic_survey/ (five synthetic copies and codebook), raw/ (empty)
reference/            outputs of the reference run, restricted_aggregates/, tariff_calendar/
outputs/              new runs (not tracked)
```

## Anonymity

The repository contains no author names, institution names or personal paths, and its history starts with this version.

## Licenses and sources

- Code: MIT License (see `LICENSE`).
- Documents and the data files created for this project (the corrected tariff calendar, the measurement outputs and the reference outputs): CC BY 4.0. Third-party data keep the terms listed below. The aggregates in `reference/restricted_aggregates/` come from restricted sources and are not covered by the CC BY license.
- TiRADE: Kalemli-Ozcan, Soylu, and Yildirim (2026), version 1.5, Global Linkages Lab, Brown University; free to use and redistribute with citation.
- Truth Social posts: public archive at github.com/stiles/trump-truth-social-archive (CC0-1.0), continued from ix.cnn.io/data/truth-social/truth_archive.json. Presidential statements: Compilation of Presidential Documents (govinfo.gov, U.S. government works).
- Federal Register documents: federalregister.gov (U.S. government works).
- Public classifier: `MoritzLaurer/deberta-v3-large-zeroshot-v2.0` (MIT; Laurer, van Atteveldt, Casas, and Welbers 2023, arXiv:2312.17543).

## Later analyses

`src_later/` holds code and aggregate tables for the later exploratory analyses in appendix A9. See `src_later/README.md`.
