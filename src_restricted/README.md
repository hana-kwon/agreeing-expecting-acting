# Analysis code for restricted data

These scripts need data that cannot be shared. They are separate from `src/` and are not called by `run_all.py`. Every script reads its inputs through `restricted_paths.py`, which needs one environment variable and stops with a clear error if it is unset:

```bash
export RESTRICTED_DATA_DIR=/path/to/restricted
export RESTRICTED_OUTPUT_DIR=/path/to/results      # optional; default outputs/restricted/
```

Expected layout under `RESTRICTED_DATA_DIR` (file names as used in the project):

```
survey/inputs/analysis_all.dta                       owner survey, one row per response
survey/inputs/respondent_waves.csv
survey/outputs/alignment_for_stata.dta               owners' agreement with the leader's positions (wave, importer indicator)
survey/trump_alignment_v2/trump_alignment_v2.dta     second version of the agreement scores
survey/trump_alignment_v2/trump_sourcing_positions_weekly.csv
jev_scores/event_question_option_probabilities.parquet   scores of the proprietary classifier
customs/smb_trade_duns_biweekly_260512.parquet       importer x origin x two-week customs records
customs/smb_firm_features_for_L2_260626.parquet
customs/china_sentiment/smb_china_sentiment_tariff_for_L2_261008.csv   two-week calendar
customs/hs6_panel/smb_tariff_h0_origin_month_261026.parquet
customs/hs6_panel/smb_duns_supplier_hs6_origin_biweekly_261024.parquet
```

Public inputs the scripts also read (China duty schedule) come from `data/public/` in this repository.

## survey/

| Script | What it does |
|---|---|
| `01_signal_expectation_plan_261101.py` | Links the classifier's coding of the leader's statements in the 28 days before each response to owners' expected tariffs and plans |
| `02_robustness_261101.py`, `03_shift_placebo_261101.py` | Robustness checks and date-shift placebos (windows shifted by 5 to 52 weeks) |
| `04_customs_signal_261101.py` | Whether imports fall from regions the leader says to leave (survey-independent customs test) |
| `06_scenario_compare_261101.py`, `07_two_by_two_261101.py` | Expectation slopes, fixed-scenario comparison, two-by-two of message and exposure |
| `08_followup_rerespondents_261101.py`, `11_followup_reports_261101.py` | Owners who answered a later wave |
| `09_capacity_by_message_261101.py`, `10_slopes_no_agreement_261101.py` | Capacity by message; slopes without the agreement control |

`11_followup_reports_261101.py` is the first script of the longitudinal extension; it is kept here because it reads the survey.

## customs/

| Folder | What it does |
|---|---|
| `pause_switch/` | Firms with arrivals from China, Hong Kong or Macau: stopped, switched origin or continued; heterogeneity; sector link |
| `extension/` | Policy episodes, seasonal reference, within-firm transitions, new-origin diagnostic, identification support, figures, and a power calculation on synthetic allocations (`05_experiment_dryrun_261101.py`, needs no data but shares the variable check) |

Outputs go to `RESTRICTED_OUTPUT_DIR/survey/`, `customs_pause_switch/` and `customs_extension/<run>/`. The extension scripts refuse to overwrite an existing run folder. Run `pause_switch/01_pause_switch_261101.py` before `pause_switch/02_sector_link_261101.py` and `extension/08_new_origin_diagnostic_261101.py`, and the other extension scripts before `extension/06_figures_261101.py`.

Status: adapted from working code; not executed in this repository because the data are absent. Aggregate results of the original runs are in `reference/restricted_aggregates/`. Do not copy outputs that hold respondent or firm rows out of `RESTRICTED_OUTPUT_DIR`.
