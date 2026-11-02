# Aggregate results from restricted data

Result tables from the scripts in `src_restricted/`: coefficients, standard errors, cell means, counts of at least 20, and the customs decomposition. Each file was checked before copying: no firm identifiers, DUNS numbers, names, response or user identifiers, and no row per respondent or per firm. A column-name scan flagged `long_e` (long-run expectation), `triple_diff` and nine-digit trade totals as false positives; they were read and kept.

| Folder | File | Script that makes it |
|---|---|---|
| survey | `01_results`, `01_sample` | `01_signal_expectation_plan` |
| survey | `02_robust` | `02_robustness` |
| survey | `03_shift_placebo` | `03_shift_placebo` |
| survey | `04_customs_results` | `04_customs_signal` |
| survey | `06_rerespondents`, `06_scenario_compare` | `06_scenario_compare` |
| survey | `07_two_by_two` | `07_two_by_two` |
| survey | `08_followup` | `08_followup_rerespondents` |
| survey | `09_capacity_by_message` | `09_capacity_by_message` |
| survey | `10_slopes_no_agreement` | `10_slopes_no_agreement` |
| customs | `01_decomposition`, `01_heterogeneity`, `01_sector_link`, `01_triple_diff` | `pause_switch/01_pause_switch` |
| customs | `02_sector_corr` | `pause_switch/02_sector_link` |
| customs | `02_customs_run_v2__*` | `extension/02_policy_episodes` |
| customs | `03_seasonal_run__*` | `extension/03_seasonal_reference` |
| customs | `04_transition_run__*` | `extension/04_within_firm_transitions` |
| customs | `08_new_origin_diagnostic_v2__*` | `extension/08_new_origin_diagnostic` |

File names keep the date suffix `_261101` of the working run.

Not copied, on purpose:

- `01_signals_by_date` and `04_region_biweek` (survey): series of the proprietary classifier's scores.
- `07_cells` (survey): a cross-tabulation with cells of 0 and 1 respondents.
- The follow-up tables of `extension/01_followup` (arms of 11 and 16 respondents; transition cells of 1) and the sector-level survey means of `pause_switch/02_sector_link` (1 to 3 respondents in some sectors). The estimate of `08_followup` rests on 32 respondents or more and is included.
- Receipts of the working runs (they hold local paths) and earlier superseded runs of the same analyses.

These tables come from the working runs of the project, not from a rerun inside this repository. Their exact inputs cannot be checked here.
