# Synthetic owner survey (W4, U.S. owners), release v2.1

This folder holds five statistical synthetic copies of the wave-4 U.S. owner survey (`synthetic_survey_w4_us_v2_1_copy1.csv` to `copy5.csv`, 2,085 to 2,150 rows each, 40 columns) and a codebook (`codebook.csv`). The original survey stays private. The data owner allows release of a synthetic copy only.

## What it is and what it can show

Every row is generated; no row is a real respondent. The generator learned relations among the survey variables from the original, so the copies have realistic scales, skip patterns and relations, and they contain the randomized message arm as in the original design. All five copies are released and should be analyzed together (see `src/08_synthetic_survey_estimates.py`); the spread between copies shows the uncertainty added by synthesis.

- **They can** let a reader run the analysis code end to end on survey-shaped data and check that code and generator behave as described.
- **They cannot** confirm any result about real owners. Estimates computed on them match the original ones to the extent that the generator learned those relations from the original data, so a match checks the code and the generator; it is not independent evidence that an original estimate is true. Relations the generator did not learn are absent: repeated answers by the same owner, the item-by-item structure behind the agreement scores, rare combinations of background traits. The agreement scores (`loo_who_pays`, `loo_china_path`, `align`, `align_trump`) are generated variables, not recomputed from item-level judgments.
- Owners who gave almost no answers are not released, so each copy has about half the rows of the original sample, and the share of firms with a suppressed background trait is small (see below).

## How it was made

1. Sequential classification and regression trees (the synthpop approach), one variable at a time in questionnaire order: background, answers before the message, the randomized message arm, answers after the message (a separate model for each message arm), then the agreement scores. Each value is drawn from the observed values in the leaf a synthetic row falls into, so answer scales are preserved. Missingness is modeled before the value to keep skip patterns. Minimum leaf size 10.
2. Five copies, each fitted on a bootstrap sample of the original, so that differences between copies reflect parameter uncertainty. The message arm and the exposure flag are drawn by design at the original allocation shares.
3. The generator learns from the original after local suppression of rare background combinations (below), and generates from all original rows. An earlier version (v2) excluded rows in rare combinations instead; that removed larger firms with many answers, shrank the estimation samples and attenuated the estimates, so it was replaced after comparing it with the original (an exploratory choice made after seeing the v2 comparison, not a pre-set rule).

## Disclosure steps

| Step | What was done | Result |
|---|---|---|
| String answers | `vote_us_early` and `tariff_opin_us_1` encoded as categories and synthesized. `trade_opin_us_1` is empty in every original row of this sample and is not released. | Both variables present |
| Identifiers and free text | No response id, user id, free text, exact date or location. Dates are kept as the Monday of the response week. A random `cluster_id` replaces the user id and only clusters standard errors. No state or region is in the file. | None released |
| Coarsening | Industry: 11 codes kept (at least 100 original rows each), the rest merged into 9999. Employees: 4 bins. Sales: 3 bins, no exact values. Political group: 5 groups. | |
| Local suppression | Original rows whose combination of industry, employee bin, sales bin, importer flag and political group occurs in fewer than 5 original rows are recoded in a fixed order until it reaches 5: industry to other, then employee bin to blank, sales bin to blank, political group to Suppressed. | Rows recoded: industry 596, employee bin 61, sales bin 23, political group 6 (of 3,950). One original row stays in a combination of fewer than 5 rows after all steps and that combination is never generated |
| Small groups | Combinations of the five background traits in the released copies | Every released combination occurs at least 5 times in the (suppressed) original data (minimum 5); combinations unique within a released copy: 0.4 to 0.6% of rows |
| Near-empty rows | Rows with fewer than 3 answers beyond background removed. | 1,800 to 1,865 of 3,950 generated rows per copy removed |
| Exact matches | Rows equal to an original row on all synthesized variables and with 10 or more answers removed. | Removed: 1 row over the five copies; after removal 0 in every copy. Rows with fewer than 10 answers that equal an original row were kept by the rule: 18 to 27 rows per copy (0.8 to 1.3%; 21.4% of original rows equal another original row) |
| Distance to the original | Nearest original row (Gower distance), compared with original rows among themselves. | Mean 0.067 against 0.039 between original rows; for rows with 10 or more answers 0.102 against 0.093; no row with 10 or more answers at distance 0 |

These diagnostics are partial and are not a proof of safety. Membership and attribute inference attacks were not tested.

## Comparison with the original estimates (restricted data; aggregates only)

Original estimates use the survey's full background controls; the synthetic estimates use the coarsened controls in these files. The combined estimate and standard error follow Raghunathan et al. (2003) with the fallback ubar + b/m when the formula is negative (as in `src/08`); overlap is the interval overlap of Karr et al. (2006) between the original 95% interval and the combined interval.

| Estimate | Original (SE) | Combined five copies (SE) | Overlap |
|---|---|---|---|
| Message effect on the revision of the expected tariff on imported materials (percentage points) | 5.658 (1.270) | 7.193 (1.892) | 0.78 |
| Message effect on the revision of the believed current tariff on imported materials | 0.050 (1.749) | 3.670 (1.333) | 0.41 |
| Message effect on the revision of the believed current tariff on competing products | 4.786 (1.420) | 2.958 (2.752) | 0.76 |
| Message effect on the revision of the expected tariff on competing products | 6.153 (1.485) | 4.382 (3.192) | 0.73 |
| Slope of planned price change on the expectation held before the message | 0.145 (0.023) | 0.146 (0.034) | 0.84 |
| Slope of planned investment change on the expectation held before the message | -0.479 (0.069) | -0.416 (0.064) | 0.76 |
| Slope of the price change in the 25 percent scenario | 0.098 (0.025) | 0.121 (0.005) | 0.59 |
| Slope of the investment change in the 25 percent scenario | -0.334 (0.093) | -0.302 (0.088) | 0.91 |

Over all 30 targets of the evaluation (message effects, same-owner slopes with and without the agreement score, sourcing coefficients, shares by agreement group), the combined synthetic estimate lies inside the original 95% interval for **29 of 30** targets, the mean interval overlap is **0.76** (lowest 0.41), and for the 22 targets that differ significantly from zero in the original, the sign agrees in all five copies for 21 and in at least four copies for 22. The outlier is the revision of the believed current tariff on imported materials, which is about 0 in the original and 3.7 in the combined copies. The message effect on the expected tariff is overshot (7.2 against 5.7), and the largest differences between copies are in the message effects (between-copy SD 2 to 3). Estimation samples are close to the original (for example 711 against 717 for the message effect, 443 against 455 for the planned-investment slope). Utility: the mean absolute difference in pairwise correlations is 0.063; a classifier can still tell original from synthetic rows (pMSE about 2.4 times its value for identical distributions).

## Reproduce

```bash
python run_all.py outputs/run_<name>     # step 08 reads the five copies and writes <run>/08_synthetic_survey_estimates.csv
```

Step 08 uses the five synthetic files and `data/public/china_additional_duty_steps.csv` only, and reports the combined estimate, the combined standard error, and the per-copy values. `src/06_synthetic_survey.py` is a different object: respondents with arbitrary parameters that exercise the comparison code.
