# Public filings (SEC forms 10-K and 10-Q)

Scripts for the analysis in technical appendix A4.5: the tariff paragraphs of annual and quarterly reports filed from 2024-07-01 to 2026-10-31 by the 239 current S&P 500 firms in four sectors, coded by Claude Haiku 5.5 through the Claude command-line tool. Run order: 01 sample, 02 collect filings, 03 extract passages (new versus repeated paragraphs), 04 classify, 05 analyses S0 to S3, 06 receipt (`edgar_text.py` holds shared helpers).

- SEC EDGAR asks every client to identify itself: set `SEC_USER_AGENT` (for example "Research yourname@example.org") before running 01 and 02.
- The scripts were written for the project's working layout (folders set at the top of each script); change those paths to run them here. Step 05 reads the leader signal from `data/jev/event_question_option_probabilities.parquet`.
- Aggregate result tables, the timing figure, and the quotation match table are in `reference/sec/`. Filing text is public at sec.gov and is not stored here.
- All results are exploratory associations in what firms wrote, not measures of what they believed; no person has validated the codes.
