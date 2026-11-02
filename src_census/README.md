# Public import statistics (U.S. Census)

Scripts for the Census analyses in the technical appendix (A5.4): monthly U.S. general imports by country and HS chapter, and world totals by HS heading, 2023-01 to 2026-08, from the Census International Trade API (`api.census.gov/data/timeseries/intltrade/imports/hs`). Run order: 01 download, 02 country events and threats, 03 product events, 04 the 2025 China episode, 05 figures, 06 receipt.

- The download needs a free Census API key in `~/.config/census/api_key.txt`; the key is read from that file and never written out.
- The scripts were written for the project's working layout; `common_261102.py` sets the folders at its top. Change those paths to run them here. They use the corrected calendar in `data/public/tariff_calendar/` and the country-week statement panel built from the Jev scores.
- Result tables and figures of the reference run are in `reference/census/`.
- All results are exploratory and aggregate; they do not describe individual firms.
