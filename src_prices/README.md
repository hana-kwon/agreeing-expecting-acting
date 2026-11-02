# Daily retail prices and the leader's statements

Scripts for technical appendix A5.5. Step 01 downloads two published daily price indices (by country of origin, and imported versus domestic) from the Pricing Lab tariff tracker (pricinglab.org/tariff-tracker; Cavallo, Llamas, and Vazquez 2026). The files carry a copyright notice and no license, so they are not stored here; the script fetches them. Steps 02 to 04 build the daily China-path signal from `data/jev/`, read the corrected calendar, estimate the price changes after escalation days with block-bootstrap standard errors and date-shift placebos, and draw the figure; 05 writes a receipt.

- `ROOT` in `common_261102.py` points to the project's working layout for data and output folders; change it to run here.
- Result tables and the figure of the reference run are in `reference/prices/`. All results are exploratory and rest on few escalation days.
