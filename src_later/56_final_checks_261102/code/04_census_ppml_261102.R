# ============================================================
# START DATE   : 2026-11-02
# CODE RUN DATE: 2026-11-02
# TIME TAKEN   : ~5 min
# DESCRIPTION  : imp = exp(b tariff/10pp + country^HS2 + HS2^month), country-clustered, score bootstrap as in 54 D: rows as recorded, then absent rows filled with zero.
# INPUT FILES  : PRIVATE_CACHE
# OUTPUT FILES : analysis/56_final_checks_261102/04_census_ppml_261102.csv
# ============================================================
suppressMessages({library(data.table); library(fixest)})
this_file <- gsub("~+~", " ", sub("--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE)[1]), fixed = TRUE)
TOPIC <- normalizePath(file.path(dirname(this_file), "..", "..")); f1 <- file.path(TOPIC, "analysis/56_final_checks_261102/04_census_ppml_261102.csv")
if (file.exists(f1)) stop("refuse to overwrite")
p <- fread(file.path(Sys.getenv("HOME"), "private_cache/final56_261102/d_panel.csv"), colClasses = list(character = "hs2"))
p[, cell := paste(iso, hs2)]; p[, `:=`(h = tau_head / 10, pr = tau_prod / 10, pn = tau_prod_no232 / 10, hs2mi = paste(hs2, mi))]
set.seed(261102)
boot <- function(d, xv, B = 9999) {
  d <- copy(d); d[, cell := factor(cell)]; m0 <- fepois(imp ~ 1 | cell + hs2mi, data = d, notes = FALSE, warn = FALSE)
  ob <- obs(m0); d <- d[ob]; mu0 <- fitted(m0); xt <- demean(as.matrix(d[[xv]]), d[, .(cell, hs2mi)], weights = mu0)[, 1]
  s <- tapply(xt * (d$imp - mu0), d$iso, sum); den <- sqrt(sum(s^2)); tobs <- sum(s) / den
  tb <- replicate(B, sum(sample(c(-1, 1), length(s), TRUE) * s) / den); (1 + sum(abs(tb) >= abs(tobs))) / (B + 1)
}
rows <- list()
for (samp in c("recorded_rows", "absent_filled_zero")) {
  d0 <- p[mi >= 12 & !is.na(pr) & !is.na(h)]; if (samp == "recorded_rows") d0 <- d0[recorded == 1]
  for (xv in c("h", "pr", "pn")) {
    m <- fepois(as.formula(paste("imp ~", xv, "| cell + hs2mi")), data = d0, cluster = ~iso, notes = FALSE, warn = FALSE); ct <- coeftable(m)
    rows[[length(rows) + 1]] <- data.table(sample = samp, x = xv, coef = ct[1, 1], se = ct[1, 2], pct_per_10pp = 100 * (exp(ct[1, 1]) - 1), p_boot = boot(d0, xv), nobs = nobs(m))
  }
}
r <- rbindlist(rows); fwrite(r, f1); print(r)
