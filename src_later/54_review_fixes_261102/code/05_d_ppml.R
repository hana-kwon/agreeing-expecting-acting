# ============================================================
# START DATE   : 2026-11-02
# CODE RUN DATE: 2026-11-02
# TIME TAKEN   : ~5 min
# DESCRIPTION  : D. PPML on import levels, country x HS2 x month (2024-01..2025-12), x = additional tariff rate in 10pp units
#                (D0 headline country rate, D1 product-level rate aggregated to HS2 with 2024 import weights, D2 = D1 without Section 232),
#                FE country^HS2 + HS2^month, SE clustered by country; Kline-Santos wild score bootstrap by country (Rademacher, 9999 draws);
#                top-90%-of-value cells; log-YoY OLS comparison (unweighted and base-value weighted).
# INPUT FILES  : PRIVATE_CACHE
# OUTPUT FILES : analysis/54_review_fixes_261102/05_d_ppml_261102.csv, 05_d_yoy_ols_261102.csv
# ============================================================
suppressMessages({library(data.table); library(fixest)})
this_file <- gsub("~+~", " ", sub("--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE)[1]), fixed = TRUE)
TOPIC <- normalizePath(file.path(dirname(this_file), "..", ".."))
OUT <- file.path(TOPIC, "analysis/54_review_fixes_261102")
f1 <- file.path(OUT, "05_d_ppml_261102.csv"); f2 <- file.path(OUT, "05_d_yoy_ols_261102.csv")
if (file.exists(f1) || file.exists(f2)) stop("refuse to overwrite")
p <- fread(file.path(Sys.getenv("HOME"), "private_cache/review54_261102/d_panel.csv"))
p[, cell := paste(iso, hs2)]
p[, `:=`(h = tau_head / 10, pr = tau_prod / 10, pn = tau_prod_no232 / 10, hs2mi = paste(hs2, mi))]
set.seed(261102)
boot <- function(d, xv, B = 9999) {
  d <- copy(d); d[, cell := factor(cell)]
  m0 <- fepois(imp ~ 1 | cell + hs2mi, data = d, notes = FALSE, warn = FALSE)
  ob <- obs(m0); d <- d[ob]; mu0 <- fitted(m0)
  xt <- demean(as.matrix(d[[xv]]), d[, .(cell, hs2mi)], weights = mu0)[, 1]
  s <- tapply(xt * (d$imp - mu0), d$iso, sum)
  den <- sqrt(sum(s^2)); tobs <- sum(s) / den
  tb <- replicate(B, sum(sample(c(-1, 1), length(s), TRUE) * s) / den)
  c(t_obs_score = tobs, p_boot = (1 + sum(abs(tb) >= abs(tobs))) / (B + 1), G = length(s))
}
rows <- list()
base24 <- p[mi >= 12 & mi <= 23, .(b = sum(imp)), by = cell]; setorder(base24, -b); base24[, cum := cumsum(b) / sum(b)]
top90 <- base24[cum <= 0.9 | shift(cum, fill = 0) < 0.9, cell]
runs <- list(list("all_weighted_cells", p), list("top90_value_cells", p[cell %in% top90]))
for (rr in runs) {
  d0 <- rr[[2]][mi >= 12 & !is.na(pr) & !is.na(h)]
  for (xv in c("h", "pr", "pn")) {
    m <- fepois(as.formula(paste("imp ~", xv, "| cell + hs2mi")), data = d0, cluster = ~iso, notes = FALSE, warn = FALSE)
    ct <- coeftable(m); b <- boot(d0, xv, 9999)
    rows[[length(rows) + 1]] <- data.table(sample = rr[[1]], x = c(h = "D0_headline", pr = "D1_product_all_components", pn = "D2_product_no232")[xv], coef = ct[1, 1], se = ct[1, 2],
      pct_per_10pp = 100 * (exp(ct[1, 1]) - 1), lo = ct[1, 1] - 1.96 * ct[1, 2], hi = ct[1, 1] + 1.96 * ct[1, 2], t_obs_score = b["t_obs_score"], p_boot_score = b["p_boot"],
      n_clusters = b["G"], nobs = nobs(m), mean_x_10pp = mean(d0[[xv]]), sd_x_10pp = sd(d0[[xv]]))
    cat(rr[[1]], xv, ct[1, 1], ct[1, 2], b["p_boot"], "\n")
  }
}
fwrite(rbindlist(rows), f1)
# YoY OLS comparison
y <- p[, .(cell, iso, hs2, mi, imp, h, pr, hs2mi)]
yy <- merge(y[mi >= 24], y[, .(cell, mi = mi + 12, imp0 = imp, h0 = h, pr0 = pr)], by = c("cell", "mi"))
yy <- yy[imp > 0 & imp0 >= 1e5 & !is.na(pr) & !is.na(pr0) & !is.na(h)]
yy[, `:=`(dy = log(imp) - log(imp0), dh = log1p(h * 10 / 100) - log1p(h0 * 10 / 100), dp = log1p(pr * 10 / 100) - log1p(pr0 * 10 / 100))]
o <- list()
for (xv in c("dh", "dp")) for (wt in c("unweighted", "base_value")) {
  m <- feols(as.formula(paste("dy ~", xv, "| cell + hs2mi")), data = yy, weights = if (wt == "unweighted") NULL else yy$imp0, cluster = ~iso, notes = FALSE)
  ct <- coeftable(m); o[[length(o) + 1]] <- data.table(x = xv, weights = wt, coef = ct[1, 1], se = ct[1, 2], nobs = nobs(m), n_clusters = uniqueN(yy$iso))
}
fwrite(rbindlist(o), f2); print(rbindlist(rows)[, .(sample, x, coef, se, pct_per_10pp, p_boot_score, nobs)]); print(rbindlist(o))
