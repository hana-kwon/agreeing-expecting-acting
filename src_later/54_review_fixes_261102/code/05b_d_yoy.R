# ============================================================
# START DATE   : 2026-11-02
# CODE RUN DATE: 2026-11-02
# TIME TAKEN   : ~2 min
# DESCRIPTION  : D comparison: log-YoY OLS (36/07 form) with headline vs product-level rate, unweighted and base-value weighted. Split from 05_d_ppml.R after the weights argument failed there.
# INPUT FILES  : PRIVATE_CACHE
# OUTPUT FILES : analysis/54_review_fixes_261102/05_d_yoy_ols_261102.csv
# ============================================================
suppressMessages({library(data.table); library(fixest)})
this_file <- gsub("~+~", " ", sub("--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE)[1]), fixed = TRUE)
TOPIC <- normalizePath(file.path(dirname(this_file), "..", ".."))
OUT <- file.path(TOPIC, "analysis/54_review_fixes_261102")
f2 <- file.path(OUT, "05_d_yoy_ols_261102.csv"); if (file.exists(f2)) stop("refuse to overwrite")
p <- fread(file.path(Sys.getenv("HOME"), "private_cache/review54_261102/d_panel.csv"))
p[, cell := paste(iso, hs2)]
p[, `:=`(h = tau_head / 10, pr = tau_prod / 10, hs2mi = paste(hs2, mi))]
# YoY OLS comparison
y <- p[, .(cell, iso, hs2, mi, imp, h, pr, hs2mi)]
yy <- merge(y[mi >= 24], y[, .(cell, mi = mi + 12, imp0 = imp, h0 = h, pr0 = pr)], by = c("cell", "mi"))
yy <- yy[imp > 0 & imp0 >= 1e5 & !is.na(pr) & !is.na(pr0) & !is.na(h)]
yy[, `:=`(dy = log(imp) - log(imp0), dh = log1p(h * 10 / 100) - log1p(h0 * 10 / 100), dp = log1p(pr * 10 / 100) - log1p(pr0 * 10 / 100))]
o <- list()
for (xv in c("dh", "dp")) for (wt in c("unweighted", "base_value")) {
  fm <- as.formula(paste("dy ~", xv, "| cell + hs2mi"))
  m <- if (wt == "unweighted") feols(fm, data = yy, cluster = ~iso, notes = FALSE) else feols(fm, data = yy, weights = yy$imp0, cluster = ~iso, notes = FALSE)
  ct <- coeftable(m); o[[length(o) + 1]] <- data.table(x = xv, weights = wt, coef = ct[1, 1], se = ct[1, 2], nobs = nobs(m), n_clusters = uniqueN(yy$iso))
}
fwrite(rbindlist(o), f2); print(rbindlist(o))
