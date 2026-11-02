# ============================================================
# START DATE   : 2026-11-02
# CODE RUN DATE: 2026-11-02
# TIME TAKEN   : ~15 min
# DESCRIPTION  : China-share x post on new trademark applications; PPML (fepois) and any-filing LPM (feols);
#                firm FE + SIC2 x quarter FE; cluster by firm. Dynamic version, employment bins, NICE groups,
#                filing basis (ITU 1(b) vs use 1(a)), SOU/registration (right-censored), region shares.
#                Every specification is logged to 04_spec_log.csv (one row per estimate).
# INPUT FILES  : PRIVATE_CACHE<TAG>.parquet
# OUTPUT FILES : P/analysis/47_innovation_trademarks_261102/04_estimates<TAG>.csv, 04_spec_log<TAG>.txt (aggregates only)
# ============================================================
suppressMessages({library(fixest); library(arrow); library(data.table)})
args <- commandArgs(TRUE); TAG <- args[1]; LASTQ <- as.integer(args[2])
C <- path.expand("PRIVATE_CACHE")
O <- "HOME"
outf <- paste0(O, "04_estimates", TAG, ".csv"); logf <- paste0(O, "04_spec_log", TAG, ".txt")
stopifnot(!file.exists(outf), !file.exists(logf))   # refuse to overwrite
setFixest_nthreads(4)
d <- as.data.table(read_parquet(paste0(C, "panel_firm_quarter", TAG, ".parquet")))
d <- d[q <= LASTQ & !is.na(sic2) & !is.na(china_share24)]
d[, post := as.integer(q >= 20252)]
d[, any_n := as.integer(n > 0)]
d[, sic2q := paste(sic2, q)]
d[, small := !is.na(emp_bin) & emp_bin != "250+"]
res <- list(); sink(logf, split = FALSE)
cat("Panel:", nrow(d), "rows;", uniqueN(d$final_dunsno), "firms; quarters", min(d$q), "-", max(d$q), "\n")
add <- function(spec, sample, outcome, model, est, term_keep = NULL) {
  ct <- as.data.table(coeftable(model), keep.rownames = "term")
  setnames(ct, c("term", "coef", "se", "stat", "p"))
  if (!is.null(term_keep)) ct <- ct[grepl(term_keep, term)]
  ct[, `:=`(spec = spec, sample = sample, outcome = outcome, nobs = model$nobs)]
  res[[length(res) + 1]] <<- ct
  cat(sprintf("[%s | %s | %s] n=%d  %s\n", spec, sample, outcome, model$nobs, paste(sprintf("%s=%.4f(%.4f)", ct$term, ct$coef, ct$se), collapse = "; ")))
}
samples <- list(small_lt250 = d[small == TRUE], all_sizes = d, emp_unknown = d[is.na(emp_bin)])
# ★ CORE LOGIC: Main spec. Exposure = 2024 China import share (time-invariant, main effect absorbed by firm FE);
# post = 2025Q2+. SIC2 x quarter FE absorb industry-wide shocks (e.g., tariff-exempt sectors, trademark filing waves).
for (s in names(samples)) {
  x <- samples[[s]]
  for (y in c("n", "n_itu", "n_use")) {
    yy <- c(n = "n", n_itu = "itu", n_use = "use")[y]
    m <- tryCatch(fepois(as.formula(paste0(yy, " ~ china_share24:post | final_dunsno + sic2q")), data = x, cluster = ~final_dunsno), error = function(e) NULL)
    if (!is.null(m)) add(paste0("PPML_", y), s, yy, m, "china_share24:post")
  }
  m <- feols(any_n ~ china_share24:post | final_dunsno + sic2q, data = x, cluster = ~final_dunsno); add("LPM_any", s, "any_n", m)
}
x <- samples$small_lt250
# Dynamic (reference quarter 2025Q1 = last quarter before the 2025-04-02 announcement)
for (yy in c("n", "any_n")) {
  f <- as.formula(paste0(yy, " ~ i(q, china_share24, ref = 20251) | final_dunsno + sic2q"))
  m <- if (yy == "n") fepois(f, data = x, cluster = ~final_dunsno) else feols(f, data = x, cluster = ~final_dunsno)
  add(paste0("DYN_", ifelse(yy == "n", "PPML", "LPM")), "small_lt250", yy, m)
}
# By employment bin
for (b in c("1-19", "20-49", "50-249", "250+")) {
  xb <- d[emp_bin == b]
  m <- tryCatch(fepois(n ~ china_share24:post | final_dunsno + sic2q, data = xb, cluster = ~final_dunsno), error = function(e) NULL); if (!is.null(m)) add("PPML_bin", b, "n", m)
  m <- feols(any_n ~ china_share24:post | final_dunsno + sic2q, data = xb, cluster = ~final_dunsno); add("LPM_bin", b, "any_n", m)
}
# NICE groups
for (g in c("g_any_goods", "g_apparel_textile", "g_electronics_machinery", "g_household_consumer", "g_chem_health_food", "g_other_goods", "g_services")) {
  m <- tryCatch(fepois(as.formula(paste0(g, " ~ china_share24:post | final_dunsno + sic2q")), data = x, cluster = ~final_dunsno), error = function(e) NULL); if (!is.null(m)) add("PPML_nice", "small_lt250", g, m)
  x[, tmp := as.integer(get(g) > 0)]
  m <- feols(tmp ~ china_share24:post | final_dunsno + sic2q, data = x, cluster = ~final_dunsno); add("LPM_nice", "small_lt250", g, m)
}
# Commercialization steps (right-censored snapshot): statement of use, registration
for (g in c("sou", "reg")) {
  m <- tryCatch(fepois(as.formula(paste0(g, " ~ china_share24:post | final_dunsno + sic2q")), data = x, cluster = ~final_dunsno), error = function(e) NULL); if (!is.null(m)) add("PPML_commercial", "small_lt250", g, m)
}
# Region shares jointly (omitted: ROW)
m <- fepois(n ~ china_share24:post + asia_ex_china_share24:post + europe_share24:post + north_america_share24:post + latam_carib_share24:post | final_dunsno + sic2q, data = x, cluster = ~final_dunsno)
add("PPML_regions", "small_lt250", "n", m)
# Ever-filer subsample (selected on outcome; reference only)
ever <- x[, .(e = sum(n)), by = final_dunsno][e > 0, final_dunsno]
xe <- x[final_dunsno %in% ever]
m <- feols(any_n ~ china_share24:post | final_dunsno + sic2q, data = xe, cluster = ~final_dunsno); add("LPM_everfiler", "small_ever", "any_n", m)
m <- fepois(n ~ china_share24:post | final_dunsno + sic2q, data = xe, cluster = ~final_dunsno); add("PPML_everfiler", "small_ever", "n", m)
# Binary high-China exposure (>=50% share), for interpretability
x[, hi := as.integer(china_share24 >= 0.5)]
m <- fepois(n ~ hi:post | final_dunsno + sic2q, data = x, cluster = ~final_dunsno); add("PPML_hi50", "small_lt250", "n", m)
# Descriptives (aggregates)
cat("\nMean apps/firm-quarter small pre(<=2025Q1):", x[post == 0, mean(n)], " post:", x[post == 1, mean(n)], "\n")
cat("Share firm-quarters any filing small pre:", x[post == 0, mean(any_n)], " post:", x[post == 1, mean(any_n)], "\n")
cat("ITU share of apps pre:", x[post == 0, sum(itu) / sum(n)], " post:", x[post == 1, sum(itu) / sum(n)], "\n")
cat("Mean china_share24 (small):", x[!duplicated(final_dunsno), mean(china_share24)], " share>=0.5:", x[!duplicated(final_dunsno), mean(china_share24 >= .5)], "\n")
sink()
fwrite(rbindlist(res, fill = TRUE), outf)
