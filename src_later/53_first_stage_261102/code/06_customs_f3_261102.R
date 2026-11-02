# ============================================================
# START DATE   : 2026-11-02
# CODE RUN DATE: 2026-11-02
# TIME TAKEN   : ~15 min compute
# DESCRIPTION  : Customs test of analysis/50 with the chosen first-stage measure f3 (count of higher-direction future claims naming a country
#                before any increase was in force there). Same PPML design (firm^origin + firm^period, two-way cluster origin and firm),
#                joint six lags (2,4,6,8,12,16 weeks) and cumulative effect, rate-in-force controls Rind_k0 + Rind_k8. Only bins whose
#                every lag window is observed (no zero filling of unobserved periods). Job MAIN: 10% firms. Job PLACEBO: 3% firms,
#                country permutation within bin (40) and unique circular shifts of the cumulative signal over observed bins (24).
# INPUT FILES  : PRIVATE_CACHE{cells,firms}.parquet, PRIVATE_CACHE, analysis/50 01_origin_map
# OUTPUT FILES : analysis/53_first_stage_261102/06_customs_f3_<JOB>_261102.csv
# ============================================================
suppressMessages({library(arrow); library(data.table); library(fixest)})
setFixest_nthreads(2)
JOB <- commandArgs(trailingOnly = TRUE)[1]
this_file <- gsub("~+~", " ", sub("--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE)[1]), fixed = TRUE)
TOPIC <- normalizePath(file.path(dirname(this_file), "..", "..")); OUT <- file.path(TOPIC, "analysis/53_first_stage_261102")
C50 <- file.path(Sys.getenv("HOME"), "private_cache/customs50_261102"); C53 <- file.path(Sys.getenv("HOME"), "private_cache/customs53_261102")
of <- file.path(OUT, paste0("06_customs_f3_", JOB, "_261102.csv")); if (file.exists(of)) stop("output exists; refusing to overwrite")
LAGS <- c(2, 4, 6, 8, 12, 16)
build <- function(frac, seed = 261102) {
  cells <- as.data.table(read_parquet(file.path(C50, "cells.parquet"))); sig <- as.data.table(read_parquet(file.path(C53, "sig_f3.parquet")))
  om <- fread(file.path(TOPIC, "analysis/50_customs_country_signal_261102/01_origin_map_261102.csv"))[!is.na(iso3) & origin != "Other"]
  cells <- merge(cells, om[, .(origin, iso3)], by = "origin")[trade_value > 0]
  np <- cells[, .(n = uniqueN(origin)), by = final_dunsno][n >= 2]; cells <- cells[final_dunsno %in% np$final_dunsno]
  set.seed(seed); keep <- sample(unique(cells$final_dunsno), round(frac * uniqueN(cells$final_dunsno))); cells <- cells[final_dunsno %in% keep]
  pairs <- unique(cells[, .(final_dunsno, origin, iso3)]); fb <- unique(cells[, .(final_dunsno, bin_index)])
  g <- merge(fb, pairs, by = "final_dunsno", allow.cartesian = TRUE)
  g <- merge(g, cells[, .(final_dunsno, origin, bin_index, y = trade_value)], by = c("final_dunsno", "origin", "bin_index"), all.x = TRUE); g[is.na(y), y := 0]
  g <- merge(g, sig[lag16_observed == 1], by = c("iso3", "bin_index"))                 # ★ IMPORTANT: drop bins with unobserved lag windows
  u <- unique(g[, c("iso3", "bin_index", paste0("f3_k", LAGS)), with = FALSE])
  for (k in LAGS) set(g, j = paste0("S_k", k), value = g[[paste0("f3_k", k)]] / sd(u[[paste0("f3_k", k)]]))
  cum <- Reduce(`+`, lapply(LAGS, function(k) u[[paste0("f3_k", k)]])); set(g, j = "S_cum", value = Reduce(`+`, lapply(LAGS, function(k) g[[paste0("f3_k", k)]])) / sd(cum))
  g[, period := bin_index]; g
}
FE <- "final_dunsno^origin + final_dunsno^period"; R2 <- "Rind_k0 + Rind_k8"; res <- list()
fit <- function(d, rhs, spec, sample) {
  m <- tryCatch(fepois(as.formula(paste("y ~", rhs, "|", FE)), d, cluster = ~origin + final_dunsno, warn = FALSE, notes = FALSE), error = function(e) { message("FAIL ", spec); NULL })
  if (is.null(m)) return(NULL)
  ct <- coeftable(m); keep <- grep("^S_", rownames(ct)); r <- data.table(spec = spec, sample = sample, term = rownames(ct)[keep], est = ct[keep, 1], se = ct[keep, 2], p = ct[keep, 4], nobs = nobs(m))
  r[, mde := 2.8 * se]; res[[length(res) + 1]] <<- r; message(spec, " ", sample); m
}
if (JOB == "MAIN") {
  g0 <- build(0.10)
  for (ex in c(FALSE, TRUE)) {
    g <- if (ex) g0[origin != "China"] else g0; sm <- ifelse(ex, "exChina_10pct", "all_10pct")
    fit(g, paste(c(paste0("S_k", LAGS), R2), collapse = " + "), "joint_six_lags", sm)
    fit(g, paste("S_cum +", R2), "cumulative", sm)
    for (k in LAGS) fit(g, sprintf("S_k%d + Rind_k%d", k, k), paste0("single_lag", k), sm)
  }
  fwrite(rbindlist(res), of)
} else {
  g0 <- build(0.03); set.seed(261102); out <- list()
  m0 <- fit(g0, paste("S_cum +", R2), "actual", "all_3pct"); b0 <- coef(m0)["S_cum"]
  sr <- unique(g0[, .(iso3, bin_index, S = S_cum)])
  run <- function(sv, kind, id) {
    d <- merge(g0[, !"S_cum"], sv, by = c("iso3", "bin_index")); setnames(d, "S", "S_cum")
    m <- tryCatch(fepois(as.formula(paste("y ~ S_cum +", R2, "|", FE)), d, cluster = ~final_dunsno, warn = FALSE, notes = FALSE), error = function(e) NULL)
    if (!is.null(m)) out[[length(out) + 1]] <<- data.table(kind = kind, id = id, est = coef(m)["S_cum"], actual = b0)
  }
  for (i in 1:40) { p <- copy(sr); p[, S := S[sample.int(.N)], by = bin_index]; run(p[, .(iso3, bin_index, S)], "perm_country", i) }
  nb <- uniqueN(sr$bin_index)
  for (j in unique(c(-12:-3, 3:12))) {      # unique cyclic shifts (in bins) over the OBSERVED bins only; 20 shifts
    p <- copy(sr)[order(iso3, bin_index)]; p[, S := shift(S, j, type = "cyclic"), by = iso3]; run(p[, .(iso3, bin_index, S)], "date_shift", j)
  }
  fwrite(rbindlist(out), of)
}
