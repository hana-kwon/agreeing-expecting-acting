# ============================================================
# START DATE   : 2026-11-02
# CODE RUN DATE: 2026-11-02
# TIME TAKEN   : ~10 min
# DESCRIPTION  : G. Analysis 50 re-estimated on the sample where every signal lag (2-16 weeks) lies inside the statement panel (bins from
#                2025-05, lag16_in_panel == 1; no zero-filling before 2025-01-24) and the rate month is coded (<= 2025-12), the same
#                bins/firms for all lags, with the numeric product-level origin rate path (Rprod, 10pp units, lags 0 and 8 weeks) in place of
#                the in-force indicator. Compared with the old indicator on the identical sample. Date-shift placebo with DISTINCT circular
#                shifts only (shift modulo number of bins; the count of distinct shifts is reported). Usage: Rscript 08_g_country_signal_261102.R
# INPUT FILES  : code/50 common (build_grid), PRIVATE_CACHE*.parquet, PRIVATE_CACHE
# OUTPUT FILES : analysis/54_review_fixes_261102/08_g_coefs_261102.csv, 08_g_placebo_261102.csv
# ============================================================
this_file54 <- gsub("~+~", " ", sub("--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE)[1]), fixed = TRUE)
TOPIC_DIR54 <- normalizePath(file.path(dirname(this_file54), "..", ".."))
Sys.setenv(TOPIC_DIR = TOPIC_DIR54)
commandArgs_orig <- commandArgs
source(file.path(TOPIC_DIR54, "code/50_customs_country_signal_261102/00_common_261102.R"))
OUT54 <- file.path(TOPIC_DIR54, "analysis/54_review_fixes_261102"); lg <- function(...) cat(format(Sys.time(), "%H:%M:%S"), ..., "\n")
f1 <- file.path(OUT54, "08_g_coefs_261102.csv"); f2 <- file.path(OUT54, "08_g_placebo_261102.csv"); if (file.exists(f1) || file.exists(f2)) stop("refuse to overwrite")
rp <- as.data.table(read_parquet(file.path(Sys.getenv("HOME"), "private_cache/review54_261102/rate_prod.parquet")))
prep <- function(frac) { g <- build_grid(frac); g <- merge(g, rp, by = c("iso3", "bin_index")); g[lag16_in_panel == 1 & !is.na(Rprod_k0) & !is.na(Rprod_k8)] }
FE <- "final_dunsno^origin + final_dunsno^period"
res <- list()
fit <- function(d, rhs, spec, sample) {
  m <- tryCatch(fepois(as.formula(paste("y ~", rhs, "|", FE)), d, cluster = ~origin + final_dunsno, warn = FALSE, notes = FALSE), error = function(e) { message("FAIL ", spec, e$message); NULL })
  if (is.null(m)) return(NULL); r <- tidy_fit(m, spec, sample); r[, `:=`(n_bins = uniqueN(d$bin_index), n_origin_iso = uniqueN(d$iso3), n_firms_grid = uniqueN(d$final_dunsno))]
  res[[length(res) + 1]] <<- r; lg(spec, sample); m
}
g <- prep(0.10); lg("grid rows", nrow(g), "bins", uniqueN(g$bin_index), "firms", uniqueN(g$final_dunsno))
for (ex in c(FALSE, TRUE)) {
  d <- if (ex) g[origin != "China"] else g; sm <- ifelse(ex, "exChina_10pct_commonlags", "all_10pct_commonlags")
  fit(d, "S_threats4_cum + Rprod_k0 + Rprod_k8", "G1_cum_numeric_rate", sm)
  fit(d, "S_threats4_cum + Rind_k0 + Rind_k8", "G1b_cum_indicator_same_sample", sm)
  fit(d, "S_threats4_cum", "G1c_cum_no_rate", sm)
  for (k in LAGS) fit(d, sprintf("S_threats4_k%d + Rprod_k0 + Rprod_k8", k), paste0("G2_lag", k, "_numeric_rate"), sm)
  fit(d, "S_thr_f_cum + S_thr_u_cum + Rprod_k0 + Rprod_k8", "G3_followed_vs_unfollowed_numeric_rate", sm)
}
fwrite(rbindlist(res), f1)
# placebo: distinct circular shifts of the signal across bins within jurisdiction
g3 <- prep(0.03); lg("placebo grid rows", nrow(g3)); out <- list()
for (ex in c(FALSE, TRUE)) {
  d <- if (ex) g3[origin != "China"] else g3; sm <- ifelse(ex, "exChina_3pct", "all_3pct")
  m0 <- fepois(y ~ S_threats4_cum + Rprod_k0 + Rprod_k8 | final_dunsno^origin + final_dunsno^period, d, cluster = ~final_dunsno, warn = FALSE, notes = FALSE); b0 <- coef(m0)["S_threats4_cum"]
  sr <- unique(d[, .(iso3, bin_index, S = S_threats4_cum)])[order(iso3, bin_index)]; nb <- uniqueN(sr$bin_index)
  shifts <- setdiff(1:(nb - 1), c(0)); shifts <- shifts[!(shifts %in% c(1))]   # shift of 1 bin kept out: adjacent-bin overlap of the 28-day windows
  shifts <- sort(unique(shifts %% nb)); shifts <- shifts[shifts >= 2 & shifts <= nb - 2]
  for (j in shifts) {
    p <- copy(sr); p[, S := shift(S, j, type = "cyclic"), by = iso3]
    dd <- merge(d[, !"S_threats4_cum"], p[, .(iso3, bin_index, S)], by = c("iso3", "bin_index")); setnames(dd, "S", "S_threats4_cum")
    m <- tryCatch(fepois(y ~ S_threats4_cum + Rprod_k0 + Rprod_k8 | final_dunsno^origin + final_dunsno^period, dd, cluster = ~final_dunsno, warn = FALSE, notes = FALSE), error = function(e) NULL)
    if (!is.null(m)) out[[length(out) + 1]] <- data.table(sample = sm, shift_bins = j, est = coef(m)["S_threats4_cum"], actual = b0, n_bins = nb, n_distinct_shifts = length(shifts))
  }
  lg("placebo done", sm, "distinct shifts", length(shifts))
}
fwrite(rbindlist(out), f2)
