# ============================================================
# START DATE   : 2026-11-02
# CODE RUN DATE: 2026-11-02
# TIME TAKEN   : ~10 min
# DESCRIPTION  : Shared panel builder and PPML estimation helper for code/50 (firm x origin x two-week, country-specific signals).
# INPUT FILES  : PRIVATE_CACHE{cells,firms,sig_long}.parquet (from 01_prep_261102.py)
# OUTPUT FILES : none (helpers)
# ============================================================
suppressMessages({library(arrow); library(data.table); library(fixest)})
setFixest_nthreads(2)
this_file <- gsub("~+~", " ", sub("--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE)[1]), fixed = TRUE)
TOPIC <- normalizePath(file.path(dirname(this_file), "..", ".."))
if (nzchar(Sys.getenv("TOPIC_DIR"))) TOPIC <- Sys.getenv("TOPIC_DIR")
OUT   <- file.path(TOPIC, "analysis/50_customs_country_signal_261102")
CACHE <- file.path(Sys.getenv("HOME"), "private_cache/customs50_261102")
LAGS  <- c(2, 4, 6, 8, 12, 16)
SIGS  <- c("threats4", "mentions", "jev", "typed4")

safe_write <- function(x, path) { if (file.exists(path)) stop("output exists; refusing to overwrite: ", path); fwrite(x, path) }

# ★ CORE LOGIC: grid = (firm, origin) pairs with any positive arrival x bins where the firm has any arrival. Under Poisson with
# firm^origin and firm^period FE this equals the full balanced panel, because all-zero groups carry no information.
build_grid <- function(frac = 1, seed = 261102) {
  cells <- as.data.table(read_parquet(file.path(CACHE, "cells.parquet")))
  firms <- as.data.table(read_parquet(file.path(CACHE, "firms.parquet")))
  sig   <- as.data.table(read_parquet(file.path(CACHE, "sig_long.parquet")))
  om    <- fread(file.path(OUT, "01_origin_map_261102.csv"))[!is.na(iso3) & origin != "Other"]
  cells <- merge(cells, om[, .(origin, iso3)], by = "origin")                      # drops Other and Panama (no statement panel row)
  cells <- cells[trade_value > 0]
  np <- cells[, .(n_pairs = uniqueN(origin)), by = final_dunsno][n_pairs >= 2]   # singleton firms cannot be identified
  cells <- cells[final_dunsno %in% np$final_dunsno]
  if (frac < 1) { set.seed(seed); keep <- sample(unique(cells$final_dunsno), round(frac * uniqueN(cells$final_dunsno))); cells <- cells[final_dunsno %in% keep] }
  pairs <- unique(cells[, .(final_dunsno, origin, iso3)])
  fb    <- unique(cells[, .(final_dunsno, bin_index)])
  g <- merge(fb, pairs, by = "final_dunsno", allow.cartesian = TRUE)
  g <- merge(g, cells[, .(final_dunsno, origin, bin_index, y = trade_value)], by = c("final_dunsno", "origin", "bin_index"), all.x = TRUE)
  g[is.na(y), y := 0]
  g <- merge(g, sig, by = c("iso3", "bin_index"))
  g <- merge(g, firms[, .(final_dunsno, alt23, china23, emp_bin)], by = "final_dunsno")
  # standardize: per-SD units over the unique origin-bin cross-section (jurisdiction x bin)
  u <- unique(g[, c("iso3", "bin_index", grep("^(threats4|mentions|jev|typed4|thr_f|thr_u)_k", names(g), value = TRUE)), with = FALSE])
  for (v in SIGS) {
    for (k in LAGS) { sd1 <- sd(u[[paste0(v, "_k", k)]]); set(g, j = paste0("S_", v, "_k", k), value = g[[paste0(v, "_k", k)]] / sd1) }
    cum <- Reduce(`+`, lapply(LAGS, function(k) u[[paste0(v, "_k", k)]])); sdc <- sd(cum)
    set(g, j = paste0("S_", v, "_cum"), value = Reduce(`+`, lapply(LAGS, function(k) g[[paste0(v, "_k", k)]])) / sdc)
  }
  for (nm in c("thr_f", "thr_u")) {
    cum <- Reduce(`+`, lapply(LAGS, function(k) u[[paste0(nm, "_k", k)]])); sdc <- sd(cum)
    set(g, j = paste0("S_", nm, "_cum"), value = Reduce(`+`, lapply(LAGS, function(k) g[[paste0(nm, "_k", k)]])) / sdc)
  }
  g[, period := bin_index]
  g
}

# Two-way (origin, firm) clustered table. MDE = 2.8 x SE (80% power, 5% two-sided).
tidy_fit <- function(m, spec, sample, terms = NULL) {
  v <- tryCatch(summary(m, cluster = ~origin + final_dunsno), error = function(e) NULL)
  ct <- if (is.null(v)) coeftable(m) else coeftable(v)
  v1 <- tryCatch(coeftable(summary(m, cluster = ~final_dunsno)), error = function(e) NULL)
  d <- data.table(spec = spec, sample = sample, term = rownames(ct), est = ct[, 1], se_2way = ct[, 2], p_2way = ct[, 4])
  d[, se_firm := if (is.null(v1)) NA_real_ else v1[term, 2]]
  d[, mde := 2.8 * se_2way]
  d[, `:=`(nobs = nobs(m), n_firms = uniqueN(m$fixef_vars) * 0 + NA_integer_)]
  if (!is.null(terms)) d <- d[term %in% terms]
  d
}
