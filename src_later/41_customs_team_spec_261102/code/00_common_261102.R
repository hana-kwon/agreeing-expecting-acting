# ============================================================
# START DATE   : 2026-11-02
# CODE RUN DATE: 2026-11-02
# TIME TAKEN   : ~10 min
# DESCRIPTION  : Shared paths and panel builder for code/41 (firm x origin-group x two-week panel with China-interacted
#                regressors). Firm rows are read from and kept in PRIVATE_CACHE only.
# INPUT FILES  : PRIVATE_CACHE{cells,firms}.parquet (from 01_prep_261102.py),
#                analysis/41_customs_team_spec_261102/01_period_series_261102.csv, 01_signal_shifts_261102.csv
# OUTPUT FILES : none
# ============================================================
suppressMessages({library(arrow); library(data.table); library(fixest)})

this_file <- gsub("~+~", " ", sub("--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE)[1]), fixed = TRUE)
TOPIC <- normalizePath(file.path(dirname(this_file), "..", ".."))
OUT   <- file.path(TOPIC, "analysis/41_customs_team_spec_261102")
CACHE <- file.path(Sys.getenv("HOME"), "private_cache/customs40_261102")
GR    <- c("china", "east_asia", "eu", "mexico", "canada", "rest")

safe_write <- function(x, path) {
  if (file.exists(path)) stop("output exists; refusing to overwrite: ", path)
  fwrite(x, path)
}

# ★ CORE LOGIC: Poisson drops firm-period and firm-origin groups that are all zero, so the balanced grid is built only
# for firm-periods with any positive arrival (6 origin groups each). Same likelihood as the full balanced panel.
build_grid <- function(terc, frac = 1, seed = 261102) {
  cells <- as.data.table(read_parquet(file.path(CACHE, "cells.parquet")))
  firms <- as.data.table(read_parquet(file.path(CACHE, "firms.parquet")))[, .(final_dunsno, tercile)]
  firms <- firms[tercile == terc]
  # compute-driven random firm subsample (seeded); decided from timing only, not from results
  if (frac < 1) { set.seed(seed); firms <- firms[sample(.N, round(frac * .N))] }
  cells <- cells[final_dunsno %in% firms$final_dunsno]
  per   <- fread(file.path(OUT, "01_period_series_261102.csv"))
  fb    <- unique(cells[, .(final_dunsno, bin_index)])
  grid  <- fb[, .(grp = GR), by = .(final_dunsno, bin_index)]
  grid  <- merge(grid, cells, by = c("final_dunsno", "grp", "bin_index"), all.x = TRUE)
  for (y in c("trade_value", "weightkg", "volumeteu")) set(grid, which(is.na(grid[[y]])), y, 0)
  grid  <- merge(grid, firms, by = "final_dunsno")
  grid  <- merge(grid, per[, .(bin_index, bin_start, mid = as.character(as.Date(bin_start) + 7), duty10, sig_china, sig_broad, threat4_z)],
                 by = "bin_index")
  grid[, china := as.numeric(grp == "china")]
  grid[, `:=`(sigC = china * sig_china, sigB = china * sig_broad, dutyC = china * duty10, thrC = china * threat4_z)]
  grid[, bin := bin_index]
  grid
}

# Coefficient table with firm-clustered SE (as specified) and firm+period two-way SE (sensitivity).
tidy_fit <- function(m, dat, spec, outcome, tercile) {
  ct <- coeftable(m)
  v2 <- tryCatch(coeftable(summary(m, cluster = as.data.frame(dat[, .(final_dunsno, bin)]))), error = function(e) NULL)
  used <- dat[obs(m)]
  data.table(spec = spec, outcome = outcome, tercile = tercile, term = rownames(ct), est = ct[, 1], se_firm = ct[, 2],
             p_firm = ct[, 4], se_2way = if (is.null(v2)) NA_real_ else v2[rownames(ct), 2],
             p_2way = if (is.null(v2)) NA_real_ else v2[rownames(ct), 4],
             nobs = nobs(m), n_firms = uniqueN(used$final_dunsno), n_periods = uniqueN(used$bin))
}

# ---- additions 2026-11-02 (extra specifications requested by the coordinator after the main run started) ----
# 2024 firm attributes (07_prep_extras_261102.py) and the China-interacted versions used in the extra models.
add_attrs <- function(dat) {
  fa <- as.data.table(read_parquet(file.path(CACHE, "firm_attrs.parquet")))[, .(final_dunsno, n_origins_2024, china_share_2024, has_alt, emp_bin, sic2)]
  dat <- merge(dat, fa, by = "final_dunsno", all.x = TRUE)
  dat[, `:=`(lno = log(n_origins_2024), alt = has_alt, sh = china_share_2024)]
  dat[, `:=`(post_lib = as.numeric(bin >= 87))]
  dat[, `:=`(libC = china * post_lib,
             sigC_alt = sigC * alt, dutyC_alt = dutyC * alt, sigC_lno = sigC * lno, dutyC_lno = dutyC * lno, sigC_sh = sigC * sh, dutyC_sh = dutyC * sh,
             libC_alt = china * post_lib * alt, libC_lno = china * post_lib * lno, libC_sh = china * post_lib * sh)]
  dat
}

# ★ CORE LOGIC: firm x period panel for the total-stop outcome. any = 1 if the firm has any arrival in the period (any origin).
# All 58 periods are kept for every firm (zeros are the outcome), so switching origins (any stays 1) is told apart from stopping (any = 0).
build_any <- function(terc, frac = 1, seed = 261102) {
  cells <- as.data.table(read_parquet(file.path(CACHE, "cells.parquet")))
  firms <- as.data.table(read_parquet(file.path(CACHE, "firms.parquet")))[, .(final_dunsno, tercile)][tercile == terc]
  if (frac < 1) { set.seed(seed); firms <- firms[sample(.N, round(frac * .N))] }
  per <- fread(file.path(OUT, "01_period_series_261102.csv"))
  fb  <- unique(cells[final_dunsno %in% firms$final_dunsno & weightkg > 0, .(final_dunsno, bin_index)])[, any := 1]
  g   <- CJ(final_dunsno = firms$final_dunsno, bin_index = per$bin_index)
  g   <- merge(g, fb, by = c("final_dunsno", "bin_index"), all.x = TRUE); g[is.na(any), any := 0]
  g   <- merge(g, per[, .(bin_index, duty10, sig_china)], by = "bin_index")
  fa  <- as.data.table(read_parquet(file.path(CACHE, "firm_attrs.parquet")))[, .(final_dunsno, n_origins_2024, china_share_2024, has_alt, sic2)]
  g   <- merge(g, fa, by = "final_dunsno", all.x = TRUE)
  g[, `:=`(bin = bin_index, alt = has_alt, lno = log(n_origins_2024), post_lib = as.numeric(bin_index >= 87))]
  g[, `:=`(sig_alt = sig_china * alt, duty_alt = duty10 * alt, lib_alt = post_lib * alt)]
  g
}
