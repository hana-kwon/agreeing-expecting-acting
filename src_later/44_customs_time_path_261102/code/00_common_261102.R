# ============================================================
# START DATE   : 2026-11-02
# CODE RUN DATE: 2026-11-02
# TIME TAKEN   : ~10 min
# DESCRIPTION  : Shared helpers for code/44. Re-uses code/41/00_common_261102.R (build_grid: firm x origin-group x two-week
#                panel, same firms, same size terciles, same seeded subsamples) and adds the 2023-2024 placebo grid, the
#                period table with lagged China duty, and linear-combination helpers. Firm rows only in PRIVATE_CACHE
# INPUT FILES  : code/41_customs_team_spec_261102/00_common_261102.R, PRIVATE_CACHE*.parquet,
#                analysis/44_customs_time_path_261102/01_period_table_261102.csv, 01_placebo_windows_261102.csv
# OUTPUT FILES : none
# ============================================================
this_file44 <- gsub("~+~", " ", sub("--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE)[1]), fixed = TRUE)
# 41's common file derives TOPIC from the running script (two levels up), which is the same for code/44; its OUT is analysis/41 (read-only use here)
source(file.path(dirname(this_file44), "..", "41_customs_team_spec_261102", "00_common_261102.R"))
OUT44 <- file.path(TOPIC, "analysis/44_customs_time_path_261102")
PT    <- fread(file.path(OUT44, "01_period_table_261102.csv"))
PT[, `:=`(bin = bin_index, mid_d = as.Date(mid))]

safe_write44 <- function(x, path) {
  if (file.exists(path)) stop("output exists; refusing to overwrite: ", path)
  fwrite(x, path)
}
lg <- function(...) cat(format(Sys.time(), "%H:%M:%S"), ..., "\n")

# ★ KEY FUNCTION: linear combination w'b with firm-clustered SE (V1) and optional firm+period two-way SE (V2).
lincomb <- function(b, V1, w, V2 = NULL) {
  nm <- names(w); est <- sum(w * b[nm]); se1 <- sqrt(drop(t(w) %*% V1[nm, nm, drop = FALSE] %*% w))
  se2 <- if (is.null(V2)) NA_real_ else sqrt(drop(t(w) %*% V2[nm, nm, drop = FALSE] %*% w))
  c(est = est, se_firm = se1, se_2way = se2)
}

# China x window dummies exactly as in code/41/02_estimate_261102.R (W_A): month of the bin midpoint, resolved China-area date 2025-05-12
add_windows_actual <- function(dat) {
  M <- function(y, m) y * 12 + m - 1
  dat[, mon := year(as.Date(mid)) * 12 + month(as.Date(mid)) - 1]
  dat[, `:=`(wF = as.numeric(mon >= M(2025, 1) & mon <= M(2025, 4)), wT = as.numeric(mon >= M(2025, 5) & mon <= M(2025, 6)),
             wR = as.numeric(mon >= M(2025, 7)))]
  dat[, `:=`(cF = china * wF, cT = china * wT, cR = china * wR)]
  dat
}

# Placebo grid: 2023-2024 cells (bins 27-78), same cohort firms and the same seeded firm subsample as build_grid().
build_grid_p <- function(terc, frac = 1, seed = 261102) {
  cells <- as.data.table(read_parquet(file.path(CACHE, "cells_placebo_261102.parquet")))
  firms <- as.data.table(read_parquet(file.path(CACHE, "firms.parquet")))[, .(final_dunsno, tercile)][tercile == terc]
  if (frac < 1) { set.seed(seed); firms <- firms[sample(.N, round(frac * .N))] }
  cells <- cells[final_dunsno %in% firms$final_dunsno]
  fb    <- unique(cells[, .(final_dunsno, bin_index)])
  grid  <- fb[, .(grp = GR), by = .(final_dunsno, bin_index)]
  grid  <- merge(grid, cells, by = c("final_dunsno", "grp", "bin_index"), all.x = TRUE)
  for (y in c("trade_value", "weightkg", "volumeteu")) set(grid, which(is.na(grid[[y]])), y, 0)
  grid  <- merge(grid, PT[, .(bin_index, mid_d)], by = "bin_index")
  grid[, `:=`(china = as.numeric(grp == "china"), bin = bin_index)]
  grid
}
