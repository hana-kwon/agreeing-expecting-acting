# ============================================================
# START DATE   : 2026-11-02
# CODE RUN DATE: 2026-11-02
# TIME TAKEN   : ~? (see log)
# DESCRIPTION  : H3/H4/H5 customs tests: PPML China x signal (all DIME R/D firms) and China x signal x Rep, firm^grp + firm^bin FE,
#                firm-clustered SE, date-shift placebos. Usage: Rscript 03_customs_tests_261102.R <signal_index_1..10> [step]
# INPUT FILES  : PRIVATE_CACHE (code/41 cohort), PRIVATE_CACHE,
#                analysis/41_.../01_period_series_261102.csv, analysis/43_.../02_bin_signals_261102.csv
# OUTPUT FILES : analysis/43_.../03_customs_<signal>_261102.csv (aggregates)
# ============================================================
suppressMessages({library(arrow); library(data.table); library(fixest)})
this_file <- gsub("~+~", " ", sub("--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE)[1]), fixed = TRUE)
TOPIC <- normalizePath(file.path(dirname(this_file), "..", ".."))
OUT <- file.path(TOPIC, "analysis/43_heterogeneity_search_261102")
HC <- file.path(Sys.getenv("HOME"), "private_cache/het43_261102")
C40 <- file.path(Sys.getenv("HOME"), "private_cache/customs40_261102")
SIGS <- c("path_china", "path_broad", "F1_china", "F2_china", "F3_china", "F1_broad", "F2_broad", "F3_broad", "dur_china", "dur_broad")
args <- commandArgs(TRUE); sg <- SIGS[as.integer(args[1])]; step <- if (length(args) > 1) as.integer(args[2]) else 1L
outf <- file.path(OUT, sprintf("03_customs_%s_261102.csv", sg))
if (file.exists(outf)) stop("output exists; refusing to overwrite: ", outf)
# ★ IMPORTANT (change 2026-11-02 03:56, time-driven; 6-group version cost ~100 s per fit): the five non-China origin groups are
# collapsed into one "other" group (China x signal is identified against non-China pooled; firm^group FE now firm^{china,other}).
GR <- c("china", "other")
cells <- as.data.table(read_parquet(file.path(C40, "cells.parquet")))
cells[, grp := ifelse(grp == "china", "china", "other")]
cells <- cells[, .(trade_value = sum(trade_value), weightkg = sum(weightkg)), by = .(final_dunsno, grp, bin_index)]
dm <- as.data.table(read_parquet(file.path(HC, "dime_firms.parquet")))
cells <- cells[final_dunsno %in% dm$final_dunsno]
per <- fread(file.path(TOPIC, "analysis/41_customs_team_spec_261102/01_period_series_261102.csv"))
fb <- unique(cells[, .(final_dunsno, bin_index)])
grid <- fb[, .(grp = GR), by = .(final_dunsno, bin_index)]
grid <- merge(grid, cells, by = c("final_dunsno", "grp", "bin_index"), all.x = TRUE)
set(grid, which(is.na(grid$trade_value)), "trade_value", 0)
grid <- merge(grid, dm, by = "final_dunsno")
grid <- merge(grid, per[, .(bin_index, duty10)], by = "bin_index")
grid[, `:=`(china = as.numeric(grp == "china"), bin = bin_index)]
grid[, dutyC := china * duty10]
# ★ IMPORTANT: Poisson drops firms with all-zero or single-group support anyway; keep only firms with positive cells in >= 2 groups to save time.
ng <- cells[trade_value > 0, .(ng = uniqueN(grp)), by = final_dunsno]
grid <- grid[final_dunsno %in% ng[ng >= 2, final_dunsno]]
cat(sg, "rows", nrow(grid), "firms", uniqueN(grid$final_dunsno), "rep_strict R/D firms",
    grid[, .(r = first(rep_strict)), by = final_dunsno][, table(r, useNA = "ifany")], "\n")
sigs <- fread(file.path(OUT, "02_bin_signals_261102.csv"))[signal == sg]
FE <- "| final_dunsno^grp + final_dunsno^bin"
get_sig <- function(k) { s <- sigs[shift_weeks == k]; setNames(s$z, s$bin_index) }
fit_one <- function(k, dat, rep_col, inter) {
  zz <- get_sig(k)[as.character(dat$bin)]
  d <- copy(dat); d[, sigC := china * zz]
  if (inter) { d[, `:=`(sigCR = sigC * get(rep_col), dutyCR = dutyC * get(rep_col))]
    f <- "trade_value ~ sigC + sigCR + dutyC + dutyCR"; key <- "sigCR" } else { f <- "trade_value ~ sigC + dutyC"; key <- "sigC" }
  d <- d[!is.na(zz)]
  m <- tryCatch(fepois(as.formula(paste(f, FE)), data = d, cluster = ~final_dunsno, nthreads = 1, notes = FALSE, warn = FALSE), error = function(e) NULL)
  if (is.null(m)) return(NULL)
  ct <- coeftable(m)
  list(b = ct[key, 1], se = ct[key, 2], p = ct[key, 4], n = nobs(m), nf = uniqueN(d[obs(m)]$final_dunsno), b_main = if (inter) ct["sigC", 1] else NA_real_)
}
SH <- c(-52:-5, 5:52); SH <- SH[seq(1, length(SH), by = step)]
specs <- list(list(id = "A_china_x_signal_allDIME", rep = NA, inter = FALSE, data = grid[!is.na(rep_strict)], hyp = "H4_H5_avg"),
              list(id = "B_china_x_signal_x_Rep_strict", rep = "rep_strict", inter = TRUE, data = grid[!is.na(rep_strict)], hyp = "H3"))
if (sg %in% c("path_china", "path_broad"))
  specs[[3]] <- list(id = "B_china_x_signal_x_Rep_broad", rep = "rep_broad", inter = TRUE, data = grid[!is.na(rep_broad)], hyp = "H3")
res <- list()
for (sp in specs) {
  t0 <- Sys.time(); a <- fit_one(0, sp$data, sp$rep, sp$inter)
  cat(format(Sys.time(), "%H:%M:%S"), sp$id, "actual", a$b, a$se, "n", a$n, "firms", a$nf, "(", round(as.numeric(Sys.time() - t0, units = "secs")), "s )\n")
  pl <- sapply(SH, function(k) { r <- fit_one(k, sp$data, sp$rep, sp$inter); if (is.null(r)) NA_real_ else r$b })
  pl <- pl[!is.na(pl)]
  res[[length(res) + 1]] <- data.table(hyp = sp$hyp, spec = sp$id, signal = sg, coef = a$b, se = a$se, p_cluster = a$p, n = a$n, n_firms = a$nf,
                                      coef_signal_main_in_B = a$b_main, expected_sign = if (sp$inter) -1 else NA_real_,
                                      placebo_p = mean(abs(pl) >= abs(a$b)), n_shifts = length(pl), shift_mean = mean(pl), shift_sd = sd(pl))
  fwrite(rbindlist(res), file.path(HC, sprintf("partial_%s.csv", sg)))
}
fwrite(rbindlist(res), outf); cat("saved", outf, "\n")
