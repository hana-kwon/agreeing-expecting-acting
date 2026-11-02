# ============================================================
# START DATE   : 2026-11-02
# CODE RUN DATE: 2026-11-02
# TIME TAKEN   : ~15 min per tercile (parallel)
# DESCRIPTION  : Date-shift placebo for the China-path signal: the 28-day statement window is moved by k weeks and three models
#                are re-estimated with Duty kept at its actual values: P_M1 main model (China x Signal), P_X1 China x Signal x
#                "had a non-China origin in 2024", P_A1 total-stop LPM (Signal x alternatives, firm and industry x period FE).
#                |b(shift)| is compared with |b(actual)| on the same subsample. Usage: Rscript 03_placebo_261102.R <tercile> <firm_fraction>
# INPUT FILES  : PRIVATE_CACHE{cells,firms,firm_attrs}.parquet, analysis/41_.../01_signal_shifts_261102.csv
# OUTPUT FILES : analysis/41_customs_team_spec_261102/03_placebo_<tercile>_261102.csv
# ============================================================
source(file.path(dirname(gsub("~+~", " ", sub("--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE)[1]), fixed = TRUE)), "00_common_261102.R"))
args <- commandArgs(TRUE); terc <- args[1]; frac <- as.numeric(args[2])
# optional args: 3 = comma-separated shifts (subset run), 4 = file suffix; used to split the slow large tercile across two processes
ks_arg <- if (length(args) >= 3) as.numeric(strsplit(args[3], ",")[[1]]) else NULL
suffix <- if (length(args) >= 4) args[4] else ""
outf <- file.path(OUT, sprintf("03_placebo_%s%s_261102.csv", terc, suffix))
if (file.exists(outf)) stop("output exists; refusing to overwrite: ", outf)
dat <- add_attrs(build_grid(terc, frac)); g <- build_any(terc, frac); g <- g[!is.na(alt) & !is.na(sic2)]
sh <- fread(file.path(OUT, "01_signal_shifts_261102.csv"))[signal == "china"]
# ★ IMPORTANT: shift grid is every third week from 5 to 50 weeks in both directions (32 shifts), reduced from 5..52 by one week for compute.
# The reduction and the firm subsample were decided from fit timing before any result was seen.
KS <- c(0, sort(c(seq(5, 50, by = 3), -seq(5, 50, by = 3))))
if (!is.null(ks_arg)) KS <- ks_arg
rows <- list(); lg <- function(...) cat(format(Sys.time(), "%H:%M:%S"), ...)
grab <- function(m, model, k, term) {
  ct <- coeftable(m); data.table(tercile = terc, firm_fraction = frac, model = model, shift_weeks = k, term = term, b = ct[term, 1], se = ct[term, 2], nobs = nobs(m))
}
for (k in KS) {
  zm <- sh[shift_weeks == k]; zmap <- setNames(zm$z, zm$bin_index)
  dat[, sigS := china * zmap[as.character(bin)]]; dat[, sigS_alt := sigS * alt]
  m1 <- tryCatch(fepois(trade_value ~ sigS + dutyC | final_dunsno^grp + final_dunsno^bin, data = dat, cluster = ~final_dunsno, nthreads = 1, notes = FALSE, warn = FALSE), error = function(e) NULL)
  if (!is.null(m1)) { rows[[length(rows) + 1]] <- grab(m1, "P_M1", k, "sigS"); rows[[length(rows) + 1]] <- grab(m1, "P_M1", k, "dutyC") }
  m2 <- tryCatch(fepois(trade_value ~ sigS + sigS_alt + dutyC + dutyC_alt | final_dunsno^grp + final_dunsno^bin, data = dat[!is.na(alt)], cluster = ~final_dunsno, nthreads = 1, notes = FALSE, warn = FALSE), error = function(e) NULL)
  if (!is.null(m2)) { rows[[length(rows) + 1]] <- grab(m2, "P_X1", k, "sigS"); rows[[length(rows) + 1]] <- grab(m2, "P_X1", k, "sigS_alt") }
  g[, sigA := zmap[as.character(bin)] * alt]
  m3 <- tryCatch(feols(any ~ sigA + duty_alt | final_dunsno + sic2^bin, data = g, cluster = ~final_dunsno, notes = FALSE, nthreads = 1), error = function(e) NULL)
  if (!is.null(m3)) rows[[length(rows) + 1]] <- grab(m3, "P_A1", k, "sigA")
  fwrite(rbindlist(rows), file.path(CACHE, sprintf("partial_placebo_%s%s.csv", terc, suffix))); lg(terc, "shift", k, "\n")
}
fwrite(rbindlist(rows), outf)
