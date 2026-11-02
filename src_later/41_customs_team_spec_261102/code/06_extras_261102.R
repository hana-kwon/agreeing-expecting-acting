# ============================================================
# START DATE   : 2026-11-02
# CODE RUN DATE: 2026-11-02
# TIME TAKEN   : ~15 min per tercile (parallel)
# DESCRIPTION  : Extra specifications: (1) pre-existing alternatives (had a non-China origin in 2024, log number of 2024 origins,
#                2024 China share) interacted with China x signal and China x post-Liberation Day; (2) the same with origin-group x
#                period fixed effects (difference across firms, not the average China decline); (3) total import stop (any arrival
#                in the period) with firm and industry x period fixed effects. One trade-size tercile per run.
#                Usage: Rscript 06_extras_261102.R <tercile> <firm_fraction>
# INPUT FILES  : PRIVATE_CACHE{cells,firms,firm_attrs}.parquet, analysis/41_.../01_period_series_261102.csv
# OUTPUT FILES : analysis/41_customs_team_spec_261102/06_coefs_<tercile>_261102.csv
# ============================================================
source(file.path(dirname(gsub("~+~", " ", sub("--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE)[1]), fixed = TRUE)), "00_common_261102.R"))
args <- commandArgs(TRUE); terc <- args[1]; frac <- as.numeric(args[2])
outf <- file.path(OUT, sprintf("06_coefs_%s_261102.csv", terc))
if (file.exists(outf)) stop("output exists; refusing to overwrite: ", outf)
dat <- add_attrs(build_grid(terc, frac)); dat <- dat[!is.na(alt)]   # firms with 2024 arrivals only
cat(terc, "rows", nrow(dat), "firms", uniqueN(dat$final_dunsno), "alt share", mean(dat[!duplicated(final_dunsno)]$alt), "\n")
res <- list(); lg <- function(...) cat(format(Sys.time(), "%H:%M:%S"), ..., "\n")
FE1 <- "| final_dunsno^grp + final_dunsno^bin"; FE2 <- "| final_dunsno^grp + final_dunsno^bin + grp^bin"
run <- function(spec, y, rhs, fe, d = dat) {
  lg("start", spec)
  m <- tryCatch(fepois(as.formula(paste(y, "~", rhs, fe)), data = d, cluster = ~final_dunsno, nthreads = 1, notes = FALSE, warn = FALSE),
                error = function(e) { lg("FAILED", spec, conditionMessage(e)); NULL })
  if (is.null(m)) return(invisible(NULL))
  res[[length(res) + 1]] <<- tidy_fit(m, d, spec, y, terc); fwrite(rbindlist(res), file.path(CACHE, sprintf("partial06_%s.csv", terc))); lg("done", spec)
}
d_lib <- dat[bin != 86]
# (1) alternatives, same FE as the main model
run("X0_baseline_same_sample", "trade_value", "sigC + dutyC", FE1)
run("X1_alt_signal", "trade_value", "sigC + sigC_alt + dutyC + dutyC_alt", FE1)
run("X2_lnorigins_signal", "trade_value", "sigC + sigC_lno + dutyC + dutyC_lno", FE1)
run("X3_chinashare_signal", "trade_value", "sigC + sigC_sh + dutyC + dutyC_sh", FE1)
run("L0_baseline_same_sample", "trade_value", "libC", FE1, d_lib)
run("L1_alt_lib", "trade_value", "libC + libC_alt", FE1, d_lib)
run("L2_lnorigins_lib", "trade_value", "libC + libC_lno", FE1, d_lib)
run("L3_chinashare_lib", "trade_value", "libC + libC_sh", FE1, d_lib)
# (2) origin-group x period FE: the China x period average is absorbed, only differences across firms remain
run("X1b_alt_signal_grpbinFE", "trade_value", "sigC_alt + dutyC_alt", FE2)
run("X3b_chinashare_signal_grpbinFE", "trade_value", "sigC_sh + dutyC_sh", FE2)
run("L1b_alt_lib_grpbinFE", "trade_value", "libC_alt", FE2, d_lib)
run("L3b_chinashare_lib_grpbinFE", "trade_value", "libC_sh", FE2, d_lib)
rm(dat, d_lib); invisible(gc())
# (3) total stop: any-import indicator, LPM, firm FE and industry (2-digit SIC) x period FE
g <- build_any(terc, frac); g <- g[!is.na(alt) & !is.na(sic2)]
runA <- function(spec, rhs, fe, d = g) {
  lg("start", spec)
  m <- tryCatch(feols(as.formula(paste("any ~", rhs, fe)), data = d, cluster = ~final_dunsno, notes = FALSE, nthreads = 1), error = function(e) { lg("FAILED", spec, conditionMessage(e)); NULL })
  if (is.null(m)) return(invisible(NULL))
  tf <- tidy_fit(m, d, spec, "any_import", terc); tf[, c("p_firm") := .(2 * pnorm(-abs(est / se_firm)))]
  res[[length(res) + 1]] <<- tf; lg("done", spec)
}
runA("A1_any_signal_alt_indperiodFE", "sig_alt + duty_alt", "| final_dunsno + sic2^bin")
runA("A2_any_signal_alt_firmFE_only", "sig_china + sig_alt + duty10 + duty_alt", "| final_dunsno")
runA("A3_any_lib_alt_indperiodFE", "lib_alt", "| final_dunsno + sic2^bin", g[bin != 86])
runA("A4_any_lib_alt_firmFE_only", "post_lib + lib_alt", "| final_dunsno", g[bin != 86])
fwrite(rbindlist(res), outf); lg("saved", outf)
