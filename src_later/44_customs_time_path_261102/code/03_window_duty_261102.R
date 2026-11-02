# ============================================================
# START DATE   : 2026-11-02
# CODE RUN DATE: 2026-11-02
# TIME TAKEN   : ~10-20 min per tercile
# DESCRIPTION  : P2. Window model W_A (China x frontload/threat/resolved) with China x duty in force added: duty at the arrival
#                period (lag 0), lagged one and two periods, all three together, and duty alone without windows. PPML on trade value,
#                firm^origin_group + firm^period FE, firm-clustered SE and firm+period two-way SE. Also contrasts cR-cT and cT-cF.
#                Usage: Rscript 03_window_duty_261102.R <tercile> <firm_fraction>
# INPUT FILES  : PRIVATE_CACHE{cells,firms}.parquet, analysis/41_.../01_period_series_261102.csv,
#                analysis/44_.../01_period_table_261102.csv (duty10_lag0..2)
# OUTPUT FILES : analysis/44_customs_time_path_261102/03_window_duty_<tercile>_261102.csv
# ============================================================
source(file.path(dirname(gsub("~+~", " ", sub("--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE)[1]), fixed = TRUE)), "00_common_261102.R"))
args <- commandArgs(TRUE); terc <- args[1]; frac <- as.numeric(args[2]); nth <- if (length(args) >= 3) as.integer(args[3]) else 1L
outf <- file.path(OUT44, sprintf("03_window_duty_%s_261102.csv", terc))
if (file.exists(outf)) stop("output exists; refusing to overwrite: ", outf)
dat <- build_grid(terc, frac)
dat <- merge(dat, PT[, .(bin_index, d1 = duty10_lag1, d2 = duty10_lag2, d0 = duty10_lag0)], by = "bin_index")
stopifnot(max(abs(dat$d0 - dat$duty10)) < 1e-9)   # lag-0 duty in the new table equals 41's duty10
dat[, `:=`(dutyC1 = china * d1, dutyC2 = china * d2)]
dat <- add_windows_actual(dat)
lg(terc, "firm fraction", frac, "rows", nrow(dat), "firms", uniqueN(dat$final_dunsno))
FE <- "| final_dunsno^grp + final_dunsno^bin"
dcl <- as.data.frame(dat[, .(final_dunsno, bin)])
res <- list()
run <- function(spec, rhs, contrasts) {
  lg("start", spec)
  m <- fepois(as.formula(paste("trade_value ~", rhs, FE)), data = dat, cluster = ~final_dunsno, nthreads = nth, notes = FALSE, warn = FALSE)
  b <- coef(m); V1 <- vcov(m); V2 <- tryCatch(summary(m, cluster = dcl)$cov.scaled, error = function(e) NULL)
  used <- dat[obs(m)]; ct <- coeftable(m)
  base <- data.table(spec = spec, tercile = terc, nobs = nobs(m), n_firms = uniqueN(used$final_dunsno))
  for (tm in names(b)) {
    r <- lincomb(b, V1, setNames(1, tm), V2)
    res[[length(res) + 1]] <<- cbind(base, term = tm, est = r["est"], se_firm = r["se_firm"], se_2way = r["se_2way"])
  }
  for (cn in names(contrasts)) {
    w <- contrasts[[cn]]; if (!all(names(w) %in% names(b))) next
    r <- lincomb(b, V1, w, V2)
    res[[length(res) + 1]] <<- cbind(base, term = cn, est = r["est"], se_firm = r["se_firm"], se_2way = r["se_2way"])
  }
  fwrite(rbindlist(res), file.path(CACHE, sprintf("partial44_p2_%s.csv", terc))); lg("done", spec)
}
CT <- list("cR - cT" = c(cR = 1, cT = -1), "cT - cF" = c(cT = 1, cF = -1))
run("P2_0_windows_only",       "cF + cT + cR",                        CT)
run("P2_a_plus_duty_t",        "cF + cT + cR + dutyC",                CT)
run("P2_b_plus_duty_lag1",     "cF + cT + cR + dutyC1",               CT)
run("P2_c_plus_duty_lag2",     "cF + cT + cR + dutyC2",               CT)
run("P2_d_plus_duty_t_lag1_lag2", "cF + cT + cR + dutyC + dutyC1 + dutyC2", c(CT, list("duty sum t+lag1+lag2" = c(dutyC = 1, dutyC1 = 1, dutyC2 = 1))))
run("P2_e_duty_only",          "dutyC",                               list())
safe_write44(rbindlist(res), outf); lg("saved", outf)
