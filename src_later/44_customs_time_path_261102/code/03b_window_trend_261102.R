# ============================================================
# START DATE   : 2026-11-02
# CODE RUN DATE: 2026-11-02
# TIME TAKEN   : ~5-15 min per tercile
# DESCRIPTION  : P2 extra (exploratory, added after seeing the 2024 path of the small-firm event study): window model W_A plus a
#                China x linear time trend ((bin-53)/26, one unit = about one year), with and without China x duty at arrival.
#                Usage: Rscript 03b_window_trend_261102.R <tercile> <firm_fraction>
# INPUT FILES  : PRIVATE_CACHE{cells,firms}.parquet, analysis/41_.../01_period_series_261102.csv
# OUTPUT FILES : analysis/44_customs_time_path_261102/03b_window_trend_<tercile>_261102.csv
# ============================================================
source(file.path(dirname(gsub("~+~", " ", sub("--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE)[1]), fixed = TRUE)), "00_common_261102.R"))
args <- commandArgs(TRUE); terc <- args[1]; frac <- as.numeric(args[2]); nth <- if (length(args) >= 3) as.integer(args[3]) else 1L
outf <- file.path(OUT44, sprintf("03b_window_trend_%s_261102.csv", terc))
if (file.exists(outf)) stop("output exists; refusing to overwrite: ", outf)
dat <- add_windows_actual(build_grid(terc, frac))
dat[, trC := china * (bin - 53) / 26]
lg(terc, "firm fraction", frac, "rows", nrow(dat), "firms", uniqueN(dat$final_dunsno))
dcl <- as.data.frame(dat[, .(final_dunsno, bin)])
res <- list()
run <- function(spec, rhs) {
  lg("start", spec)
  m <- fepois(as.formula(paste("trade_value ~", rhs, "| final_dunsno^grp + final_dunsno^bin")), data = dat, cluster = ~final_dunsno, nthreads = nth, notes = FALSE, warn = FALSE)
  b <- coef(m); V1 <- vcov(m); V2 <- tryCatch(summary(m, cluster = dcl)$cov.scaled, error = function(e) NULL)
  used <- dat[obs(m)]; base <- data.table(spec = spec, tercile = terc, nobs = nobs(m), n_firms = uniqueN(used$final_dunsno))
  ws <- c(as.list(setNames(as.list(rep(1, length(b))), names(b))), list())
  for (tm in names(b)) { r <- lincomb(b, V1, setNames(1, tm), V2); res[[length(res) + 1]] <<- cbind(base, term = tm, est = r["est"], se_firm = r["se_firm"], se_2way = r["se_2way"]) }
  for (cn in c("cR - cT", "cT - cF")) {
    w <- if (cn == "cR - cT") c(cR = 1, cT = -1) else c(cT = 1, cF = -1); r <- lincomb(b, V1, w, V2)
    res[[length(res) + 1]] <<- cbind(base, term = cn, est = r["est"], se_firm = r["se_firm"], se_2way = r["se_2way"])
  }
  fwrite(rbindlist(res), file.path(CACHE, sprintf("partial44_p2b_%s.csv", terc))); lg("done", spec)
}
run("P2_f_windows_plus_trend", "cF + cT + cR + trC")
run("P2_g_windows_plus_trend_plus_duty", "cF + cT + cR + trC + dutyC")
safe_write44(rbindlist(res), outf); lg("saved", outf)
