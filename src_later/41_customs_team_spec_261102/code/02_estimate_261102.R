# ============================================================
# START DATE   : 2026-11-02
# CODE RUN DATE: 2026-11-02
# TIME TAKEN   : ~15 min per tercile (parallel)
# DESCRIPTION  : Main signal model (China x Signal_{t-1} + China x Duty_t), broad-path and threat-count variants,
#                team event replication (post-Liberation Day x China) and the TPU_Small threat/resolved window
#                specification, one firm-size tercile per run. Usage: Rscript 02_estimate_261102.R <tercile> <firm_fraction>
# INPUT FILES  : PRIVATE_CACHE{cells,firms}.parquet, analysis/41_.../01_period_series_261102.csv
# OUTPUT FILES : analysis/41_customs_team_spec_261102/02_coefs_<tercile>_261102.csv
# ============================================================
source(file.path(dirname(gsub("~+~", " ", sub("--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE)[1]), fixed = TRUE)), "00_common_261102.R"))
args <- commandArgs(TRUE); terc <- args[1]; frac <- as.numeric(args[2])
outf <- file.path(OUT, sprintf("02_coefs_%s_261102.csv", terc))
if (file.exists(outf)) stop("output exists; refusing to overwrite: ", outf)
dat <- build_grid(terc, frac)
cat(terc, "firm fraction", frac, "rows", nrow(dat), "firms", uniqueN(dat$final_dunsno), "\n")
FE <- "| final_dunsno^grp + final_dunsno^bin"
res <- list(); lg <- function(...) cat(format(Sys.time(), "%H:%M:%S"), ..., "\n")
run <- function(spec, y, rhs, d = dat, extra = NULL) {
  lg("start", spec, y)
  m <- tryCatch(fepois(as.formula(paste(y, "~", rhs, FE)), data = d, cluster = ~final_dunsno, nthreads = 2, notes = FALSE, warn = FALSE),
                error = function(e) { lg("FAILED", spec, conditionMessage(e)); NULL })
  if (is.null(m)) return(invisible(NULL))
  tf <- tidy_fit(m, d, spec, y, terc)
  if (!is.null(extra)) {   # linear combination of two coefficients, firm-clustered
    for (ex in extra) {
      b <- coef(m); V <- vcov(m); est <- b[ex[1]] - b[ex[2]]; se <- sqrt(V[ex[1], ex[1]] + V[ex[2], ex[2]] - 2 * V[ex[1], ex[2]])
      tf <- rbind(tf, data.table(spec = spec, outcome = y, tercile = terc, term = paste(ex[1], "-", ex[2]), est = est, se_firm = se,
                                 p_firm = 2 * pnorm(-abs(est / se)), se_2way = NA_real_, p_2way = NA_real_, nobs = nobs(m),
                                 n_firms = tf$n_firms[1], n_periods = tf$n_periods[1]))
    }
  }
  res[[length(res) + 1]] <<- tf; fwrite(rbindlist(res), file.path(CACHE, sprintf("partial_%s.csv", terc))); lg("done", spec, y)
}

# ---- main signal models ----
run("M1_signal_china_plus_duty", "trade_value", "sigC + dutyC")
run("M2_signal_china_only", "trade_value", "sigC")
run("M4_signal_broad_plus_duty", "trade_value", "sigB + dutyC")
run("M5_threat_count_plus_duty", "trade_value", "thrC + dutyC")
run("M1_signal_china_plus_duty", "weightkg", "sigC + dutyC")

# ---- team replication: post-Liberation Day x China (event bin 86 dropped; post = bin >= 87) ----
dat[, `:=`(post_lib = as.numeric(bin >= 87))]; dat[, libC := china * post_lib]
d_lib <- dat[bin != 86]; d_lib91 <- dat[bin != 86 & bin <= 91]
run("R_lib_full_window", "trade_value", "libC", d_lib)
run("R_lib_through_2025_06", "trade_value", "libC", d_lib91)
run("R_lib_full_window", "weightkg", "libC", d_lib)
run("R_lib_full_window", "volumeteu", "libC", d_lib)

# ---- TPU_Small windows (frontload 2025-01..04, threat 2025-05..month(R)+1, resolved month(R)+2..); bin month = month of midpoint ----
M <- function(y, m) y * 12 + m - 1
dat[, mon := year(as.Date(mid)) * 12 + month(as.Date(mid)) - 1]
Rm <- c(china = M(2025, 5), east_asia = M(2025, 7), eu = M(2025, 7), mexico = M(2025, 7), canada = M(2025, 7), rest = M(2025, 7))
dat[, Rmon := Rm[grp]]
dat[, `:=`(wF = as.numeric(mon >= M(2025, 1) & mon <= M(2025, 4)), wT = as.numeric(mon >= M(2025, 5) & mon <= Rmon + 1), wR = as.numeric(mon >= Rmon + 2))]
dat[, `:=`(cF = (grp == "china") * wF, cT = (grp == "china") * wT, cR = (grp == "china") * wR,
           eF = (grp == "eu") * wF, eT = (grp == "eu") * wT, eR = (grp == "eu") * wR,
           aF = (grp == "east_asia") * wF, aT = (grp == "east_asia") * wT, aR = (grp == "east_asia") * wR)]
run("W_A_china_windows", "trade_value", "cF + cT + cR", dat, extra = list(c("cT", "cR"), c("cT", "cF")))
run("W_B_china_eu_eastasia_windows", "trade_value", "cF + cT + cR + eF + eT + eR + aF + aT + aR", dat,
    extra = list(c("cT", "cR"), c("eT", "eR"), c("aT", "aR")))
safe_out <- rbindlist(res); fwrite(safe_out, outf); lg("saved", outf)
