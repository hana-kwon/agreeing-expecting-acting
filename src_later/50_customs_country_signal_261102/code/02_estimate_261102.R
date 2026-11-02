# ============================================================
# START DATE   : 2026-11-02
# CODE RUN DATE: 2026-11-02
# TIME TAKEN   : ~15 min compute (three jobs in parallel)
# DESCRIPTION  : PPML estimates for the country-specific signal design. Job A: lags, cumulative, signal variants, followed/unfollowed,
#                China-excluded (10% firms). Job B: employment bins, alt-origin split, S x alt with origin^period FE (25% firms).
#                Job C: country-level permutation and date-shift placebo for the cumulative threat signal (3% firms).
# INPUT FILES  : PRIVATE_CACHE*.parquet
# OUTPUT FILES : analysis/50_customs_country_signal_261102/02_job<A|B|C>_coefs_261102.csv, 02_jobC_placebo_261102.csv
# ============================================================
args <- commandArgs(trailingOnly = TRUE); JOB <- args[1]
source(file.path(dirname(gsub("~+~", " ", sub("--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE)[1]), fixed = TRUE)), "00_common_261102.R"))
FE <- "final_dunsno^origin + final_dunsno^period"
res <- list()
fit <- function(d, rhs, spec, sample, fe = FE, keep = NULL) {
  t0 <- Sys.time()
  m <- tryCatch(fepois(as.formula(paste("y ~", rhs, "|", fe)), d, cluster = ~origin + final_dunsno, warn = FALSE, notes = FALSE), error = function(e) { message("FAIL ", spec, ": ", e$message); NULL })
  if (is.null(m)) return(invisible(NULL))
  cn <- names(coef(m)); if (is.null(keep)) keep <- cn[grepl("^S_|^Rnum|^Rind|:", cn)]
  r <- tidy_fit(m, spec, sample, keep); r[, `:=`(n_firms_grid = uniqueN(d$final_dunsno), secs = as.numeric(difftime(Sys.time(), t0, units = "secs")))]
  res[[length(res) + 1]] <<- r; message(spec, " | ", sample, " | ", round(r$secs[1]), "s")
  m
}
R2 <- "Rind_k0 + Rind_k8"
if (JOB == "A") {
  g0 <- build_grid(0.10)
  for (ex in c(FALSE, TRUE)) {
    g <- if (ex) g0[origin != "China"] else g0; sm <- ifelse(ex, "exChina_10pct", "all_10pct")
    for (k in LAGS) fit(g, sprintf("S_threats4_k%d + Rind_k%d", k, k), paste0("A1_lag", k), sm)
    fit(g, paste(sprintf("S_threats4_k%d", LAGS), collapse = " + ") |> paste("+", R2), "A2_joint", sm)
    fit(g, paste("S_threats4_cum +", R2), "A3_cum_threats4", sm)
    fit(g[!is.na(Rnum_k0) & !is.na(Rnum_k8)], "S_threats4_cum + Rnum_k0 + Rnum_k8", "A4_cum_Rnum", sm)
    for (v in c("mentions", "jev", "typed4")) fit(g, paste0("S_", v, "_cum + ", R2), paste0("D_cum_", v), sm)
    fit(g, paste("S_thr_f_cum + S_thr_u_cum +", R2), "E_followed_vs_unfollowed", sm)
    fit(g[lag16_in_panel == 1], paste("S_threats4_cum +", R2), "R1_cum_all_lags_in_panel", sm)
  }
  safe_write(rbindlist(res), file.path(OUT, "02_jobA_coefs_261102.csv"))
}
if (JOB == "B") {
  g0 <- build_grid(0.25)
  for (ex in c(FALSE, TRUE)) {
    g <- if (ex) g0[origin != "China"] else g0; sm0 <- ifelse(ex, "exChina", "all")
    for (e in c("e1_19", "e20_49", "e50_249", "e250p")) { d <- g[emp_bin == e]; if (nrow(d) > 0) fit(d, paste("S_threats4_cum +", R2), paste0("F_emp_", e), paste0(sm0, "_25pct")) }
    for (a in c(0, 1)) fit(g[alt23 == a], paste("S_threats4_cum +", R2), paste0("F_alt23_", a), paste0(sm0, "_25pct"))
    # S main is absorbed by origin^period FE; only the interaction with the firm's 2023 alternative origin is identified
    fit(g, "S_threats4_cum:alt23 + Rind_k0:alt23", "G_Sxalt_originperiodFE", paste0(sm0, "_25pct"), fe = paste(FE, "+ origin^period"))
  }
  safe_write(rbindlist(res), file.path(OUT, "02_jobB_coefs_261102.csv"))
}
if (JOB %in% c("C", "D")) {
  # Job D (added after seeing job A, labeled   ): same placebo design restricted to bins where all 16-week lags are inside the statement panel
  g0 <- build_grid(0.03); if (JOB == "D") g0 <- g0[lag16_in_panel == 1]; set.seed(261102); NPERM <- 60
  out <- list()
  for (ex in c(FALSE, TRUE)) {
    g <- if (ex) g0[origin != "China"] else g0; sm <- paste0(ifelse(ex, "exChina_3pct", "all_3pct"), ifelse(JOB == "D", "_lagsInPanel", ""))
    m0 <- fit(g, paste("S_threats4_cum +", R2), "C0_actual", sm); b0 <- coef(m0)["S_threats4_cum"]
    sr <- unique(g[, .(iso3, bin_index, S = S_threats4_cum)])
    run <- function(sv, kind, id) {
      d <- merge(g[, !"S_threats4_cum"], sv, by = c("iso3", "bin_index")); setnames(d, "S", "S_threats4_cum")
      m <- tryCatch(fepois(as.formula(paste("y ~ S_threats4_cum +", R2, "|", FE)), d, cluster = ~final_dunsno, warn = FALSE, notes = FALSE), error = function(e) NULL)
      if (!is.null(m)) out[[length(out) + 1]] <<- data.table(sample = sm, kind = kind, id = id, est = coef(m)["S_threats4_cum"], actual = b0)
    }
    for (i in 1:NPERM) {   # permute the signal across jurisdictions within each bin
      p <- copy(sr); p[, S := S[sample.int(.N)], by = bin_index]; run(p[, .(iso3, bin_index, S)], "perm_country", i)
    }
    bins <- sort(unique(sr$bin_index)); nb <- length(bins)
    for (j in c(-25, -23, -21, -19, -17, -15, -13, -11, -9, -7, -5, -3, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25)) {   # circular date shift in bins (6-50 weeks)
      p <- copy(sr)[order(iso3, bin_index)]; p[, S := shift(S, j, type = "cyclic"), by = iso3]; run(p[, .(iso3, bin_index, S)], "date_shift", j)
    }
    message("C done ", sm)
  }
  safe_write(rbindlist(out), file.path(OUT, paste0("02_job", JOB, "_placebo_261102.csv")))
  safe_write(rbindlist(res), file.path(OUT, paste0("02_job", JOB, "_coefs_261102.csv")))
}
