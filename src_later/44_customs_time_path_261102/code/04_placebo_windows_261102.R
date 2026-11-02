# ============================================================
# START DATE   : 2026-11-02
# CODE RUN DATE: 2026-11-02
# TIME TAKEN   : ~15-30 min per tercile
# DESCRIPTION  : P3. Placebo for the window contrast (resolved minus threat). The actual window rule (frontload 4 months, threat 2
#                months, resolved the next 9 months) is placed on a fake year (2024) with start dates shifted by k weeks
#                (k = +-8..+-26 weekly, plus k = 0), sample = 2023-2024 (bins 27-78), reference = periods outside every window.
#                Same fepois model as W_A. The actual contrast is re-estimated on the same firm subsample (2024-2026 panel).
#                Usage: Rscript 04_placebo_windows_261102.R <tercile> <firm_fraction>
# INPUT FILES  : PRIVATE_CACHE{cells_placebo_261102,cells,firms}.parquet,
#                analysis/44_.../01_placebo_windows_261102.csv, 01_period_table_261102.csv
# OUTPUT FILES : analysis/44_customs_time_path_261102/04_placebo_windows_<tercile>_261102.csv
# ============================================================
source(file.path(dirname(gsub("~+~", " ", sub("--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE)[1]), fixed = TRUE)), "00_common_261102.R"))
args <- commandArgs(TRUE); terc <- args[1]; frac <- as.numeric(args[2])
outf <- file.path(OUT44, sprintf("04_placebo_windows_%s_261102.csv", terc))
if (file.exists(outf)) stop("output exists; refusing to overwrite: ", outf)
PW <- fread(file.path(OUT44, "01_placebo_windows_261102.csv"))
rows <- list()
grab <- function(m, k, kind) {
  b <- coef(m); V <- vcov(m); used <- NULL
  out <- list()
  for (tm in c("cF", "cT", "cR")) { r <- lincomb(b, V, setNames(1, tm)); out[[tm]] <- data.table(kind = kind, shift_weeks = k, term = tm, est = r["est"], se = r["se_firm"]) }
  r <- lincomb(b, V, c(cR = 1, cT = -1)); out[["c"]] <- data.table(kind = kind, shift_weeks = k, term = "cR - cT", est = r["est"], se = r["se_firm"])
  rbindlist(out)[, `:=`(tercile = terc, firm_fraction = frac, nobs = nobs(m))]
}
# ---- actual contrast on the same firm subsample (2024-2026 panel, same windows as code/41 W_A) ----
act <- add_windows_actual(build_grid(terc, frac))
m0 <- fepois(trade_value ~ cF + cT + cR | final_dunsno^grp + final_dunsno^bin, data = act, cluster = ~final_dunsno, nthreads = 1, notes = FALSE, warn = FALSE)
rows[[1]] <- grab(m0, NA_real_, "actual_same_subsample"); rm(act); invisible(gc())
lg(terc, "actual done", rows[[1]][term == "cR - cT", est])
# ---- placebo shifts ----
dat <- build_grid_p(terc, frac)
lg(terc, "placebo grid rows", nrow(dat), "firms", uniqueN(dat$final_dunsno))
for (i in seq_len(nrow(PW))) {
  p <- PW[i]
  if (!isTRUE(p$estimable)) { rows[[length(rows) + 1]] <- data.table(kind = "not_estimable", shift_weeks = p$shift_weeks, term = NA_character_, est = NA_real_, se = NA_real_, tercile = terc, firm_fraction = frac, nobs = NA_integer_); next }
  S <- as.Date(p$S); t1 <- as.Date(p$thr_start); t2 <- as.Date(p$res_start); t3 <- as.Date(p$res_end)
  dat[, `:=`(cF = china * as.numeric(mid_d >= S & mid_d < t1), cT = china * as.numeric(mid_d >= t1 & mid_d < t2), cR = china * as.numeric(mid_d >= t2 & mid_d < t3))]
  m <- tryCatch(fepois(trade_value ~ cF + cT + cR | final_dunsno^grp + final_dunsno^bin, data = dat, cluster = ~final_dunsno, nthreads = 1, notes = FALSE, warn = FALSE),
                error = function(e) { lg("FAILED", p$shift_weeks, conditionMessage(e)); NULL })
  if (!is.null(m)) rows[[length(rows) + 1]] <- grab(m, p$shift_weeks, "placebo")
  fwrite(rbindlist(rows), file.path(CACHE, sprintf("partial44_p3_%s.csv", terc))); lg(terc, "shift", p$shift_weeks)
}
safe_write44(rbindlist(rows), outf); lg("saved", outf)
