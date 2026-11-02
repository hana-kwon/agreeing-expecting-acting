# ============================================================
# START DATE   : 2026-11-02
# CODE RUN DATE: 2026-11-02
# TIME TAKEN   : ~5-20 min per tercile
# DESCRIPTION  : C1-C4. Cohort = firms active in 2022 or 2023 (sample cap 4000 per tercile). PPML (fepois, firm^grp + firm^bin FE, firm-clustered SE).
#                M1 saturated China x bin (bins 1-111): same-calendar placebo (2024 vs 2023, 2023 vs 2022) and real (2025 vs 2024) period
#                differences, and China-trend-adjusted path (linear fit of the China coefficient over 2023-2024 bins, extrapolated).
#                M2 origin-group x window models, real (2024 reference) and one-year-earlier placebo (2023 reference).
#                M3 cut windows: China x post (+ China x linear relative position), real and same-calendar placebos 1 and 2 years earlier.
#                Usage: Rscript 03_c_es.R <tercile>
# INPUT FILES  : PRIVATE_CACHE, analysis/54.../03_bin_calendar_261102.csv
# OUTPUT FILES : analysis/54_review_fixes_261102/03_{m1_coefs,m1_summary,m2_groups,m3_cuts}_<tercile>_261102.csv
# ============================================================
suppressMessages({library(arrow); library(data.table); library(fixest)})
this_file <- gsub("~+~", " ", sub("--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE)[1]), fixed = TRUE)
TOPIC <- normalizePath(file.path(dirname(this_file), "..", ".."))
OUT <- file.path(TOPIC, "analysis/54_review_fixes_261102"); CACHE <- file.path(Sys.getenv("HOME"), "private_cache/review54_261102")
terc <- commandArgs(TRUE)[1]; nth <- 2L
w <- function(name) { f <- file.path(OUT, sprintf("03_%s_%s_261102.csv", name, terc)); if (file.exists(f)) stop("refuse to overwrite ", f); f }
f1 <- w("m1_coefs"); f2 <- w("m1_summary"); f3 <- w("m2_groups"); f4 <- w("m3_cuts")
lg <- function(...) cat(format(Sys.time(), "%H:%M:%S"), ..., "\n")
cal <- fread(file.path(OUT, "03_bin_calendar_261102.csv")); cal[, `:=`(bs = as.Date(bin_start), be = as.Date(bin_end), mid = as.Date(bin_start) + 7)]
cal[, mon := year(mid) * 12 + month(mid) - 1]
cells <- as.data.table(read_parquet(file.path(CACHE, "cells7.parquet")))[tercile == terc]
GR <- c("china", "vietnam", "east_asia_other", "eu", "mexico", "canada", "rest")
fb <- unique(cells[, .(final_dunsno, bin_index)])
grid <- fb[, .(grp = GR), by = .(final_dunsno, bin_index)]
grid <- merge(grid, cells[, .(final_dunsno, grp, bin_index, trade_value)], by = c("final_dunsno", "grp", "bin_index"), all.x = TRUE)
grid[is.na(trade_value), trade_value := 0]
grid[, `:=`(bin = bin_index, china = as.numeric(grp == "china"))]
grid <- merge(grid, cal[, .(bin = bin_index, mon, bs, be)], by = "bin")
lg(terc, "rows", nrow(grid), "firms", uniqueN(grid$final_dunsno))
M <- function(y, m) y * 12 + m - 1
lc <- function(b, V, wt) { nm <- names(wt); c(est = sum(wt * b[nm]), se = sqrt(drop(t(wt) %*% V[nm, nm, drop = FALSE] %*% wt))) }

# ---------- M1 ----------
m <- fepois(trade_value ~ i(bin, china, ref = 1) | final_dunsno^grp + final_dunsno^bin, data = grid, cluster = ~final_dunsno, nthreads = nth, notes = FALSE, warn = FALSE)
b0 <- coef(m); V0 <- vcov(m); bn <- as.integer(sub("^bin::([0-9]+):china$", "\\1", names(b0)))
BINS <- sort(c(1L, bn)); nm <- as.character(BINS)
b <- setNames(rep(0, length(BINS)), nm); b[as.character(bn)] <- b0
V <- matrix(0, length(BINS), length(BINS), dimnames = list(nm, nm)); V[as.character(bn), as.character(bn)] <- V0
fwrite(data.table(bin = BINS, est = b, se = sqrt(diag(V)), nobs = nobs(m), n_firms = uniqueN(grid[obs(m)]$final_dunsno)), f1)
nfirm <- uniqueN(grid[obs(m)]$final_dunsno)
pos <- function(k, yr) c(`2022` = 0L, `2023` = 26L, `2024` = 52L, `2025` = 79L)[as.character(yr)] + k   # bin of period k in year
res <- list()
add <- function(spec, label, wt) { r <- lc(b, V, wt); res[[length(res) + 1]] <<- data.table(tercile = terc, spec = spec, label = label, est = r["est"], se = r["se"], nobs = nobs(m), n_firms = nfirm) }
avgdiff <- function(ks, ya, yb) { wt <- setNames(rep(0, length(BINS)), nm); for (k in ks) { wt[as.character(pos(k, ya))] <- wt[as.character(pos(k, ya))] + 1 / length(ks); wt[as.character(pos(k, yb))] <- wt[as.character(pos(k, yb))] - 1 / length(ks) }; wt[wt != 0] }
for (cmp in list(c(2024, 2023), c(2023, 2022), c(2025, 2024))) {
  lab <- sprintf("%d_minus_%d", cmp[1], cmp[2])
  add("C1_same_calendar", paste0(lab, "_all"), avgdiff(1:26, cmp[1], cmp[2]))
  add("C1_same_calendar", paste0(lab, "_H1"), avgdiff(1:13, cmp[1], cmp[2]))
  add("C1_same_calendar", paste0(lab, "_H2"), avgdiff(14:26, cmp[1], cmp[2]))
}
for (k in 1:26) add("C1_same_calendar_by_period", sprintf("2024_minus_2023_k%02d", k), avgdiff(k, 2024, 2023))
for (k in 1:26) add("C1_real_by_period", sprintf("2025_minus_2024_k%02d", k), avgdiff(k, 2025, 2024))
# C2 trend adjusted: line over bins 27..78 (period index t = bin for bins <= 78, bin - 1 for bins > 79)
tt <- function(x) ifelse(x > 79, x - 1, x)
fit_b <- 27:78; tbar <- mean(tt(fit_b)); Sxx <- sum((tt(fit_b) - tbar)^2)
detr <- function(targets) {
  wt <- setNames(rep(0, length(BINS)), nm)
  for (t in targets) { wt[as.character(t)] <- wt[as.character(t)] + 1 / length(targets)
    h <- 1 / length(fit_b) + (tt(t) - tbar) * (tt(fit_b) - tbar) / Sxx
    wt[as.character(fit_b)] <- wt[as.character(fit_b)] - h / length(targets) }
  wt[abs(wt) > 1e-14]
}
for (t in c(54:78, 80:111)) { mt <- cal[bin_index == t]; add("C2_trend_adjusted_path", sprintf("%d|%s", t, mt$bs), detr(t)) }
add("C2_trend_adjusted_mean", "resolved_bins_93_111", detr(93:111))
add("C2_trend_adjusted_mean", "bins_80_91_(2025-01..2025-06)", detr(80:91))
add("C2_trend_adjusted_mean", "2024_in_sample_check_bins_53_78", detr(53:78))
add("C2_raw_mean", "resolved_bins_93_111_vs_2024_mean", { wt <- setNames(rep(0, length(BINS)), nm); wt[as.character(93:111)] <- 1 / 19; wt[as.character(53:78)] <- wt[as.character(53:78)] - 1 / 26; wt[wt != 0] })
fwrite(rbindlist(res), f2); lg("M1 saved")

# ---------- M2 ----------
GN <- c("china", "vietnam", "east_asia_other", "eu", "mexico", "canada")
run_m2 <- function(binsel, wins, tag) {
  d <- copy(grid[bin %in% binsel]); terms <- c()
  for (g in GN) for (k in names(wins)) { v <- sprintf("%s_%s", g, k); d[, (v) := as.numeric(grp == g & mon >= wins[[k]][1] & mon <= wins[[k]][2])]; if (sum(d[[v]]) > 0) terms <- c(terms, v) }
  mm <- fepois(as.formula(paste("trade_value ~", paste(terms, collapse = " + "), "| final_dunsno^grp + final_dunsno^bin")), data = d, cluster = ~final_dunsno, nthreads = nth, notes = FALSE, warn = FALSE)
  ct <- coeftable(mm); data.table(tercile = terc, design = tag, term = rownames(ct), est = ct[, 1], se = ct[, 2], nobs = nobs(mm), n_firms = uniqueN(d[obs(mm)]$final_dunsno))
}
realw <- list(W1 = c(M(2025, 1), M(2025, 4)), W2 = c(M(2025, 5), M(2025, 6)), W3 = c(M(2025, 7), M(2025, 10)), W4 = c(M(2025, 11), M(2026, 1)), W5 = c(M(2026, 2), M(2026, 3)))
plw <- list(W1 = c(M(2024, 1), M(2024, 4)), W2 = c(M(2024, 5), M(2024, 6)), W3 = c(M(2024, 7), M(2024, 10)), W4 = c(M(2024, 11), M(2024, 12)))
r2 <- rbind(run_m2(c(53:78, 80:111), realw, "real_ref2024"), run_m2(27:78, plw, "placebo_2024_ref2023"))
fwrite(r2, f3); lg("M2 saved")

# ---------- M3 ----------
vb <- cal$bin_index
cutbin <- function(dt) cal[bs <= dt & be >= dt, bin_index]
run_cut <- function(dt, nb, na, tag, cap = 111L) {
  cb <- cutbin(dt); if (!length(cb)) return(NULL); ci <- match(cb, vb)
  sel <- vb[max(1, ci - nb):min(length(vb), ci + na - 1)]; sel <- sel[sel <= cap]
  d <- grid[bin %in% sel]; d[, `:=`(post = as.numeric(bin >= cb), rel = (match(bin, vb) - ci) / 8)]
  d[, `:=`(cP = china * post, cR = china * rel)]
  out <- list()
  for (sp in c("post", "post+trend")) {
    rhs <- if (sp == "post") "cP" else "cP + cR"
    mm <- tryCatch(fepois(as.formula(paste("trade_value ~", rhs, "| final_dunsno^grp + final_dunsno^bin")), data = d, cluster = ~final_dunsno, nthreads = nth, notes = FALSE, warn = FALSE), error = function(e) NULL)
    if (is.null(mm)) next
    ct <- coeftable(mm)
    out[[sp]] <- data.table(tercile = terc, design = tag, spec = sp, cut = as.character(dt), cut_bin = cb, n_pre = sum(sel < cb), n_post = sum(sel >= cb), term = rownames(ct), est = ct[, 1], se = ct[, 2], nobs = nobs(mm), n_firms = uniqueN(d[obs(mm)]$final_dunsno))
  }
  rbindlist(out)
}
cuts <- as.Date(c("2025-05-14", "2025-11-10", "2026-02-24"))
r3 <- list()
for (dt in cuts) {
  for (win in list(c(8, 8), c(4, 4))) {
    lab <- sprintf("%d+%d", win[1], win[2])
    r3[[length(r3) + 1]] <- run_cut(dt, win[1], win[2], paste0("real_", lab))
    for (yrs in 1:2) { pdt <- dt - as.integer(round(365.25 * yrs)); if (yrs == 1 && dt == as.Date("2026-02-24")) next
      r3[[length(r3) + 1]] <- run_cut(pdt, win[1], win[2], sprintf("placebo_minus%dy_%s", yrs, lab), cap = 78L) }
  }
  lg("cut", as.character(dt), "done")
}
fwrite(rbindlist(r3), f4); lg("M3 saved")
