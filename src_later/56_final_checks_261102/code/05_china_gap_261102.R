# ============================================================
# START DATE   : 2026-11-02
# CODE RUN DATE: 2026-11-02
# TIME TAKEN   : ~5-20 min per size group
# DESCRIPTION  : Same saturated model as 54 C: trade = exp(sum b_bin China x bin + firm^origin + firm^bin), firm-clustered. For each tariff cut:
#                recovery = mean b(2-9 bins after the cut bin, stopping before the next cut) - mean b(4 bins before); net recovery subtracts the same-calendar 2024 change;
#                gap = mean b(after) - mean b(same calendar bins 2024); trend gap removes the 2023-2024 straight line. Usage: Rscript 05_china_gap_261102.R <tercile>
# INPUT FILES  : PRIVATE_CACHE, analysis/54_review_fixes_261102/03_bin_calendar_261102.csv
# OUTPUT FILES : analysis/56_final_checks_261102/05_china_gap_<tercile>_261102.csv
# ============================================================
suppressMessages({library(arrow); library(data.table); library(fixest)})
this_file <- gsub("~+~", " ", sub("--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE)[1]), fixed = TRUE)
TOPIC <- normalizePath(file.path(dirname(this_file), "..", "..")); terc <- commandArgs(TRUE)[1]
f1 <- file.path(TOPIC, sprintf("analysis/56_final_checks_261102/05_china_gap_%s_261102.csv", terc)); if (file.exists(f1)) stop("refuse to overwrite")
cal <- fread(file.path(TOPIC, "analysis/54_review_fixes_261102/03_bin_calendar_261102.csv")); cal[, `:=`(bs = as.Date(bin_start), be = as.Date(bin_end))]
cells <- as.data.table(read_parquet(file.path(Sys.getenv("HOME"), "private_cache/review54_261102/cells7.parquet")))[tercile == terc]
GR <- c("china", "vietnam", "east_asia_other", "eu", "mexico", "canada", "rest")
fb <- unique(cells[, .(final_dunsno, bin_index)]); grid <- fb[, .(grp = GR), by = .(final_dunsno, bin_index)]
grid <- merge(grid, cells[, .(final_dunsno, grp, bin_index, trade_value)], by = c("final_dunsno", "grp", "bin_index"), all.x = TRUE); grid[is.na(trade_value), trade_value := 0]
grid[, `:=`(bin = bin_index, china = as.numeric(grp == "china"))]
m <- fepois(trade_value ~ i(bin, china, ref = 1) | final_dunsno^grp + final_dunsno^bin, data = grid, cluster = ~final_dunsno, nthreads = 2L, notes = FALSE, warn = FALSE)
b0 <- coef(m); V0 <- vcov(m); bn <- as.integer(sub("^bin::([0-9]+):china$", "\\1", names(b0))); BINS <- sort(c(1L, bn)); nm <- as.character(BINS)
b <- setNames(rep(0, length(BINS)), nm); b[as.character(bn)] <- b0; V <- matrix(0, length(BINS), length(BINS), dimnames = list(nm, nm)); V[as.character(bn), as.character(bn)] <- V0
same24 <- function(x) ifelse(x > 105, x - 105, x - 79) + 52L     # ★ IMPORTANT: 2025 = 79 + k, 2026 = 105 + k, 2024 = 52 + k, as in 54 C
tt <- function(x) ifelse(x > 79, x - 1, x); fit_b <- 27:78; tbar <- mean(tt(fit_b)); Sxx <- sum((tt(fit_b) - tbar)^2)
W <- function(pos, neg) { w <- setNames(rep(0, length(BINS)), nm); for (x in pos) w[as.character(x)] <- w[as.character(x)] + 1 / length(pos); for (x in neg) w[as.character(x)] <- w[as.character(x)] - 1 / length(neg); w }
detr <- function(ts) { w <- setNames(rep(0, length(BINS)), nm); for (t in ts) { w[as.character(t)] <- w[as.character(t)] + 1 / length(ts); h <- 1 / length(fit_b) + (tt(t) - tbar) * (tt(fit_b) - tbar) / Sxx; w[as.character(fit_b)] <- w[as.character(fit_b)] - h / length(ts) }; w }
lc <- function(w) c(sum(w * b), sqrt(drop(t(w) %*% V %*% w)))
cb <- function(d) cal[bs <= as.Date(d) & be >= as.Date(d), bin_index]
cuts <- c("2025-05-14", "2025-11-10", "2026-02-24"); cbs <- sapply(cuts, cb); nxt <- c(cbs[-1], 112L); rows <- list()
for (i in seq_along(cuts)) {
  c0 <- cbs[i]; after <- (c0 + 2):min(c0 + 9, nxt[i] - 1); pre <- (c0 - 4):(c0 - 1); pre <- pre[pre != 79]
  add <- function(lab, w) { r <- lc(w); rows[[length(rows) + 1]] <<- data.table(tercile = terc, cut = cuts[i], contrast = lab, est = r[1], se = r[2], n_after_bins = length(after), n_before_bins = length(pre), first_after = cal[bin_index == min(after), bs], last_after = cal[bin_index == max(after), be], n_firms = uniqueN(grid[obs(m)]$final_dunsno)) }
  rec <- W(after, pre); add("recovery_from_before_cut", rec); add("same_calendar_2024_change", W(same24(after), same24(pre))); add("net_recovery", rec - W(same24(after), same24(pre)))
  add("gap_from_same_calendar_2024", W(after, same24(after))); add("gap_after_straight_trend", detr(after))
}
r <- rbindlist(rows); fwrite(r, f1); print(r[, .(tercile, cut, contrast, round(est, 3), round(se, 3), n_after_bins)])
