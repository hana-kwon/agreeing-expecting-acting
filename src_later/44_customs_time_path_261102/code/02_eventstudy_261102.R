# ============================================================
# START DATE   : 2026-11-02
# CODE RUN DATE: 2026-11-02
# TIME TAKEN   : ~10-30 min per tercile
# DESCRIPTION  : P1. China x two-week-period event study (saturated: China x every bin except bin 53), PPML on trade value,
#                firm^origin_group + firm^period FE, one size tercile per run. V1 = relative to the mean of 2024 Q1 (bins 53-58);
#                V2 = relative to the same year_biweek number in 2024 (2025 and 2026 periods). Usage: Rscript 02_eventstudy_261102.R <tercile> <firm_fraction>
# INPUT FILES  : PRIVATE_CACHE{cells,firms}.parquet, analysis/41_.../01_period_series_261102.csv,
#                analysis/44_.../01_period_table_261102.csv
# OUTPUT FILES : analysis/44_customs_time_path_261102/02_eventstudy_<tercile>_261102.csv
# ============================================================
source(file.path(dirname(gsub("~+~", " ", sub("--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE)[1]), fixed = TRUE)), "00_common_261102.R"))
args <- commandArgs(TRUE); terc <- args[1]; frac <- as.numeric(args[2]); nth <- if (length(args) >= 3) as.integer(args[3]) else 2L
outf <- file.path(OUT44, sprintf("02_eventstudy_%s_261102.csv", terc))
if (file.exists(outf)) stop("output exists; refusing to overwrite: ", outf)
dat <- build_grid(terc, frac)
lg(terc, "firm fraction", frac, "rows", nrow(dat), "firms", uniqueN(dat$final_dunsno))
# ★ CORE LOGIC: saturated model, one omitted bin (53). V1 and V2 are linear combinations of these coefficients, so the reference
# set (2024 Q1 or the same 2024 period) does not constrain the 2024 Q1 periods to share one China effect.
m <- fepois(trade_value ~ i(bin, china, ref = 53) | final_dunsno^grp + final_dunsno^bin, data = dat, cluster = ~final_dunsno,
            nthreads = nth, notes = FALSE, warn = FALSE)
lg("estimated; nobs", nobs(m))
b <- coef(m); V <- vcov(m)
bn <- as.integer(sub("^bin::([0-9]+):china$", "\\1", names(b))); stopifnot(!anyNA(bn))
BINS <- sort(c(53L, bn)); nmf <- as.character(BINS)
bf <- setNames(rep(0, length(BINS)), nmf); bf[as.character(bn)] <- b
Vf <- matrix(0, length(BINS), length(BINS), dimnames = list(nmf, nmf)); Vf[as.character(bn), as.character(bn)] <- V
used <- dat[obs(m)]
meta <- PT[bin_index %in% BINS, .(bin = bin_index, year_biweek, bin_start, bin_end, year, poy)]
rows <- list()
add <- function(spec, t, w, ref_bin = NA_integer_) {
  r <- lincomb(bf, Vf, w); mt <- meta[bin == t]
  rows[[length(rows) + 1]] <<- data.table(spec = spec, tercile = terc, bin = t, year_biweek = mt$year_biweek, bin_start = mt$bin_start,
    bin_end = mt$bin_end, est = r["est"], se = r["se_firm"], lo = r["est"] - 1.96 * r["se_firm"], hi = r["est"] + 1.96 * r["se_firm"],
    ref_bin = ref_bin, nobs = nobs(m), n_firms = uniqueN(used$final_dunsno))
}
e <- function(t) { w <- setNames(1, as.character(t)); w }
for (t in BINS) {
  w1 <- setNames(rep(0, length(BINS)), nmf); w1[as.character(t)] <- w1[as.character(t)] + 1; w1[as.character(53:58)] <- w1[as.character(53:58)] - 1 / 6
  add("V1_ref_2024Q1", t, w1[w1 != 0])
  add("S_rel_bin53", t, e(t))
  mt <- meta[bin == t]
  if (mt$year >= 2025) {
    rb <- 52L + mt$poy; w2 <- setNames(c(1, -1), as.character(c(t, rb))); add("V2_same_period_2024", t, w2, rb)
  }
}
res <- rbindlist(rows); res[, `:=`(ref_start = meta$bin_start[match(ref_bin, meta$bin)])]
safe_write44(res, outf); lg("saved", outf)
