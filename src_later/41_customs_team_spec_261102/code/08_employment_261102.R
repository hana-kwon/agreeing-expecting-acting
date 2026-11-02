# ============================================================
# START DATE   : 2026-11-02
# CODE RUN DATE: 2026-11-02
# TIME TAKEN   : ~10 min
# DESCRIPTION  : Employment size instead of trade terciles. 2024 D&B EMPLOYEESALLSITES in bins 1-19, 20-49, 50-249, 250+.
#                (a) main signal model separately by bin; (b) pooled model with 250+ as the omitted bin and origin-group x period FE,
#                so coefficients are differences from 250+ firms. Firm subsample cap per bin for compute (seeded).
# INPUT FILES  : PRIVATE_CACHE{cells,firms,firm_attrs}.parquet, analysis/41_.../01_period_series_261102.csv
# OUTPUT FILES : analysis/41_customs_team_spec_261102/08_coefs_employment_261102.csv
# ============================================================
source(file.path(dirname(gsub("~+~", " ", sub("--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE)[1]), fixed = TRUE)), "00_common_261102.R"))
outf <- file.path(OUT, "08_coefs_employment_261102.csv"); if (file.exists(outf)) stop("output exists; refusing to overwrite: ", outf)
CAP <- 6000L; set.seed(261102)
cells <- as.data.table(read_parquet(file.path(CACHE, "cells.parquet")))
fa <- as.data.table(read_parquet(file.path(CACHE, "firm_attrs.parquet")))[!is.na(emp_bin), .(final_dunsno, emp_bin, tercile)]
fa <- fa[final_dunsno %in% unique(cells$final_dunsno)]
cnt <- fa[, .N, by = emp_bin]; cat("firms with positive cells by bin (before cap):\n"); print(cnt)
fa <- fa[, .SD[sample(.N, min(.N, CAP))], by = emp_bin]
per <- fread(file.path(OUT, "01_period_series_261102.csv"))
cells <- cells[final_dunsno %in% fa$final_dunsno]
fb <- unique(cells[, .(final_dunsno, bin_index)])
grid <- fb[, .(grp = GR), by = .(final_dunsno, bin_index)]
grid <- merge(grid, cells, by = c("final_dunsno", "grp", "bin_index"), all.x = TRUE)
for (y in c("trade_value", "weightkg", "volumeteu")) set(grid, which(is.na(grid[[y]])), y, 0)
grid <- merge(grid, fa[, .(final_dunsno, emp_bin)], by = "final_dunsno")
grid <- merge(grid, per[, .(bin_index, duty10, sig_china)], by = "bin_index")
grid[, `:=`(bin = bin_index, china = as.numeric(grp == "china"), post_lib = as.numeric(bin_index >= 87))]
grid[, `:=`(sigC = china * sig_china, dutyC = china * duty10, libC = china * post_lib)]
res <- list(); lg <- function(...) cat(format(Sys.time(), "%H:%M:%S"), ..., "\n")
for (b in c("e1_19", "e20_49", "e50_249", "e250p")) {
  d <- grid[emp_bin == b]; lg("bin", b, "rows", nrow(d), "firms", uniqueN(d$final_dunsno))
  m <- fepois(trade_value ~ sigC + dutyC | final_dunsno^grp + final_dunsno^bin, data = d, cluster = ~final_dunsno, nthreads = 1, notes = FALSE, warn = FALSE)
  res[[length(res) + 1]] <- tidy_fit(m, d, "E1_signal_by_employment_bin", "trade_value", b)
  d <- d[bin != 86]
  m <- fepois(trade_value ~ libC | final_dunsno^grp + final_dunsno^bin, data = d, cluster = ~final_dunsno, nthreads = 1, notes = FALSE, warn = FALSE)
  res[[length(res) + 1]] <- tidy_fit(m, d, "E2_lib_by_employment_bin", "trade_value", b)
  fwrite(rbindlist(res), file.path(CACHE, "partial08.csv")); lg("done bin", b)
}
# pooled: differences from the omitted 250+ bin; China x period average absorbed by origin-group x period FE
for (b in c("e1_19", "e20_49", "e50_249")) grid[, (paste0("sigC_", b)) := sigC * (emp_bin == b)]
for (b in c("e1_19", "e20_49", "e50_249")) grid[, (paste0("libC_", b)) := libC * (emp_bin == b)]
for (b in c("e1_19", "e20_49", "e50_249")) grid[, (paste0("dutyC_", b)) := dutyC * (emp_bin == b)]
lg("pooled signal"); m <- fepois(trade_value ~ sigC_e1_19 + sigC_e20_49 + sigC_e50_249 + dutyC_e1_19 + dutyC_e20_49 + dutyC_e50_249 |
                                  final_dunsno^grp + final_dunsno^bin + grp^bin, data = grid, cluster = ~final_dunsno, nthreads = 1, notes = FALSE, warn = FALSE)
res[[length(res) + 1]] <- tidy_fit(m, grid, "E3_pooled_signal_diff_from_250plus_grpbinFE", "trade_value", "pooled")
d <- grid[bin != 86]; lg("pooled lib")
m <- fepois(trade_value ~ libC_e1_19 + libC_e20_49 + libC_e50_249 | final_dunsno^grp + final_dunsno^bin + grp^bin, data = d, cluster = ~final_dunsno, nthreads = 1, notes = FALSE, warn = FALSE)
res[[length(res) + 1]] <- tidy_fit(m, d, "E4_pooled_lib_diff_from_250plus_grpbinFE", "trade_value", "pooled")
fwrite(rbindlist(res), outf); lg("saved", outf)
