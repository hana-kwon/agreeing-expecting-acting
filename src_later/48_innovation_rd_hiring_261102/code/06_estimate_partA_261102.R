# ============================================================
# START DATE   : 2026-11-02
# CODE RUN DATE: 2026-11-02
# TIME TAKEN   : ~15 min
# DESCRIPTION  : Part A. Share of postings in R&D / engineering / product-development roles ~ China share x Post,
#                firm and SIC2 x quarter FE, clustered by firm; dynamic version; by employment bin. Logs every spec.
# INPUT FILES  : PRIVATE_CACHE (firm-level cache)
# OUTPUT FILES : analysis/48_.../spec_log.csv (appended), 06_partA_dynamic_coefs_261102.csv, 06_partA_pretrend_261102.csv
# ============================================================
suppressMessages(library(fixest)); suppressMessages(library(data.table))
C <- path.expand("PRIVATE_CACHE")
A <- normalizePath("../../analysis/48_innovation_rd_hiring_261102/", mustWork = TRUE)
if (!file.exists(file.path(A, "spec_log.csv")) ) {}
outs <- c("06_partA_dynamic_coefs_261102.csv", "06_partA_pretrend_261102.csv", "06_partA_main_261102.csv")
for (f in outs) if (file.exists(file.path(A, f))) stop(paste("refuse to overwrite", f))
d <- fread(paste0(C, "panel_duns_quarter.csv"))
d[, RD := ENG + SCI + PRODDEV]
for (v in c("RD", "ENG", "SCI", "PRODDEV", "SOFT", "PROC", "SALES")) d[, paste0("s_", v) := 100 * get(v) / N]
d[, s_RDSOFT := s_RD + s_SOFT]; d[, a_RD := asinh(RD)]
d[, firm := final_dunsno]; d[, qn := q]
d[, post := as.integer(q >= "2025Q2")]
d <- d[!is.na(china_share_2024) & !is.na(sic2) & N >= 1]
d[, cs := china_share_2024]; d[, cs_alt := china_share_pre_2022_2024]
sdcs <- sd(d[!duplicated(firm)]$cs)
cat("firms", uniqueN(d$firm), "firm-quarters", nrow(d), "sd(cs)", sdcs, "\n")
logf <- file.path(A, "spec_log.csv")
logrow <- function(part, id, outcome, spec, sample, m, coefname) {
  ct <- coeftable(m); i <- grep(coefname, rownames(ct))[1]
  r <- data.table(part = part, id = id, outcome = outcome, spec = spec, sample = sample, n_obs = nobs(m),
    n_units = m$fixef_sizes[["firm"]], coef = ct[i, 1], se = ct[i, 2], p = ct[i, 4])
  fwrite(r, logf, append = file.exists(logf)); r
}
run <- function(dt, y, tag, sample, wt = NULL, expo = "cs") {
  f <- as.formula(paste0(y, " ~ ", expo, ":post | firm + sic2^qn"))
  m <- if (is.null(wt)) feols(f, dt, cluster = ~firm, notes = FALSE) else feols(f, dt, weights = dt[[wt]], cluster = ~firm, notes = FALSE)
  logrow("A", paste0("A_", tag, "_", y), y, "static", sample, m, "post")
}
res <- list()
for (y in c("s_RD", "s_ENG", "s_SCI", "s_PRODDEV", "s_SOFT", "s_RDSOFT", "s_PROC", "s_SALES", "a_RD"))
  res[[y]] <- run(d, y, "main", "N>=1")
d5 <- d[N >= 5]
for (y in c("s_RD", "s_PROC", "s_SALES")) { res[[paste0(y, "5")]] <- run(d5, y, "n5", "N>=5") }
res[["w"]] <- run(d, "s_RD", "wt", "N>=1, weights=N", wt = "N")
res[["strict"]] <- run(d[strict_link == TRUE], "s_RD", "strict", "strict-name-state links only")
res[["alt"]] <- run(d[!is.na(cs_alt)], "s_RD", "alt_expo", "N>=1, exposure=2022-2024 pooled", expo = "cs_alt")
# employment bins
for (b in c("1-19", "20-49", "50-249", "250+")) {
  db <- d[emp_bin == b]
  if (nrow(db) > 200) for (y in c("s_RD", "s_PROC")) res[[paste(y, b)]] <- run(db, y, paste0("bin", b), paste0("emp ", b))
}
fwrite(rbindlist(res), file.path(A, "06_partA_main_261102.csv"))
# dynamic
dyn <- list(); pre <- list()
for (y in c("s_RD", "s_SOFT", "s_PROC", "s_SALES")) {
  for (grp in c("all", "emp<50", "emp50+")) {
    dd <- if (grp == "all") d else if (grp == "emp<50") d[emp_bin %in% c("1-19", "20-49")] else d[emp_bin %in% c("50-249", "250+")]
    if (nrow(dd) < 500) next
    m <- feols(as.formula(paste0(y, " ~ i(qn, cs, ref = '2025Q1') | firm + sic2^qn")), dd, cluster = ~firm, notes = FALSE)
    ct <- as.data.table(coeftable(m), keep.rownames = "term"); ct[, `:=`(outcome = y, group = grp)]; dyn[[paste(y, grp)]] <- ct
    w <- tryCatch(wald(m, "qn::202[34]", print = FALSE), error = function(e) NULL)
    pt <- data.table(outcome = y, group = grp, n_obs = nobs(m), n_firms = m$fixef_sizes[["firm"]], pre_wald_p = if (is.null(w)) NA else w$p, pre_wald_stat = if (is.null(w)) NA else w$stat)
    pre[[paste(y, grp)]] <- pt
    fwrite(data.table(part = "A", id = paste0("A_dyn_", y, "_", grp), outcome = y, spec = "dynamic pre-trend joint Wald p", sample = grp,
                      n_obs = nobs(m), n_units = m$fixef_sizes[["firm"]], coef = pt$pre_wald_stat, se = NA, p = pt$pre_wald_p), logf, append = TRUE)
  }
}
fwrite(rbindlist(dyn, fill = TRUE), file.path(A, "06_partA_dynamic_coefs_261102.csv"))
fwrite(rbindlist(pre), file.path(A, "06_partA_pretrend_261102.csv"))
print(rbindlist(res)[, .(id, n_obs, n_units, coef = round(coef, 3), se = round(se, 3), p = round(p, 3))])
print(rbindlist(pre))
