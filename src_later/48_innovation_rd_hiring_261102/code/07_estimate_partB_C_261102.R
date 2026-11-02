# ============================================================
# START DATE   : 2026-11-02
# CODE RUN DATE: 2026-11-02
# TIME TAKEN   : ~10 min
# DESCRIPTION  : Part B: R&D intensity (xrdq/saleq) and log xrdq of listed importers ~ China share x Post (static, dynamic).
#                Part C (added after start): SEC-stated tariff expectation (Hexp, Last at filing quarter t) -> R&D intensity
#                and R&D/engineering posting share at t+1, firm FE and sector x quarter FE.
# INPUT FILES  : cache: fundq_xrdq_wrds.csv (WRDS comp.fundq; the volume's fundq lacks xrdq), gvkey_china_share.parquet (built from
#                .../compustat_parent_imports/{importer_parent_link,importer_hs6_origin_2024}), ticker_gvkey.csv, gvkey_rcid.csv,
#                rcid_month_function_counts.parquet (as csv: rcid_quarter_rd.csv);
#                analysis/35_sec_tariff_expectations_261102/02_panel_261102.csv
# OUTPUT FILES : analysis/48_.../spec_log.csv (appended), 07_* csv files
# ============================================================
suppressMessages(library(fixest)); suppressMessages(library(data.table)); suppressMessages(library(arrow))
C <- path.expand("PRIVATE_CACHE")
A <- normalizePath("../../analysis/48_innovation_rd_hiring_261102/", mustWork = TRUE)
S <- normalizePath("../../analysis/35_sec_tariff_expectations_261102/02_panel_261102.csv", mustWork = TRUE)
outs <- c("07_partB_dynamic_coefs_261102.csv", "07_partB_reporting_counts_261102.csv", "07_partC_261102.csv")
for (f in outs) if (file.exists(file.path(A, f))) stop(paste("refuse to overwrite", f))
logf <- file.path(A, "spec_log.csv")
nfirm <- function(m) { u <- tryCatch(m$fixef_sizes[["firm"]], error = function(e) NULL); if (!is.null(u)) return(u)
  dt <- eval(m$call$data, parent.frame(3)); ob <- m$obs_selection$obsRemoved; idx <- if (is.null(ob)) seq_len(nrow(dt)) else seq_len(nrow(dt))[ob]; uniqueN(dt$firm[idx]) }
logrow <- function(part, id, outcome, spec, sample, m, coefname) {
  ct <- coeftable(m); i <- grep(coefname, rownames(ct), fixed = TRUE)[1]
  r <- data.table(part = part, id = id, outcome = outcome, spec = spec, sample = sample, n_obs = nobs(m),
    n_units = nfirm(m), coef = ct[i, 1], se = ct[i, 2], p = ct[i, 4])
  fwrite(r, logf, append = file.exists(logf)); r
}
# ---- Part B data
q <- fread(paste0(C, "fundq_xrdq_wrds.csv"), colClasses = list(character = "gvkey"))
cs <- as.data.table(read_parquet(paste0(C, "gvkey_china_share.parquet"))); cs[, gvkey := sprintf("%06s", gvkey)]; cs[, gvkey := gsub(" ", "0", gvkey)]
pi <- as.data.table(read_parquet("PRIVATE_SERVER_PATH", col_select = c("gvkey", "sic", "in_us")))
pi[, sic2 := floor(as.numeric(sic) / 100)]
q <- merge(q, cs[, .(gvkey, cs = china_share)], by = "gvkey"); q <- merge(q, unique(pi[, .(gvkey, sic2, in_us)]), by = "gvkey")
q <- q[curcdq == "USD" & datacqtr >= "2023Q1" & datacqtr <= "2025Q4" & !is.na(sic2) & !is.na(cs)]
q <- q[!duplicated(q[, .(gvkey, datacqtr)])]
q[, firm := gvkey]; q[, qn := datacqtr]; q[, post := as.integer(qn >= "2025Q2")]
q[, rdint := ifelse(saleq > 0 & !is.na(xrdq), 100 * xrdq / saleq, NA_real_)]
q[, lxrd := ifelse(!is.na(xrdq) & xrdq > 0, log(xrdq), NA_real_)]
w <- function(x) { l <- quantile(x, c(.01, .99), na.rm = TRUE); pmin(pmax(x, l[1]), l[2]) }
q[, rdint_w := w(rdint)]
rep <- q[, .(n_q_xrdq = sum(!is.na(xrdq)), n_q_pos = sum(!is.na(xrdq) & xrdq > 0)), by = gvkey]
cnt <- data.table(item = c("importer-linked US-dollar firm-quarters 2023Q1-2025Q4", "gvkeys in that sample", "gvkeys with any non-missing xrdq", "gvkeys with xrdq > 0 in any quarter",
  "gvkeys with xrdq in all 12 quarters", "firm-quarters with xrdq non-missing", "firm-quarters with xrdq > 0 and saleq > 0"),
  value = c(nrow(q), uniqueN(q$gvkey), sum(rep$n_q_xrdq > 0), sum(rep$n_q_pos > 0), sum(rep$n_q_xrdq == 12), sum(!is.na(q$xrdq)), sum(!is.na(q$rdint) & q$xrdq > 0)))
fwrite(cnt, file.path(A, "07_partB_reporting_counts_261102.csv")); print(cnt)
d <- q[!is.na(rdint)]
for (y in c("rdint_w", "rdint", "lxrd")) {
  dd <- if (y == "lxrd") q[!is.na(lxrd)] else d
  m <- feols(as.formula(paste0(y, " ~ cs:post | firm + sic2^qn")), dd, cluster = ~firm, notes = FALSE)
  print(logrow("B", paste0("B_static_", y), y, "static", "listed importers, xrdq reported", m, "cs:post")[, .(id, n_obs, n_units, coef, se, p)])
  m2 <- feols(as.formula(paste0(y, " ~ i(qn, cs, ref = '2025Q1') | firm + sic2^qn")), dd, cluster = ~firm, notes = FALSE)
  ct <- as.data.table(coeftable(m2), keep.rownames = "term"); ct[, outcome := y]
  fwrite(ct, file.path(A, "07_partB_dynamic_coefs_261102.csv"), append = file.exists(file.path(A, "07_partB_dynamic_coefs_261102.csv")))
  wd <- wald(m2, "qn::202[34]", print = FALSE)
  fwrite(data.table(part = "B", id = paste0("B_dyn_", y), outcome = y, spec = "dynamic pre-trend joint Wald stat/p", sample = "listed importers",
    n_obs = nobs(m2), n_units = m2$fixef_sizes[["firm"]], coef = wd$stat, se = NA, p = wd$p), logf, append = TRUE)
}
# balanced-in-reporting robustness: firms with xrdq in all 12 quarters
bal <- rep[n_q_xrdq == 12, gvkey]
m <- feols(rdint_w ~ cs:post | firm + sic2^qn, d[gvkey %in% bal], cluster = ~firm, notes = FALSE)
print(logrow("B", "B_static_rdint_w_balanced", "rdint_w", "static", "xrdq reported in all 12 quarters", m, "cs:post")[, .(n_obs, n_units, coef, se, p)])
#    (added after seeing that the winsorized intensity has very large SEs): firm-quarters with quarterly sales >= 10 million USD
m <- feols(rdint_w ~ cs:post | firm + sic2^qn, d[saleq >= 10], cluster = ~firm, notes = FALSE)
print(logrow("B", "B_static_rdint_w_sales10m_POSTHOC", "rdint_w", "static (  )", "saleq >= 10 (USD millions)", m, "cs:post")[, .(n_obs, n_units, coef, se, p)])
m <- feols(rdint_w ~ i(qn, cs, ref = "2025Q1") | firm + sic2^qn, d[saleq >= 10], cluster = ~firm, notes = FALSE)
ct <- as.data.table(coeftable(m), keep.rownames = "term"); ct[, outcome := "rdint_w_sales10m_POSTHOC"]
fwrite(ct, file.path(A, "07_partB_dynamic_coefs_261102.csv"), append = TRUE)
# ---- Part C: SEC expectations
sec <- fread(S)
tg <- fread(paste0(C, "ticker_gvkey.csv"), colClasses = list(character = c("gvkey", "cik")))
sec[, cik10 := sprintf("%010d", as.integer(cik))]
sec <- merge(sec, unique(tg[, .(cik10 = cik, gvkey)]), by = "cik10")
sec[, nextq := paste0(as.integer(substr(quarter, 1, 4)) + (as.integer(substr(quarter, 6, 6)) == 4), "Q", (as.integer(substr(quarter, 6, 6)) %% 4) + 1)]
sec <- sec[, .(Hexp = max(Hexp), Last = max(Last), n_all = max(n_all), Expo = max(Expo)), by = .(gvkey, sector, quarter, nextq)]
cat("SEC firm-quarters matched to gvkey:", nrow(sec), "firms", uniqueN(sec$gvkey), "\n")
qa <- fread(paste0(C, "fundq_xrdq_wrds.csv"), colClasses = list(character = "gvkey"))[curcdq == "USD"]
qa[, rdint := ifelse(saleq > 0 & !is.na(xrdq), 100 * xrdq / saleq, NA_real_)]; qa[, rdint_w := w(rdint)]
qa <- qa[!duplicated(qa[, .(gvkey, datacqtr)]), .(gvkey, nextq = datacqtr, rdint, rdint_w, xrdq)]
x <- merge(sec, qa, by = c("gvkey", "nextq"))
x <- merge(x, cs[, .(gvkey, cs = china_share)], by = "gvkey", all.x = TRUE)
x[, `:=`(firm = gvkey, ln_n = log1p(n_all), sq = paste(sector, quarter))]
cres <- list()
xx <- x[!is.na(rdint_w)]
cat("C Compustat: obs", nrow(xx), "firms", uniqueN(xx$gvkey), "firms with Hexp=1 at least once", uniqueN(xx[Hexp == 1]$gvkey), "\n")
for (fe in c("sector^quarter", "firm + sector^quarter")) {
  m <- feols(as.formula(paste0("rdint_w ~ Hexp + Last + ln_n + Expo | ", fe)), xx, cluster = ~firm, notes = FALSE)
  for (cn in c("Hexp", "Last")) cres[[paste(fe, cn)]] <- logrow("C", paste0("C_rdint_", cn, "_", gsub("[^a-z]", "", fe)), "rdint_w(t+1)", paste0("FE: ", fe), "SEC firms matched to Compustat", m, cn)
}
xc <- xx[!is.na(cs)]
m <- feols(rdint_w ~ Hexp + Hexp:cs + Last + ln_n + Expo | firm + sector^quarter, xc, cluster = ~firm, notes = FALSE)
cres[["int"]] <- logrow("C", "C_rdint_HexpXcs", "rdint_w(t+1)", "Hexp x cs (cs absorbed by firm FE)", "matched, exposure available", m, "Hexp:cs")
# ---- Part C postings branch (bridge-extracted rcids only)
rp <- file.path(C, "rcid_quarter_rd.csv")
if (file.exists(rp)) {
  rc <- fread(paste0(C, "gvkey_rcid.csv"), colClasses = list(character = "gvkey")); rq <- fread(rp)
  rq <- merge(rq, unique(rc[, .(rcid, gvkey)]), by = "rcid")
  rq <- rq[, .(RD = sum(RD), N = sum(N)), by = .(gvkey, nextq = q)][N >= 1][, s_RD := 100 * RD / N]
  y <- merge(sec, rq, by = c("gvkey", "nextq")); y[, firm := gvkey]; y[, ln_n := log1p(n_all)]
  cat("C postings: obs", nrow(y), "firms", uniqueN(y$gvkey), "\n")
  if (nrow(y) > 30) for (fe in c("sector^quarter", "firm + sector^quarter")) {
    m <- feols(as.formula(paste0("s_RD ~ Hexp + Last + ln_n + Expo | ", fe)), y, cluster = ~firm, notes = FALSE)
    for (cn in c("Hexp", "Last")) cres[[paste("p", fe, cn)]] <- logrow("C", paste0("C_postRD_", cn, "_", gsub("[^a-z]", "", fe)), "RD posting share(t+1)", paste0("FE: ", fe), "SEC firms with Revelio rcid in bridge extract", m, cn)
  }
}
fwrite(rbindlist(cres), file.path(A, "07_partC_261102.csv")); print(rbindlist(cres)[, .(id, n_obs, n_units, coef, se, p)])
