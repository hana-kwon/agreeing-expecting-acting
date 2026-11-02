# ============================================================
# START DATE   : 2026-11-02
# CODE RUN DATE: 2026-11-02
# TIME TAKEN   : ~15 min run
# DESCRIPTION  : RERUN of the quarterly dynamic/pre-trend part only (47, 48A) after fixing the loop-variable bug in dyn(). Two-exposure (input tariff shock IN, output protection OUT, IN x OUT) difference-in-differences for four innovation
#                outcome families; manufacturers vs wholesale/retail; employment bins; placebo/dynamic pre-trend tests; MDEs.
# INPUT FILES  : PRIVATE_CACHE, firm_cov_261102.parquet; innov47_261102/panel_firm_quarter_final.parquet;
#                innov48_261102/panel_duns_quarter.csv, fundq_xrdq_wrds.csv; innov52_261102/firm_exposure.parquet, gvkey_exposure.parquet
# OUTPUT FILES : analysis/52_innovation_two_exposures_261102/04_unused.csv, 03_dynamic.csv, 03_log.txt (aggregates only)
# ============================================================
suppressMessages({library(data.table); library(fixest); library(arrow)})
setFixest_notes(FALSE)
args <- commandArgs(trailingOnly = TRUE)
ROOT <- args[1]  # project root passed explicitly (Rscript mangles spaces in --file)
ANA <- file.path(ROOT, "analysis/52_innovation_two_exposures_261102")
CA <- path.expand("PRIVATE_CACHE")
fe_out <- file.path(ANA, "04_unused_never_written.csv"); dyn_out <- file.path(ANA, "04_dynamic_quarterly.csv"); lg <- file.path(ANA, "04_log.txt")
for (f in c(dyn_out, lg)) if (file.exists(f)) stop("refusing to overwrite ", f)  # never overwrite
sink(lg, split = TRUE)
EST <- list(); DYN <- list()

fx <- as.data.table(read_parquet(paste0(CA, "innov52_261102/firm_exposure.parquet")))
gx <- as.data.table(read_parquet(paste0(CA, "innov52_261102/gvkey_exposure.parquet")))
grp <- function(s4) fifelse(s4 %/% 100 >= 20 & s4 %/% 100 <= 39, "mfg", fifelse(s4 %/% 100 >= 50 & s4 %/% 100 <= 59, "wr", NA_character_))
fx[, grp := grp(sic4)]; gx[, grp := grp(sic4)]
ebin <- function(e) fifelse(is.na(e), "NA", fifelse(e < 20, "1-19", fifelse(e < 50, "20-49", fifelse(e < 250, "50-249", "250+"))))
fx[, ebin := ebin(emp)]

# ★ CORE LOGIC: standardize IN, OUT, O301 within each estimation sample so coefficients are per +1 SD, and the interaction is
# product of z-scores; net effect for a firm +1SD in both = b_IN + b_OUT + b_INxOUT.
prep <- function(d, g, unit) {
  d <- d[grp == g]
  if (g == "mfg") d <- d[!is.na(OUT)]
  d <- d[!is.na(IN) & !is.na(sic4)]
  u <- unique(d[, c(unit, "IN", "OUT", "O301"), with = FALSE])
  z <- function(x) (x - mean(x, na.rm = TRUE)) / sd(x, na.rm = TRUE)
  m <- list(IN = mean(u$IN), sIN = sd(u$IN), OUT = mean(u$OUT), sOUT = sd(u$OUT), O301 = mean(u$O301), sO301 = sd(u$O301))
  d[, `:=`(zIN = (IN - m$IN) / m$sIN, zO301 = (O301 - m$O301) / m$sO301)]
  if (g == "mfg") d[, zOUT := (OUT - m$OUT) / m$sOUT]
  d[, sic1 := sic4 %/% 1000]; d[, sic2 := sic4 %/% 100]
  attr(d, "m") <- m
  d
}
fit <- function(d, y, g, ds, sample, fe = "A", ppml = FALSE, post, per, unit, spec = "both") {
  d <- copy(d); set(d, j = "p", value = as.numeric(d[[post]]))
  d[, pIN := zIN * p]; d[, pO301 := zO301 * p]
  rhs <- "pIN + pO301"
  if (g == "mfg") { d[, pOUT := zOUT * p]; d[, pX := zIN * zOUT * p]
    rhs <- if (spec == "in_only") "pIN + pO301" else "pIN + pOUT + pX + pO301" }
  fes <- if (fe == "A") paste0(unit, " + sic1^", per) else paste0(unit, " + sic2^", per)
  f <- as.formula(paste0(y, " ~ ", rhs, " | ", fes))
  mod <- tryCatch(if (ppml) fepois(f, d, cluster = as.formula(paste0("~", "sic4"))) else feols(f, d, cluster = as.formula("~sic4")), error = function(e) NULL)
  if (is.null(mod)) return(NULL)
  mf <- tryCatch(if (ppml) fepois(f, d, cluster = as.formula(paste0("~", unit))) else feols(f, d, cluster = as.formula(paste0("~", unit))), error = function(e) NULL)
  cf <- coef(mod); se <- se(mod); sef <- if (!is.null(mf)) se(mf) else NA
  nm <- names(cf)
  pre <- mean(d[[y]][d[[post]] == 0], na.rm = TRUE)
  out <- data.table(dataset = ds, outcome = y, group = g, sample = sample, fe = fe, spec = spec, ppml = ppml, term = nm, coef = cf, se_sic4 = se,
                    se_unit = sef[nm], n_obs = nobs(mod), n_units = uniqueN(d[[unit]]), pre_mean = pre)
  out[, `:=`(p_sic4 = 2 * pnorm(-abs(coef / se_sic4)), mde = 2.8 * se_sic4)]
  if (g == "mfg" && spec == "both") {   # net effect +1SD in both
    V <- vcov(mod); k <- c("pIN", "pOUT", "pX"); b <- sum(cf[k]); s <- sqrt(sum(V[k, k]))
    out <- rbind(out, data.table(dataset = ds, outcome = y, group = g, sample = sample, fe = fe, spec = spec, ppml = ppml, term = "NET(IN+OUT+IN*OUT)",
                                 coef = b, se_sic4 = s, se_unit = NA, n_obs = nobs(mod), n_units = uniqueN(d[[unit]]), pre_mean = pre,
                                 p_sic4 = 2 * pnorm(-abs(b / s)), mde = 2.8 * s))
  }
  out
}
add <- function(x) if (!is.null(x)) EST[[length(EST) + 1]] <<- x

dyn <- function(d, y, g, ds, sample, base, per, unit, ppml = FALSE, fe = "A", pre_periods) {
  d <- copy(d); set(d, j = "pp", value = as.character(d[[per]]))
  ps <- setdiff(sort(unique(d$pp)), as.character(base))
  # ★ IMPORTANT: loop variable must not be named like a data column (a column "q" shadowed it inside data.table j in run 3; fixed 2026-11-02)
  for (qq_ in ps) { set(d, j = paste0("I", qq_), value = d$zIN * (d$pp == qq_)); if (g == "mfg") set(d, j = paste0("O", qq_), value = d$zOUT * (d$pp == qq_)) }
  d[, pO301d := zO301]
  terms <- c(paste0("I", ps), if (g == "mfg") paste0("O", ps))
  fes <- paste0(unit, " + sic1^", per)
  f <- as.formula(paste0(y, " ~ ", paste(c(paste0("`", terms, "`"), paste0("i(pp, zO301, ref = '", base, "')")), collapse = " + "), " | ", fes))
  mod <- tryCatch(if (ppml) fepois(f, d, cluster = ~sic4) else feols(f, d, cluster = ~sic4), error = function(e) {message(conditionMessage(e)); NULL})
  if (is.null(mod)) return(NULL)
  cf <- coef(mod); se <- se(mod)
  keep <- names(cf)[grepl("^`?[IO]", names(cf))]
  wi <- tryCatch(wald(mod, keep = paste0("^`?I(", paste(pre_periods, collapse = "|"), ")`?$"), print = FALSE)$p, error = function(e) NA)
  wo <- if (g == "mfg") tryCatch(wald(mod, keep = paste0("^`?O(", paste(pre_periods, collapse = "|"), ")`?$"), print = FALSE)$p, error = function(e) NA) else NA
  DYN[[length(DYN) + 1]] <<- data.table(dataset = ds, outcome = y, group = g, sample = sample, term = gsub("`", "", keep), coef = cf[keep], se_sic4 = se[keep], n_obs = nobs(mod),
                                       pre_wald_p_IN = wi, pre_wald_p_OUT = wo)
}


tq <- as.data.table(read_parquet(paste0(CA, "innov47_261102/panel_firm_quarter_final.parquet")))[, .(id = final_dunsno, q, n)]
tq <- merge(tq, fx[, .(id, IN, OUT, O301, sic4, grp, ebin)], by = "id"); tq[, anyf := as.numeric(n > 0)]
pre47 <- c(20221:20224, 20231:20234, 20241:20244)
for (g in c("mfg", "wr")) { d <- prep(tq, g, "id")
  dyn(d, "n", g, "47_trademarks_PPML", "A", base = 20251, per = "q", unit = "id", ppml = TRUE, pre_periods = pre47)
  dyn(d, "anyf", g, "47_trademarks_LPM", "A", base = 20251, per = "q", unit = "id", pre_periods = pre47) }
pq <- fread(paste0(CA, "innov48_261102/panel_duns_quarter.csv"))[, .(id = final_dunsno, q, ENG, SCI, PRODDEV, N)]
pq <- unique(pq, by = c("id", "q")); pq[, s_RD := 100 * (ENG + SCI + PRODDEV) / N]; pq <- pq[N >= 1]
pq <- merge(pq, fx[, .(id, IN, OUT, O301, sic4, grp, ebin)], by = "id")
pre48 <- c("2023Q1", "2023Q2", "2023Q3", "2023Q4", "2024Q1", "2024Q2", "2024Q3", "2024Q4")
for (g in c("mfg", "wr")) { d <- prep(pq, g, "id"); dyn(d, "s_RD", g, "48_rd_posting_share", "A", base = "2025Q1", per = "q", unit = "id", pre_periods = pre48) }
if (length(DYN)) fwrite(rbindlist(DYN, fill = TRUE), dyn_out)
cat("wrote", length(DYN), "dynamic blocks\n"); sink()
