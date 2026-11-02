# ============================================================
# START DATE   : 2026-11-02
# CODE RUN DATE: 2026-11-02
# TIME TAKEN   : ~5 min
# DESCRIPTION  : Linear probability models with country and two-week-bin fixed effects, two-way clustered (country, bin), per SD of each measure.
#                M1 measure only; M2 + same-outcome indicators for the PAST and concurrent windows (+ pending increases for O1).
#                Same models for the PAST window as outcome (statement after the event) and for sub-windows. Spec: analysis/53 00_spec_261102.md.
# INPUT FILES  : analysis/53_first_stage_261102/01_panel_261102.csv
# OUTPUT FILES : analysis/53_first_stage_261102/04_feols_261102.csv
# ============================================================
suppressMessages({library(fixest); library(data.table)})
args <- commandArgs(trailingOnly = FALSE); f <- sub("--file=", "", args[grep("--file=", args)]); f <- gsub("~+~", " ", f, fixed = TRUE)
OUT <- file.path(dirname(dirname(dirname(normalizePath(f)))), "analysis/53_first_stage_261102")
of <- file.path(OUT, "04_feols_261102.csv"); if (file.exists(of)) stop("output exists; refusing to overwrite")
P <- fread(file.path(OUT, "01_panel_261102.csv"))
MS <- c("a1_jev_score","a2_jev_share","b1_n_future","b2_n_threat","b3_n_intent","c1_first_threat","c2_first_threat_or_intent","d1_n_specific","d2_n_dated","e1_n_notinforce","e2_n_notinforce_higher","f1_n_future_higher","f2_n_new_vs_recent","f3_n_higher_no_policy_yet","ref_mentions28","ref_jev28","ref_tirade_threats4")
for (m in MS) { v <- P[[m]]; v[is.na(v)] <- 0; P[[paste0(m, "_z")]] <- (v - mean(v)) / sd(v) }
OUTC <- c("O1_eff_incr","O2_signing","O3_tir_threat","O3_tir_announce","O3_tir_implement","O3_tir_any")
res <- list()
for (o in OUTC) for (w in c("fut_2_12","past_2_12","imm_0_2","fut_2_4","fut_5_8","fut_9_12")) {
  y <- paste0(o, "__", w); d <- P[!is.na(get(y))]; d[, Y := get(y)]
  d[, pastc := get(paste0(o, "__past_2_12"))]; d[, concc := get(paste0(o, "__conc"))]
  for (m in MS) for (mod in c("M1","M2")) {
    if (mod == "M2" && !(w %in% c("fut_2_12"))) next
    rhs <- paste0(m, "_z"); if (mod == "M2") rhs <- paste(rhs, "+ pastc + concc", if (o == "O1_eff_incr") "+ pending" else "")
    mdl <- tryCatch(feols(as.formula(paste("Y ~", rhs, "| iso3 + k")), d, cluster = ~iso3 + k, notes = FALSE, warn = FALSE), error = function(e) NULL)
    if (is.null(mdl)) next
    cn <- paste0(m, "_z"); ct <- coeftable(mdl)
    if (!(cn %in% rownames(ct))) next
    res[[length(res) + 1]] <- data.table(outcome = o, window = w, model = mod, measure = m, coef_per_sd = ct[cn, 1], se = ct[cn, 2], p = ct[cn, 4], n = nobs(mdl), mean_y = mean(d$Y))
  }
}
fwrite(rbindlist(res), of); cat("rows", length(res), "\n")
