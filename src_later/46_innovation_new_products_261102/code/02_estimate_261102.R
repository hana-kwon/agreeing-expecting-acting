# ============================================================
# START DATE   : 2026-11-02
# CODE RUN DATE: 2026-11-02
# TIME TAKEN   : ~10 min
# DESCRIPTION  : Long-difference regressions of firm-level new-product / new-pair outcomes on 2025 tariff exposure (2024 weights),
#                with pre-trend placebo, mechanical-margin checks, heterogeneity; every spec logged with n, estimate, SE.
# INPUT FILES  : PRIVATE_CACHE{firm_long_261102.parquet, firm_cov_261102.parquet}
# OUTPUT FILES : P/analysis/46_innovation_new_products_261102/{log_specs_261102.csv, descriptive_2026_261102.csv, sample_counts_261102.csv}
# ============================================================
suppressMessages({library(arrow); library(fixest); library(data.table)})
loc <- file.path(Sys.getenv("HOME"), "private_cache/innov46_261102")
P <- "HOME"
od <- file.path(P, "analysis/46_innovation_new_products_261102")
f_log <- file.path(od, "log_specs_261102.csv")
if (file.exists(f_log)) stop("refusing to overwrite ", f_log)
L <- as.data.table(read_parquet(file.path(loc, "firm_long_261102.parquet")))
C <- as.data.table(read_parquet(file.path(loc, "firm_cov_261102.parquet")))
OUTS <- c("new_vs","new_n","new_any","new_frac","newpair_vs","newpair_n","neworig_n","drop_n","n_hs6",
          "new_vs_nonCN","new_vs_CN","new_n_nonCN","new_n_CN","newch_n","newch_vs")
L <- L[has_hist == TRUE]
W <- dcast(L, id ~ year, value.var = OUTS)
dd <- merge(C, W, by = "id", all.x = TRUE)
# size bins (D&B 2024 employment), missing as own category
dd[, sizebin := cut(emp, c(-Inf, 19, 49, 249, Inf), labels = c("1-19","20-49","50-249","250+"))]
dd[, sizebin := as.character(sizebin)][is.na(sizebin), sizebin := "NA"]
dd[, sic2c := ifelse(is.na(sic2), "NA", as.character(sic2))]
dd[, `:=`(x = exp_tot/10, x232 = exp_232/10, xieepa = exp_ieepa/10, x301 = exp_301/10,
          limp = log(imp24), cn = cn_share24, lhs6 = log(n_hs6_24), lhist = log1p(hist_n_pre24))]
dd <- dd[!is.na(x) & is.finite(limp)]
has <- function(o, y) !is.na(dd[[paste0(o, "_", y)]])
logrows <- list()
addlog <- function(m, spec, samp, y, expo, ctrl, note = "", term = "x", sd_y = NA, mean_y = NA) {
  ct <- coeftable(m); nm <- rownames(ct)[grepl(term, rownames(ct), fixed = TRUE)]
  for (r in nm) logrows[[length(logrows) + 1]] <<- data.table(spec = spec, sample = samp, outcome = y, exposure = r,
    controls = ctrl, n = nobs(m), est_per10pp = ct[r, 1], se = ct[r, 2], p = ct[r, 4], sd_dep = sd_y, mean_dep = mean_y, note = note)
}
run <- function(dat, y, rhs, spec, samp, ctrl, note = "", term = "x") {
  dat <- copy(dat); dat[, yy := get(y)]
  dat <- dat[is.finite(yy)]
  if (nrow(dat) < 500) return(invisible(NULL))
  m <- tryCatch(feols(as.formula(paste("yy ~", rhs, "| sic2c + sizebin")), data = dat, cluster = ~id), error = function(e) NULL)
  if (is.null(m)) return(invisible(NULL))
  addlog(m, spec, samp, y, rhs, ctrl, note, term, sd(dat$yy), mean(dat$yy)); invisible(m)
}
base <- "x + limp + cn"; aug <- "x + limp + cn + lhs6 + lhist"

# ---- sample definitions -------------------------------------------------
# A: has history and imports in both t-1 and t (conditional on importing in both years); B: balanced, imported every year 2022-2025
mk <- function(t0, t1) {
  s <- copy(dd); ok <- has("new_vs", t0) & has("new_vs", t1)
  s <- s[ok]
  for (o in OUTS) s[[paste0("d_", o)]] <- s[[paste0(o, "_", t1)]] - s[[paste0(o, "_", t0)]]
  s
}
A25 <- mk(2024, 2025); A24 <- mk(2023, 2024)
B  <- dd[has("new_vs",2023) & has("new_vs",2024) & has("new_vs",2025)]
mkB <- function(s, t0, t1) { for (o in OUTS) s[[paste0("d_", o)]] <- s[[paste0(o,"_",t1)]] - s[[paste0(o,"_",t0)]]; s }
B25 <- mkB(copy(B), 2024, 2025); B24 <- mkB(copy(B), 2023, 2024)
counts <- data.table(sample = c("A: both-year importers 2024&2025 with history (main)", "A: placebo 2023&2024 with history",
  "B: balanced importers 2022-2025 (main/placebo)", "2024 importers in panel with exposure", "2024 importers with history (any import 2022-2023)", "of which importing in 2025"),
  n_firms = c(nrow(A25), nrow(A24), nrow(B), sum(dd$imp_2024), sum(dd$imp_2024 & (dd$imp_2022 | dd$imp_2023)), sum(dd$imp_2024 & (dd$imp_2022 | dd$imp_2023) & dd$imp_2025)))
fwrite(counts, file.path(od, "sample_counts_261102.csv"))

# ---- 1. main + placebo, plain and augmented controls ----------------------
for (o in OUTS) {
  for (nmS in c("A", "B")) {
    S25 <- if (nmS == "A") A25 else B25; S24 <- if (nmS == "A") A24 else B24
    y <- paste0("d_", o)
    run(S25, y, base, "main_2025m2024", nmS, "SIC2+size FE, log imp, CN share")
    run(S24, y, base, "placebo_2024m2023", nmS, "SIC2+size FE, log imp, CN share")
    run(S25, y, aug, "main_2025m2024_aug", nmS, "+ log HS6 count 2024, log history size")
    run(S24, y, aug, "placebo_2024m2023_aug", nmS, "+ log HS6 count 2024, log history size")
  }
}
# ---- 2. exposure components (232 / IEEPA / 301 jointly) --------------------
cmp <- "x232 + xieepa + x301 + limp + cn"
for (o in c("new_vs","new_n","new_any","newpair_vs","newpair_n","neworig_n","drop_n","n_hs6")) for (nmS in c("A","B")) {
  S25 <- if (nmS == "A") A25 else B25; S24 <- if (nmS == "A") A24 else B24
  for (tm in c("x232","xieepa","x301")) {
    run(S25, paste0("d_", o), cmp, "components_main", nmS, "joint 232+IEEPA+301", term = tm)
    run(S24, paste0("d_", o), cmp, "components_placebo", nmS, "joint 232+IEEPA+301", term = tm)
  }
}
# ---- 3. mechanical margins -------------------------------------------------
# 3a. extensive margin: stopping among 2024 importers (with history) -- any coded import in 2025
E <- dd[has("new_vs", 2024)]
E[, stay := as.numeric(!is.na(new_vs_2025))]
run(E, "stay", base, "extensive_stay", "2024 importers w/ history", "base", "1[any coded import 2025]")
# 3b. unconditional: stoppers get 0 on new_* (mechanical zero), n_hs6 = 0
for (o in c("new_n","new_vs","new_any","newpair_n","newpair_vs","neworig_n")) {
  U <- copy(E); U[, y25 := fifelse(stay == 1, get(paste0(o,"_2025")), 0)]
  U[, dy := y25 - get(paste0(o, "_2024"))]
  run(U, "dy", base, paste0("unconditional_", o), "2024 importers w/ history", "base", "stoppers set to 0 in 2025; dep = ", o)
}
# 3c. quartile of baseline HS6 count: regression to the mean/mechanics
A25[, q := cut(n_hs6_24, c(0, 2, 5, 15, Inf), labels = c("1-2","3-5","6-15","16+"))]
for (qq in levels(A25$q)) for (o in c("new_vs","new_n","newpair_n"))
  run(A25[q == qq], paste0("d_", o), base, paste0("byHS6ct_", qq), "A", "base", "split by 2024 HS6 count")
A24[, q := cut(n_hs6_24, c(0, 2, 5, 15, Inf), labels = c("1-2","3-5","6-15","16+"))]
for (qq in levels(A24$q)) for (o in c("new_vs","new_n","newpair_n"))
  run(A24[q == qq], paste0("d_", o), base, paste0("byHS6ct_placebo_", qq), "A", "base", "split by 2024 HS6 count")
# 3d. baseline-level control (regression to the mean): add lagged outcome level
for (o in c("new_vs","new_n","newpair_n")) {
  A25[, lag := get(paste0(o, "_2024"))]
  run(A25, paste0(o, "_2025"), paste(base, "+ lag + lhs6"), "levelY_lag_control", "A", "base+lag Y+lhs6", "dep = Y2025 level, Y2024 on rhs")
}
# ---- 4. heterogeneity (slopes by group, common controls/FE) -----------------
het <- function(S, grpcol, tag, samp) {
  for (o in c("new_vs","new_n","newpair_n","n_hs6")) {
    s <- copy(S); s[, g := get(grpcol)]; s <- s[!is.na(g)]; s[, yy := get(paste0("d_", o))]
    s <- s[is.finite(yy)]
    m <- tryCatch(feols(yy ~ i(g, x) + i(g) + limp + cn | sic2c + sizebin, data = s, cluster = ~id), error = function(e) NULL)
    if (is.null(m)) next
    ct <- coeftable(m); rn <- rownames(ct)[grepl("^g::.*:x$", rownames(ct))]
    for (r in rn) logrows[[length(logrows) + 1]] <<- data.table(spec = paste0("het_", tag, "_", samp), sample = "A", outcome = paste0("d_", o),
      exposure = r, controls = "group-specific slope; base ctrl", n = nobs(m), est_per10pp = ct[r,1], se = ct[r,2], p = ct[r,4],
      sd_dep = sd(s$yy), mean_dep = mean(s$yy), note = paste("n in group =", sum(s$g == sub(":x$","",sub("^g::","",r)))))
  }
}
for (nm in c("main","placebo")) {
  S <- if (nm == "main") copy(A25) else copy(A24)
  S[, g_emp := sizebin]
  S[, g_orig := fcase(n_orig24 == 1, "1 origin", n_orig24 <= 3, "2-3 origins", n_orig24 >= 4, "4+ origins")]
  S[, g_freq3 := fifelse(years_imp_22_24 == 3, "imports all 3 yrs", "fewer yrs")]
  S[, g_rec := fifelse(rec24 > median(rec24, na.rm = TRUE), "records above median", "records at/below median")]
  S[, g_party := fifelse(dime_party %in% c("Republican","Democratic"), dime_party, NA_character_)]
  for (gg in c("g_emp","g_orig","g_freq3","g_rec","g_party")) het(S, gg, gg, nm)
}
# ---- 5. descriptive 2026 partial year ---------------------------------------
dd[, tert := cut(x, quantile(x, c(0, 1/3, 2/3, 1), na.rm = TRUE), include.lowest = TRUE, labels = c("low","mid","high"))]
ds <- rbindlist(lapply(2024:2026, function(y) {
  z <- dd[has("new_vs", y)]; z[, v := get(paste0("new_vs_", y))]; z[, n := get(paste0("new_n_", y))]; z[, h := get(paste0("n_hs6_", y))]
  z[, .(year = y, n_firms = .N, mean_new_vs = mean(v), mean_new_n = mean(n), mean_hs6 = mean(h), mean_new_frac = mean(get(paste0("new_frac_", y)))), by = tert]
}))
fwrite(ds[order(year, tert)], file.path(od, "descriptive_2026_261102.csv"))
fwrite(rbindlist(logrows), f_log)
cat("specs logged:", length(logrows), "\n"); print(counts)
