# ============================================================
# START DATE   : 2026-11-02
# CODE RUN DATE: 2026-11-02
# TIME TAKEN   : ~10 min
# DESCRIPTION  : Post-results additions (  ): fixed unconditional (stoppers = 0) specs; asinh versions of count outcomes
#                (main, placebo, heterogeneity). Same samples, controls and FE as 02.
# INPUT FILES  : PRIVATE_CACHE{firm_long_261102.parquet, firm_cov_261102.parquet}
# OUTPUT FILES : P/analysis/46_innovation_new_products_261102/log_specs_robust_261102.csv
# ============================================================
suppressMessages({library(arrow); library(fixest); library(data.table)})
loc <- file.path(Sys.getenv("HOME"), "private_cache/innov46_261102")
P <- "HOME"
f_log <- file.path(P, "analysis/46_innovation_new_products_261102/log_specs_robust_261102.csv")
if (file.exists(f_log)) stop("refusing to overwrite ", f_log)
L <- as.data.table(read_parquet(file.path(loc, "firm_long_261102.parquet")))[has_hist == TRUE]
C <- as.data.table(read_parquet(file.path(loc, "firm_cov_261102.parquet")))
OUTS <- c("new_vs","new_n","new_any","newpair_n","neworig_n","drop_n","n_hs6")
dd <- merge(C, dcast(L, id ~ year, value.var = OUTS), by = "id", all.x = TRUE)
dd[, sizebin := as.character(cut(emp, c(-Inf,19,49,249,Inf), labels = c("1-19","20-49","50-249","250+")))][is.na(sizebin), sizebin := "NA"]
dd[, sic2c := ifelse(is.na(sic2), "NA", as.character(sic2))]
dd[, `:=`(x = exp_tot/10, limp = log(imp24), cn = cn_share24)]
dd <- dd[!is.na(x) & is.finite(limp)]
has <- function(o, y) !is.na(dd[[paste0(o, "_", y)]])
rows <- list()
run <- function(dat, y, spec, samp, note = "") {
  dat <- copy(dat); dat[, yy := get(y)]; dat <- dat[is.finite(yy)]
  m <- feols(yy ~ x + limp + cn | sic2c + sizebin, data = dat, cluster = ~id)
  ct <- coeftable(m)
  rows[[length(rows) + 1]] <<- data.table(spec = spec, sample = samp, outcome = y, exposure = "x", n = nobs(m), est_per10pp = ct["x",1],
    se = ct["x",2], p = ct["x",4], sd_dep = sd(dat$yy), mean_dep = mean(dat$yy), note = note)
}
ihs <- function(v) asinh(v)
A <- function(t0, t1) dd[has("new_vs", t0) & has("new_vs", t1)]
B <- dd[has("new_vs",2023) & has("new_vs",2024) & has("new_vs",2025)]
add <- function(S, t0, t1) { for (o in c("new_n","newpair_n","neworig_n","drop_n","n_hs6"))
  S[[paste0("a_", o)]] <- ihs(S[[paste0(o,"_",t1)]]) - ihs(S[[paste0(o,"_",t0)]]); S }
S <- list(A_main = add(copy(A(2024,2025)), 2024, 2025), A_plac = add(copy(A(2023,2024)), 2023, 2024),
          B_main = add(copy(B), 2024, 2025), B_plac = add(copy(B), 2023, 2024))
for (nm in names(S)) for (o in c("new_n","newpair_n","neworig_n","drop_n","n_hs6"))
  run(S[[nm]], paste0("a_", o), paste0("asinh_", ifelse(grepl("main", nm), "main_2025m2024", "placebo_2024m2023")), substr(nm,1,1), "asinh difference")
# fixed unconditional specs: among 2024 importers with history, stoppers get 0 in 2025
E <- dd[has("new_vs", 2024)]; E[, stay := as.numeric(has("new_vs", 2025)[match(E$id, dd$id)])]
for (o in c("new_n","new_vs","new_any","newpair_n","neworig_n")) {
  E[, y25 := fifelse(stay == 1, get(paste0(o,"_2025")), 0)]; E[, dy := y25 - get(paste0(o,"_2024"))]
  run(E, "dy", paste0("unconditional_", o), "2024 importers w/ history", "stoppers set to 0 in 2025 (dep = 2025 minus 2024 of this outcome)")
}
# heterogeneity with asinh (common controls/FE, group-specific slopes)
Sm <- copy(S$A_main); Sp <- copy(S$A_plac)
for (nm in c("main","placebo")) {
  T <- if (nm == "main") Sm else Sp
  T[, g_orig := fcase(n_orig24 == 1, "1 origin", n_orig24 <= 3, "2-3 origins", n_orig24 >= 4, "4+ origins")]
  T[, g_rec := fifelse(rec24 > median(rec24, na.rm = TRUE), "records above median", "records at/below median")]
  T[, g_emp := sizebin]
  T[, g_hs6 := as.character(cut(n_hs6_24, c(0,2,5,15,Inf), labels = c("1-2","3-5","6-15","16+")))]
  T[, g_party := fifelse(dime_party %in% c("Republican","Democratic"), dime_party, NA_character_)]
  for (gg in c("g_orig","g_rec","g_emp","g_hs6","g_party")) for (o in c("a_new_n","a_n_hs6","a_newpair_n")) {
    s <- copy(T); s[, g := get(gg)]; s <- s[!is.na(g)]; s[, yy := get(o)]
    m <- feols(yy ~ i(g, x) + i(g) + limp + cn | sic2c + sizebin, data = s, cluster = ~id); ct <- coeftable(m)
    for (r in rownames(ct)[grepl("^g::.*:x$", rownames(ct))])
      rows[[length(rows)+1]] <- data.table(spec = paste0("asinh_het_", gg, "_", nm), sample = "A", outcome = o, exposure = r, n = nobs(m),
        est_per10pp = ct[r,1], se = ct[r,2], p = ct[r,4], sd_dep = sd(s$yy), mean_dep = mean(s$yy), note = "asinh difference")
  }
}
fwrite(rbindlist(rows), f_log); cat("logged", length(rows), "\n")
