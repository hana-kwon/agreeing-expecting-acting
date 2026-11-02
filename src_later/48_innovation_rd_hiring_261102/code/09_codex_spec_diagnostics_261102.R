# ============================================================
# START DATE   : 2026-11-02
# CODE RUN DATE: 2026-11-02
# TIME TAKEN   : Limited diagnostic run; elapsed time in console log
# DESCRIPTION  : Reproduce Part B and compare a longer existing follow-up and baseline controls.
# INPUT FILES  : Existing innov48 local cache and read-only parent_imports_2024.parquet
# OUTPUT FILES : analysis/48_innovation_rd_hiring_261102/10_codex_spec_diagnostics_261102.csv
# ============================================================
# Exploratory diagnostic choices fixed before this run; original results already known.
# No claim of preregistration or causal validation. No new API calls or microdata outputs.
suppressMessages({library(data.table); library(fixest); library(arrow)})
setFixest_nthreads(2)
args <- commandArgs(TRUE)
out <- args[1]
stopifnot(length(out)==1, !file.exists(out))
C <- path.expand("PRIVATE_CACHE")
q <- fread(paste0(C,"fundq_xrdq_wrds.csv"), colClasses=list(character="gvkey"))
cs <- as.data.table(read_parquet(paste0(C,"gvkey_china_share.parquet")))
cs[,gvkey:=gsub(" ","0",sprintf("%06s",gvkey))]
pi <- as.data.table(read_parquet("PRIVATE_SERVER_PATH",col_select=c("gvkey","sic","in_us")))
pi[,sic2:=floor(as.numeric(sic)/100)]
q <- merge(q,cs[,.(gvkey,cs=china_share)],by="gvkey")
q <- merge(q,unique(pi[,.(gvkey,sic2,in_us)]),by="gvkey")
q <- q[curcdq=="USD" & datacqtr>="2023Q1" & datacqtr<="2026Q2" & !is.na(sic2) & !is.na(cs)]
cat("Duplicates before original first-row rule:",sum(duplicated(q[,.(gvkey,datacqtr)])),"\n")
q <- q[!duplicated(q[,.(gvkey,datacqtr)])]
q[,`:=`(firm=gvkey,qn=datacqtr,post=as.integer(datacqtr>="2025Q2"))]
q[,lxrd:=ifelse(xrdq>0 & !is.na(xrdq),log(xrdq),NA_real_)]
cat("Aggregate R&D coverage by calendar quarter:\n")
print(q[,.(firms=uniqueN(firm),rd_nonmissing=sum(!is.na(xrdq)),rd_positive=sum(xrdq>0,na.rm=TRUE)),by=qn])
cat("USD reporters' in_us composition:\n")
print(unique(q[,.(firm,in_us)])[,.(firms=.N),by=in_us])
# Baseline covariates only. Current sales/assets may be downstream of the tariff shock.
b <- q[qn>="2024Q1" & qn<="2024Q4",.(asset0=mean(atq[atq>0],na.rm=TRUE),rd0=mean(xrdq,na.rm=TRUE)),by=firm]
b[,`:=`(lnasset0=log(asset0),rdasset0=100*rd0/asset0)]
q <- merge(q,b,by="firm",all.x=TRUE)
q[,rd_asset_fixed:=ifelse(!is.na(xrdq)&is.finite(asset0)&asset0>0,100*xrdq/asset0,NA_real_)]
rows <- list()
run <- function(id,d,y="lxrd",controls=FALSE,seasonal=FALSE) {
 rhs <- if(controls) "cs:post + lnasset0:post + rdasset0:post" else "cs:post"
 fe <- if(seasonal) "firm^fqtr + sic2^qn" else "firm + sic2^qn"
 m <- feols(as.formula(paste(y,"~",rhs,"|",fe)),d,cluster=~firm,notes=FALSE)
 ct <- coeftable(m); k <- match("cs:post",rownames(ct)); ci <- confint(m,"cs:post")
 r <- data.table(spec=id,outcome=y,n=nobs(m),coef=ct[k,1],se=ct[k,2],p=ct[k,4],lo95=ci[1,1],hi95=ci[1,2])
 rows[[length(rows)+1]] <<- r; print(r)
}
run("01_original_2025Q4",q[qn<="2025Q4" & !is.na(lxrd)])
run("02_extend_2026Q2",q[!is.na(lxrd)])
# The next three checks use one common covariate-available sample.
d <- q[!is.na(lxrd)&is.finite(lnasset0)&is.finite(rdasset0)]
run("03_common_sample_2025Q4",d[qn<="2025Q4"])
run("04_common_sample_2026Q2",d)
run("05_baseline_assets_RD_controls_2026Q2",d,controls=TRUE)
run("06_firm_fiscal_quarter_FE_2026Q2",d,seasonal=TRUE)
run("07_RD_scaled_by_fixed_2024_assets",q[!is.na(rd_asset_fixed)&is.finite(lnasset0)&is.finite(rdasset0)],y="rd_asset_fixed",controls=TRUE)
fwrite(rbindlist(rows),out)
cat("Only aggregate estimates written. Original scripts and results unchanged.\n")
