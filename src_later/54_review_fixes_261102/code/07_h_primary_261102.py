# ============================================================
# START DATE   : 2026-11-02
# CODE RUN DATE: 2026-11-02
# TIME TAKEN   : ~20 min
# DESCRIPTION  : H. Re-run of the PRIMARY tests of 51_survey_exposure_signal (expectation outcomes, pre-message Z samples, O1-O3 contrasts,
#                two 28-day China signals) with (1) a date-shift placebo that DROPS responses whose shifted signal is undefined (no zero fill) and
#                re-orthogonalizes on the retained rows, (2) slopes of the outcome on the signal reported separately for the Z=1 and Z=0 groups
#                (controls + wave + industry, no date FE since the signal is date-level; SE clustered by date and owner), and (3) the number of
#                response dates with both groups present. Adapted from 51/01_exposure_signal_261102.py (copied, not imported: that file runs on import).
# INPUT FILES  : same as 51/01_exposure_signal_261102.py
# OUTPUT FILES : analysis/54_review_fixes_261102/07_h_primary_261102.csv (aggregates only)
# ============================================================
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.linalg import qr

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve()
TOPIC = HERE.parents[2]
SHARED = HERE.parents[7] / "PRIVATE_DATA_ROOT"
IA = SHARED / "SMB_tariff/ideology_analysis"
OUT = TOPIC / "analysis/54_review_fixes_261102"
outf = OUT / "07_h_primary_261102.csv"
if outf.exists():
    sys.exit(f"{outf} exists; refusing to overwrite.")
SHIFTS = [k for k in range(-52, 53) if abs(k) >= 5]  # weeks; 96 shifts

# ---------------- survey data ----------------
keep = ["responseid", "uid", "startdate", "country", "three_info", "long_term_prior", "short_term_prior", "import_tariff_expect",
        "knowledge_new2_1", "knowledge_new1_1", "tariff_guess_chn", "invest_post2_1", "prices_post2_1",
        "import_locs1_1", "import_locs1_2", "import_locs1_3", "media_top_us"]
d = pd.read_stata(IA / "inputs/analysis_all.dta", columns=keep, convert_categoricals=False)
d = d[d.country == 1]
a = pd.read_stata(IA / "outputs/alignment_for_stata.dta", convert_categoricals=False)
acols = ["responseid", "wave", "politics_us_group", "importer_dum", "export_dum", "fcomp_dum", "industry_code", "emp_2024_bins",
         "sales_2024_val", "import_loc_post2_1__toward", "import_loc_post2_3__away"]
d = d.merge(a[acols], on="responseid", validate="1:1")
d["date"] = pd.to_datetime(d.startdate, errors="coerce").dt.normalize()
d = d.dropna(subset=["date"]).reset_index(drop=True)
for c in ["emp_2024_bins", "sales_2024_val", "importer_dum", "export_dum", "fcomp_dum"]:
    d["miss_" + c] = d[c].isna().astype(float)
    d[c] = d[c].fillna(d[c].median() if "bins" in c or "val" in c else 0.0)
d["industry_code"] = d["industry_code"].fillna(-1).astype(int)
d["three_info"] = d.three_info.fillna(0).astype(int)

# --- Z variables ---
L = d[["import_locs1_1", "import_locs1_2", "import_locs1_3"]].apply(lambda s: s.fillna("").str.strip())
listed = L.apply(lambda s: ~s.isin(["", "Not Applicable"])).any(axis=1)
china_any = (L == "China").any(axis=1)
d["grp"] = np.where(china_any, "china", np.where(listed, "other", np.where(d.importer_dum == 0, "non", "x")))
d["china_top1"] = (L["import_locs1_1"] == "China")
m = d.media_top_us.fillna("")
outlets = m.apply(lambda s: [t.strip() for t in s.split(",") if t.strip() and not t.startswith("Other")])
d["n_out"] = outlets.apply(len).astype(float)
d["n_out"] = np.where(m == "", np.nan, d.n_out)
d["biz"] = np.where(m == "", np.nan, outlets.apply(lambda l: float(any(t in ("WSJ", "Financial Times", "Reuters/AP") for t in l))))
d["foxtruth"] = np.where(m == "", np.nan, outlets.apply(lambda l: float(any(t in ("Fox", "Truth Social", "Twitter/X") for t in l))))
d["rep"] = d.politics_us_group.map({"Republican": 1.0, "Democrat": 0.0})
steps = pd.read_csv(TOPIC / "data/16_tariff_schedule_261029/china_additional_duty_steps_261029.csv").iloc[:, :2]
steps.columns = ["eff", "duty_cn"]
steps["eff"] = pd.to_datetime(steps.eff)
d = pd.merge_asof(d.sort_values("date"), steps.sort_values("eff"), left_on="date", right_on="eff", direction="backward").reset_index(drop=True)
guess = np.where(d.wave.astype(str).str.contains("W4"), d.knowledge_new1_1, d.tariff_guess_chn)
d["know"] = np.where(np.isnan(guess), np.nan, (np.abs(guess - d.duty_cn) <= 10).astype(float))
W = d.wave.astype(str)
d["wv"] = W.str.extract(r"(W\d)")[0]

# ---------------- signals ----------------
S = pd.read_csv(TOPIC / "analysis/49_statement_type_261102/05_daily_series_261102.csv", parse_dates=["date"])
idx = pd.date_range(S.date.min(), S.date.max())


def build_signals():
    sig = {}
    for var in ["base", "thrint"]:
        for q in ["broad", "china"]:
            x = S[(S.variant == var) & (S.question == q)].set_index("date").reindex(idx)
            num, ev = x.num.fillna(0.0), x.ev.fillna(0.0)
            # ★ CORE LOGIC: window [t-W, t-1] ends the day before the response date (same convention as file s28/surprise).
            prior = (num.rolling(90).sum() / ev.rolling(90).sum().replace(0, np.nan)).shift(29)  # [t-118, t-29]
            for w in [3, 7, 14, 28]:
                lev = (num.rolling(w).sum() / ev.rolling(w).sum().replace(0, np.nan)).shift(1)
                sig[f"{var}_{q}_w{w}_level"] = lev
                sig[f"{var}_{q}_w{w}_surprise"] = lev - prior
    return sig


SIG = build_signals()
# check: reconstructed 28-day level and surprise vs the stored columns of the file
chk = S[(S.variant == "base") & (S.question == "china")].set_index("date").reindex(idx)
dif_l = (SIG["base_china_w28_level"] - chk.s28).abs().max()
dif_s = (SIG["base_china_w28_surprise"] - chk.surprise).abs().max()
print("signal reconstruction max abs diff: level", dif_l, "surprise", dif_s)

# ---------------- contrasts ----------------
CONTR = {  # name: (family, rows mask fn, Z fn)
    "O1_china_vs_otherimp": ("origin", lambda t: t.grp.isin(["china", "other"]), lambda t: (t.grp == "china").astype(float)),
    "O2_china_vs_nonimp": ("origin", lambda t: t.grp.isin(["china", "non"]), lambda t: (t.grp == "china").astype(float)),
    "O3_otherimp_vs_nonimp": ("origin", lambda t: t.grp.isin(["other", "non"]), lambda t: (t.grp == "other").astype(float)),
    "O4_china_top1_vs_otherimp": ("origin", lambda t: t.grp.isin(["china", "other"]), lambda t: t.china_top1.astype(float)),
    "N1_n_outlets_z": ("news", lambda t: t.n_out.notna(), lambda t: (t.n_out - t.n_out.mean()) / t.n_out.std()),
    "N2_wsj_ft_reuters": ("news", lambda t: t.biz.notna(), lambda t: t.biz),
    "N3_fox_truth_x": ("news", lambda t: t.foxtruth.notna(), lambda t: t.foxtruth),
    "K1_knows_china_duty": ("knowledge", lambda t: t.know.notna(), lambda t: t.know),
    "P1_republican": ("party", lambda t: t.rep.notna(), lambda t: t.rep),
}
# waves where Z was measured before the Information_A_B message (QSF block order); None = no pre-message sample
PRE = {"origin": ["W1", "W2", "W4"], "knowledge": ["W1", "W2", "W4"], "party": ["W0", "W1", "W2", "W3", "W4", "W5"], "news": []}
# waves where Z precedes the specific outcome question (QSF question order)
ZBEFORE_Y = {("origin", "long_term_prior"): "no (W1,W2: import_locs after the prior; before the message)", ("origin", "short_term_prior"): "no (same)",
             ("origin", "import_tariff_expect"): "yes (W4)", ("origin", "knowledge_new2_1"): "no (knowledge_new precedes import_locs in W4; before message)"}
EXP = {"expect": [("long_term_prior", ["W1", "W2", "W3"]), ("short_term_prior", ["W1", "W2", "W3"]),
                  ("import_tariff_expect", ["W4"]), ("knowledge_new2_1", ["W4"])],
       "plan": [(y, ws) for y in ["invest_post2_1", "prices_post2_1", "import_loc_post2_1__toward", "import_loc_post2_3__away"]
                for ws in (["W1", "W2", "W3"], ["W4"])]}
CTRL = ["emp_2024_bins", "sales_2024_val", "miss_emp_2024_bins", "miss_sales_2024_val", "importer_dum", "export_dum", "fcomp_dum",
        "miss_importer_dum", "miss_export_dum", "miss_fcomp_dum"]
PRIMARY_Z = {"O1_china_vs_otherimp", "O2_china_vs_nonimp", "O3_otherimp_vs_nonimp"}
PRIMARY_SIG = {"base_china_w28_level", "base_china_w28_surprise"}


def fe(v):
    return pd.get_dummies(v, drop_first=True).astype(float)


def clustvar(sc, g):
    """One-way cluster variance of sum(score*e)."""
    return np.sum(pd.Series(sc).groupby(g).sum().values ** 2)


def run_cell(t, z_name, y, sample_label, waves):
    fam, rowfn, zfn = CONTR[z_name]
    t = t[t.wv.isin(waves) & t[y].notna()]
    t = t[rowfn(t)].copy()
    t["Z"] = zfn(t)
    t = t[t.Z.notna()]
    n0 = len(t)
    if n0 < 200 or t.Z.nunique() < 2 or min(t.Z.eq(t.Z.max()).sum(), t.Z.ne(t.Z.max()).sum()) < 30:
        return [dict(z=z_name, family=fam, outcome=y, sample=sample_label, waves="+".join(waves), n=n0, note="n<200 or smaller Z group <30; not run")]
    rows = []
    # design that does not depend on the signal's missing pattern: date FE + controls + Z main
    dates = t.date.values
    base_cols = [t[CTRL + ["Z"]].astype(float), fe(t.date), fe(t.industry_code), fe(t.wv), fe(t.three_info)]
    X0 = pd.concat(base_cols, axis=1)
    X0 = X0.loc[:, X0.std() > 0].values
    X0 = np.column_stack([np.ones(n0), X0])
    yv = t[y].astype(float).values
    uid = pd.factorize(t.uid)[0]
    dt = pd.factorize(t.date)[0]
    pair = pd.factorize(pd.Series(uid.astype(str)) + "_" + pd.Series(dt.astype(str)))[0]
    cache = {}
    for sname, sser in {k: v for k, v in SIG.items() if k in PRIMARY_SIG}.items():
        S0 = sser.reindex(t.date).values
        ok = ~np.isnan(S0)
        if ok.sum() < 200:
            rows.append(dict(z=z_name, family=fam, outcome=y, sample=sample_label, waves="+".join(waves), signal=sname, n=int(ok.sum()), note="signal defined <200 rows; not run"))
            continue
        key = ok.tobytes()
        if key not in cache:
            Xs = X0[ok]
            Q, R, piv = qr(Xs, mode="economic", pivoting=True)
            r = int((np.abs(np.diag(R)) > 1e-8 * np.abs(R[0, 0])).sum())
            Q = Q[:, :r]
            cache[key] = (Q, yv[ok] - Q @ (Q.T @ yv[ok]))
        Q, yt = cache[key]
        Zv = t.Z.values[ok]

        def beta(sig_full):
            s = sig_full.copy()
            m_ = np.isnan(s)
            s = (s - np.nanmean(s)) / np.nanstd(s)  # standardize on defined rows
            s[m_] = 0.0
            v = (s * t.Z.values)[ok]
            vt = v - Q @ (Q.T @ v)
            den = vt @ vt
            return (vt @ yt) / den if den > 1e-10 else np.nan, vt, den
        b, vt, den = beta(S0)
        if not np.isfinite(b):
            continue
        e = yt - b * vt
        sc = vt * e
        u = uid[ok]; dd = dt[ok]; pp = pair[ok]
        Gu, Gd = len(np.unique(u)), len(np.unique(dd))
        Vd = clustvar(sc, dd) * Gd / max(Gd - 1, 1)
        Vu = clustvar(sc, u) * Gu / max(Gu - 1, 1)
        Vp = clustvar(sc, pp)
        V = Vd + Vu - Vp
        if V <= 0:
            V = max(Vd, Vu)
        se = np.sqrt(V) / den
        df_ = max(min(Gd, Gu) - 1, 1)
        p = 2 * stats.t.sf(abs(b / se), df_)
        sh = []; nvalid = []
        for k in SHIFTS:
            Sk = sser.reindex(t.date + pd.Timedelta(days=7 * k)).values
            ok2 = ok & ~np.isnan(Sk)
            if ok2.sum() < 200 or len(np.unique(t.Z.values[ok2])) < 2:
                continue
            Q2, R2, _ = qr(X0[ok2], mode="economic", pivoting=True)
            r2 = int((np.abs(np.diag(R2)) > 1e-8 * np.abs(R2[0, 0])).sum()); Q2 = Q2[:, :r2]
            y2 = yv[ok2] - Q2 @ (Q2.T @ yv[ok2])
            s2 = Sk[ok2]; s2 = (s2 - s2.mean()) / s2.std(); v2 = s2 * t.Z.values[ok2]; v2 = v2 - Q2 @ (Q2.T @ v2)
            if v2 @ v2 > 1e-10:
                sh.append((v2 @ y2) / (v2 @ v2)); nvalid.append(int(ok2.sum()))
        # group-specific slopes (no date FE)
        grp_out = {}
        dts = pd.Series(t.date.values[ok]); zz = t.Z.values[ok]
        both = int(len(set(dts[zz == 1]) & set(dts[zz == 0])))
        for g in (0.0, 1.0):
            mk = ok & (t.Z.values == g)
            tt = t[mk]; sg = (sser.reindex(tt.date).values); sg = (sg - sg.mean()) / sg.std()
            Xg = pd.concat([pd.Series(sg, index=tt.index, name="sig"), tt[[c for c in CTRL if c not in ("importer_dum", "miss_importer_dum")]].astype(float), fe(tt.wv), fe(tt.industry_code), fe(tt.three_info)], axis=1)
            Xg = Xg.loc[:, Xg.std() > 0]; Xg = pd.concat([pd.Series(1.0, index=Xg.index, name="const"), Xg], axis=1)
            import statsmodels.api as sm
            mg = sm.OLS(tt[y].astype(float), Xg).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(tt.date)[0]})
            ug = sm.OLS(tt[y].astype(float), Xg).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(tt.uid)[0]})
            grp_out[g] = (mg.params["sig"], max(mg.bse["sig"], ug.bse["sig"]), int(mk.sum()), tt.date.nunique())
        sh = np.array(sh)
        ysd = float(yt.std())
        rows.append(dict(z=z_name, family=fam, outcome=y, sample=sample_label, waves="+".join(waves), signal=sname,
                         variant=sname.split("_")[0], question=sname.split("_")[1], window=int(sname.split("_")[2][1:]), kind=sname.split("_")[3],
                         n=int(ok.sum()), n_date_clusters=Gd, n_owner_clusters=Gu, z_share=float(Zv.mean()), coef=b, se=se, p_cluster=p,
                         placebo_p=float((np.abs(sh) >= abs(b)).mean()) if len(sh) else np.nan, n_shifts=len(sh),
                         shift_sd=float(sh.std()) if len(sh) else np.nan, n_dates_both_groups=both, median_n_placebo_rows=float(np.median(nvalid)) if nvalid else np.nan,
                         slope_Z1=grp_out[1.0][0], se_Z1=grp_out[1.0][1], n_Z1=grp_out[1.0][2], dates_Z1=grp_out[1.0][3],
                         slope_Z0=grp_out[0.0][0], se_Z0=grp_out[0.0][1], n_Z0=grp_out[0.0][2], dates_Z0=grp_out[0.0][3], mde_units=2.8 * se, y_sd=float(yv[ok].std()),
                         mde_sd=2.8 * se / float(yv[ok].std()), note=""))
    return rows


rows = []
for kind, specs in {"expect": EXP["expect"]}.items():
    for y, ws in specs:
        for z_name, (fam, _, _) in {k: v for k, v in CONTR.items() if k in PRIMARY_Z}.items():
            pre = [w for w in ws if w[:2] in PRE[fam]]
            samples = [("all", ws)]
            if pre:
                samples.insert(0, ("pre", pre))
            for sl, wl in samples:
                if sl == "all":
                    continue
                if sl == "pre" and pre == ws:
                    sl = "pre(=all)"
                for r in run_cell(d, z_name, y, sl, [w[:2] for w in wl]):
                    r.update(kind=kind_ if (kind_ := kind) else kind, primary=int(kind == "expect" and sl.startswith("pre") and z_name in PRIMARY_Z and r.get("signal") in PRIMARY_SIG),
                             z_before_message=("yes" if sl.startswith("pre") else "no/not for all waves"),
                             z_before_outcome=ZBEFORE_Y.get((fam, y), "n/a (news/party/knowledge: see spec section 3)"))
                    rows.append(r)
        print(kind, y, ws, len(rows), flush=True)
res = pd.DataFrame(rows)
res.to_csv(outf, index=False)
print(res[res.note.fillna('')==''][['z','outcome','signal','n','coef','se','placebo_p','n_shifts','slope_Z1','se_Z1','slope_Z0','se_Z0','n_dates_both_groups']].round(3).to_string())
