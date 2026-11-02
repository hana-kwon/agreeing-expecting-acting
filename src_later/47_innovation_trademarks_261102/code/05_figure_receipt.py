# ============================================================
# START DATE   : 2026-11-02
# CODE RUN DATE: 2026-11-02
# TIME TAKEN   : ~5 min
# DESCRIPTION  : Event-study figure (PPML and LPM dynamic coefficients, small importers) and receipt JSON with hashes.
# INPUT FILES  : P/analysis/47_innovation_trademarks_261102/04_estimates<TAG>.csv, 03_match_diagnostics<TAG>.json
# OUTPUT FILES : 05_fig_dynamic<TAG>.png, receipt_trademarks_261102.json
# ============================================================
import sys, os, json, glob, hashlib, subprocess, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
TAG = sys.argv[1]
O = "HOME"
CODE = O.replace("analysis", "code")
C = os.path.expanduser("PRIVATE_CACHE")
out_png, rec = O + f"05_fig_dynamic{TAG}.png", O + "receipt_trademarks_261102.json"
assert not os.path.exists(out_png) and not os.path.exists(rec), "refuse to overwrite"
e = pd.read_csv(O + f"04_estimates{TAG}.csv")
fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
for a, spec, ttl in zip(ax, ["DYN_PPML", "DYN_LPM"], ["PPML: new trademark applications (count)", "Linear: any new application (0/1)"]):
    d = e[e.spec == spec].copy(); d["q"] = d.term.str.extract(r"::(\d+):")[0].astype(int)
    d["lab"] = d.q.astype(str).str[:4] + "Q" + d.q.astype(str).str[4]
    d = pd.concat([d, pd.DataFrame({"q": [20251], "lab": ["2025Q1"], "coef": [0.0], "se": [0.0]})]).sort_values("q")
    x = range(len(d))
    a.errorbar(x, d.coef, yerr=1.96 * d.se, fmt="o-", color="#1f4e79", capsize=2, lw=1)
    a.axhline(0, color="gray", lw=.8); a.axvline(list(d.q).index(20252) - .5, color="firebrick", ls="--", lw=1)
    a.set_xticks(list(x)); a.set_xticklabels(d.lab, rotation=90, fontsize=7); a.set_title(ttl, fontsize=10)
    a.set_ylabel("Coefficient on 2024 China import share x quarter\n(0% to 100% China share; ref 2025Q1)", fontsize=8)
fig.suptitle("Importers with <250 employees: China exposure and new trademark applications (dashed = 2025Q2 tariff shock)", fontsize=10)
plt.tight_layout(); plt.savefig(out_png, dpi=160)
def h(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()
ins = {}
for f in sorted(glob.glob(C + "raw/apc-*.zip") + glob.glob(C + "raw/daily-*.zip")): ins[os.path.basename(f)] = h(f)
S = "PRIVATE_DATA_ROOT/SMB_tariff/intermediate_data/"
for f in ["prep_reg_L2_update/smb_trade_duns_biweekly_260512.parquet", "prep_reg_L2_update/smb_duns_matched_firm_attributes_employment_260428.parquet", "prep_reg_L2_update/smb_panjiva_duns_concordance_final_260512.parquet.gzip"]:
    ins[f] = h(S + f)
outs = {os.path.basename(f): h(f) for f in glob.glob(O + "*") if os.path.isfile(f) and "receipt" not in f}
codes = {os.path.basename(f): h(f) for f in sorted(glob.glob(CODE + "*"))}
json.dump(dict(date="2026-11-02", inputs_sha256=ins, code_sha256=codes, run_order=["01_parse_trademark_xml.py (per zip)", "02_build_universe.py", "TAG=%s 03_match_build_panel.py" % TAG, "Rscript 04_estimate.R %s <lastq>" % TAG, "05_figure_receipt.py"],
                env=dict(python=sys.version.split()[0], pandas=pd.__version__, R="fixest (Rscript /usr/local/bin/Rscript)"), outputs_sha256=outs,
                not_done=["no human validation of matches (agent reading of 50 pairs only)", "no patent assignee file pulled", "no case-file API sample", "no launch/product data: filings are not launches", "no paid calls, no external sends"],
                note="USPTO zips live in PRIVATE_CACHE (too large for the project folder); hashes above."), open(rec, "w"), indent=1)
print("ok")
