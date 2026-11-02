# ============================================================
# START DATE   : 2026-11-02
# CODE RUN DATE: 2026-11-02
# TIME TAKEN   : ~5 min
# DESCRIPTION  : Event-study figure from the logged dynamic coefficients, plus receipt JSON with SHA-256 of inputs/outputs.
# INPUT FILES  : analysis/48_.../06_partA_dynamic_coefs_261102.csv, 07_partB_dynamic_coefs_261102.csv
# OUTPUT FILES : 08_event_study_261102.png, receipt_innov48_261102.json
# ============================================================
import os, sys, json, hashlib, glob, platform, pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
A = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../analysis/48_innovation_rd_hiring_261102/")) + "/"
C = os.path.expanduser("PRIVATE_CACHE")
for f in ("08_event_study_261102.png", "receipt_innov48_261102.json"):
    if os.path.exists(A + f): sys.exit("refuse to overwrite " + f)
def load(path, outc, grp=None):
    d = pd.read_csv(path); d = d[d.outcome == outc]
    if grp: d = d[d.group == grp]
    d["q"] = d.term.str.extract(r"qn::(\d{4}Q\d)")[0]
    d = d.set_index("q"); d.loc["2025Q1"] = [None, 0, 0, None, None, outc] + ([grp] if grp else [])
    return d.sort_index()
fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
def plot(a, d, title, col, sc=1):
    x = range(len(d)); b = d.Estimate.astype(float) * sc; s = d["Std. Error"].astype(float) * sc
    a.errorbar(x, b, yerr=1.96 * s, fmt="o-", color=col, capsize=3); a.axhline(0, c="grey", lw=.8)
    a.axvline(list(d.index).index("2025Q1") + .5, c="red", ls="--", lw=.8); a.set_xticks(list(x)); a.set_xticklabels(d.index, rotation=60, fontsize=8); a.set_title(title, fontsize=10)
plot(ax[0], load(A + "06_partA_dynamic_coefs_261102.csv", "s_RD", "all"), "A. R&D/eng/product-dev share of postings\n(pp per 1.0 China share; firms n=18,511)", "#1f77b4")
plot(ax[1], load(A + "06_partA_dynamic_coefs_261102.csv", "s_PROC", "all"), "A. Comparison: procurement share of postings\n(pp per 1.0 China share)", "#7f7f7f")
plot(ax[2], load(A + "07_partB_dynamic_coefs_261102.csv", "lxrd"), "B. Listed importers: log quarterly R&D (xrdq)\n(log points per 1.0 China share; firms n=1,165)", "#d62728")
fig.suptitle("China import share (2024) x quarter; base 2025Q1; red line = tariff period starts 2025Q2; 95% CI clustered by firm; firm and SIC2 x quarter FE", fontsize=9)
fig.tight_layout(); fig.savefig(A + "08_event_study_261102.png", dpi=150)
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
D = "PRIVATE_DATA_ROOT/SMB_tariff/intermediate_data/prep_reg_L2_update/"
inputs = [D + "revelio_smb/revelio_rcid_duns_bridge_full_261024.parquet", D + "smb_trade_duns_biweekly_260512.parquet", D + "smb_firm_features_for_L2_260626.parquet",
          D + "smb_duns_matched_firm_attributes_employment_260428.parquet", A + "../35_sec_tariff_expectations_261102/02_panel_261102.csv"]
inputs += sorted(glob.glob(C + "postings_bridge_*.parquet")) + [C + "fundq_xrdq_wrds.csv", C + "gvkey_china_share.parquet"]
vol = "PRIVATE_SERVER_PATH"
outs = sorted(glob.glob(A + "*.csv")) + sorted(glob.glob(A + "*.md")) + [A + "08_event_study_261102.png"]
r = {"created": "2026-11-02", "status": "exploratory; one agent; no cross-check",
     "inputs_sha256": {p: sha(p) for p in inputs if os.path.getsize(p) < 3e9},
     "volume_raw_postings_read_only": "year_2023/2024/2025.parquet (not hashed here, 40 GB; see SHA256SUMS on volume): " + open(vol).read().replace("\n", " | "),
     "code_order": sorted(os.listdir(os.path.dirname(os.path.abspath(__file__)))),
     "environment": {"python": platform.python_version(), "pandas": pd.__version__, "R": "/usr/local/bin/Rscript with fixest, data.table, arrow"},
     "outputs_sha256": {p: sha(p) for p in outs},
     "comparison_to_reference": "bridge-company posting totals recomputed from job-level files: 4,984,097 (2023), 5,527,917 (2024), 10,832,447 (2025); identical to revelio dictionary_revelio_full_261026.md",
     "not_done": ["no writes to private server or #Shared_", "no cross-agent verification", "Compustat xrdq came from WRDS comp.fundq because the volume file lacks it (253,653 rows vs 252,262)",
                  "job-level postings not re-read for SEC-firm rcids outside the SMB bridge", "no title-text analysis (job-level files have no title column)"]}
json.dump(r, open(A + "receipt_innov48_261102.json", "w"), indent=1, ensure_ascii=False)
