# ============================================================
# START DATE   : 2026-10-29
# CODE RUN DATE: 2026-10-31
# TIME TAKEN   : ~1 s
# DESCRIPTION  : Measurement step 5. Links TiRADE rows to the public statements and compares the measure's timing
#                score with TiRADE's own label.
# INPUT FILES  : data/public/tirade-v1.5-rate-statements.csv, data/public/public_tariff_statements.csv, <run>/02_scores.csv
# OUTPUT FILES : <run>/05_tirade_links.csv, <run>/05_tirade_benchmark.csv
# ============================================================
# Step 5. External benchmark: TiRADE v1.5 (Kalemli-Ozcan, Soylu and Yildirim 2026) codes each stated U.S. tariff
# rate as threatened, announced or implemented. Link TiRADE rows to the public statements (Truth Social by post
# ID; presidential documents by text: share of the excerpt's word 5-grams found in a passage dated within one day,
# at least 0.5), then compare the measure's timing score with TiRADE's label on the same passages.
# Outputs: <run>/05_tirade_links.csv, <run>/05_tirade_benchmark.csv.
import re
from datetime import timedelta

import pandas as pd

from common import PUBLIC as DATA, run_dir

out = run_dir()
EXCLUDE = {"TRK-0611", "TRK-0612", "TRK-0063"}  # two rows on China's tariff on U.S. goods; one miscoded row
BROAD = {"All countries", "All foreign", "All Annex I partners"}
t = pd.read_csv(DATA / "tirade-v1.5-rate-statements.csv", dtype={"ref_id": str})
t = t[~t.row_id.isin(EXCLUDE)].copy()
st = pd.read_csv(DATA / "public_tariff_statements.csv")
st["d"] = pd.to_datetime(st.date)
W = re.compile(r"[^a-z0-9%]+")
words = lambda s: W.sub(" ", str(s).lower().replace("’", "").replace("'", "")).split()
grams = lambda w: {" ".join(w[i:i + 5]) for i in range(len(w) - 4)}
st["g"] = st.text.map(lambda s: grams(words(s)))
ts = dict(zip(st[st.source == "truth_social"].statement_id.str[3:], st[st.source == "truth_social"].statement_id))

links = []
for r in t.itertuples():
    sid, how = None, None
    if r.source_type == "Social" and str(r.ref_id) in ts:
        sid, how = ts[str(r.ref_id)], "post_id"
    elif r.source_type == "Speech" and isinstance(r.excerpt, str):
        ex = grams(words(r.excerpt))
        if len(ex) >= 3:
            d = pd.Timestamp(r.date)
            c = st[(st.d >= d - timedelta(days=1)) & (st.d <= d + timedelta(days=1)) & (st.source == "govinfo_cpd")]
            if len(c):
                sh = c.g.map(lambda g: len(ex & g) / len(ex))
                if sh.max() >= 0.5:
                    sid, how = c.loc[sh.idxmax(), "statement_id"], "text"
    if sid:
        et, role = str(r.event_type), str(r.role)
        grp = ("no_rate" if role == "no_rate" or et in ("Court event", "Announced (deal)") else "mixed" if "/" in et
               else "in_force" if (et == "Implemented" or role == "status_description") else "future")
        q = "china_path" if "China" in str(r.countries) else "broad_path" if (r.countries in BROAD or str(r.countries).startswith("All countries (")) else None
        links.append(dict(row_id=r.row_id, statement_id=sid, link=how, question_id=q, event_type=et, role=role, tirade_group=grp))
Lk = pd.DataFrame(links)
Lk.to_csv(out / "05_tirade_links.csv", index=False)

sc = pd.read_csv(out / "02_scores.csv")
x = Lk[Lk.question_id.notna()].merge(sc, on=["question_id", "statement_id"], how="inner")
lab = x.groupby(["question_id", "statement_id"]).agg(groups=("tirade_group", set), p_future=("p_future", "first"),
                                                       p_past=("p_in_force_or_past", "first")).reset_index()
lab = lab[lab.groups.map(lambda g: bool(g - {"no_rate"}))]
lab["tirade"] = lab.groups.map(lambda g: "future claim" if g & {"future", "mixed"} else "in force or past")
lab["measure_future"] = lab.p_future > lab.p_past
res = lab.groupby("tirade").agg(passages=("statement_id", "size"), share_scored_future=("measure_future", "mean")).round(3).reset_index()
res.to_csv(out / "05_tirade_benchmark.csv", index=False)
print(f"TiRADE rows linked: {len(Lk)} ({Lk.link.value_counts().to_dict()}); passages compared: {len(lab)}")
print(res.to_string(index=False))
