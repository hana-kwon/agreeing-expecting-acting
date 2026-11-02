# ============================================================
# START DATE   : 2026-11-01
# CODE RUN DATE: 2026-11-01
# TIME TAKEN   : See receipt
# DESCRIPTION  : Simulated allocation integrity and transparent sample-size scenarios.
# INPUT FILES  : None; synthetic IDs only, no participants or API connections.
# OUTPUT FILES : Aggregate allocation checks and power table in 05_experiment_dryrun.
# ============================================================
from pathlib import Path
from datetime import datetime
import json
import hashlib
import math
import numpy as np
import pandas as pd
from scipy.stats import norm
import scipy
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from restricted_paths import SURVEY_DIR, CUSTOMS_DIR, JEV_DIR, OUTPUT_ROOT, out_dir, fresh_out_dir

HERE=Path(__file__).resolve()
OUT=fresh_out_dir('customs_extension/05_experiment_dryrun')
OUT.mkdir(exist_ok=False)

def allocate(strata, seed):
    """Return in-memory assignments; a real platform must persist each assignment exactly once."""
    rng=np.random.default_rng(seed)
    result=pd.Series(index=strata.index,dtype=int)
    for _,ids in strata.groupby(strata,sort=True).groups.items():
        ids=list(ids)
        for start in range(0,len(ids),6):
            block=ids[start:start+6]
            labels=rng.permutation([0,0,1,1,2,2])
            result.loc[block]=labels[:len(block)]
    return result.astype(int)

# ★ CORE LOGIC: test assignment only on synthetic cases; no respondent-level file is produced.
strata=pd.Series(np.repeat(['batch1_passage1','batch1_passage2','batch2_passage1','batch2_passage2'],300))
assigned=allocate(strata,261101)
assert assigned.index.equals(strata.index) and assigned.notna().all()
assert assigned.equals(allocate(strata,261101))
counts=pd.crosstab(strata,assigned)
assert counts.eq(100).all().all()
for i in range(0,len(assigned),6):
    assert assigned.iloc[i:i+6].value_counts().sort_index().tolist()==[2,2,2]
counts.rename_axis('stratum').to_csv(OUT/'01_synthetic_allocation_counts.csv')
z=norm.ppf(1-0.05/2/2)+norm.ppf(0.8)
rows=[]
for delta in [0.05,0.075,0.10]:
    # Conservative two-arm binary variance bound, 0.25 + 0.25, for each planned contrast.
    per_arm=math.ceil(0.5*z*z/(delta*delta))
    for retention in [0.60,0.80,0.90]:
        randomized_per_arm=math.ceil(per_arm/retention)
        rows.append(dict(risk_difference=delta,retention=retention,observed_per_arm=per_arm,
                         observed_total=3*per_arm,assigned_per_arm=randomized_per_arm,assigned_total=3*randomized_per_arm,
                         power=0.8,familywise_alpha=0.05,contrasts=2))
pd.DataFrame(rows).to_csv(OUT/'02_power_scenarios.csv',index=False)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
receipt=dict(timestamp=datetime.now().astimezone().isoformat(),code={str(HERE):sha(HERE)},
             environment=dict(numpy=np.__version__,pandas=pd.__version__,scipy=scipy.__version__),
             outputs={p.name:sha(p) for p in OUT.glob('*.csv')},
             checks=['Complete synthetic blocks contain two assignments per arm.','Identical seed reproduces assignments.','Synthetic strata are balanced.'],
             not_done=['No actual participant assignment, survey-platform upload, model call, recruitment, payment, or follow-up.'])
(OUT/'receipt_experiment_dryrun_261101.json').write_text(json.dumps(receipt,indent=2))
print(pd.DataFrame(rows).query('risk_difference == 0.10').to_string(index=False))
