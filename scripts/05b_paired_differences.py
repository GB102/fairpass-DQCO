"""Step 5b: paired bootstrap differences between schedules (same success-model replicate for both schedules),
which is the right uncertainty for 'is A better than B' - the absolute intervals in schedule_study.json are wide
because a few high-rate satellites dominate the absolute data volume."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fairpass.problem import Contacts, FastEval, schedule_metrics  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
inst = json.load(open(ROOT / "data/processed/instance_2026-10-01.json"))
ss = json.load(open(ROOT / "results/schedule_study.json"))
c = Contacts.from_dict(inst["contacts"])
pb = np.array(inst["p_bootstrap"])
act = [int(s) for s in np.unique(c.sat) if c.p[c.sat == s].mean() >= 0.3]
fe = FastEval(c)
X = {}
for k, idx in ss["x"].items():
    x = np.zeros(c.n, int)
    x[idx] = 1
    X[k] = x
pairs = [("cpsat", "station_value"), ("cpsat", "station_maxel"), ("cpsat", "network_greedy"),
         ("cpsat_fair_200", "cpsat"), ("cpsat_fair_800", "cpsat"), ("cpsat_fair_200", "station_value")]
out = {}
for a, b in pairs:
    d_kb, d_st = [], []
    for p in pb:
        ma = schedule_metrics(c, X[a], active=act, fast=fe, p=p)
        mb = schedule_metrics(c, X[b], active=act, fast=fe, p=p)
        d_kb.append((ma["expected_unique_kB"] - mb["expected_unique_kB"]) / 1000)
        d_st.append(ma["starved"] - mb["starved"])
    rel = [d / (schedule_metrics(c, X[b], active=act, fast=fe, p=p)["expected_unique_kB"] / 1000) for d, p in zip(d_kb, pb)]
    out[f"{a} - {b}"] = {"dMB_mean": float(np.mean(d_kb)), "dMB_p05": float(np.percentile(d_kb, 5)),
                         "dMB_p95": float(np.percentile(d_kb, 95)), "rel_mean": float(np.mean(rel)),
                         "rel_p05": float(np.percentile(rel, 5)), "rel_p95": float(np.percentile(rel, 95)),
                         "d_starved_nominal": ss["schedules"][a]["starved"] - ss["schedules"][b]["starved"]}
(ROOT / "results/paired_differences.json").write_text(json.dumps(out, indent=1))
for k, v in out.items():
    print(f"{k:32s} dMB {v['dMB_mean']:8.1f} [{v['dMB_p05']:.1f}, {v['dMB_p95']:.1f}]  rel {100*v['rel_mean']:+.1f}% "
          f"[{100*v['rel_p05']:+.1f}, {100*v['rel_p95']:+.1f}]  d_starved {v['d_starved_nominal']}")
