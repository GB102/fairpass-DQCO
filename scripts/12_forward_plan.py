"""Step 12: forward plan - tomorrow's fairness-aware schedule for the same 10 stations from current TLEs.

* TLEs: CelesTrak GP 'active' group, downloaded once and cached (CelesTrak policy: one download per update)
* satellites/transmitters/stations/horizons/success model: as in the retrospective instance
* horizon: PLAN_START (UTC) + 24 h
* outputs: results/forward_plan.json and results/forward_plan_<station>.csv (pass lists an operator can load)
This is a *prediction* - nothing here is validated until the day has passed and its observations are vetted.
"""
import csv
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fairpass import MODEL_VERSION  # noqa: E402
from fairpass import data_sources as ds  # noqa: E402
from fairpass.classical import exact_cpsat, local_search, network_greedy, station_greedy  # noqa: E402
from fairpass.linkmodel import SuccessModel  # noqa: E402
from fairpass.passes import find_passes, satrec  # noqa: E402
from fairpass.problem import Contacts, FastEval, build_problem, schedule_metrics  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PLAN_START = datetime(2026, 10, 11, tzinfo=timezone.utc)
HOURS, LAM, K, TAU = 24.0, 200.0, 2, 60.0


def parse_tle_text(txt):
    lines = [l.rstrip() for l in txt.splitlines() if l.strip()]
    out = {}
    for i in range(len(lines) - 2):
        if lines[i + 1].startswith("1 ") and lines[i + 2].startswith("2 "):
            out[int(lines[i + 1][2:7])] = (lines[i].strip(), lines[i + 1], lines[i + 2])
    return out


def main():
    inst = json.load(open(ROOT / "data/processed/instance_2026-10-01.json"))
    c0 = inst["contacts"]
    sat_tx = {}
    for s, r, m in zip(c0["sat"], c0["rate"], c0["mode"]):
        sat_tx[s] = (r, m)
    tle = parse_tle_text(ds.celestrak_group("active"))
    st_meta = {s["id"]: s for s in json.load(open(ROOT / "data/raw/satnogs_stations.json", encoding="utf-8"))}
    stations = inst["stations"]
    model = SuccessModel.from_dict(json.load(open(ROOT / "results/link_model.json")))
    # band compatibility is inherited: only (sat, station) pairs that were candidates in the instance
    pairs = set(zip(c0["sat"], c0["station"]))
    rows = []
    missing = [s for s in sat_tx if s not in tle]
    for s, (rate, mode) in sat_tx.items():
        if s not in tle:
            continue
        sat = satrec(tle[s][1], tle[s][2])
        for st in stations:
            g = st["id"]
            if (s, g) not in pairs:
                continue
            m = st_meta[g]
            for p in find_passes(sat, s, g, m["lat"], m["lng"], m["altitude"] or 0, PLAN_START, HOURS,
                                 horizon_deg=st["effective_horizon_deg"], step_s=10.0):
                rows.append((s, g, (p.aos - PLAN_START).total_seconds(), (p.los - PLAN_START).total_seconds(),
                             p.max_el, rate, mode))
    sat_, st_, a_, l_, el_, r_, md_ = map(list, zip(*rows))
    p = model.predict(np.array(el_), md_, st_, sat_)
    c = Contacts(np.array(sat_), np.array(st_), np.array(a_), np.array(l_), np.array(el_), np.array(r_, float), p,
                 np.ones(len(rows)))
    act = [int(s) for s in np.unique(c.sat) if c.p[c.sat == s].mean() >= 0.3]
    pb = build_problem(c, lam=LAM, K=K, active=act)
    fe = FastEval(c)
    x_g = station_greedy(c, pb, key="value", tau=TAU)
    x_h = local_search(pb, network_greedy(pb))
    x, info = exact_cpsat(pb, time_limit=180, hint=x_h)
    x = local_search(pb, x)
    res = {"model_version": MODEL_VERSION, "plan_start": PLAN_START.isoformat(), "hours": HOURS, "lam": LAM, "K": K,
           "n_candidates": c.n, "n_satellites": int(len(np.unique(c.sat))), "n_active": len(act),
           "satellites_without_current_tle": missing, "solver": info,
           "plan_metrics": schedule_metrics(c, x, K=K, active=act, fast=fe),
           "uncoordinated_metrics": schedule_metrics(c, x_g, K=K, active=act, fast=fe),
           "status": "PREDICTION - not yet observable"}
    names = {s["norad_cat_id"]: s["name"] for s in
             json.load(open(ROOT / "data/raw/satnogs_db_satellites.json", encoding="utf-8")) if s.get("norad_cat_id")}
    for st in stations:
        g = st["id"]
        idx = [i for i in np.flatnonzero((c.station == g) & (x == 1))]
        idx.sort(key=lambda i: c.aos[i])
        with open(ROOT / f"results/forward_plan_{g}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["station", "norad", "satellite", "aos_utc", "los_utc", "max_el_deg", "p_success",
                        "expected_kB"])
            for i in idx:
                w.writerow([st["name"], int(c.sat[i]), names.get(int(c.sat[i]), ""),
                            (PLAN_START + timedelta(seconds=float(c.aos[i]))).isoformat(timespec="seconds"),
                            (PLAN_START + timedelta(seconds=float(c.los[i]))).isoformat(timespec="seconds"),
                            round(float(c.max_el[i]), 1), round(float(c.p[i]), 3), round(float(c.value_kB()[i]), 1)])
    (ROOT / "results/forward_plan.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ("n_candidates", "n_satellites", "n_active", "plan_metrics",
                                          "uncoordinated_metrics")}, indent=1))


if __name__ == "__main__":
    main()
