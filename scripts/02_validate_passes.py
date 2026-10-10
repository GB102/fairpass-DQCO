"""Step 2: reproduce SatNOGS's own pass geometry from the TLE stored with each observation.

For every observation we propagate its TLE (SGP4) from the station's coordinates and compare:
  * max elevation of the pass  vs  the network's recorded `max_altitude` (integer degrees)
  * rise/set crossing of the station horizon mask  vs  the observation start/end
Output: results/pass_validation.json and per-observation CSV.
"""
import csv
import glob
import json
import sys
from datetime import timedelta
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fairpass.passes import (ecef_positions, find_passes, look_angles, parse_utc, satrec,  # noqa: E402
                             time_grid)

ROOT = Path(__file__).resolve().parents[1]


def load_obs(pattern):
    rows = []
    for p in sorted(glob.glob(str(ROOT / "data" / "raw" / "obs" / pattern))):
        rows += json.load(open(p, encoding="utf-8"))
    seen, out = set(), []
    for r in rows:
        if r["id"] not in seen:
            seen.add(r["id"])
            out.append(r)
    return out


def elev_at(sat, o, key):
    _, jd, fr = time_grid(parse_utc(o[key]), 0, 1)
    el, _, _ = look_angles(ecef_positions(sat, jd, fr), o["station_lat"], o["station_lng"], o["station_alt"] or 0)
    return float(el[0])


def effective_horizons(obs):
    """Median satellite elevation at observation start, per station: the horizon the scheduler really used
    (the live station config can differ from the value in force when the observations were scheduled)."""
    acc = {}
    for o in obs:
        try:
            sat = satrec(o["tle1"], o["tle2"])
        except Exception:
            continue
        acc.setdefault(o["ground_station"], []).append(elev_at(sat, o, "start"))
    return {gs: float(np.median(v)) for gs, v in acc.items()}


def main(pattern="*.json", limit=None):
    stations = {s["id"]: s for s in json.load(open(ROOT / "data/raw/satnogs_stations.json", encoding="utf-8"))}
    obs = [o for o in load_obs(pattern) if o.get("tle1") and o.get("max_altitude") is not None]
    if limit:
        obs = obs[:limit]
    eff = effective_horizons(obs)
    recs = []
    for o in obs:
        st = stations.get(o["ground_station"], {})
        hor = eff.get(o["ground_station"], float(st.get("min_horizon") or 0))
        t0 = parse_utc(o["start"]) - timedelta(minutes=25)
        try:
            sat = satrec(o["tle1"], o["tle2"])
        except Exception:
            continue
        ps = find_passes(sat, o["norad_cat_id"], o["ground_station"], o["station_lat"], o["station_lng"],
                         o["station_alt"] or 0, t0, hours=1.2, horizon_deg=0.0, step_s=5.0)
        s, e = parse_utc(o["start"]), parse_utc(o["end"])
        mid = s + (e - s) / 2
        best = None
        for p in ps:
            if p.aos <= mid <= p.los:
                best = p
        if best is None:
            recs.append({"id": o["id"], "matched": 0})
            continue
        # horizon-mask crossings for start/end comparison
        ph = find_passes(sat, o["norad_cat_id"], o["ground_station"], o["station_lat"], o["station_lng"],
                         o["station_alt"] or 0, t0, hours=1.2, horizon_deg=hor, step_s=5.0)
        ph = [p for p in ph if p.aos <= mid <= p.los]
        d_start = (ph[0].aos - s).total_seconds() if ph else np.nan
        d_end = (ph[0].los - e).total_seconds() if ph else np.nan
        recs.append({"id": o["id"], "matched": 1, "station": o["ground_station"], "norad": o["norad_cat_id"],
                     "recorded_max_el": o["max_altitude"], "computed_max_el": round(best.max_el, 3),
                     "d_max_el": best.max_el - o["max_altitude"], "horizon": hor,
                     "d_start_s": d_start, "d_end_s": d_end, "obs_dur_s": (e - s).total_seconds()})
    m = [r for r in recs if r["matched"]]
    d = np.array([r["d_max_el"] for r in m])
    ds_ = np.array([r["d_start_s"] for r in m], float)
    de_ = np.array([r["d_end_s"] for r in m], float)
    summary = {
        "n_obs": len(recs), "n_matched": len(m),
        "max_el_error_deg": {"mean": float(d.mean()), "median_abs": float(np.median(np.abs(d))),
                             "p95_abs": float(np.percentile(np.abs(d), 95)),
                             "frac_within_1deg": float(np.mean(np.abs(d) <= 1.0))},
        "start_offset_s": {"median": float(np.nanmedian(ds_)), "median_abs": float(np.nanmedian(np.abs(ds_))),
                           "frac_within_30s": float(np.nanmean(np.abs(ds_) <= 30))},
        "end_offset_s": {"median": float(np.nanmedian(de_)), "median_abs": float(np.nanmedian(np.abs(de_))),
                         "frac_within_30s": float(np.nanmean(np.abs(de_) <= 30))},
        "effective_horizon_deg": {str(k): round(v, 2) for k, v in eff.items()},
        "config_min_horizon_deg": {str(k): stations.get(k, {}).get("min_horizon") for k in eff},
        "note": "recorded max_altitude is an integer; |error|<=1 deg is the resolution limit. Observation "
                "start/end can be shortened by the scheduler (e.g. to avoid overlaps), so start/end "
                "offsets are expected to be one-sided.",
    }
    out = ROOT / "results"
    (out / "pass_validation.json").write_text(json.dumps(summary, indent=1))
    with open(out / "pass_validation.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sorted({k for r in m for k in r}))
        w.writeheader()
        w.writerows(m)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main(*(sys.argv[1:2] or ["*.json"]))
