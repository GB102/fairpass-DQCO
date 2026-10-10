"""Step 4: build the retrospective network-day instance (2026-10-01 UTC, 10 SatNOGS stations).

* satellites: every satellite the 10 stations observed in the train/test/day periods; TLE = the most recent
  one stored with an observation at or before the planning day (all epochs within days of the horizon)
* transmitter: the satellite's most-observed (mode, baud) in those records
* passes: SGP4 over 24 h above each station's *effective* horizon (inferred from its own observations)
* p(success): fitted full model (results/link_model.json), plus 40 bootstrap replicates
* actual schedule: the observations the stations really ran that day, matched to candidate passes,
  with their real outcomes (good/bad/failed/unknown)
Output: data/processed/instance_2026-10-01.json
"""
import collections
import glob
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fairpass import MODEL_VERSION  # noqa: E402
from fairpass.linkmodel import SuccessModel  # noqa: E402
from fairpass.passes import find_passes, parse_utc, satrec  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
T0 = datetime(2026, 10, 1, tzinfo=timezone.utc)
HOURS = 24.0
TAU = 60.0
MAX_TLE_AGE_H = 14 * 24
# Value proxy uses min(listed baud, RATE_CAP): typical SatNOGS receive chains decode <= 9600 bit/s, and for
# LoRa the listed "baud" is not a payload bit rate. The uncapped listed value is kept as a sensitivity case.
RATE_CAP = 9600.0


def load(pattern):
    rows, seen = [], set()
    for p in sorted(glob.glob(str(ROOT / "data/raw/obs" / pattern))):
        for r in json.load(open(p, encoding="utf-8")):
            if r["id"] not in seen:
                seen.add(r["id"])
                rows.append(r)
    return rows


def tle_epoch(l1):
    yy, doy = int(l1[18:20]), float(l1[20:32])
    year = 2000 + yy if yy < 57 else 1900 + yy
    return datetime(year, 1, 1, tzinfo=timezone.utc).timestamp() + (doy - 1) * 86400


def main():
    sys.path.insert(0, str(ROOT / "scripts"))
    from importlib import import_module
    stations_cfg = import_module("01_fetch_data").STATIONS
    vp = import_module("02_validate_passes")

    allobs = load("train_*.json") + load("test_*.json") + load("day_*.json")
    day = [r for r in load("day_*.json")]
    st_meta = {s["id"]: s for s in json.load(open(ROOT / "data/raw/satnogs_stations.json", encoding="utf-8"))}
    eff_h = vp.effective_horizons([r for r in allobs if r.get("tle1")])
    stations = [g for g in stations_cfg if g in eff_h]

    # TLE per satellite: latest epoch not after the end of the planning day
    t_end = T0.timestamp() + HOURS * 3600
    tle, txc = {}, collections.defaultdict(collections.Counter)
    for r in allobs:
        if not r.get("tle1"):
            continue
        e = tle_epoch(r["tle1"])
        s = r["norad_cat_id"]
        if e <= t_end and (s not in tle or e > tle[s][2]):  # latest usable epoch
            tle[s] = (r["tle1"], r["tle2"], e)
        w = 3 if r["status"] == "good" else 1
        txc[s][(r["transmitter_mode"], r["transmitter_baud"], r.get("transmitter_downlink_low"))] += w
    model = json.load(open(ROOT / "results/link_model.json"))
    full = SuccessModel.from_dict(model)
    boots = [SuccessModel.from_dict(b) for b in model["bootstrap"]]
    gpath = ROOT / "results/link_model_geom.json"
    geom = SuccessModel.from_dict(json.load(open(gpath))) if gpath.exists() else None

    def hears(g, f):
        """Station g has an antenna whose range covers downlink frequency f (Hz)."""
        if not f:
            return True
        return any((a.get("frequency") or 0) <= f <= (a.get("frequency_max") or 0) for a in st_meta[g]["antenna"])

    rows, band_skipped, stale = [], 0, []
    for s, (l1, l2, ep) in sorted(tle.items()):
        if (T0.timestamp() - ep) / 3600 > MAX_TLE_AGE_H:  # SGP4 error grows ~1-3 km/day: drop stale elements
            stale.append(int(s))
            continue
        mode, baud, f_dl = txc[s].most_common(1)[0][0]
        if not baud or baud <= 0:
            continue
        sat = satrec(l1, l2)
        for g in stations:
            m = st_meta[g]
            if not hears(g, f_dl):
                band_skipped += 1
                continue
            for p in find_passes(sat, s, g, m["lat"], m["lng"], m["altitude"] or 0, T0, HOURS,
                                 horizon_deg=eff_h[g], step_s=10.0):
                rows.append({"sat": s, "station": g, "aos": (p.aos - T0).total_seconds(),
                             "los": (p.los - T0).total_seconds(), "max_el": p.max_el, "rate": min(float(baud), RATE_CAP), "baud_listed": float(baud),
                             "mode": mode, "tle_age_h": (T0.timestamp() - ep) / 3600})
    el = np.array([r["max_el"] for r in rows])
    md = [r["mode"] for r in rows]
    stl = [r["station"] for r in rows]
    sal = [r["sat"] for r in rows]
    p = full.predict(el, md, stl, sal)
    pb = np.array([b.predict(el, md, stl, sal) for b in boots])
    seen = set(full.sats)

    # actual schedule that day: match each observation to the candidate pass (same sat+station, overlapping)
    idx_by = collections.defaultdict(list)
    for k, r in enumerate(rows):
        idx_by[(r["sat"], r["station"])].append(k)
    actual, outcome, unmatched = {}, {}, 0
    for o in day:
        if o["ground_station"] not in stations:
            continue
        s, e = (parse_utc(o["start"]) - T0).total_seconds(), (parse_utc(o["end"]) - T0).total_seconds()
        hit = [k for k in idx_by.get((o["norad_cat_id"], o["ground_station"]), [])
               if rows[k]["aos"] < e and s < rows[k]["los"]]
        if hit:
            actual[hit[0]] = o["id"]
            outcome[hit[0]] = o["status"]
        else:
            unmatched += 1

    inst = {
        "model_version": MODEL_VERSION, "t0": T0.isoformat(), "hours": HOURS, "tau_s": TAU,
        "stations": [{"id": g, "name": st_meta[g]["name"], "lat": st_meta[g]["lat"], "lng": st_meta[g]["lng"],
                      "effective_horizon_deg": eff_h[g], "config_min_horizon_deg": st_meta[g]["min_horizon"]}
                     for g in stations],
        "contacts": {
            "sat": sal, "station": stl, "aos": [r["aos"] for r in rows], "los": [r["los"] for r in rows],
            "max_el": el.tolist(), "rate": [r["rate"] for r in rows], "p": p.tolist(),
            "baud_listed": [r["baud_listed"] for r in rows],
            "p_geom": (geom.predict(el, md, stl, sal) if geom else p).tolist(),
            "weight": [1.0] * len(rows), "mode": md, "tle_age_h": [r["tle_age_h"] for r in rows],
            "sat_seen_in_training": [s in seen for s in sal],
        },
        "p_bootstrap": pb.round(5).tolist(),
        "actual": {"index": list(actual.keys()), "obs_id": list(actual.values()),
                   "status": [outcome[k] for k in actual], "unmatched_observations": unmatched},
        "band_pairs_skipped": band_skipped, "stale_tle_satellites_dropped": stale,
        "provenance": {"tle_source": "TLEs stored with SatNOGS observations (CC BY-SA 4.0)",
                       "success_model": "results/link_model.json", "weights": "all 1.0 (no mission priorities "
                       "are public; weights are a user input)"},
    }
    out = ROOT / "data/processed/instance_2026-10-01.json"
    out.write_text(json.dumps(inst))
    n_sat = len(set(sal))
    print(f"stations {len(stations)}  satellites {n_sat}  candidate passes {len(rows)}  "
          f"(satellite-station pairs skipped for antenna band: {band_skipped})")
    print(f"satellites dropped for TLE older than 14 days: {len(stale)}")
    print(f"actual observations matched {len(actual)}  unmatched {unmatched}")
    print("TLE age (h) median %.1f max %.1f" % (np.median([r['tle_age_h'] for r in rows]),
                                               np.max([r['tle_age_h'] for r in rows])))
    print("p distribution:", np.percentile(p, [10, 25, 50, 75, 90]).round(3))


if __name__ == "__main__":
    main()
