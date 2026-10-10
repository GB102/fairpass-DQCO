"""Step 1: fetch and cache public data (SatNOGS Network/DB). Everything is cached in data/cache/ and
each completed query is written to data/raw/obs/<period>_<station>.json, so partial runs are usable.

The SatNOGS API allows anonymous clients roughly 60 requests/hour, so the pull is deliberately narrow:
10 geographically spread European stations, fetched in priority order.

Periods (UTC):
  day    2026-10-01 .. 2026-10-02   retrospective scheduling day (+ propagation validation)
  test   2026-09-26 .. 2026-09-30   held-out evaluation of the success model (time split)
  train  2026-09-19 .. 2026-09-24   fit the success model (M0EYT: 09-13 .. 09-24 from an earlier cached pull)
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fairpass import data_sources as ds  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OBS = RAW / "obs"
OBS.mkdir(parents=True, exist_ok=True)

# UK, NL, AT, DE, BG, ES, GR, SE, SK, TR - Online stations with long records and UHF antennas
STATIONS = [91, 2173, 2410, 37, 4460, 342, 26, 858, 2138, 1616]
PERIODS = [
    ("day", "2026-10-01T00:00:00Z", "2026-10-02T00:00:00Z"),
    ("test", "2026-09-26T00:00:00Z", "2026-09-30T00:00:00Z"),
    ("train", "2026-09-19T00:00:00Z", "2026-09-24T00:00:00Z"),
]

KEEP = ["id", "start", "end", "ground_station", "norad_cat_id", "status", "waterfall_status",
        "vetted_status", "max_altitude", "rise_azimuth", "set_azimuth", "transmitter_uuid",
        "transmitter_mode", "transmitter_baud", "transmitter_downlink_low", "tle0", "tle1", "tle2",
        "tle_source", "station_lat", "station_lng", "station_alt", "observation_frequency"]


def slim(o):
    r = {k: o.get(k) for k in KEEP}
    r["n_demod"] = len(o.get("demoddata") or [])
    return r


def main():
    if not (RAW / "satnogs_stations.json").exists():
        (RAW / "satnogs_stations.json").write_text(json.dumps(ds.stations()), encoding="utf-8")
    if not (RAW / "satnogs_db_transmitters.json").exists():
        (RAW / "satnogs_db_transmitters.json").write_text(json.dumps(ds.db_transmitters()), encoding="utf-8")
        (RAW / "satnogs_db_satellites.json").write_text(json.dumps(ds.db_satellites()), encoding="utf-8")
    probe = f"{ds.NETWORK}/observations/?format=json&start={PERIODS[0][1]}&end={PERIODS[0][2]}&ground_station=91"
    print("page_size probe ->", ds.probe_page_size(probe), flush=True)
    for name, a, b in PERIODS:
        for gs in STATIONS:
            out = OBS / f"{name}_{gs}.json"
            if out.exists():
                continue
            obs = ds.observations(start=a, end=b, ground_station=gs)
            out.write_text(json.dumps([slim(o) for o in obs]), encoding="utf-8")
            print(name, gs, len(obs), flush=True)


if __name__ == "__main__":
    main()
