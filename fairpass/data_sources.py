"""Polite, cached access to public data sources.

* SatNOGS Network API (stations, observations)  - CC BY-SA 4.0, Libre Space Foundation
* SatNOGS DB API (satellites, transmitters)      - CC BY-SA 4.0, Libre Space Foundation
* CelesTrak GP data (TLEs)                       - CelesTrak usage policy: one download per update

Every response is cached under data/cache/ keyed by URL, and a provenance record
(url, UTC fetch time, sha256) is appended to data/cache/provenance.jsonl.
"""
from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "cache"
CACHE.mkdir(parents=True, exist_ok=True)
PROVENANCE = CACHE / "provenance.jsonl"

NETWORK = "https://network.satnogs.org/api"
DB = "https://db.satnogs.org/api"
CELESTRAK = "https://celestrak.org/NORAD/elements/gp.php"

_session = requests.Session()
_session.headers["User-Agent"] = "FairPass-hackathon-research/0.1 (Qollab x IonQ 2026)"


def _key(url: str) -> Path:
    return CACHE / (hashlib.sha1(url.encode()).hexdigest() + ".json")


def _log(url: str, body: bytes) -> None:
    rec = {
        "url": url,
        "fetched_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sha256": hashlib.sha256(body).hexdigest(),
        "bytes": len(body),
    }
    with PROVENANCE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")


def get_json(url: str, *, pause: float = 1.0, retries: int = 6):
    """GET a JSON document with caching. Returns (data, next_url).

    The SatNOGS API throttles anonymous clients (~60 requests/hour); on HTTP 429 we sleep for the full
    Retry-After period instead of hammering the server."""
    path = _key(url)
    if path.exists():
        cached = json.loads(path.read_text(encoding="utf-8"))
        return cached["data"], cached.get("next")
    for attempt in range(retries):
        r = _session.get(url, timeout=90)
        if r.status_code == 429:
            wait = float(r.headers.get("Retry-After", 60)) + 5
            print(f"  throttled; sleeping {wait:.0f}s", flush=True)
            time.sleep(min(wait, 3700))
            continue
        if r.status_code >= 500:
            time.sleep(10 * (attempt + 1))
            continue
        r.raise_for_status()
        nxt = r.links.get("next", {}).get("url")
        data = r.json()
        path.write_text(json.dumps({"url": url, "next": nxt, "data": data}), encoding="utf-8")
        _log(url, r.content)
        time.sleep(pause)
        return data, nxt
    raise RuntimeError(f"giving up on {url}")


def get_text(url: str, *, pause: float = 1.0) -> str:
    """GET plain text once (CelesTrak asks for one download per update cycle)."""
    path = CACHE / (hashlib.sha1(url.encode()).hexdigest() + ".txt")
    if path.exists():
        return path.read_text(encoding="utf-8")
    r = _session.get(url, timeout=90)
    if r.status_code != 200:  # CelesTrak policy: stop on any non-200
        raise RuntimeError(f"HTTP {r.status_code} for {url}; stopping as CelesTrak policy requires")
    path.write_text(r.text, encoding="utf-8")
    _log(url, r.content)
    time.sleep(pause)
    return r.text


def paginate(url: str, *, max_pages: int = 10_000, pause: float = 1.0) -> list:
    out, nxt, n = [], url, 0
    while nxt and n < max_pages:
        data, nxt = get_json(nxt, pause=pause)
        out.extend(data if isinstance(data, list) else data.get("results", []))
        n += 1
    return out


def stations() -> list:
    data, _ = get_json(f"{NETWORK}/stations/?format=json")
    return data


PAGE_SIZE = None  # set by probe_page_size() if the API honours a larger page


def probe_page_size(url_base: str, size: int = 100) -> int | None:
    """One request: does the observations endpoint honour ?page_size=? Returns the size if it does."""
    global PAGE_SIZE
    data, _ = get_json(f"{url_base}&page_size={size}")
    if isinstance(data, list) and len(data) > 25:
        PAGE_SIZE = size
    return PAGE_SIZE


def observations(*, start: str, end: str, ground_station: int | None = None,
                 status: str | None = None, norad: int | None = None) -> list:
    q = f"{NETWORK}/observations/?format=json&start={start}&end={end}"
    if PAGE_SIZE:
        q += f"&page_size={PAGE_SIZE}"
    if ground_station is not None:
        q += f"&ground_station={ground_station}"
    if status is not None:
        q += f"&status={status}"
    if norad is not None:
        q += f"&satellite__norad_cat_id={norad}"
    return paginate(q)


def db_satellites() -> list:
    data, _ = get_json(f"{DB}/satellites/?format=json")
    return data


def db_transmitters() -> list:
    data, _ = get_json(f"{DB}/transmitters/?format=json")
    return data


def celestrak_group(group: str) -> str:
    return get_text(f"{CELESTRAK}?GROUP={group}&FORMAT=TLE")
