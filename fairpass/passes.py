"""SGP4 propagation and pass (visibility window) computation.

TEME -> ECEF uses a GMST (IAU-82) rotation and ignores polar motion and the equation of the
equinoxes; the induced pointing error is well below 0.1 deg for LEO, far below the 1 deg
granularity of SatNOGS's reported max elevation.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import numpy as np
from sgp4.api import Satrec, jday

WGS84_A = 6378.137
WGS84_F = 1 / 298.257223563
WGS84_E2 = WGS84_F * (2 - WGS84_F)


def gmst_rad(jd_ut1: np.ndarray) -> np.ndarray:
    """Greenwich mean sidereal time (IAU-82, as used with SGP4/TEME), radians."""
    t = (jd_ut1 - 2451545.0) / 36525.0
    g = 67310.54841 + (876600.0 * 3600 + 8640184.812866) * t + 0.093104 * t**2 - 6.2e-6 * t**3
    return np.mod(np.deg2rad(g / 240.0), 2 * np.pi)


def station_ecef(lat_deg: float, lon_deg: float, alt_m: float) -> np.ndarray:
    lat, lon = np.deg2rad(lat_deg), np.deg2rad(lon_deg)
    n = WGS84_A / np.sqrt(1 - WGS84_E2 * np.sin(lat) ** 2)
    h = alt_m / 1000.0
    return np.array([(n + h) * np.cos(lat) * np.cos(lon),
                     (n + h) * np.cos(lat) * np.sin(lon),
                     (n * (1 - WGS84_E2) + h) * np.sin(lat)])


def enu_matrix(lat_deg: float, lon_deg: float) -> np.ndarray:
    lat, lon = np.deg2rad(lat_deg), np.deg2rad(lon_deg)
    return np.array([[-np.sin(lon), np.cos(lon), 0.0],
                     [-np.sin(lat) * np.cos(lon), -np.sin(lat) * np.sin(lon), np.cos(lat)],
                     [np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)]])


def time_grid(t0: datetime, hours: float, step_s: float):
    n = int(round(hours * 3600 / step_s)) + 1
    secs = np.arange(n) * step_s
    jd0, fr0 = jday(t0.year, t0.month, t0.day, t0.hour, t0.minute, t0.second + t0.microsecond * 1e-6)
    fr = fr0 + secs / 86400.0
    jd = np.full(n, jd0)
    return secs, jd, fr


def ecef_positions(sat: Satrec, jd: np.ndarray, fr: np.ndarray) -> np.ndarray:
    err, r, _ = sat.sgp4_array(jd, fr)
    r = np.asarray(r, dtype=float)
    r[err != 0] = np.nan
    th = gmst_rad(jd + fr)
    c, s = np.cos(th), np.sin(th)
    x = c * r[:, 0] + s * r[:, 1]
    y = -s * r[:, 0] + c * r[:, 1]
    return np.stack([x, y, r[:, 2]], axis=1)


def look_angles(r_ecef: np.ndarray, lat: float, lon: float, alt_m: float):
    """Elevation and azimuth (deg) and slant range (km) of positions seen from a station."""
    d = r_ecef - station_ecef(lat, lon, alt_m)
    enu = d @ enu_matrix(lat, lon).T
    rng = np.linalg.norm(enu, axis=1)
    el = np.rad2deg(np.arcsin(enu[:, 2] / rng))
    az = np.mod(np.rad2deg(np.arctan2(enu[:, 0], enu[:, 1])), 360)
    return el, az, rng


@dataclass
class Pass:
    norad: int
    station: int
    aos: datetime
    los: datetime
    tca: datetime
    max_el: float
    aos_az: float
    los_az: float

    @property
    def duration_s(self) -> float:
        return (self.los - self.aos).total_seconds()


def _crossing(t: np.ndarray, el: np.ndarray, i: int, horizon: float) -> float:
    """Linear interpolation of the horizon crossing between samples i and i+1."""
    e0, e1 = el[i] - horizon, el[i + 1] - horizon
    if e1 == e0:
        return float(t[i])
    return float(t[i] + (t[i + 1] - t[i]) * (-e0) / (e1 - e0))


def find_passes(sat: Satrec, norad: int, station_id: int, lat: float, lon: float, alt_m: float,
                t0: datetime, hours: float, horizon_deg: float = 0.0, step_s: float = 10.0,
                min_culmination: float = 0.0) -> list[Pass]:
    secs, jd, fr = time_grid(t0, hours, step_s)
    el, az, _ = look_angles(ecef_positions(sat, jd, fr), lat, lon, alt_m)
    up = el > horizon_deg
    out: list[Pass] = []
    if not up.any():
        return out
    d = np.diff(up.astype(int))
    rises = np.flatnonzero(d == 1)   # up[i] False -> up[i+1] True
    sets = np.flatnonzero(d == -1)   # up[j] True  -> up[j+1] False
    for i in rises:  # passes truncated by the edge of the time grid are dropped
        later = sets[sets > i]
        if not len(later):
            break
        j = later[0]
        a, b = i + 1, j
        ta = _crossing(secs, el, i, horizon_deg)
        tb = _crossing(secs, el, j, horizon_deg)
        k = a + int(np.nanargmax(el[a:b + 1]))
        # parabolic refinement of culmination
        if 0 < k < len(el) - 1:
            y0, y1, y2 = el[k - 1], el[k], el[k + 1]
            den = y0 - 2 * y1 + y2
            dk = 0.5 * (y0 - y2) / den if den != 0 else 0.0
            mx = y1 - 0.25 * (y0 - y2) * dk
            tk = secs[k] + dk * step_s
        else:
            mx, tk = el[k], secs[k]
        if mx < min_culmination:
            continue
        out.append(Pass(norad, station_id, t0 + timedelta(seconds=ta), t0 + timedelta(seconds=tb),
                        t0 + timedelta(seconds=float(tk)), float(mx), float(az[a]), float(az[b])))
    return out


def satrec(l1: str, l2: str) -> Satrec:
    return Satrec.twoline2rv(l1.strip(), l2.strip())


def parse_utc(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(timezone.utc)
