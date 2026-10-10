"""Pass-success model fitted on vetted SatNOGS observations.

P(good | pass) = sigmoid( b0 + f(max_el) + beta_mode + u_station + v_satellite )

* f(max_el): quadratic in s = sin(max_el) (smooth, monotone over the data range in practice)
* beta_mode: fixed effect per modulation family
* u_station, v_satellite: L2-shrunk ("random") effects; unseen levels fall back to 0

Labels: status 'good' -> 1, 'bad' -> 0. 'failed' (station/software failure) and 'unknown'
(not vetted) are excluded from the link model; station failure rate is estimated separately.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize

MODES = ["FSK", "GMSK", "GFSK", "MSK", "BPSK", "AFSK", "CW", "FM", "LORA", "OTHER"]


def mode_family(m: str | None) -> str:
    m = (m or "").upper()
    for k in ["GMSK", "GFSK", "AFSK", "BPSK", "LORA", "MSK", "FSK", "CW"]:
        if k in m:
            return k
    if m.startswith("FM"):
        return "FM"
    return "OTHER"


def _sigmoid(z):
    return 0.5 * (1 + np.tanh(0.5 * z))


@dataclass
class SuccessModel:
    kind: str = "full"           # const | geom | sat | full
    l2_station: float = 2.0
    l2_sat: float = 1.0
    params: np.ndarray | None = None
    stations: list = field(default_factory=list)
    sats: list = field(default_factory=list)

    # ---- design ---------------------------------------------------------
    def _design(self, el, mode, st, sat):
        el = np.asarray(el, float)
        s = np.sin(np.deg2rad(np.clip(el, 0, 90)))
        n = len(el)
        cols = [np.ones(n)]
        if self.kind in ("geom", "full"):
            cols += [s - 0.5, (s - 0.5) ** 2]
        if self.kind == "full":
            fam = np.array([mode_family(m) for m in mode])
            cols += [(fam == m).astype(float) for m in MODES[1:]]
        X = np.stack(cols, 1)
        si = np.array([self._sidx.get(x, -1) for x in st]) if self.kind == "full" else None
        vi = np.array([self._vidx.get(x, -1) for x in sat]) if self.kind in ("sat", "full") else None
        return X, si, vi

    def _logit(self, theta, X, si, vi):
        k = X.shape[1]
        z = X @ theta[:k]
        off = k
        if si is not None:
            u = np.concatenate([theta[off:off + len(self.stations)], [0.0]])
            z = z + u[si]
            off += len(self.stations)
        if vi is not None:
            v = np.concatenate([theta[off:off + len(self.sats)], [0.0]])
            z = z + v[vi]
        return z

    def fit(self, el, mode, st, sat, y):
        self.stations = sorted(set(st)) if self.kind == "full" else []
        self.sats = sorted(set(sat)) if self.kind in ("sat", "full") else []
        self._sidx = {x: i for i, x in enumerate(self.stations)}
        self._vidx = {x: i for i, x in enumerate(self.sats)}
        X, si, vi = self._design(el, mode, st, sat)
        y = np.asarray(y, float)
        k, ns, nv = X.shape[1], len(self.stations), len(self.sats)

        def loss(theta):
            z = self._logit(theta, X, si, vi)
            p = _sigmoid(z)
            nll = np.sum(np.logaddexp(0, z) - y * z)
            reg = 0.5 * self.l2_station * np.sum(theta[k:k + ns] ** 2) \
                + 0.5 * self.l2_sat * np.sum(theta[k + ns:] ** 2) + 1e-4 * np.sum(theta[:k] ** 2)
            g_lin = p - y
            grad = np.zeros_like(theta)
            grad[:k] = X.T @ g_lin + 2e-4 * theta[:k]
            if ns:
                grad[k:k + ns] = np.bincount(si, g_lin, minlength=ns + 1)[:ns] \
                    + self.l2_station * theta[k:k + ns]
            if nv:
                grad[k + ns:] = np.bincount(vi, g_lin, minlength=nv + 1)[:nv] \
                    + self.l2_sat * theta[k + ns:]
            return nll + reg, grad

        theta0 = np.zeros(k + ns + nv)
        theta0[0] = np.log((y.mean() + 1e-3) / (1 - y.mean() + 1e-3))
        res = minimize(loss, theta0, jac=True, method="L-BFGS-B", options={"maxiter": 2000})
        self.params = res.x
        return self

    def predict(self, el, mode, st, sat):
        X, si, vi = self._design(el, mode, st, sat)
        return _sigmoid(self._logit(self.params, X, si, vi))

    def to_dict(self):
        return {"kind": self.kind, "params": self.params.tolist(), "stations": self.stations,
                "sats": self.sats, "l2_station": self.l2_station, "l2_sat": self.l2_sat, "modes": MODES}

    @classmethod
    def from_dict(cls, d):
        m = cls(kind=d["kind"], l2_station=d["l2_station"], l2_sat=d["l2_sat"])
        m.params = np.array(d["params"])
        m.stations, m.sats = d["stations"], d["sats"]
        m._sidx = {x: i for i, x in enumerate(m.stations)}
        m._vidx = {x: i for i, x in enumerate(m.sats)}
        return m


# ---- metrics ------------------------------------------------------------------
def log_loss(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def brier(y, p):
    return float(np.mean((p - y) ** 2))


def auc(y, p):
    y = np.asarray(y)
    order = np.argsort(p)
    ranks = np.empty(len(p))
    ranks[order] = np.arange(1, len(p) + 1)
    # average ties
    _, inv, cnt = np.unique(p, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, ranks)
    ranks = sums[inv] / cnt[inv]
    n1 = y.sum()
    n0 = len(y) - n1
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
