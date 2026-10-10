"""Candidate contacts -> quadratic scheduling objective (QUBO with hard pairs).

Objective (maximise):
    f(x) = sum_i h_i x_i + sum_{i<j} J_ij x_i x_j + sum_s lam * E[g(n_s)]
    subject to  x_i + x_j <= 1 for (i,j) in H

Terms
  value        h_i = v_i = w_s * p_i * rate_s * dur_i / 8000          (expected delivered data, kB)
  redundancy   J_ij = -w_s * rate_s * overlap_ij * p_i * p_j / 8000   (same satellite seen by two stations at
               once, broadcast mode) -> exact expected *unique* data for pairs; a lower bound (Bonferroni)
               when three or more stations overlap
  fairness     g(n) = n - n(n-1)/(2K) is concave (marginal gains 1, 1-1/K, 1-2/K, ...).  With independent
               pass successes n_s = sum_i Bernoulli(p_i x_i):
                   E[g(n_s)] = sum_i p_i x_i - (1/K) sum_{i<j in s} p_i p_j x_i x_j
               so dead satellites (p ~ 0) earn no fairness credit. Only "active missions" are grouped.
  hard pairs   same station and intervals closer than the re-pointing gap tau
               (+ same satellite overlapping, in 'commanded' downlink mode)
For QUBO solvers the hard pairs become a penalty P*x_i*x_j with P > max_i h_i, which makes every optimum of
the penalised QUBO feasible: all other couplings are non-positive, so removing one endpoint of a violated
pair gains at least P - h_i > 0.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Contacts:
    sat: np.ndarray        # norad id
    station: np.ndarray    # station id
    aos: np.ndarray        # seconds from t0
    los: np.ndarray
    max_el: np.ndarray
    rate: np.ndarray       # baud (bit/s)
    p: np.ndarray          # success probability
    weight: np.ndarray     # mission priority w_s

    FIELDS = ("sat", "station", "aos", "los", "max_el", "rate", "p", "weight")

    @property
    def n(self):
        return len(self.sat)

    @property
    def dur(self):
        return self.los - self.aos

    def value_kB(self):
        return self.weight * self.p * self.rate * self.dur / 8000.0

    def subset(self, idx):
        idx = np.asarray(idx)
        return Contacts(*(getattr(self, f)[idx] for f in self.FIELDS))

    def to_dict(self):
        return {f: np.asarray(getattr(self, f)).tolist() for f in self.FIELDS}

    @classmethod
    def from_dict(cls, d):
        return cls(*(np.asarray(d[f]) for f in cls.FIELDS))


@dataclass
class Problem:
    h: np.ndarray
    J: dict                    # (i,j) i<j -> coupling (non-positive)
    hard: set                  # {(i,j)} i<j
    groups: list = field(default_factory=list)  # fairness: [(idx array, p array)], scaled by lam/K
    lam: float = 0.0
    K: int = 2
    meta: dict = field(default_factory=dict)

    @property
    def n(self):
        return len(self.h)

    def objective(self, X):
        """f(x) for one (n,) or many (m,n) 0/1 vectors (ignores hard constraints)."""
        X = np.atleast_2d(X).astype(float)
        f = X @ self.h
        if self.J:
            ij = np.array(list(self.J.keys()))
            w = np.array(list(self.J.values()))
            f = f + (X[:, ij[:, 0]] * X[:, ij[:, 1]]) @ w
        if self.lam > 0:
            for idx, p in self.groups:
                s = X[:, idx] @ p
                s2 = X[:, idx] @ (p * p)
                f = f + self.lam * (s - (s * s - s2) / (2 * self.K))
        return f

    def violations(self, X):
        X = np.atleast_2d(X)
        if not self.hard:
            return np.zeros(len(X), int)
        hp = np.array(sorted(self.hard))
        return (X[:, hp[:, 0]] * X[:, hp[:, 1]]).sum(1)

    def linear_upper(self):
        """Largest possible single-variable gain (used for the penalty weight)."""
        lin = self.h.copy()
        if self.lam > 0:
            for idx, p in self.groups:
                lin[idx] += self.lam * p
        return lin

    def penalty_weight(self):
        return 1.5 * float(np.max(self.linear_upper(), initial=0.0)) + 1e-9

    def dense_J(self):
        """All pairwise couplings including the expanded fairness pairs."""
        J = dict(self.J)
        if self.lam > 0:
            for idx, p in self.groups:
                for a in range(len(idx)):
                    for b in range(a + 1, len(idx)):
                        i, j = int(min(idx[a], idx[b])), int(max(idx[a], idx[b]))
                        J[(i, j)] = J.get((i, j), 0.0) - self.lam * p[a] * p[b] / self.K
        return J

    def qubo_matrix(self, P=None):
        """Upper-triangular Q with f_pen(x) = x^T Q x (diag = linear), to MAXIMISE."""
        P = self.penalty_weight() if P is None else P
        Q = np.diag(self.linear_upper().astype(float))
        for (i, j), w in self.dense_J().items():
            Q[i, j] += w
        for (i, j) in self.hard:
            Q[i, j] -= P
        return Q

    def penalised(self, X, P=None):
        P = self.penalty_weight() if P is None else P
        return self.objective(X) - P * self.violations(X)

    def restrict(self, free, x):
        """Sub-problem over `free` variables with all others fixed to x, returned as a plain QUBO
        (fairness expanded into pairs). Free variables in hard conflict with a fixed selected variable are
        dropped (forced to 0). Returns (sub, kept_free)."""
        free = np.asarray(free)
        free_set = set(free.tolist())
        fixed_on = set(np.flatnonzero(x).tolist()) - free_set
        blocked = set()
        for (i, j) in self.hard:
            if i in fixed_on and j in free_set:
                blocked.add(j)
            if j in fixed_on and i in free_set:
                blocked.add(i)
        kept = np.array([f for f in free if f not in blocked], int)
        pos = {v: k for k, v in enumerate(kept)}
        h = self.h[kept].astype(float).copy()
        J, hard = {}, set()
        for (i, j), w in self.J.items():
            if i in pos and j in pos:
                J[(pos[i], pos[j])] = w
            elif i in pos and j in fixed_on:
                h[pos[i]] += w
            elif j in pos and i in fixed_on:
                h[pos[j]] += w
        if self.lam > 0:
            for idx, p in self.groups:
                loc = [(k, pp) for k, pp in zip(idx, p) if k in pos]
                if not loc:
                    continue
                s_fixed = sum(pp for k, pp in zip(idx, p) if k in fixed_on)
                for k, pp in loc:
                    h[pos[k]] += self.lam * pp * (1 - s_fixed / self.K)
                for a in range(len(loc)):
                    for b in range(a + 1, len(loc)):
                        i, j = pos[loc[a][0]], pos[loc[b][0]]
                        key = (min(i, j), max(i, j))
                        J[key] = J.get(key, 0.0) - self.lam * loc[a][1] * loc[b][1] / self.K
        for (i, j) in self.hard:
            if i in pos and j in pos:
                hard.add((min(pos[i], pos[j]), max(pos[i], pos[j])))
        J = {k: w for k, w in J.items() if k not in hard}
        return Problem(h, J, hard, meta={"parent_idx": kept.tolist()}), kept


def build_problem(c: Contacts, *, lam: float = 0.0, K: int = 2, tau: float = 60.0,
                  mode: str = "broadcast", active=None) -> Problem:
    """active: iterable of satellite ids that count as missions for fairness (default: all)."""
    v = c.value_kB()
    J: dict = {}
    hard: set = set()
    order = np.argsort(c.aos, kind="stable")
    aos, los = c.aos[order], c.los[order]
    for a_pos in range(len(order)):
        i = order[a_pos]
        stop = np.searchsorted(aos, los[a_pos] + tau, side="left")
        for j in order[a_pos + 1:stop]:
            ii, jj = (int(i), int(j)) if i < j else (int(j), int(i))
            if c.station[i] == c.station[j]:
                hard.add((ii, jj))
            elif c.sat[i] == c.sat[j]:
                ov = min(c.los[i], c.los[j]) - max(c.aos[i], c.aos[j])
                if ov <= 0:
                    continue
                if mode == "commanded":
                    hard.add((ii, jj))
                else:
                    J[(ii, jj)] = J.get((ii, jj), 0.0) - c.weight[i] * c.rate[i] * ov * c.p[i] * c.p[j] / 8000.0
    J = {k: w for k, w in J.items() if k not in hard and w != 0.0}
    groups = []
    act = set(np.unique(c.sat).tolist()) if active is None else set(active)
    for s in np.unique(c.sat):
        if s in act:
            idx = np.flatnonzero(c.sat == s)
            groups.append((idx, c.p[idx].astype(float)))
    return Problem(v.astype(float), J, hard, groups, lam, K, {"tau": tau, "mode": mode})


# ---- evaluation metrics on a schedule ------------------------------------------
class FastEval:
    """Expected unique data of schedules under many success-probability vectors, without rebuilding the
    problem: the redundancy pair list (same satellite, different stations, overlapping) is built once."""

    def __init__(self, c: Contacts):
        self.c = c
        base = build_problem(c, lam=0.0, mode="broadcast")
        self.hard = np.array(sorted(base.hard)) if base.hard else np.zeros((0, 2), int)
        pairs = np.array(list(base.J.keys())) if base.J else np.zeros((0, 2), int)
        self.pi, self.pj = pairs[:, 0], pairs[:, 1]
        ov = np.minimum(c.los[self.pi], c.los[self.pj]) - np.maximum(c.aos[self.pi], c.aos[self.pj])
        self.pw = c.weight[self.pi] * c.rate[self.pi] * ov / 8000.0  # times p_i p_j
        self.lin = c.weight * c.rate * c.dur / 8000.0                 # times p_i

    def unique_kB(self, x, p):
        x = np.asarray(x, float)
        return float(self.lin @ (p * x) - self.pw @ (p[self.pi] * p[self.pj] * x[self.pi] * x[self.pj]))


def schedule_metrics(c: Contacts, x: np.ndarray, *, K: int = 2, active=None, fast: FastEval | None = None,
                     p=None) -> dict:
    if fast is not None:
        return _metrics_fast(fast, x, K=K, active=active, p=c.p if p is None else p)
    return _metrics_slow(c, x, K=K, active=active)


def _metrics_fast(fe: FastEval, x, *, K, active, p):
    c = fe.c
    x = np.asarray(x).astype(int)
    sats = np.unique(c.sat) if active is None else np.array(sorted(active))
    vals = c.weight * p * c.rate * c.dur / 8000.0
    on = x == 1
    cnt, ecnt, share = [], [], []
    for s in sats:
        m = c.sat == s
        cnt.append((m & on).sum())
        ecnt.append(p[m & on].sum())
        share.append(vals[m & on].sum() / max(vals[m].sum(), 1e-9))
    cnt, ecnt, share = map(np.array, (cnt, ecnt, share))
    viol = int((x[fe.hard[:, 0]] * x[fe.hard[:, 1]]).sum()) if len(fe.hard) else 0
    return {
        "expected_unique_kB": fe.unique_kB(x, p), "n_contacts": int(x.sum()), "missions": int(len(sats)),
        "starved": int(np.sum(cnt == 0)), "exp_successful_lt1": int(np.sum(ecnt < 1.0)),
        "below_K": int(np.sum(cnt < K)),
        "jain_share": float(share.sum() ** 2 / (len(share) * np.sum(share ** 2) + 1e-12)),
        "min_share": float(share.min()) if len(share) else 0.0, "violations": viol,
    }


def _metrics_slow(c: Contacts, x: np.ndarray, *, K: int = 2, active=None) -> dict:
    """Outcome metrics under the broadcast (expected unique data) value model, independent of lam."""
    base = build_problem(c, lam=0.0, mode="broadcast")
    x = np.asarray(x).astype(int)
    sats = np.unique(c.sat) if active is None else np.array(sorted(active))
    vals = c.value_kB()
    cnt, ecnt, share = [], [], []
    for s in sats:
        m = c.sat == s
        on = m & (x == 1)
        cnt.append(on.sum())
        ecnt.append(c.p[on].sum())
        share.append(vals[on].sum() / max(vals[m].sum(), 1e-9))
    cnt, ecnt, share = map(np.array, (cnt, ecnt, share))
    jain = float(share.sum() ** 2 / (len(share) * np.sum(share ** 2) + 1e-12))
    return {
        "expected_unique_kB": float(base.objective(x)[0]),
        "n_contacts": int(x.sum()),
        "missions": int(len(sats)),
        "starved": int(np.sum(cnt == 0)),
        "exp_successful_lt1": int(np.sum(ecnt < 1.0)),
        "below_K": int(np.sum(cnt < K)),
        "jain_share": jain,
        "min_share": float(share.min()) if len(share) else 0.0,
        "violations": int(base.violations(x)[0]),
    }
