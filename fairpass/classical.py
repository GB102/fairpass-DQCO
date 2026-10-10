"""Classical schedulers for a `Problem` (maximise f, hard pairs as constraints).

* exact_cpsat     - CP-SAT on the exactly linearised objective (fairness via one integer square per mission);
                    the reference optimum (status and bound reported)
* brute_force     - exhaustive enumeration for small sub-problems (n <= ~22)
* station_greedy  - SatNOGS auto-scheduler style: each station independently walks its passes in
                    priority order and keeps every pass that does not overlap one already kept
* network_greedy  - coordinated greedy on marginal objective gain (aware of redundancy and fairness)
* local_search    - add / drop / swap hill climbing to a local optimum
* simulated_annealing - single-flip Metropolis on the penalised QUBO (sub-problems)
* random_sampling - best of N uniform random bitstrings (matched-budget baseline for QAOA)
"""
from __future__ import annotations

import time

import numpy as np
from ortools.sat.python import cp_model

from .problem import Problem

SCALE = 1e6      # objective units -> integers
PSCALE = 1000    # probabilities -> integers inside fairness sums


def exact_cpsat(pb: Problem, *, time_limit: float = 60.0, hint=None, workers: int = 8, log=False):
    m = cp_model.CpModel()
    x = [m.NewBoolVar(f"x{i}") for i in range(pb.n)]
    for (i, j) in pb.hard:
        m.AddBoolOr([x[i].Not(), x[j].Not()])
    obj = [int(round(SCALE * pb.h[i])) * x[i] for i in range(pb.n)]
    for (i, j), w in pb.J.items():
        y = m.NewBoolVar("")
        cw = int(round(SCALE * w))
        if cw < 0:  # maximising with a negative weight: y >= x_i AND x_j is exact
            m.AddBoolOr([x[i].Not(), x[j].Not(), y])
        else:
            m.AddImplication(y, x[i])
            m.AddImplication(y, x[j])
        obj.append(cw * y)
    if pb.lam > 0:
        # lam * [ S - (S^2 - sum p_i^2 x_i) / (2K) ],  S = sum p_i x_i,  with p in units of 1/PSCALE
        c_lin = pb.lam * SCALE / PSCALE
        c_sq = pb.lam * SCALE / (2 * pb.K * PSCALE ** 2)
        for idx, p in pb.groups:
            P = [int(round(PSCALE * v)) for v in p]
            ub = sum(P)
            S = m.NewIntVar(0, ub, "")
            m.Add(S == sum(Pi * x[i] for Pi, i in zip(P, idx)))
            sq = m.NewIntVar(0, ub * ub, "")
            m.AddMultiplicationEquality(sq, [S, S])
            obj.append(int(round(c_lin)) * S)
            obj.append(-int(round(c_sq)) * sq)
            obj += [int(round(c_sq * Pi * Pi)) * x[i] for Pi, i in zip(P, idx)]
    m.Maximize(sum(obj))
    if hint is not None:
        for i, v in enumerate(hint):
            m.AddHint(x[i], int(v))
    s = cp_model.CpSolver()
    s.parameters.max_time_in_seconds = time_limit
    s.parameters.num_workers = workers
    s.parameters.log_search_progress = log
    t = time.perf_counter()
    st = s.Solve(m)
    sol = np.array([s.Value(v) for v in x], int)
    return sol, {"status": s.StatusName(st), "wall_s": time.perf_counter() - t,
                 "bound": s.BestObjectiveBound() / SCALE, "objective_int": s.ObjectiveValue() / SCALE,
                 "gap_rel": (s.BestObjectiveBound() - s.ObjectiveValue()) / max(abs(s.ObjectiveValue()), 1)}


def all_bitstrings(n):
    return ((np.arange(2 ** n)[:, None] >> np.arange(n)[None, :]) & 1).astype(np.int8)


def brute_force(pb: Problem, P=None):
    """Exact optimum of the penalised QUBO (== constrained optimum, see problem.py) and the whole
    landscape. Bit i of the integer index = variable i."""
    X = all_bitstrings(pb.n)
    f = pb.penalised(X, P)
    k = int(np.argmax(f))
    return X[k].astype(int), f


def station_greedy(c, pb: Problem, *, key="value", tau=60.0):
    """Independent per-station scheduling (no network coordination), as satnogs-auto-scheduler's
    ordered_scheduler run separately on every station. key: 'value' (expected data) or 'max_el'."""
    score = c.value_kB() if key == "value" else c.max_el
    x = np.zeros(pb.n, int)
    for g in np.unique(c.station):
        idx = np.flatnonzero(c.station == g)
        kept = []
        for i in idx[np.argsort(-score[idx], kind="stable")]:
            if score[i] <= 0:
                continue
            if all(c.aos[i] >= c.los[k] + tau or c.aos[k] >= c.los[i] + tau for k in kept):
                kept.append(i)
        x[kept] = 1
    return x


class _State:
    """Incremental marginal-gain bookkeeping for greedy / local search."""

    def __init__(self, pb: Problem, x):
        self.pb = pb
        self.x = np.array(x, int)
        n = pb.n
        self.nb = [[] for _ in range(n)]
        for (i, j), w in pb.J.items():
            self.nb[i].append((j, w))
            self.nb[j].append((i, w))
        self.hard_arr = np.array(sorted(pb.hard), int).reshape(-1, 2)
        self.hard = [[] for _ in range(n)]
        for (i, j) in pb.hard:
            self.hard[i].append(j)
            self.hard[j].append(i)
        self.grp = np.full(n, -1)
        self.gp = np.zeros(n)
        self.S = np.zeros(len(pb.groups))
        for g, (idx, p) in enumerate(pb.groups):
            self.grp[idx] = g
            self.gp[idx] = p
            self.S[g] = float(p @ self.x[idx])
        self.coup = pb.h.astype(float).copy()  # h_i + sum_{j on} J_ij
        for i in np.flatnonzero(self.x):
            for j, w in self.nb[i]:
                self.coup[j] += w

    def gain(self, i):
        """Objective change from flipping x_i."""
        pb, g = self.pb, self.grp[i]
        fair = 0.0
        if pb.lam > 0 and g >= 0:
            S_other = self.S[g] - (self.gp[i] if self.x[i] else 0.0)
            fair = pb.lam * self.gp[i] * (1 - S_other / pb.K)
        d = self.coup[i] + fair
        return -d if self.x[i] else d

    def flip(self, i):
        sgn = -1 if self.x[i] else 1
        self.x[i] ^= 1
        for j, w in self.nb[i]:
            self.coup[j] += sgn * w
        g = self.grp[i]
        if g >= 0:
            self.S[g] += sgn * self.gp[i]

    def blocked(self, i):
        return any(self.x[j] for j in self.hard[i])

    def all_gains(self):
        """Vectorised flip gains for every variable."""
        pb = self.pb
        d = self.coup.copy()
        if pb.lam > 0:
            g = self.grp
            m = g >= 0
            S_other = self.S[g[m]] - np.where(self.x[m] == 1, self.gp[m], 0.0)
            d[m] += pb.lam * self.gp[m] * (1 - S_other / pb.K)
        return np.where(self.x == 1, -d, d)

    def n_blocking(self):
        """Number of selected hard-conflict partners of each variable."""
        nb = np.zeros(self.pb.n, int)
        h = self.hard_arr
        if len(h):
            np.add.at(nb, h[:, 0], self.x[h[:, 1]])
            np.add.at(nb, h[:, 1], self.x[h[:, 0]])
        return nb


def network_greedy(pb: Problem, x0=None):
    """Repeatedly add the feasible variable with the largest positive marginal gain."""
    st = _State(pb, np.zeros(pb.n, int) if x0 is None else x0)
    blocked = st.n_blocking() > 0
    while True:
        g = st.all_gains()
        g[(st.x == 1) | blocked] = -np.inf
        bi = int(np.argmax(g))
        if not np.isfinite(g[bi]) or g[bi] <= 1e-12:
            break
        st.flip(bi)
        for j in st.hard[bi]:
            blocked[j] = True
    return st.x


def local_search(pb: Problem, x, max_iter=5000, swap_candidates=300):
    """Best-improvement hill climbing with drop, add, and swap (add i after dropping its selected conflicts).
    Swap moves are evaluated exactly for the `swap_candidates` unselected variables with the largest
    conflict-free gain (all add/drop moves are evaluated)."""
    st = _State(pb, x)
    for _ in range(max_iter):
        g = st.all_gains()
        nblk = st.n_blocking()
        cand_free = (st.x == 0) & (nblk == 0)
        best, move = 1e-9, None
        for mask in (st.x == 1, cand_free):  # drop / add
            if mask.any():
                k = int(np.argmax(np.where(mask, g, -np.inf)))
                if g[k] > best:
                    best, move = g[k], (k, [])
        sw = np.flatnonzero((st.x == 0) & (nblk > 0))
        if len(sw):
            sw = sw[np.argsort(-g[sw])[:swap_candidates]]
            for i in sw:
                conf = [j for j in st.hard[i] if st.x[j]]
                tot = 0.0
                for j in conf:
                    tot += st.gain(j)
                    st.flip(j)
                tot += st.gain(i)
                for j in conf:
                    st.flip(j)
                if tot > best:
                    best, move = tot, (int(i), conf)
        if move is None:
            break
        i, conf = move
        for j in conf:
            st.flip(j)
        st.flip(i)
    return st.x


def simulated_annealing(pb: Problem, *, sweeps=200, restarts=1, seed=0, P=None, T0=None, T1=None):
    """Metropolis single-flip annealing on the penalised QUBO; returns the best state *visited*.
    Budget = sweeps*n proposals (objective-change evaluations) per restart. Temperatures are set from the
    non-penalty coefficients so that the penalty scale does not freeze the walk. The defaults
    (T0 = 10x, T1 = 0.3x the median |linear coefficient|) were tuned on the TRAINING windows of the
    QAOA study (grid over T0 in {0.3,1,3,10}, T1 in {0.003,0.03,0.3}), the same data QAOA angles see."""
    rng = np.random.default_rng(seed)
    Q = pb.qubo_matrix(P)
    S = Q + Q.T - np.diag(np.diag(Q))
    n = pb.n
    lin = np.abs(np.diag(Q))
    scale = float(np.median(lin[lin > 0])) if np.any(lin > 0) else 1.0
    T0 = T0 or 10.0 * scale
    T1 = T1 or 0.3 * scale
    best_x, best_f, evals = None, -np.inf, 0
    for _ in range(restarts):
        x = rng.integers(0, 2, n)
        f = float(x @ Q @ x)
        if f > best_f:
            best_f, best_x = f, x.copy()
        for T in np.geomspace(T0, T1, sweeps):
            for i in rng.permutation(n):
                d = (1 - 2 * x[i]) * (S[i, i] + S[i] @ x - S[i, i] * x[i])
                evals += 1
                if d >= 0 or rng.random() < np.exp(d / T):
                    x[i] ^= 1
                    f += d
                    if f > best_f + 1e-12:
                        best_f, best_x = f, x.copy()
    return best_x, {"flips": evals}


def random_sampling(pb: Problem, shots: int, seed=0, P=None):
    rng = np.random.default_rng(seed)
    X = rng.integers(0, 2, (shots, pb.n))
    f = pb.penalised(X, P)
    return X[int(np.argmax(f))], f
