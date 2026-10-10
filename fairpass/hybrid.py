"""Window extraction and large-neighbourhood search (LNS) with a pluggable sub-problem solver.

A window = the n highest-value candidate passes whose AOS falls in [t, t+W). All other passes are fixed
to the incumbent; `Problem.restrict` folds their couplings into the window's linear terms. The quantum
step (QAOA) is one possible sub-solver; exact enumeration, random sampling and simulated annealing are
drop-in alternatives with the same interface, so the contribution of the quantum step is isolated.
"""
from __future__ import annotations

import numpy as np

from .classical import brute_force, network_greedy, random_sampling, simulated_annealing
from .problem import Problem


def window_free_set(c, pb: Problem, t: float, W: float, n: int):
    m = np.flatnonzero((c.aos >= t) & (c.aos < t + W) & (pb.h > 1e-6))
    if len(m) == 0:
        return m
    return m[np.argsort(-pb.h[m], kind="stable")[:n]]


def extract_windows(c, pb: Problem, x_inc, *, n: int, W: float = 1200.0, step: float = 600.0,
                    t_max: float = 86400.0, min_pairs: int = 3):
    """Sub-problems of exactly n free variables, with >= min_pairs couplings/conflicts."""
    out = []
    for t in np.arange(0.0, t_max - W + 1, step):
        free = window_free_set(c, pb, t, W, n + 6)  # spare candidates, some may be blocked
        if len(free) < n:
            continue
        sub, kept = pb.restrict(free, x_inc)
        if len(kept) < n:
            continue
        sub, kept = pb.restrict(kept[:n], x_inc)
        if len(kept) != n or len(sub.hard) + len(sub.J) < min_pairs:
            continue
        out.append({"t": float(t), "kept": kept, "sub": sub})
    return out


# ---- sub-problem solvers: (sub, rng) -> 0/1 vector ------------------------------------------
def solver_exact(sub, rng):
    x, _ = brute_force(sub)
    return x


def solver_random(shots):
    def f(sub, rng):
        x, _ = random_sampling(sub, shots, seed=int(rng.integers(1 << 31)))
        return x
    return f


def solver_sa(flips):
    def f(sub, rng):
        sweeps = max(1, flips // max(sub.n, 1))
        x, _ = simulated_annealing(sub, sweeps=sweeps, seed=int(rng.integers(1 << 31)))
        return x
    return f


def solver_greedy(sub, rng):
    return network_greedy(sub)


def lns(c, pb: Problem, x0, solver, *, n=10, W=1200.0, step=600.0, rounds=2, seed=0, t_max=86400.0):
    """Sweep windows over the horizon `rounds` times; accept a window re-solve only if feasible and better."""
    rng = np.random.default_rng(seed)
    x = np.array(x0, int)
    f = pb.objective(x)[0]
    trace = [f]
    calls = 0
    for _ in range(rounds):
        for t in np.arange(0.0, t_max - W + 1, step):
            free = window_free_set(c, pb, t, W, n + 6)
            if len(free) == 0:
                continue
            sub, kept = pb.restrict(free, x)
            if len(kept) == 0:
                continue
            if len(kept) > n:
                free = kept[:n]
                sub, kept = pb.restrict(free, x)
            xs = solver(sub, rng)
            calls += 1
            if sub.violations(xs)[0]:
                continue
            y = x.copy()
            # blocked members of `free` conflict with a fixed selected pass, so they are 0 in x already
            y[free] = 0
            y[kept] = xs
            fy = pb.objective(y)[0]
            if pb.violations(y)[0] == 0 and fy > f + 1e-9:
                x, f = y, fy
            trace.append(f)
    return x, {"trace": trace, "calls": calls}
