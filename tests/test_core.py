"""Consistency checks: CP-SAT == brute force on random small instances; penalised optimum is feasible;
sub-problem restriction preserves the objective."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fairpass.classical import brute_force, exact_cpsat, local_search, network_greedy  # noqa: E402
from fairpass.problem import Contacts, build_problem  # noqa: E402


def random_contacts(n, seed):
    rng = np.random.default_rng(seed)
    aos = np.sort(rng.uniform(0, 3600, n))
    dur = rng.uniform(240, 720, n)
    return Contacts(sat=rng.integers(0, 4, n), station=rng.integers(0, 3, n), aos=aos, los=aos + dur,
                    max_el=rng.uniform(10, 90, n), rate=rng.choice([1200, 9600], n).astype(float),
                    p=rng.uniform(0.2, 0.9, n), weight=np.ones(n))


def test_exact_matches_bruteforce():
    for seed in range(12):
        c = random_contacts(12, seed)
        for lam, mode in [(0, "broadcast"), (200, "broadcast"), (200, "commanded")]:
            pb = build_problem(c, lam=lam, K=2, mode=mode)
            xb, f = brute_force(pb)
            xc, info = exact_cpsat(pb, time_limit=10)
            assert pb.violations(xb)[0] == 0
            assert pb.violations(xc)[0] == 0
            assert abs(pb.objective(xb)[0] - pb.objective(xc)[0]) < 1e-2 * max(1, abs(f.max())), (seed, lam, mode)
            xg = local_search(pb, network_greedy(pb))
            assert pb.violations(xg)[0] == 0
            assert pb.objective(xg)[0] <= pb.objective(xb)[0] + 1e-6


def test_restrict_consistent():
    c = random_contacts(14, 3)
    pb = build_problem(c, lam=100, K=2)
    x = local_search(pb, network_greedy(pb))
    free = np.arange(4, 11)
    sub, kept = pb.restrict(free, x)
    xs, _ = brute_force(sub)
    y = x.copy()
    y[free] = 0
    y[kept] = xs
    assert pb.violations(y)[0] == 0
    assert pb.objective(y)[0] >= pb.objective(x)[0] - 1e-6  # exact re-optimisation never worsens



def test_lns_monotone_feasible():
    from fairpass.classical import station_greedy
    from fairpass.hybrid import lns, solver_exact, solver_random
    rng = np.random.default_rng(5)
    n = 60
    aos = np.sort(rng.uniform(0, 7200, n))
    c = Contacts(sat=rng.integers(0, 8, n), station=rng.integers(0, 4, n), aos=aos,
                 los=aos + rng.uniform(240, 720, n), max_el=rng.uniform(10, 90, n),
                 rate=rng.choice([1200, 9600], n).astype(float), p=rng.uniform(0.2, 0.9, n), weight=np.ones(n))
    pb = build_problem(c, lam=200, K=2)
    x0 = station_greedy(c, pb)
    for solver in (solver_exact, solver_random(50)):
        x, info = lns(c, pb, x0, solver, n=8, W=1200, step=600, t_max=7200)
        assert pb.violations(x)[0] == 0
        assert np.all(np.diff(info["trace"]) >= -1e-9)
        assert pb.objective(x)[0] >= pb.objective(x0)[0] - 1e-9


def test_exact_unique_data():
    from fairpass.problem import FastEval
    # one satellite, three stations: exact union vs closed form; pairwise bound is exact for <= 2 overlaps
    c = Contacts(sat=np.array([1, 1, 1]), station=np.array([1, 2, 3]), aos=np.array([0.0, 0, 0]),
                 los=np.array([100.0, 100, 100]), max_el=np.array([50.0] * 3), rate=np.array([8000.0] * 3),
                 p=np.array([0.5, 0.5, 0.5]), weight=np.ones(3))
    fe = FastEval(c)
    assert abs(fe.unique_kB(np.array([1, 1, 1]), c.p) - 100 * (1 - 0.125)) < 1e-9
    assert abs(fe.unique_kB(np.array([1, 1, 0]), c.p) - fe.unique_kB_pairwise(np.array([1, 1, 0]), c.p)) < 1e-9
    assert fe.unique_kB_pairwise(np.array([1, 1, 1]), c.p) < fe.unique_kB(np.array([1, 1, 1]), c.p)


if __name__ == "__main__":
    test_exact_matches_bruteforce()
    test_restrict_consistent()
    test_lns_monotone_feasible()
    test_exact_unique_data()
    print("ok")
