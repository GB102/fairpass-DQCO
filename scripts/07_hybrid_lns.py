"""Step 7: end-to-end hybrid schedule - classical LNS with a QAOA window step, against matched alternatives.

Start: per-station value greedy (no coordination).  Problem: fairness-aware objective (lam = 200, K = 2).
Window step solvers (same windows, same acceptance rule):
  exact      enumeration of the window (classical reference for the sub-step)
  qaoa       transferred angles (no runtime optimisation), N shots sampled from the exact QAOA distribution
  qaoa_noisy same, Aer density matrix with Forte-like noise (2q 0.4%, 1q 0.04%, readout 0.5%)
  random     best of N uniform random bitstrings
  sa         simulated annealing with N single-flip evaluations
  greedy     coordinated greedy on the window
Shots N in {100, 1000}; window size n = 10; 3 seeds for stochastic solvers.
Reference: global CP-SAT optimum of the same objective (results/schedule_study.json).
Output: results/hybrid_lns.json
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fairpass import MODEL_VERSION  # noqa: E402
from fairpass.classical import exact_cpsat, local_search, station_greedy  # noqa: E402
from fairpass.hybrid import lns, solver_exact, solver_greedy, solver_random, solver_sa  # noqa: E402
from fairpass.problem import Contacts, build_problem, schedule_metrics  # noqa: E402
from fairpass.qaoa import NumpyQAOA  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
LAM, K, N_WIN = 200.0, 2, 10
FORTE = {"e2": 0.004, "readout": 0.005}


def qaoa_params(n):
    P = json.load(open(ROOT / "results/qaoa_params.json"))
    sizes = sorted({int(k.split("_")[0][1:]) for k in P})
    m = min(sizes, key=lambda s: abs(s - n))
    # depth chosen by training loss (never by test results): the best energy-trained depth for that size
    keys = [k for k in P if k.startswith(f"n{m}_") and k.endswith("energy")]
    k = min(keys, key=lambda k: P[k]["train_loss"])
    return np.array(P[k]["gammas"]), np.array(P[k]["betas"]), k


def solver_qaoa(shots, noisy=False):
    sys.path.insert(0, str(ROOT / "scripts"))
    from importlib import import_module
    q6 = import_module("06_qaoa_study")

    def f(sub, rng):
        Q = sub.qubo_matrix()
        g, b, _ = qaoa_params(sub.n)
        if noisy:
            q6.READOUT = FORTE["readout"]
            pr = q6.noisy_probs(Q, g, b, FORTE["e2"])
        else:
            pr = NumpyQAOA(Q).probs(g, b)
        k = rng.choice(len(pr), size=shots, p=pr)
        X = ((k[:, None] >> np.arange(sub.n)[None, :]) & 1).astype(int)
        fx = sub.penalised(X)
        return X[int(np.argmax(fx))]
    return f


def main():
    inst = json.load(open(ROOT / "data/processed/instance_2026-10-01.json"))
    c = Contacts.from_dict(inst["contacts"])
    act = [int(s) for s in np.unique(c.sat) if c.p[c.sat == s].mean() >= 0.3]
    pb = build_problem(c, lam=LAM, K=K, active=act)
    x0 = station_greedy(c, pb, key="value", tau=inst["tau_s"])
    f0 = pb.objective(x0)[0]
    ss = json.load(open(ROOT / "results/schedule_study.json"))
    x_opt = np.zeros(c.n, int)
    x_opt[ss["x"][f"cpsat_fair_{int(LAM)}"]] = 1
    f_opt = pb.objective(x_opt)[0]
    out = {"model_version": MODEL_VERSION, "lam": LAM, "K": K, "n_window": N_WIN, "f_start": f0, "f_opt": f_opt,
           "start_metrics": schedule_metrics(c, x0, K=K, active=act),
           "opt_metrics": schedule_metrics(c, x_opt, K=K, active=act),
           "qaoa_params_used": qaoa_params(N_WIN)[2], "runs": []}
    print(f"start {f0:.0f}  CP-SAT optimum {f_opt:.0f}")
    configs = [("exact", None, solver_exact, [0]), ("greedy", None, solver_greedy, [0])]
    for N in (100, 1000):
        configs += [("qaoa", N, solver_qaoa(N), [0, 1, 2]), ("random", N, solver_random(N), [0, 1, 2]),
                    ("sa", N, solver_sa(N), [0, 1, 2])]
    configs += [("qaoa_noisy_forte", 100, solver_qaoa(100, noisy=True), [0])]
    for name, N, solver, seeds in configs:
        for sd in seeds:
            t = time.perf_counter()
            x, info = lns(c, pb, x0, solver, n=N_WIN, rounds=2, seed=sd)
            f = pb.objective(x)[0]
            rec = {"solver": name, "shots": N, "seed": sd, "f": f,
                   "gap_closed": (f - f0) / (f_opt - f0) if f_opt > f0 else None,
                   "rel_gap_to_opt": (f_opt - f) / f_opt, "calls": info["calls"],
                   "wall_s": time.perf_counter() - t, "trace": info["trace"],
                   "metrics": schedule_metrics(c, x, K=K, active=act)}
            out["runs"].append(rec)
            print(f"{name:17s} N={N}  seed={sd}  f={f:.0f}  gap closed={rec['gap_closed']:.3f}  "
                  f"starved={rec['metrics']['starved']}  {rec['wall_s']:.0f}s", flush=True)
    # polishing check: does a classical local search after LNS reach the optimum anyway?
    x_ls = local_search(pb, x0)
    out["local_search_from_start"] = {"f": pb.objective(x_ls)[0],
                                      "gap_closed": (pb.objective(x_ls)[0] - f0) / (f_opt - f0)}
    (ROOT / "results/hybrid_lns.json").write_text(json.dumps(out))


if __name__ == "__main__":
    main()
