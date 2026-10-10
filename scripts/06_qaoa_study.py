"""Step 6 (H3): does a QAOA sub-solver beat matched-budget classical sampling on real scheduling windows,
and where does IonQ-like noise erase the effect?

Design (fixed before looking at test results):
  problem      fairness-aware day instance (lam = LAM, K = 2), incumbent = per-station value greedy
  windows      n in {8, 10, 12} highest-value passes in 20-min windows; anchors before 12:00 UTC -> TRAIN,
               after 12:00 UTC -> TEST (time split; parameters never see test windows)
  QAOA         standard X-mixer QAOA on the penalised QUBO (normalised Ising), p = 1, 2, 3; angles trained
               on TRAIN windows (mean expected energy, and a CVaR(0.1) variant), then transferred unchanged
  baselines    uniform random sampling with the SAME number of shots; simulated annealing with the same
               number of objective evaluations (single flips); greedy; exact enumeration (reference)
  metrics      P(optimum) per shot, expected best-of-N approximation ratio (N = 100, 1000), feasibility
  noise        Aer density-matrix simulation, depolarising e2 on each RZZ, e1 = e2/10 on 1q gates,
               readout flip 0.5%; e2 swept 0 .. 2%; n = 8, 10 (exact distributions, no shot noise)
  resources    RZZ count, depth; transpiled CX count (fixed-angle MS ~ CX) and IonQ partial-MS estimate
Outputs: results/qaoa_study.json, results/qaoa_params.json
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fairpass import MODEL_VERSION  # noqa: E402
from fairpass.classical import brute_force, network_greedy, simulated_annealing, station_greedy  # noqa: E402
from fairpass.hybrid import extract_windows  # noqa: E402
from fairpass.problem import Contacts, build_problem  # noqa: E402
from fairpass.qaoa import NumpyQAOA, distribution_metrics, qaoa_circuit  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
LAM, K = 200.0, 2
SIZES = [int(s) for s in os.environ.get("FP_SIZES", "8,10,12").split(",")]
PS = [1, 2, 3]
SHOTS = [100, 1000]
E2 = [0.0, 0.001, 0.0025, 0.005, 0.01, 0.02]
READOUT = 0.005
SEED = 7
SPLIT_T = 43200.0


def load():
    inst = json.load(open(ROOT / "data/processed/instance_2026-10-01.json"))
    c = Contacts.from_dict(inst["contacts"])
    act = [int(s) for s in np.unique(c.sat) if c.p[c.sat == s].mean() >= 0.3]
    pb = build_problem(c, lam=LAM, K=K, active=act)
    x_inc = station_greedy(c, pb, key="value", tau=inst["tau_s"])
    return inst, c, pb, x_inc


def prep(w):
    sub = w["sub"]
    Q = sub.qubo_matrix()
    sim = NumpyQAOA(Q)
    f = sub.penalised(sim.X)
    feas = sub.violations(sim.X) == 0
    w.update(Q=Q, sim=sim, f=f, feas=feas, fmax=f.max())
    return w


def expected_cost(sim, probs, f, fmax, cvar=None):
    """Normalised cost to MINIMISE: -E[f]/fmax (or CVaR of the upper tail)."""
    if cvar is None:
        return -float(probs @ f) / fmax
    order = np.argsort(-f)
    cp = np.cumsum(probs[order])
    k = np.searchsorted(cp, cvar) + 1
    w = probs[order][:k].copy()
    w[-1] -= cp[k - 1] - cvar if cp[k - 1] > cvar else 0.0
    return -float(w @ f[order][:k]) / (cvar * fmax)


def train(windows, p, cvar=None, init=None, seed=SEED):
    rng = np.random.default_rng(seed)

    def loss(th):
        g, b = th[:p], th[p:]
        return np.mean([expected_cost(w["sim"], w["sim"].probs(g, b), w["f"], w["fmax"], cvar) for w in windows])

    starts = []
    if init is not None:
        starts.append(init)
    if p == 1:
        # (gamma, beta) and (-gamma, -beta) give identical distributions and beta has period pi, so
        # gamma in [-1.5, 1.5] x beta in (0, pi) covers both inequivalent sign sectors
        best = None
        for g in np.linspace(-1.5, 1.5, 25):
            for b in np.linspace(0.05, np.pi - 0.05, 25):
                v = loss(np.array([g, b]))
                if best is None or v < best[0]:
                    best = (v, np.array([g, b]))
        starts.append(best[1])
    starts += [np.concatenate([rng.uniform(-1.0, 1.0, p), rng.uniform(0.05, np.pi - 0.05, p)]) for _ in range(2)]
    best = None
    for s in starts:
        r = minimize(loss, s, method="Nelder-Mead", options={"maxiter": 400 * p, "xatol": 1e-3, "fatol": 1e-5})
        if best is None or r.fun < best.fun:
            best = r
    return best.x, float(best.fun)


def interp(th, p):
    """INTERP initialisation (Zhou et al. 2020) from depth p-1 to p."""
    g, b = th[:p - 1], th[p - 1:]

    def up(v):
        v = np.concatenate([[0.0], v, [0.0]])
        return np.array([(i / (p - 1)) * v[i] + ((p - 1 - i) / (p - 1)) * v[i + 1] for i in range(p)])
    return np.concatenate([up(g), up(b)])


def baselines(w, shots, rng):
    n = w["sub"].n
    uni = np.full(2 ** n, 2.0 ** -n)
    out = {"uniform": distribution_metrics(uni, w["f"], w["feas"], shots)}
    sa = []
    for k in range(20):
        x, _ = simulated_annealing(w["sub"], sweeps=max(1, shots // n), seed=int(rng.integers(1 << 31)))
        fx = w["sub"].penalised(x)[0]
        sa.append(max(fx, 0.0) / w["fmax"] if w["sub"].violations(x)[0] == 0 else 0.0)
    out["sa_matched_ratio"] = float(np.mean(sa))
    out["sa_matched_hit_opt"] = float(np.mean(np.isclose(sa, 1.0)))
    xg = network_greedy(w["sub"])
    out["greedy_ratio"] = float(max(w["sub"].objective(xg)[0], 0) / w["fmax"])
    xi = w["x_inc_sub"]
    out["incumbent_ratio"] = float(max(w["sub"].objective(xi)[0], 0) / w["fmax"])
    return out


def resources(Q, p):
    from qiskit import transpile
    qc = qaoa_circuit(Q, [0.3] * p, [0.4] * p)
    ops = qc.count_ops()
    tc = transpile(qc, basis_gates=["rz", "sx", "x", "cx"], optimization_level=1, seed_transpiler=SEED)
    return {"n_qubits": qc.num_qubits, "rzz": int(ops.get("rzz", 0)), "depth_logical": qc.depth(),
            "cx_transpiled": int(tc.count_ops().get("cx", 0)), "depth_transpiled": tc.depth(),
            "ionq_partial_ms_estimate": int(ops.get("rzz", 0)),
            "ionq_fixed_ms_estimate": 2 * int(ops.get("rzz", 0))}


def noisy_probs(Q, g, b, e2):
    from qiskit import transpile
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error
    n = len(Q)
    qc = qaoa_circuit(Q, g, b, measure=False)
    qc.save_density_matrix()
    nm = NoiseModel()
    if e2 > 0:
        # Aer depolarising parameter lam relates to average gate infidelity r by r = lam*(d-1)/d... for the
        # 2-qubit channel r = 0.75*lam, for 1-qubit r = 0.5*lam
        nm.add_all_qubit_quantum_error(depolarizing_error(e2 / 0.75, 2), ["rzz"])
        nm.add_all_qubit_quantum_error(depolarizing_error((e2 / 10) / 0.5, 1), ["rz", "rx", "h"])
    sim = AerSimulator(method="density_matrix", noise_model=nm if e2 > 0 else None)
    rho = sim.run(transpile(qc, sim, optimization_level=0)).result().data()["density_matrix"]
    probs = np.clip(np.real(np.diag(np.asarray(rho))), 0, None)
    # independent symmetric readout flips (applied exactly to the distribution)
    if READOUT > 0:
        P = probs.reshape([2] * n)
        for q in range(n):
            ax = n - 1 - q
            a0, a1 = np.take(P, 0, axis=ax), np.take(P, 1, axis=ax)
            P = np.stack([(1 - READOUT) * a0 + READOUT * a1, READOUT * a0 + (1 - READOUT) * a1], axis=ax)
        probs = P.reshape(-1)
    return probs / probs.sum()


def main():
    t0 = time.perf_counter()
    inst, c, pb, x_inc = load()
    rng = np.random.default_rng(SEED)
    res = {"model_version": MODEL_VERSION, "lam": LAM, "K": K, "seed": SEED, "shots": SHOTS, "e2": E2,
           "readout": READOUT, "sizes": {}}
    params = {}
    for n in SIZES:
        wins = extract_windows(c, pb, x_inc, n=n)
        for w in wins:
            w["x_inc_sub"] = x_inc[w["kept"]]
        tr = [prep(w) for w in wins if w["t"] < SPLIT_T]
        te = [prep(w) for w in wins if w["t"] >= SPLIT_T]
        print(f"n={n}: train {len(tr)} test {len(te)} windows", flush=True)
        if len(tr) < 3 or len(te) < 3:
            continue
        rec = {"n_train": len(tr), "n_test": len(te),
               "test_windows": [{"t": w["t"], "parent_idx": w["kept"].tolist(), "Q": w["Q"].tolist(),
                                 "n_hard": len(w["sub"].hard), "n_J": len(w["sub"].J),
                                 "n_optima": int(np.isclose(w["f"], w["fmax"]).sum())} for w in te],
               "density": float(np.mean([(len(w["sub"].hard) + len(w["sub"].J)) / (n * (n - 1) / 2) for w in te])),
               "qaoa": {}, "baselines": {}}
        for N in SHOTS:
            b = [baselines(w, N, rng) for w in te]
            rec["baselines"][N] = {
                "uniform_p_opt": float(np.mean([x["uniform"]["p_opt"] for x in b])),
                "uniform_best_ratio": float(np.mean([x["uniform"]["best_of_N_ratio"] for x in b])),
                "uniform_hit_opt": float(np.mean([x["uniform"]["p_hit_opt_N"] for x in b])),
                "sa_matched_ratio": float(np.mean([x["sa_matched_ratio"] for x in b])),
                "sa_matched_hit_opt": float(np.mean([x["sa_matched_hit_opt"] for x in b])),
                "greedy_ratio": float(np.mean([x["greedy_ratio"] for x in b])),
                "incumbent_ratio": float(np.mean([x["incumbent_ratio"] for x in b])),
            }
        th_prev = None
        for p in PS:
            for obj, cv in (("energy", None), ("cvar0.1", 0.1)):
                init = interp(th_prev[obj], p) if th_prev and p > 1 else None
                th, loss = train(tr, p, cv, init=init)
                th_prev = th_prev or {}
                th_prev[obj] = th
                g, bb = th[:p], th[p:]
                params[f"n{n}_p{p}_{obj}"] = {"gammas": g.tolist(), "betas": bb.tolist(), "train_loss": loss}
                m = {}
                for N in SHOTS:
                    ms = [distribution_metrics(w["sim"].probs(g, bb), w["f"], w["feas"], N) for w in te]
                    m[N] = {k: float(np.mean([x[k] for x in ms])) for k in ms[0]}
                    m[N]["per_window_p_opt"] = [x["p_opt"] for x in ms]
                    m[N]["per_window_best_ratio"] = [x["best_of_N_ratio"] for x in ms]
                # train-set metric for overfitting check
                mtr = [distribution_metrics(w["sim"].probs(g, bb), w["f"], w["feas"], 1000) for w in tr]
                m["train_p_opt"] = float(np.mean([x["p_opt"] for x in mtr]))
                m["resources"] = resources(te[0]["Q"], p)
                rec["qaoa"][f"p{p}_{obj}"] = m
                print(f"  p={p} {obj:7s} test P_opt={m[1000]['p_opt']:.4f} (uniform "
                      f"{rec['baselines'][1000]['uniform_p_opt']:.4f})  best1000={m[1000]['best_of_N_ratio']:.4f} "
                      f"(uniform {rec['baselines'][1000]['uniform_best_ratio']:.4f}, SA "
                      f"{rec['baselines'][1000]['sa_matched_ratio']:.4f})  best100={m[100]['best_of_N_ratio']:.4f} "
                      f"(uniform {rec['baselines'][100]['uniform_best_ratio']:.4f})", flush=True)
        # noise sweep on the best noiseless depth/objective (chosen by TRAIN loss ordering, not test)
        if n <= 10:
            cand = {k: v for k, v in params.items() if k.startswith(f"n{n}_") and "energy" in k}
            sweep = {}
            for key in cand:
                pq = int(key.split("_p")[1].split("_")[0])
                g, bb = np.array(cand[key]["gammas"]), np.array(cand[key]["betas"])
                rows = []
                for e2 in E2:
                    ms100, ms1000 = [], []
                    for w in te[:12]:
                        pr = noisy_probs(w["Q"], g, bb, e2)
                        ms100.append(distribution_metrics(pr, w["f"], w["feas"], 100))
                        ms1000.append(distribution_metrics(pr, w["f"], w["feas"], 1000))
                    rows.append({"e2": e2,
                                 "p_opt": float(np.mean([x["p_opt"] for x in ms1000])),
                                 "best100": float(np.mean([x["best_of_N_ratio"] for x in ms100])),
                                 "best1000": float(np.mean([x["best_of_N_ratio"] for x in ms1000])),
                                 "p_feasible": float(np.mean([x["p_feasible"] for x in ms1000]))})
                    print(f"    noise {key} e2={e2:.4f} P_opt={rows[-1]['p_opt']:.4f} best100={rows[-1]['best100']:.4f}",
                          flush=True)
                sweep[key] = rows
            uni100 = float(np.mean([distribution_metrics(np.full(2 ** n, 2.0 ** -n), w["f"], w["feas"], 100)
                                    ["best_of_N_ratio"] for w in te[:12]]))
            uni_p = float(np.mean([distribution_metrics(np.full(2 ** n, 2.0 ** -n), w["f"], w["feas"], 100)
                                   ["p_opt"] for w in te[:12]]))
            rec["noise_sweep"] = {"windows_used": min(12, len(te)), "uniform_best100": uni100,
                                  "uniform_p_opt": uni_p, "rows": sweep}
        res["sizes"][n] = rec
    res["wall_s"] = time.perf_counter() - t0
    (ROOT / "results/qaoa_study.json").write_text(json.dumps(res))
    (ROOT / "results/qaoa_params.json").write_text(json.dumps(params, indent=1))
    print("done in %.0fs" % res["wall_s"])


if __name__ == "__main__":
    main()
