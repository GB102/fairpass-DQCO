# FairPass - fairness-aware contact scheduling for open ground-station networks (Qollab x IonQ 2026)
#
# 'backend' is provided by Qollab: it is whatever you pick in the Run on Qollab dialog (built-in simulator,
# an IonQ noise-model simulator, or IonQ hardware). This run submits exactly ONE job, as IonQ requires.
#
# What is precomputed (classical, offline, in the full repository) and what runs live is printed explicitly:
#   precomputed : SatNOGS data -> validated pass/success model -> network-day schedules (CP-SAT, greedy)
#                 -> QAOA angles trained on *other* windows (first half of the day)
#   live        : one QAOA circuit for one held-out scheduling window, sampled on your chosen backend,
#                 decoded into antenna schedules and compared with matched-shot random sampling
#
# This effort is supported by Qollab & IonQ.  MIT license. Data: SatNOGS (CC BY-SA 4.0), see PROJECT_CARD.md.
import json
import time

import numpy as np

from fairpass_core import (all_bitstrings, parse_counts, qaoa_circuit, qubo_value, uniform_best_of,
                           violations)

BUNDLE = "data/demo_bundle.json"


def _hr(title=""):
    print("\n" + "=" * 78 + ("\n" + title if title else ""))


def _wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, c - h), min(1.0, c + h)


def _run_one_job(be, qc, shots):
    job = be.run(qc, shots=shots)
    try:
        from qiskit.providers.jobstatus import JobStatus
        while job.status() not in (JobStatus.DONE, JobStatus.ERROR, JobStatus.CANCELLED):
            time.sleep(3)
    except Exception:
        pass
    return job.result().get_counts()


def print_day_summary(B):
    d = B["day"]
    _hr("A. NETWORK-DAY RESULT  [precomputed offline - classical]")
    print(f"Day {d['date']} UTC | {len(d['stations'])} SatNOGS stations | {d['n_satellites']} satellites | "
          f"{d['n_candidates']:,} candidate passes | {d['n_active']} active missions")
    v = d["validation"]
    print(f"Pass geometry reproduced from stored TLEs: median |max-elevation error| "
          f"{v['geom_median_abs_deg']:.2f} deg over {v['geom_n']} observations")
    print(f"Success model on held-out days: AUC {v['auc_full']:.2f} (geometry-only {v['auc_geom']:.2f}); "
          f"top-30% picks succeed {100 * v['top30_full']:.0f}% vs {100 * v['top30_maxel']:.0f}% by elevation")
    print(f"\n{'schedule':44s} {'expected MB':>12s} {'90% interval':>16s} {'starved':>8s}")
    for name, s in d["schedules"].items():
        print(f"{name:44s} {s['MB']:12.1f} {'[%.1f, %.1f]' % (s['MB_p05'], s['MB_p95']):>16s} "
              f"{s['starved']:>5d}/{d['n_active']}")
    r = d["recommendation"]
    print(f"\nRecommendation: {r['text']}")
    if d.get("next_measurement"):
        print(f"Next measurement: {d['next_measurement']['text']}")


def main(window: int = 0, depth: int = 2, shots: int = 500, seed: int = 7):
    B = json.load(open(BUNDLE))
    print_day_summary(B)

    W = B["windows"][window % len(B["windows"])]
    n = W["n"]
    Q = np.array(W["Q"], float)
    hard = [tuple(h) for h in W["hard"]]
    prm = W["params"][str(depth)]
    _hr(f"B. LIVE QUANTUM STEP  [this run, backend: {getattr(backend, 'name', backend)}]")
    print(f"Window {window}: {W['t_utc']} UTC, {n} candidate passes, {len(hard)} antenna conflicts, "
          f"{W['n_couplings']} redundancy/fairness couplings. {W['selection_note']}")
    for i, p in enumerate(W["passes"]):
        print(f"  q{i:<2d} {p['name'][:22]:22s} @ {p['station'][:18]:18s} {p['aos']}-{p['los']}  "
              f"el {p['max_el']:4.0f}  P(ok) {p['p']:.2f}  {p['value_kB']:7.1f} kB")
    qc = qaoa_circuit(Q, prm["gammas"], prm["betas"])
    ops = qc.count_ops()
    print(f"\nCircuit: {qc.num_qubits} qubits, no ancillas, depth {qc.depth()}, RZZ (2-qubit) gates "
          f"{ops.get('rzz', 0)}, single-qubit rotations {ops.get('rz', 0) + ops.get('rx', 0) + ops.get('h', 0)}, "
          f"p = {depth}, angles transferred from training windows (no optimisation in this run).")
    print(f"Submitting ONE job with {shots} shots ...")
    t0 = time.time()
    counts = _run_one_job(backend, qc, shots)
    print(f"Job finished in {time.time() - t0:.1f}s.")

    X, mult = parse_counts(counts, n)
    f = qubo_value(Q, X)
    feas = violations(hard, X) == 0
    f_opt = W["f_opt"]
    opt_set = {tuple(o) for o in W["optima"]}
    is_opt = np.array([tuple(x) in opt_set for x in X])
    k_opt = int(mult[is_opt].sum())
    lo, hi = _wilson(k_opt, shots)
    p_uniform = W["n_optima"] / 2 ** n
    best = int(np.argmax(np.where(feas, f, -np.inf))) if feas.any() else None

    xr, fr = uniform_best_of(Q, hard, None, shots, seed)
    k_opt_r = int(sum(tuple(x) in opt_set for x in np.random.default_rng(seed).integers(0, 2, (shots, n))))
    f_inc = float(qubo_value(Q, np.array(W["incumbent"]))[0])

    _hr("C. WHAT THE LIVE SAMPLES SHOW")
    print(f"P(optimal schedule) per shot: QAOA {k_opt}/{shots} = {k_opt / shots:.3f} "
          f"[95% CI {lo:.3f}-{hi:.3f}]  vs uniform {p_uniform:.4f}  -> x{(k_opt / shots) / p_uniform:.1f}")
    print(f"  (precomputed for this window: ideal p={depth} {W['precomputed']['p_opt_ideal'][str(depth)]:.3f}, "
          f"Forte-like noise {W['precomputed']['p_opt_forte_like'][str(depth)]:.3f})")
    print(f"Feasible samples: {mult[feas].sum()}/{shots}")
    if best is not None:
        print(f"Best sampled schedule reaches {f[best] / f_opt:.3f} of the window optimum.")
    print(f"Matched-budget classical sampling ({shots} uniform random schedules): best {max(fr.max(), 0) / f_opt:.3f}"
          f" of optimum, optimum hit {k_opt_r} times.")
    print(f"Uncoordinated per-station greedy (what stations run today): {f_inc / f_opt:.3f} of optimum.")

    if best is not None:
        xb = X[best]
        inc = np.array(W["incumbent"])
        _hr("D. RECOMMENDED ANTENNA PLAN FOR THIS WINDOW  [from the best live sample]")
        for i, p in enumerate(W["passes"]):
            tag = "keep" if xb[i] and inc[i] else ("ADD " if xb[i] else ("DROP" if inc[i] else "    "))
            if xb[i] or inc[i]:
                print(f"  {tag} {p['name'][:22]:22s} @ {p['station'][:18]:18s} {p['aos']}-{p['los']} "
                      f"P(ok) {p['p']:.2f} {p['value_kB']:7.1f} kB")
        sats_new = {W["passes"][i]["name"] for i in range(n) if xb[i]}
        sats_old = {W["passes"][i]["name"] for i in range(n) if inc[i]}
        print(f"Objective vs today's greedy: {f[best] - f_inc:+.1f} (kB-equivalent incl. fairness credit); "
              f"missions served in window {len(sats_old)} -> {len(sats_new)}")

    _hr("E. HONEST SCOPE")
    print("* The window has 2^%d = %d schedules; exhaustive search solves it instantly on a laptop. The live" % (n, 2 ** n))
    print("  circuit demonstrates amplitude concentration on good schedules with transferred angles, not speed-up.")
    print("* Material result of the project is classical: coordinated, fairness-aware scheduling on real data.")
    print("* Backend noise (if any) is the chosen backend's own; no emulated noise is added in this run.")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        ratio_q = np.repeat(np.where(feas, np.maximum(f, 0) / f_opt, 0.0), mult)
        ratio_r = np.where(violations(hard, np.random.default_rng(seed).integers(0, 2, (shots, n))) == 0,
                           np.maximum(qubo_value(Q, np.random.default_rng(seed).integers(0, 2, (shots, n))), 0)
                           / f_opt, 0.0)
        fig, ax = plt.subplots(figsize=(7, 3.2))
        bins = np.linspace(0, 1.0001, 21)
        ax.hist(ratio_r, bins=bins, color="#eb6834", alpha=0.9, label="uniform random sampling", rwidth=0.85)
        ax.hist(ratio_q, bins=bins, color="#2a78d6", alpha=0.75, label=f"QAOA p={depth} (this run)", rwidth=0.85)
        ax.set_xlabel("schedule quality (fraction of window optimum; infeasible = 0)")
        ax.set_ylabel("shots")
        ax.legend(frameon=False)
        ax.set_title(f"Window {window}: {shots} shots each", loc="left")
        fig.tight_layout()
        plt.savefig("fairpass_window.png")
    except Exception as e:  # plotting is optional
        print("(plot skipped:", e, ")")

    summary = {"window": window, "depth": depth, "shots": shots, "backend": str(getattr(backend, "name", backend)),
               "p_opt_qaoa": k_opt / shots, "p_opt_uniform": p_uniform, "feasible": int(mult[feas].sum()),
               "best_ratio": float(f[best] / f_opt) if best is not None else 0.0, "counts": counts}
    try:
        from js import sendFile
        sendFile("fairpass_run.json", "application/json", json.dumps(summary))
    except Exception:
        pass
    return summary


if "backend" not in globals():  # running outside Qollab: use a local simulator and call main() ourselves
    try:
        from qiskit_aer import AerSimulator
        backend = AerSimulator()
    except ImportError:
        from qiskit.providers.basic_provider import BasicSimulator
        backend = BasicSimulator()
    main()
