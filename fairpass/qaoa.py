"""QAOA for scheduling sub-problems.

Encoding: maximise f(x) = x^T Q x  (Q upper-triangular incl. diagonal, penalties folded in)
          <=> minimise C(x) = -f(x).  With x_i = (1 - z_i)/2 the cost is an Ising Hamiltonian
          C = c0 + sum_i a_i Z_i + sum_{i<j} b_ij Z_i Z_j.
Normalisation: coefficients are divided by s = max(|a|, |b|) so that angles transfer between windows.

Two simulators:
  * `NumpyQAOA` - exact statevector via the diagonal cost (all 2^n energies) and per-qubit RX mixers;
                  used for parameter training (thousands of evaluations) and noiseless metrics.
  * `qaoa_circuit` - the same ansatz as a Qiskit circuit (RZ / RZZ / RX), used for transpilation,
                  resource counts, noisy (Aer) runs and the IonQ / Qollab demo. Bit i of a Qiskit result
                  string (read right-to-left) is variable i, the same convention as `classical.all_bitstrings`.
"""
from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit

from .classical import all_bitstrings


def ising_from_qubo(Q: np.ndarray):
    """Return (c0, a, b) for C(z) = -x^T Q x with x = (1 - z)/2.  b is a dict {(i,j): coeff}."""
    n = len(Q)
    Qs = -np.triu(Q)  # minimise
    c0 = 0.0
    a = np.zeros(n)
    b = {}
    for i in range(n):
        c0 += Qs[i, i] / 2
        a[i] -= Qs[i, i] / 2
    for i in range(n):
        for j in range(i + 1, n):
            w = Qs[i, j]
            if w == 0:
                continue
            c0 += w / 4
            a[i] -= w / 4
            a[j] -= w / 4
            b[(i, j)] = b.get((i, j), 0.0) + w / 4
    return c0, a, b


def normalised_ising(Q):
    c0, a, b = ising_from_qubo(Q)
    s = max(np.max(np.abs(a), initial=0.0), max((abs(v) for v in b.values()), default=0.0)) or 1.0
    return a / s, {k: v / s for k, v in b.items()}, s


class NumpyQAOA:
    def __init__(self, Q: np.ndarray):
        self.n = len(Q)
        self.X = all_bitstrings(self.n)
        self.f = np.einsum("ki,ij,kj->k", self.X.astype(float), np.triu(Q), self.X.astype(float))
        a, b, self.scale = normalised_ising(Q)
        Z = 1 - 2 * self.X.astype(float)
        e = Z @ a
        for (i, j), v in b.items():
            e = e + v * Z[:, i] * Z[:, j]
        self.energy = e  # normalised Ising energy (minimise), equals (-f - c0)/s

    def state(self, gammas, betas, init=None):
        n = self.n
        psi = np.full(2 ** n, 2 ** (-n / 2), complex) if init is None else init.astype(complex)
        for g, bt in zip(gammas, betas):
            psi = psi * np.exp(-1j * g * self.energy)
            c, s = np.cos(bt), -1j * np.sin(bt)
            psi = psi.reshape([2] * n)  # axis k <-> bit (n-1-k) because index bit i = variable i
            for q in range(n):
                ax = n - 1 - q
                p0 = np.take(psi, 0, axis=ax)
                p1 = np.take(psi, 1, axis=ax)
                psi = np.stack([c * p0 + s * p1, s * p0 + c * p1], axis=ax)
            psi = psi.reshape(-1)
        return psi

    def probs(self, gammas, betas):
        p = np.abs(self.state(gammas, betas)) ** 2
        return p / p.sum()


def qaoa_circuit(Q: np.ndarray, gammas, betas, measure=True) -> QuantumCircuit:
    a, b, _ = normalised_ising(Q)
    n = len(Q)
    qc = QuantumCircuit(n, n if measure else 0)
    qc.h(range(n))
    for g, bt in zip(gammas, betas):
        for i in range(n):
            if a[i] != 0:
                qc.rz(2 * g * a[i], i)
        for (i, j), v in b.items():
            qc.rzz(2 * g * v, i, j)
        qc.rx(2 * bt, range(n))
    if measure:
        qc.measure(range(n), range(n))
    return qc


def counts_to_matrix(counts: dict, n: int):
    """Qiskit counts -> (X (m,n) int array, multiplicities)."""
    keys = list(counts)
    X = np.array([[int(k[::-1][i]) for i in range(n)] for k in keys], int)
    return X, np.array([counts[k] for k in keys])


# ---- metrics -------------------------------------------------------------------------------
def distribution_metrics(probs: np.ndarray, f_pen: np.ndarray, feasible: np.ndarray, shots: int = 1000):
    """Metrics of a sampling distribution over all 2^n bitstrings for the penalised objective f_pen."""
    fmax = f_pen.max()
    opt = np.isclose(f_pen, fmax, rtol=1e-9, atol=1e-9)
    p_opt = float(probs[opt].sum())
    # expected best-of-N approximation ratio (feasible-only value, infeasible counted as 0)
    val = np.where(feasible, f_pen, 0.0)
    val = np.maximum(val, 0.0)
    order = np.argsort(val)
    v_sorted, p_sorted = val[order], probs[order]
    cdf = np.clip(np.cumsum(p_sorted), 0, 1)
    cdf_prev = np.concatenate([[0.0], cdf[:-1]])
    p_best = cdf ** shots - cdf_prev ** shots   # P(best of N == this value)
    e_best = float(np.sum(p_best * v_sorted))
    return {
        "p_opt": p_opt,
        "p_feasible": float(probs[feasible].sum()),
        "exp_ratio": float(np.sum(probs * val) / fmax) if fmax > 0 else 0.0,
        "best_of_N_ratio": e_best / fmax if fmax > 0 else 0.0,
        "p_hit_opt_N": float(1 - (1 - p_opt) ** shots),
    }
