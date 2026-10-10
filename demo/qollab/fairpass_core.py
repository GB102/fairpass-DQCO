"""Minimal, dependency-light core for the Qollab demo (NumPy + Qiskit only; runs in Pyodide).

Mirrors fairpass/qaoa.py and fairpass/problem.py from the full repository:
  * a window is a QUBO to MAXIMISE, f_pen(x) = x^T Q x (Q upper-triangular, diagonal = linear terms),
    where hard conflicts (same antenna, overlapping passes) carry a penalty -P
  * the QAOA circuit is built from the normalised Ising form of -f_pen (max |coefficient| = 1),
    so that angles trained on other windows transfer without re-optimisation
  * bit i of every bitstring is pass i of the window (Qiskit count keys are read right-to-left)
"""
import numpy as np
from qiskit import QuantumCircuit


def all_bitstrings(n):
    return ((np.arange(2 ** n)[:, None] >> np.arange(n)[None, :]) & 1).astype(np.int8)


def qubo_value(Q, X):
    X = np.atleast_2d(X).astype(float)
    return np.einsum("ki,ij,kj->k", X, np.triu(Q), X)


def violations(hard, X):
    X = np.atleast_2d(X)
    if not hard:
        return np.zeros(len(X), int)
    h = np.array(hard)
    return (X[:, h[:, 0]] * X[:, h[:, 1]]).sum(1)


def normalised_ising(Q):
    n = len(Q)
    Qs = -np.triu(np.asarray(Q, float))
    a = np.zeros(n)
    b = {}
    for i in range(n):
        a[i] -= Qs[i, i] / 2
    for i in range(n):
        for j in range(i + 1, n):
            w = Qs[i, j]
            if w:
                a[i] -= w / 4
                a[j] -= w / 4
                b[(i, j)] = b.get((i, j), 0.0) + w / 4
    s = max(np.max(np.abs(a)), max((abs(v) for v in b.values()), default=0.0)) or 1.0
    return a / s, {k: v / s for k, v in b.items()}


def qaoa_circuit(Q, gammas, betas):
    a, b = normalised_ising(Q)
    n = len(Q)
    qc = QuantumCircuit(n, n, name="fairpass_qaoa")
    qc.h(range(n))
    for g, bt in zip(gammas, betas):
        for i in range(n):
            if a[i]:
                qc.rz(2 * g * a[i], i)
        for (i, j), v in b.items():
            qc.rzz(2 * g * v, i, j)
        qc.rx(2 * bt, range(n))
    qc.measure(range(n), range(n))
    return qc


def parse_counts(counts, n):
    """Counts dict -> (X (m,n) int array, multiplicities). Accepts binary or hex keys."""
    X, m = [], []
    for k, v in counts.items():
        k = str(k).replace(" ", "")
        val = int(k, 16) if k.startswith("0x") else int(k, 2)
        X.append([(val >> i) & 1 for i in range(n)])
        m.append(int(v))
    return np.array(X, int), np.array(m)


def uniform_best_of(Q, hard, P, shots, seed):
    rng = np.random.default_rng(seed)
    X = rng.integers(0, 2, (shots, len(Q)))
    f = qubo_value(Q, X)
    return X[int(np.argmax(f))], f
