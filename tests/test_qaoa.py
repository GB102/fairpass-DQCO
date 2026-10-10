"""The numpy QAOA simulator must agree with the Qiskit circuit (bit order and angle conventions)."""
import sys
from pathlib import Path

import numpy as np
from qiskit.quantum_info import Statevector

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fairpass.qaoa import NumpyQAOA, qaoa_circuit  # noqa: E402
from fairpass.problem import build_problem  # noqa: E402
from tests.test_core import random_contacts  # noqa: E402


def test_numpy_matches_qiskit():
    for seed in range(4):
        pb = build_problem(random_contacts(7, seed), lam=150, K=2)
        Q = pb.qubo_matrix()
        sim = NumpyQAOA(Q)
        rng = np.random.default_rng(seed)
        g, b = rng.uniform(-1, 1, 2), rng.uniform(-1, 1, 2)
        p_np = sim.probs(g, b)
        qc = qaoa_circuit(Q, g, b, measure=False)
        p_qk = Statevector(qc).probabilities()
        assert np.allclose(p_np, p_qk, atol=1e-9), np.abs(p_np - p_qk).max()
        # energy ordering: lowest normalised energy == highest penalised objective
        f = pb.penalised(sim.X)
        assert np.argmin(sim.energy) == np.argmax(f)


if __name__ == "__main__":
    test_numpy_matches_qiskit()
    print("ok")
