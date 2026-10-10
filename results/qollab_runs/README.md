# Live Qollab runs of the demo (window 0, QAOA p = 2, 500 shots, one job each)

Raw `fairpass_run.json` outputs downloaded from the Qollab console, run by the team on 2026-10-10:

* `builtin_simulator.json`: Qollab built-in simulator (Qiskit BasicSimulator, ideal)
* `ionq_simulator.json`: IonQ remote simulator through Qollab, with IonQ's noise model of the machine picked in the
  Select QPU dialog (Forte 1). The backend object reports `ionq_simulator`. IonQ does not publish how its Forte noise
  model works (see Qollab docs, "Noise and the real machine").

Comparison with the precomputed values in `demo/qollab/data/demo_bundle.json`:

| run | P(optimal plan) per shot [95% Wilson] | × uniform (1/1024) | feasible fraction [95%] | best sampled plan |
|---|---|---|---|---|
| exact ideal distribution (numpy, precomputed) | 0.025 | 25.8 | 0.805 | n/a |
| Aer, Forte-like depolarising model (precomputed) | 0.023 | 23.4 | n/a | n/a |
| Qollab built-in simulator | 0.020 [0.011, 0.036] | 20.5 | 0.798 [0.761, 0.831] | 1.000 of optimum |
| **IonQ noise-model simulator** | **0.016 [0.008, 0.031]** | **16.4** | **0.732 [0.692, 0.769]** | **1.000 of optimum** |
