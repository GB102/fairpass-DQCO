# 2. Methods (fixed before the main results were computed)

## 2.1 Data and the propagation layer
* **Stations.** 10 Online SatNOGS stations spread across Europe (UK, NL, AT, DE, BG, ES, GR, SE, SK, TR), chosen for
  long observing records and UHF capability. Each station's horizon is the **effective** one, the median elevation
  at observation start in its own records, because the live configuration can differ from the value used when
  the observations were scheduled. For M0EYT the configured horizon is 20° but the records show 10°.
* **Satellites and TLEs.** Every satellite these stations observed between 2026-09-13 and 2026-10-01. For each, we
  use the TLE stored with the most recent observation that was not later than the planning day. We use the
  satellite's most-observed transmitter (mode, baud, downlink frequency). A satellite–station pair is a candidate only
  if the station has an antenna whose frequency range covers the downlink.
* **Passes.** SGP4 (`sgp4` 2.27) gives TEME positions, a GMST rotation converts them to Earth-fixed, and WGS-84
  topocentric elevation is computed every 10 s with linear interpolation of horizon crossings and parabolic refinement
  of culmination.
* **Validation (reproduction of a published quantity).** For every downloaded observation we propagate *its own
  stored TLE* from the station's recorded coordinates. We compare the computed pass maximum elevation with the
  network's recorded `max_altitude` (whole degrees), and the computed horizon crossings with the observation start/end.

## 2.2 Pass-success model (H1)
* Label: vetted `good` = 1, `bad` = 0. `failed` (station/software failure) and `unknown` (unvetted) are excluded.
* Models, nested:
  * `const`: one base rate.
  * `geom`: quadratic in sin(max elevation).
  * `sat`: L2-shrunk satellite effects.
  * `full`: geometry + modulation family + L2-shrunk station effects + L2-shrunk satellite effects.
* Shrinkage strengths are chosen on an inner time split of the training period.
* **Time split:** fit on 09-13…09-24 (M0EYT) and 09-19…09-24 (other stations); evaluate on 09-26…09-30. Metrics:
  log-loss, Brier, AUC, decile calibration, and a decision metric: the success rate of the top 30% of passes each
  station would pick by each score.
* Uncertainty: 40 observation-level bootstrap replicates of the full model, propagated to every schedule score.

## 2.3 Scheduling objective (QUBO)
For candidate pass *i* (satellite *s*, station *g*, window [AOS, LOS]):

| term | form | meaning |
|---|---|---|
| value | h_i = w_s · p_i · baud_s · dur_i / 8000 | expected delivered data (kB) |
| redundancy | J_ij = −w_s · baud_s · overlap_ij · p_i p_j / 8000 | same satellite seen by two stations at once: exact expected *unique* data for pairs |
| fairness | λ · E[g(n_s)], g(n) = n − n(n−1)/(2K) | concave utility per active mission; under independent successes E[g] = Σp_i x_i − (1/K) Σ_{i<j} p_i p_j x_i x_j |
| antenna | x_i + x_j ≤ 1 (penalty P = 1.5 · max linear gain in the QUBO) | same station, intervals closer than the 60 s re-pointing gap |

* *Active mission*: a satellite whose mean predicted pass success is ≥ 0.3.
* Mission weights w_s = 1, because no public priorities exist. Weights are a user input.
* The penalised QUBO has only feasible optima: every other coupling is ≤ 0, so removing one endpoint of a
  violated pair gains at least P − h_i > 0.
* CP-SAT models fairness exactly with one integer square per mission, and CP-SAT's optimum is cross-checked against
  brute force (`tests/test_core.py`).

## 2.4 Schedulers compared (H2)
All schedulers share the same candidate set and are judged by the same validated model:

* **actual:** what the stations really observed that day.
* **per-station greedy by elevation / by expected data:** each station independently keeps the best non-overlapping
  passes. This is the rule of `satnogs-auto-scheduler`'s ordered scheduler, re-implemented.
* **coordinated greedy + local search.**
* **coordinated CP-SAT optimum:** throughput only (λ = 0) and a λ sweep.
* **Model-assumption sensitivity:** re-optimise under (a) commanded downlink (one station per satellite at a time),
  (b) contact-time value (baud ignored), (c) the geometry-only success model; score every schedule under the primary model.

## 2.5 Quantum step (H3)
* **Windows:** the n ∈ {8, 10, 12} highest-value candidate passes with AOS in a 20-minute window. All other passes
  are fixed to today's per-station greedy schedule, and their couplings are folded into the window's linear terms.
  Windows anchored before 12:00 UTC are **training** windows; windows from 12:00 UTC are **test** windows.
* **QAOA:** standard X-mixer, p = 1–3, on the normalised Ising form (max |coefficient| = 1), with RZ and RZZ gates.
  The angles minimise the mean expected energy (or a CVaR(0.1) variant) over the training windows. INTERP
  initialisation is used for p > 1. The angles are then **transferred unchanged** to the test windows.
* **Matched baselines:**
  * uniform random sampling with the same number of shots;
  * simulated annealing with the same number of objective evaluations, temperatures tuned on the same training windows;
  * coordinated greedy;
  * exact enumeration.
* **Metrics:** probability of an optimal schedule per shot, expected best-of-N approximation ratio (N = 100, 1000),
  feasible fraction.
* **Noise:** Aer density-matrix simulation with depolarising error e2 on each RZZ, e1 = e2/10 on single-qubit gates,
  and 0.5% symmetric readout flips. The sweep covers e2 = 0…2%. IonQ's published 2Q error is about 0.4% (Forte) and
  0.6% (Aria).
* **Resources:** qubits, ancillas (none), logical depth, RZZ count (≈ native ZZ on Forte), and CX count after
  transpiling to {rz, sx, x, cx}.
* **End-to-end hybrid:** large-neighbourhood search over the day (20-minute windows every 10 minutes, 2 rounds,
  accept only feasible improvements). The window solver is QAOA, uniform random, SA, greedy or exact, at
  N = 100 and 1000 shots/evaluations, with 3 seeds. The reference is the global CP-SAT optimum.
