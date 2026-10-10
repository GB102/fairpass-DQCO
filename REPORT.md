# FairPass: research report

*Qollab × IonQ Quantum Hackathon 2026, Quantum Impact (space). This effort is supported by Qollab & IonQ.*
All numbers come from `results/*.json`, produced by `scripts/run_pipeline.sh` (model version `fairpass-0.1.0`).
Each value is labelled **measured** (third-party records), **computed**, **fitted**, **simulated** or
**hypothetical**.

## 0. Verdict in five lines
1. **The decision that matters is classical.** On a real network-day (10 SatNOGS stations, 2026-10-01, 10,727 candidate
   passes), coordinating the stations raises expected delivered data by **+44%** (paired 90% interval +42% to +45%)
   over each station greedily scheduling itself by expected data. The gain is **+72%** over scheduling by elevation.
   Today's stations use this per-station rule.
2. **Fairness has a price, and the frontier shows it.** A concave per-mission fairness term cuts active missions left
   with zero contacts from **49 to 22 for -3.2%** of data (lambda = 100), or **to 12 for -7.7%** (lambda = 200).
   Reaching 0 costs -21%.
3. **Measured outcomes rank passes far better than geometry** (held-out AUC 0.96 vs 0.65), and the model predicted the
   day's real outcomes (167 [165-171] vs 165 observed). But after coordination, a geometry-only plan loses only
   **1.6%**. The assumption that moves the plan most is how data rate is valued: capped vs listed baud changes three
   quarters of the passes and 14% of the data.
4. **The quantum step is a small, honest result.** On held-out windows, QAOA with *transferred* angles samples the
   optimal 8-12-pass plan **55-435x more often** than uniform random sampling, and stays above x25 at 2% two-qubit
   error. It edges tuned simulated annealing at 100 samples (by 0.01-0.07 in approximation ratio) but not greedy. In
   the end-to-end hybrid it is no better than classical window solvers, and plain local search over the whole day
   beats every hybrid.
5. **No quantum speed-up is claimed.** The windows have at most 4,096 plans and are solved exactly by enumeration.

## 1. Question and who needs it
University cubesat teams and amateur missions rely on open ground-station networks such as SatNOGS. Each station
schedules itself, using the auto-scheduler's ordered-greedy rule. Antenna time is the scarce resource. The decision we
improve: **which passes should a set of stations take tomorrow** to deliver the most data without leaving active
missions with no contact. The secondary question: **is a small QAOA step a useful component** inside such a scheduler?

The go/no-go memo ([docs/01_feasibility.md](docs/01_feasibility.md)) narrowed the original idea:

* "QAOA improves the schedule" was rejected as untestable-by-construction at 12–20 variables.
* The weather layer was dropped: VHF/UHF rain fade is negligible.
* "One station per satellite at a time" was replaced by a redundancy model for broadcast telemetry. It is kept as a
  sensitivity case.

## 2. Data and what we reproduced (tested on existing data)
* **Sources.** SatNOGS Network observations (start/end, stored TLE, vetted outcome, max elevation, mode/baud) and
  stations, plus the SatNOGS DB (CC BY-SA 4.0). We used 10 European stations: 2,453 vetted training observations
  (09-13…09-24), 1,445 test observations (09-26…09-29), and 524 observations on the target day (2026-10-01).
  See [docs/DATA_MANIFEST.md](docs/DATA_MANIFEST.md).
* **Reproduction of a published quantity (computed vs measured).** Propagating each observation's own stored TLE with
  SGP4 reproduces the network's recorded pass maximum elevation for **5,971 of 5,977** observations (all periods):

  * median |error| **0.25°**, 95th percentile **0.49°**;
  * 99.2% within 1°;
  * the records are whole degrees, so the 95th percentile matches the 0.475° expected from rounding alone;
  * observation start times match horizon crossings with a median offset of **0.3 s** (92% within 30 s).

  This is only true once each station's **effective** horizon is inferred from its own records. Four stations'
  live configurations differ from what was used, by 8–25°. That is a data-quality finding in itself.
* **Pass-success model (fitted, time-split validation).** Held-out results:

  | model | log-loss | Brier | AUC | top-30% picks that succeed |
  |---|---|---|---|---|
  | constant (base rate 0.53) | 0.695 | 0.251 | 0.50 | 0.64* |
  | geometry only (max elevation) | 0.663 | 0.235 | 0.65 | 0.64 |
  | satellite effects only | 0.311 | 0.088 | 0.95 | 0.88 |
  | **full** (geometry + mode + station + satellite) | **0.258** | **0.073** | **0.96** | **0.88** |

  \*Ties are broken by elevation. Decile calibration is close to the diagonal (Fig. 1c), with mild over-prediction
  between 0.2 and 0.5. The 4.7% of test observations whose satellite was unseen in training reach only AUC 0.78. **What
  this validates:** whether a satellite is transmitting dominates pass success; elevation is second-order. **What it
  does not validate:** byte counts. 'Good' is a vetting label, and demodulated frames are too sparse to use.
* **Schedule-level check of the model.** For the vetted observations the stations actually ran on 2026-10-01, the model
  predicts **167 successes (bootstrap 90% interval 165–171); 165 were observed.**

## 3. The decision study (generated by us, judged by the validated model)
**Instance.**
* 10,727 candidate passes, 381 satellites, 340 active missions (mean predicted success >= 0.3).
* We dropped 37 satellites whose newest TLE was older than 14 days, and 1,064 satellite-station pairs for
  antenna-band mismatch.
* **Value** = success probability x min(listed baud, 9,600 bps) x pass duration. Typical SatNOGS chains decode at most
  about 9,600 bps, and for LoRa the listed "baud" is not a payload rate. The uncapped variant is a sensitivity case.
* **Scoring.** Every schedule is scored by the **exact** expected unique data: per satellite, the time-integral of
  1 - prod(1 - p) over the stations listening. The optimiser uses the pairwise (second-order) form of the same
  quantity, which is exact up to two simultaneous stations and conservative beyond. We report expected data as the
  bootstrap mean with a 90% interval over 40 replicates of the success model (Fig. 2).

| schedule | expected MB/day | starved missions (of 340) | contacts |
|---|---|---|---|
| actual SatNOGS schedule that day (see note) | 82 [80, 83] | 168 | 387 |
| per-station greedy by elevation | 211 [202, 222] | 45 | 1,342 |
| per-station greedy by expected data | 252 [246, 258] | 99 | 1,395 |
| coordinated greedy + local search | 334 [323, 343] | 50 | 1,390 |
| coordinated CP-SAT, throughput only | 362 [351, 373] | 49 | 1,498 |
| coordinated CP-SAT + fairness, lambda = 50 | 359 [348, 370] | 34 | 1,440 |
| coordinated CP-SAT + fairness, lambda = 100 | 351 [340, 362] | 22 | 1,421 |
| **coordinated CP-SAT + fairness, lambda = 200** | **335 [324, 346]** | **12** | 1,380 |
| coordinated CP-SAT + fairness, lambda = 400 | 309 [299, 321] | 2 | 1,341 |
| coordinated CP-SAT + fairness, lambda = 800 | 285 [276, 297] | 0 | 1,325 |

Note: the actual schedule is not like-for-like. The stations ran far fewer observations than antenna time allows (387
vs about 1,400), and 137 of their observations were FM-voice or CW transmitters with no data rate. It shows unused
antenna time, not a scheduler comparison.

**Paired differences** (same model replicate for both schedules; `results/paired_differences.json`):

| comparison | change in expected data | change in starved |
|---|---|---|
| throughput optimum vs per-station greedy by expected data | **+110 MB (+43.7%, 90% interval +42.4% to +45.2%)** | -50 |
| throughput optimum vs per-station greedy by elevation | +151 MB (+71.5%, +68.5% to +74.9%) | +4 |
| throughput optimum vs coordinated greedy | +29 MB (+8.7%, +8.1% to +9.2%) | -1 |
| fair (lambda = 200) vs throughput optimum | **-28 MB (-7.7%, -8.1% to -7.3%)** | **-37** |
| fair (lambda = 800) vs throughput optimum | -77 MB (-21.3%, -22.0% to -20.6%) | -49 |
| fair (lambda = 200) vs per-station greedy by expected data | +82 MB (+32.7%, +31.0% to +34.4%) | -87 |

Findings:
* **Uncoordinated scheduling wastes antenna time on duplicates.** Stations ranking by expected data all chase the same
  satellites at the same moment, and ranking by elevation scatters effort onto silent satellites. Coordination fixes
  both.
* **Throughput-only optimisation starves low-rate missions.** Of the 49 missions it leaves at zero, the median rate is
  1,200 baud, against 4,800 for active missions.
* **The fairness frontier (Fig. 3a) is the operator's decision.** lambda = 50 rescues 15 missions for -0.9%;
  lambda = 100 rescues 27 for -3.2%; lambda = 200 rescues 37 for -7.7%; the last 12 cost another 13.6%. We use
  lambda = 200 in the demo and the forward plan. lambda of about 100 is the cheaper knee.
* **Robustness.** Re-optimising under 10 bootstrap replicates, the lambda = 200 plan's regret averages **0.7%**
  (max 0.9%).
* **Solver status.** CP-SAT hit its 180 s limit on every run. The incumbents are within **7-8%** of its proven upper
  bound; the bound is loose because of the fairness squares. Coordinated greedy with local search is 8.7% below the
  incumbent.

**Model-assumption sensitivity.** We re-optimised under each alternative assumption (lambda = 200) and scored the
result under the primary model:

| alternative assumption | MB/day | starved | overlap with primary plan (Jaccard) |
|---|---|---|---|
| primary | 335 | 12 | 1 |
| geometry-only success model | 330 (-1.6%) | 10 | 0.72 |
| commanded downlink (one station per satellite at a time) | 325 (-3.0%) | 7 | 0.65 |
| value = contact time (baud ignored) | 285 (-14.9%) | 1 | 0.61 |
| value = listed baud, uncapped | 289 (-13.7%) | 11 | 0.25 |

The qualitative recommendation (coordinate, then buy fairness along the frontier) holds under every assumption.
*Which* passes are chosen depends most on how data rate is valued, and an operator must state that.

**Forward plan** (`results/forward_plan*.csv`, a *prediction* that is not yet observable): for 2026-10-11, from
current CelesTrak TLEs, the lambda = 200 plan for the same 10 stations covers 7,852 candidates and 263 active missions.
It gives **263 MB, 0 starved**, against **224 MB, 58 starved** for uncoordinated per-station greedy.

## 4. The quantum step (simulated, held-out)
**What the quantum component is.** For a 20-minute window, the 8–12 highest-value candidate passes become qubits.
All other passes stay fixed to the incumbent, and their couplings fold into the window's linear terms. The circuit is a
standard X-mixer QAOA on the normalised Ising form of the penalised window QUBO. It encodes value, redundancy and
fairness couplings as RZ/RZZ gates, with antenna conflicts as RZZ penalties. Angles were trained on **morning** windows
(before 12:00 UTC) and **transferred unchanged** to afternoon windows. No optimisation runs at execution time. Each
measured bitstring is a candidate antenna plan, and the best feasible sample is kept. There are no ancillas.

**Held-out results.** About 50 test windows per size. Matched budget means the same number of samples or objective
evaluations. The QAOA configuration is the one with the highest P(optimum) on the **training** windows (p = 3,
CVaR(0.1)-trained, at every size):

| n | QAOA P(optimum)/shot (train / **test**) | uniform | ×uniform | best-of-100: QAOA | uniform | SA (100 evals, tuned) | greedy | today's per-station plan |
|---|---|---|---|---|---|---|---|---|
| 8 | 0.236 / **0.217** | 0.0039 | 55 | 1.000 | 0.642 | 0.991 | 0.998 | 0.00 |
| 10 | 0.146 / **0.115** | 0.0010 | 118 | 1.000 | 0.464 | 0.975 | 0.999 | 0.00 |
| 12 | 0.151 / **0.106** | 0.00024 | 435 | 0.999 | 0.329 | 0.926 | 0.988 | 0.00 |

* **Transfer works, with a modest generalisation gap.** Test P(optimum) is 70–92% of training.
* **Training choices matter.** Energy-trained p = 3 sometimes lands in a worse optimum than p = 2 (n = 10: 0.051 vs
  0.102). CVaR training fixes this. We report the choice made on training data, never on test data.
* **At 1,000 samples every method converges.** SA reaches 1.000; uniform reaches 0.76–0.99.
* The "today" column is the ratio of the window's restriction of the uncoordinated plan. In these windows it is
  dominated by redundant passes, so its window value is ≤ 0.
* **Noise** (Fig. 3b,c; Aer density matrix; depolarising e2 on every RZZ, e1 = e2/10, 0.5% readout; 12 test windows
  per size; energy-trained angles). The amplification over uniform for n = 10, p = 2 falls from **×50** (noiseless)
  to **×45 at e2 = 0.5%** (Forte-like) and **×34 at e2 = 2%**. For n = 8 it falls from ×37 to ×34 to ×27. **No
  break-even point appears within 0–2%.**
* **Resources.** A mean of 9 / 14 / 18 RZZ per layer at n = 8 / 10 / 12 (range 4–34). For the demo-size circuit
  (n = 10, p = 3) that is about 40 two-qubit gates (≈ native ZZ on Forte) and logical depth about 20. Transpiled to
  {rz, sx, x, cx}, the CX count is 2× the RZZ count. No ancillas.

**End-to-end hybrid** (Fig. 4): large-neighbourhood search over the day with 20-minute, 10-pass windows, 2 rounds,
starting from the per-station greedy plan, against the global CP-SAT incumbent:

| window solver | gap closed (3 seeds) |
|---|---|
| exact enumeration | 71.5% |
| greedy | 71.4% |
| simulated annealing, N = 100 | 71.3–71.6% |
| QAOA, N = 100 (energy-trained p = 3) | 70.7–70.9% |
| QAOA, N = 100, Forte-like noise | 71.2% |
| uniform random, N = 100 | 68.9–69.9% |
| all solvers, N = 1,000 | 71.2–71.7% |
| *plain local search on the whole day (no windows)* | *96.7%* |

**Interpretation.** QAOA's amplitude concentration is real, transfers to unseen windows, and survives IonQ-level
noise. Within one window it is worth about 1–2 points of day-level gap-closure over blind sampling at equal shots.
But the windowed LNS is itself the bottleneck: it leaves 90 missions starved, against 8 at the optimum. Tuned
classical heuristics at equal budget match or beat QAOA. **H3 is rejected as a practical claim and supported only as a
measurement principle.**

## 5. What is new, and what kind of contribution
* **Empirical (new for quantum-scheduling work).** Pass values grounded in vetted outcomes from an open network, with
  held-out and schedule-level validation. Prior QUBO/QAOA ground-station studies (QUARGS, SpaceOps 2025; QAOA for
  oversubscribed scheduling, 2025/26; Makarov et al. 2024) use synthetic values.
* **Methodological.** Fairness as an exact expected concave utility under uncertain success, which is native to the
  QUBO (no slack qubits). Redundancy as an exact pairwise expected-unique-data term. Matched-budget baselines with tuned
  SA, angle transfer to held-out windows, and a noise sweep against IonQ reference rates.
* **Engineering.** An open, reproducible coordinated scheduler on real SatNOGS data. It includes per-station
  horizon inference and a forward 24-hour plan from current CelesTrak TLEs (`results/forward_plan*`).
* **Not new:** QAOA, CVaR, INTERP, CP-SAT, fairness in satellite scheduling as such.

## 6. Next measurement worth making
We asked which uncertainty, if resolved, would change the plan most.
* **Success-rate uncertainty barely matters.** We screened the 25 satellites whose bootstrap uncertainty moves the most
  value, then locally re-optimised under 20 replicates each (partial expected value of perfect information). Total
  partial EVPI is **82 kB/day, 0.02% of the day's value**. Regret of the recommended plan under model uncertainty
  averages 0.7%.
* **The value assumptions matter a lot** (section 3 sensitivity table). Using listed baud uncapped instead of capped
  at 9,600 bps changes **about three quarters of the chosen passes** (Jaccard 0.25). Scored under the capped model,
  the uncapped plan delivers **13.7% less** (289 vs 335 MB).

**Recommended experiment: a one-day decode campaign on the 42 high-rate satellites** (listed > 9,600 baud, including
LoRa) at the same 10 stations. Run their highest passes, with at least 15 vetted passes per rate class, enough to bound
a success fraction to ±0.1. Count *decoded frames*, not just 'good' vetting. If high-rate downlinks prove decodable at
their listed rate, switch to the uncapped value model; if not, keep the cap. That choice moves about 14% of the day's
expected data and three quarters of the antenna plan, so its value of information is about **46 MB/day** for this
network. A second, cheaper step: run the recommended plan for one day and compare vetted outcomes with the predicted
165 ± 3 successes.

## 7. Limitations and what remains unverified
* All schedule values are **model expectations**. No optimised schedule has been executed on the network. The
  real test is to run the recommended plan for a day and vet the outcomes.
* 'Good' is a vetting label, not bytes. Baud × duration is a link-capacity proxy, and broadcast beacons are
  intermittent. The contact-time variant changes which passes are chosen.
* Mission weights are hypothetical (all equal). λ, K = 2, the 60 s re-pointing gap and the 0.3 activity threshold
  are design choices.
* The 10 stations are a non-random European sub-network. Unvetted observations are excluded, which may bias labels.
* CP-SAT is near-optimal, not proven optimal (gap 1–3%). The redundancy term is a lower bound when three or more
  stations overlap.
* QAOA results are classical simulations. The depolarising noise model ignores crosstalk, drift and coherent error.
  Running on IonQ's noise models or hardware through Qollab is the next check. The demo makes that one click.
* The quantum step offers **no** computational advantage at these sizes. Whether it would help at sizes beyond exact
  enumeration is untested here.

## 8. Reproduce
See [README.md](README.md). `bash scripts/run_pipeline.sh` regenerates every number and figure from the data in
`data/raw/`. Re-downloading the data is throttled by SatNOGS to about 60 requests/hour.
