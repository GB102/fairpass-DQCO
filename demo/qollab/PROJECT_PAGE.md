# FairPass: who gets the antenna?

*Fairness-aware satellite contact scheduling for open ground-station networks, with a QAOA step tested honestly.*
**Quantum Impact (space)** · This effort is supported by Qollab & IonQ.

▶ **Demo video:** <paste your unlisted YouTube link here>
💻 **Code and full report:** https://github.com/JuL-sezaR/superpositioned

---

## 1. What is it?
FairPass decides **which satellite passes each antenna in a ground-station network should take tomorrow**. The goal is
to deliver the most data without leaving any active mission with no contact at all.

It works on real public data:
* **Data:** about 4,000 vetted observations from 10 SatNOGS stations across Europe, plus the satellites' orbit data.
* **Model:** a pass-success model fitted on those observations and checked on later days.
* **Plan:** a network-wide schedule that trades data volume against fairness.

The quantum part is a QAOA circuit that re-plans one busy 20-minute window at a time. It runs as **one job** on the
backend you pick here.

## 2. Why does it matter, and who is it for?
Hundreds of university and amateur satellites depend on open networks like SatNOGS to hear from home. Each station
schedules itself, so stations chase the same satellites at the same time, while low-data-rate missions get nothing.
**FairPass is for network operators and the small missions that rely on them.**

On a real day (2026-10-01; 10 stations; 10,727 possible passes; 340 active missions):

| | Expected data | Missions with zero contacts |
|---|---|---|
| Each station schedules itself (today) | 252 MB | 99 |
| Coordinated plan (FairPass, throughput only) | 362 MB (**+44%**) | 49 |
| Coordinated plan + fairness | 335 MB | **12** |

The data behind this holds up:
* Our orbit calculations reproduce SatNOGS's own pass records to a median of **0.25°**.
* The success model predicts later days far better than pass elevation alone (AUC **0.96 vs 0.65**).
* On the target day it predicted **167** successful observations; **165** happened.

## 3. How does the quantum part work?
1. **Pick a window.** We take a busy 20-minute window. Its 10 most valuable candidate passes become **10 qubits**
   (1 = take the pass).
2. **Encode the window.** Everything else in the day's plan is fixed. The window's choice is written as a QUBO with
   three ingredients:
   * expected data (rewarded);
   * the same satellite heard twice (redundancy, penalised);
   * fairness: a concave reward per mission, so a mission's first contact is worth more than its fifth.

   Two passes on the same antenna at the same time get a penalty.
3. **Build the circuit.** The QUBO becomes a **QAOA circuit**: cost layers of RZ and RZZ gates and mixer layers of RX
   gates, p = 2 here. That's 10 qubits, no ancillas, and about 20 two-qubit gates.
4. **No live optimisation.** The circuit angles were **trained on morning windows and reused unchanged** on afternoon
   windows, so this run sends one circuit and nothing else.
5. **Decode.** Every shot decodes to an antenna plan. We keep the best feasible one and compare it with random
   sampling using the same number of shots.

**What we found (honestly):**

| Window size | QAOA best of 100 shots | Random (100) | Simulated annealing (100) | Greedy |
|---|---|---|---|---|
| 8 passes | 1.000 | 0.642 | 0.991 | 0.998 |
| 10 passes | 1.000 | 0.464 | 0.975 | 0.999 |
| 12 passes | 0.999 | 0.329 | 0.926 | 0.988 |

*(Quality of the best plan as a fraction of the optimum; about 50 unseen windows per size.)*

* **QAOA finds the optimal plan 55–435× more often than random sampling.** On **IonQ's Forte 1 noise model**, run
  here, it hit the optimum **16× more often** than random, and its best plan was optimal.
* **Classical wins at this size.** Simulated annealing and greedy match QAOA, and plain local search over the whole
  day beats every hybrid we built (96.7% vs about 71% of the gap closed). We claim a measured principle, **not a
  speed-up**.

## 4. How do I run it?
1. Press **Run on Qollab** and choose a backend:
   * **Built-in simulator** is instant.
   * **IonQ Forte 1** (remote simulator, free) uses IonQ's own noise model.
2. Optional Run settings:
   * `window`: which held-out window to plan, 0–11.
   * `depth`: QAOA layers, 1–3.
   * `shots`: 300–500 recommended for IonQ.
   * `seed`: for the classical comparison.
3. Read the console, which prints five sections:
   * **A**, the network-day result (precomputed, classical);
   * **B**, the circuit and its size;
   * **C**, live QAOA results vs random sampling;
   * **D**, the recommended antenna plan for the window;
   * **E**, the honest scope.

   A histogram and `fairpass_run.json` are offered for download.

**Replicate or extend:** the full pipeline is on GitHub: data download, orbit validation, model fitting, CP-SAT
scheduling, the QAOA study and the figures. Run `bash scripts/run_pipeline.sh`. To extend it:
* change the stations or mission priorities;
* change the fairness weight;
* or plug in another window solver, for example a constraint-preserving QAOA mixer.

---

**Limitations.**
* Results are model expectations; the plan hasn't been executed on the network.
* Data volume uses a baud-rate proxy.
* QAOA ran on simulators and IonQ's noise model, not hardware.

The next real experiment is to run the plan for one day and compare the outcomes, and to measure decoded frames on
high-rate satellites.

**Licences and credits.**
* Code: **MIT**.
* Satellite data: © Libre Space Foundation and SatNOGS contributors, **CC BY-SA 4.0**, not covered by MIT.
* Orbit elements for the forward plan come from CelesTrak.
* Built during the hackathon with help from an AI coding assistant (Claude), as disclosed per the rules.
* Libraries: Qiskit, Qiskit Aer, OR-Tools, sgp4, NumPy, SciPy, Matplotlib.
