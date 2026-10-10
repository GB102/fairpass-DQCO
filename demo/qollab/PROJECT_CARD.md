# FairPass: fairness-aware antenna scheduling for open ground-station networks

**Quantum Impact (space).** This effort is supported by Qollab & IonQ.

## The problem
Small satellites, mostly university cubesats, are heard through open ground-station networks such as
[SatNOGS](https://network.satnogs.org). Each station schedules itself, so antenna time goes to whatever looks best
locally. Several stations end up chasing the same satellite while low-data-rate missions get nothing.
**Which passes should a group of stations take so that more data comes down, without leaving active missions behind?**

## What we found (classical, on real data)
* **Pass geometry is checked against real records.** From the TLE stored with each observation, we reproduce SatNOGS's
  own recorded pass elevations to a median of **0.25°** across 5,971 observations.
* **We fitted a pass-success model on ~2,500 vetted observations** and tested it on later days. Elevation alone reaches
  AUC **0.65**; the model reaches **0.96**. On the target day it predicted **167** successful observations and **165**
  happened.
* **Coordination pays.** On the real day of 2026-10-01 (10 stations, 10,727 candidate passes), coordinating the stations
  delivers **+44%** expected data compared with each station scheduling itself.
* **Fairness has a visible price.** A fairness term cuts active missions with zero contacts from **49 to 22** for **−3.2%** data, or to **12** for **−7.7%**.

## What the quantum step does (this project's live run)
The **8–12 most valuable candidate passes** of a 20-minute window become qubits. All other passes are fixed. A QAOA
circuit encodes value, redundancy, fairness and antenna-conflict penalties as RZ and RZZ gates. Its angles were trained
on *morning* windows and are reused **unchanged** here, so no optimisation runs live. **One job** is sent to the backend
you pick, and each shot decodes to an antenna plan.

On held-out windows this finds the optimal plan **55–435×** more often than random sampling with the same number of
shots. In simulation that holds up to 2% two-qubit error. It edges tuned simulated annealing at 100 samples (by
0.01–0.07 of the optimum) but does **not** beat greedy, and the windows are small enough to solve exactly. This is a measured principle, **not a speed-up**.

### Quantum vs classical, at a glance
Same held-out windows, same budget of samples or objective evaluations. Metric: quality of the best plan found, as a
fraction of the exact optimum (1.00 = optimal). About 50 windows per size.

| Window size | QAOA (p = 3, 100 shots) | Uniform random (100 samples) | Simulated annealing (100 evals) | Greedy |
|---|---|---|---|---|
| 8 passes | 1.000 | 0.642 | 0.991 | 0.998 |
| 10 passes | 1.000 | 0.464 | 0.975 | 0.999 |
| 12 passes | 0.999 | 0.329 | 0.926 | 0.988 |

Inside the full-day scheduler (share of the gap to the best schedule closed): QAOA 70.8%, simulated annealing 71.4%,
greedy 71.4%, exact enumeration 71.5%, random sampling 69.4%. **Plain classical local search over the whole day closes
96.7%, so classical wins at this problem size.** On IonQ's Forte 1 noise model (this demo, window 0), QAOA hit the
optimal plan 16× more often than random sampling. Full details are in `REPORT.md`, section 4.

## How to run
1. Press **Run on Qollab** and pick a backend. The built-in simulator is instant. **IonQ Forte 1**
   (remote noise-model simulator, free) runs IonQ's own noise model.
2. Optional parameters (Run settings):
   * `window`: which held-out window. 0 is the window where today's greedy loses most; the others are in time order.
   * `depth`: QAOA p, 1–3.
   * `shots`: default 500. IonQ simulation time grows with shots.
   * `seed`: for the classical comparison.
3. The console prints, in order:
   * **A**, the precomputed network-day result (classical, offline);
   * **B**, the circuit and its resources;
   * **C**, live statistics against matched random sampling;
   * **D**, the recommended antenna plan for the window;
   * **E**, the scope.

   A histogram and `fairpass_run.json` are offered for download.

## How to replicate or extend
The full pipeline is open source (MIT): data download, SGP4 validation, model fitting, CP-SAT scheduling, the QAOA
study and the figures. See the repository README (link below). `bash scripts/run_pipeline.sh` regenerates every number.
To extend it:
* swap in your own stations (`scripts/01_fetch_data.py`);
* set mission priorities (`weight`);
* change λ;
* switch the downlink model to `commanded`;
* or plug another window solver into `fairpass/hybrid.py`.

Repository: <add your public GitHub link here>
Report and limitations: `REPORT.md` in the repository.

## Data, licences and AI disclosure
* Code and documentation: **MIT license. Fork it, change it, keep it.**
* `data/demo_bundle.json` is derived from SatNOGS Network and SatNOGS DB data, © Libre Space Foundation and
  contributors, licensed **CC BY-SA 4.0**. It is shared under the same licence, not MIT.
* Built during the hackathon window with the help of an AI coding assistant (Claude), as the rules permit when
  disclosed. Open-source libraries: Qiskit, Qiskit Aer, OR-Tools, sgp4, NumPy, SciPy, Matplotlib.
