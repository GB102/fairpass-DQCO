# 1. Feasibility assessment and go/no-go

*Written at about the 15% mark of the build window (2026-10-10, 08:30 UTC). Later results are in `REPORT.md`.*

## The question as proposed

> Given a limited set of ground stations and a fixed contact budget, which satellite passes should be
> scheduled to maximise priority-weighted downlinked data across as many satellites as possible,
> without leaving missions behind?

The proposed method was a QUBO with one binary variable per pass, solved with QAOA, warm-start QAOA
or CVaR-QAOA on 12–20-pass windows, and benchmarked against greedy scheduling and CP-SAT.

## Hackathon constraints (verified 2026-10-10 against qollab.xyz/programs/hackathon and /terms)

* Build window: 2026-10-09 19:00 ET to 2026-10-11 19:00 ET. Submission is due at the end of the window.
* The project must use Qiskit, be published runnable on Qollab under MIT, and include a project page
  and a 2–3 minute public demo video. It must also include the line "This effort is supported by Qollab & IonQ".
* **Judges run projects on a free local simulator. IonQ simulator or hardware results "neither add nor
  subtract points".** The Qollab playground runs circuits on the IonQ noise-model simulators
  (aria-1, forte-1, ...). The public rules list **no** job, circuit, qubit or shot limits. Because a
  single-job-per-run limit can't be ruled out, the demo is built to need **one circuit submission**.
* Quantum Impact theme: "public dataset in logistics, manufacturing, **space**, ... build a strong
  classical baseline, then show where a quantum subroutine makes a measurable difference."
* A submission must go beyond tutorial level. Generic QAOA-on-a-QUBO risks a zero on Novelty.

## Closest prior work

| Work | What it showed | Gap relevant to us |
|---|---|---|
| Leidreiter, Petrak, Anderle, Prüfer, *QUARGS – Quantum Reinforced Ground Station Scheduling*, SpaceOps 2025 (DLR GSOC) | Ground-station scheduling for a MEO constellation as a QUBO/PUBO. Compares QA, QAOA and E-VQE with SCIP on 16 instances of 5–20 contact slots built from real visibilities. Noiseless simulation. QAOA finds optima only on the smallest instances. | Objective is "maximise number of contacts". Fairness is mentioned but **not used** in the quantum instances. No link-quality data, no noise, no matched-budget sampling baseline. |
| Fagiolo, Oddi, Rasconi, Baioletti, *Applying QAOA to the Oversubscribed Satellite Scheduling Problem* (2025/26) | OSSP as a QUBO, with layer-wise QAOA parameter growth. Simulator only. | Same: synthetic task values, no outcome data, no fairness. |
| Makarov et al., *Quantum Optimization Methods for Satellite Mission Planning*, IEEE Access 12 (2024) | EO-satellite planning: QA and QAOA on a realistic dataset, with bounds on solvable sizes. | EO imaging, not downlink. Quality degrades with size. Framed as a baseline, not an advantage. |
| Stollenwerk et al., *Agile EO satellite scheduling with a quantum annealer*, IEEE TAES (2021) | QUBO for EO scheduling on D-Wave. | Annealing, EO imaging. |
| Classical satellite range scheduling: Barbulescu et al. (2004, AFSCN); Vazquez & Erwin (2014, tractability); Spangelo et al. (2015); Lemaître, Verfaillie et al. (fair sharing of EO satellites, 2003) | Range scheduling is NP-hard in general but solved well in practice by MIP/CP and heuristics. Fair allocation is an established classical topic. | None of these uses open-network outcome data such as SatNOGS. |
| SatNOGS `satnogs-auto-scheduler` (AGPL-3.0) | The tool open-network stations actually use. It schedules **one station at a time**, walking a priority-ordered or shuffled pass list and keeping each pass that doesn't overlap one already kept. | No network coordination and no fairness. It's a real-world greedy baseline. |
| QAOA methodology: Farhi et al. 2014; Hadfield et al. 2019 (constraint-preserving mixers); Barkoutsos et al. 2020 (CVaR); Egger et al. 2021 (warm start); Brandão et al. 2018 / Galda et al. 2021 (parameter concentration/transfer) | Established techniques. | We use them; we don't claim them. |

## Assessment of the original hypotheses

1. **"QAOA improves the schedule."** At 12–20 binary variables, exhaustive search (≤ 10⁶ states)
   and CP-SAT give the exact optimum in milliseconds. A QAOA *simulation* at this size can't
   improve on exact classical optimisation, and hardware wouldn't change that. Any claimed
   "improvement" would have to be over a deliberately weaker classical method. **As stated,
   this is not defensible.**
2. **"Fairness changes the schedule."** This is plausible and testable on real data, but it is a
   classical decision-science result.
3. **Weather layer (Open-Meteo / SRTM).** Most SatNOGS stations receive amateur VHF/UHF downlinks
   (145/435 MHz), where rain attenuation is negligible (< 0.01 dB/km). A weather layer would add
   complexity with no physical effect. **Dropped.** Terrain is already in each station's
   `min_horizon` setting.
4. **"A satellite can talk to only one station at a time."** That holds for commanded downlinks.
   It is wrong for the broadcast telemetry that dominates SatNOGS, where several stations can
   receive the same pass at once and the second copy is *redundant* rather than infeasible. We
   model both cases with the same pairwise QUBO term: a hard conflict in commanded mode, and a
   redundancy discount equal to the overlap value in broadcast mode.

## What real data makes possible (checked before committing)

The SatNOGS Network API (CC BY-SA 4.0) exposes each historical observation with:

* the station, its coordinates and horizon mask;
* the exact TLE used;
* start and end times, maximum elevation and azimuths;
* transmitter mode and baud rate;
* the vetted outcome (good/bad/failed);
* the number of demodulated frames.

These records let us:

* **reproduce** the network's own pass geometry (max elevation) from the stored TLEs, which checks the propagation layer;
* **fit and validate on held-out time** a pass-success model, P(good | geometry, station, mode), so pass *value*
  rests on measured outcomes rather than an assumed link budget;
* **compare against what the network actually scheduled** on a real day, judged by the same validated model.

## Narrowed project (GO)

**Decision: GO, with a narrowed and re-weighted contribution.**

* **H1 (data, scientific/empirical).** Pass geometry alone (duration or max elevation) mis-ranks passes relative to
  expected delivered data from a success model fitted on real observations. *Test:* time-split
  held-out log-loss and AUC against geometry-only and constant baselines.
* **H2 (decision, methodological).** On a real network-day, coordinated scheduling (exact CP-SAT on
  the QUBO objective) beats independent per-station greedy scheduling (what SatNOGS stations run).
  A concave per-mission fairness term removes starved missions at a measurable throughput cost.
  *Test:* paired comparison under the validated model, with bootstrap uncertainty.
  *Rejection:* coordination gain within the noise, or fairness cost larger than the coverage gain.
* **H3 (quantum, narrow).** Inside a classical large-neighbourhood search, a QAOA step re-optimises
  conflict-dense windows of 8–14 passes, using parameters trained on *other* windows and transferred
  without any runtime optimisation. *Test:* against a **matched-shot** random-sampling step,
  simulated annealing, and an exact subsolver, under IonQ-like noise. Report the noise/size boundary
  where any QAOA benefit over sampling disappears. *Expected and accepted outcome:* QAOA never beats the exact
  classical subsolver; the open question is whether it beats matched-budget sampling and for how long under noise.

**Contribution type.** H1 is empirical and data-grounded (new for quantum-scheduling work, which uses synthetic
values). H2 is methodological plus engineering: an open, reproducible network-coordinated scheduler on
real SatNOGS data, and a measured fairness/throughput frontier. H3 is a *small, honest quantum
measurement*: amplitude concentration from transferred parameters and its noise boundary. **No speedup is claimed.**

**Is it substantial enough?** For a 48-hour hackathon, yes. Prior quantum-scheduling studies use synthetic
values and skip matched-budget baselines; adding real outcome data, fairness and a noise boundary is a
credible narrow step. The main impact (H1/H2) is classical, and we will say so.

## Background requested in the brief that does not apply here

The generalized amplitude-damping capacity results (classical capacity, strong-converse exponent) belong to the
quantum-memory direction (MemoryMap) that we did not pursue. They give no bound relevant to a classical scheduling
QUBO, so we don't use them.
