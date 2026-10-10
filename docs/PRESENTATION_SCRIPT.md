# FairPass: 2–3 minute demo-video script

*About 380 spoken words, roughly 2 min 40 s at a calm pace. Screen directions are in [brackets].*

---

**[0:00 – Title slide: "FairPass: who gets the antenna?"]**
Hundreds of university and amateur satellites depend on open ground-station networks like SatNOGS to hear from
home. Each station schedules itself, and antenna time is the scarce resource.
We asked one question: **which passes should a group of stations take so that more data comes down, without leaving
missions with no contact at all?**

**[0:20 – Fig. 1]**
First, we tested ourselves against real records. We downloaded about four thousand vetted SatNOGS observations from
ten European stations. From the orbit data stored with each observation, we reproduce the network's own pass
geometry to a quarter of a degree. Then we fitted a pass-success model and tested it on *later* days. Geometry alone
barely predicts success: AUC 0.65. Knowing which satellites are actually transmitting gives 0.96. On the target day,
the model predicted 167 successful observations and 165 happened.

**[0:55 – Fig. 2]**
Then we generated schedules for a real day, the first of October: ten stations, almost eleven thousand candidate
passes. Compared with each station scheduling itself, coordinating them delivers **44% more expected data**.
Throughput-only optimisation still leaves 49 active missions with no contact. Those are mostly the low-data-rate
cubesats.

**[1:20 – Fig. 3a]**
Adding a fairness term brings that down to **22 missions for 3% less data, or 12 for 8% less**. This curve is the decision a network
operator actually makes.

**[1:35 – Qollab demo running, one IonQ job]**
Where is the quantum step? For a crowded 20-minute window, the candidate passes become qubits. A QAOA circuit, with
angles learned on *morning* windows and reused unchanged in the *afternoon*, runs as **a single job** on IonQ's
simulator. Each measurement is an antenna plan.
On held-out windows it finds the best plan **55 to 400 times more often** than random sampling with the same number of
shots. And on IonQ's own Forte 1 noise model, run through Qollab, it still hit the optimal plan 16 times more often
than random, and its best sample was the optimum.

**[2:05 – Fig. 4]**
Now the honest part. Simulated annealing and plain greedy do just as well at the same budget. Inside the full
scheduler, swapping them for QAOA changes the day's result by less than one percentage point. Plain local search over
the whole day beats every windowed hybrid, and these windows are small enough to solve exactly. So we claim a measured principle, not a speed-up.

**[2:25 – Limitations slide]**
What's still unverified: these are model expectations, not an executed schedule. Data volume is a baud-rate proxy, and
mission priorities are assumptions. The next real experiment: run the recommended plan on the network for one day,
vet the outcomes, and compare.
FairPass is open source, MIT for code, CC BY-SA for SatNOGS data. This effort is supported by Qollab and IonQ.

**[2:45 – end]**

---

### Answers to the questions judges will ask
* **Who needs this?** Operators of open or shared ground-station networks (SatNOGS, university networks) and the
  small missions that depend on them.
* **What decision does it improve?** Which passes each antenna takes over the next day, and how much throughput to trade
  for fairness (λ).
* **What did we test on existing data?** Pass geometry against the network's records, and pass success on held-out
  days and on the target day.
* **What did we generate ourselves?** Schedules, the fairness frontier, QAOA output distributions, noise sweeps and the
  hybrid runs. All are model-based.
* **What is new?** Outcome-grounded values and an exact expected fairness/redundancy QUBO, plus a matched-budget,
  held-out, noise-swept test of a QAOA window step.
* **What remains unverified?** Real execution of the plan, byte-level volumes, mission priorities, and QAOA on IonQ
  noise models or hardware.
