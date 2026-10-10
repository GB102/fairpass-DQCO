# FairPass: fairness-aware contact scheduling for open ground-station networks

*Qollab × IonQ Quantum Hackathon 2026, Quantum Impact theme.*
**This effort is supported by Qollab & IonQ.**

Small-satellite teams depend on open ground-station networks such as [SatNOGS](https://network.satnogs.org).
Antenna time is scarce, and each station schedules itself. FairPass asks a narrow question on real network data:

> Which passes should a set of stations take so that the most data is delivered **without leaving active missions with
> no contact at all**? And can a small QAOA step inside a classical scheduler do anything measurable?

Pipeline:

```
public data (SatNOGS observations + stored TLEs, SatNOGS DB)
  -> SGP4 passes, validated against the network's own records
  -> pass-success model fitted on vetted outcomes, validated on later days
  -> candidate-contact QUBO (value, redundancy, concave fairness, antenna conflicts)
  -> schedules: per-station greedy (today), coordinated greedy, CP-SAT optimum, hybrid LNS with a QAOA window step
  -> recommendation with bootstrap uncertainty + the next measurement worth making
```

Results and limitations are in [`REPORT.md`](REPORT.md). The video script is in
[`docs/PRESENTATION_SCRIPT.md`](docs/PRESENTATION_SCRIPT.md), and the Qollab project page text in
[`demo/qollab/PROJECT_CARD.md`](demo/qollab/PROJECT_CARD.md). The go/no-go assessment written
at the 15% mark is in [`docs/01_feasibility.md`](docs/01_feasibility.md).

## Run it

**Qollab (judging path).** The [`demo/qollab`](demo/qollab) folder is a self-contained Qollab Python project
(`main.py`, `fairpass_core.py`, `data/demo_bundle.json`). Press *Run on Qollab* and pick any backend. The run sends
**one** job: a 10–12-qubit QAOA circuit for one held-out scheduling window. Everything else is precomputed and labelled
as such in the output.

**Locally.** Windows/macOS/Linux, Python 3.12:

```bash
python -m venv .venv && .venv/Scripts/python -m pip install -r requirements.txt
```

```bash
cd demo/qollab && ../../.venv/Scripts/python main.py
```

The demo falls back to Qiskit Aer, or to the BasicSimulator, when no Qollab `backend` exists.

**Full pipeline** (data download is throttled by SatNOGS to about 60 requests/hour and is cached in `data/cache/`):

| step | script | output |
|---|---|---|
| 1 | `scripts/01_fetch_data.py` | `data/raw/` (stations, DB, observations per station and period) |
| 2 | `scripts/02_validate_passes.py` | `results/pass_validation.*` (SGP4 vs network records) |
| 3 | `scripts/03_fit_link_model.py` | `results/link_model*.json`, `results/fig1_data.json` |
| 4 | `scripts/04_build_instance.py` | `data/processed/instance_2026-10-01.json` |
| 5 | `scripts/05_schedule_study.py`, `05b_paired_differences.py` | `results/schedule_study.json`, `results/paired_differences.json` |
| 6 | `scripts/06_qaoa_study.py` | `results/qaoa_study.json`, `results/qaoa_params.json` |
| 7 | `scripts/07_hybrid_lns.py` | `results/hybrid_lns.json` |
| 8 | `scripts/08_figures.py` | `figures/fig1–4.{png,svg}` |
| 9 | `scripts/09_next_measurement.py` | `results/next_measurement.json` |
| 10 | `scripts/10_build_demo.py` | `demo/qollab/data/demo_bundle.json` |
| 11 | `scripts/11_manifest.py` | `docs/DATA_MANIFEST.md` |
| 12 | `scripts/12_forward_plan.py` | `results/forward_plan*.{json,csv}` (tomorrow's plan from current CelesTrak TLEs) |

Tests: `python -m tests.test_core` checks CP-SAT against brute force and sub-problem consistency. `python -m tests.test_qaoa`
checks the numpy QAOA simulator against the Qiskit circuit.

## Repository map

```
fairpass/        library: data_sources, passes (SGP4), linkmodel, problem (QUBO), classical, qaoa, hybrid
scripts/         numbered pipeline steps (above)
demo/qollab/     the Qollab project (MIT)
data/            raw SatNOGS extracts (CC BY-SA 4.0), processed instance, provenance log
results/         every number in the report, as JSON (model version, seeds, parameters)
figures/         the three report figures
docs/            feasibility memo, data manifest
```

## Licences and attribution

* Code and documentation: MIT (see `LICENSE`).
* CelesTrak GP data (`active` group, cached once for the forward plan) is © CelesTrak, used under its usage policy.
* SatNOGS Network and SatNOGS DB data (observations, stations, transmitters, and the TLEs stored with observations):
  © Libre Space Foundation and SatNOGS contributors, **CC BY-SA 4.0**. Files under `data/` derived from them keep that
  licence. The data are not relicensed under MIT. See [`docs/DATA_MANIFEST.md`](docs/DATA_MANIFEST.md).
* The SatNOGS auto-scheduler (AGPL-3.0) was read for its algorithm only. No code was copied. Our per-station greedy
  baseline is an independent re-implementation of the same rule.
* AI assistance: the code, analysis and text were developed with an AI coding assistant (Claude) during the build
  window, as the hackathon rules permit when disclosed.
