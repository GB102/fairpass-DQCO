"""Step 5 (H2): classical scheduling study on the real network-day instance.

Schedulers compared on the same candidate set, judged by the same validated success model:
  actual            what the 10 stations really observed on 2026-10-01 (SatNOGS records)
  station_maxel     per-station greedy by max elevation   (geometry-only priorities, no coordination)
  station_value     per-station greedy by expected data   (model-informed, no coordination ~ auto-scheduler)
  network_greedy    coordinated greedy + local search      (lam = 0)
  cpsat             coordinated exact optimum              (lam = 0, throughput-optimal)
  cpsat_fair[lam]   coordinated exact optimum with the concave fairness term, lam sweep (K = 2)
Uncertainty: every schedule is re-scored under 40 bootstrap replicates of the success model; for the
throughput-optimal and the recommended fair schedule we also re-optimise under 10 replicates and report
the regret of the nominal schedule (decision robustness).
Outputs: results/schedule_study.json
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fairpass import MODEL_VERSION  # noqa: E402
from fairpass.classical import exact_cpsat, local_search, network_greedy, station_greedy  # noqa: E402
from fairpass.problem import Contacts, FastEval, build_problem, schedule_metrics  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
K = 2
LAMS = [0, 50, 100, 200, 400, 800]
ACTIVE_P = 0.3
TIME_LIMIT = 180


def load_instance():
    inst = json.load(open(ROOT / "data/processed/instance_2026-10-01.json"))
    c = Contacts.from_dict(inst["contacts"])
    return inst, c


def active_set(c, pmat=None):
    """Active missions: satellites seen in training whose mean predicted pass success >= ACTIVE_P."""
    return [int(s) for s in np.unique(c.sat) if c.p[c.sat == s].mean() >= ACTIVE_P]


def with_p(c, p):
    d = {f: getattr(c, f) for f in Contacts.FIELDS}
    d["p"] = np.asarray(p)
    return Contacts(**d)


def boot_scores(c, x, pboot, act, fe):
    vals = [schedule_metrics(c, x, K=K, active=act, fast=fe, p=pb) for pb in pboot]
    keys = ["expected_unique_kB", "starved", "exp_successful_lt1", "jain_share"]
    return {k: {"mean": float(np.mean([v[k] for v in vals])), "sd": float(np.std([v[k] for v in vals])),
                "p05": float(np.percentile([v[k] for v in vals], 5)),
                "p95": float(np.percentile([v[k] for v in vals], 95))} for k in keys}


def main():
    inst, c = load_instance()
    pboot = np.array(inst["p_bootstrap"])
    act = active_set(c)
    print(f"candidates {c.n}  satellites {len(np.unique(c.sat))}  active missions {len(act)}")
    out = {"model_version": MODEL_VERSION, "K": K, "active_p": ACTIVE_P, "n_candidates": c.n,
           "n_satellites": int(len(np.unique(c.sat))), "n_active": len(act), "schedules": {}}
    pb0 = build_problem(c, lam=0, K=K, active=act)
    fe = FastEval(c)
    out["n_hard_pairs"], out["n_redundancy_pairs"] = len(pb0.hard), len(pb0.J)
    xs = {}

    x_act = np.zeros(c.n, int)
    x_act[inst["actual"]["index"]] = 1
    xs["actual"] = x_act
    st = inst["actual"]["status"]
    out["actual_outcomes"] = {"n": len(st), "good": st.count("good"), "bad": st.count("bad"),
                              "failed": st.count("failed"), "unknown": st.count("unknown"),
                              "predicted_successes": float(c.p[x_act == 1].sum()),
                              "unmatched": inst["actual"]["unmatched_observations"]}
    # schedule-level check of the model: predicted vs realised successes among vetted actual observations
    idx = np.array(inst["actual"]["index"])
    vet = np.array([s in ("good", "bad") for s in st])
    if vet.any():
        out["actual_outcomes"]["vetted_predicted_good"] = float(c.p[idx[vet]].sum())
        out["actual_outcomes"]["vetted_realised_good"] = int(sum(s == "good" for s in st))
        pv = pboot[:, idx[vet]].sum(1)
        out["actual_outcomes"]["vetted_predicted_good_boot_p05_p95"] = [float(np.percentile(pv, 5)),
                                                                        float(np.percentile(pv, 95))]

    xs["station_maxel"] = station_greedy(c, pb0, key="max_el", tau=inst["tau_s"])
    xs["station_value"] = station_greedy(c, pb0, key="value", tau=inst["tau_s"])
    t = time.perf_counter()
    xs["network_greedy"] = local_search(pb0, network_greedy(pb0))
    out["network_greedy_wall_s"] = time.perf_counter() - t

    infos = {}
    hint = xs["network_greedy"]
    for lam in LAMS:
        pb = build_problem(c, lam=lam, K=K, active=act)
        x, info = exact_cpsat(pb, time_limit=TIME_LIMIT, hint=hint)
        x = local_search(pb, x)  # removes any integer-rounding suboptimality of the CP-SAT objective
        name = "cpsat" if lam == 0 else f"cpsat_fair_{lam}"
        xs[name], infos[name] = x, info
        hint = x
        print(name, info["status"], f"{info['wall_s']:.1f}s", schedule_metrics(c, x, K=K, active=act, fast=fe),
              flush=True)

    for name, x in xs.items():
        m = schedule_metrics(c, x, K=K, active=act, fast=fe)
        m["boot"] = boot_scores(c, x, pboot, act, fe)
        if name in infos:
            m["solver"] = infos[name]
        out["schedules"][name] = m

    # starved missions under the throughput optimum: what kind of missions are they?
    starved = [s for s in act if not xs["cpsat"][c.sat == s].any()]
    out["starved_under_cpsat"] = [{"norad": int(s), "baud": float(c.rate[c.sat == s][0]),
                                   "n_candidates": int((c.sat == s).sum()),
                                   "mean_p": float(c.p[c.sat == s].mean())} for s in starved]
    out["active_baud_median"] = float(np.median([c.rate[c.sat == s][0] for s in act]))

    # model-assumption sensitivity: re-optimise under an alternative assumption, score under the primary model
    sens = {}
    lam_s = 200
    alts = {
        "commanded_downlink": dict(c=c, mode="commanded"),
        "value_contact_time": dict(c=Contacts(c.sat, c.station, c.aos, c.los, c.max_el,
                                              np.full(c.n, float(np.median(c.rate))), c.p, c.weight),
                                   mode="broadcast"),
        "geometry_only_success": dict(c=with_p(c, np.array(inst["contacts"]["p_geom"])), mode="broadcast"),
    }
    for name, a in alts.items():
        pa = build_problem(a["c"], lam=lam_s, K=K, mode=a["mode"], active=act)
        xa, info = exact_cpsat(pa, time_limit=TIME_LIMIT, hint=xs.get(f"cpsat_fair_{lam_s}"))
        m = schedule_metrics(c, xa, K=K, active=act, fast=fe)
        ref = xs.get(f"cpsat_fair_{lam_s}")
        jac = float(((xa == 1) & (ref == 1)).sum() / max(((xa == 1) | (ref == 1)).sum(), 1)) if ref is not None else None
        sens[name] = {"metrics_under_primary_model": m, "boot": boot_scores(c, xa, pboot, act, fe),
                      "jaccard_vs_primary_schedule": jac, "solver": info}
        print("sensitivity", name, m, "jaccard", jac, flush=True)
    out["sensitivity"] = sens

    # decision robustness: regret of nominal schedules under re-optimisation on bootstrap replicates
    rob = {}
    for name, lam in [("cpsat", 0), ("cpsat_fair_200", 200)]:
        if name not in xs:
            continue
        regrets = []
        for b in range(min(10, len(pboot))):
            cb = with_p(c, pboot[b])
            pbb = build_problem(cb, lam=lam, K=K, active=act)
            xb, _ = exact_cpsat(pbb, time_limit=60, hint=xs[name])
            fb, fn = pbb.objective(xb)[0], pbb.objective(xs[name])[0]
            regrets.append((fb - fn) / max(fb, 1e-9))
        rob[name] = {"regret_rel_mean": float(np.mean(regrets)), "regret_rel_max": float(np.max(regrets))}
    out["robustness"] = rob
    out["x"] = {k: np.flatnonzero(v).tolist() for k, v in xs.items()}
    (ROOT / "results/schedule_study.json").write_text(json.dumps(out, indent=1))
    for k, v in out["schedules"].items():
        print(f"{k:18s} kB={v['expected_unique_kB']:10.0f} contacts={v['n_contacts']:4d} "
              f"starved={v['starved']:3d}/{len(act)} jain={v['jain_share']:.3f}")


if __name__ == "__main__":
    main()
