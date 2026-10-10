"""Step 10: package precomputed results + held-out QAOA windows into demo/qollab/data/demo_bundle.json.

Window selection is stated in the bundle: window 0 is the held-out (afternoon) window in which today's
uncoordinated greedy loses most against the window optimum; the rest follow in time order. Aggregate
statistics over *all* held-out windows are in REPORT.md - the demo window is illustrative, not evidence.
"""
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fairpass import MODEL_VERSION  # noqa: E402
from fairpass.classical import brute_force, station_greedy  # noqa: E402
from fairpass.hybrid import extract_windows  # noqa: E402
from fairpass.problem import Contacts, build_problem  # noqa: E402
from fairpass.qaoa import NumpyQAOA  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "demo/qollab/data/demo_bundle.json"
LAM, K, N = 200.0, 2, 10
MAX_WINDOWS = 12


def main():
    sys.path.insert(0, str(ROOT / "scripts"))
    from importlib import import_module
    q6 = import_module("06_qaoa_study")
    inst = json.load(open(ROOT / "data/processed/instance_2026-10-01.json"))
    c = Contacts.from_dict(inst["contacts"])
    t0 = datetime.fromisoformat(inst["t0"])
    act = [int(s) for s in np.unique(c.sat) if c.p[c.sat == s].mean() >= 0.3]
    pb = build_problem(c, lam=LAM, K=K, active=act)
    x_inc = station_greedy(c, pb, key="value", tau=inst["tau_s"])
    names = {s["norad_cat_id"]: s["name"] for s in
             json.load(open(ROOT / "data/raw/satnogs_db_satellites.json", encoding="utf-8")) if s.get("norad_cat_id")}
    st_names = {s["id"]: s["name"] for s in inst["stations"]}
    params = json.load(open(ROOT / "results/qaoa_params.json"))
    ss = json.load(open(ROOT / "results/schedule_study.json"))
    ev = json.load(open(ROOT / "results/link_model_eval.json"))
    pv = json.load(open(ROOT / "results/pass_validation.json"))
    nm = json.load(open(ROOT / "results/next_measurement.json")) if (ROOT / "results/next_measurement.json").exists() else None

    wins = [w for w in extract_windows(c, pb, x_inc, n=N) if w["t"] >= 43200.0]
    recs = []
    for w in wins:
        sub, kept = w["sub"], w["kept"]
        Q = sub.qubo_matrix()
        xo, f = brute_force(sub)
        fmax = f.max()
        optima = [list(map(int, ((k >> np.arange(N)) & 1))) for k in np.flatnonzero(np.isclose(f, fmax))]
        inc = x_inc[kept].astype(int)
        f_inc = float(sub.penalised(inc)[0])
        sim = NumpyQAOA(Q)
        pr_ideal, pr_noisy, prm = {}, {}, {}
        for p in (1, 2, 3):
            k = f"n{N}_p{p}_energy"
            g, b = np.array(params[k]["gammas"]), np.array(params[k]["betas"])
            prm[str(p)] = {"gammas": g.tolist(), "betas": b.tolist()}
            opt_mask = np.isclose(f, fmax)
            pr_ideal[str(p)] = float(sim.probs(g, b)[opt_mask].sum())
            q6.READOUT = 0.005
            pr_noisy[str(p)] = float(q6.noisy_probs(Q, g, b, 0.004)[opt_mask].sum())
        passes = []
        for i in kept:
            passes.append({"norad": int(c.sat[i]), "name": names.get(int(c.sat[i]), str(int(c.sat[i]))),
                           "station": st_names.get(int(c.station[i]), str(int(c.station[i]))),
                           "aos": (t0 + timedelta(seconds=float(c.aos[i]))).strftime("%H:%M:%S"),
                           "los": (t0 + timedelta(seconds=float(c.los[i]))).strftime("%H:%M:%S"),
                           "max_el": round(float(c.max_el[i]), 1), "p": round(float(c.p[i]), 3),
                           "value_kB": round(float(c.value_kB()[i]), 1)})
        recs.append({"t": w["t"], "t_utc": (t0 + timedelta(seconds=w["t"])).strftime("%H:%M") + "-" +
                     (t0 + timedelta(seconds=w["t"] + 1200)).strftime("%H:%M"),
                     "n": N, "Q": np.round(Q, 4).tolist(), "hard": [list(h) for h in sorted(sub.hard)],
                     "n_couplings": len(sub.J), "passes": passes, "incumbent": inc.tolist(),
                     "f_opt": float(fmax), "optima": optima, "n_optima": len(optima),
                     "gap_incumbent": float((fmax - f_inc) / fmax), "params": prm,
                     "precomputed": {"p_opt_ideal": pr_ideal, "p_opt_forte_like": pr_noisy}})
    # default window: the held-out window whose optimal plan keeps the most passes (a busy, readable example);
    # ties -> larger loss of today's greedy
    recs.sort(key=lambda r: (-max(sum(o) for o in r["optima"]), -r["gap_incumbent"], r["t"]))
    first = recs[:1]
    rest = sorted(recs[1:], key=lambda r: r["t"])[:MAX_WINDOWS - 1]
    windows = first + rest
    for i, r in enumerate(windows):
        r["selection_note"] = ("Default: the held-out window whose optimal plan keeps the most passes."
                               if i == 0 else "Held-out window (time order).")

    label = {"actual": "Actual SatNOGS schedule that day", "station_maxel": "Per-station greedy by elevation",
             "station_value": "Per-station greedy by expected data", "network_greedy": "Coordinated greedy",
             "cpsat": "Coordinated optimum (throughput only)",
             f"cpsat_fair_{int(LAM)}": f"Coordinated optimum + fairness (lambda={int(LAM)})"}
    sched = {}
    for k, lab in label.items():
        if k in ss["schedules"]:
            s = ss["schedules"][k]
            sched[lab] = {"MB": s["boot"]["expected_unique_kB"]["mean"] / 1000,
                          "MB_p05": s["boot"]["expected_unique_kB"]["p05"] / 1000,
                          "MB_p95": s["boot"]["expected_unique_kB"]["p95"] / 1000, "starved": s["starved"]}
    fair, thr, inc_s = (ss["schedules"].get(f"cpsat_fair_{int(LAM)}"), ss["schedules"]["cpsat"],
                        ss["schedules"]["station_value"])
    rec_text = (f"coordinate the {len(inst['stations'])} stations with the fairness-aware optimum (lambda={int(LAM)}): "
                f"{fair['starved']} starved missions vs {thr['starved']} (throughput-only) and {inc_s['starved']} "
                f"(uncoordinated), at {100 * (1 - fair['expected_unique_kB'] / thr['expected_unique_kB']):.1f}% "
                f"less expected data than the throughput optimum.") if fair else "see REPORT.md"
    sens = ss.get("sensitivity", {}).get("value_uncapped_listed_baud")
    nm_text = None
    if sens:
        fair_kb = ss["schedules"][f"cpsat_fair_{int(LAM)}"]["expected_unique_kB"]
        loss = 1 - sens["metrics_under_primary_model"]["expected_unique_kB"] / fair_kb
        head = (f"success-rate uncertainty barely matters (partial EVPI {nm['sum_partial_evpi_kB']:.0f} kB/day); "
                if nm else "")
        nm_text = head + (
                   f"the plan hinges on how data rate is valued: capping listed baud at 9,600 bit/s or not changes "
                   f"{100 * (1 - sens['jaccard_vs_primary_schedule']):.0f}% of chosen passes and {100 * loss:.0f}% of "
                   f"expected data. Measure it: a one-day decode campaign (>= 15 vetted passes per rate class) on the "
                   f"high-rate and LoRa satellites, counting decoded frames.")
    bundle = {
        "meta": {"model_version": MODEL_VERSION, "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                 "data_licence": "Derived from SatNOGS Network/DB data, CC BY-SA 4.0 (Libre Space Foundation)",
                 "values": "measured = SatNOGS observation outcomes; fitted = success model; simulated = schedules, "
                           "QAOA distributions; hypothetical = equal mission weights"},
        "day": {"date": inst["t0"][:10], "stations": [s["name"] for s in inst["stations"]],
                "n_satellites": ss["n_satellites"], "n_candidates": ss["n_candidates"], "n_active": ss["n_active"],
                "validation": {"geom_median_abs_deg": pv["max_el_error_deg"]["median_abs"], "geom_n": pv["n_matched"],
                               "auc_full": ev["full"]["auc"], "auc_geom": ev["geom"]["auc"],
                               "top30_full": ev["full"]["top30_success"], "top30_maxel": ev["top30_by_max_el_success"]},
                "schedules": sched, "recommendation": {"lam": LAM, "text": rec_text},
                "next_measurement": {"text": nm_text} if nm_text else None},
        "windows": windows,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(bundle))
    print(f"bundle: {len(windows)} windows, {OUT.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
