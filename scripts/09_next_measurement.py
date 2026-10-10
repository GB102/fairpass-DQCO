"""Step 9: which single measurement campaign would most improve the scheduling decision?

For each satellite s we ask: if we knew its success probability exactly, how much better would the
day's fairness-aware schedule be (partial expected value of perfect information, EVPI_s)?
  * uncertainty: the 40 observation-level bootstrap replicates of the success model
  * for replicate b, only satellite s's passes take their replicate probabilities; all else nominal
  * the nominal schedule is re-optimised locally - CP-SAT on every pass that overlaps a pass of s at the
    same station or the same satellite (all other passes fixed) - and the objective gain is the regret
  * EVPI_s = mean regret over replicates (kB/day under the replicate's own model)
A satellite with a high EVPI is one whose liveness/link quality is both uncertain and decision-relevant.
The recommended campaign: vetted observations of the top satellite(s) at high elevation, sized by the
binomial standard error needed to settle the decision.
Output: results/next_measurement.json
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fairpass import MODEL_VERSION  # noqa: E402
from fairpass.classical import exact_cpsat  # noqa: E402
from fairpass.problem import Contacts, build_problem  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
LAM, K = 200.0, 2


def neighbourhood(c, s, tau=60.0):
    """Passes of satellite s plus every pass that overlaps one of them at the same station."""
    own = np.flatnonzero(c.sat == s)
    nb = set(own.tolist())
    for i in own:
        m = (c.station == c.station[i]) & (c.aos < c.los[i] + tau) & (c.los + tau > c.aos[i])
        nb |= set(np.flatnonzero(m).tolist())
    return np.array(sorted(nb))


def main():
    inst = json.load(open(ROOT / "data/processed/instance_2026-10-01.json"))
    c = Contacts.from_dict(inst["contacts"])
    pboot = np.array(inst["p_bootstrap"])
    act = [int(s) for s in np.unique(c.sat) if c.p[c.sat == s].mean() >= 0.3]
    ss = json.load(open(ROOT / "results/schedule_study.json"))
    x_nom = np.zeros(c.n, int)
    x_nom[ss["x"][f"cpsat_fair_{int(LAM)}"]] = 1
    obs_counts = {}
    for name in ("train", "test"):
        import glob
        for p in glob.glob(str(ROOT / f"data/raw/obs/{name}_*.json")):
            for r in json.load(open(p, encoding="utf-8")):
                if r["status"] in ("good", "bad"):
                    obs_counts[r["norad_cat_id"]] = obs_counts.get(r["norad_cat_id"], 0) + 1
    rows = []
    sats = np.unique(c.sat)
    # screen: satellites whose replicate spread in mean p is non-trivial
    spread = {int(s): float(np.std(pboot[:, c.sat == s].mean(1))) for s in sats}
    cand = [s for s in sats if spread[int(s)] > 0.02]
    print(f"{len(cand)} of {len(sats)} satellites have bootstrap spread > 0.02")
    for s in cand:
        idx_s = np.flatnonzero(c.sat == s)
        nb = neighbourhood(c, s)
        regrets = []
        for b in range(len(pboot)):
            p = c.p.copy()
            p[idx_s] = pboot[b, idx_s]
            cb = Contacts(c.sat, c.station, c.aos, c.los, c.max_el, c.rate, p, c.weight)
            pb = build_problem(cb, lam=LAM, K=K, active=act)
            sub, kept = pb.restrict(nb, x_nom)
            if sub.n == 0:
                regrets.append(0.0)
                continue
            xs, info = exact_cpsat(sub, time_limit=5, workers=1)
            y = x_nom.copy()
            y[nb] = 0
            y[kept] = xs
            regrets.append(max(0.0, pb.objective(y)[0] - pb.objective(x_nom)[0]))
        rows.append({"norad": int(s), "evpi_kB": float(np.mean(regrets)), "regret_p90": float(np.percentile(regrets, 90)),
                     "p_nominal_mean": float(c.p[idx_s].mean()), "p_boot_sd": spread[int(s)],
                     "n_vetted_obs": int(obs_counts.get(int(s), 0)), "n_candidate_passes": int(len(idx_s)),
                     "baud": float(c.rate[idx_s][0]), "scheduled_nominal": int(x_nom[idx_s].sum())})
    rows.sort(key=lambda r: -r["evpi_kB"])
    total = float(sum(r["evpi_kB"] for r in rows))
    out = {"model_version": MODEL_VERSION, "lam": LAM, "K": K, "ranking": rows, "sum_partial_evpi_kB": total,
           "day_objective_kB": float(build_problem(c, lam=LAM, K=K, active=act).objective(x_nom)[0])}
    if rows:
        top = rows[0]
        # passes needed so that the binomial SE on p is <= 0.1 around the nominal value
        pn = min(max(top["p_nominal_mean"], 0.05), 0.95)
        n_needed = int(np.ceil(pn * (1 - pn) / 0.1 ** 2))
        out["recommendation"] = {"norad": top["norad"], "extra_vetted_passes": n_needed,
                                 "evpi_kB_per_day": top["evpi_kB"]}
    (ROOT / "results/next_measurement.json").write_text(json.dumps(out, indent=1))
    for r in rows[:10]:
        print(r)
    print("recommendation:", out.get("recommendation"))


if __name__ == "__main__":
    main()
