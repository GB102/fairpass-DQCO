"""Step 3: fit and validate the pass-success model on real SatNOGS outcomes (time split).

H1 test: does a model fitted on observed outcomes rank passes better than pass geometry alone?
  models: const | geom (max elevation) | sat (per-satellite rate) | full (geom + mode + station + satellite)
  held-out metrics on the later test period: log-loss, Brier, AUC, calibration, and the decision metric
  "success rate of the top-30% passes a station would pick" under each ranking.
Outputs: results/link_model.json (fitted full model + bootstrap replicates), results/link_model_eval.json,
         results/fig1_data.json
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fairpass import MODEL_VERSION  # noqa: E402
from fairpass.linkmodel import SuccessModel, auc, brier, log_loss, mode_family  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
SEED = 20261010


def load(period):
    rows, seen = [], set()
    for p in sorted(glob.glob(str(ROOT / "data/raw/obs" / f"{period}_*.json"))):
        for r in json.load(open(p, encoding="utf-8")):
            if r["id"] not in seen:
                seen.add(r["id"])
                rows.append(r)
    return rows


def arrays(rows):
    lab = [r for r in rows if r["status"] in ("good", "bad") and r.get("max_altitude") is not None]
    return (np.array([r["max_altitude"] for r in lab], float), [r["transmitter_mode"] for r in lab],
            [r["ground_station"] for r in lab], [r["norad_cat_id"] for r in lab],
            np.array([r["status"] == "good" for r in lab], float), lab)


def metrics(y, p):
    return {"log_loss": log_loss(y, p), "brier": brier(y, p), "auc": auc(y, p) if 0 < y.sum() < len(y) else None}


def top_frac_success(y, score, st, frac=0.3):
    """Each station keeps its top `frac` passes by `score`; success rate of the kept passes."""
    st = np.asarray(st)
    kept = []
    for g in np.unique(st):
        idx = np.flatnonzero(st == g)
        k = max(1, int(round(frac * len(idx))))
        kept += list(idx[np.argsort(-score[idx], kind="stable")[:k]])
    return float(y[kept].mean()), len(kept)


def main():
    train, test = load("train"), load("test")
    el, md, st, sat, y, _ = arrays(train)
    el_t, md_t, st_t, sat_t, y_t, lab_t = arrays(test)
    print(f"train labelled {len(y)} (good {y.mean():.3f}); test labelled {len(y_t)} (good {y_t.mean():.3f})")

    # inner time split of train for the shrinkage strength
    order = np.argsort([r["start"] for r in arrays(train)[5]])
    cut = int(0.8 * len(order))
    a, b = order[:cut], order[cut:]
    best = None
    for l2s in (0.3, 1.0, 3.0):
        for l2g in (1.0, 3.0):
            m = SuccessModel("full", l2_station=l2g, l2_sat=l2s).fit(
                el[a], [md[i] for i in a], [st[i] for i in a], [sat[i] for i in a], y[a])
            ll = log_loss(y[b], m.predict(el[b], [md[i] for i in b], [st[i] for i in b], [sat[i] for i in b]))
            if best is None or ll < best[0]:
                best = (ll, l2s, l2g)
    _, l2s, l2g = best
    print("chosen l2_sat", l2s, "l2_station", l2g)

    models, ev = {}, {}
    for kind in ("const", "geom", "sat", "full"):
        m = SuccessModel(kind, l2_station=l2g, l2_sat=l2s).fit(el, md, st, sat, y)
        models[kind] = m
        p = m.predict(el_t, md_t, st_t, sat_t)
        ev[kind] = metrics(y_t, p)
        ev[kind]["top30_success"], ev[kind]["top30_n"] = top_frac_success(y_t, p + 1e-9 * el_t, st_t)
    seen = np.array([s in set(sat) for s in sat_t])
    pf = models["full"].predict(el_t, md_t, st_t, sat_t)
    ev["full_seen_sats"] = metrics(y_t[seen], pf[seen])
    ev["full_unseen_sats"] = metrics(y_t[~seen], pf[~seen]) if (~seen).sum() > 20 else None
    ev["frac_test_unseen_sat"] = float((~seen).mean())
    ev["top30_by_max_el_success"], _ = top_frac_success(y_t, el_t, st_t)
    ev["base_rate_test"] = float(y_t.mean())

    # calibration (deciles of predicted p)
    qs = np.quantile(pf, np.linspace(0, 1, 11))
    cal = []
    for lo, hi in zip(qs[:-1], qs[1:]):
        mm = (pf >= lo) & (pf <= hi)
        if mm.sum():
            cal.append({"p_mean": float(pf[mm].mean()), "y_mean": float(y_t[mm].mean()), "n": int(mm.sum())})
    ev["calibration_full"] = cal

    # bootstrap replicates of the full model (observation-level resampling: a satellite or station with few
    # observations gets a wide replicate spread, which is what the next-measurement analysis needs)
    rng = np.random.default_rng(SEED)
    boots = []
    for _ in range(40):
        idx = rng.integers(0, len(y), len(y))
        mb = SuccessModel("full", l2_station=l2g, l2_sat=l2s).fit(
            el[idx], [md[i] for i in idx], [st[i] for i in idx], [sat[i] for i in idx], y[idx])
        boots.append(mb.to_dict())

    # figure-1 data: empirical success vs max elevation on the TEST period, with model curves
    bins = [0, 15, 25, 35, 45, 60, 75, 91]
    fig = {"bins": bins, "empirical": [], "geom": [], "full": []}
    pg = models["geom"].predict(el_t, md_t, st_t, sat_t)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mm = (el_t >= lo) & (el_t < hi)
        k, n = int(y_t[mm].sum()), int(mm.sum())
        # Wilson 95% interval
        if n:
            ph = k / n
            z = 1.96
            cen = (ph + z * z / (2 * n)) / (1 + z * z / n)
            half = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / (1 + z * z / n)
        else:
            ph = cen = half = np.nan
        fig["empirical"].append({"lo": lo, "hi": hi, "k": k, "n": n, "p": ph, "ci": [cen - half, cen + half]})
        fig["geom"].append(float(pg[mm].mean()) if n else None)
        fig["full"].append(float(pf[mm].mean()) if n else None)
    # per-satellite: empirical test rate vs predicted mean (satellites with >= 8 test passes)
    fig["per_sat"] = []
    for s in sorted(set(sat_t)):
        mm = np.array(sat_t) == s
        if mm.sum() >= 8:
            fig["per_sat"].append({"norad": int(s), "n": int(mm.sum()), "y": float(y_t[mm].mean()),
                                   "p": float(pf[mm].mean()), "seen": bool(s in set(sat))})
    fig["mode_family_test_counts"] = {f: int(sum(mode_family(m) == f for m in md_t)) for f in set(map(mode_family, md_t))}

    full = models["full"].to_dict()
    full.update({"bootstrap": boots, "model_version": MODEL_VERSION, "seed": SEED,
                 "train_n": int(len(y)), "train_good_rate": float(y.mean())})
    (RES / "link_model.json").write_text(json.dumps(full))
    (RES / "link_model_geom.json").write_text(json.dumps(models["geom"].to_dict()))
    (RES / "link_model_eval.json").write_text(json.dumps(ev, indent=1))
    (RES / "fig1_data.json").write_text(json.dumps(fig, indent=1))
    print(json.dumps({k: v for k, v in ev.items() if k != "calibration_full"}, indent=1))


if __name__ == "__main__":
    main()
