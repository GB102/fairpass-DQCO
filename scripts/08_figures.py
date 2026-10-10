"""Step 8: the three report figures (PNG + SVG) from results/*.json.

Fig 1  model vs real measurements: (a) pass-geometry reproduction error, (b) held-out success vs max
       elevation with Wilson 95% CIs and model curves, (c) held-out calibration by decile
Fig 2  operational performance of schedulers on the real network-day: expected delivered data (bootstrap
       90% interval) and starved active missions - two panels, one measure each (no dual axis)
Fig 3  decision / benefit boundaries: (a) fairness-throughput frontier vs lam, (b,c) QAOA amplification of
       the optimal schedule's probability vs two-qubit error, with IonQ reference error rates
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES, FIG = ROOT / "results", ROOT / "figures"
FIG.mkdir(exist_ok=True)

# validated reference palette (light mode) - categorical slots in fixed order
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#fcfcfb"
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10, "axes.titlesize": 11,
    "axes.titleweight": "bold", "axes.titlelocation": "left", "lines.linewidth": 2,
    "lines.solid_capstyle": "round", "legend.frameon": False, "axes.axisbelow": True,
})


def save(fig, name):
    fig.savefig(FIG / f"{name}.png", dpi=200, bbox_inches="tight")
    fig.savefig(FIG / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)


def fig1():
    f1 = json.load(open(RES / "fig1_data.json"))
    ev = json.load(open(RES / "link_model_eval.json"))
    import csv
    d = np.array([float(r["d_max_el"]) for r in csv.DictReader(open(RES / "pass_validation.csv"))])
    pv = json.load(open(RES / "pass_validation.json"))
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8), gridspec_kw={"width_ratios": [1, 1.25, 1]})

    a = ax[0]
    a.hist(d, bins=np.arange(-1.5, 1.51, 0.1), color=S1, rwidth=0.85)
    a.set_xlabel("computed − recorded max elevation (deg)")
    a.set_ylabel("observations")
    a.set_title("a  Pass geometry reproduced from stored TLEs")
    a.text(0.02, 0.97, f"n = {pv['n_matched']}\nmedian |err| {pv['max_el_error_deg']['median_abs']:.2f}°\n"
           f"{100 * pv['max_el_error_deg']['frac_within_1deg']:.1f}% within 1°\n(records are whole degrees)",
           transform=a.transAxes, va="top", fontsize=9, color=INK2)

    a = ax[1]
    mids = [(e["lo"] + e["hi"]) / 2 for e in f1["empirical"]]
    ok = [e["n"] > 0 for e in f1["empirical"]]
    m = np.array(mids)[ok]
    emp = np.array([e["p"] for e in f1["empirical"]])[ok]
    lo = np.array([e["ci"][0] for e in f1["empirical"]])[ok]
    hi = np.array([e["ci"][1] for e in f1["empirical"]])[ok]
    a.errorbar(m, emp, yerr=[emp - lo, hi - emp], fmt="o", ms=7, color=INK, ecolor=MUTED, elinewidth=1.2,
               capsize=0, label="observed (test period, 95% CI)", zorder=3,
               markeredgecolor=SURF, markeredgewidth=2)
    a.plot(m, np.array(f1["geom"])[ok], color=S2, label="geometry-only model")
    a.plot(m, np.array(f1["full"])[ok], color=S1, label="full model (satellite, station, mode, geometry)")
    a.set_xlabel("pass max elevation (deg)")
    a.set_ylabel("P(observation vetted 'good')")
    a.set_ylim(0, 1)
    a.set_title("b  Held-out pass success vs elevation")
    a.legend(loc="upper left", fontsize=8.5)

    a = ax[2]
    cal = ev["calibration_full"]
    a.plot([0, 1], [0, 1], color=GRID, lw=1.5, zorder=1)
    a.plot([c["p_mean"] for c in cal], [c["y_mean"] for c in cal], "-o", color=S1, ms=7,
           markeredgecolor=SURF, markeredgewidth=2, zorder=3)
    a.set_xlabel("predicted success (decile mean)")
    a.set_ylabel("observed success rate")
    a.set_xlim(0, 1)
    a.set_ylim(0, 1)
    a.set_title("c  Held-out calibration")
    a.text(0.03, 0.97, f"AUC full {ev['full']['auc']:.2f} vs geometry {ev['geom']['auc']:.2f}\n"
           f"log-loss {ev['full']['log_loss']:.3f} vs {ev['geom']['log_loss']:.3f} (const "
           f"{ev['const']['log_loss']:.3f})", transform=a.transAxes, va="top", fontsize=9, color=INK2)
    fig.suptitle("Fig. 1  Model layer tested on real SatNOGS observations (train 09-13…09-24, test 09-26…09-30)",
                 x=0.01, ha="left", fontsize=11.5, fontweight="bold")
    fig.tight_layout()
    save(fig, "fig1_model_vs_measurements")


NAMES = {"actual": "Actual network schedule (2026-10-01)", "station_maxel": "Per-station greedy, by elevation",
         "station_value": "Per-station greedy, by expected data", "network_greedy": "Coordinated greedy + local search",
         "cpsat": "Coordinated optimum (throughput)"}


def fig2(lam_star):
    ss = json.load(open(RES / "schedule_study.json"))
    n_st = len(json.load(open(ROOT / "data/processed/instance_2026-10-01.json"))["stations"])
    keys = ["actual", "station_maxel", "station_value", "network_greedy", "cpsat", f"cpsat_fair_{lam_star}"]
    keys = [k for k in keys if k in ss["schedules"]]
    labels = [NAMES.get(k, f"Coordinated optimum + fairness (λ={lam_star})") for k in keys]
    sch = [ss["schedules"][k] for k in keys]
    y = np.arange(len(keys))[::-1]
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 3.9), sharey=True)
    a = ax[0]
    mb = np.array([s["boot"]["expected_unique_kB"]["mean"] / 1000 for s in sch])
    lo = np.array([s["boot"]["expected_unique_kB"]["p05"] / 1000 for s in sch])
    hi = np.array([s["boot"]["expected_unique_kB"]["p95"] / 1000 for s in sch])
    a.barh(y, mb, height=0.55, color=S1)
    a.errorbar(mb, y, xerr=[np.maximum(mb - lo, 0), np.maximum(hi - mb, 0)], fmt="none", ecolor=INK2, elinewidth=1.2)
    for yy, v, h in zip(y, mb, hi):
        a.text(h + mb.max() * 0.015, yy, f"{v:,.0f}", va="center", fontsize=9, color=INK)
    a.set_yticks(y, labels)
    a.set_xlabel("expected unique data delivered (MB / day, bootstrap 90% interval)")
    a.set_title("a  Throughput")
    a.set_xlim(0, hi.max() * 1.15)
    a.grid(axis="y", visible=False)
    a = ax[1]
    st = np.array([s["starved"] for s in sch])
    a.barh(y, st, height=0.55, color=S1)
    for yy, v in zip(y, st):
        a.text(v + st.max() * 0.015, yy, f"{v}", va="center", fontsize=9, color=INK)
    a.set_xlabel(f"active missions with zero contacts (of {ss['n_active']})")
    a.set_title("b  Missions left behind")
    a.set_xlim(0, st.max() * 1.15 + 1)
    a.grid(axis="y", visible=False)
    fig.suptitle(f"Fig. 2  Schedulers on the real 2026-10-01 network-day: {ss['n_candidates']:,} candidate passes, "
                 f"{n_st} stations, {ss['n_satellites']} satellites; all scored by the validated success model",
                 x=0.01, ha="left", fontsize=11.5, fontweight="bold")
    fig.tight_layout()
    save(fig, "fig2_operational_performance")


def fig3(lam_star):
    ss = json.load(open(RES / "schedule_study.json"))
    q = json.load(open(RES / "qaoa_study.json"))
    sizes = [s for s in ("8", "10") if s in q["sizes"] and "noise_sweep" in q["sizes"][s]]
    fig, ax = plt.subplots(1, 1 + len(sizes), figsize=(4.6 * (1 + len(sizes)), 4.0))
    ax = np.atleast_1d(ax)
    a = ax[0]
    lams = sorted([0] + [int(k.split("_")[-1]) for k in ss["schedules"] if k.startswith("cpsat_fair_")])
    pts = [ss["schedules"]["cpsat" if l == 0 else f"cpsat_fair_{l}"] for l in lams]
    xs_ = [p["starved"] for p in pts]
    ys_ = [p["boot"]["expected_unique_kB"]["mean"] / 1000 for p in pts]
    a.plot(xs_, ys_, "-o", color=S1, ms=7, markeredgecolor=SURF, markeredgewidth=2,
           label="coordinated optimum, fairness weight λ")
    for l, x_, y_ in zip(lams, xs_, ys_):
        if l in (0, lam_star, lams[-1]):
            a.annotate(f"λ={l}", (x_, y_), textcoords="offset points", xytext=(6, 6), fontsize=8.5, color=INK2)
    for k, mk in (("station_value", "s"), ("actual", "D")):
        if k in ss["schedules"]:
            s = ss["schedules"][k]
            a.plot(s["starved"], s["boot"]["expected_unique_kB"]["mean"] / 1000, mk, color=S2, ms=8,
                   markeredgecolor=SURF, markeredgewidth=2, label=NAMES[k].split(",")[0] if k == "actual"
                   else "per-station greedy (uncoordinated)")
    a.set_xlabel("active missions with zero contacts")
    a.set_ylabel("expected unique data (MB / day)")
    a.set_title("a  Fairness–throughput frontier")
    a.legend(fontsize=8.5, loc="lower left")
    ref = {"Forte ≈0.4%": 0.4, "Aria ≈0.6%": 0.6}
    for a, n in zip(ax[1:], sizes):
        ns = q["sizes"][n]["noise_sweep"]
        for (key, rows), col in zip(sorted(ns["rows"].items()), (S1, S2, S3)):
            p = key.split("_p")[1].split("_")[0]
            e2 = [100 * r["e2"] for r in rows]
            amp = [r["p_opt"] / ns["uniform_p_opt"] for r in rows]
            a.plot(e2, amp, "-o", color=col, ms=6, markeredgecolor=SURF, markeredgewidth=1.5, label=f"QAOA p={p}")
        a.axhline(1.0, color=INK2, lw=1)
        a.text(0.98, 1.08, "uniform random sampling = 1", fontsize=8, color=INK2, ha="right",
               transform=a.get_yaxis_transform())
        for lab, v in ref.items():
            a.axvline(v, color=GRID, lw=1.5, zorder=0)
            a.text(v, 0.03, lab, rotation=90, va="bottom", ha="right", fontsize=8, color=MUTED,
                   transform=a.get_xaxis_transform())
        a.set_yscale("log")
        a.set_xlabel("two-qubit gate error (%)  [1q = e/10, readout 0.5%]")
        a.set_ylabel("P(optimal schedule) ÷ uniform")
        a.set_title(f"{'bc'[sizes.index(n)]}  QAOA amplification, {n}-pass windows")
        a.legend(fontsize=8.5, loc="upper right")
    fig.suptitle("Fig. 3  Decision boundaries: how much fairness to buy, and where the quantum step stops helping "
                 "(held-out windows, transferred angles)", x=0.01, ha="left", fontsize=11.5, fontweight="bold")
    fig.tight_layout()
    save(fig, "fig3_decision_boundaries")


def fig4():
    h = json.load(open(RES / "hybrid_lns.json"))
    order = [("exact", None, "exact enumeration"), ("greedy", None, "coordinated greedy"),
             ("qaoa", 100, "QAOA (ideal)"), ("qaoa_noisy_forte", 100, "QAOA (Forte-like noise)"),
             ("random", 100, "uniform random"), ("sa", 100, "simulated annealing"),
             ("qaoa", 1000, "QAOA (ideal)"), ("random", 1000, "uniform random"), ("sa", 1000, "simulated annealing")]
    labels, means, lo, hi = [], [], [], []
    for name, N, lab in order:
        v = [r["gap_closed"] for r in h["runs"] if r["solver"] == name and r["shots"] == N]
        if not v:
            continue
        labels.append(lab + (f", N={N}" if N else ""))
        means.append(100 * np.mean(v))
        lo.append(100 * np.min(v))
        hi.append(100 * np.max(v))
    y = np.arange(len(labels))[::-1]
    fig, a = plt.subplots(figsize=(8.5, 0.42 * len(labels) + 1.2))
    a.barh(y, means, height=0.55, color=S1)
    m, l_, h_ = map(np.array, (means, lo, hi))
    a.errorbar(m, y, xerr=[np.maximum(m - l_, 0), np.maximum(h_ - m, 0)], fmt="none", ecolor=INK2, elinewidth=1.2)
    for yy, v, top in zip(y, means, hi):
        a.text(max(v, top) + 1.5, yy, f"{v:.0f}%", va="center", fontsize=9)
    a.set_yticks(y, labels)
    a.set_xlim(0, 110)
    a.set_xlabel("share of the gap (today's greedy → global CP-SAT optimum) closed by the hybrid LNS "
                 "(range over 3 seeds)")
    a.grid(axis="y", visible=False)
    a.set_title(f"Hybrid LNS, {h['n_window']}-pass windows, same windows and acceptance rule for every solver")
    fig.tight_layout()
    save(fig, "fig4_hybrid_lns")


if __name__ == "__main__":
    lam_star = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    for f in (fig1, lambda: fig2(lam_star), lambda: fig3(lam_star), fig4):
        try:
            f()
        except FileNotFoundError as e:
            print("skip:", e)
    print("figures written to", FIG)
