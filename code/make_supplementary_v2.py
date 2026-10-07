#!/usr/bin/env python3
"""Supplementary figures S1-S4 and the numbers behind them, rebuilt entirely from
the current authoritative runs (replacing the stale make_supplementary.py, which
read the old n=50 runs).

Sources:
  - COCO n=500 primary run : results/signal_ra_coco_n500_msg256_d15_20260913_231014.json
  - 600-image comparison   : results/sota_comparison_20260910_174221.json
  - fidelity (600)         : results/fidelity_ssim_lpips_600.json
  - damage distribution    : results/damage_distribution_n1200.json, damage_ring_n1200.json
  - area strata (n=200)    : results/attackset_area_strata_20260929_124128.json
  - Delta sweep (n=200)    : newest results/delta_sweep_attackset200_*.json

Outputs: paper/figures/fig_s1_distribution.{pdf,png}, fig_s2_scatter, fig_s3_intensity,
         fig_s4_delta, and results/supplementary_numbers.json
Usage: cd code && python make_supplementary_v2.py
"""
import sys, os, json, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = f"{BASE}/paper/figures"
N500 = f"{BASE}/results/signal_ra_coco_n500_msg256_d15_20260913_231014.json"
SOTA = f"{BASE}/results/sota_comparison_20260910_174221.json"
FID = f"{BASE}/results/fidelity_ssim_lpips_600.json"
DAM = f"{BASE}/results/damage_distribution_n1200.json"
RING = f"{BASE}/results/damage_ring_n1200.json"
AREA = f"{BASE}/results/attackset_area_strata_20260929_124128.json"

plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.25,
                     "grid.linestyle": "--", "axes.spines.top": False,
                     "axes.spines.right": False})


def boot(v, n_boot=10000, seed=42):
    v = np.asarray(v, dtype=float)
    rng = np.random.RandomState(seed)
    m = np.array([rng.choice(v, len(v), replace=True).mean() for _ in range(n_boot)])
    return float(v.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def load(path):
    return json.load(open(path))


def series(pi, key):
    return np.array([r[key] for r in pi if r.get(key) is not None], dtype=float) * 100


def main():
    d = load(N500)
    pi = d["per_image"]
    S = {k: series(pi, k) for k in ["uniform_ba", "signal_ra_ba", "signal_soft_ba",
                                    "signal_soft_dilate_ba", "bbox_ra_ba"]}
    fail = series(pi, "att_fail_frac")
    out = {"n500": {}, "ci": {}, "damage": {}, "delta": {}, "fidelity": {}, "area": {}}

    for label, k in [("uniform", "uniform_ba"), ("v1", "signal_ra_ba"),
                     ("soft", "signal_soft_ba"), ("ours", "signal_soft_dilate_ba"),
                     ("oracle", "bbox_ra_ba")]:
        m, lo, hi = boot(S[k])
        out["ci"][label] = {"mean": round(m, 2), "lo": round(lo, 2), "hi": round(hi, 2)}
    out["n500"] = {"n": len(pi),
                   "mean": {k: round(float(np.mean(v)), 2) for k, v in
                            zip(["uniform", "v1", "soft", "ours", "oracle"],
                                [S["uniform_ba"], S["signal_ra_ba"], S["signal_soft_ba"],
                                 S["signal_soft_dilate_ba"], S["bbox_ra_ba"]])},
                   "frac_ge95": {k: round(float(np.mean(v >= 95) * 100), 1) for k, v in
                                 zip(["uniform", "v1", "soft", "ours", "oracle"],
                                     [S["uniform_ba"], S["signal_ra_ba"], S["signal_soft_ba"],
                                      S["signal_soft_dilate_ba"], S["bbox_ra_ba"]])}}

    # ---- Fig S1: per-image improvement + per-variant BA ----
    gain = S["signal_soft_dilate_ba"] - S["uniform_ba"]
    out["gain"] = {"mean": round(float(gain.mean()), 2), "median": round(float(np.median(gain)), 2),
                   "positive_pct": round(float((gain > 0).mean() * 100), 1),
                   "max": round(float(gain.max()), 1), "min": round(float(gain.min()), 1)}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    ax = axes[0]
    ax.hist(gain, bins=25, color="#2c6fbb", alpha=0.8, edgecolor="white")
    ax.axvline(0, color="#b03030", ls="--", lw=1.2)
    ax.axvline(gain.mean(), color="black", ls="-", lw=1.2,
               label=f"mean {gain.mean():+.1f} pt")
    ax.set_xlabel("BA improvement (ours $-$ undefended, pt)")
    ax.set_ylabel("Images")
    ax.set_title(f"(a) Per-image improvement (COCO, $n={len(gain)}$)", fontsize=10, loc="left")
    ax.annotate(f"positive: {(gain>0).mean()*100:.0f}%\nmedian {np.median(gain):+.1f} pt",
                xy=(0.97, 0.95), xycoords="axes fraction", ha="right", va="top", fontsize=8.5,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.85))
    ax = axes[1]
    labels = ["undef.", "binary", "soft", "ours", "oracle"]
    keys = ["uniform_ba", "signal_ra_ba", "signal_soft_ba",
            "signal_soft_dilate_ba", "bbox_ra_ba"]
    colors = ["#999999", "#6baed6", "#4292c6", "#2171b5", "#c77d1e"]
    for i, (k, c) in enumerate(zip(keys, colors)):
        ax.scatter(np.full(len(S[k]), i) + np.random.RandomState(42).uniform(-.18, .18, len(S[k])),
                   S[k], s=9, color=c, alpha=0.35, zorder=2)
        ax.errorbar(i, S[k].mean(), yerr=S[k].std(), fmt="D", color=c, ms=5,
                    capsize=3, zorder=3)
    ax.axhline(95, color="#555", ls=":", lw=1)
    ax.set_xticks(range(5)); ax.set_xticklabels(labels)
    ax.set_ylabel("Bit accuracy (%)")
    ax.set_title("(b) Per-image BA by decoder variant", fontsize=10, loc="left")
    fig.tight_layout(); fig.savefig(f"{FIG}/fig_s1_distribution.pdf", bbox_inches="tight")
    fig.savefig(f"{FIG}/fig_s1_distribution.png", dpi=200, bbox_inches="tight"); plt.close(fig)

    # ---- Fig S2: uniform vs ours scatter ----
    fig, ax = plt.subplots(figsize=(4.8, 4.6))
    u, v = S["uniform_ba"], S["signal_soft_dilate_ba"]
    ax.scatter(u, v, s=14, color="#2171b5", alpha=0.5, edgecolors="white", linewidth=0.4)
    lim = [min(u.min(), v.min()) - 3, 101]
    ax.plot(lim, lim, "k--", lw=0.9, alpha=0.6, label="$y=x$")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("Undefended BA (%)"); ax.set_ylabel("Ours BA (%)")
    ax.set_title(f"Per-image: ours vs undefended ($n={len(u)}$)", fontsize=10)
    above = int((v > u).sum())
    ax.text(0.04, 0.96, f"ours $>$ undefended: {above}/{len(u)} ({above/len(u)*100:.0f}%)\n"
            f"mean gain {np.mean(v-u):+.1f} pt",
            transform=ax.transAxes, va="top", fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.85))
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout(); fig.savefig(f"{FIG}/fig_s2_scatter.pdf", bbox_inches="tight")
    fig.savefig(f"{FIG}/fig_s2_scatter.png", dpi=200, bbox_inches="tight"); plt.close(fig)

    # ---- Fig S3: severity vs gain, stratified bars ----
    bins = [(0, 15, "Small"), (15, 35, "Mid"), (35, 101, "Large")]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    ax = axes[0]
    ax.scatter(fail, gain, s=12, color="#2171b5", alpha=0.5, edgecolors="white", linewidth=0.4)
    ax.axhline(0, color="#b03030", ls="--", lw=1)
    ax.set_xlabel("Attack severity: check-failing blocks (%)")
    ax.set_ylabel("BA improvement (pt)")
    ax.set_title("(a) Severity vs. defense gain", fontsize=10, loc="left")
    ax = axes[1]
    xs, um, vm, om, ns = [], [], [], [], []
    for lo, hi, lab in bins:
        m = (fail >= lo) & (fail < hi)
        if m.sum() == 0:
            continue
        xs.append(f"{lab}\n$n={int(m.sum())}$")
        um.append(S["uniform_ba"][m].mean()); vm.append(S["signal_soft_dilate_ba"][m].mean())
        om.append(S["bbox_ra_ba"][m].mean()); ns.append(int(m.sum()))
    x = np.arange(len(xs)); w = 0.26
    ax.bar(x - w, um, w, label="undefended", color="#bbbbbb", edgecolor="white")
    ax.bar(x, vm, w, label="ours", color="#2171b5", edgecolor="white")
    ax.bar(x + w, om, w, label="oracle", color="#c77d1e", edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels(xs)
    ax.set_ylim(50, 105); ax.set_ylabel("Mean BA (%)")
    ax.set_title("(b) Stratified mean BA", fontsize=10, loc="left")
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout(); fig.savefig(f"{FIG}/fig_s3_intensity.pdf", bbox_inches="tight")
    fig.savefig(f"{FIG}/fig_s3_intensity.png", dpi=200, bbox_inches="tight"); plt.close(fig)
    out["severity_bins"] = {lab: {"n": n, "undefended": round(float(a), 1),
                                  "ours": round(float(b), 1), "oracle": round(float(c), 1)}
                            for (lo, hi, lab), n, a, b, c in zip(bins, ns, um, vm, om)}

    # ---- damage distribution ----
    dmg = load(DAM); ring = load(RING) if os.path.exists(RING) else {}
    out["damage"] = {**{k: round(v, 4) for k, v in dmg.items() if isinstance(v, float)},
                     "ring": ring, "n": dmg.get("n")}

    # ---- area strata (n=200) ----
    ar = load(AREA)
    out["area"] = {"strata": ar["strata"], "corr_area_vs_checkfail": ar["corr"]["area_vs_checkfail"],
                   "summary": ar["summary"]}

    # ---- fidelity paired ----
    fid = load(FID)
    out["fidelity"] = {"per_method": fid["per_method"], "paired": fid["paired_vs_ours"]}

    # ---- Delta sweep ----
    cand = sorted(glob.glob(f"{BASE}/results/delta_sweep_attackset200_*.json"))
    if cand:
        ds = load(cand[-1]); out["delta"] = {"source": os.path.basename(cand[-1]),
                                             "summary": ds["summary"]}
        dl = sorted(int(k) for k in ds["summary"])
        vm2 = [100 * ds["summary"][str(k)]["v2_ba"] for k in dl]
        um2 = [100 * ds["summary"][str(k)]["uniform_ba"] for k in dl]
        om2 = [100 * ds["summary"][str(k)]["oracle_ba"] for k in dl]
        psm = [ds["summary"][str(k)]["psnr"] for k in dl]
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
        axes[0].plot(dl, um2, "o-", color="#7f7f7f", label="undefended")
        axes[0].plot(dl, vm2, "s-", color="#b03030", label="ours")
        axes[0].plot(dl, om2, "^--", color="#c77d1e", label="oracle")
        axes[0].set_xticks(dl); axes[0].set_xlabel(r"Embedding step size $\Delta$")
        axes[0].set_ylabel("Bit accuracy (%)")
        axes[0].set_title(r"(a) Defense vs. $\Delta$", fontsize=10, loc="left")
        axes[0].legend(fontsize=8.5, frameon=False)
        axes[1].plot(dl, psm, "D-", color="#2c6fbb")
        for x0, y0 in zip(dl, psm):
            axes[1].annotate(f"{y0:.1f}", (x0, y0), textcoords="offset points",
                             xytext=(0, 7), ha="center", fontsize=8.5)
        axes[1].set_xticks(dl); axes[1].set_xlabel(r"Embedding step size $\Delta$")
        axes[1].set_ylabel("PSNR (dB)")
        axes[1].set_title(r"(b) Embedding fidelity vs. $\Delta$", fontsize=10, loc="left")
        fig.tight_layout(); fig.savefig(f"{FIG}/fig_s4_delta.pdf", bbox_inches="tight")
        fig.savefig(f"{FIG}/fig_s4_delta.png", dpi=200, bbox_inches="tight"); plt.close(fig)

    json.dump(out, open(f"{BASE}/results/supplementary_numbers.json", "w"), indent=1)
    print(json.dumps({k: out[k] for k in ["n500", "ci", "gain", "damage", "delta"]}, indent=1)[:4000])


if __name__ == "__main__":
    main()
