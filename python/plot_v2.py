#!/usr/bin/env python3
"""Phase-4 figures — PROTOCOL.md sections 5, 10, 8.1/8.3, 11.

matplotlib (Agg backend, headless) -> PDF + PNG under code/results/. Data comes
from the aggregated timing CSVs (agg_timing.csv, scaling_fits.csv) and, for the
convergence-spread figure, from the raw convergence-pass CSVs (bench_*_conv.csv,
which retain one iteration count per seeded RHS); nothing is transcribed by hand
(section 11). Four figures:

  1. fig_v2_work_identity   cross-language work-count identity (section 5): for
     every shared experimental unit, Julia vs Python work counts must lie on the
     y = x line (equivalent implementations produce identical counts).
  2. fig_v2_time_vs_n       solve time vs n on log-log axes with the fitted
     scaling line log T = beta0 + beta1 log n per (method, language) for the SPD
     ladder (section 10), single_thread regime.
  3. fig_v2_convergence_spread  iteration-count spread across the seeded RHS
     (section 3.3): per instance a box (IQR) with min-max whiskers.
  4. fig_v2_drazin_error    Drazin error of DGMRES vs ordinary GMRES per singular
     instance (section 8.1/8.3): DGMRES recovers A^D b (error ~ machine epsilon)
     via the exported x_star; GMRES does not (O(1) error).

No pandas; stdlib csv + numpy + matplotlib only.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
from collections import defaultdict

import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# size-encoding tokens (co-vary with n); everything else is a held-fixed
# secondary parameter -- matches aggregate.family_group so figure series line up
# with the per-family scaling fits.
_SIZE_TOKEN = re.compile(r"^(n\d+|m\d+|ncore\d+)$")


def family_group(family, instance_key):
    base = instance_key.split("/")[-1]
    kept = [t for t in base.split("_") if not _SIZE_TOKEN.match(t)]
    label = "_".join(kept)
    return f"{family}/{label}" if label else family

HERE = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.normpath(os.path.join(HERE, ".."))
RESULTS_DIR = os.path.join(CODE_DIR, "results")
BENCH_DIR = os.path.join(RESULTS_DIR, "benchmarks")

# colorblind-friendly (Wong) palette
CB = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73",
      "vermillion": "#D55E00", "purple": "#CC79A7", "black": "#000000"}
METHOD_COLOR = {"cg": CB["blue"], "pcg_jacobi": CB["orange"],
                "pcg_ic0": CB["green"], "gmres": CB["vermillion"],
                "dgmres": CB["blue"]}
LANG_MARKER = {"julia": "o", "python": "s"}


def load_csv(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def g(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def savefig(fig, stem):
    for ext in ("pdf", "png"):
        p = os.path.join(RESULTS_DIR, f"{stem}.{ext}")
        fig.savefig(p, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return os.path.join(RESULTS_DIR, f"{stem}.pdf")


def fig_work_identity(units):
    """Julia vs Python work counts for every shared unit; must be on y = x."""
    key = lambda u: (u["phase"], u["method"], u["instance_key"], u["n"], u["seed"])
    by = defaultdict(dict)
    for u in units:
        by[key(u)][u["language"]] = u
    metrics = [("matvecs", CB["blue"], "o"),
               ("orthogonalizations", CB["orange"], "s"),
               ("precond_applications", CB["green"], "^")]
    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    max_rel = 0.0
    plotted = set()
    for name, color, marker in metrics:
        xs, ys = [], []
        for pair in by.values():
            if "julia" in pair and "python" in pair:
                xv = g(pair["julia"][name])
                yv = g(pair["python"][name])
                if xv > 0 or yv > 0:
                    xs.append(xv)
                    ys.append(yv)
                    if max(xv, yv) > 0:
                        max_rel = max(max_rel, abs(xv - yv) / max(xv, yv))
        if xs:
            ax.scatter(xs, ys, s=42, facecolors="none", edgecolors=color,
                       marker=marker, linewidths=1.4, label=name)
            plotted.update(xs + ys)
    lim = max(plotted) if plotted else 1
    lim *= 1.3
    ax.plot([1, lim], [1, lim], "--", color=CB["black"], lw=1, label="y = x")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.8, lim)
    ax.set_ylim(0.8, lim)
    ax.set_xlabel("Julia work count")
    ax.set_ylabel("Python work count")
    ax.set_title(f"Cross-language work-count identity\n(max relative deviation "
                 f"= {max_rel:.1e})")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.set_aspect("equal", adjustable="box")
    return savefig(fig, "fig_v2_work_identity"), max_rel


def fig_time_vs_n(units, fits):
    """SPD solve time vs n (log-log), ONE PANEL PER VALID FAMILY. Each family that
    retains a genuine n-ladder (>= 3 distinct n, secondary params held fixed) gets
    its own panel; within it, every (method, language) series carries its own
    fitted scaling line (PROTOCOL.md 10). No line is ever drawn across families."""
    spd = [u for u in units if u["phase"] == "spd"]
    # per-family fits with a valid exponent
    fit_by = {(f["family"], f["language"], f["method"]): f
              for f in fits if f["phase"] == "spd" and f["status"] == "ok"}
    valid_fams = sorted({fam for (fam, _l, _m) in fit_by})
    if not valid_fams:
        valid_fams = sorted({family_group(u["family"], u["instance_key"]) for u in spd})

    # median over RHS of the per-unit median t_solve, per (family, lang, method, n)
    agg = defaultdict(list)
    for u in spd:
        fam = family_group(u["family"], u["instance_key"])
        agg[(fam, u["language"], u["method"], int(u["n"]))].append(g(u["t_solve_median_ns"]))

    ncol = 2
    nrow = int(math.ceil(len(valid_fams) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(6.0 * ncol, 4.3 * nrow),
                             squeeze=False)
    axf = axes.flatten()
    for ax, fam in zip(axf, valid_fams):
        methods = sorted({m for (f, l, m, n) in agg if f == fam})
        for method in methods:
            color = METHOD_COLOR.get(method, CB["black"])
            for lang in ("julia", "python"):
                ns = sorted({n for (f, l, m, n) in agg
                             if f == fam and l == lang and m == method})
                if not ns:
                    continue
                ys = [np.median(agg[(fam, lang, method, n)]) / 1e6 for n in ns]
                ls = "-" if lang == "julia" else "--"
                ax.scatter(ns, ys, color=color, s=38, zorder=3,
                           marker=LANG_MARKER[lang], facecolors="none"
                           if lang == "python" else color, linewidths=1.3)
                fit = fit_by.get((fam, lang, method))
                if fit and math.isfinite(g(fit["beta1"])):
                    b0, b1 = g(fit["beta0"]), g(fit["beta1"])
                    xline = np.array([min(ns), max(ns)], dtype=float)
                    yline = np.exp(b0) * xline ** b1 / 1e6
                    ax.plot(xline, yline, ls=ls, color=color, lw=1.5,
                            label=fr"{method} {lang[:2]}: $\beta_1$={b1:.2f}")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("n (unknowns)")
        ax.set_ylabel(r"median $t_\mathrm{solve}$ [ms]")
        ax.set_title(fam, fontsize=9)
        ax.grid(True, which="both", ls=":", lw=0.5, alpha=0.6)
        ax.legend(frameon=False, fontsize=6.5, loc="upper left", ncol=1)
    for ax in axf[len(valid_fams):]:
        ax.set_visible(False)
    fig.suptitle(r"Per-family SPD scaling: $\log T_\mathrm{solve} = \beta_0 + "
                 r"\beta_1 \log n$ (solid = Julia, dashed = Python)", y=1.005)
    fig.tight_layout()
    return savefig(fig, "fig_v2_time_vs_n")


def fig_convergence_spread(conv_units):
    """Iteration-count spread across the seeded RHS (PROTOCOL.md 3.2/3.3), from the
    convergence pass (1 rep, up to 20 RHS, single thread). For each (phase, method)
    the per-instance distribution of iteration counts over RHS seeds is drawn as a
    box (IQR) with whiskers (min-max); a wide box means RHS-sensitive convergence."""
    # iterations are language- and rep-invariant here; use julia rows, one per seed.
    rows = [u for u in conv_units if u["language"] == "julia"]
    # group by (phase, method, instance) -> list of per-seed iteration counts
    by = defaultdict(lambda: defaultdict(list))
    for u in rows:
        by[(u["phase"], u["method"])][(u["instance_key"], int(u["n"]))].append(
            g(u["iterations"]))

    panels = [("spd", ["cg", "pcg_jacobi", "pcg_ic0"]),
              ("drazin", ["gmres", "dgmres"])]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0))
    for ax, (phase, methods) in zip(axes, panels):
        # instances present in this phase, ordered by n
        insts = sorted({inst for m in methods for inst in
                        {k for (p, mm) in by for k in by[(p, mm)]
                         if p == phase and mm == m}}, key=lambda t: (t[1], t[0]))
        labels = [f"{k[0].split('/')[-1]}" for k in insts]
        xbase = np.arange(len(insts))
        width = 0.8 / max(len(methods), 1)
        for mi, method in enumerate(methods):
            data, pos = [], []
            for xi, inst in enumerate(insts):
                vals = [v for v in by[(phase, method)].get(inst, []) if math.isfinite(v)]
                if vals:
                    data.append(vals)
                    pos.append(xi + (mi - (len(methods) - 1) / 2) * width)
            if not data:
                continue
            color = METHOD_COLOR.get(method, CB["black"])
            bp = ax.boxplot(data, positions=pos, widths=width * 0.9, patch_artist=True,
                            manage_ticks=False, showfliers=True)
            for box in bp["boxes"]:
                box.set(facecolor=color, alpha=0.45, edgecolor=color)
            for med in bp["medians"]:
                med.set(color=CB["black"], lw=1.2)
            ax.plot([], [], color=color, lw=6, alpha=0.45, label=method)
        ax.set_xticks(xbase)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=7)
        ax.set_ylabel("iterations to convergence")
        ax.set_title(f"{phase}: iteration-count spread across RHS")
        ax.legend(frameon=False, fontsize=8)
        ax.grid(True, axis="y", ls=":", lw=0.5, alpha=0.6)
    fig.suptitle("Convergence sensitivity to the right-hand side (per seeded RHS)",
                 y=1.02)
    return savefig(fig, "fig_v2_convergence_spread")


def fig_drazin_error(units):
    """DGMRES vs GMRES Drazin error per singular instance (log scale)."""
    dr = [u for u in units if u["phase"] == "drazin"]
    # median over RHS+language of drazin_error, per (instance, method)
    agg = defaultdict(list)
    insts = []
    for u in dr:
        short = u["instance_key"].split("/")[-1]
        if short not in insts:
            insts.append(short)
        agg[(short, u["method"])].append(g(u["drazin_error"]))
    insts = sorted(insts)
    methods = ["gmres", "dgmres"]
    floor = 1e-16
    x = np.arange(len(insts))
    w = 0.38
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    for i, method in enumerate(methods):
        vals = [max(np.median(agg.get((inst, method), [floor])), floor) for inst in insts]
        color = CB["vermillion"] if method == "gmres" else CB["green"]
        ax.bar(x + (i - 0.5) * w, vals, width=w, color=color, label=method,
               edgecolor="black", linewidth=0.5)
    ax.axhline(1e-12, ls="--", color=CB["black"], lw=1,
               label=r"$10^{-12}$ (recovery threshold)")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(insts, rotation=15, ha="right", fontsize=8)
    ax.set_ylabel(r"Drazin error $\|x - A^D b\| / \|A^D b\|$")
    ax.set_title("DGMRES recovers the Drazin solution; ordinary GMRES does not")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(True, axis="y", which="both", ls=":", lw=0.5, alpha=0.6)
    return savefig(fig, "fig_v2_drazin_error")


def load_raw_conv(bench_dir, spd_name, drazin_name):
    """Raw convergence rows (one iteration count per seeded RHS), tagged with a
    'phase' column so the spread figure can panel SPD vs singular."""
    rows = []
    for name, phase in ((spd_name, "spd"), (drazin_name, "drazin")):
        path = os.path.join(bench_dir, name)
        if os.path.isfile(path):
            for r in load_csv(path):
                r["phase"] = phase
                rows.append(r)
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bench-dir", default=BENCH_DIR)
    p.add_argument("--units", default="agg_timing.csv")
    p.add_argument("--fits", default="scaling_fits.csv")
    p.add_argument("--spd-conv", default="bench_spd_conv.csv",
                   help="raw SPD convergence CSV (per-seed iteration counts)")
    p.add_argument("--drazin-conv", default="bench_drazin_conv.csv")
    p.add_argument("--regime", default="single_thread",
                   help="regime used for the timing/scaling/work/Drazin figures")
    args = p.parse_args()
    units_all = load_csv(os.path.join(args.bench_dir, args.units))
    fits_all = load_csv(os.path.join(args.bench_dir, args.fits))
    # the work-identity, scaling and Drazin figures use one regime (work counts and
    # Drazin error are regime-invariant; scaling is fitted per regime).
    units = [u for u in units_all if u.get("regime") == args.regime] or units_all
    fits = [f for f in fits_all if f.get("regime", args.regime) == args.regime] or fits_all

    conv_rows = load_raw_conv(args.bench_dir, args.spd_conv, args.drazin_conv)

    p1, max_rel = fig_work_identity(units)
    p2 = fig_time_vs_n(units, fits)
    p3 = fig_drazin_error(units)
    print(f"wrote {p1} (+.png)  [work-count max rel dev = {max_rel:.2e}]")
    print(f"wrote {p2} (+.png)")
    print(f"wrote {p3} (+.png)")
    if conv_rows:
        p4 = fig_convergence_spread(conv_rows)
        print(f"wrote {p4} (+.png)")
    else:
        print("  (no raw convergence CSVs; skipped convergence-spread figure)")


if __name__ == "__main__":
    main()
