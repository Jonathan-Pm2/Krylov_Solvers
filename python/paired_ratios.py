#!/usr/bin/env python3
"""Round-2 analysis 1 -- PAIRED cross-language comparison.

Replaces the round-1 visual claim ("Python faster across the mid-to-large
range") with a proper PAIRED analysis. Julia and Python solve the SAME
experimental cell -- identical matrix, n, RHS seed, method, regime -- so the two
timings are paired, not two independent samples. For every shared cell we form

    R = log( T_Python / T_Julia )      on the per-unit median t_solve

so R < 0 means Python is faster on that cell. Per (family, regime, method) and
overall we report:

  * the geometric-mean ratio  exp(median R)  and  exp(mean R)
    (a ratio > 1 means Python is slower by that factor; < 1 means faster),
  * a PAIRED bootstrap 95% CI (resampling the shared cells, i.e. the paired
    differences -- the correct paired resampling unit),
  * the PROPORTION of cells where Python is faster (R < 0),
  * the ratio as a function of n, to locate any crossover.

No new benchmark runs: reads only the raw per-rep timing CSVs.

Outputs:
  * results/benchmarks/paired_ratios.csv   one row per (regime, family, method)
        group and per overall/by-n rollups
  * results/paired_ratios_v2.tex           booktabs summary table
  * results/fig_v2_paired_ratio.{pdf,png}  ratio vs n (crossover) figure
"""
from __future__ import annotations

import argparse
import csv
import math
import os
from collections import defaultdict

import numpy as np

import statlib_v2 as sl


def geomean_stats(R, B=5000, seed=20260728):
    """From a vector of paired log-ratios R = log(T_Py/T_Jl): geometric-mean
    ratio via exp(mean R) and exp(median R), paired bootstrap 95% CI of
    exp(mean R), and the proportion of cells with R < 0 (Python faster)."""
    R = np.asarray(R, dtype=float)
    R = R[np.isfinite(R)]
    if R.size == 0:
        return dict(n_cells=0, gm_mean=float("nan"), gm_median=float("nan"),
                    ci_lo=float("nan"), ci_hi=float("nan"),
                    prop_py_faster=float("nan"))
    _, lo, hi = sl.bootstrap_ci(R, stat=lambda a, axis=None: np.mean(a, axis=axis),
                                B=B, seed=seed)
    return dict(
        n_cells=int(R.size),
        gm_mean=float(np.exp(np.mean(R))),
        gm_median=float(np.exp(np.median(R))),
        ci_lo=float(np.exp(lo)),
        ci_hi=float(np.exp(hi)),
        prop_py_faster=float(np.mean(R < 0.0)),
    )


def build_rows(paired):
    """Return (group_rows, byn_rows, all_R_by_regime). group_rows: overall +
    per (family, method) within each regime. byn_rows: ratio vs n per regime."""
    # collect per-cell log ratios, tagged by regime/family_group/method/n
    recs = []
    for ck, langs in paired.items():
        jl = langs["julia"]["median"]
        py = langs["python"]["median"]
        if not (math.isfinite(jl) and math.isfinite(py) and jl > 0 and py > 0):
            continue
        d = langs["julia"]["key"]
        recs.append(dict(regime=d["regime"], phase=d["phase"],
                         family=langs["julia"]["family_group"],
                         method=d["method"], n=int(d["n"]),
                         R=math.log(py / jl)))

    group_rows = []
    # overall per regime, and (phase-level) overall too
    by_regime = defaultdict(list)
    for r in recs:
        by_regime[r["regime"]].append(r["R"])
    for regime in sorted(by_regime):
        st = geomean_stats(by_regime[regime])
        group_rows.append(dict(regime=regime, scope="overall", phase="all",
                               family="ALL", method="ALL", **st))

    # per (regime, phase) overall
    by_rp = defaultdict(list)
    for r in recs:
        by_rp[(r["regime"], r["phase"])].append(r["R"])
    for (regime, phase) in sorted(by_rp):
        st = geomean_stats(by_rp[(regime, phase)])
        group_rows.append(dict(regime=regime, scope="phase", phase=phase,
                               family="ALL", method="ALL", **st))

    # per (regime, family, method)
    by_fam = defaultdict(list)
    fam_phase = {}
    for r in recs:
        by_fam[(r["regime"], r["family"], r["method"])].append(r["R"])
        fam_phase[(r["family"], r["method"])] = r["phase"]
    for (regime, family, method) in sorted(by_fam):
        st = geomean_stats(by_fam[(regime, family, method)])
        group_rows.append(dict(regime=regime, scope="family_method",
                               phase=fam_phase[(family, method)],
                               family=family, method=method, **st))

    # ratio as a function of n (per regime, pooling families/methods at each n)
    byn = defaultdict(list)
    for r in recs:
        byn[(r["regime"], r["n"])].append(r["R"])
    byn_rows = []
    for (regime, n) in sorted(byn):
        st = geomean_stats(byn[(regime, n)])
        byn_rows.append(dict(regime=regime, n=n, **st))

    return group_rows, byn_rows


def write_csv(path, group_rows, byn_rows):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["kind", "regime", "scope", "phase", "family", "method", "n",
                    "n_cells", "gm_ratio_mean", "gm_ratio_median",
                    "ci_lo", "ci_hi", "prop_python_faster"])
        for r in group_rows:
            w.writerow(["group", r["regime"], r["scope"], r["phase"],
                        r["family"], r["method"], "",
                        r["n_cells"], f"{r['gm_mean']:.6f}",
                        f"{r['gm_median']:.6f}", f"{r['ci_lo']:.6f}",
                        f"{r['ci_hi']:.6f}", f"{r['prop_py_faster']:.6f}"])
        for r in byn_rows:
            w.writerow(["by_n", r["regime"], "n_slice", "", "", "", r["n"],
                        r["n_cells"], f"{r['gm_mean']:.6f}",
                        f"{r['gm_median']:.6f}", f"{r['ci_lo']:.6f}",
                        f"{r['ci_hi']:.6f}", f"{r['prop_py_faster']:.6f}"])


REG = {"single_thread": "1T", "multi_thread": "8T"}


def build_tex(group_rows):
    lines = []
    # This table has 100+ data rows and is taller than a page, so it is emitted
    # as a longtable: it anchors in place (like table[H], no floating/clumping)
    # AND breaks across pages. booktabs rules and \scriptsize are preserved.
    header = (r"Reg. & Family & Method & \#cells & GM ratio (95\% CI) & "
              r"median & \% Py faster \\")
    lines.append(r"{\scriptsize")
    lines.append(r"\begin{longtable}{lllrrrr}")
    lines.append(r"\caption{Paired cross-language solve-time ratio. For every "
                 r"shared experimental cell (identical matrix, $n$, RHS seed, "
                 r"method, regime) $R=\log(T_\mathrm{Py}/T_\mathrm{Jl})$ is formed "
                 r"on the per-cell median $t_\mathrm{solve}$; the geometric-mean "
                 r"ratio is $\exp(\overline{R})$ (a value $>1$ means Python is "
                 r"slower by that factor, $<1$ means faster). The 95\% CI is a "
                 r"paired bootstrap over the shared cells; ``\% Py faster'' is the "
                 r"fraction of cells with $R<0$. Generated from CSV.}"
                 r"\label{tab:paired_ratios_v2}\\")
    lines.append(r"\toprule")
    lines.append(header)
    lines.append(r"\midrule")
    lines.append(r"\endfirsthead")
    lines.append(r"\toprule")
    lines.append(header)
    lines.append(r"\midrule")
    lines.append(r"\endhead")
    lines.append(r"\bottomrule")
    lines.append(r"\endfoot")

    def order(r):
        scope_rank = {"overall": 0, "phase": 1, "family_method": 2}[r["scope"]]
        return (0 if r["regime"] == "single_thread" else 1, scope_rank,
                r["phase"], r["family"], r["method"])

    last_regime = None
    for r in sorted(group_rows, key=order):
        if r["regime"] != last_regime:
            if last_regime is not None:
                lines.append(r"\midrule")
            last_regime = r["regime"]
        if r["scope"] == "overall":
            fam, meth = r"\textbf{ALL}", r"\textbf{ALL}"
        elif r["scope"] == "phase":
            fam, meth = f"\\emph{{{r['phase']}}}", r"\emph{all}"
        else:
            fam, meth = sl.tex_escape(r["family"]), sl.tex_escape(r["method"])
        gm = r["gm_mean"]; lo = r["ci_lo"]; hi = r["ci_hi"]
        gmcell = (f"{gm:.3f} [{lo:.3f}, {hi:.3f}]"
                  if math.isfinite(gm) else "--")
        med = f"{r['gm_median']:.3f}" if math.isfinite(r["gm_median"]) else "--"
        pf = (f"{100*r['prop_py_faster']:.0f}\\%"
              if math.isfinite(r["prop_py_faster"]) else "--")
        lines.append(f"{REG.get(r['regime'], r['regime'])} & {fam} & {meth} & "
                     f"{r['n_cells']} & {gmcell} & {med} & {pf} \\\\")
    lines.append(r"\end{longtable}")
    lines.append(r"}")
    return "\n".join(lines) + "\n"


def make_figure(byn_rows, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    CB = {"single_thread": "#0072B2", "multi_thread": "#E69F00"}
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for regime in ("single_thread", "multi_thread"):
        rr = sorted([r for r in byn_rows if r["regime"] == regime],
                    key=lambda x: x["n"])
        if not rr:
            continue
        ns = [r["n"] for r in rr]
        gm = [r["gm_mean"] for r in rr]
        lo = [r["ci_lo"] for r in rr]
        hi = [r["ci_hi"] for r in rr]
        ax.plot(ns, gm, "-o", color=CB[regime], ms=4, lw=1.4,
                label=REG[regime])
        ax.fill_between(ns, lo, hi, color=CB[regime], alpha=0.15)
    ax.axhline(1.0, ls="--", color="black", lw=1,
               label="parity (Py = Jl)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("n (unknowns)")
    ax.set_ylabel(r"geometric-mean $T_\mathrm{Py}/T_\mathrm{Jl}$")
    ax.set_title("Paired Python/Julia solve-time ratio vs problem size\n"
                 "(below parity = Python faster; band = paired bootstrap 95% CI)")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(True, which="both", ls=":", lw=0.5, alpha=0.6)
    paths = []
    for ext in ("pdf", "png"):
        p = os.path.join(out_dir, f"fig_v2_paired_ratio.{ext}")
        fig.savefig(p, bbox_inches="tight", dpi=150)
        paths.append(p)
    plt.close(fig)
    return paths


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bench-dir", default=sl.BENCH_DIR)
    p.add_argument("--out-dir", default=sl.RESULTS_DIR)
    p.add_argument("--no-fig", action="store_true")
    args = p.parse_args()

    rows = sl.load_timing_rows(args.bench_dir)
    units = sl.unit_samples(rows, "t_solve_ns")
    paired = sl.pair_by_cell(units)
    group_rows, byn_rows = build_rows(paired)

    csv_path = os.path.join(args.bench_dir, "paired_ratios.csv")
    write_csv(csv_path, group_rows, byn_rows)
    tex_path = os.path.join(args.out_dir, "paired_ratios_v2.tex")
    with open(tex_path, "w") as fh:
        fh.write(build_tex(group_rows))
    print(f"paired cells: {len(paired)}")
    print(f"wrote {csv_path}")
    print(f"wrote {tex_path}")
    if not args.no_fig:
        for pth in make_figure(byn_rows, args.out_dir):
            print(f"wrote {pth}")

    # console summary
    print("\n--- OVERALL (per regime) ---")
    for r in group_rows:
        if r["scope"] == "overall":
            print(f"  {r['regime']:14} GM={r['gm_mean']:.3f} "
                  f"CI[{r['ci_lo']:.3f},{r['ci_hi']:.3f}] "
                  f"medianGM={r['gm_median']:.3f} "
                  f"%PyFaster={100*r['prop_py_faster']:.0f} "
                  f"n_cells={r['n_cells']}")
    print("--- per family/method (single_thread) ---")
    for r in sorted(group_rows, key=lambda x: (x["phase"], x["family"], x["method"])):
        if r["scope"] == "family_method" and r["regime"] == "single_thread":
            print(f"  {r['family']:34} {r['method']:11} GM={r['gm_mean']:.3f} "
                  f"CI[{r['ci_lo']:.3f},{r['ci_hi']:.3f}] "
                  f"%PyFaster={100*r['prop_py_faster']:.0f} nc={r['n_cells']}")


if __name__ == "__main__":
    main()
