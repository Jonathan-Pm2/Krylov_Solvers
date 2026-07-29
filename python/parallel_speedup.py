#!/usr/bin/env python3
"""Round-2 analysis 5 -- 8-thread parallel speedup and efficiency.

The harness ran every unit under two regimes: single_thread (1T) and
multi_thread (8T). For each shared cell (phase, method, family, instance, n, RHS
seed, language) we form

    S8 = T1 / T8            (parallel speedup on the per-cell median t_solve)
    E8 = S8 / 8             (parallel efficiency; 1.0 = perfect 8x)

and summarize where 8 threads actually help (S8 > 1), where they do not (S8 <= 1,
i.e. 8T is no faster or slower than 1T), broken down by problem size. Sparse
iterative solvers are memory-bandwidth bound, so small problems often show
S8 <= 1 (threading overhead dominates) while only larger problems approach any
useful speedup.

THREAT TO VALIDITY (must be stated in the manuscript, not glossed over): this
study did NOT pin P-core/E-core affinity, did NOT disable turbo boost, and did
NOT fix the CPU frequency governor. On a hybrid CPU the 8 software threads may
land on a mix of performance and efficiency cores, and turbo/governor behavior
differs between the 1-thread and 8-thread runs. The speedup numbers below are
therefore indicative of end-to-end wall-clock behavior on an unpinned desktop,
NOT a clean strong-scaling measurement. Do not claim controlled scaling.

No new benchmark runs: reads only the raw per-rep timing CSVs.

Outputs:
  * results/benchmarks/parallel_speedup.csv   one row per shared cell (S8, E8)
  * results/speedup_v2.tex                     summary table by size / method
  * results/fig_v2_speedup.{pdf,png}           S8 vs n per method x language
"""
from __future__ import annotations

import argparse
import csv
import math
import os
from collections import defaultdict

import numpy as np

import statlib_v2 as sl

# key that identifies a cell independent of the thread regime
CELL = ("phase", "method", "family", "instance_key", "n", "nnz", "seed",
        "language")


def paired_regimes(units):
    """{cell_key: {"single_thread": median, "multi_thread": median, ...meta}}"""
    cells = defaultdict(dict)
    for u in units:
        d = u["key"]
        ck = (d["phase"], d["method"], d["family"], d["instance_key"], d["n"],
              d["nnz"], d["seed"], u["language"])
        cells[ck][d["regime"]] = u
    out = {}
    for ck, regs in cells.items():
        if "single_thread" in regs and "multi_thread" in regs:
            out[ck] = regs
    return out


def build_records(paired):
    recs = []
    for ck, regs in paired.items():
        t1 = regs["single_thread"]["median"]
        t8 = regs["multi_thread"]["median"]
        if not (math.isfinite(t1) and math.isfinite(t8) and t1 > 0 and t8 > 0):
            continue
        phase, method, family, inst, n, nnz, seed, lang = ck
        s8 = t1 / t8
        recs.append(dict(phase=phase, method=method,
                         family=sl.family_group(family, inst),
                         instance_key=inst, n=int(n), nnz=int(nnz), seed=seed,
                         language=lang, t1_ns=t1, t8_ns=t8, s8=s8, e8=s8 / 8.0))
    return recs


def write_csv(path, recs):
    cols = ["phase", "language", "method", "family", "instance_key", "n", "nnz",
            "seed", "t1_ns", "t8_ns", "s8", "e8"]
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for r in sorted(recs, key=lambda x: (x["phase"], x["language"],
                                             x["method"], x["n"], x["family"])):
            w.writerow([r["phase"], r["language"], r["method"], r["family"],
                        r["instance_key"], r["n"], r["nnz"], r["seed"],
                        f"{r['t1_ns']:.1f}", f"{r['t8_ns']:.1f}",
                        f"{r['s8']:.4f}", f"{r['e8']:.4f}"])


def size_bucket(n):
    if n < 2000:
        return "small (n<2k)"
    if n < 20000:
        return "medium (2k-20k)"
    return "large (n>=20k)"


BUCKET_ORDER = {"small (n<2k)": 0, "medium (2k-20k)": 1, "large (n>=20k)": 2}


def summarize(recs):
    """Median S8/E8 and fraction with S8>1, per (language, method, size bucket)."""
    groups = defaultdict(list)
    for r in recs:
        groups[(r["language"], r["method"], size_bucket(r["n"]))].append(r)
    rows = []
    for (lang, method, bucket), rr in groups.items():
        s8 = np.array([r["s8"] for r in rr])
        rows.append(dict(language=lang, method=method, bucket=bucket,
                         n_cells=len(rr), s8_median=float(np.median(s8)),
                         e8_median=float(np.median(s8) / 8.0),
                         frac_help=float(np.mean(s8 > 1.0)),
                         s8_max=float(np.max(s8))))
    return rows


def build_tex(summary):
    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{8-thread parallel speedup $S_8=T_1/T_8$ and efficiency "
                 r"$E_8=S_8/8$ (median over shared cells), by problem-size bucket. "
                 r"``\% help'' is the fraction of cells with $S_8>1$ (8 threads "
                 r"faster than 1). Sparse iterative kernels are bandwidth-bound, so "
                 r"small problems typically show $S_8\le 1$. \emph{Threat to "
                 r"validity: core affinity, turbo boost and the CPU frequency "
                 r"governor were not pinned; these are unpinned wall-clock "
                 r"speedups, not a controlled strong-scaling study.} Generated from "
                 r"CSV.}")
    lines.append(r"\label{tab:speedup_v2}")
    lines.append(r"\scriptsize")
    lines.append(r"\begin{tabular}{lllrrrr}")
    lines.append(r"\toprule")
    lines.append(r"Lang & Method & Size & \#cells & median $S_8$ & median $E_8$ & "
                 r"\% help \\")
    lines.append(r"\midrule")
    last_lang = None
    for r in sorted(summary, key=lambda x: (x["language"], x["method"],
                                            BUCKET_ORDER[x["bucket"]])):
        if r["language"] != last_lang:
            if last_lang is not None:
                lines.append(r"\midrule")
            last_lang = r["language"]
        lines.append(f"{r['language']} & {sl.tex_escape(r['method'])} & {r['bucket']} & "
                     f"{r['n_cells']} & {r['s8_median']:.2f} & {r['e8_median']:.2f} & "
                     f"{100*r['frac_help']:.0f}\\% \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


def make_figure(recs, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    CB = {"cg": "#0072B2", "pcg_jacobi": "#E69F00", "pcg_ic0": "#009E73",
          "gmres": "#D55E00", "dgmres": "#CC79A7"}
    MK = {"julia": "o", "python": "s"}
    # median S8 per (method, language, n)
    agg = defaultdict(list)
    for r in recs:
        agg[(r["method"], r["language"], r["n"])].append(r["s8"])
    methods = sorted({m for (m, l, n) in agg})
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for method in methods:
        for lang in ("julia", "python"):
            ns = sorted({n for (m, l, n) in agg if m == method and l == lang})
            if not ns:
                continue
            ys = [float(np.median(agg[(method, lang, n)])) for n in ns]
            ax.plot(ns, ys, marker=MK[lang], ls="-" if lang == "julia" else "--",
                    color=CB.get(method, "black"), ms=4, lw=1.2,
                    label=f"{method} {lang[:2]}")
    ax.axhline(1.0, ls=":", color="black", lw=1.2, label="S8 = 1 (no gain)")
    ax.set_xscale("log")
    ax.set_xlabel("n (unknowns)")
    ax.set_ylabel(r"parallel speedup $S_8 = T_1 / T_8$")
    ax.set_title("8-thread speedup vs problem size\n"
                 "(unpinned affinity/turbo/governor -- indicative, not controlled)")
    ax.legend(frameon=False, fontsize=6.5, ncol=2, loc="upper left")
    ax.grid(True, which="both", ls=":", lw=0.5, alpha=0.6)
    paths = []
    for ext in ("pdf", "png"):
        p = os.path.join(out_dir, f"fig_v2_speedup.{ext}")
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
    paired = paired_regimes(units)
    recs = build_records(paired)
    summary = summarize(recs)

    csv_path = os.path.join(args.bench_dir, "parallel_speedup.csv")
    write_csv(csv_path, recs)
    tex_path = os.path.join(args.out_dir, "speedup_v2.tex")
    with open(tex_path, "w") as fh:
        fh.write(build_tex(summary))
    print(f"paired cells: {len(recs)}")
    print(f"wrote {csv_path}")
    print(f"wrote {tex_path}")
    if not args.no_fig:
        for pth in make_figure(recs, args.out_dir):
            print(f"wrote {pth}")

    s8all = np.array([r["s8"] for r in recs])
    print(f"\n  overall median S8 = {np.median(s8all):.2f}  "
          f"E8 = {np.median(s8all)/8:.2f}  "
          f"cells with S8>1: {100*np.mean(s8all>1):.0f}%  "
          f"max S8 = {np.max(s8all):.2f}")
    print("  by language / method / size:")
    for r in sorted(summary, key=lambda x: (x["language"], x["method"],
                                            BUCKET_ORDER[x["bucket"]])):
        print(f"    {r['language']:7} {r['method']:11} {r['bucket']:16} "
              f"n_cells={r['n_cells']:3} S8={r['s8_median']:.2f} "
              f"E8={r['e8_median']:.2f} %help={100*r['frac_help']:.0f} "
              f"maxS8={r['s8_max']:.2f}")


if __name__ == "__main__":
    main()
