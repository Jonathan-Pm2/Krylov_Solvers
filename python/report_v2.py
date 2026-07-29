#!/usr/bin/env python3
"""Round-2 numbers report -- reproducible digest of the re-analysis.

Auto-generated from the round-2 output CSVs (paired_ratios.csv,
variance_components.csv, scaling_fits.csv, parallel_speedup.csv). This is a
PIPELINE ARTIFACT, not hand-written prose: rerun after the analysis scripts to
refresh every number. It exists so the manuscript author can lift figures with a
single source of truth. Run order:

    python paired_ratios.py
    python variance_components.py
    python scaling_sensitivity.py     # writes the corrected columns into scaling_fits.csv
    python scaling_decomp.py
    python parallel_speedup.py
    python report_v2.py               # <- this file

Output: results/numbers_report_v2.md
"""
from __future__ import annotations

import argparse
import csv
import math
import os

import numpy as np

import statlib_v2 as sl


def load(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def sec_paired(bench):
    rows = load(os.path.join(bench, "paired_ratios.csv"))
    out = ["## 1. Paired cross-language ratio (R = log T_Py/T_Jl on t_solve)\n"]
    out.append("Geometric-mean ratio exp(mean R): >1 = Python slower, <1 = Python faster.\n")
    out.append("| scope | regime | GM ratio | 95% CI (paired boot) | median GM | % Py faster | #cells |")
    out.append("|---|---|---|---|---|---|---|")
    for r in rows:
        if r["kind"] == "group" and r["scope"] in ("overall", "phase"):
            name = "OVERALL" if r["scope"] == "overall" else r["phase"]
            out.append(f"| {name} | {r['regime']} | {f(r['gm_ratio_mean']):.3f} | "
                       f"[{f(r['ci_lo']):.3f}, {f(r['ci_hi']):.3f}] | "
                       f"{f(r['gm_ratio_median']):.3f} | "
                       f"{100*f(r['prop_python_faster']):.0f}% | {r['n_cells']} |")
    out.append("\nPer-family/method (single_thread):\n")
    out.append("| family | method | GM ratio | % Py faster | #cells |")
    out.append("|---|---|---|---|---|")
    for r in rows:
        if r["kind"] == "group" and r["scope"] == "family_method" \
                and r["regime"] == "single_thread":
            out.append(f"| {r['family']} | {r['method']} | {f(r['gm_ratio_mean']):.3f} | "
                       f"{100*f(r['prop_python_faster']):.0f}% | {r['n_cells']} |")
    # crossover: ratio vs n (single_thread)
    byn = [r for r in rows if r["kind"] == "by_n" and r["regime"] == "single_thread"]
    out.append("\nRatio vs n (single_thread) -- crossover locator:\n")
    out.append("| n | GM ratio | % Py faster | #cells |")
    out.append("|---|---|---|---|")
    for r in sorted(byn, key=lambda x: int(x["n"])):
        out.append(f"| {r['n']} | {f(r['gm_ratio_mean']):.3f} | "
                   f"{100*f(r['prop_python_faster']):.0f}% | {r['n_cells']} |")
    return "\n".join(out) + "\n"


def sec_variance(bench):
    rows = load(os.path.join(bench, "variance_components.csv"))
    fin = [r for r in rows if math.isfinite(f(r["between_rhs_cv"]))
           and math.isfinite(f(r["within_rhs_cv"]))]
    b = np.array([f(r["between_rhs_cv"]) for r in fin])
    w = np.array([f(r["within_rhs_cv"]) for r in fin])
    out = ["## 2. Variance components (single_thread; CV = std / headline)\n"]
    out.append(f"- units with >= 2 RHS: {len(fin)}\n")
    out.append(f"- median CV within-RHS (rep): {100*np.median(w):.1f}%\n")
    out.append(f"- median CV between-RHS (RHS): {100*np.median(b):.1f}%\n")
    out.append(f"- between-RHS > within-RHS in {int(np.sum(b>w))}/{len(fin)} units "
               f"({100*np.mean(b>w):.0f}%)\n")
    for lang in ("julia", "python"):
        wl = [f(r["within_rhs_cv"]) for r in fin if r["language"] == lang]
        bl = [f(r["between_rhs_cv"]) for r in fin if r["language"] == lang]
        out.append(f"- {lang}: median CV_rep = {100*np.median(wl):.1f}%, "
                   f"median CV_RHS = {100*np.median(bl):.1f}%\n")
    out.append("\nNote: within-RHS spread is heavy-tailed for Julia (GC/JIT), which "
               "is why the headline uses a median. See aggregate_note_v2.md.\n")
    return "".join(out)


def sec_scaling(bench):
    rows = load(os.path.join(bench, "scaling_fits.csv"))
    ok = [r for r in rows if r["status"] == "ok" and r["regime"] == "single_thread"]
    out = ["## 3. Scaling exponent beta1 -- naive vs corrected observation level (single_thread)\n"]
    out.append("naive CI: df = #units-2 (RHS treated as independent). "
               "corrected CI: df = #distinct_n-2 (RHS medians pooled).\n")
    out.append("| family | method | lang | #n | naive beta1 [CI] | corrected beta1 [CI] | "
               "drop-small beta1 | overhead alpha |")
    out.append("|---|---|---|---|---|---|---|---|")
    for r in sorted(ok, key=lambda x: (x["family"], x["method"], x["language"])):
        b, lo, hi = f(r["beta1"]), f(r["beta1_ci_lo"]), f(r["beta1_ci_hi"])
        bc, loc, hic = f(r["beta1_corrected"]), f(r["beta1_ci_lo_corrected"]), f(r["beta1_ci_hi_corrected"])
        drop = f(r["drop_small_beta1"])
        al = f(r["overhead_alpha"])
        out.append(f"| {r['family']} | {r['method']} | {r['language'][:2]} | "
                   f"{r['distinct_n_used']} | {b:.3f} [{lo:.3f}, {hi:.3f}] | "
                   f"{bc:.3f} [{loc:.3f}, {hic:.3f}] | "
                   f"{drop:.3f} | {al:.3f} |")
    return "\n".join(out) + "\n"


def sec_decomp(out_dir):
    # decomposition numbers are printed by scaling_decomp.py; re-derive here from raw
    import scaling_decomp as sd
    rows = sl.load_timing_rows()
    res = sd.analyze(rows, "single_thread")
    out = ["## 4. Scaling decomposition beta_total = beta_cost + beta_count (single_thread)\n"]
    out.append("beta_iter (iteration-count) is language-independent within a family; "
               "the cross-language total difference is carried by beta_cost.\n")
    out.append("| family | method | lang | beta_cost | beta_iter | beta_count | "
               "beta_total | cost+count |")
    out.append("|---|---|---|---|---|---|---|---|")
    for r in sorted(res, key=lambda x: (x["family"], x["method"], x["language"])):
        out.append(f"| {r['family']} | {r['method']} | {r['language'][:2]} | "
                   f"{r['beta_cost']:.3f} | {r['beta_iter']:.3f} | {r['beta_count']:.3f} | "
                   f"{r['beta_total']:.3f} | {r['beta_sum']:.3f} |")
    return "\n".join(out) + "\n"


def sec_speedup(bench):
    rows = load(os.path.join(bench, "parallel_speedup.csv"))
    s8 = np.array([f(r["s8"]) for r in rows])
    out = ["## 5. 8-thread speedup S8 = T1/T8, efficiency E8 = S8/8\n"]
    out.append(f"- paired cells: {len(rows)}\n")
    out.append(f"- overall median S8 = {np.median(s8):.2f}, median E8 = {np.median(s8)/8:.2f}\n")
    out.append(f"- cells with S8 > 1 (8T helps): {100*np.mean(s8>1):.0f}%; max S8 = {np.max(s8):.2f}\n")
    # by size bucket overall
    def bucket(n):
        n = int(n)
        return "small(<2k)" if n < 2000 else ("medium(2k-20k)" if n < 20000 else "large(>=20k)")
    out.append("\n| size | median S8 | % cells S8>1 | #cells |")
    out.append("|---|---|---|---|")
    from collections import defaultdict
    by = defaultdict(list)
    for r in rows:
        by[bucket(r["n"])].append(f(r["s8"]))
    for name in ("small(<2k)", "medium(2k-20k)", "large(>=20k)"):
        v = np.array(by[name])
        out.append(f"| {name} | {np.median(v):.2f} | {100*np.mean(v>1):.0f}% | {len(v)} |")
    out.append("\nTHREAT TO VALIDITY: core affinity (P/E), turbo boost and CPU governor "
               "were NOT pinned. These are unpinned wall-clock speedups, not controlled "
               "strong scaling.\n")
    return "\n".join(out) if out[-1].endswith("\n") else "\n".join(out) + "\n"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bench-dir", default=sl.BENCH_DIR)
    p.add_argument("--out-dir", default=sl.RESULTS_DIR)
    args = p.parse_args()

    parts = [
        "# Round-2 statistical re-analysis -- numbers report\n",
        "_Auto-generated by report_v2.py from the round-2 CSVs. No new benchmark "
        "runs; the existing per-repetition timings were re-analyzed._\n",
        sec_paired(args.bench_dir),
        sec_variance(args.bench_dir),
        sec_scaling(args.bench_dir),
        sec_decomp(args.out_dir),
        sec_speedup(args.bench_dir),
    ]
    path = os.path.join(args.out_dir, "numbers_report_v2.md")
    with open(path, "w") as fh:
        fh.write("\n".join(parts))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
