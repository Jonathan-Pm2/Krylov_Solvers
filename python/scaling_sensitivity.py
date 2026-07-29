#!/usr/bin/env python3
"""Round-2 analysis 3 -- scaling-fit robustness and honest confidence intervals.

The round-1 scaling exponents beta1 (from log T = beta0 + beta1 log n) were
fitted over EVERY (n, RHS) unit and their CI used df = (#units - 2). But the RHS
seeds at a fixed n are not independent observations of the scaling law -- they
are repeated draws at the SAME problem size. Treating them as independent inflates
the degrees of freedom and produces implausibly tight CIs
(e.g. Python varied-cond cg single-thread beta1 = 2.823 [2.818, 2.828]).

This script, for every genuine per-family fit (status "ok" in scaling_fits.csv):

  1. reports the EXACT number of distinct n points, the observation unit, and the
     inference method + degrees of freedom;
  2. recomputes the exponent and its 95% CI at the CORRECT observation level --
     one point per distinct n (RHS medians pooled), df = (#distinct_n - 2) -- and
     reports the widened, honest interval next to the naive one;
  3. SENSITIVITY (a): refits at the corrected level after DROPPING the smallest n,
     to check the exponent is not an artifact of a single small-n anchor;
  4. SENSITIVITY (b): fits an overhead model T = c0 + c1 * n^alpha on the same
     per-n points, so a pure-power-law exponent is cross-checked against a model
     that admits a constant-overhead term.

No new benchmark runs: reads the raw per-rep timing CSVs plus scaling_fits.csv.

Outputs:
  * results/scaling_sensitivity_v2.tex   booktabs table (naive vs corrected CI,
        drop-small-n exponent, overhead-model alpha)
  * results/benchmarks/scaling_fits.csv  UPDATED IN PLACE with added columns:
        ci_method, beta1_corrected, beta1_ci_lo_corrected, beta1_ci_hi_corrected,
        df_corrected, distinct_n_used, drop_small_beta1, overhead_c0,
        overhead_c1, overhead_alpha
"""
from __future__ import annotations

import argparse
import csv
import math
import os
from collections import defaultdict

import numpy as np

import statlib_v2 as sl

ADDED_COLS = ["ci_method", "beta1_corrected", "beta1_ci_lo_corrected",
              "beta1_ci_hi_corrected", "df_corrected", "distinct_n_used",
              "drop_small_beta1", "overhead_c0", "overhead_c1", "overhead_alpha"]


def per_n_points(units):
    """(n_array, T_array) with one T per distinct n = median over the RHS-level
    per-unit medians at that n. Returns arrays sorted by n."""
    by_n = defaultdict(list)
    for u in units:
        if math.isfinite(u["median"]) and u["median"] > 0:
            by_n[u["n"]].append(u["median"])
    ns = sorted(by_n)
    T = [float(np.median(by_n[n])) for n in ns]
    return np.array(ns, dtype=float), np.array(T, dtype=float)


def fit_overhead(n, T):
    """T = c0 + c1 * n^alpha via nonlinear least squares on the per-n points.
    Returns (c0, c1, alpha) or NaNs if the fit fails / too few points."""
    if n.size < 4:  # 3 params + slack
        return float("nan"), float("nan"), float("nan")
    from scipy.optimize import curve_fit
    model = lambda x, c0, c1, a: c0 + c1 * np.power(x, a)
    # scale T to ~O(1) for conditioning; recover c0, c1 afterwards
    scale = float(np.median(T))
    y = T / scale
    p0 = [float(y[0]) * 0.1, float(y[-1]) / (n[-1] ** 1.5), 1.5]
    try:
        popt, _ = curve_fit(model, n, y, p0=p0, maxfev=20000,
                            bounds=([-np.inf, 0, 0], [np.inf, np.inf, 6]))
        c0, c1, a = popt
        return float(c0 * scale), float(c1 * scale), float(a)
    except Exception:
        return float("nan"), float("nan"), float("nan")


def analyze(fits, units_all):
    """Attach the corrected fit, drop-small-n fit and overhead fit to every ok
    row of scaling_fits. `units_all` is the raw single- and multi-thread unit
    list from statlib.unit_samples."""
    # index units by (phase, family_group, language, method, regime)
    idx = defaultdict(list)
    for u in units_all:
        d = u["key"]
        idx[(d["phase"], u["family_group"], u["language"], d["method"],
             d["regime"])].append(u)

    enriched = []
    report = []  # analysis rows for the tex table (ok fits only)
    for f in fits:
        row = dict(f)
        for c in ADDED_COLS:
            row.setdefault(c, "")
        if f.get("status") != "ok":
            enriched.append(row)
            continue
        key = (f["phase"], f["family"], f["language"], f["method"], f["regime"])
        us = idx.get(key, [])
        n, T = per_n_points(us)
        row["ci_method"] = "analytic-t, df = distinct_n - 2 (RHS medians pooled)"
        row["distinct_n_used"] = int(np.unique(n).size)

        corrected = sl.ols_loglog(n, T)
        row["beta1_corrected"] = f"{corrected['beta1']:.6f}"
        row["beta1_ci_lo_corrected"] = f"{corrected['beta1_ci_lo']:.6f}"
        row["beta1_ci_hi_corrected"] = f"{corrected['beta1_ci_hi']:.6f}"
        row["df_corrected"] = corrected["df"]

        # drop smallest n
        if n.size >= 4:
            drop = sl.ols_loglog(n[1:], T[1:])
            row["drop_small_beta1"] = f"{drop['beta1']:.6f}"
        else:
            drop = {"beta1": float("nan")}
            row["drop_small_beta1"] = ""

        c0, c1, a = fit_overhead(n, T)
        row["overhead_c0"] = f"{c0:.6g}" if math.isfinite(c0) else ""
        row["overhead_c1"] = f"{c1:.6g}" if math.isfinite(c1) else ""
        row["overhead_alpha"] = f"{a:.6f}" if math.isfinite(a) else ""

        enriched.append(row)
        report.append({
            "phase": f["phase"], "family": f["family"], "method": f["method"],
            "language": f["language"], "regime": f["regime"],
            "distinct_n": int(np.unique(n).size),
            "n_points_naive": int(sl._f(f.get("n_points", "nan"))),
            "beta1_naive": sl._f(f["beta1"]),
            "ci_lo_naive": sl._f(f["beta1_ci_lo"]),
            "ci_hi_naive": sl._f(f["beta1_ci_hi"]),
            "beta1_corr": corrected["beta1"],
            "ci_lo_corr": corrected["beta1_ci_lo"],
            "ci_hi_corr": corrected["beta1_ci_hi"],
            "df_corr": corrected["df"],
            "drop_small_beta1": drop["beta1"],
            "overhead_alpha": a,
        })
    return enriched, report


def load_fits(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def write_fits(path, rows):
    # preserve original column order, then append the new columns
    base_cols = [c for c in rows[0].keys() if c not in ADDED_COLS]
    cols = base_cols + ADDED_COLS
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})


REG = {"single_thread": "1T", "multi_thread": "8T"}


def build_tex(report):
    """One row per ok fit: naive vs corrected 95% CI, drop-small-n exponent,
    overhead-model alpha. Focus on the single-thread canonical regime to keep the
    table readable; multi-thread rows are in the CSV."""
    rows = [r for r in report if r["regime"] == "single_thread"]

    def width(lo, hi):
        return hi - lo if (math.isfinite(lo) and math.isfinite(hi)) else float("nan")

    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Scaling-exponent robustness (single-thread regime). "
                 r"The naive 95\% CI treats every $(n,\mathrm{RHS})$ unit as an "
                 r"independent observation ($\mathrm{df}=\#\text{units}-2$); the "
                 r"corrected CI uses one point per distinct $n$ with the RHS "
                 r"medians pooled ($\mathrm{df}=\#\text{distinct }n-2$), which is "
                 r"the true observation level for a scaling law. "
                 r"$\beta_1^{\text{drop}}$ refits after dropping the smallest $n$; "
                 r"$\alpha$ is the exponent of the overhead model "
                 r"$T=c_0+c_1 n^{\alpha}$ on the same points. Generated from CSV.}")
    lines.append(r"\label{tab:scaling_sensitivity_v2}")
    lines.append(r"\scriptsize")
    lines.append(r"\begin{tabular}{lllrll rl}")
    lines.append(r"\toprule")
    lines.append(r"Family & Method & Lang & \#$n$ & $\beta_1$ naive CI & "
                 r"$\beta_1$ corrected CI & $\beta_1^{\text{drop}}$ & $\alpha$ \\")
    lines.append(r"\midrule")
    last_phase = None
    for r in sorted(rows, key=lambda x: (0 if x["phase"] == "spd" else 1,
                                         x["family"], x["method"], x["language"])):
        if r["phase"] != last_phase:
            if last_phase is not None:
                lines.append(r"\midrule")
            last_phase = r["phase"]

        def ci(b, lo, hi):
            if not math.isfinite(b):
                return "--"
            if math.isfinite(lo):
                return f"{b:.3f} [{lo:.3f}, {hi:.3f}]"
            return f"{b:.3f} [n/a]"
        naive = ci(r["beta1_naive"], r["ci_lo_naive"], r["ci_hi_naive"])
        corr = ci(r["beta1_corr"], r["ci_lo_corr"], r["ci_hi_corr"])
        drop = f"{r['drop_small_beta1']:.3f}" if math.isfinite(r["drop_small_beta1"]) else "--"
        al = f"{r['overhead_alpha']:.3f}" if math.isfinite(r["overhead_alpha"]) else "--"
        lines.append(f"{sl.tex_escape(r['family'])} & {sl.tex_escape(r['method'])} & "
                     f"{r['language'][:2]} & {r['distinct_n']} & {naive} & {corr} & "
                     f"{drop} & {al} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\par\smallskip\footnotesize The corrected interval is the "
                 r"honest one: with only 3--8 distinct problem sizes the exponent "
                 r"is weakly constrained, and the naive interval's apparent "
                 r"precision is an artifact of counting RHS repetitions as "
                 r"independent evidence about the slope.")
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bench-dir", default=sl.BENCH_DIR)
    p.add_argument("--out-dir", default=sl.RESULTS_DIR)
    p.add_argument("--fits", default="scaling_fits.csv")
    args = p.parse_args()

    fits_path = os.path.join(args.bench_dir, args.fits)
    fits = load_fits(fits_path)
    rows = sl.load_timing_rows(args.bench_dir)
    units = sl.unit_samples(rows, "t_solve_ns")

    enriched, report = analyze(fits, units)
    write_fits(fits_path, enriched)
    tex_path = os.path.join(args.out_dir, "scaling_sensitivity_v2.tex")
    with open(tex_path, "w") as fh:
        fh.write(build_tex(report))
    print(f"updated {fits_path} (+{len(ADDED_COLS)} columns)")
    print(f"wrote {tex_path}")

    print("\n--- naive vs corrected CI (single_thread) ---")
    for r in sorted(report, key=lambda x: (x["phase"], x["family"], x["method"],
                                           x["language"])):
        if r["regime"] != "single_thread":
            continue
        wn = r["ci_hi_naive"] - r["ci_lo_naive"]
        wc = (r["ci_hi_corr"] - r["ci_lo_corr"]
              if math.isfinite(r["ci_lo_corr"]) else float("nan"))
        print(f"  {r['family']:26} {r['method']:11} {r['language'][:2]} "
              f"#n={r['distinct_n']} "
              f"naive b1={r['beta1_naive']:.3f}[{r['ci_lo_naive']:.3f},"
              f"{r['ci_hi_naive']:.3f}](w={wn:.3f}) "
              f"corr b1={r['beta1_corr']:.3f}"
              f"[{r['ci_lo_corr']:.3f},{r['ci_hi_corr']:.3f}](w={wc:.3f}) "
              f"drop={r['drop_small_beta1']:.3f} alpha={r['overhead_alpha']:.3f}")


if __name__ == "__main__":
    main()
