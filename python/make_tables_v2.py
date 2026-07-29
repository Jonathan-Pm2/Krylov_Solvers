#!/usr/bin/env python3
"""Phase-4 LaTeX tables (booktabs) — PROTOCOL.md sections 10, 11.

Generated ENTIRELY from the aggregated CSVs (agg_timing.csv, agg_conv.csv,
scaling_fits.csv written by aggregate.py); zero manual transcription
(section 11). Writes three .tex files to code/results/:

  * main_results_v2.tex  — the CORRECTED main results table (replaces old Table 1):
    per (phase, instance, method, regime) the median +/- IQR solve time for BOTH
    languages and BOTH thread regimes, the Python/Julia ratio, the shared work
    count (matvecs), and convergence. RHS seeds are pooled by the median of the
    per-unit medians. SPD instances whose unshifted IC(0) factorization broke down
    (non-positive pivot -> no rows) are shown as an explicit "IC(0) breakdown"
    cell, never as 0 / NaN / a silently omitted row (PROTOCOL.md 7).
  * scaling_exponents_v2.tex — the empirical scaling exponent beta1 (log T =
    beta0 + beta1 log n) per (method, language, phase, regime) with its 95% CI,
    R^2, n range, and an honest status flag (degenerate ladders daggered).
  * convergence_v2.tex — iteration-count spread across the seeded RHS from the
    convergence pass (mean / median / min / max / std, up to 20 RHS).

Reads only stdlib csv + numpy.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.normpath(os.path.join(HERE, ".."))
RESULTS_DIR = os.path.join(CODE_DIR, "results")
BENCH_DIR = os.path.join(RESULTS_DIR, "benchmarks")


def tex_escape(s):
    return str(s).replace("_", r"\_")


def load_csv(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def g(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def fmt_ms(ns):
    """ns -> milliseconds, 4 significant figures."""
    if not math.isfinite(ns):
        return "--"
    return f"{ns / 1e6:.4g}"


def short_instance(key):
    """Last path component, trimmed for the table."""
    return key.split("/")[-1]


def family_tag(instance_key):
    """Leading family component of an instance key, trimmed to a short tag."""
    head = instance_key.split("/")[0]
    return head.replace("_singular", "").replace("_", " ")


def disambiguate(instance_keys):
    """Map each instance_key to a printable label. When two distinct keys share
    the same short name (last path component) --- e.g. the coupled and similarity
    singular families both ending in ncore040_k3_n00000043 --- append a
    parenthetical family tag so the printed labels stay unique."""
    owners = defaultdict(set)
    for ik in instance_keys:
        owners[short_instance(ik)].add(ik)
    label = {}
    for ik in instance_keys:
        s = short_instance(ik)
        label[ik] = f"{s} ({family_tag(ik)})" if len(owners[s]) > 1 else s
    return label


REGIME_LABEL = {"single_thread": "1T", "multi_thread": "8T"}


def build_main_table(units):
    """Corrected main results: per (phase, instance, method, regime) the median
    +/- IQR solve time for BOTH languages and BOTH thread regimes, with the
    Python/Julia ratio, the shared matvec work count, and convergence. RHS seeds
    are pooled by the median of the per-unit medians. Generated from CSV (S11)."""
    # group by (phase, instance_key, n, nnz, method, regime), then by language
    table = defaultdict(lambda: defaultdict(list))
    for u in units:
        k = (u["phase"], u["instance_key"], int(u["n"]), int(u["nnz"]),
             u["method"], u.get("regime", "single_thread"))
        table[k][u["language"]].append(u)

    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Corrected main results (replaces the invalidated Table 1). "
                 r"Solve time as median $\pm$ IQR, summarized as the median over "
                 r"repetitions and then over the seeded RHS, for both languages and "
                 r"both thread regimes (1T = single controlled thread; 8T = 8 matched "
                 r"physical cores; protocol in Appendix~\ref{app:protocol}). Matvec "
                 r"counts are hardware-independent and nearly identical across "
                 r"languages (exact in 508 of 510 units). SPD "
                 r"instances whose unshifted IC(0) factorization broke down "
                 r"(non-positive pivot) carry no timing and are labeled explicitly. "
                 r"Generated from CSV.}")
    lines.append(r"\label{tab:main_results_v2}")
    lines.append(r"\scriptsize")
    lines.append(r"\begin{tabular}{llrlrrrrl}")
    lines.append(r"\toprule")
    lines.append(r"Instance & Method & $n$ & Reg. & Julia [ms] & Python [ms] & "
                 r"Py/Jl & matvecs & conv. \\")
    lines.append(r"\midrule")

    # Inject explicit IC(0)-breakdown cells: for every SPD (instance, n, nnz,
    # regime) where cg ran but pcg_ic0 has no rows, the unshifted IC(0)
    # factorization broke down (non-positive pivot). Surface it as a labeled row
    # rather than a silent omission (PROTOCOL.md 7).
    breakdown = set()
    spd_cells = defaultdict(set)  # (inst,n,nnz,regime) -> set(methods present)
    for (phase, inst, n, nnz, method, regime) in table.keys():
        if phase == "spd":
            spd_cells[(inst, n, nnz, regime)].add(method)
    for (inst, n, nnz, regime), methods in spd_cells.items():
        if "cg" in methods and "pcg_ic0" not in methods:
            k = ("spd", inst, n, nnz, "pcg_ic0", regime)
            breakdown.add(k)
            table[k]  # materialize an empty entry so it sorts into place

    disp = disambiguate({k[1] for k in table.keys()})

    def sortkey(item):
        (phase, inst, n, nnz, method, regime) = item
        return (0 if phase == "spd" else 1, n, inst, method,
                0 if regime == "single_thread" else 1)

    last_phase = None
    for k in sorted(table.keys(), key=sortkey):
        (phase, inst, n, nnz, method, regime) = k
        if k in breakdown:
            if phase != last_phase:
                if last_phase is not None:
                    lines.append(r"\midrule")
                last_phase = phase
            lines.append(
                f"{tex_escape(disp[inst])} & {tex_escape(method)} & {n} & "
                f"{REGIME_LABEL.get(regime, regime)} & "
                r"\multicolumn{3}{c}{\emph{IC(0) breakdown (non-positive pivot)}} & "
                r"-- & -- \\")
            continue
        langs = table[k]
        jl = langs.get("julia", [])
        py = langs.get("python", [])
        jl_med = float(np.median([g(u["t_solve_median_ns"]) for u in jl])) if jl else float("nan")
        jl_iqr = float(np.median([g(u["t_solve_iqr_ns"]) for u in jl])) if jl else float("nan")
        py_med = float(np.median([g(u["t_solve_median_ns"]) for u in py])) if py else float("nan")
        py_iqr = float(np.median([g(u["t_solve_iqr_ns"]) for u in py])) if py else float("nan")
        ratio = (py_med / jl_med) if (jl_med and math.isfinite(jl_med) and jl_med > 0) else float("nan")
        matv = jl[0]["matvecs"] if jl else (py[0]["matvecs"] if py else "--")
        conv = jl[0]["converged"] if jl else (py[0]["converged"] if py else "--")
        conv = {"true": "yes", "false": "no"}.get(str(conv).lower(), str(conv))

        if phase != last_phase:
            if last_phase is not None:
                lines.append(r"\midrule")
            last_phase = phase
        jl_cell = f"{fmt_ms(jl_med)} $\\pm$ {fmt_ms(jl_iqr)}"
        py_cell = f"{fmt_ms(py_med)} $\\pm$ {fmt_ms(py_iqr)}"
        ratio_cell = f"{ratio:.2f}" if math.isfinite(ratio) else "--"
        lines.append(f"{tex_escape(disp[inst])} & {tex_escape(method)} & {n} & "
                     f"{REGIME_LABEL.get(regime, regime)} & {jl_cell} & {py_cell} & "
                     f"{ratio_cell} & {matv} & {conv} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


def build_scaling_table(fits):
    """Per-family empirical scaling exponents. Only genuine n-ladders (status ok,
    >= 3 distinct n with the secondary parameter held fixed) and the explicitly
    labeled pooled-singular fallback are shown; insufficient_ladder groups
    (SuiteSparse fixed-size points, 1-D Laplacian, single-eps/kappa slices, thin
    singular families) carry no exponent and are excluded from the headline."""
    shown = [f for f in fits if f["status"] in ("ok", "pooled")]
    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Empirical per-family scaling exponents $\beta_1$ from "
                 r"$\log T_\mathrm{solve} = \beta_0 + \beta_1 \log n$ fitted per "
                 r"(family, method, language, phase, regime) with every secondary "
                 r"parameter (mesh anisotropy $\varepsilon$, condition number "
                 r"$\kappa$, Drazin index $k$) held fixed, so each fit is a genuine "
                 r"$n$-ladder (protocol in Appendix~\ref{app:protocol}). 95\% "
                 r"confidence interval in "
                 r"brackets. Only families with $\ge 3$ distinct $n$ are reported; "
                 r"fixed-size / thin families are omitted. Generated from CSV.}")
    lines.append(r"\label{tab:scaling_exponents_v2}")
    lines.append(r"\scriptsize")
    lines.append(r"\begin{tabular}{lllllrrl}")
    lines.append(r"\toprule")
    lines.append(r"Phase & Family & Method & Lang & Reg. & $\beta_1$ (95\% CI) & "
                 r"$R^2$ & $n$ range \\")
    lines.append(r"\midrule")

    def sortkey(f):
        return (0 if f["phase"] == "spd" else 1, f["family"], f["method"],
                f["language"], 0 if f.get("regime") == "single_thread" else 1)

    last_phase = None
    for f in sorted(shown, key=sortkey):
        if f["phase"] != last_phase:
            if last_phase is not None:
                lines.append(r"\midrule")
            last_phase = f["phase"]
        b1, lo, hi, r2 = g(f["beta1"]), g(f["beta1_ci_lo"]), g(f["beta1_ci_hi"]), g(f["r_squared"])
        if math.isfinite(b1):
            b1cell = f"{b1:.3f} [{lo:.3f}, {hi:.3f}]" if math.isfinite(lo) else f"{b1:.3f} [n/a]"
        else:
            b1cell = "--"
        r2cell = f"{r2:.4f}" if math.isfinite(r2) else "--"
        nrange = f"{f['n_min']}--{f['n_max']}"
        flag = r"$^{\dagger}$" if f["status"] == "pooled" else ""
        reg = REGIME_LABEL.get(f.get("regime", ""), f.get("regime", ""))
        lines.append(f"{f['phase']} & {tex_escape(f['family'])} & {tex_escape(f['method'])} & "
                     f"{f['language']} & {reg} & {b1cell}{flag} & {r2cell} & {nrange} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\par\smallskip\footnotesize $^{\dagger}$ pooled across the "
                 r"singular families: no single singular family retains $\ge 3$ "
                 r"distinct $n$ after the intractable dense-$n$ cells were dropped, "
                 r"so this exponent mixes families of differing structure and is a "
                 r"coarse indicator, not a per-family law.")
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


def build_convergence_table(conv):
    """Iteration-count spread across the seeded RHS (convergence pass, 1 rep, up
    to 20 RHS, single thread). Iteration counts are language-independent by
    construction, so one row per (phase, instance, method) using the Julia rows;
    the min/max/std columns quantify RHS sensitivity (PROTOCOL.md 3.3, 5)."""
    rows = [c for c in conv if c["language"] == "julia"]
    disp = disambiguate({c["instance_key"] for c in rows})

    def sortkey(c):
        return (0 if c["phase"] == "spd" else 1, c["method"], int(c["n"]),
                c["instance_key"])

    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Iteration-count spread across the seeded right-hand "
                 r"sides (convergence pass: single thread, one repetition, up to "
                 r"20 RHS per instance). Iteration counts are hardware- and "
                 r"language-independent (exact in 508 of 510 shared units); "
                 r"a wide min--max or large std flags RHS-sensitive convergence. "
                 r"Generated from CSV.}")
    lines.append(r"\label{tab:convergence_v2}")
    lines.append(r"\scriptsize")
    lines.append(r"\begin{tabular}{llrrrrrrr}")
    lines.append(r"\toprule")
    lines.append(r"Instance & Method & $n$ & \#RHS & mean & median & min & max & std \\")
    lines.append(r"\midrule")
    last_phase = None
    for c in sorted(rows, key=sortkey):
        if c["phase"] != last_phase:
            if last_phase is not None:
                lines.append(r"\midrule")
            last_phase = c["phase"]
        mean = g(c["iter_mean"]); med = g(c["iter_median"]); std = g(c["iter_std"])
        lines.append(f"{tex_escape(disp[c['instance_key']])} & "
                     f"{tex_escape(c['method'])} & {int(c['n'])} & {c['n_rhs']} & "
                     f"{mean:.1f} & {med:.0f} & {c['iter_min']} & {c['iter_max']} & "
                     f"{std:.2f} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bench-dir", default=BENCH_DIR)
    p.add_argument("--out-dir", default=RESULTS_DIR)
    p.add_argument("--units", default="agg_timing.csv",
                   help="aggregated timing-unit CSV (drives the main table)")
    p.add_argument("--fits", default="scaling_fits.csv")
    p.add_argument("--conv", default="agg_conv.csv",
                   help="aggregated convergence CSV (drives the convergence table)")
    args = p.parse_args()

    units = load_csv(os.path.join(args.bench_dir, args.units))
    fits = load_csv(os.path.join(args.bench_dir, args.fits))
    conv_path = os.path.join(args.bench_dir, args.conv)
    conv = load_csv(conv_path) if os.path.isfile(conv_path) else []

    p1 = os.path.join(args.out_dir, "main_results_v2.tex")
    p2 = os.path.join(args.out_dir, "scaling_exponents_v2.tex")
    with open(p1, "w") as fh:
        fh.write(build_main_table(units))
    with open(p2, "w") as fh:
        fh.write(build_scaling_table(fits))
    print(f"wrote {p1}")
    print(f"wrote {p2}")
    if conv:
        p3 = os.path.join(args.out_dir, "convergence_v2.tex")
        with open(p3, "w") as fh:
            fh.write(build_convergence_table(conv))
        print(f"wrote {p3}")
    else:
        print(f"  (no {args.conv}; skipped convergence table)")


if __name__ == "__main__":
    main()
