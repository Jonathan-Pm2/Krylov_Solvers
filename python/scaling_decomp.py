#!/usr/bin/env python3
"""Round-2 analysis 4 -- decomposing the total scaling exponent.

A larger total solve-time exponent can come from two very different causes:
the per-matvec (per-iteration) COST growing with n, or the ITERATION COUNT
growing with n. Blaming a language for a steep total slope without separating
these is a mistake. Because the harness records t_solve, matvecs and iterations
per unit, and time_per_matvec = t_solve / matvecs by construction, the total
exponent decomposes exactly:

    log T_solve = log(time_per_matvec) + log(matvecs)
    =>  beta_total = beta_percost + beta_count           (algebraic identity)

and matvecs ~ iterations for both CG and (D)GMRES, so beta_count ~ beta_iter.

For each (family, method, language) with a genuine n-ladder we fit three
log-log exponents at the CORRECT observation level (one point per distinct n,
RHS medians pooled):

  (a) beta_percost : log(time_per_matvec) vs log n   -- per-iteration cost growth
  (b) beta_iter    : log(iterations)      vs log n   -- iteration-count growth
      beta_count   : log(matvecs)         vs log n   -- (identity partner of a+c)
  (c) beta_total   : log(t_solve)         vs log n   -- total

and report beta_percost + beta_count next to beta_total to confirm the identity.
This attributes a steep total slope to cost-per-iteration vs iteration growth,
rather than to "the language".

No new benchmark runs: reads only the raw per-rep timing CSVs.

Output: results/scaling_decomp_v2.tex
"""
from __future__ import annotations

import argparse
import math
import os
from collections import defaultdict

import numpy as np

import statlib_v2 as sl


def per_unit_medians(rows, cols):
    """Return {(unit_key_tuple, language): {col: median-over-reps}} for the given
    value columns, plus n and family_group."""
    KEY = sl.UNIT_KEYS
    groups = defaultdict(list)
    for r in rows:
        groups[tuple(r[k] for k in KEY) + (r["language"],)].append(r)
    out = {}
    for key, recs in groups.items():
        d = dict(zip(KEY, key[:-1]))
        entry = {"n": int(d["n"]),
                 "family_group": sl.family_group(d["family"], d["instance_key"]),
                 "phase": d["phase"], "method": d["method"],
                 "regime": d["regime"], "language": key[-1]}
        for c in cols:
            s = np.array([sl._f(r[c]) for r in recs], dtype=float)
            s = s[np.isfinite(s)]
            entry[c] = float(np.median(s)) if s.size else float("nan")
        out[key] = entry
    return out


def per_n(entries, col):
    by_n = defaultdict(list)
    for e in entries:
        v = e[col]
        if math.isfinite(v) and v > 0:
            by_n[e["n"]].append(v)
    ns = sorted(by_n)
    T = [float(np.median(by_n[n])) for n in ns]
    return np.array(ns, dtype=float), np.array(T, dtype=float)


def fit_beta(entries, col):
    n, T = per_n(entries, col)
    if np.unique(n).size < 3:
        return float("nan"), int(np.unique(n).size)
    res = sl.ols_loglog(n, T)
    return res["beta1"], res["distinct_n"]


def analyze(rows, regime="single_thread"):
    cols = ["t_solve_ns", "time_per_matvec_ns", "matvecs", "iterations"]
    um = per_unit_medians(rows, cols)
    groups = defaultdict(list)
    for e in um.values():
        if e["regime"] != regime:
            continue
        groups[(e["phase"], e["family_group"], e["method"], e["language"])].append(e)

    out = []
    for (phase, fam, method, lang), entries in sorted(groups.items()):
        b_total, dn = fit_beta(entries, "t_solve_ns")
        b_cost, _ = fit_beta(entries, "time_per_matvec_ns")
        b_count, _ = fit_beta(entries, "matvecs")
        b_iter, _ = fit_beta(entries, "iterations")
        if not math.isfinite(b_total) or dn < 3:
            continue
        out.append(dict(phase=phase, family=fam, method=method, language=lang,
                        distinct_n=dn, beta_total=b_total, beta_cost=b_cost,
                        beta_count=b_count, beta_iter=b_iter,
                        beta_sum=b_cost + b_count))
    return out


def build_tex(rows):
    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Decomposition of the total solve-time scaling exponent "
                 r"(single-thread regime, correct per-$n$ observation level). "
                 r"$\beta_{\text{cost}}$ is the exponent of the per-matvec cost, "
                 r"$\beta_{\text{iter}}$ that of the iteration count, "
                 r"$\beta_{\text{count}}$ that of the matvec count, and "
                 r"$\beta_{\text{total}}$ that of $t_\mathrm{solve}$. Since "
                 r"$t_\mathrm{solve}=\text{matvecs}\times\text{time\_per\_matvec}$ "
                 r"by construction, $\beta_{\text{cost}}+\beta_{\text{count}}$ "
                 r"reproduces $\beta_{\text{total}}$ (identity check). A steep total "
                 r"slope is thereby attributed to per-iteration cost growth vs "
                 r"iteration-count growth, not to the language. Generated from CSV.}")
    lines.append(r"\label{tab:scaling_decomp_v2}")
    lines.append(r"\scriptsize")
    lines.append(r"\begin{tabular}{lllr rrrr r}")
    lines.append(r"\toprule")
    lines.append(r"Family & Method & Lang & \#$n$ & $\beta_{\text{cost}}$ & "
                 r"$\beta_{\text{iter}}$ & $\beta_{\text{count}}$ & "
                 r"$\beta_{\text{total}}$ & $\beta_{\text{cost}}{+}\beta_{\text{count}}$ \\")
    lines.append(r"\midrule")
    last_phase = None

    def f(x):
        return f"{x:.3f}" if math.isfinite(x) else "--"

    for r in sorted(rows, key=lambda x: (0 if x["phase"] == "spd" else 1,
                                         x["family"], x["method"], x["language"])):
        if r["phase"] != last_phase:
            if last_phase is not None:
                lines.append(r"\midrule")
            last_phase = r["phase"]
        lines.append(f"{sl.tex_escape(r['family'])} & {sl.tex_escape(r['method'])} & "
                     f"{r['language'][:2]} & {r['distinct_n']} & {f(r['beta_cost'])} & "
                     f"{f(r['beta_iter'])} & {f(r['beta_count'])} & "
                     f"{f(r['beta_total'])} & {f(r['beta_sum'])} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bench-dir", default=sl.BENCH_DIR)
    p.add_argument("--out-dir", default=sl.RESULTS_DIR)
    p.add_argument("--regime", default="single_thread")
    args = p.parse_args()

    rows = sl.load_timing_rows(args.bench_dir)
    res = analyze(rows, args.regime)
    tex_path = os.path.join(args.out_dir, "scaling_decomp_v2.tex")
    with open(tex_path, "w") as fh:
        fh.write(build_tex(res))
    print(f"wrote {tex_path}  ({len(res)} decomposed fits)")
    print("\n  family / method / lang : cost + count = total  (iter)")
    for r in sorted(res, key=lambda x: (x["phase"], x["family"], x["method"], x["language"])):
        print(f"    {r['phase']:6} {r['family']:26} {r['method']:11} {r['language'][:2]} "
              f": {r['beta_cost']:.3f} + {r['beta_count']:.3f} = {r['beta_sum']:.3f}  "
              f"(total={r['beta_total']:.3f}, iter={r['beta_iter']:.3f})")


if __name__ == "__main__":
    main()
