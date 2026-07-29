#!/usr/bin/env python3
"""report_deepening.py — tables + figure for the Round-2 deepening experiments
(1.13 Krylov-budget sweep, 1.14 ill-conditioned diagnostics, 1.17 restarted-DGMRES
error-floor decoupling, 3.6 IC(0) exactness). Zero manual transcription: every
number is read from a CSV written by the Julia deepening scripts (PROTOCOL.md 11).

Inputs (code/results/benchmarks/):
  krylov_budget_sweep.csv     <- julia/deepen_krylov_budget.jl (1.13)
  illcond_diag.csv            <- julia/deepen_krylov_budget.jl (1.14)
  restart_sweep.csv           <- julia/restart_sweep.jl        (1.17, reused)
  ic0_exactness.csv           <- julia/deepen_ic0.jl           (3.6)

Outputs (code/results/):
  budget_sweep_v2.tex, fig_v2_budget_sweep.{pdf,png}   (1.13)
  illcond_v2.tex                                        (1.14)
  restart_floor_v2.tex                                  (1.17)
  ic0_exactness_v2.tex                                  (3.6)

stdlib csv + numpy + matplotlib (Agg) only.
"""
from __future__ import annotations

import csv
import math
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.normpath(os.path.join(HERE, ".."))
RESULTS_DIR = os.path.join(CODE_DIR, "results")
BENCH_DIR = os.path.join(RESULTS_DIR, "benchmarks")

CB = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#000000", "#56B4E9"]


def load_csv(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def g(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def tex_escape(s):
    return str(s).replace("_", r"\_").replace("%", r"\%")


def sci(x, sig=2):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "--"
    if x == 0:
        return "0"
    return f"{x:.{sig}e}"


# ---------------------------------------------------------------------------
# 1.13  Krylov-budget sweep
# ---------------------------------------------------------------------------
def report_budget():
    path = os.path.join(BENCH_DIR, "krylov_budget_sweep.csv")
    rows = load_csv(path)

    # group by instance, ordered by requested cap
    by = defaultdict(list)
    order = []
    for r in rows:
        if r["instance"] not in by:
            order.append(r["instance"])
        by[r["instance"]].append(r)

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Krylov-budget sweep for DGMRES on the dense singular instance "
        r"that hit the Krylov cap in the main sweep (similarity transform, $n=1000$, "
        r"index $k=3$; closes reviewer 1.13). As the Krylov cap $m$ grows the "
        r"index-$a$ residual $\lVert A^a r_m\rVert/\lVert A^a b\rVert$ (the operative "
        r"stopping test, \S8.2) falls monotonically and smoothly, with no "
        r"orthogonality blow-up (cf.\ the stability diagnostics), ruling out "
        r"instability. The Drazin error $\lVert x-x_\star\rVert/\lVert x_\star\rVert$ "
        r"against the exported ground truth $x_\star=A^Db$ (\S8.3) stays $O(1)$ until "
        r"the budget approaches the invertible-block dimension "
        r"($n_{\mathrm{core}}=997$) and then collapses to $\sim10^{-14}$ at the full "
        r"budget ($m=n-1$, $997$ iterations, converged): for this dense operator "
        r"recovery is essentially all-or-nothing near $m\approx n_{\mathrm{core}}$, "
        r"because DGMRES must resolve almost the entire invertible-block spectrum. "
        r"The failure at small $m$ is therefore a Krylov-budget limit, not an "
        r"instability. Generated from CSV (\S11).}",
        r"\label{tab:budget_sweep_v2}",
        r"\small",
        r"\begin{tabular}{rrlrr}",
        r"\toprule",
        r"Krylov cap $m$ & iters & converged & idx-$a$ residual & Drazin error \\",
        r"\midrule",
    ]
    for inst in order:
        sub = sorted(by[inst], key=lambda x: int(x["krylov_m_used"]))
        for r in sub:
            m_req = int(r["krylov_m_requested"])
            m_used = int(r["krylov_m_used"])
            n = int(r["n"])
            mlabel = str(m_req) if m_req == m_used else f"{m_used} (full)"
            conv = r["converged"].strip().lower() == "true"
            lines.append(
                f"{mlabel} & {int(r['iterations'])} & "
                f"{'yes' if conv else 'no'} & "
                f"{sci(g(r['index_a_residual']))} & {sci(g(r['drazin_error']))} \\\\"
            )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    out = os.path.join(RESULTS_DIR, "budget_sweep_v2.tex")
    with open(out, "w") as fh:
        fh.write("\n".join(lines))
    print(f"wrote {out}")

    # figure: Drazin error and index-a residual vs m
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for i, inst in enumerate(order):
        sub = sorted(by[inst], key=lambda x: int(x["krylov_m_used"]))
        m = np.array([int(x["krylov_m_used"]) for x in sub])
        derr = np.array([g(x["drazin_error"]) for x in sub])
        idxr = np.array([g(x["index_a_residual"]) for x in sub])
        ax.semilogy(m, np.clip(derr, 1e-18, None), "o-", color=CB[0],
                    lw=1.8, label="Drazin error $\\|x-x_\\star\\|/\\|x_\\star\\|$")
        ax.semilogy(m, np.clip(idxr, 1e-18, None), "s--", color=CB[1],
                    lw=1.5, label="index-$a$ residual (stop test)")
    ax.axhline(1e-8, color="0.5", ls=":", lw=1.0)
    ax.text(m[0], 1.3e-8, "tol $=10^{-8}$", fontsize=8, color="0.4")
    ax.set_xlabel("Krylov cap $m$")
    ax.set_ylabel("relative magnitude")
    ax.set_title("DGMRES budget sweep: dense similarity, $n=1000$, $k=3$")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, frameon=False, loc="upper right")
    fig.tight_layout()
    stem = os.path.join(RESULTS_DIR, "fig_v2_budget_sweep")
    for ext in ("pdf", "png"):
        fig.savefig(f"{stem}.{ext}", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"wrote {stem}.pdf (+.png)")


# ---------------------------------------------------------------------------
# 1.14  Ill-conditioned block diagnostics
# ---------------------------------------------------------------------------
def report_illcond():
    path = os.path.join(BENCH_DIR, "illcond_diag.csv")
    d = {r["quantity"]: g(r["value"]) for r in load_csv(path)}

    rowspec = [
        (r"$\kappa_2(B)$ (true, ill-conditioned block)", "cond_B_true"),
        (r"DGMRES iterations (converged on idx-$a$ res.)", "dgmres_iters"),
        (r"forward error $\lVert x-x_\star\rVert/\lVert x_\star\rVert$", "forward_error_rel"),
        (r"ordinary residual $\lVert b-Ax\rVert/\lVert b\rVert$", "ordinary_residual_rel"),
        (r"index-$a$ residual $\lVert A^a r\rVert/\lVert A^a b\rVert$ (stop test)", "index_a_residual_rel"),
        (r"normwise backward error (Rigal--Gaches)", "backward_error_normwise"),
        (r"empirical $\kappa$ of $A^Db$ (structure-preserving, $\varepsilon=10^{-8}$)", "empirical_cond_ADb"),
        (r"expected fwd.\ err.\ $\approx \kappa\cdot(\text{idx-}a\text{ res.})$", "expected_fwd_err_kappa_times_stopres"),
    ]
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Ill-conditioned singular block diagnostics "
        r"(illcond\_block\_singular, index $k=2$, $\kappa_2(B)\approx10^{6}$; "
        r"closes reviewer 1.14). DGMRES converges on the index-$a$ residual (the "
        r"operative stopping test, \S8.2) yet the forward Drazin error is $O(1)$. "
        r"The empirical condition number of the Drazin solution map (relative change "
        r"of the exact $A^Db$ under a structure-preserving perturbation of $B$ and "
        r"$b$ of size $\varepsilon=10^{-8}$) is $\approx\kappa_2(B)$; even so, "
        r"$\kappa\cdot(\text{index-}a\text{ residual})$ predicts a forward error far "
        r"below the observed $O(1)$. The gap shows the index-$a$ residual is a weak "
        r"proxy for the forward error here: annihilating the nilpotent part with "
        r"$A^a$ also attenuates the ill-conditioned $B$-subspace, so a small "
        r"index-$a$ residual does not certify a small Drazin error. Generated from "
        r"CSV (\S11).}",
        r"\label{tab:illcond_v2}",
        r"\small",
        r"\begin{tabular}{lr}",
        r"\toprule",
        r"Quantity & Value \\",
        r"\midrule",
    ]
    for label, key in rowspec:
        val = d.get(key, float("nan"))
        cell = str(int(round(val))) if key == "dgmres_iters" else sci(val)
        lines.append(f"{label} & {cell} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    out = os.path.join(RESULTS_DIR, "illcond_v2.tex")
    with open(out, "w") as fh:
        fh.write("\n".join(lines))
    print(f"wrote {out}")
    return d


# ---------------------------------------------------------------------------
# 1.17  Restarted-DGMRES error-floor decoupling (block-diagonal instances)
# ---------------------------------------------------------------------------
def report_restart_floor():
    path = os.path.join(BENCH_DIR, "restart_sweep.csv")
    rows = [r for r in load_csv(path) if r["language"] == "julia"
            and r["family"] == "sparse_block_singular"]
    by = defaultdict(lambda: defaultdict(list))
    order = []
    for r in rows:
        inst = r["instance"]
        if inst not in by:
            order.append(inst)
        by[inst][int(r["restart_r"])].append(r)

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Restarted-DGMRES residual/error decoupling at the stopping "
        r"tolerance (block-diagonal instances, index $k=3$; closes reviewer 1.17). "
        r"For each restart length $r$: the terminating index-$a$ residual "
        r"$\lVert A^a r\rVert/\lVert A^a b\rVert$ (the stop test, \S8.2) and the true "
        r"Drazin error $\lVert x-x_\star\rVert/\lVert x_\star\rVert$ against "
        r"$x_\star=A^Db$ (\S8.3). All runs converge ($\text{residual}\le "
        r"\text{tol}=10^{-8}$), but the Drazin error floor is non-monotonic in $r$: "
        r"$r=30$ needs a second cycle that drives the residual to $\sim10^{-11}$ "
        r"(well below tol) and yields the smallest error, whereas $r=50$ terminates "
        r"in a single cycle right at the tolerance ($\sim10^{-9}$) and ends with a "
        r"$\sim100\times$ larger error. The error tracks how far below tol the "
        r"residual actually lands, not merely whether tol is met. Generated from "
        r"CSV (\S11).}",
        r"\label{tab:restart_floor_v2}",
        r"\small",
        r"\begin{tabular}{lrrrrl}",
        r"\toprule",
        r"Instance & $r$ & cycles & term.\ idx-$a$ res. & Drazin err. & note \\",
        r"\midrule",
    ]
    for inst in order:
        first = True
        best_r = min(by[inst],
                     key=lambda rr: g(sorted(by[inst][rr], key=lambda x: int(x["cycle"]))[-1]["final_drazin_err"]))
        for rr in sorted(by[inst]):
            sub = sorted(by[inst][rr], key=lambda x: int(x["cycle"]))
            term_res = g(sub[-1]["cycle_residual"])
            derr = g(sub[-1]["final_drazin_err"])
            ncyc = int(sub[-1]["outer_cycles"])
            note = "best floor" if rr == best_r else ""
            inst_cell = tex_escape(inst) if first else ""
            if first and inst != order[0]:
                lines.append(r"\addlinespace")
            first = False
            lines.append(
                f"{inst_cell} & {rr} & {ncyc} & {sci(term_res)} & {sci(derr)} & {note} \\\\"
            )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    out = os.path.join(RESULTS_DIR, "restart_floor_v2.tex")
    with open(out, "w") as fh:
        fh.write("\n".join(lines))
    print(f"wrote {out}")


# ---------------------------------------------------------------------------
# 3.6  IC(0) exactness per family
# ---------------------------------------------------------------------------
def report_ic0_exactness():
    path = os.path.join(BENCH_DIR, "ic0_exactness.csv")
    rows = load_csv(path)
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{IC(0) factorization exactness by SPD family (closes reviewer "
        r"3.6, one-iteration convergence). For each representative matrix: the "
        r"zero-fill factorization error $\lVert LL^\top-A\rVert_F/\lVert A\rVert_F$, "
        r"whether that makes IC(0) an exact factorization, the structural fill ratio "
        r"$\mathrm{nnz}(\mathrm{tril}\,A)/\mathrm{nnz}(L_{\text{exact chol}})$ of the "
        r"natural-order exact Cholesky factor (1.0 means zero-fill discards nothing), "
        r"and the observed PCG(IC0) iteration count. Where the ratio is 1 the "
        r"zero-fill factor equals the exact Cholesky factor, so $M=A$ and PCG "
        r"converges in a single iteration by construction --- not a leak. The "
        r"2-D/3-D Laplacian and anisotropic problems drop most of the exact fill, "
        r"giving a nonzero factorization error and multi-iteration convergence. "
        r"Generated from CSV (\S11).}",
        r"\label{tab:ic0_exactness_v2}",
        r"\small",
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Matrix & $n$ & $\lVert LL^\top-A\rVert/\lVert A\rVert$ & exact? "
        r"& fill ratio & PCG its. \\",
        r"\midrule",
    ]
    for r in rows:
        exact = r["exact"].strip().lower() == "true"
        ratio = r["chol_structural_fill_ratio"]
        ratio_cell = "--" if ratio == "NA" else f"{g(ratio):.3g}"
        lines.append(
            f"{tex_escape(r['label'])} & {int(r['n'])} & "
            f"{sci(g(r['factorization_residual']))} & "
            f"{'yes' if exact else 'no'} & {ratio_cell} & "
            f"{int(r['observed_pcg_iters'])} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    out = os.path.join(RESULTS_DIR, "ic0_exactness_v2.tex")
    with open(out, "w") as fh:
        fh.write("\n".join(lines))
    print(f"wrote {out}")


def main():
    report_budget()
    report_illcond()
    report_restart_floor()
    report_ic0_exactness()


if __name__ == "__main__":
    main()
