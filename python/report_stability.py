#!/usr/bin/env python3
"""report_stability.py — table + figure for DGMRES stability diagnostics
(PROTOCOL.md 8.4, reviewer 3.11).

Reads code/results/benchmarks/stability_dgmres.csv (written by
code/julia/stability_dgmres.jl) and generates, with zero manual transcription
(PROTOCOL.md 11):

  * code/results/stability_v2.tex          booktabs table: per (instance,
    precision) the max/final orthogonality drift delta_m, the max Arnoldi-relation
    residual, the max least-squares condition number, the final index-a residual,
    the final Drazin error vs exported x_star, and convergence.
  * code/results/fig_v2_stability.{pdf,png}  two panels vs Arnoldi step m:
    (left) orthogonality drift delta_m, (right) least-squares condition number;
    Float64 solid, Float32 dashed, one colour per instance.

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
CSV_PATH = os.path.join(BENCH_DIR, "stability_dgmres.csv")

# colorblind-friendly (Wong) palette, matching plot_v2.py
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
    return str(s).replace("_", r"\_")


def sci(x, sig=2):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "--"
    if x == 0:
        return "0"
    return f"\\num{{{x:.{sig}e}}}" if False else f"{x:.{sig}e}"


def build_table(rows):
    # group per (instance, precision) preserving instance order of appearance
    order = []
    groups = defaultdict(list)
    for r in rows:
        key = (r["instance"], r["precision"])
        if key not in groups:
            order.append(key)
        groups[key].append(r)

    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{DGMRES numerical-stability diagnostics (protocol in "
                 r"Appendix~\ref{app:protocol}; closes reviewer 3.11). Per representative singular "
                 r"instance and working precision: peak/final orthogonality drift "
                 r"$\delta_m=\lVert V_m^\top V_m-I\rVert_2$, peak Arnoldi-relation "
                 r"residual $\lVert AV_m-V_{m+1}\bar H_m\rVert_2$, peak "
                 r"least-squares condition number $\kappa_2$ of the step-$m$ "
                 r"objective operator, the final index-$a$ residual "
                 r"$\lVert A^a r\rVert/\lVert A^a b\rVert$ (the stopping test, "
                 r"\S8.2), and the final Drazin error "
                 r"$\lVert x-x_\star\rVert/\lVert x_\star\rVert$ against the "
                 r"exported ground truth $x_\star=A^Db$ (\S8.3). The DGMRES "
                 r"algorithm is byte-identical across Julia and Python (work-count "
                 r"identity already established), so these floating-point "
                 r"diagnostics are computed once, in Julia. Generated from CSV "
                 r"(\S11).}")
    lines.append(r"\label{tab:stability_v2}")
    lines.append(r"\scriptsize\setlength{\tabcolsep}{3pt}")
    lines.append(r"\begin{tabular}{llrrrrrrl}")
    lines.append(r"\toprule")
    lines.append(r"Instance & Prec. & $n$ & $\delta_m^{\max}$ & $\delta_m^{\text{fin}}$ "
                 r"& Arnoldi$^{\max}$ & $\kappa_2^{\max}$ & idx-$a$ res. "
                 r"& Drazin err. \\")
    lines.append(r"\midrule")

    last_inst = None
    for (inst, prec) in order:
        sub = groups[(inst, prec)]
        ms = [int(x["m"]) for x in sub]
        delta = [g(x["delta_m"]) for x in sub]
        arn = [g(x["arnoldi_resid"]) for x in sub]
        lscond = [g(x["ls_cond"]) for x in sub]
        lscond_f = [c for c in lscond if math.isfinite(c)]
        # rows share the final columns
        n = int(sub[0]["n"])
        derr = g(sub[0]["final_drazin_err"])
        fin_res = g(sub[0]["final_res_rel"])
        conv = sub[0]["converged"].strip().lower() == "true"
        delta_max = max(delta) if delta else float("nan")
        delta_fin = delta[-1] if delta else float("nan")
        arn_max = max(arn) if arn else float("nan")
        ls_max = max(lscond_f) if lscond_f else float("nan")

        inst_cell = tex_escape(inst) if inst != last_inst else ""
        if inst != last_inst and last_inst is not None:
            lines.append(r"\addlinespace")
        last_inst = inst
        conv_cell = "(conv.)" if conv else "(no conv.)"
        lines.append(
            f"{inst_cell} & {prec.replace('float', 'f')} & {n} & "
            f"{sci(delta_max)} & {sci(delta_fin)} & {sci(arn_max)} & "
            f"{sci(ls_max)} & {sci(fin_res)} & {sci(derr)}~{conv_cell} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    lines.append("")
    return "\n".join(lines)


def build_figure(rows):
    # per instance: m-series for float64 (solid) and float32 (dashed, if present)
    by = defaultdict(lambda: defaultdict(list))  # instance -> precision -> rows
    inst_order = []
    for r in rows:
        if r["instance"] not in by:
            inst_order.append(r["instance"])
        by[r["instance"]][r["precision"]].append(r)

    color = {inst: CB[i % len(CB)] for i, inst in enumerate(inst_order)}
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12.5, 5.0))

    for inst in inst_order:
        for prec, rws in by[inst].items():
            rws = sorted(rws, key=lambda x: int(x["m"]))
            m = np.array([int(x["m"]) for x in rws])
            delta = np.array([g(x["delta_m"]) for x in rws])
            ls = np.array([g(x["ls_cond"]) for x in rws])
            style = "-" if prec == "float64" else "--"
            lw = 1.8 if prec == "float64" else 1.2
            lbl = inst if prec == "float64" else None
            axL.semilogy(m, np.clip(delta, 1e-18, None), style, color=color[inst],
                         lw=lw, label=lbl)
            axR.semilogy(m, np.clip(ls, 1e-18, None), style, color=color[inst],
                         lw=lw, label=lbl)

    axL.set_xlabel("Arnoldi step $m$")
    axL.set_ylabel(r"orthogonality drift $\delta_m=\|V_m^\top V_m-I\|_2$")
    axL.set_title("(a) Loss of orthogonality")
    axL.axhline(1.0, color="0.6", ls=":", lw=1.0)
    axL.grid(True, which="both", alpha=0.3)

    axR.set_xlabel("Arnoldi step $m$")
    axR.set_ylabel(r"least-squares condition number $\kappa_2$")
    axR.set_title("(b) Conditioning of the step-$m$ objective")
    axR.grid(True, which="both", alpha=0.3)

    handles, labels = axL.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8,
               frameon=False, bbox_to_anchor=(0.5, -0.08))
    fig.text(0.5, 0.005, "solid = Float64, dashed = Float32", ha="center",
             fontsize=8, style="italic")
    fig.tight_layout(rect=(0, 0.05, 1, 1))

    stem = os.path.join(RESULTS_DIR, "fig_v2_stability")
    for ext in ("pdf", "png"):
        fig.savefig(f"{stem}.{ext}", bbox_inches="tight", dpi=150)
    plt.close(fig)
    return stem + ".pdf"


def main():
    rows = load_csv(CSV_PATH)
    tex = build_table(rows)
    tex_path = os.path.join(RESULTS_DIR, "stability_v2.tex")
    with open(tex_path, "w") as fh:
        fh.write(tex)
    print(f"wrote {tex_path}")
    fig_path = build_figure(rows)
    print(f"wrote {fig_path} (+.png)")


if __name__ == "__main__":
    main()
