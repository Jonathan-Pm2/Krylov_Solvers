#!/usr/bin/env python3
"""report_restart_sweep.py — table + figure for restarted-DGMRES sensitivity to
the restart length r (PROTOCOL.md 8.2, reviewer 3.12).

Reads code/results/benchmarks/restart_sweep.csv (written by
code/julia/restart_sweep.jl, with a Python timing point appended by
code/python/restart_sweep_py.py) and generates, with zero manual transcription
(PROTOCOL.md 11):

  * code/results/restart_sweep_v2.tex        longtable: per (instance, language,
    r) the outer cycles, total inner iterations, total solve time, bounded
    Arnoldi-basis memory r*n*8, convergence, final Drazin error, and an explicit
    stagnation flag (residual not decreasing across cycles).
  * code/results/fig_v2_restart_percycle.{pdf,png}  per-instance panels of the
    per-cycle index-a residual trajectory for each r, so convergence vs
    stagnation is visible; language-independent, drawn from the Julia rows.

longtable is used because the julia+python cross product can exceed one page.
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
CSV_PATH = os.path.join(BENCH_DIR, "restart_sweep.csv")

TOL = 1e-8
CB = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7"]


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
    return f"{x:.{sig}e}"


def build_table(rows):
    # one summary row per (instance, language, r) — the per-cycle rows share the
    # run-level fields, so any one of them carries the summary.
    summary = {}
    inst_order, r_order = [], []
    for r in rows:
        key = (r["instance"], r["language"], int(r["restart_r"]))
        if key not in summary:
            summary[key] = r
        if r["instance"] not in inst_order:
            inst_order.append(r["instance"])
        if int(r["restart_r"]) not in r_order:
            r_order.append(int(r["restart_r"]))
    r_order = sorted(r_order)

    lines = []
    lines.append(r"\footnotesize")
    lines.append(r"\setlength{\tabcolsep}{4pt}")
    lines.append(r"\begin{longtable}{llrrrrrll}")
    lines.append(r"\caption{Restarted DGMRES sensitivity to the restart length "
                 r"$r$ (protocol in Appendix~\ref{app:protocol}; closes reviewer 3.12). Per instance, "
                 r"language, and $r$: outer restart cycles, total inner Arnoldi "
                 r"iterations, median total solve time, the bounded Arnoldi-basis "
                 r"memory $r\cdot n\cdot 8$ bytes (the quantity restarting caps, "
                 r"independent of cycle count), convergence on the index-$a$ "
                 r"residual, the final Drazin error vs $x_\star=A^Db$, and an "
                 r"explicit stagnation flag (index-$a$ residual flat across "
                 r"cycles). The per-cycle trajectory is language-independent by "
                 r"the work-count identity; a Python timing point is included for "
                 r"one instance. Generated from CSV (\S11).}"
                 r"\label{tab:restart_sweep_v2}\\")
    lines.append(r"\toprule")
    header = (r"Instance & Lang. & $r$ & cycles & inner it. & time [ms] & "
              r"basis [KB] & conv. & Drazin err. \\")
    lines.append(header)
    lines.append(r"\midrule")
    lines.append(r"\endfirsthead")
    lines.append(r"\toprule")
    lines.append(header)
    lines.append(r"\midrule")
    lines.append(r"\endhead")
    lines.append(r"\bottomrule")
    lines.append(r"\endfoot")

    last_inst = None
    for inst in inst_order:
        for lang in ("julia", "python"):
            for r in r_order:
                key = (inst, lang, r)
                if key not in summary:
                    continue
                s = summary[key]
                cycles = int(s["outer_cycles"])
                inner = int(s["total_inner_iters"])
                tms = g(s["total_time_s"]) * 1e3
                basis_kb = g(s["basis_bytes"]) / 1024.0
                conv = s["converged"].strip().lower() == "true"
                derr = g(s["final_drazin_err"])
                stag = s["stagnation"].strip().lower() == "true"
                inst_cell = tex_escape(inst) if inst != last_inst else ""
                if inst != last_inst and last_inst is not None:
                    lines.append(r"\addlinespace")
                last_inst = inst
                conv_cell = "yes" if conv else ("STAG." if stag else "no")
                derr_cell = sci(derr)
                lines.append(
                    f"{inst_cell} & {lang[:2]} & {r} & {cycles} & {inner} & "
                    f"{tms:.3g} & {basis_kb:.1f} & {conv_cell} & {derr_cell} \\\\"
                )
    lines.append(r"\end{longtable}")
    lines.append("")
    return "\n".join(lines)


def build_figure(rows):
    # per-cycle residual trajectories, one panel per instance, one line per r,
    # Julia rows only (language-independent).
    jl = [r for r in rows if r["language"] == "julia"]
    inst_order = []
    traj = defaultdict(lambda: defaultdict(list))  # instance -> r -> [(cycle,res)]
    conv_map = {}
    for r in jl:
        inst = r["instance"]
        if inst not in inst_order:
            inst_order.append(inst)
        rr = int(r["restart_r"])
        traj[inst][rr].append((int(r["cycle"]), g(r["cycle_residual"])))
        conv_map[(inst, rr)] = r["converged"].strip().lower() == "true"

    ninst = len(inst_order)
    ncol = 3
    nrow = math.ceil(ninst / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.2 * ncol, 4.0 * nrow),
                             squeeze=False)
    r_colors = {}
    all_r = sorted({int(r["restart_r"]) for r in jl})
    for i, rr in enumerate(all_r):
        r_colors[rr] = CB[i % len(CB)]

    for idx, inst in enumerate(inst_order):
        ax = axes[idx // ncol][idx % ncol]
        for rr in sorted(traj[inst].keys()):
            pts = sorted(traj[inst][rr])
            xs = np.array([p[0] for p in pts])
            ys = np.clip(np.array([p[1] for p in pts]), 1e-18, None)
            # cap the flat tail for readability but keep enough to show the plateau
            if len(xs) > 60:
                xs, ys = xs[:60], ys[:60]
            conv = conv_map.get((inst, rr), False)
            style = "-" if conv else "--"
            ax.semilogy(xs, ys, style, color=r_colors[rr], lw=1.6,
                        marker="o", ms=3,
                        label=f"r={rr} ({'conv' if conv else 'stag'})")
        ax.axhline(TOL, color="0.5", ls=":", lw=1.0)
        ax.set_title(inst, fontsize=10)
        ax.set_xlabel("restart cycle")
        ax.set_ylabel(r"$\|A^a(b-Ax)\|/\|A^a b\|$")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=7, frameon=False)

    # blank any unused panels
    for j in range(ninst, nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")

    fig.suptitle("Per-cycle index-$a$ residual vs restart length $r$ "
                 "(solid = converged, dashed = stagnated; tail capped at 60 cycles)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    stem = os.path.join(RESULTS_DIR, "fig_v2_restart_percycle")
    for ext in ("pdf", "png"):
        fig.savefig(f"{stem}.{ext}", bbox_inches="tight", dpi=150)
    plt.close(fig)
    return stem + ".pdf"


def main():
    rows = load_csv(CSV_PATH)
    tex = build_table(rows)
    tex_path = os.path.join(RESULTS_DIR, "restart_sweep_v2.tex")
    with open(tex_path, "w") as fh:
        fh.write(tex)
    print(f"wrote {tex_path}")
    fig_path = build_figure(rows)
    print(f"wrote {fig_path} (+.png)")


if __name__ == "__main__":
    main()
