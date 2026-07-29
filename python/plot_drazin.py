"""Figures for the Drazin experiment. Run: python code/python/plot_drazin.py
Produces (PDF for LaTeX + PNG for preview) in code/results/:
  fig_drazin_error_vs_iter   error trajectory: GMRES plateaus, DGMRES -> A^D b
  fig_drazin_error_vs_n      final Drazin error across sizes (the systematic gap)
  fig_drazin_time_vs_n       wall-clock GMRES vs DGMRES across sizes
"""
import csv
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from drazin_krylov import sparse_block_singular, dgmres

RES = os.path.join(os.path.dirname(__file__), "..", "results")

# Okabe-Ito colorblind-safe pair (validated: CVD ΔE 91.9). Secondary encoding via
# distinct linestyle + marker so the two series survive greyscale printing.
C_GMRES, C_DGMRES = "#D55E00", "#0072B2"
INK, MUTED, GRID = "#1a1a1a", "#666666", "#dddddd"

plt.rcParams.update({
    "font.size": 11, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "figure.dpi": 120,
})


def drazin_solution_block(A, b, n_core):
    import scipy.sparse.linalg as spla
    x = np.zeros_like(b)
    x[:n_core] = spla.spsolve(A[:n_core, :n_core].tocsc(), b[:n_core])
    return x


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(RES, name + ".pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(RES, name + ".png"), bbox_inches="tight", dpi=150)
    plt.close(fig)


# ---- Figure 1: error vs iteration (representative case) --------------------
def fig_error_vs_iter():
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    for ax, k in zip(axes, (1, 3)):
        A, _, _ = sparse_block_singular(2000, k, seed=12345)
        n = 2000 + k
        b = np.random.default_rng(999).standard_normal(n)
        xD = drazin_solution_block(A, b, 2000)
        g = dgmres(A, b, index=0, m=min(n, 400), tol=1e-10, xtrue=xD)
        d = dgmres(A, b, index=k, m=min(n, 400), tol=1e-10, xtrue=xD)
        ax.semilogy(range(1, len(g.error_history) + 1), g.error_history,
                    color=C_GMRES, ls="--", lw=2, label="GMRES (classical)")
        ax.semilogy(range(1, len(d.error_history) + 1), d.error_history,
                    color=C_DGMRES, ls="-", lw=2, marker="o", markevery=8,
                    ms=5, label="DGMRES")
        ax.set_title(f"Drazin index $k={k}$", color=INK)
        ax.set_xlabel("iteration")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel(r"relative error $\|x_m - A^{D}b\| / \|A^{D}b\|$")
    axes[0].legend(frameon=False, loc="upper right")
    fig.suptitle("GMRES stalls far from the Drazin solution; DGMRES converges to it",
                 y=1.04, fontsize=12)
    save(fig, "fig_drazin_error_vs_iter")


# ---- read benchmark CSVs ---------------------------------------------------
def load(lang):
    rows = []
    with open(os.path.join(RES, f"drazin_{lang}.csv")) as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def series(rows, method, k, field):
    xs, ys = [], []
    for r in rows:
        if r["method"] == method and int(r["index"]) == k:
            xs.append(int(r["n"]))
            ys.append(float(r[field]))
    order = np.argsort(xs)
    return np.array(xs)[order], np.array(ys)[order]


# ---- Figure 2: final Drazin error across n (k=1) ---------------------------
def fig_error_vs_n():
    jl = load("julia")
    fig, ax = plt.subplots(figsize=(6, 3.8))
    for method, color, ls, mk in (("GMRES", C_GMRES, "--", "s"),
                                  ("DGMRES", C_DGMRES, "-", "o")):
        x, y = series(jl, method, 1, "drazin_error")
        ax.semilogy(x, y, color=color, ls=ls, lw=2, marker=mk, ms=7,
                    label=f"{method}")
    ax.axhline(1e-9, color=MUTED, lw=1, ls=":")
    ax.text(x[0], 1.4e-9, "tolerance", color=MUTED, fontsize=9, va="bottom")
    ax.set_xlabel("system dimension $n$")
    ax.set_ylabel(r"final $\|x - A^{D}b\| / \|A^{D}b\|$")
    ax.set_title("Correctness gap is systematic across sizes (Julia, $k=1$)", color=INK)
    ax.legend(frameon=False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    save(fig, "fig_drazin_error_vs_n")


# ---- Figure 3: wall-clock vs n ---------------------------------------------
def fig_time_vs_n():
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    for ax, lang in zip(axes, ("julia", "python")):
        rows = load(lang)
        for method, color, ls, mk in (("GMRES", C_GMRES, "--", "s"),
                                      ("DGMRES", C_DGMRES, "-", "o")):
            x, y = series(rows, method, 1, "time_s")
            ax.semilogy(x, y, color=color, ls=ls, lw=2, marker=mk, ms=7, label=method)
        ax.set_title(lang.capitalize(), color=INK)
        ax.set_xlabel("system dimension $n$")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("wall-clock time (s)")
    axes[0].legend(frameon=False)
    fig.suptitle("DGMRES converges (and so returns) far faster than stalled GMRES ($k=1$)",
                 y=1.04, fontsize=12)
    save(fig, "fig_drazin_time_vs_n")


if __name__ == "__main__":
    fig_error_vs_iter()
    fig_error_vs_n()
    fig_time_vs_n()
    print("wrote figures to code/results/ (*.pdf and *.png)")
