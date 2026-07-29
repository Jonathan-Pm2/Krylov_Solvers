#!/usr/bin/env python3
"""restart_sweep_py.py — Python cross-language timing point for the restart sweep
(PROTOCOL.md 8.2 / 3.12).

The full restart sweep (per-cycle residual trajectories, memory, stagnation) is
produced in Julia by code/julia/restart_sweep.jl. The per-cycle residual
behaviour is language-independent (identical algorithm, work-count identity
already proven), so this script only adds a Python TIMING point for ONE
representative converging instance across r in {10,20,30,50}, satisfying the
"timing in both languages for at least one instance" requirement of 3.12.

It APPENDS language=python rows to results/benchmarks/restart_sweep.csv,
mirroring the Julia column schema. The script is idempotent: it keeps the
existing language=julia rows, drops any prior python rows, and rewrites.

Run AFTER restart_sweep.jl. stdlib csv + numpy + scipy only.
"""
from __future__ import annotations

import csv
import os
import resource
import time

import numpy as np
import scipy.sparse.linalg as spla

import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.normpath(os.path.join(HERE, ".."))
DATA_DIR = os.path.join(CODE_DIR, "data")
BENCH_DIR = os.path.join(CODE_DIR, "results", "benchmarks")
CSV_PATH = os.path.join(BENCH_DIR, "restart_sweep.csv")

sys.path.insert(0, HERE)
from data_layer import read_mtx, read_npy  # noqa: E402
from drazin_krylov import dgmres_restarted  # noqa: E402

RESTARTS = [10, 20, 30, 50]
TOL = 1e-8
MAXOUTER = 500
REPS = 3

# One representative converging block-diagonal instance (matches a Julia row).
INSTANCE = {
    "label": "block-diag n=503 k=3",
    "family": "sparse_block_singular",
    "dir": "sparse_block_singular/n00000503",
    "k": 3,
    "n_core": 500,
    "seed": 0,
}


def block_x_star(A, b, n_core):
    """Closed-form Drazin solution for A = diag(B, N): x[core] = B^{-1} b[core]."""
    n = A.shape[0]
    B = A[:n_core, :n_core].tocsc()
    x = np.zeros(n)
    x[:n_core] = spla.spsolve(B, np.asarray(b[:n_core]).ravel())
    return x


def main():
    A = read_mtx(os.path.join(DATA_DIR, INSTANCE["dir"], "A.mtx"))
    b = read_npy(os.path.join(DATA_DIR, INSTANCE["dir"], "b.npy"))
    n = A.shape[0]
    xstar = block_x_star(A, b, INSTANCE["n_core"])
    nrm_xstar = float(np.linalg.norm(xstar))

    new_rows = []
    for r in RESTARTS:
        # warmup (page/cache warm-up), then median of REPS timed runs
        res = dgmres_restarted(A, b, index=INSTANCE["k"], restart=r,
                               maxouter=MAXOUTER, tol=TOL)
        times = []
        for _ in range(REPS):
            t0 = time.perf_counter_ns()
            dgmres_restarted(A, b, index=INSTANCE["k"], restart=r,
                             maxouter=MAXOUTER, tol=TOL)
            times.append((time.perf_counter_ns() - t0) / 1e9)
        tmed = float(np.median(times))
        vmhwm = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024  # KB->bytes (Linux)

        reshist = res.residual_history
        ncyc = len(reshist)
        derr = (np.linalg.norm(res.x) if nrm_xstar == 0
                else float(np.linalg.norm(res.x - xstar) / nrm_xstar))
        basis_bytes = r * n * 8
        tail = max(0, ncyc - min(10, ncyc))
        stagnation = (not res.converged) and ncyc >= 2 and (reshist[-1] >= 0.99 * reshist[tail])

        for ci, g in enumerate(reshist, start=1):
            new_rows.append([
                INSTANCE["label"], INSTANCE["family"], str(n), str(INSTANCE["k"]),
                "python", str(r), str(ci), f"{g:.8g}", str(ncyc),
                str(res.iterations), f"{tmed:.6g}", str(basis_bytes),
                "", str(vmhwm), str(res.converged).lower(), f"{derr:.8g}",
                str(stagnation).lower(), str(INSTANCE["seed"]),
            ])
        print(f"  {INSTANCE['label']:<28} [python] r={r:2d} cycles={ncyc:3d} "
              f"inner={res.iterations:4d} t={tmed:.4g}s derr={derr:.2e} "
              f"conv={res.converged} stag={stagnation}")

    # idempotent rewrite: keep header + existing julia rows, replace python rows
    with open(CSV_PATH, newline="") as fh:
        reader = list(csv.reader(fh))
    header, body = reader[0], reader[1:]
    lang_col = header.index("language")
    kept = [row for row in body if row[lang_col] != "python"]
    with open(CSV_PATH, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(kept)
        w.writerows(new_rows)
    print(f"appended {len(new_rows)} python rows to {CSV_PATH}")


if __name__ == "__main__":
    main()
