# Restarted DGMRES at scale (Python mirror of benchmark_restart.jl).
# Run: python code/python/benchmark_restart.py
import os
import time
import numpy as np
import scipy.sparse.linalg as spla
from drazin_krylov import sparse_block_singular, dgmres_restarted

SIZES = (10_000, 25_000, 50_000)
INDICES = (1, 2)
RESTART = 20
RES = os.path.join(os.path.dirname(__file__), "..", "results")


def drazin_solution_block(A, b, n_core):
    x = np.zeros_like(b)
    x[:n_core] = spla.spsolve(A[:n_core, :n_core].tocsc(), b[:n_core])
    return x


# warmup
_A, _, _ = sparse_block_singular(50, 2, seed=1)
_b = np.random.default_rng(1).standard_normal(52)
dgmres_restarted(_A, _b, index=2, restart=10, tol=1e-10)

hdr = (f'{"lang":<6}{"n":<8}{"k":<3}{"restart":<8}{"total_iters":<12}'
       f'{"cycles":<7}{"conv":<6}{"drazin_err":<13}{"time_s":<9}')
print(hdr)
with open(os.path.join(RES, "restart_python.csv"), "w") as f:
    f.write("lang,n,index,restart,total_iters,cycles,converged,drazin_error,time_s\n")
    for n_core in SIZES:
        for k in INDICES:
            A, _, _ = sparse_block_singular(n_core, k, seed=12345)
            n = n_core + k
            b = np.random.default_rng(999).standard_normal(n)
            xD = drazin_solution_block(A, b, n_core)
            t0 = time.perf_counter()
            res = dgmres_restarted(A, b, index=k, restart=RESTART, tol=1e-10)
            t = time.perf_counter() - t0
            derr = np.linalg.norm(res.x - xD) / max(np.linalg.norm(xD), 1e-30)
            cycles = len(res.residual_history)
            print(f'{"Python":<6}{n:<8}{k:<3}{RESTART:<8}{res.iterations:<12}'
                  f'{cycles:<7}{str(res.converged):<6}{derr:<13.3e}{t:<9.3f}')
            f.write(f"Python,{n},{k},{RESTART},{res.iterations},{cycles},"
                    f"{res.converged},{derr:.6e},{t:.6f}\n")
print("\nwrote code/results/restart_python.csv")
