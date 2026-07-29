# Headline experiment (Python mirror of benchmark_drazin.jl).
# GMRES (dgmres index=0) converges to the WRONG solution on non-symmetric
# singular systems; DGMRES (index=k) recovers the exact Drazin solution A^D b.
# Run:  python code/python/benchmark_drazin.py
import os
import time
import numpy as np
import scipy.sparse.linalg as spla
from drazin_krylov import sparse_block_singular, dgmres

SIZES = (1000, 2000, 4000)
INDICES = (1, 2, 3)
CAP_M = 400
RESULTS = os.path.join(os.path.dirname(__file__), "..", "results", "drazin_python.csv")


def drazin_solution_block(A, b, n_core):
    """Closed-form A^D b for sparse_block_singular: x[core]=B^-1 b[core], x[nil]=0."""
    x = np.zeros_like(b)
    B = A[:n_core, :n_core].tocsc()
    x[:n_core] = spla.spsolve(B, b[:n_core])
    return x


def run_case(n_core, k):
    A, _, _ = sparse_block_singular(n_core, k, seed=12345)
    n = n_core + k
    b = np.random.default_rng(999).standard_normal(n)
    xD = drazin_solution_block(A, b, n_core)
    nrmb = np.linalg.norm(b)
    nrmxD = max(np.linalg.norm(xD), 1e-30)
    out = {}
    for name, idx in (("GMRES", 0), ("DGMRES", k)):
        t0 = time.perf_counter()
        res = dgmres(A, b, index=idx, m=min(n, CAP_M), tol=1e-10)
        t = time.perf_counter() - t0
        x = res.x
        rr = np.linalg.norm(A @ x - b) / nrmb
        de = np.linalg.norm(x - xD) / nrmxD
        out[name] = dict(t=t, iters=res.iterations, conv=res.converged, res=rr, derr=de)
    return out


# JIT/import warmup so timing isn't polluted by first-call overhead
_A, _, _ = sparse_block_singular(20, 2, seed=1)
_b = np.random.default_rng(1).standard_normal(22)
dgmres(_A, _b, index=0, m=22, tol=1e-10)
dgmres(_A, _b, index=2, m=22, tol=1e-10)

hdr = f'{"lang":<7}{"n":<7}{"k":<4}{"method":<8}{"time_s":>10}{"iters":>7}{"conv":>6}{"rel_res":>12}{"drazin_err":>13}'
print(hdr)
with open(RESULTS, "w") as f:
    f.write("lang,n,index,method,time_s,iters,converged,rel_residual,drazin_error\n")
    for n_core in SIZES:
        for k in INDICES:
            r = run_case(n_core, k)
            for method in ("GMRES", "DGMRES"):
                m = r[method]
                print(f'{"Python":<7}{n_core + k:<7}{k:<4}{method:<8}{m["t"]:>10.4f}'
                      f'{m["iters"]:>7}{str(m["conv"]):>6}{m["res"]:>12.3e}{m["derr"]:>13.3e}')
                f.write(f'Python,{n_core + k},{k},{method},{m["t"]:.6f},{m["iters"]},'
                        f'{m["conv"]},{m["res"]:.6e},{m["derr"]:.6e}\n')
print("\nwrote code/results/drazin_python.csv")
