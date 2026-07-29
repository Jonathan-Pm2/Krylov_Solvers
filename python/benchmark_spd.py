# Experiment B (Python): does CG on the 1-D Laplacian really converge in ~59
# iterations, or does it genuinely need O(n)? Matched stopping criterion.
# Run: python code/python/benchmark_spd.py
import os
import time
import numpy as np
from cg_ref import laplacian_1d, rhs_deterministic, cg_ref, scipy_cg

SIZES = (1000, 2000, 5000)
RTOL = 1e-8
RES = os.path.join(os.path.dirname(__file__), "..", "results")

# warmup
_A = laplacian_1d(50); _b = rhs_deterministic(50)
cg_ref(_A, _b); cg_ref(_A, _b, jacobi=True); scipy_cg(_A, _b)

hdr = f'{"lang":<7}{"n":<7}{"kappa":>10}{"solver":<14}{"iters":>7}{"conv":>6}{"relres":>11}{"time_s":>9}'
print(hdr)
with open(os.path.join(RES, "spd_python.csv"), "w") as f:
    f.write("lang,n,kappa,solver,iters,converged,relres,time_s\n")
    for n in SIZES:
        A = laplacian_1d(n)
        b = rhs_deterministic(n)
        kappa = (2 - 2 * np.cos(n * np.pi / (n + 1))) / (2 - 2 * np.cos(np.pi / (n + 1)))
        runs = [
            ("CG", lambda: cg_ref(A, b, rtol=RTOL, maxiter=n)),
            ("CG-Jacobi", lambda: cg_ref(A, b, rtol=RTOL, maxiter=n, jacobi=True)),
            ("scipy.cg", lambda: scipy_cg(A, b, rtol=RTOL, maxiter=n)),
        ]
        for name, fn in runs:
            t0 = time.perf_counter()
            out = fn()
            t = time.perf_counter() - t0
            iters, conv, relres = out[1], out[2], out[3]
            print(f'{"Python":<7}{n:<7}{kappa:>10.2e}{name:<14}{iters:>7}'
                  f'{str(conv):>6}{relres:>11.2e}{t:>9.3f}')
            f.write(f"Python,{n},{kappa:.4e},{name},{iters},{conv},{relres:.4e},{t:.6f}\n")
print("\nwrote code/results/spd_python.csv")
