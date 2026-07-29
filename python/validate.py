# Python-side validation mirror. Run: python code/python/validate.py
import numpy as np
import scipy.sparse as sp
from drazin_krylov import (drazin_index, drazin_inverse, drazin_solution,
                           block_nilpotent, graph_laplacian_singular, dgmres)

npass = 0
nfail = 0


def check(name, cond):
    global npass, nfail
    if cond:
        npass += 1
        print(f"  PASS  {name}")
    else:
        nfail += 1
        print(f"  FAIL  {name}")


print("== 1. Drazin inverse axioms (block_nilpotent, index k) ==")
for k in (1, 2, 3):
    A = block_nilpotent(6, k, seed=100 + k)
    kdet = drazin_index(A)
    Ad = drazin_inverse(A)
    e1 = np.linalg.norm(np.linalg.matrix_power(A, k + 1) @ Ad
                        - np.linalg.matrix_power(A, k)) / np.linalg.norm(np.linalg.matrix_power(A, k))
    e2 = np.linalg.norm(Ad @ A @ Ad - Ad) / np.linalg.norm(Ad)
    e3 = np.linalg.norm(A @ Ad - Ad @ A)
    check(f"index detected == {k} (got {kdet})", kdet == k)
    check(f"A^(k+1) A^D = A^k   (rel err {e1:.2e})", e1 < 1e-8)
    check(f"A^D A A^D = A^D     (rel err {e2:.2e})", e2 < 1e-8)
    check(f"A A^D = A^D A       (abs err {e3:.2e})", e3 < 1e-8)

print("\n== 2. Singular graph Laplacian (Neumann) has index 1 ==")
for n in (5, 10, 25):
    L = graph_laplacian_singular(n).toarray()
    check(f"ind(L_{n}) == 1", drazin_index(L) == 1)
    check(f"L_{n} is singular (min sv ~ 0)", np.linalg.svd(L, compute_uv=False).min() < 1e-10)

print("\n== 3. DGMRES converges to the Drazin solution A^D b ==")
for k in (1, 2, 3):
    A = block_nilpotent(8, k, seed=200 + k)
    n = A.shape[0]
    rng = np.random.default_rng(7)
    b = A @ drazin_solution(A, A @ rng.standard_normal(n))  # consistent RHS in range(A^k)
    xD = drazin_solution(A, b)
    res = dgmres(A, b, m=n + k + 2, tol=1e-10)
    err = np.linalg.norm(res.x - xD) / max(np.linalg.norm(xD), 1e-30)
    check(f"DGMRES == A^D b, index {k} (rel err {err:.2e}, iters {res.iterations}, conv {res.converged})",
          err < 1e-6)

print("\n== 4. DGMRES on singular Laplacian vs ground truth ==")
for n in (20, 50):
    L = graph_laplacian_singular(n).toarray()
    rng = np.random.default_rng(3)
    b = L @ rng.standard_normal(n)          # consistent (b in range L)
    xD = drazin_solution(L, b)
    res = dgmres(L, b, index=1, m=n, tol=1e-10)
    err = np.linalg.norm(res.x - xD) / max(np.linalg.norm(xD), 1e-30)
    check(f"DGMRES==A^D b on L_{n} (rel err {err:.2e}, iters {res.iterations})", err < 1e-5)

print(f"\n== SUMMARY: {npass} passed, {nfail} failed ==")
raise SystemExit(0 if nfail == 0 else 1)
