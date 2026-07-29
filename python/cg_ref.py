"""Reference CG (Jacobi-PCG) mirror of code/julia/cg_ref.jl, plus a wrapper around
scipy.sparse.linalg.cg to reproduce and explain the library's behaviour.
Identical deterministic RHS and stopping rule as the Julia version.
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def laplacian_1d(n):
    """1-D Dirichlet Laplacian tridiag(-1,2,-1): SPD, sparse, cond ~ n^2."""
    d = 2.0 * np.ones(n)
    o = -np.ones(n - 1)
    return sp.diags([o, d, o], [-1, 0, 1], format="csr")


def rhs_deterministic(n):
    """b_i = sin(0.1 i), normalised — identical to the Julia closed form."""
    b = np.sin(0.1 * np.arange(1, n + 1))
    return b / np.linalg.norm(b)


def cg_ref(A, b, rtol=1e-8, atol=0.0, maxiter=None, jacobi=False):
    """Preconditioned CG, same algorithm/stopping rule as CGRef.cg_ref in Julia."""
    n = len(b)
    if maxiter is None:
        maxiter = n
    x = np.zeros(n)
    r = b.copy()
    Minv = (1.0 / A.diagonal()) if jacobi else np.ones(n)
    z = Minv * r
    p = z.copy()
    rz = r @ z
    nb = np.linalg.norm(b)
    thresh = max(rtol * nb, atol)
    iters = 0
    converged = False
    for k in range(1, maxiter + 1):
        Ap = A @ p
        alpha = rz / (p @ Ap)
        x += alpha * p
        r -= alpha * Ap
        iters = k
        if np.linalg.norm(r) <= thresh:
            converged = True
            break
        z = Minv * r
        rz_new = r @ z
        p = z + (rz_new / rz) * p
        rz = rz_new
    return x, iters, converged, np.linalg.norm(b - A @ x) / nb


def scipy_cg(A, b, rtol=1e-8, atol=0.0, maxiter=None):
    """scipy.sparse.linalg.cg with explicit modern params + iteration counter."""
    if maxiter is None:
        maxiter = len(b)
    cnt = {"k": 0}

    def cb(xk):
        cnt["k"] += 1

    x, info = spla.cg(A, b, rtol=rtol, atol=atol, maxiter=maxiter, callback=cb)
    relres = np.linalg.norm(b - A @ x) / np.linalg.norm(b)
    return x, cnt["k"], (info == 0), relres, info
