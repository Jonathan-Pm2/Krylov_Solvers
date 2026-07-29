"""
drazin_krylov: Python mirror of the Julia DrazinKrylov module.

Ground-truth Drazin inverse (exact identity A^D = A^k (A^{2k+1})^+ A^k), singular
test-matrix generators with controlled Drazin index, and DGMRES (Sidi 2001).

Kept deliberately parallel to code/julia/DrazinKrylov.jl so the cross-language
comparison measures the runtimes, not divergent algorithms.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from numpy.linalg import matrix_rank, pinv, inv, norm, svd

TOL_RANK = 1e-10


# ---------------------------------------------------------------------------
# Drazin index and ground-truth Drazin inverse
# ---------------------------------------------------------------------------
def drazin_index(A: np.ndarray, tol: float = TOL_RANK) -> int:
    """Smallest k with rank(A^k) == rank(A^(k+1)); 0 iff A nonsingular."""
    A = np.asarray(A, dtype=float)
    n = A.shape[0]
    Ak = np.eye(n)
    rprev = matrix_rank(Ak, tol=tol)
    for k in range(1, n + 1):
        Ak = Ak @ A
        r = matrix_rank(Ak, tol=tol)
        if r == rprev:
            return k - 1
        rprev = r
    return n


def drazin_inverse(A: np.ndarray, tol: float = TOL_RANK) -> np.ndarray:
    """Exact Drazin inverse via A^D = A^k (A^{2k+1})^+ A^k (validation ground truth)."""
    A = np.asarray(A, dtype=float)
    k = drazin_index(A, tol=tol)
    if k == 0:
        return inv(A)
    Ak = np.linalg.matrix_power(A, k)
    mid = pinv(np.linalg.matrix_power(A, 2 * k + 1))
    return Ak @ mid @ Ak


def drazin_solution(A: np.ndarray, b: np.ndarray, tol: float = TOL_RANK) -> np.ndarray:
    return drazin_inverse(A, tol=tol) @ b


# ---------------------------------------------------------------------------
# Singular test-matrix generators (controlled Drazin index)
# ---------------------------------------------------------------------------
def block_nilpotent(m: int, k: int, cond_core: float = 10.0, seed: int = 12345) -> np.ndarray:
    """A = diag(B, N_k) rotated into a random orthonormal basis; ind(A) == k."""
    assert m >= 1 and k >= 1
    n = m + k
    U = np.linalg.qr(np.random.default_rng(seed).standard_normal((m, m)))[0]
    V = np.linalg.qr(np.random.default_rng(seed + 1).standard_normal((m, m)))[0]
    svals = np.linspace(1.0, cond_core, m)
    B = U @ np.diag(svals) @ V.T
    N = np.zeros((k, k))
    for i in range(k - 1):
        N[i, i + 1] = 1.0
    A = np.zeros((n, n))
    A[:m, :m] = B
    A[m:, m:] = N
    Q = np.linalg.qr(np.random.default_rng(seed + 2).standard_normal((n, n)))[0]
    return Q @ A @ Q.T


def sparse_block_singular(n_core: int, k: int, gamma: float = 0.5, seed: int = 12345):
    """Sparse, NON-symmetric, singular operator with exact Drazin index k.

    A = diag(B, N): B is n_core x n_core non-symmetric tridiagonal, strictly
    diagonally dominant (diag 3, sub -1-gamma, super -1+gamma) => invertible;
    N is a k x k nilpotent Jordan block => ind(A) = k. Since A^D = diag(B^-1, 0),
    the Drazin solution is known in closed form and cheap even at large scale.
    Returns (A, core_slice, nil_slice).
    """
    assert n_core >= 1 and k >= 1
    n = n_core + k
    rows, cols, vals = [], [], []

    def add(i, j, v):
        rows.append(i); cols.append(j); vals.append(v)

    for i in range(n_core):
        add(i, i, 3.0)
        if i > 0:
            add(i, i - 1, -1.0 - gamma)
        if i < n_core - 1:
            add(i, i + 1, -1.0 + gamma)
    for i in range(k - 1):
        add(n_core + i, n_core + i + 1, 1.0)
    A = sp.csr_matrix((vals, (rows, cols)), shape=(n, n))
    return A, slice(0, n_core), slice(n_core, n)


def graph_laplacian_singular(n: int) -> sp.csr_matrix:
    """1-D path (Neumann) Laplacian: symmetric PSD, singular, index 1."""
    assert n >= 2
    main = np.full(n, 2.0)
    main[0] = 1.0
    main[-1] = 1.0
    off = -np.ones(n - 1)
    L = sp.diags([off, main, off], [-1, 0, 1], format="csr")
    return L


# ---------------------------------------------------------------------------
# DGMRES (Sidi, ETNA 2001)
# ---------------------------------------------------------------------------
class DGMRESResult:
    def __init__(self, x, residual_history, iterations, converged, index_used,
                 error_history=None):
        self.x = x
        self.residual_history = residual_history      # ‖A^a r_m‖ per iteration
        self.iterations = iterations
        self.converged = converged
        self.index_used = index_used
        self.error_history = error_history or []      # ‖x_m - xtrue‖/‖xtrue‖ if tracked


def _apply_power(A, M, p):
    R = np.asarray(M, dtype=float)
    for _ in range(p):
        R = A @ R
    return R


def dgmres_restarted(A, b, index=None, restart=30, maxouter=500, tol=1e-10) -> DGMRESResult:
    """Restarted DGMRES(restart) — mirror of the Julia version.

    Each cycle runs full DGMRES with Krylov cap `restart` from the current
    iterate; memory is bounded by the restart-column basis, so it scales to large
    n. Global convergence on the a-th residual: ‖A^a(b-Ax)‖ ≤ tol·‖A^a b‖. Every
    cycle's correction stays in range(A^a), so the limit is still A^D b.
    """
    n = A.shape[0]
    a = drazin_index(np.asarray(A.todense()) if sp.issparse(A) else A) if index is None else index

    def ath(v):
        return _apply_power(A, v.reshape(n, 1), a)[:, 0]

    x = np.zeros(n)
    ref = norm(ath(b))
    ref = ref if ref != 0 else 1.0
    reshist = []
    total = 0
    converged = False
    for _ in range(maxouter):
        res = dgmres(A, b, index=a, m=restart, tol=tol, x0=x)
        x = res.x
        total += res.iterations
        g = norm(ath(b - A @ x)) / ref
        reshist.append(g)
        if g <= tol:
            converged = True
            break
    return DGMRESResult(x, reshist, total, converged, a)


def dgmres(A, b, index=None, m=None, tol=1e-8, x0=None, xtrue=None) -> DGMRESResult:
    """Drazin-inverse solution of A x = b.

    Minimises ‖A^a r_m‖ (a = ind(A)) over the Krylov space K_m(A, A^a r0), which
    annihilates the null-space/nilpotent component tied to the zero eigenvalue
    and so targets exactly A^D b — where plain GMRES stagnates.

    If `xtrue` is given, the per-iteration relative error ‖x_m - xtrue‖/‖xtrue‖
    is recorded in `result.error_history` (used for convergence figures).
    """
    A_op = A
    n = A.shape[0]
    if m is None:
        m = min(n, 200)
    a = drazin_index(np.asarray(A.todense()) if sp.issparse(A) else A) if index is None else index
    x = np.zeros(n) if x0 is None else np.array(x0, dtype=float)

    r0 = b - A_op @ x
    reshist = []
    errhist = []
    nrm_xt = None if xtrue is None else max(np.linalg.norm(xtrue), 1e-30)

    # Sidi's key step: build K_m(A, A^a r0) ⊆ range(A^a) so the correction
    # cannot introduce a nilpotent/null component. Arnoldi starts from A^a r0.
    w0 = _apply_power(A_op, r0.reshape(n, 1), a)[:, 0]   # A^a r0
    beta = norm(w0)
    if beta == 0:
        return DGMRESResult(x, reshist, 0, True, a, errhist)  # A^a b = 0 ⇒ A^D b = 0

    V = np.zeros((n, m + 1))
    H = np.zeros((m + 1, m))
    V[:, 0] = w0 / beta

    converged = False
    used = 0
    for j in range(1, m + 1):
        w = A_op @ V[:, j - 1]
        for i in range(1, j + 1):
            H[i - 1, j - 1] = V[:, i - 1] @ w
            w = w - H[i - 1, j - 1] * V[:, i - 1]
        H[j, j - 1] = norm(w)
        if H[j, j - 1] > 1e-14:
            V[:, j] = w / H[j, j - 1]

        Vj1 = V[:, : j + 1]
        Hj = H[: j + 1, :j]
        if a == 0:
            # standard, numerically stable GMRES: min ‖β e1 - H_j z‖ (V is orthonormal,
            # so w0 = β v1 projects to β e1). Keeps the baseline textbook-fair.
            g = np.zeros(j + 1); g[0] = beta
            z, *_ = np.linalg.lstsq(Hj, g, rcond=None)
            rres = norm(g - Hj @ z)
        else:
            P = _apply_power(A_op, Vj1, a)          # A^a V_{j+1}
            M_ls = P @ Hj                            # n x j
            z, *_ = np.linalg.lstsq(M_ls, w0, rcond=None)
            rres = norm(w0 - M_ls @ z)
        reshist.append(rres)
        if nrm_xt is not None:
            x_iter = x + Vj1[:, :j] @ z
            errhist.append(norm(x_iter - xtrue) / nrm_xt)
        used = j
        if rres <= tol * beta:
            x = x + Vj1[:, :j] @ z
            converged = True
            break
        if j == m or H[j, j - 1] <= 1e-14:
            x = x + Vj1[:, :j] @ z
            break
    return DGMRESResult(x, reshist, used, converged, a, errhist)
