"""
    DrazinKrylov

Reference implementations for the Drazin-inverse solution of singular linear
systems and a Krylov-subspace solver (DGMRES, Sidi 2001).

The module has two jobs:
  1. Provide a *ground-truth* Drazin inverse (via the exact identity
     `A^D = A^k (A^{2k+1})^+ A^k`) so that any iterative method can be validated
     against a known-correct answer.
  2. Provide the DGMRES iterative solver that the paper actually claims to study.

Only stdlib dependencies (LinearAlgebra, SparseArrays).
"""
module DrazinKrylov

using LinearAlgebra
using SparseArrays
using Random: MersenneTwister

export drazin_index, drazin_inverse, drazin_solution,
       block_nilpotent, graph_laplacian_singular, sparse_block_singular,
       drazin_solution_block, dgmres, dgmres_restarted, DGMRESResult

# ---------------------------------------------------------------------------
# Drazin index and ground-truth Drazin inverse
# ---------------------------------------------------------------------------

"""
    drazin_index(A; tol) -> k

Smallest non-negative integer `k` such that `rank(A^k) == rank(A^(k+1))`.
`k == 0` iff `A` is nonsingular. Numerical rank uses `rank(_, atol=tol)`.
"""
function drazin_index(A::AbstractMatrix; tol::Real = 1e-10)
    n = size(A, 1)
    Ak = Matrix{Float64}(I, n, n)   # A^0
    rprev = rank(Ak; atol = tol)    # = n
    for k in 1:n
        Ak = Ak * A
        r = rank(Ak; atol = tol)
        if r == rprev
            return k - 1
        end
        rprev = r
    end
    return n
end

"""
    drazin_inverse(A; tol) -> A^D

Exact Drazin inverse via `A^D = A^k (A^{2k+1})^+ A^k`, `k = ind(A)`.
This is a standard closed-form identity; used here as validation ground truth.
"""
function drazin_inverse(A::AbstractMatrix; tol::Real = 1e-10)
    Ad = Matrix{Float64}(A)
    k = drazin_index(Ad; tol = tol)
    if k == 0
        return inv(Ad)
    end
    Ak = Ad^k
    mid = pinv(Ad^(2k + 1))
    return Ak * mid * Ak
end

"""
    drazin_solution(A, b; tol) -> x_D = A^D b
"""
drazin_solution(A::AbstractMatrix, b::AbstractVector; tol::Real = 1e-10) =
    drazin_inverse(A; tol = tol) * b

# ---------------------------------------------------------------------------
# Singular test-matrix generators (with controlled Drazin index)
# ---------------------------------------------------------------------------

"""
    block_nilpotent(m, k; cond_core, rng) -> A (dense)

Block-diagonal `A = diag(B, N)` in a random (well-conditioned) basis, where
`B` is `m x m` invertible and `N` is a single `k x k` nilpotent Jordan block.
Hence `ind(A) == k` exactly. Lets us study how a Drazin solver scales with the
Drazin index — something the SPD Laplacian benchmark cannot probe.
"""
function block_nilpotent(m::Int, k::Int; cond_core::Real = 10.0, seed::Int = 12345)
    @assert m >= 1 && k >= 1
    n = m + k
    # invertible core with a controlled condition number
    U = qr(_seeded_randn(seed, m, m)).Q |> Matrix
    V = qr(_seeded_randn(seed + 1, m, m)).Q |> Matrix
    svals = range(1.0, cond_core; length = m)
    B = U * Diagonal(collect(svals)) * V'
    # single nilpotent Jordan block of size k (super-diagonal ones)
    N = zeros(k, k)
    for i in 1:(k - 1)
        N[i, i + 1] = 1.0
    end
    A = zeros(n, n)
    A[1:m, 1:m] = B
    A[(m + 1):n, (m + 1):n] = N
    # rotate into a random orthonormal basis so structure isn't trivially aligned
    Q = qr(_seeded_randn(seed + 2, n, n)).Q |> Matrix
    return Q * A * Q'
end

"""
    graph_laplacian_singular(n) -> L (sparse SPD-singular, index 1)

1-D path-graph (Neumann) Laplacian: symmetric positive *semi*-definite, singular
with a one-dimensional null space (the constant vector). `ind(L) == 1`.
This keeps continuity with the paper's Laplacian theme while being genuinely
singular — the case the current benchmark never tests.
"""
function graph_laplacian_singular(n::Int)
    @assert n >= 2
    I_idx = Int[]; J_idx = Int[]; Vv = Float64[]
    deg(i) = (i == 1 || i == n) ? 1.0 : 2.0
    for i in 1:n
        push!(I_idx, i); push!(J_idx, i); push!(Vv, deg(i))
        if i > 1
            push!(I_idx, i); push!(J_idx, i - 1); push!(Vv, -1.0)
        end
        if i < n
            push!(I_idx, i); push!(J_idx, i + 1); push!(Vv, -1.0)
        end
    end
    return sparse(I_idx, J_idx, Vv, n, n)
end

"""
    sparse_block_singular(n_core, k; gamma, seed) -> (A, coreblk, nilblk)

Sparse, NON-symmetric, singular operator with exact Drazin index `k`, scalable
to large `n_core`. Block layout (kept axis-aligned for sparsity/scalability):

  A = [ B  0 ]     B : n_core x n_core, non-symmetric tridiagonal, strictly
      [ 0  N ]         diagonally dominant (diag 3, sub -1-γ, super -1+γ) ⇒ invertible
                   N : k x k single nilpotent Jordan block ⇒ ind(A) = k

Because A^D = diag(B^{-1}, 0), the Drazin solution of `A x = b` is known in
closed form (`drazin_solution_block`), giving an exact, O(n) ground truth even
at scales where a dense pseudoinverse is impossible. Solvers receive the full
`A` and never exploit the partition.
"""
function sparse_block_singular(n_core::Int, k::Int; gamma::Real = 0.5, seed::Int = 12345)
    @assert n_core >= 1 && k >= 1
    n = n_core + k
    I_idx = Int[]; J_idx = Int[]; Vv = Float64[]
    add!(i, j, v) = (push!(I_idx, i); push!(J_idx, j); push!(Vv, v))
    for i in 1:n_core
        add!(i, i, 3.0)
        i > 1        && add!(i, i - 1, -1.0 - gamma)
        i < n_core   && add!(i, i + 1, -1.0 + gamma)
    end
    for i in 1:(k - 1)          # nilpotent Jordan block on the trailing k indices
        add!(n_core + i, n_core + i + 1, 1.0)
    end
    A = sparse(I_idx, J_idx, Vv, n, n)
    return A, 1:n_core, (n_core + 1):n
end

"""
    drazin_solution_block(A, b, coreblk, nilblk) -> A^D b (exact, cheap)

Closed-form Drazin solution for a `sparse_block_singular` operator:
`x[core] = B \\ b[core]`, `x[nilpotent] = 0`.
"""
function drazin_solution_block(A::AbstractMatrix, b::AbstractVector, coreblk, nilblk)
    x = zeros(length(b))
    B = A[coreblk, coreblk]
    x[coreblk] = B \ Vector(b[coreblk])
    return x
end

# reproducible randn without touching Random's global state
_seeded_randn(seed::Int, r::Int, c::Int) = randn(MersenneTwister(seed), r, c)

# ---------------------------------------------------------------------------
# DGMRES  (Sidi, ETNA 2001) — Drazin-inverse solution via a GMRES-type method
# ---------------------------------------------------------------------------

struct DGMRESResult
    x::Vector{Float64}          # approximate Drazin solution
    residual_history::Vector{Float64}   # ‖A^a r_m‖ per iteration
    iterations::Int
    converged::Bool
    index_used::Int
end

"""
    dgmres(A, b; index, m, tol, x0) -> DGMRESResult

Compute the Drazin-inverse solution of `A x = b`.

DGMRES builds the Krylov space `K_m(A, r0)` via Arnoldi and, at step `m`,
chooses the correction that minimises `‖A^a r_m‖_2` (the *a-th* residual,
`a = ind(A)`), rather than `‖r_m‖_2` as in plain GMRES. Pre-multiplying by
`A^a` annihilates the nilpotent / null-space component tied to the zero
eigenvalue, so the minimiser targets exactly `A^D b`.

`index` may be given to skip index detection; `m` caps the Krylov dimension.
"""
function dgmres(A::AbstractMatrix, b::AbstractVector;
                index::Union{Int,Nothing} = nothing,
                m::Int = min(size(A, 1), 200),
                tol::Real = 1e-8,
                x0::Union{AbstractVector,Nothing} = nothing)
    n = size(A, 1)
    a = index === nothing ? drazin_index(A) : index
    x = x0 === nothing ? zeros(n) : Vector{Float64}(x0)

    r0 = b - A * x
    reshist = Float64[]

    # Sidi's key step: the correction must live in K_m(A, A^a r0) ⊆ range(A^a),
    # so the nilpotent/null component is annihilated up front and x_m stays on
    # the subspace where the Drazin solution lives. Arnoldi therefore starts
    # from A^a r0, NOT r0.
    w0 = _apply_power(A, reshape(r0, n, 1), a)[:, 1]   # A^a r0
    beta = norm(w0)
    if beta == 0
        return DGMRESResult(x, reshist, 0, true, a)    # A^a b = 0 ⇒ A^D b = 0
    end

    V = zeros(n, m + 1)
    H = zeros(m + 1, m)
    V[:, 1] = w0 ./ beta

    converged = false
    used = 0
    for j in 1:m
        w = A * V[:, j]
        for i in 1:j
            H[i, j] = dot(V[:, i], w)
            w -= H[i, j] .* V[:, i]
        end
        H[j + 1, j] = norm(w)
        if H[j + 1, j] > 1e-14
            V[:, j + 1] = w ./ H[j + 1, j]
        end

        # min_z ‖A^a r_m‖ = ‖ A^a r0 - (A^a V_{j+1}) H_j z ‖ , since A V_j = V_{j+1} H_j
        Vj1 = @view V[:, 1:(j + 1)]
        Hj = @view H[1:(j + 1), 1:j]
        local z, rres
        if a == 0
            # standard, numerically stable GMRES: min ‖β e1 - H_j z‖ (V orthonormal,
            # so w0 = β v1 projects to β e1). Keeps the baseline textbook-fair.
            g = zeros(j + 1); g[1] = beta
            z = Hj \ g
            rres = norm(g - Hj * z)
        else
            P = _apply_power(A, Matrix(Vj1), a) # A^a V_{j+1}  (n x (j+1))
            z = (P * Hj) \ w0                   # least-squares correction
            rres = norm(w0 - P * Hj * z)
        end
        push!(reshist, rres)
        used = j
        if rres <= tol * beta
            x = x + V[:, 1:j] * z
            converged = true
            break
        end
        if j == m || H[j + 1, j] <= 1e-14       # cap reached or Krylov exhausted
            x = x + V[:, 1:j] * z
            break
        end
    end
    return DGMRESResult(x, reshist, used, converged, a)
end

# A^p * M for small p (p = 0,1,2,...) — reused by the least-squares objective
function _apply_power(A::AbstractMatrix, M::AbstractMatrix, p::Int)
    R = Matrix{Float64}(M)
    for _ in 1:p
        R = A * R
    end
    return R
end

"""
    dgmres_restarted(A, b; index, restart, maxouter, tol) -> DGMRESResult

Restarted DGMRES(`restart`). Each cycle runs the full DGMRES on the current
iterate with a Krylov cap of `restart`, then uses the result as the next
starting guess. Memory is bounded by the `restart`-column Arnoldi basis
(independent of the total iteration count), so this scales to large `n` where
the non-restarted method's dense basis would be prohibitive.

Convergence is judged globally on the `a`-th residual:
`‖A^a (b - A x)‖ ≤ tol · ‖A^a b‖`. Because every cycle's correction lies in
`range(A^a)`, the nilpotent component is never reintroduced and the limit is
still `A^D b`.
"""
function dgmres_restarted(A::AbstractMatrix, b::AbstractVector;
                          index::Union{Int,Nothing} = nothing,
                          restart::Int = 30, maxouter::Int = 500,
                          tol::Real = 1e-10)
    n = size(A, 1)
    a = index === nothing ? drazin_index(A) : index
    x = zeros(n)
    ath(v) = _apply_power(A, reshape(v, n, 1), a)[:, 1]   # A^a v
    ref = norm(ath(b))
    ref = ref == 0 ? 1.0 : ref
    reshist = Float64[]
    total = 0
    converged = false
    for cyc in 1:maxouter
        res = dgmres(A, b; index = a, m = restart, tol = tol, x0 = x)
        x = res.x
        total += res.iterations
        g = norm(ath(b - A * x)) / ref          # global a-th residual
        push!(reshist, g)
        if g <= tol
            converged = true
            break
        end
    end
    return DGMRESResult(x, reshist, total, converged, a)
end

end # module
