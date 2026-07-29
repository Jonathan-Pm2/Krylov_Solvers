"""
    FamiliesPhase3

Phase-3 matrix-family generators (PROTOCOL.md sections 3.1, 3.2, 3.3, 8.3).
Julia remains the single source of truth: every instance is generated once, in
memory, here, then exported by the shared `DataLayer` code path
(`generate_data_phase3.jl`). This module broadens the test set far beyond the
1-D Laplacian (closes red flags 1.12, 3.13) and beyond the trivial
block-diagonal singular case (closes 1.14).

Contents:

  SPD families (section 3.1)
    - laplacian_2d       : 5-point Dirichlet Laplacian on an m x m grid.
    - laplacian_3d       : 7-point Dirichlet Laplacian on an m x m x m grid.
    - anisotropic_2d     : anisotropic 2-D diffusion with ratio `eps`.
    - varied_cond_spd    : SPD family with a *controlled* condition number.

  Singular families (section 3.2), each with an EXACT closed-form Drazin
  solution `x_star = A^D b` (section 8.3 ground truth):
    - similarity_singular : A' = S diag(B,N) S^{-1}, S well-conditioned but
                            NON-orthogonal (couples the invariant subspaces).
                            A'^D = S diag(B^{-1},0) S^{-1}.
    - coupled_singular    : block *upper*-triangular A = [B C; 0 N] with
                            genuine off-diagonal coupling C.  Closed form
                            A^D = [B^{-1}  Z; 0 0],  Z = sum_{i=0}^{k-1}
                            (B^{-1})^{i+2} C N^i.
    - illcond_block_singular : block-diagonal diag(B,N) with an ILL-conditioned
                            invertible block B (controlled cond(B)).
    - (higher Drazin index k >= 4 is obtained by choosing the nilpotent block
       size k in any of the above.)

  Condition-number helper (`spd_condition`) and a symmetric-aware Matrix Market
  loader (`read_mtx_symmetric`) for SuiteSparse `.mtx` files.

Only stdlib dependencies (LinearAlgebra, SparseArrays, Random).
"""
module FamiliesPhase3

using LinearAlgebra
using SparseArrays
using Random: MersenneTwister

export laplacian_2d, laplacian_3d, anisotropic_2d, varied_cond_spd,
       similarity_singular, coupled_singular, illcond_block_singular,
       drazin_sol_blockdiag, drazin_sol_similarity, drazin_sol_coupled,
       spd_condition, read_mtx_symmetric, seeded_rhs, RHS_SEEDS

# 20 fixed RHS seeds (PROTOCOL.md section 3.3). Recorded in the manifest.
const RHS_SEEDS = collect(1000:1019)

"Seeded standard-normal RHS with a local RNG (global stream untouched)."
seeded_rhs(n::Int, seed::Int) = randn(MersenneTwister(seed), n)

# ---------------------------------------------------------------------------
# Small linear-algebra helpers
# ---------------------------------------------------------------------------

"1-D Dirichlet stiffness tridiag(-1, 2, -1), size m (sparse)."
tridiag_1d(m::Int) = spdiagm(-1 => fill(-1.0, m - 1),
                              0 => fill(2.0, m),
                              1 => fill(-1.0, m - 1))

"Orthonormal m x m matrix from the QR of a seeded random matrix."
orthonormal(seed::Int, m::Int) = Matrix(qr(randn(MersenneTwister(seed), m, m)).Q)

"Logarithmically spaced values in [1, kappa] of length m."
logspaced(kappa::Real, m::Int) = exp10.(range(0.0, log10(kappa); length = m))

"Single k x k nilpotent Jordan block (ones on the super-diagonal), dense."
function nilpotent_block(k::Int)
    N = zeros(k, k)
    for i in 1:(k - 1)
        N[i, i + 1] = 1.0
    end
    return N
end

# ---------------------------------------------------------------------------
# SPD family 3.1
# ---------------------------------------------------------------------------

"""
    laplacian_2d(m) -> A (sparse SPD), n = m^2

5-point Dirichlet Laplacian on an m x m interior grid:
`A = I_m ⊗ T_m + T_m ⊗ I_m`, `T_m = tridiag(-1,2,-1)`. SPD, 5 nonzeros/row
(interior). Isotropic; condition number grows like m^2.
"""
function laplacian_2d(m::Int)
    @assert m >= 2
    T = tridiag_1d(m)
    Im = sparse(1.0I, m, m)
    return kron(Im, T) + kron(T, Im)
end

"""
    laplacian_3d(m) -> A (sparse SPD), n = m^3

7-point Dirichlet Laplacian on an m x m x m interior grid:
`A = I⊗I⊗T + I⊗T⊗I + T⊗I⊗I`. SPD, 7 nonzeros/row (interior).
"""
function laplacian_3d(m::Int)
    @assert m >= 2
    T = tridiag_1d(m)
    Im = sparse(1.0I, m, m)
    I2 = kron(Im, Im)
    return kron(I2, T) + kron(kron(Im, T), Im) + kron(T, I2)
end

"""
    anisotropic_2d(m, eps) -> A (sparse SPD), n = m^2

Anisotropic 2-D diffusion operator `-(eps ∂xx + ∂yy)` on an m x m grid:
`A = eps (I_m ⊗ T_m) + (T_m ⊗ I_m)`. SPD for every `eps > 0`; the anisotropy
ratio is `eps` (eps << 1 = strong anisotropy). Both Kronecker terms are SPD so
their positive combination is SPD.
"""
function anisotropic_2d(m::Int, eps::Real)
    @assert m >= 2 && eps > 0
    T = tridiag_1d(m)
    Im = sparse(1.0I, m, m)
    return eps * kron(Im, T) + kron(T, Im)
end

"""
    varied_cond_spd(n, kappa; seed) -> (A dense SPD, kappa_exact)

SPD matrix with a DIRECTLY CONTROLLED condition number. Construction:
`A = Q diag(λ) Qᵀ`, `Q` orthogonal (QR of a seeded random matrix), eigenvalues
`λ` logarithmically spaced in `[1, kappa]`. Hence `κ₂(A) = λ_max/λ_min = kappa`
exactly (up to floating-point rounding), and every `λ_i > 0` so `A` is SPD.
Returns the matrix and the realized condition number `λ_max/λ_min`.
"""
function varied_cond_spd(n::Int, kappa::Real; seed::Int = 20240)
    @assert n >= 2 && kappa >= 1
    Q = orthonormal(seed, n)
    lam = logspaced(kappa, n)
    A = Symmetric(Q * Diagonal(lam) * Q')
    return Matrix(A), maximum(lam) / minimum(lam)
end

# ---------------------------------------------------------------------------
# Singular family 3.2 + exact Drazin ground truth (section 8.3)
# ---------------------------------------------------------------------------

"""
    _invertible_core(n_core; cond, seed) -> B (dense, invertible)

Non-symmetric invertible block with a controlled 2-norm condition number
`B = U diag(s) Vᵀ`, `U,V` distinct orthogonal, `s` logspaced in `[1, cond]`.
"""
function _invertible_core(n_core::Int; cond::Real = 10.0, seed::Int = 30000)
    U = orthonormal(seed, n_core)
    V = orthonormal(seed + 1, n_core)
    s = logspaced(cond, n_core)
    return U * Diagonal(s) * V'
end

"""
    similarity_singular(n_core, k; cond_B, cond_S, seed)
        -> (A, x_star_fn, meta)

`A = S · diag(B, N) · S⁻¹` where:
  - `B` is `n_core x n_core` invertible (cond ~ `cond_B`),
  - `N` is a `k x k` nilpotent Jordan block  ⇒ `ind(A) = k`,
  - `S` is `n x n`, WELL-conditioned (cond ~ `cond_S`) but NON-orthogonal, built
    as `S = U diag(s) Vᵀ` with `U ≠ V` and singular values in `[1, cond_S]`
    (singular values ≠ 1 ⇒ `S` is not orthogonal, so the similarity genuinely
    couples the invertible and nilpotent subspaces — destroys the trivial split).

Exact Drazin inverse: `A^D = S · diag(B⁻¹, 0) · S⁻¹` (section 8.3), so
`x_star(b) = S · diag(B⁻¹,0) · (S⁻¹ b)` is returned in closed form.
`meta` records `cond(S)`, `cond(B)`, and the construction.
"""
function similarity_singular(n_core::Int, k::Int;
                             cond_B::Real = 10.0, cond_S::Real = 3.0,
                             seed::Int = 40000)
    @assert n_core >= 1 && k >= 1
    n = n_core + k
    B = _invertible_core(n_core; cond = cond_B, seed = seed)
    N = nilpotent_block(k)
    D = zeros(n, n)
    D[1:n_core, 1:n_core] = B
    D[(n_core + 1):n, (n_core + 1):n] = N
    # well-conditioned NON-orthogonal S
    U = orthonormal(seed + 10, n)
    V = orthonormal(seed + 11, n)
    s = logspaced(cond_S, n)
    S = U * Diagonal(s) * V'
    Sinv = inv(S)
    A = S * D * Sinv
    Bfact = lu(B)
    x_star_fn = b -> begin
        y = Sinv * b                 # S⁻¹ b
        z = zeros(n)
        z[1:n_core] = Bfact \ y[1:n_core]   # diag(B⁻¹,0) applied
        return S * z                 # S · (…)
    end
    meta = Dict{String,Any}(
        "n_core" => n_core, "k" => k,
        "cond_B" => cond(B), "cond_S" => cond(S),
        "construction" => "A = S*diag(B,N)*inv(S); S well-conditioned NON-orthogonal; A^D = S*diag(B^{-1},0)*inv(S)",
    )
    return A, x_star_fn, meta
end

"""
    coupled_singular(n_core, k; cond_B, coupling, seed) -> (A, x_star_fn, meta)

Block *upper*-triangular operator with genuine off-diagonal coupling between the
invertible and nilpotent parts:

    A = [ B  C ]     B invertible (n_core x n_core),
        [ 0  N ]     N nilpotent Jordan block (k x k)  ⇒ ind(A) = k,
                     C = `coupling` · R,  R seeded random (n_core x k).

Closed-form Drazin inverse of a block-triangular matrix (Campbell & Meyer):
`A^D = [B⁻¹  Z; 0  0]`, `Z = Σ_{i=0}^{k-1} (B⁻¹)^{i+2} C N^i` (finite since
`N^k = 0`). Hence `x_star(b) = [B⁻¹ b₁ + Z b₂; 0]` is exact.
"""
function coupled_singular(n_core::Int, k::Int;
                          cond_B::Real = 10.0, coupling::Real = 1.0,
                          seed::Int = 50000)
    @assert n_core >= 1 && k >= 1
    n = n_core + k
    B = _invertible_core(n_core; cond = cond_B, seed = seed)
    N = nilpotent_block(k)
    C = coupling .* randn(MersenneTwister(seed + 5), n_core, k)
    A = zeros(n, n)
    A[1:n_core, 1:n_core] = B
    A[1:n_core, (n_core + 1):n] = C
    A[(n_core + 1):n, (n_core + 1):n] = N
    Binv = inv(B)
    # Z = Σ_{i=0}^{k-1} (B⁻¹)^{i+2} C N^i
    Z = zeros(n_core, k)
    Ni = Matrix{Float64}(I, k, k)         # N^0
    Binv_pow = Binv * Binv                 # (B⁻¹)^{2} for i = 0
    for i in 0:(k - 1)
        Z .+= Binv_pow * C * Ni
        Ni = Ni * N
        Binv_pow = Binv_pow * Binv
    end
    x_star_fn = b -> begin
        b1 = b[1:n_core]; b2 = b[(n_core + 1):n]
        x = zeros(n)
        x[1:n_core] = Binv * b1 + Z * b2
        return x
    end
    meta = Dict{String,Any}(
        "n_core" => n_core, "k" => k,
        "cond_B" => cond(B), "coupling" => Float64(coupling),
        "construction" => "A = [B C; 0 N] block-upper-triangular; A^D = [B^{-1} Z; 0 0], Z = sum_{i=0}^{k-1} (B^{-1})^{i+2} C N^i",
    )
    return A, x_star_fn, meta
end

"""
    illcond_block_singular(n_core, k; cond_B, seed) -> (A, x_star_fn, meta)

Block-diagonal `A = diag(B, N)` with an ILL-conditioned invertible block `B`
(cond ~ `cond_B`, e.g. 1e6). `N` is a `k x k` nilpotent block ⇒ `ind(A) = k`.
Exact Drazin: `A^D = diag(B⁻¹, 0)`, `x_star(b) = [B⁻¹ b₁; 0]`. The ill
conditioning stresses the core solve while the ground truth stays exact.
"""
function illcond_block_singular(n_core::Int, k::Int;
                                cond_B::Real = 1.0e6, seed::Int = 60000)
    @assert n_core >= 1 && k >= 1
    n = n_core + k
    B = _invertible_core(n_core; cond = cond_B, seed = seed)
    N = nilpotent_block(k)
    A = zeros(n, n)
    A[1:n_core, 1:n_core] = B
    A[(n_core + 1):n, (n_core + 1):n] = N
    Bfact = lu(B)
    x_star_fn = b -> begin
        x = zeros(n)
        x[1:n_core] = Bfact \ b[1:n_core]
        return x
    end
    meta = Dict{String,Any}(
        "n_core" => n_core, "k" => k,
        "cond_B" => cond(B),
        "construction" => "A = diag(B, N) with ill-conditioned B; A^D = diag(B^{-1}, 0)",
    )
    return A, x_star_fn, meta
end

# closed-form helpers exported for reuse/testing
"Drazin solution of diag(B,N): x = [B\\b1 ; 0]."
function drazin_sol_blockdiag(B::AbstractMatrix, b::AbstractVector)
    nc = size(B, 1); n = length(b)
    x = zeros(n)
    x[1:nc] = B \ Vector(b[1:nc])
    return x
end

# ---------------------------------------------------------------------------
# Condition number of an SPD matrix (extremal symmetric eigenvalues)
# ---------------------------------------------------------------------------

"""
    spd_condition(A; nmax) -> (kappa_or_nothing, method_string)

2-norm condition number of an SPD matrix via the extremal eigenvalues of the
dense `Symmetric(A)`. Skipped (returns `nothing`) when `n > nmax` because a
dense symmetric eigensolve is then too expensive; the manifest records why.
"""
function spd_condition(A::AbstractMatrix; nmax::Int = 5000)
    n = size(A, 1)
    if n > nmax
        return nothing, "skipped: n=$n > nmax=$nmax (dense symmetric eigensolve too expensive)"
    end
    ev = eigvals(Symmetric(Matrix(A)))
    lmin = minimum(ev); lmax = maximum(ev)
    (lmin <= 0) && return nothing, "not SPD (min eigenvalue $lmin <= 0)"
    return lmax / lmin, "extremal eigenvalues of dense Symmetric(A)"
end

# ---------------------------------------------------------------------------
# Symmetric-aware Matrix Market loader (for SuiteSparse .mtx downloads)
# ---------------------------------------------------------------------------

"""
    read_mtx_symmetric(path) -> (A::SparseMatrixCSC, header::NamedTuple)

Read a Matrix Market coordinate file, honoring the `symmetric` qualifier by
mirroring stored below-diagonal entries above the diagonal (the diagonal is not
duplicated). Supports `real` and `pattern` (pattern → value 1.0). This is used
ONLY to ingest downloaded SuiteSparse matrices; the ingested matrix is then
re-exported via `DataLayer.write_mtx` as a fully-expanded `general` file, so the
data layer on disk stays in one canonical format.

`header` reports `(symmetric::Bool, field::String, stored::Int)` for provenance.
"""
function read_mtx_symmetric(path::AbstractString)
    rows = Int[]; cols = Int[]; vals = Float64[]
    m = 0; n = 0; declared = -1; seen_size = false
    symmetric = false; field = "real"
    open(path, "r") do io
        for raw in eachline(io)
            line = strip(raw)
            isempty(line) && continue
            if startswith(line, "%%MatrixMarket")
                lc = lowercase(line)
                symmetric = occursin("symmetric", lc)
                field = occursin("pattern", lc) ? "pattern" :
                        occursin("integer", lc) ? "integer" :
                        occursin("complex", lc) ? "complex" : "real"
                continue
            end
            startswith(line, "%") && continue
            parts = split(line)
            if !seen_size
                m = parse(Int, parts[1]); n = parse(Int, parts[2])
                declared = parse(Int, parts[3]); seen_size = true
            else
                i = parse(Int, parts[1]); j = parse(Int, parts[2])
                v = field == "pattern" ? 1.0 : parse(Float64, parts[3])
                push!(rows, i); push!(cols, j); push!(vals, v)
                if symmetric && i != j
                    push!(rows, j); push!(cols, i); push!(vals, v)
                end
            end
        end
    end
    field == "complex" && error("read_mtx_symmetric: complex field unsupported ($path)")
    A = sparse(rows, cols, vals, m, n)
    return A, (symmetric = symmetric, field = field, stored = declared)
end

end # module
