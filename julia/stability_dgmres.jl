# stability_dgmres.jl — DGMRES numerical-stability diagnostics (PROTOCOL.md 8.4, 3.11)
#
# Reviewer red flag 3.11 asks for MEASURED stability, not asserted stability.
# This script runs a DIAGNOSTIC variant of the DGMRES Arnoldi process (it never
# touches the timed harness path in harness.jl) that records, per Arnoldi step m:
#
#   * orthogonality drift        delta_m       = ||V_m^T V_m - I||_2
#   * Arnoldi-relation residual  arnoldi_resid = ||A V_m - V_{m+1} Hbar_m||_2
#   * least-squares condition    ls_cond       = cond_2( LS operator solved at step m )
#   * Hessenberg condition       cond_H        = cond_2( Hbar_m )                (secondary)
#   * a-th residual              lsq_res_rel   = ||A^a r_m|| / ||A^a b||         (the stop test)
#
# and, at the end, the final Drazin error against the exported ground truth x_star
# (PROTOCOL.md 8.3), i.e. ||x_final - x_star|| / ||x_star||.
#
# The DGMRES algorithm is byte-identical across Julia and Python (the work-count
# identity is already proven in the Phase-4 sweep: fig_v2_work_identity), and the
# quantities above are language-independent floating-point diagnostics. It is
# therefore sufficient to compute them in ONE language; Julia is used here. This
# is stated explicitly in the generated table caption.
#
# A single-precision (Float32) vs double-precision (Float64) pass is run on a
# subset spanning conditioning and index, to expose precision sensitivity.
#
# Output: code/results/benchmarks/stability_dgmres.csv  (long / per-iteration).

using LinearAlgebra
using SparseArrays
using Printf

const HERE = @__DIR__
const CODE_DIR = normpath(joinpath(HERE, ".."))
const DATA_DIR = joinpath(CODE_DIR, "data")
const OUT_DIR = joinpath(CODE_DIR, "results", "benchmarks")

include(joinpath(HERE, "data_layer.jl"))
using .DataLayer: read_mtx, read_npy

# ---------------------------------------------------------------------------
# Diagnostic DGMRES (mirrors DrazinKrylov.dgmres / harness.dgmres_counted exactly,
# same MGS Arnoldi from A^a r0, same least-squares objective — but instrumented
# and generic over the working precision T). NOT on any timed path.
# ---------------------------------------------------------------------------
struct StabTrace
    m::Vector{Int}
    delta_m::Vector{Float64}
    cond_H::Vector{Float64}
    ls_cond::Vector{Float64}
    arnoldi_resid::Vector{Float64}
    lsq_res_rel::Vector{Float64}
    x::Vector{Float64}
    iters::Int
    converged::Bool
    a::Int
end

# A^p * M for small p, in precision T
function _apply_power(A, M, p::Int)
    R = copy(M)
    for _ in 1:p
        R = A * R
    end
    return R
end

"""
    dgmres_diagnostic(A, b; index, m, tol) -> StabTrace

Instrumented DGMRES. `A`, `b` carry the working precision (Float32 or Float64);
all diagnostics are computed in that precision, then promoted to Float64 for
storage. `index = a = ind(A)` and `m` (Krylov cap) match the harness defaults.
"""
function dgmres_diagnostic(A::AbstractMatrix{T}, b::AbstractVector{T};
                           index::Int, m::Int, tol::Real) where {T<:AbstractFloat}
    n = size(A, 1)
    a = index
    x = zeros(T, n)

    r0 = b - A * x
    w0 = _apply_power(A, reshape(r0, n, 1), a)[:, 1]      # A^a r0
    beta = norm(w0)

    ms = Int[]; dds = Float64[]; cHs = Float64[]; clss = Float64[]
    arn = Float64[]; lsq = Float64[]

    if beta == 0
        return StabTrace(ms, dds, cHs, clss, arn, lsq, Float64.(x), 0, true, a)
    end

    V = zeros(T, n, m + 1)
    H = zeros(T, m + 1, m)
    V[:, 1] = w0 ./ beta

    converged = false
    used = 0
    for j in 1:m
        w = A * V[:, j]
        for i in 1:j                                     # modified Gram-Schmidt
            H[i, j] = dot(V[:, i], w)
            w -= H[i, j] .* V[:, i]
        end
        H[j + 1, j] = norm(w)
        if H[j + 1, j] > T(1e-14)
            V[:, j + 1] = w ./ H[j + 1, j]
        end

        Vj1 = @view V[:, 1:(j + 1)]
        Hj = @view H[1:(j + 1), 1:j]

        # least-squares correction (a>0 for every singular instance here)
        local z, rres, ls_operator
        if a == 0
            g = zeros(T, j + 1); g[1] = beta
            z = Hj \ g
            rres = norm(g - Hj * z)
            ls_operator = Matrix{Float64}(Hj)
        else
            P = _apply_power(A, Matrix(Vj1), a)          # A^a V_{j+1}
            ls_operator = Float64.(P * Hj)               # the matrix actually solved
            z = (P * Hj) \ w0
            rres = norm(w0 - P * Hj * z)
        end

        # --- diagnostics at step j (computed in T, stored as Float64) ---
        Vj = @view V[:, 1:j]                             # n x j orthonormal basis
        G = Float64.(Vj' * Vj) - Matrix{Float64}(I, j, j)
        delta = opnorm(G, 2)                             # ||V_j^T V_j - I||_2
        # Arnoldi relation residual ||A V_j - V_{j+1} Hbar_j||_2
        AVj = Float64.(A * Vj)
        VH = Float64.((@view V[:, 1:(j + 1)]) * (@view H[1:(j + 1), 1:j]))
        arn_res = opnorm(AVj - VH, 2)
        # condition numbers via singular values
        sH = svdvals(Float64.(Hj))
        cond_H = (length(sH) == 0 || minimum(sH) == 0) ? Inf : maximum(sH) / minimum(sH)
        sL = svdvals(ls_operator)
        cond_ls = (length(sL) == 0 || minimum(sL) == 0) ? Inf : maximum(sL) / minimum(sL)

        push!(ms, j); push!(dds, delta); push!(cHs, cond_H); push!(clss, cond_ls)
        push!(arn, arn_res); push!(lsq, Float64(rres / beta))
        used = j

        if rres <= tol * beta
            x = x + Vj * z
            converged = true
            break
        end
        if j == m || H[j + 1, j] <= T(1e-14)
            x = x + Vj * z
            break
        end
    end
    return StabTrace(ms, dds, cHs, clss, arn, lsq, Float64.(x), used, converged, a)
end

# ---------------------------------------------------------------------------
# Instance table (representative singular instances, PROTOCOL.md 3.2 / 8.4).
# One representative seeded RHS (seed 1000) per instance; the diagnostics are
# per-RHS-deterministic so a single seed suffices for the stability picture.
#   ref = :exported  -> use the manifest-exported x_star = A^D b (section 8.3)
# ---------------------------------------------------------------------------
struct Inst
    label::String
    family::String
    dir::String
    k::Int
    cond_B::Float64
    precisions::Vector{DataType}   # which precisions to run
end

const SEED = 1000
const KRYLOV_M = 120   # harness default (run_full.py --krylov-m)
const TOL = 1e-8       # harness default

const INSTANCES = Inst[
    Inst("similarity k=3 (well-cond)", "similarity_singular",
         "similarity_singular/ncore040_k3_n00000043", 3, 10.0, [Float64, Float32]),
    Inst("similarity k=4",             "similarity_singular",
         "similarity_singular/ncore040_k4_n00000044", 4, 10.0, [Float64]),
    Inst("illcond_block k=2",          "illcond_block_singular",
         "illcond_block_singular/ncore040_k2_n00000042", 2, 1.0e6, [Float64, Float32]),
    Inst("blockdiag_high_index k=5",   "blockdiag_high_index_singular",
         "blockdiag_high_index_singular/ncore040_k5_n00000045", 5, 10.0, [Float64, Float32]),
    Inst("coupled k=3",                "coupled_singular",
         "coupled_singular/ncore040_k3_n00000043", 3, 10.0, [Float64]),
    Inst("similarity n=1000 (dense; fails)", "similarity_singular",
         "similarity_singular/ncore997_k3_n00001000", 3, 10.0, [Float64]),
]

function load_instance(inst::Inst)
    A = read_mtx(joinpath(DATA_DIR, inst.dir, "A.mtx"))
    b = read_npy(joinpath(DATA_DIR, inst.dir, "b_seed$(SEED).npy"))
    xstar = read_npy(joinpath(DATA_DIR, inst.dir, "x_star_seed$(SEED).npy"))
    return A, b, xstar
end

function main()
    mkpath(OUT_DIR)
    out = joinpath(OUT_DIR, "stability_dgmres.csv")
    cols = ["instance", "family", "n", "k", "cond_B", "precision", "m",
            "delta_m", "cond_H", "ls_cond", "arnoldi_resid", "lsq_res_rel",
            "final_drazin_err", "final_res_rel", "converged", "iters_total", "seed"]
    open(out, "w") do io
        println(io, join(cols, ","))
        for inst in INSTANCES
            A64, b64, xstar = load_instance(inst)
            n = size(A64, 1)
            nrm_xstar = norm(xstar)
            for T in inst.precisions
                A = T == Float64 ? A64 : SparseMatrixCSC{T,Int}(A64)
                b = T.(b64)
                # cap at n-1 so the least-squares operator A^a V_{j+1} H_j stays
                # rectangular (QR min-norm solve); a full j=n step would form a
                # square, possibly singular operator and throw in exact-\ paths.
                mcap = min(KRYLOV_M, n - 1)
                tr = dgmres_diagnostic(A, b; index = inst.k, m = mcap, tol = TOL)
                derr = nrm_xstar == 0 ? norm(tr.x) : norm(tr.x - xstar) / nrm_xstar
                final_res = isempty(tr.lsq_res_rel) ? NaN : tr.lsq_res_rel[end]
                pname = T == Float64 ? "float64" : "float32"
                for idx in eachindex(tr.m)
                    row = [inst.label, inst.family, string(n), string(inst.k),
                           @sprintf("%.6g", inst.cond_B), pname, string(tr.m[idx]),
                           @sprintf("%.8g", tr.delta_m[idx]),
                           @sprintf("%.8g", tr.cond_H[idx]),
                           @sprintf("%.8g", tr.ls_cond[idx]),
                           @sprintf("%.8g", tr.arnoldi_resid[idx]),
                           @sprintf("%.8g", tr.lsq_res_rel[idx]),
                           @sprintf("%.8g", derr),
                           @sprintf("%.8g", final_res),
                           string(tr.converged), string(tr.iters), string(SEED)]
                    println(io, join(row, ","))
                end
                @printf("  %-32s [%s]  m=%3d  δ_max=%.2e  arnoldi_max=%.2e  ls_cond_max=%.2e  derr=%.2e  conv=%s\n",
                        inst.label, pname, tr.iters,
                        isempty(tr.delta_m) ? NaN : maximum(tr.delta_m),
                        isempty(tr.arnoldi_resid) ? NaN : maximum(tr.arnoldi_resid),
                        (isempty(tr.ls_cond) || isempty(filter(isfinite, tr.ls_cond))) ? NaN : maximum(filter(isfinite, tr.ls_cond)),
                        derr, tr.converged)
            end
        end
    end
    println("wrote ", out)
end

main()
