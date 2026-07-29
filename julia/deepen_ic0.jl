# deepen_ic0.jl — Round-2 deepening experiments 3.6 and 3.7.
#
# Reuses the harness IC(0) factorization (code/julia/harness.jl: build_lower_pattern,
# ic0_factorize, ic0_build, IC0_SHIFT_SCHEDULE) so the numbers describe the ACTUAL
# code path used in the main study. Never regenerates data.
#
# 3.6  IC(0) one-iteration convergence — DEMONSTRATE why.  For the SPD families
#      whose PCG(IC0) converged in a single iteration (laplacian_1d, tridiagonal;
#      varied_cond, dense), compute the factorization error ||L Lᵀ - A|| / ||A||
#      and show it is at machine precision, i.e. the zero-fill IC(0) factor EQUALS
#      the exact Cholesky factor for that sparsity structure, so M = A exactly and
#      one PCG iteration is expected — not a leak. Contrasting non-exact families
#      (laplacian_2d/3d, anisotropic) are included to show a nonzero factorization
#      error and the corresponding multi-iteration convergence. The natural-order
#      exact-Cholesky fill ratio (nnz of the dense natural-order Cholesky factor
#      over nnz(tril(A))) is reported for the smaller instances to show zero-fill
#      discards nothing exactly when that ratio is 1.
#      Output: results/benchmarks/ic0_exactness.csv
#
# 3.7  bcsstk14 IC(0) breakdown — document the ACTUAL behavior.  Re-run the harness
#      IC(0) factorization on HB/bcsstk14 (which errored out in the main sweep) and
#      capture: the diagonal magnitude range, ||A||, the Manteuffel shift schedule
#      and increment rule actually used, matrix ordering (natural), scaling (none),
#      PCG tolerance, and the row of the first non-positive pivot for shift = 0 and
#      for every shift in the schedule. Then search for the smallest shift that
#      DOES yield an all-positive factor, to show the schedule's absolute shifts are
#      negligible relative to the diagonal magnitude of this stiffness matrix.
#      Output: results/benchmarks/bcsstk14_ic0_breakdown.csv

using LinearAlgebra
using SparseArrays
using Printf

const HERE = @__DIR__
const CODE_DIR = normpath(joinpath(HERE, ".."))
const DATA_DIR = joinpath(CODE_DIR, "data")
const OUT_DIR = joinpath(CODE_DIR, "results", "benchmarks")

include(joinpath(HERE, "data_layer.jl"))
using .DataLayer: read_mtx, read_npy
include(joinpath(HERE, "harness.jl"))
using .Harness: build_lower_pattern, ic0_factorize, ic0_build, IC0_SHIFT_SCHEDULE

# ---------------------------------------------------------------------------
# 3.6  IC(0) exactness per family
# ---------------------------------------------------------------------------
struct SpdInst
    family::String
    label::String
    dir::String
    observed_pcg_iters::Int   # from the main sweep bench_spd_conv.csv (documentation)
end

const SPD_INSTANCES = SpdInst[
    SpdInst("laplacian_1d_spd", "1-D Laplacian n=500 (tridiagonal)",
            "laplacian_1d_spd/n00000500", 1),
    SpdInst("varied_cond_spd", "dense SPD kappa=1e6 n=200",
            "varied_cond_spd/kappa6e0_n00000200", 1),
    SpdInst("varied_cond_spd", "dense SPD kappa=1e4 n=500",
            "varied_cond_spd/kappa4e0_n00000500", 1),
    SpdInst("laplacian_2d_spd", "2-D Laplacian n=1024 (5-point)",
            "laplacian_2d_spd/m032_n00001024", 35),
    SpdInst("laplacian_3d_spd", "3-D Laplacian n=1728 (7-point)",
            "laplacian_3d_spd/m012_n00001728", 18),
    SpdInst("anisotropic_2d_spd", "anisotropic 2-D eps=0.1 n=576",
            "anisotropic_2d_spd/eps0p1_n00000576", 24),
]

# nnz of the natural-order (no reordering) exact Cholesky factor, computed densely.
# Returns (exact_nnzL, feasible) — feasible=false when n is too large for a dense
# factorization (we then skip the fill ratio for that instance).
function natural_order_chol_nnz(Adense::Matrix{Float64}; drop = 1e-12)
    n = size(Adense, 1)
    C = cholesky(Symmetric(Adense, :L); check = true)
    L = Matrix(C.L)                                  # natural order: cholesky(dense) does NOT permute
    cnt = 0
    @inbounds for j in 1:n, i in j:n
        abs(L[i, j]) > drop && (cnt += 1)
    end
    return cnt
end

function run_ic0_exactness(io)
    for inst in SPD_INSTANCES
        A = read_mtx(joinpath(DATA_DIR, inst.dir, "A.mtx"))
        n = size(A, 1)
        nnzA = nnz(A)
        nnz_tril = count(!iszero, tril(A))
        Afro = norm(A)                               # Frobenius norm of A

        # Hand-written IC(0) factor L (unshifted), exactly as the harness builds it.
        built = ic0_build(A)
        L = built.factor.L
        residual = norm(L * transpose(L) - A) / Afro
        nnzL = nnz(L)
        fill_zero = nnzL / nnz_tril                  # 1.0 by construction (zero fill)

        # Natural-order exact Cholesky fill ratio (only for modest n, dense feasible).
        exact_ratio = NaN
        exact_nnzL = -1
        if n <= 2200
            Adense = Matrix(A)
            exact_nnzL = natural_order_chol_nnz(Adense)
            exact_ratio = nnz_tril / exact_nnzL       # 1.0 => zero-fill loses nothing
        end
        exact = residual < 1e-10

        row = [inst.family, inst.label, inst.dir, string(n), string(nnzA),
               string(nnz_tril), string(nnzL), @sprintf("%.6g", fill_zero),
               (exact_nnzL < 0 ? "NA" : string(exact_nnzL)),
               (isnan(exact_ratio) ? "NA" : @sprintf("%.6g", exact_ratio)),
               @sprintf("%.6e", residual), string(exact),
               string(inst.observed_pcg_iters), string(built.shift),
               string(built.breakdown)]
        println(io, join(row, ","))
        @printf("  [3.6] %-34s n=%5d  ||LLt-A||/||A||=%.3e  exact=%-5s  chol_fill_ratio=%s  pcg_iters=%d\n",
                inst.label, n, residual, string(exact),
                isnan(exact_ratio) ? "NA" : @sprintf("%.3g", exact_ratio),
                inst.observed_pcg_iters)
    end
end

# ---------------------------------------------------------------------------
# 3.7  bcsstk14 IC(0) breakdown documentation
# ---------------------------------------------------------------------------
function run_bcsstk14(io)
    dir = "suitesparse_spd/HB_bcsstk14_n00001806"
    A = read_mtx(joinpath(DATA_DIR, dir, "A.mtx"))
    n = size(A, 1)
    d = diag(A)
    dmin = minimum(abs.(d)); dmax = maximum(abs.(d))
    Afro = norm(A)
    A1 = maximum(sum(abs.(A); dims = 1))             # 1-norm (max abs column sum)
    rowcols, rowvals = build_lower_pattern(A)

    println(io, "quantity,value")
    meta = [
        ("matrix", "HB/bcsstk14"),
        ("n", string(n)),
        ("nnz", string(nnz(A))),
        ("ordering", "natural (rows factored 1..n, no fill-reducing permutation)"),
        ("scaling", "none (raw matrix values)"),
        ("pcg_tolerance", "1e-8 (relative residual)"),
        ("diag_abs_min", @sprintf("%.6e", dmin)),
        ("diag_abs_max", @sprintf("%.6e", dmax)),
        ("A_frobenius_norm", @sprintf("%.6e", Afro)),
        ("A_1norm", @sprintf("%.6e", A1)),
        ("manteuffel_increment_rule", "fixed ABSOLUTE additive schedule alpha*I (not scaled to ||A|| or diag)"),
        ("manteuffel_schedule", join(string.(IC0_SHIFT_SCHEDULE), ";")),
    ]
    for (k, v) in meta
        println(io, join([k, "\"" * string(v) * "\""], ","))
        @printf("  [3.7] %-34s = %s\n", k, v)
    end

    # First non-positive pivot at shift = 0.
    f0 = ic0_factorize(rowcols, rowvals, n; shift = 0.0)
    println(io, join(["bad_pivot_row_shift0_1based", string(f0.bad_pivot)], ","))
    @printf("  [3.7] %-34s = %d (1-based; Python 0-based reports %d)\n",
            "bad_pivot_row_shift0", f0.bad_pivot, f0.bad_pivot - 1)

    # Walk the actual Manteuffel schedule: does each shift succeed? where does it fail?
    for α in IC0_SHIFT_SCHEDULE
        fac = ic0_factorize(rowcols, rowvals, n; shift = α)
        println(io, join(["schedule_shift_$(α)_ok", string(fac.ok)], ","))
        println(io, join(["schedule_shift_$(α)_bad_pivot_1based", string(fac.bad_pivot)], ","))
        @printf("  [3.7] Manteuffel shift alpha=%-8g ok=%-5s bad_pivot=%d\n",
                α, string(fac.ok), fac.bad_pivot)
    end

    # Smallest shift (geometric search) that yields an all-positive factor.
    α_success = NaN
    for e in -3:0.5:9        # 1e-3 ... 1e9
        α = 10.0^e
        fac = ic0_factorize(rowcols, rowvals, n; shift = α)
        if fac.ok
            α_success = α
            break
        end
    end
    println(io, join(["smallest_successful_shift", isnan(α_success) ? "none<=1e9" : @sprintf("%.6g", α_success)], ","))
    if !isnan(α_success)
        println(io, join(["successful_shift_over_diag_max", @sprintf("%.6g", α_success / dmax)], ","))
        println(io, join(["successful_shift_over_Afro", @sprintf("%.6g", α_success / Afro)], ","))
        @printf("  [3.7] %-34s = %.6g  (= %.3g x diag_abs_max)\n",
                "smallest_successful_shift", α_success, α_success / dmax)
    else
        @printf("  [3.7] no successful shift found up to 1e9\n")
    end
end

function main()
    mkpath(OUT_DIR)

    out_ex = joinpath(OUT_DIR, "ic0_exactness.csv")
    open(out_ex, "w") do io
        println(io, join(["family", "label", "dir", "n", "nnz_A", "nnz_tril_A",
                          "nnz_L_ic0", "fill_factor_zero", "nnz_L_exact_chol",
                          "chol_structural_fill_ratio", "factorization_residual",
                          "exact", "observed_pcg_iters", "shift", "breakdown"], ","))
        println("== 3.6 IC(0) exactness per family ==")
        run_ic0_exactness(io)
    end
    println("wrote ", out_ex)

    out_bc = joinpath(OUT_DIR, "bcsstk14_ic0_breakdown.csv")
    open(out_bc, "w") do io
        println("\n== 3.7 bcsstk14 IC(0) breakdown ==")
        run_bcsstk14(io)
    end
    println("wrote ", out_bc)
end

main()
