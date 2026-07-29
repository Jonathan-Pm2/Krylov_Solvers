# deepen_krylov_budget.jl — Round-2 deepening experiments 1.13 and 1.14.
#
# Two DGMRES "deepening" diagnostics, both reusing the manifest singular instances
# and the exported closed-form ground truth x_star = A^D b (PROTOCOL.md 8.3). This
# script never touches the timed harness path and never regenerates data.
#
# 1.13  Krylov-budget sweep.  On the dense similarity instance (n=1000, k=3) that
#       hit the Krylov cap in the main sweep (m=120, did NOT converge, Drazin error
#       ~1.06), re-run DGMRES with a growing Krylov cap m in {120,200,400,800} and
#       record: converged?, iterations, the index-a residual (the operative stop
#       test, 8.2), and the final Drazin error vs x_star. GOAL: show the Drazin
#       error drops as m grows, i.e. the dense-case failure is a BUDGET limit, not
#       an instability. If it does NOT recover, that is reported honestly.
#       Output: results/benchmarks/krylov_budget_sweep.csv
#
# 1.14  Ill-conditioned block (illcond_block_singular, k=2, cond(B) ~ 1e6).  Pin
#       down WHICH component fails. For the DGMRES solution x at the harness
#       settings (m=120, tol=1e-8) report: the forward error, the ordinary residual
#       ||b-Ax||/||b||, the index-a residual (the stop test), a normwise backward
#       error (Rigal-Gaches), and an EMPIRICAL condition number of the Drazin
#       solution map obtained by perturbing the invertible block B and the RHS by a
#       small relative epsilon and measuring the relative change in the exact A^D b.
#       This separates inherent conditioning of A^D b from a stopping-criterion
#       quality issue.
#       Output: results/benchmarks/illcond_diag.csv
#
# Diagnostics are language-independent (the DGMRES arithmetic is byte-identical
# across Julia and Python; the work-count identity is already established), so they
# are computed once, in Julia.

using LinearAlgebra
using SparseArrays
using Printf
using Random: MersenneTwister

const HERE = @__DIR__
const CODE_DIR = normpath(joinpath(HERE, ".."))
const DATA_DIR = joinpath(CODE_DIR, "data")
const OUT_DIR = joinpath(CODE_DIR, "results", "benchmarks")

include(joinpath(HERE, "data_layer.jl"))
using .DataLayer: read_mtx, read_npy
include(joinpath(HERE, "DrazinKrylov.jl"))
using .DrazinKrylov: dgmres, drazin_inverse

const SEED = 1000
const TOL = 1e-8

# A^p * v  (dense result), matching DrazinKrylov._apply_power semantics.
function apply_power(A, M::AbstractVecOrMat, p::Int)
    R = Matrix{Float64}(reshape(M, size(M, 1), :))
    for _ in 1:p
        R = A * R
    end
    return size(M, 2) == 0 ? R : R
end

# index-a residual  ||A^a (b - A x)|| / ||A^a b||   (PROTOCOL.md 8.2)
function index_a_residual(A, b, x, a)
    r = b - A * x
    num = norm(apply_power(A, reshape(r, :, 1), a)[:, 1])
    den = norm(apply_power(A, reshape(b, :, 1), a)[:, 1])
    return den == 0 ? num : num / den
end

# ---------------------------------------------------------------------------
# 1.13  Krylov-budget sweep
# ---------------------------------------------------------------------------
struct BudgetInst
    label::String
    dir::String
    k::Int
end

const BUDGET_INSTANCES = BudgetInst[
    BudgetInst("similarity n=1000 k=3 (dense)", "similarity_singular/ncore997_k3_n00001000", 3),
]
# The four requested caps plus a full-Krylov recovery point (m = n-1). For a dense
# operator, DGMRES needs a Krylov dimension approaching the invertible-block size
# (n_core = 997 here) for full recovery, so the m = n-1 point demonstrates the
# solution IS recovered once the budget is large enough (the decisive budget-limit
# evidence); the intermediate caps show the monotone approach.
const BUDGET_M = [120, 200, 400, 800]

function run_budget_sweep(io)
    for inst in BUDGET_INSTANCES
        A = read_mtx(joinpath(DATA_DIR, inst.dir, "A.mtx"))
        b = read_npy(joinpath(DATA_DIR, inst.dir, "b_seed$(SEED).npy"))
        xstar = read_npy(joinpath(DATA_DIR, inst.dir, "x_star_seed$(SEED).npy"))
        n = size(A, 1)
        nrm_xstar = norm(xstar)
        ms = vcat(BUDGET_M, [n - 1])          # add full-Krylov recovery point
        for m in ms
            mcap = min(m, n - 1)
            t = @elapsed res = dgmres(A, b; index = inst.k, m = mcap, tol = TOL)
            idxres = index_a_residual(A, b, res.x, inst.k)
            derr = nrm_xstar == 0 ? norm(res.x) : norm(res.x - xstar) / nrm_xstar
            row = [inst.label, inst.dir, string(n), string(inst.k), string(m),
                   string(mcap), string(res.iterations), string(res.converged),
                   @sprintf("%.8g", idxres), @sprintf("%.8g", derr),
                   @sprintf("%.6g", t), string(SEED)]
            println(io, join(row, ","))
            flush(io)
            @printf("  [1.13] %-30s m=%3d (cap %3d) iters=%3d conv=%-5s idx_a_res=%.3e derr=%.3e  (%.2fs)\n",
                    inst.label, m, mcap, res.iterations, string(res.converged), idxres, derr, t)
            flush(stdout)
        end
    end
end

# ---------------------------------------------------------------------------
# 1.14  Ill-conditioned block diagnostics
# ---------------------------------------------------------------------------
# Empirical condition number of the Drazin solution map A^D b, measured on the
# index-preserving manifold: A = diag(B, N) with A^D b = [B^{-1} b_core; 0]. We
# perturb B and b_core by a small relative epsilon (keeping the nilpotent block N
# fixed, so ind(A) stays 2 and A^D b remains well defined) and measure the
# amplification  (||x_pert - x_star|| / ||x_star||) / eps_rel.  The maximum over
# random directions estimates kappa(A^D b). This isolates INHERENT conditioning
# from the DGMRES stopping test.
function run_illcond(io)
    dir = "illcond_block_singular/ncore040_k2_n00000042"
    k = 2
    n_core = 40
    A = read_mtx(joinpath(DATA_DIR, dir, "A.mtx"))
    b = read_npy(joinpath(DATA_DIR, dir, "b_seed$(SEED).npy"))
    xstar = read_npy(joinpath(DATA_DIR, dir, "x_star_seed$(SEED).npy"))
    n = size(A, 1)
    Adense = Matrix(A)
    core = 1:n_core
    B = Adense[core, core]
    bcore = b[core]
    condB = cond(B)

    # DGMRES at the harness settings (m=120, tol=1e-8)
    res = dgmres(A, b; index = k, m = min(120, n - 1), tol = TOL)
    x = res.x
    nrm_xstar = norm(xstar)
    forward_err = norm(x - xstar) / nrm_xstar
    ordinary_res = norm(b - A * x) / norm(b)
    idx_a_res = index_a_residual(A, b, x, k)
    # normwise (Rigal-Gaches) backward error for the ordinary system A x = b
    Anorm = opnorm(Adense, 2)
    backward_err = norm(b - A * x) / (Anorm * norm(x) + norm(b))

    # empirical condition number of A^D b (index-preserving perturbation of B, b_core)
    Bnorm = opnorm(B, 2)
    bcnorm = norm(bcore)
    xcore_star = B \ bcore                      # exact B^{-1} b_core (the nonzero part of A^D b)
    nrm_core = norm(xcore_star)
    eps_rel = 1e-8
    ntrial = 40
    rng = MersenneTwister(20260728)
    kappa_emp = 0.0
    for _ in 1:ntrial
        E = randn(rng, n_core, n_core)
        E .*= (eps_rel * Bnorm / opnorm(E, 2))   # ||E||_2 = eps_rel * ||B||_2
        f = randn(rng, n_core)
        f .*= (eps_rel * bcnorm / norm(f))        # ||f||_2 = eps_rel * ||b_core||_2
        xp = (B + E) \ (bcore + f)
        rel_change = norm(xp - xcore_star) / nrm_core
        amp = rel_change / eps_rel
        kappa_emp = max(kappa_emp, amp)
    end

    # cross-check: independent pinv-identity Drazin inverse (section 8.3 cross-check)
    xstar_pinv = drazin_inverse(Adense) * b
    pinv_vs_exported = norm(xstar_pinv - xstar) / nrm_xstar

    rows = [
        ("cond_B_true", condB),
        ("dgmres_iters", Float64(res.iterations)),
        ("dgmres_converged", res.converged ? 1.0 : 0.0),
        ("forward_error_rel", forward_err),
        ("ordinary_residual_rel", ordinary_res),
        ("index_a_residual_rel", idx_a_res),
        ("backward_error_normwise", backward_err),
        ("empirical_cond_ADb", kappa_emp),
        ("eps_rel_perturbation", eps_rel),
        ("expected_fwd_err_kappa_times_stopres", kappa_emp * idx_a_res),
        ("pinv_crosscheck_rel_err", pinv_vs_exported),
    ]
    for (name, val) in rows
        println(io, join([name, @sprintf("%.10g", val)], ","))
        @printf("  [1.14] %-40s = %.6g\n", name, val)
    end
    return rows
end

function main()
    mkpath(OUT_DIR)

    out_budget = joinpath(OUT_DIR, "krylov_budget_sweep.csv")
    open(out_budget, "w") do io
        println(io, join(["instance", "dir", "n", "k", "krylov_m_requested",
                          "krylov_m_used", "iterations", "converged",
                          "index_a_residual", "drazin_error", "solve_time_s", "seed"], ","))
        println("== 1.13 Krylov-budget sweep ==")
        run_budget_sweep(io)
    end
    println("wrote ", out_budget)

    out_ill = joinpath(OUT_DIR, "illcond_diag.csv")
    open(out_ill, "w") do io
        println(io, "quantity,value")
        println("\n== 1.14 Ill-conditioned block diagnostics ==")
        run_illcond(io)
    end
    println("wrote ", out_ill)
end

main()
