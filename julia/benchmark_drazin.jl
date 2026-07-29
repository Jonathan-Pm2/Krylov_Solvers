# Headline experiment (Julia): on NON-symmetric singular systems with controlled
# Drazin index, plain GMRES (dgmres index=0) converges to the WRONG solution,
# while DGMRES (index=k) recovers the exact Drazin solution A^D b.
#
# Honest metrics reported:
#   res    = ‖A x - b‖ / ‖b‖         (ordinary residual — GMRES minimises THIS)
#   derr   = ‖x - A^D b‖ / ‖A^D b‖   (error vs the true Drazin solution)
# GMRES wins on `res` yet loses on `derr`: it solves a different problem.
#
# Run:  julia code/julia/benchmark_drazin.jl
include("DrazinKrylov.jl")
using .DrazinKrylov
using .DrazinKrylov: drazin_solution_block
using LinearAlgebra, Printf, Random

const SIZES = (1000, 2000, 4000)
const INDICES = (1, 2, 3)
const CAP_M = 400

function run_case(n_core, k)
    A, coreblk, nilblk = sparse_block_singular(n_core, k; seed = 12345)
    n = n_core + k
    b = randn(MersenneTwister(999), n)
    xD = drazin_solution_block(A, b, coreblk, nilblk)
    nrmb = norm(b); nrmxD = max(norm(xD), 1e-30)

    results = Dict{String,Any}()
    for (name, idx) in (("GMRES", 0), ("DGMRES", k))
        t = @elapsed res = dgmres(A, b; index = idx, m = min(n, CAP_M), tol = 1e-10)
        x = res.x
        rr = norm(A * x - b) / nrmb
        de = norm(x - xD) / nrmxD
        results[name] = (t = t, iters = res.iterations, conv = res.converged, res = rr, derr = de)
    end
    return results
end

# --- JIT warmup (compile everything on a tiny problem; NOT timed) ---
let
    A, cb, nb = sparse_block_singular(20, 2; seed = 1)
    b = randn(MersenneTwister(1), 22)
    dgmres(A, b; index = 0, m = 22, tol = 1e-10)
    dgmres(A, b; index = 2, m = 22, tol = 1e-10)
    drazin_solution_block(A, b, cb, nb)
end

open(joinpath(@__DIR__, "..", "results", "drazin_julia.csv"), "w") do io
    println(io, "lang,n,index,method,time_s,iters,converged,rel_residual,drazin_error")
    @printf("%-6s %-7s %-4s %-7s %10s %6s %5s %12s %12s\n",
            "lang", "n", "k", "method", "time_s", "iters", "conv", "rel_res", "drazin_err")
    for n_core in SIZES, k in INDICES
        r = run_case(n_core, k)
        for method in ("GMRES", "DGMRES")
            m = r[method]
            @printf("%-6s %-7d %-4d %-7s %10.4f %6d %5s %12.3e %12.3e\n",
                    "Julia", n_core + k, k, method, m.t, m.iters, m.conv, m.res, m.derr)
            @printf(io, "Julia,%d,%d,%s,%.6f,%d,%s,%.6e,%.6e\n",
                    n_core + k, k, method, m.t, m.iters, m.conv, m.res, m.derr)
        end
    end
end
println("\nwrote code/results/drazin_julia.csv")
