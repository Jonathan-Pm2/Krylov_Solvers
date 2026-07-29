# Restarted DGMRES at scale (Julia). Two things are checked:
#   (1) validity: restarted DGMRES(20) reaches the SAME A^D b as the ground truth;
#   (2) scale: it converges at n up to 50,000 with a fixed 20-column Arnoldi basis
#       (memory bounded independently of the total iteration count).
# restart = 20 (< the ~45 iters full DGMRES needs) forces multiple cycles.
# Run: julia code/julia/benchmark_restart.jl
include("DrazinKrylov.jl")
using .DrazinKrylov
using .DrazinKrylov: drazin_solution_block
using LinearAlgebra, Printf, Random

const SIZES = (10_000, 25_000, 50_000)
const INDICES = (1, 2)
const RESTART = 20

# JIT warmup
let A, cb, nb
    A, cb, nb = sparse_block_singular(50, 2; seed = 1)
    b = randn(MersenneTwister(1), 52)
    dgmres_restarted(A, b; index = 2, restart = 10, tol = 1e-10)
end

open(joinpath(@__DIR__, "..", "results", "restart_julia.csv"), "w") do io
    println(io, "lang,n,index,restart,total_iters,cycles,converged,drazin_error,time_s")
    @printf("%-6s %-7s %-3s %-8s %-11s %-7s %-6s %-13s %-9s\n",
            "lang","n","k","restart","total_iters","cycles","conv","drazin_err","time_s")
    for n_core in SIZES, k in INDICES
        A, coreblk, nilblk = sparse_block_singular(n_core, k; seed = 12345)
        n = n_core + k
        b = randn(MersenneTwister(999), n)
        xD = drazin_solution_block(A, b, coreblk, nilblk)
        t = @elapsed res = dgmres_restarted(A, b; index = k, restart = RESTART, tol = 1e-10)
        derr = norm(res.x - xD) / max(norm(xD), 1e-30)
        cycles = length(res.residual_history)
        @printf("%-6s %-7d %-3d %-8d %-11d %-7d %-6s %-13.3e %-9.3f\n",
                "Julia", n, k, RESTART, res.iterations, cycles, res.converged, derr, t)
        @printf(io, "Julia,%d,%d,%d,%d,%d,%s,%.6e,%.6f\n",
                n, k, RESTART, res.iterations, cycles, res.converged, derr, t)
    end
end
println("\nwrote code/results/restart_julia.csv")
