# Experiment B (Julia): reference CG on the 1-D Laplacian, matched criterion.
# Run: julia code/julia/benchmark_spd.jl
include("cg_ref.jl")
using .CGRef
using Printf

const SIZES = (1000, 2000, 5000)
const RTOL = 1e-8

# JIT warmup (not timed)
let A = laplacian_1d(50), b = rhs_deterministic(50)
    cg_ref(A, b); cg_ref(A, b; jacobi = true)
end

open(joinpath(@__DIR__, "..", "results", "spd_julia.csv"), "w") do io
    println(io, "lang,n,kappa,solver,iters,converged,relres,time_s")
    @printf("%-7s%-7s%10s%-14s%7s%6s%11s%9s\n",
            "lang", "n", "kappa", "solver", "iters", "conv", "relres", "time_s")
    for n in SIZES
        A = laplacian_1d(n)
        b = rhs_deterministic(n)
        kappa = (2 - 2cos(n * pi / (n + 1))) / (2 - 2cos(pi / (n + 1)))
        for (name, jac) in (("CG", false), ("CG-Jacobi", true))
            t = @elapsed (x, iters, conv, relres) =
                cg_ref(A, b; rtol = RTOL, maxiter = n, jacobi = jac)
            @printf("%-7s%-7d%10.2e%-14s%7d%6s%11.2e%9.3f\n",
                    "Julia", n, kappa, name, iters, conv, relres, t)
            @printf(io, "Julia,%d,%.4e,%s,%d,%s,%.4e,%.6f\n",
                    n, kappa, name, iters, conv, relres, t)
        end
    end
end
println("\nwrote code/results/spd_julia.csv")
