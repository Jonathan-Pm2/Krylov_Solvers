# restart_sweep.jl — restarted-DGMRES sensitivity to the restart parameter r
# (PROTOCOL.md 8.2 / 3.12).
#
# Reviewer red flag 3.12 asks how restarted DGMRES behaves as the restart length
# r changes. This sweeps r in {10, 20, 30, 50} with DrazinKrylov.dgmres_restarted
# on representative singular instances that converge (block-diagonal + similarity
# + higher-index families), recording per (instance, r):
#
#   * outer cycles                (length of the per-cycle residual history)
#   * total inner iterations      (sum of inner DGMRES iterations across cycles)
#   * per-cycle a-th residual     ||A^a (b - A x)|| / ||A^a b||  (PROTOCOL.md 8.2)
#   * total solve time            (median of timed reps after a discarded warmup)
#   * memory                      basis_bytes = r*n*8 (bounded Arnoldi basis, the
#                                 point of restarting) and alloc_bytes (@allocated
#                                 churn); process VmHWM is recorded but is a
#                                 process high-water mark, not r-separable within a
#                                 single process, so basis/alloc are the r-signal
#   * converged flag, final Drazin error (vs exported x_star / closed form)
#   * stagnation flag             (residual not decreasing across cycles)
#
# The per-cycle residual trajectory is language-independent (identical algorithm,
# proven identical work counts). Julia produces the full sweep here; a companion
# Python script (restart_sweep_py.py) adds timing for one instance so a
# cross-language timing point exists per PROTOCOL.md 3.12.
#
# Output: code/results/benchmarks/restart_sweep.csv (long / per-cycle, language=julia).

using LinearAlgebra
using SparseArrays
using Printf
using Statistics: median

const HERE = @__DIR__
const CODE_DIR = normpath(joinpath(HERE, ".."))
const DATA_DIR = joinpath(CODE_DIR, "data")
const OUT_DIR = joinpath(CODE_DIR, "results", "benchmarks")

include(joinpath(HERE, "data_layer.jl"))
using .DataLayer: read_mtx, read_npy
include(joinpath(HERE, "DrazinKrylov.jl"))
using .DrazinKrylov: dgmres_restarted, drazin_solution_block

const RESTARTS = [10, 20, 30, 50]
const TOL = 1e-8       # match the main-study DGMRES stop (harness / stability exp.)
const MAXOUTER = 500
const REPS = 3

"Peak RSS (bytes) from /proc/self/status VmHWM; 0 if unavailable."
function vmhwm_bytes()
    isfile("/proc/self/status") || return 0
    for line in eachline("/proc/self/status")
        if startswith(line, "VmHWM:")
            parts = split(line)
            return parse(Int, parts[2]) * 1024
        end
    end
    return 0
end

# ref = :block    -> exact closed-form A^D b via diag(B^{-1},0) (sparse_block family)
# ref = :exported -> manifest-exported x_star = A^D b for the seeded RHS
struct Inst
    label::String
    family::String
    dir::String
    k::Int
    n_core::Int
    ref::Symbol
    seed::Int
end

const INSTANCES = Inst[
    Inst("block-diag n=103 k=3", "sparse_block_singular",
         "sparse_block_singular/n00000103", 3, 100, :block, 0),
    Inst("block-diag n=503 k=3", "sparse_block_singular",
         "sparse_block_singular/n00000503", 3, 500, :block, 0),
    Inst("similarity n=43 k=3", "similarity_singular",
         "similarity_singular/ncore040_k3_n00000043", 3, 40, :exported, 1000),
    Inst("similarity n=44 k=4", "similarity_singular",
         "similarity_singular/ncore040_k4_n00000044", 4, 40, :exported, 1000),
    Inst("blockdiag_high_index n=45 k=5", "blockdiag_high_index_singular",
         "blockdiag_high_index_singular/ncore040_k5_n00000045", 5, 40, :exported, 1000),
]

function load_instance(inst::Inst)
    A = read_mtx(joinpath(DATA_DIR, inst.dir, "A.mtx"))
    if inst.ref == :block
        b = read_npy(joinpath(DATA_DIR, inst.dir, "b.npy"))
        xstar = drazin_solution_block(A, b, 1:inst.n_core, (inst.n_core + 1):size(A, 1))
    else
        b = read_npy(joinpath(DATA_DIR, inst.dir, "b_seed$(inst.seed).npy"))
        xstar = read_npy(joinpath(DATA_DIR, inst.dir, "x_star_seed$(inst.seed).npy"))
    end
    return A, b, xstar
end

function main()
    mkpath(OUT_DIR)
    out = joinpath(OUT_DIR, "restart_sweep.csv")
    cols = ["instance", "family", "n", "k", "language", "restart_r", "cycle",
            "cycle_residual", "outer_cycles", "total_inner_iters", "total_time_s",
            "basis_bytes", "alloc_bytes", "vmhwm_bytes", "converged",
            "final_drazin_err", "stagnation", "seed"]
    open(out, "w") do io
        println(io, join(cols, ","))
        for inst in INSTANCES
            A, b, xstar = load_instance(inst)
            n = size(A, 1)
            nrm_xstar = norm(xstar)
            for r in RESTARTS
                # warmup (discard JIT), then measure once for cycle/mem, time over REPS
                res = dgmres_restarted(A, b; index = inst.k, restart = r,
                                       maxouter = MAXOUTER, tol = TOL)
                allocb = @allocated dgmres_restarted(A, b; index = inst.k, restart = r,
                                                     maxouter = MAXOUTER, tol = TOL)
                times = Float64[]
                for _ in 1:REPS
                    push!(times, @elapsed dgmres_restarted(A, b; index = inst.k,
                                          restart = r, maxouter = MAXOUTER, tol = TOL))
                end
                tmed = median(times)
                vm = vmhwm_bytes()

                reshist = res.residual_history
                ncyc = length(reshist)
                derr = nrm_xstar == 0 ? norm(res.x) : norm(res.x - xstar) / nrm_xstar
                basis_bytes = r * n * 8
                # stagnation: did not converge AND the a-th residual is flat over
                # the tail (< 1% reduction across the last up-to-10 cycles), i.e.
                # extra cycles buy no progress (PROTOCOL.md 3.12 "residual not
                # decreasing across cycles").
                tail = max(1, ncyc - min(9, ncyc - 1))
                stagnation = (!res.converged) && ncyc >= 2 &&
                             (reshist[end] >= 0.99 * reshist[tail])

                for (ci, g) in enumerate(reshist)
                    row = [inst.label, inst.family, string(n), string(inst.k), "julia",
                           string(r), string(ci), @sprintf("%.8g", g),
                           string(ncyc), string(res.iterations),
                           @sprintf("%.6g", tmed), string(basis_bytes),
                           string(allocb), string(vm), string(res.converged),
                           @sprintf("%.8g", derr), string(stagnation), string(inst.seed)]
                    println(io, join(row, ","))
                end
                @printf("  %-30s r=%2d  cycles=%3d  inner=%4d  t=%.4gs  alloc=%.2fMB  derr=%.2e  conv=%s  stag=%s\n",
                        inst.label, r, ncyc, res.iterations, tmed, allocb / 1e6, derr,
                        res.converged, stagnation)
            end
        end
    end
    println("wrote ", out)
end

main()
