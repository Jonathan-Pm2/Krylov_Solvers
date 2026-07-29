# Emit the Julia half of the environment record (PROTOCOL.md section 1.1) as a
# single JSON object on stdout. Invoked by code/capture_environment.py; can also
# be run standalone:  julia code/julia/capture_env_julia.jl
#
# Captured: Julia version, BLAS/LAPACK backend + loaded libraries (via
# LinearAlgebra.BLAS.get_config()), configured BLAS thread count, CPU/target
# info, and versions of the study-relevant packages that are actually installed
# (degrades gracefully to null when a package or API is unavailable).

using JSON3
using LinearAlgebra
import Pkg

function blas_info()
    info = Dict{String,Any}()
    try
        cfg = LinearAlgebra.BLAS.get_config()
        libs = Any[]
        for lib in cfg.loaded_libs
            push!(libs, Dict(
                "libname" => String(basename(lib.libname)),
                "interface" => string(lib.interface),
            ))
        end
        info["loaded_libs"] = libs
        info["summary"] = replace(string(cfg), r"\s+" => " ")
    catch e
        info["error"] = string(e)
    end
    try
        info["num_threads"] = LinearAlgebra.BLAS.get_num_threads()
    catch e
        info["num_threads"] = nothing
    end
    return info
end

function package_versions()
    wanted = ["IterativeSolvers", "Krylov", "IncompleteLU", "Preconditioners",
              "BenchmarkTools", "JSON3", "SparseArrays", "LinearAlgebra"]
    out = Dict{String,Any}()
    try
        deps = Pkg.dependencies()
        byname = Dict(info.name => info for (_, info) in deps)
        for w in wanted
            out[w] = haskey(byname, w) && byname[w].version !== nothing ?
                     string(byname[w].version) : nothing
        end
    catch e
        out["error"] = string(e)
    end
    return out
end

rec = Dict(
    "julia_version" => string(VERSION),
    "julia_base_image" => Base.julia_cmd()[1],
    "word_size" => Sys.WORD_SIZE,
    "cpu_target" => Sys.CPU_NAME,
    "machine" => Sys.MACHINE,
    "julia_threads" => Threads.nthreads(),
    "blas" => blas_info(),
    "packages" => package_versions(),
)

JSON3.write(stdout, rec)
