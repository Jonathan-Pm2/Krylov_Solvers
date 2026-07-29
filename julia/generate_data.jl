# Generate-once data exporter (PROTOCOL.md sections 1.1, 2).
#
# Julia is the single source of truth. Each test instance is generated once, in
# memory, and written to code/data/ as:
#   - A -> Matrix Market  .mtx
#   - b -> IEEE-754        .npy
#   - a manifest.json listing every instance (family, n, seed, params, file
#     paths, shape, nnz, and canonical SHA-256 checksums of A and b).
#
# This is applied to the EXISTING generators only, as a proof of the pipeline:
#   1. laplacian_1d          (1-D Dirichlet Laplacian, SPD)          -> cg_ref.jl
#   2. sparse_block_singular (block-diagonal singular, Drazin index) -> DrazinKrylov.jl
# New matrix families are out of scope for Phase 1 (they are Phase 3).
#
# Run:  julia --project=. code/julia/generate_data.jl
#   (or just: julia code/julia/generate_data.jl)

using Random: MersenneTwister
using SparseArrays
using JSON3

const HERE = @__DIR__
include(joinpath(HERE, "data_layer.jl"))
include(joinpath(HERE, "cg_ref.jl"))
include(joinpath(HERE, "DrazinKrylov.jl"))
using .DataLayer
using .CGRef: laplacian_1d, rhs_deterministic
using .DrazinKrylov: sparse_block_singular

const DATA_DIR = normpath(joinpath(HERE, "..", "data"))

# Deterministic, seeded RHS for a singular instance. Uses a local RNG so the
# global stream is untouched; recorded seed makes it exactly reproducible.
seeded_rhs(n::Int, seed::Int) = randn(MersenneTwister(seed), n)

"""
    export_instance(family, n, seed, params, A, b) -> Dict

Write one instance to disk and return its manifest entry. `n` is the reported
system size (matrix dimension). Family-agnostic: A/b are already built.
"""
function export_instance(family::String, n::Int, seed, params, A, b::Vector{Float64})
    key = "$(family)/n$(lpad(n, 8, '0'))"
    dir = joinpath(DATA_DIR, key)
    mkpath(dir)
    mtx_rel = joinpath(key, "A.mtx")
    npy_rel = joinpath(key, "b.npy")
    write_mtx(joinpath(DATA_DIR, mtx_rel), A)
    write_npy(joinpath(DATA_DIR, npy_rel), b)

    m1, m2 = size(A)
    entry = Dict(
        "family" => family,
        "n" => n,
        "seed" => seed === nothing ? nothing : seed,
        "params" => params,
        "matrix_file" => mtx_rel,
        "rhs_file" => npy_rel,
        "shape" => [m1, m2],
        "nnz" => nnz(issparse(A) ? A : sparse(A)),
        "rhs_len" => length(b),
        "matrix_values_sha256" => matrix_values_sha256(A),
        "rhs_sha256" => vector_sha256(b),
    )
    println("  wrote $key  (shape $(m1)x$(m2), nnz $(entry["nnz"]), |b|=$(length(b)))")
    return entry
end

function main()
    mkpath(DATA_DIR)
    instances = Any[]

    # --- Family 1: 1-D Laplacian SPD, deterministic RHS -------------------
    println("SPD family: laplacian_1d")
    for n in (100, 500)
        A = laplacian_1d(n)
        b = rhs_deterministic(n)
        push!(instances, export_instance("laplacian_1d_spd", n, nothing,
                                         Dict{String,Any}(), A, b))
    end

    # --- Family 2: block-diagonal singular, seeded RHS --------------------
    println("Singular family: sparse_block_singular")
    for (n_core, k) in ((100, 3), (500, 3))
        gamma = 0.5
        seed = 12345
        A, _, _ = sparse_block_singular(n_core, k; gamma = gamma, seed = seed)
        n = n_core + k
        b = seeded_rhs(n, seed)
        push!(instances, export_instance("sparse_block_singular", n, seed,
            Dict("n_core" => n_core, "k" => k, "gamma" => gamma), A, b))
    end

    manifest = Dict(
        "schema_version" => 1,
        "protocol_section" => "2",
        "generator" => "code/julia/generate_data.jl",
        "source_of_truth" => "julia",
        "julia_version" => string(VERSION),
        "generated_utc" => string(round(Int, time())),  # epoch seconds (UTC)
        "matrix_format" => "MatrixMarket coordinate real general (.mtx)",
        "rhs_format" => "NumPy .npy v1.0, <f8 C-order (IEEE-754 little-endian)",
        "checksum" => "sha256 over canonical (col,row)-sorted <i8 row, <i8 col, <f8 val triplets (matrix); raw <f8 bytes (vector)",
        "instances" => instances,
    )

    manifest_path = joinpath(DATA_DIR, "manifest.json")
    open(manifest_path, "w") do io
        JSON3.pretty(io, manifest)
    end
    println("\nwrote $(length(instances)) instances + manifest.json under code/data/")
    return manifest_path
end

main()
