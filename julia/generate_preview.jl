# Phase-4 PREVIEW data generator (PROTOCOL.md sections 2, 3.1, 3.3).
#
# De-risking helper: appends a SMALL 2-D Laplacian scaling ladder to the existing
# manifest WITHOUT touching any of the pre-existing instances. It reuses the exact
# DataLayer export code path (write_mtx / write_npy + canonical checksums) so the
# bit-for-bit invariant (section 2) still holds for the new instances.
#
# Ladder (grid-square sizes, n = m^2, ceiling ~10k):
#   m = 32 -> n =  1024   (already in the manifest; left untouched)
#   m = 64 -> n =  4096   (appended here if absent)
#   m = 96 -> n =  9216   (appended here if absent)
#
# RHS: 3 seeded RHS per NEW instance (preview budget, section 3.3 subset).
# Singular preview instances are the EXISTING similarity/coupled/blockdiag ones;
# they are NOT regenerated here.
#
# Run:  julia code/julia/generate_preview.jl

using LinearAlgebra
using SparseArrays
using JSON3

const HERE = @__DIR__
include(joinpath(HERE, "data_layer.jl"))
include(joinpath(HERE, "families_phase3.jl"))
using .DataLayer
using .FamiliesPhase3: laplacian_2d, seeded_rhs, spd_condition

const DATA_DIR = normpath(joinpath(HERE, "..", "data"))
const PREVIEW_RHS_SEEDS = [1000, 1001, 1002]   # 3 seeded RHS for the preview

# Export one SPD instance (matrix + 3 RHS, no x_star) and return the manifest entry.
function export_spd_preview(m::Int)
    A = laplacian_2d(m)
    n = m^2
    tag = "m$(lpad(m,3,'0'))_n$(lpad(n,8,'0'))"
    family = "laplacian_2d_spd"
    key = "$(family)/$(tag)"
    dir = joinpath(DATA_DIR, key)
    mkpath(dir)
    mtx_rel = joinpath(key, "A.mtx")
    write_mtx(joinpath(DATA_DIR, mtx_rel), A)

    rhs_set = Any[]
    first = nothing
    for seed in PREVIEW_RHS_SEEDS
        b = seeded_rhs(n, seed)
        b_rel = joinpath(key, "b_seed$(seed).npy")
        write_npy(joinpath(DATA_DIR, b_rel), b)
        entry = Dict{String,Any}(
            "seed" => seed,
            "rhs_file" => b_rel,
            "rhs_sha256" => vector_sha256(b),
            "x_star_file" => nothing,
            "x_star_sha256" => nothing,
        )
        push!(rhs_set, entry)
        first === nothing && (first = entry)
    end

    # kappa reported only when cheap (skip the dense eigensolve for the ladder).
    kappa, kmeth = spd_condition(A; nmax = 2000)
    e = Dict{String,Any}(
        "family" => family,
        "tag" => tag,
        "n" => n,
        "shape" => [n, n],
        "nnz" => nnz(A),
        "params" => Dict{String,Any}("m" => m, "stencil" => "5-point", "bc" => "Dirichlet"),
        "matrix_file" => mtx_rel,
        "matrix_values_sha256" => matrix_values_sha256(A),
        "n_rhs" => length(PREVIEW_RHS_SEEDS),
        "rhs_seeds" => copy(PREVIEW_RHS_SEEDS),
        "rhs_set" => rhs_set,
        "seed" => first["seed"],
        "rhs_file" => first["rhs_file"],
        "rhs_sha256" => first["rhs_sha256"],
        "rhs_len" => n,
        "spd" => true,
        "singular" => false,
        "kappa" => kappa,
        "kappa_method" => kmeth,
        "drazin_index" => nothing,
        "provenance" => nothing,
    )
    println("  wrote $key  ($(n)x$(n), nnz $(nnz(A)), $(length(PREVIEW_RHS_SEEDS)) RHS)  kappa=$(kappa === nothing ? "skipped" : round(kappa, sigdigits=5))")
    return e, key
end

function instance_key_of(inst)
    dirname(String(inst.matrix_file))
end

function main()
    manifest_path = joinpath(DATA_DIR, "manifest.json")
    isfile(manifest_path) || error("manifest.json not found: $manifest_path")
    existing = JSON3.read(read(manifest_path, String))
    instances = Any[inst for inst in existing.instances]
    existing_keys = Set(instance_key_of(inst) for inst in instances)
    n_before = length(instances)

    println("Manifest has $n_before instances; appending preview SPD ladder (m=64, m=96).")
    appended = 0
    for m in (64, 96)
        n = m^2
        key = "laplacian_2d_spd/m$(lpad(m,3,'0'))_n$(lpad(n,8,'0'))"
        if key in existing_keys
            println("  $key already present; skipping (untouched).")
            continue
        end
        e, _ = export_spd_preview(m)
        push!(instances, e)
        appended += 1
    end

    if appended == 0
        println("Nothing to append; manifest left unchanged ($n_before instances).")
        return manifest_path
    end

    merged = Dict{String,Any}()
    for (k, v) in pairs(existing)
        merged[String(k)] = v
    end
    merged["instances"] = instances
    merged["phase4_preview_note"] =
        "appended a small 2-D Laplacian ladder (m=64,96; 3 RHS each) for the Phase-4 preview; pre-existing instances untouched"
    open(manifest_path, "w") do io
        JSON3.pretty(io, merged)
    end
    println("\nAppended $appended instance(s); manifest now has $(length(instances)) instances (was $n_before).")
    return manifest_path
end

main()
