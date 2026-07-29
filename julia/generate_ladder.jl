# Phase-4 full-scale n-ladder generator (PROTOCOL.md sections 2, 3.1, 3.2, 3.3, 8.3, 10).
#
# APPEND-ONLY de-risking helper (same contract as generate_preview.jl): it grows
# the manifest with the full-scale scaling ladder needed for the empirical scaling
# fit (section 10) WITHOUT touching any pre-existing instance. It reuses the exact
# DataLayer export code path (write_mtx / write_npy + canonical checksums) so the
# bit-for-bit invariant (section 2) still holds for every new instance, and it
# mirrors the schema-2 manifest entry produced by generate_data_phase3.jl
# (export_extended), including the 20 seeded RHS per instance (section 3.3) and,
# for singular instances, the exact closed-form x_star = A^D b per RHS with the
# inversion-free integrity residual (section 8.3).
#
# Ladder (grid-aligned SPD sizes up to ~50k + a feasible singular ladder):
#   laplacian_2d_spd  : m = 71,100,158,224  -> n = 5041,10000,24964,50176  (5-point)
#   laplacian_3d_spd  : m = 20,30           -> n = 8000,27000              (7-point)
#   anisotropic_2d_spd: eps=0.1, m = 71,100 -> n = 5041,10000
#   varied_cond_spd   : kappa=1e4, n = 500,1000                            (DENSE)
#   similarity_singular k=3: n_core = 997,1997,4997 -> n = 1000,2000,5000  (DENSE)
#
# Feasibility caps (honest, documented — "up to a feasible n", section 3.2 + 10):
#   * varied_cond_spd and similarity_singular are DENSE (nnz = n^2); their sizes
#     are capped well below 50k because storage and the A^k / inv(S) operators are
#     O(n^2)-O(n^3). similarity_singular stops at n=5000 (n>=10000 would need a
#     multi-GB .mtx and O(n^3) A^a operators — intractable for generation AND the
#     later DGMRES solve). The sparse Laplacian families carry the true 50k reach.
#   * drazin_index_detected (rank-scan cross-check) and the pinv cross-check are
#     O(n^3) dense; they are computed only for small n and recorded as null above a
#     cap (the exact constructed index k and the inversion-free residual remain the
#     authoritative ground-truth checks at every size).
#
# Run:  julia code/julia/generate_ladder.jl

using Random: MersenneTwister
using LinearAlgebra
using SparseArrays
using JSON3

const HERE = @__DIR__
include(joinpath(HERE, "data_layer.jl"))
include(joinpath(HERE, "DrazinKrylov.jl"))
include(joinpath(HERE, "families_phase3.jl"))
using .DataLayer
using .DrazinKrylov: drazin_index, drazin_inverse
using .FamiliesPhase3

const DATA_DIR = normpath(joinpath(HERE, "..", "data"))

# Detected-index / pinv cross-check caps (O(n^3) dense operations).
const DETECT_INDEX_NMAX = 600
const PINV_CROSSCHECK_NMAX = 400

# ---------------------------------------------------------------------------
# Extended exporter (append-only mirror of generate_data_phase3.export_extended).
# One A, 20 seeded RHS, optional exact x_star per RHS. Returns (entry, key).
# ---------------------------------------------------------------------------
function export_instance(family::String, tag::String, A, params::Dict;
                         spd::Bool, singular::Bool,
                         kappa = nothing, kappa_method = nothing,
                         x_star_fn = nothing, drazin_k = nothing,
                         provenance = nothing)
    key = "$(family)/$(tag)"
    dir = joinpath(DATA_DIR, key)
    mkpath(dir)
    mtx_rel = joinpath(key, "A.mtx")
    write_mtx(joinpath(DATA_DIR, mtx_rel), A)
    m1, m2 = size(A)
    n = m2

    # Precompute integrity-check operators once for singular instances.
    Ak = nothing; Ak1 = nothing; AD = nothing
    id_res_max = 0.0
    pinv_err = nothing
    if singular
        Am = Matrix(A)
        Ak = Am^drazin_k
        Ak1 = Am^(drazin_k + 1)
        AD = n <= PINV_CROSSCHECK_NMAX ? drazin_inverse(Am) : nothing
    end

    rhs_set = Any[]
    first = nothing
    for seed in RHS_SEEDS
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
        if x_star_fn !== nothing
            x = x_star_fn(b)
            x_rel = joinpath(key, "x_star_seed$(seed).npy")
            write_npy(joinpath(DATA_DIR, x_rel), x)
            entry["x_star_file"] = x_rel
            entry["x_star_sha256"] = vector_sha256(x)
            # inversion-free integrity check: A^{k+1} x* == A^k b
            denom = norm(Ak * b)
            r = denom == 0 ? 0.0 : norm(Ak1 * x - Ak * b) / denom
            id_res_max = max(id_res_max, r)
            if AD !== nothing && seed == RHS_SEEDS[1]
                xn = AD * b
                pinv_err = norm(x - xn) / max(norm(xn), 1e-30)
            end
        end
        push!(rhs_set, entry)
        first === nothing && (first = entry)
    end

    Asp = issparse(A) ? A : sparse(A)
    e = Dict{String,Any}(
        "family" => family,
        "tag" => tag,
        "n" => n,
        "shape" => [m1, m2],
        "nnz" => nnz(Asp),
        "params" => params,
        "matrix_file" => mtx_rel,
        "matrix_values_sha256" => matrix_values_sha256(A),
        "n_rhs" => length(RHS_SEEDS),
        "rhs_seeds" => collect(RHS_SEEDS),
        "rhs_set" => rhs_set,
        "seed" => first["seed"],
        "rhs_file" => first["rhs_file"],
        "rhs_sha256" => first["rhs_sha256"],
        "rhs_len" => n,
        "spd" => spd,
        "singular" => singular,
        "kappa" => kappa,
        "kappa_method" => kappa_method,
        "drazin_index" => drazin_k,
        "provenance" => provenance,
    )
    if singular
        detected = n <= DETECT_INDEX_NMAX ? drazin_index(Matrix(A)) : nothing
        e["drazin_index_detected"] = detected
        e["drazin_identity_residual_max"] = id_res_max
        e["drazin_identity_residual_def"] = "max_over_RHS ||A^(k+1) x_star - A^k b|| / ||A^k b||"
        e["pinv_crosscheck_rel_err"] = pinv_err
        e["pinv_crosscheck_note"] = "independent A^D = A^k (A^(2k+1))^+ A^k cross-check (section 8.3); computed only for n <= $(PINV_CROSSCHECK_NMAX); detected index only for n <= $(DETECT_INDEX_NMAX)"
    end
    xtag = singular ? " k=$(drazin_k) idres=$(round(id_res_max,sigdigits=3))" :
                       (kappa === nothing ? "" : " kappa=$(round(kappa,sigdigits=5))")
    println("  wrote $key  ($(m1)x$(m2), nnz $(e["nnz"]), $(length(RHS_SEEDS)) RHS$(x_star_fn===nothing ? "" : "+x*"))$xtag")
    return e, key
end

function instance_key_of(inst)
    dirname(String(inst.matrix_file))
end

# ---------------------------------------------------------------------------
# Ladder specification (family, generator closures). Each returns nothing when
# its key already exists so the script is idempotent and append-only.
# ---------------------------------------------------------------------------
function main()
    manifest_path = joinpath(DATA_DIR, "manifest.json")
    isfile(manifest_path) || error("manifest.json not found: $manifest_path")
    existing = JSON3.read(read(manifest_path, String))
    instances = Any[inst for inst in existing.instances]
    existing_keys = Set(instance_key_of(inst) for inst in instances)
    n_before = length(instances)
    println("Manifest has $n_before instances; appending full-scale ladder (pre-existing untouched).")

    appended = 0
    function append_if_absent(family, tag, build)
        key = "$(family)/$(tag)"
        if key in existing_keys
            println("  $key already present; skipping (untouched).")
            return
        end
        e, _ = build()
        push!(instances, e)
        push!(existing_keys, key)
        appended += 1
    end

    println("SPD: 2-D Laplacian scaling ladder (5-point Dirichlet) -> ~50k")
    for m in (71, 100, 158, 224)
        n = m^2
        tag = "m$(lpad(m,3,'0'))_n$(lpad(n,8,'0'))"
        append_if_absent("laplacian_2d_spd", tag, () -> begin
            A = laplacian_2d(m)
            k, meth = spd_condition(A; nmax = 2000)   # skip dense eig on large sparse (no densify)
            export_instance("laplacian_2d_spd", tag, A,
                Dict{String,Any}("m" => m, "stencil" => "5-point", "bc" => "Dirichlet");
                spd = true, singular = false, kappa = k, kappa_method = meth)
        end)
    end

    println("SPD: 3-D Laplacian scaling ladder (7-point Dirichlet)")
    for m in (20, 30)
        n = m^3
        tag = "m$(lpad(m,3,'0'))_n$(lpad(n,8,'0'))"
        append_if_absent("laplacian_3d_spd", tag, () -> begin
            A = laplacian_3d(m)
            k, meth = spd_condition(A; nmax = 2000)
            export_instance("laplacian_3d_spd", tag, A,
                Dict{String,Any}("m" => m, "stencil" => "7-point", "bc" => "Dirichlet");
                spd = true, singular = false, kappa = k, kappa_method = meth)
        end)
    end

    println("SPD: anisotropic 2-D diffusion at larger sizes (eps=0.1)")
    for m in (71, 100)
        n = m^2; eps = 0.1
        tag = "eps$(replace(string(eps), "." => "p"))_n$(lpad(n,8,'0'))"
        append_if_absent("anisotropic_2d_spd", tag, () -> begin
            A = anisotropic_2d(m, eps)
            k, meth = spd_condition(A; nmax = 2000)
            export_instance("anisotropic_2d_spd", tag, A,
                Dict{String,Any}("m" => m, "eps" => eps, "operator" => "-(eps d_xx + d_yy)");
                spd = true, singular = false, kappa = k, kappa_method = meth)
        end)
    end

    println("SPD: varied-condition-number family at larger sizes (dense, kappa=1e4)")
    for n in (500, 1000)
        kap = 1.0e4
        tag = "kappa$(round(Int,log10(kap)))e0_n$(lpad(n,8,'0'))"
        append_if_absent("varied_cond_spd", tag, () -> begin
            A, kexact = varied_cond_spd(n, kap)
            export_instance("varied_cond_spd", tag, A,
                Dict{String,Any}("n" => n, "kappa_target" => kap,
                    "construction" => "A = Q diag(lambda) Q^T, lambda logspaced in [1,kappa]");
                spd = true, singular = false, kappa = kexact,
                kappa_method = "controlled: eigenvalues logspaced in [1,kappa]; kappa = lambda_max/lambda_min")
        end)
    end

    println("Singular: non-orthogonal similarity ladder  A' = S diag(B,N) S^{-1} (k=3, dense)")
    for nc in (997, 1997, 4997)
        k = 3
        n = nc + k
        tag = "ncore$(lpad(nc,3,'0'))_k$(k)_n$(lpad(n,8,'0'))"
        append_if_absent("similarity_singular", tag, () -> begin
            A, xfn, meta = similarity_singular(nc, k)
            export_instance("similarity_singular", tag, A, meta;
                spd = false, singular = true, x_star_fn = xfn, drazin_k = k)
        end)
    end

    if appended == 0
        println("Nothing to append; manifest left unchanged ($n_before instances).")
        return manifest_path
    end

    merged = Dict{String,Any}()
    for (kk, vv) in pairs(existing)
        merged[String(kk)] = vv
    end
    merged["instances"] = instances
    merged["phase4_ladder_note"] =
        "appended the full-scale n-ladder for the scaling fit (section 10): laplacian_2d_spd m=71,100,158,224 (n up to 50176); laplacian_3d_spd m=20,30 (n=8000,27000); anisotropic_2d_spd eps=0.1 m=71,100; varied_cond_spd kappa=1e4 n=500,1000 (dense); similarity_singular k=3 n=1000,2000,5000 (dense singular ladder with exact x_star=A^D b). 20 seeded RHS each (seeds 1000-1019). Pre-existing instances untouched."
    open(manifest_path, "w") do io
        JSON3.pretty(io, merged)
    end
    println("\nAppended $appended instance(s); manifest now has $(length(instances)) instances (was $n_before).")
    return manifest_path
end

main()
