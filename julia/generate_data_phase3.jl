# Phase-3 data exporter (PROTOCOL.md sections 3.1, 3.2, 3.3, 8.3).
#
# Julia is the single source of truth. This script EXTENDS the Phase-1 data
# layer: it generates the broadened matrix families, exports them through the
# SAME DataLayer code path (A -> .mtx, b -> .npy), and APPENDS the new instances
# to code/data/manifest.json. Phase-1 instances are preserved untouched.
#
# New per-instance manifest fields (schema_version 2):
#   - n_rhs, rhs_seeds, rhs_set   : PROTOCOL.md 3.3 (20 seeded RHS per instance)
#   - spd, singular               : family classification
#   - kappa, kappa_method         : SPD condition number (reported, controlled)
#   - drazin_index (k)            : PROTOCOL.md 3.2 / 8.3 (per singular instance)
#   - drazin_index_detected       : numeric rank-based detection (cross-check)
#   - drazin_identity_residual_max: max_over_RHS ||A^{k+1} x* - A^k b|| / ||A^k b||
#                                   (inversion-free integrity check of x_star)
#   - pinv_crosscheck_rel_err     : independent A^D=A^k(A^{2k+1})^+A^k cross-check
#                                   (section 8.3; degrades for ill-conditioned B)
#   - provenance                  : SuiteSparse origin (id, group, URL, timestamp)
# For each singular instance the EXACT closed-form x_star = A^D b is exported as
# an additional .npy per RHS (section 8.3: the ground truth travels with data).
#
# Run:  julia code/julia/generate_data_phase3.jl
#   (SuiteSparse matrices are read from code/data/_suitesparse_raw/, downloaded
#    by the accompanying shell step; if absent the phase still completes and the
#    manifest records suitesparse status "pending".)

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
const SS_RAW = joinpath(DATA_DIR, "_suitesparse_raw")

# Family names owned by Phase 3 (used to refresh idempotently on re-run).
const PHASE3_FAMILIES = Set([
    "laplacian_2d_spd", "laplacian_3d_spd", "anisotropic_2d_spd",
    "varied_cond_spd", "similarity_singular", "coupled_singular",
    "illcond_block_singular", "blockdiag_high_index_singular",
    "suitesparse_spd",
])

# ---------------------------------------------------------------------------
# Generic extended exporter: one A, 20 seeded RHS, optional x_star per RHS.
# ---------------------------------------------------------------------------
function export_extended(family::String, tag::String, A, params::Dict;
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
        # independent pseudoinverse cross-check (section 8.3) — bounded n only
        AD = n <= 400 ? drazin_inverse(Am) : nothing
    end

    rhs_set = Any[]
    first = nothing
    for seed in RHS_SEEDS
        b = seeded_rhs(n, seed)
        b_rel = joinpath(key, "b_seed$(seed).npy")
        write_npy(joinpath(DATA_DIR, b_rel), b)
        b_sha = vector_sha256(b)
        entry = Dict{String,Any}(
            "seed" => seed,
            "rhs_file" => b_rel,
            "rhs_sha256" => b_sha,
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
        # backward-compatible primary-RHS pointers (mirror Phase-1 schema)
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
        e["drazin_index_detected"] = drazin_index(Matrix(A))
        e["drazin_identity_residual_max"] = id_res_max
        e["drazin_identity_residual_def"] = "max_over_RHS ||A^(k+1) x_star - A^k b|| / ||A^k b||"
        e["pinv_crosscheck_rel_err"] = pinv_err
        e["pinv_crosscheck_note"] = "independent A^D = A^k (A^(2k+1))^+ A^k cross-check (section 8.3); expected to degrade for ill-conditioned B"
    end
    xtag = singular ? " k=$(drazin_k) idres=$(round(id_res_max,sigdigits=3))" :
                       (kappa === nothing ? "" : " kappa=$(round(kappa,sigdigits=5))")
    println("  wrote $key  ($(m1)x$(m2), nnz $(e["nnz"]), $(length(RHS_SEEDS)) RHS$(x_star_fn===nothing ? "" : "+x*"))$xtag")
    return e
end

# ---------------------------------------------------------------------------
# SuiteSparse ingestion (section 3.1 real SPD component).
# ---------------------------------------------------------------------------
function load_provenance_txt()
    prov = Dict{String,Any}()
    f = joinpath(SS_RAW, "_provenance.txt")
    isfile(f) || return prov
    for line in eachline(f)
        parts = split(strip(line), "|")
        length(parts) == 5 || continue
        id, grp, name, url, ts = parts
        prov[String(name)] = Dict("matrix_id" => String(id), "group" => String(grp),
            "name" => String(name), "source_url" => String(url), "download_utc" => String(ts))
    end
    return prov
end

# Locate the primary .mtx inside a staged SuiteSparse directory (skip *_coord.mtx
# geometry files that some problems ship alongside the matrix).
function primary_mtx(dir::String, name::String)
    cand = joinpath(dir, "$name.mtx")
    isfile(cand) && return cand
    for f in readdir(dir)
        endswith(f, ".mtx") && !occursin("coord", f) && return joinpath(dir, f)
    end
    return nothing
end

function ingest_suitesparse!(instances)
    result = Dict{String,Any}("status" => "pending: not attempted", "matrices" => Any[])
    if !isdir(SS_RAW)
        result["status"] = "pending: no network (raw download dir absent)"
        return result
    end
    prov = load_provenance_txt()
    subdirs = filter(d -> isdir(joinpath(SS_RAW, d)) && !startswith(d, "_"), readdir(SS_RAW))
    if isempty(subdirs)
        result["status"] = "pending: no network (no matrices downloaded)"
        return result
    end
    included = 0
    for name in sort(subdirs)
        dir = joinpath(SS_RAW, name)
        path = primary_mtx(dir, name)
        rec = Dict{String,Any}("name" => name)
        if path === nothing
            rec["status"] = "skipped: no .mtx found"; push!(result["matrices"], rec); continue
        end
        A, hdr = read_mtx_symmetric(path)
        n = size(A, 1)
        rec["n"] = n; rec["original_stored"] = hdr.stored
        rec["original_format"] = "coordinate $(hdr.field) $(hdr.symmetric ? "symmetric" : "general")"
        # SPD verification: symmetric + sparse Cholesky (CHOLMOD) succeeds.
        # Kept sparse throughout so large matrices (e.g. n=17546) never densify.
        diffmax = maximum(abs.(A - permutedims(A)))
        scale = isempty(nonzeros(A)) ? 1.0 : maximum(abs.(nonzeros(A)))
        sym = diffmax <= 1e-12 * scale
        spd = false
        if sym
            spd = try
                cholesky(Symmetric(A)); true
            catch
                false
            end
        end
        rec["symmetric"] = sym
        rec["spd_verified"] = spd
        if !(sym && spd)
            rec["status"] = "skipped: SPD verification failed (symmetric=$sym, cholesky=$spd)"
            push!(result["matrices"], rec); continue
        end
        pdata = get(prov, name, Dict{String,Any}())
        provenance = merge(Dict{String,Any}(
            "spd_verified" => true,
            "original_nnz_stored" => hdr.stored,
            "original_format" => rec["original_format"],
            "expanded_nnz" => nnz(A),
        ), pdata)
        kappa, kmeth = spd_condition(A; nmax = 5000)
        tag = "$(replace(get(pdata, "matrix_id", name), "/" => "_"))_n$(lpad(n, 8, '0'))"
        e = export_extended("suitesparse_spd", tag, A,
            Dict{String,Any}("matrix_id" => get(pdata, "matrix_id", name),
                             "group" => get(pdata, "group", "")),
            spd = true, singular = false, kappa = kappa, kappa_method = kmeth,
            provenance = provenance)
        push!(instances, e)
        rec["status"] = "included"; rec["expanded_nnz"] = nnz(A)
        rec["kappa"] = kappa
        push!(result["matrices"], rec)
        included += 1
    end
    result["status"] = included > 0 ? "ok: $included matrices verified SPD and included" :
                                       "pending: matrices downloaded but none passed SPD verification"
    return result
end

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
function main()
    isdir(DATA_DIR) || error("data dir missing: $DATA_DIR")
    manifest_path = joinpath(DATA_DIR, "manifest.json")
    isfile(manifest_path) || error("Phase-1 manifest.json not found; run generate_data.jl first")
    existing = JSON3.read(read(manifest_path, String))

    # Preserve every non-Phase-3 (i.e. Phase-1) instance untouched.
    kept = Any[]
    for inst in existing.instances
        String(inst.family) in PHASE3_FAMILIES || push!(kept, inst)
    end
    println("Preserving $(length(kept)) Phase-1 instance(s); (re)generating Phase-3 families.")

    new_inst = Any[]

    println("SPD: 2-D Laplacian (5-point Dirichlet)")
    for m in (20, 32)
        A = laplacian_2d(m); n = m^2
        k, meth = spd_condition(A)
        push!(new_inst, export_extended("laplacian_2d_spd", "m$(lpad(m,3,'0'))_n$(lpad(n,8,'0'))", A,
            Dict{String,Any}("m" => m, "stencil" => "5-point", "bc" => "Dirichlet");
            spd = true, singular = false, kappa = k, kappa_method = meth))
    end

    println("SPD: 3-D Laplacian (7-point Dirichlet)")
    for m in (8, 12)
        A = laplacian_3d(m); n = m^3
        k, meth = spd_condition(A)
        push!(new_inst, export_extended("laplacian_3d_spd", "m$(lpad(m,3,'0'))_n$(lpad(n,8,'0'))", A,
            Dict{String,Any}("m" => m, "stencil" => "7-point", "bc" => "Dirichlet");
            spd = true, singular = false, kappa = k, kappa_method = meth))
    end

    println("SPD: anisotropic 2-D diffusion (varying ratio eps)")
    for eps in (0.1, 0.01)
        m = 24; A = anisotropic_2d(m, eps); n = m^2
        k, meth = spd_condition(A)
        push!(new_inst, export_extended("anisotropic_2d_spd",
            "eps$(replace(string(eps), "." => "p"))_n$(lpad(n,8,'0'))", A,
            Dict{String,Any}("m" => m, "eps" => eps, "operator" => "-(eps d_xx + d_yy)");
            spd = true, singular = false, kappa = k, kappa_method = meth))
    end

    println("SPD: varied-condition-number family (kappa controlled)")
    for kap in (1.0e2, 1.0e4, 1.0e6)
        n = 200; A, kexact = varied_cond_spd(n, kap)
        push!(new_inst, export_extended("varied_cond_spd",
            "kappa$(round(Int,log10(kap)))e0_n$(lpad(n,8,'0'))", A,
            Dict{String,Any}("n" => n, "kappa_target" => kap,
                "construction" => "A = Q diag(lambda) Q^T, lambda logspaced in [1,kappa]");
            spd = true, singular = false, kappa = kexact,
            kappa_method = "controlled: eigenvalues logspaced in [1,kappa]; kappa = lambda_max/lambda_min"))
    end

    println("Singular: non-orthogonal similarity  A' = S diag(B,N) S^{-1}")
    for k in (3, 4)   # k=4 also satisfies the higher-index (>=4) requirement
        nc = 40
        A, xfn, meta = similarity_singular(nc, k)
        push!(new_inst, export_extended("similarity_singular",
            "ncore$(lpad(nc,3,'0'))_k$(k)_n$(lpad(nc+k,8,'0'))", A, meta;
            spd = false, singular = true, x_star_fn = xfn, drazin_k = k))
    end

    println("Singular: off-diagonal coupling  A = [B C; 0 N]")
    for k in (3,)
        nc = 40
        A, xfn, meta = coupled_singular(nc, k)
        push!(new_inst, export_extended("coupled_singular",
            "ncore$(lpad(nc,3,'0'))_k$(k)_n$(lpad(nc+k,8,'0'))", A, meta;
            spd = false, singular = true, x_star_fn = xfn, drazin_k = k))
    end

    println("Singular: ill-conditioned invertible block B")
    for (nc, k, cB) in ((40, 2, 1.0e6),)
        A, xfn, meta = illcond_block_singular(nc, k; cond_B = cB)
        push!(new_inst, export_extended("illcond_block_singular",
            "ncore$(lpad(nc,3,'0'))_k$(k)_n$(lpad(nc+k,8,'0'))", A, meta;
            spd = false, singular = true, x_star_fn = xfn, drazin_k = k))
    end

    println("Singular: higher Drazin index k >= 4 (block-diagonal, k=5)")
    for (nc, k) in ((40, 5),)
        A, xfn, meta = illcond_block_singular(nc, k; cond_B = 10.0)  # well-conditioned B, index 5
        meta["construction"] = "A = diag(B, N), well-conditioned B, nilpotent index k=5; A^D = diag(B^{-1}, 0)"
        push!(new_inst, export_extended("blockdiag_high_index_singular",
            "ncore$(lpad(nc,3,'0'))_k$(k)_n$(lpad(nc+k,8,'0'))", A, meta;
            spd = false, singular = true, x_star_fn = xfn, drazin_k = k))
    end

    println("SuiteSparse: real SPD matrices")
    ss = ingest_suitesparse!(new_inst)
    println("  SuiteSparse status: ", ss["status"])

    instances = vcat(kept, new_inst)
    manifest = Dict{String,Any}(
        "schema_version" => 2,
        "protocol_section" => "2,3,8.3",
        "generator" => "code/julia/generate_data.jl (phase 1) + code/julia/generate_data_phase3.jl (phase 3)",
        "source_of_truth" => "julia",
        "julia_version" => string(VERSION),
        "generated_utc" => existing.generated_utc,
        "phase3_generated_utc" => string(round(Int, time())),
        "matrix_format" => "MatrixMarket coordinate real general (.mtx)",
        "rhs_format" => "NumPy .npy v1.0, <f8 C-order (IEEE-754 little-endian)",
        "checksum" => "sha256 over canonical (col,row)-sorted <i8 row, <i8 col, <f8 val triplets (matrix); raw <f8 bytes (vector)",
        "rhs_seeds" => collect(RHS_SEEDS),
        "rhs_policy" => "PROTOCOL.md 3.3: 20 seeded RHS per Phase-3 instance; singular instances also export x_star = A^D b per RHS (8.3)",
        "suitesparse" => ss,
        "instances" => instances,
    )
    open(manifest_path, "w") do io
        JSON3.pretty(io, manifest)
    end
    println("\nwrote $(length(instances)) instances ($(length(kept)) phase-1 + $(length(new_inst)) phase-3) + manifest.json")
    return manifest_path
end

main()
