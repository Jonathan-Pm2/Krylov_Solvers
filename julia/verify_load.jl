# Julia half of the cross-language bit-for-bit verification (PROTOCOL.md section 2).
#
# Loads every instance listed in code/data/manifest.json from disk (never
# regenerates) and asserts, against the checksums recorded at generation time:
#   - matrix A: shape, nnz, and canonical value checksum match the manifest;
#   - RHS   b: length and raw-bytes checksum match the manifest.
# Any mismatch aborts with a nonzero exit code (fail loudly).
#
# It also writes a small cross-check dump per instance under
# code/data/_verify_julia/<key>/ (the Julia-loaded b as .npy and an info.json
# with the Julia-recomputed shape/nnz/checksum) so the Python verifier can do a
# direct element-wise comparison of b and an independent A-checksum comparison.
#
# Run:  julia code/julia/verify_load.jl

using JSON3
using SparseArrays

const HERE = @__DIR__
include(joinpath(HERE, "data_layer.jl"))
using .DataLayer

const DATA_DIR = normpath(joinpath(HERE, "..", "data"))
const DUMP_DIR = joinpath(DATA_DIR, "_verify_julia")

function main()
    manifest_path = joinpath(DATA_DIR, "manifest.json")
    isfile(manifest_path) || error("manifest not found: $manifest_path (run generate_data.jl first)")
    manifest = JSON3.read(read(manifest_path, String))

    mkpath(DUMP_DIR)
    failures = String[]
    maxbdiff = 0.0

    for inst in manifest.instances
        key = dirname(String(inst.matrix_file))
        A = read_mtx(joinpath(DATA_DIR, String(inst.matrix_file)))
        b = read_npy(joinpath(DATA_DIR, String(inst.rhs_file)))

        # --- assert A against the manifest -------------------------------
        got_shape = [size(A, 1), size(A, 2)]
        got_shape == collect(inst.shape) ||
            push!(failures, "$key: A shape $got_shape != manifest $(collect(inst.shape))")
        nnz(A) == inst.nnz ||
            push!(failures, "$key: A nnz $(nnz(A)) != manifest $(inst.nnz)")
        amv = matrix_values_sha256(A)
        amv == String(inst.matrix_values_sha256) ||
            push!(failures, "$key: A checksum mismatch (loaded $amv != manifest $(inst.matrix_values_sha256))")

        # --- assert b against the manifest -------------------------------
        length(b) == inst.rhs_len ||
            push!(failures, "$key: |b| $(length(b)) != manifest $(inst.rhs_len)")
        bv = vector_sha256(b)
        bv == String(inst.rhs_sha256) ||
            push!(failures, "$key: b checksum mismatch (loaded $bv != manifest $(inst.rhs_sha256))")

        # --- cross-check dump for the Python verifier --------------------
        dkey = joinpath(DUMP_DIR, key)
        mkpath(dkey)
        write_npy(joinpath(dkey, "b_loaded.npy"), b)
        open(joinpath(dkey, "info.json"), "w") do io
            JSON3.write(io, Dict(
                "shape" => got_shape,
                "nnz" => nnz(A),
                "matrix_values_sha256" => amv,
            ))
        end

        # --- Phase-3 extension: full rhs_set + x_star (section 3.3, 8.3) --
        # Instances may carry a `rhs_set` of {seed, rhs_file, rhs_sha256,
        # x_star_file, x_star_sha256}. Verify every RHS and every exported
        # Drazin ground-truth vector, and dump each for the Python verifier.
        if hasproperty(inst, :rhs_set)
            for r in inst.rhs_set
                bi = read_npy(joinpath(DATA_DIR, String(r.rhs_file)))
                vector_sha256(bi) == String(r.rhs_sha256) ||
                    push!(failures, "$key: rhs $(r.rhs_file) checksum mismatch")
                write_npy(joinpath(dkey, "b_seed$(r.seed).npy"), bi)
                if r.x_star_file !== nothing
                    xi = read_npy(joinpath(DATA_DIR, String(r.x_star_file)))
                    vector_sha256(xi) == String(r.x_star_sha256) ||
                        push!(failures, "$key: x_star $(r.x_star_file) checksum mismatch")
                    write_npy(joinpath(dkey, "x_star_seed$(r.seed).npy"), xi)
                end
            end
        end
    end

    if isempty(failures)
        println("Julia verify_load: PASS — $(length(manifest.instances)) instances match manifest to bit precision.")
        println("  cross-check dumps written under code/data/_verify_julia/")
        exit(0)
    else
        println("Julia verify_load: FAIL")
        for f in failures
            println("  - ", f)
        end
        exit(1)
    end
end

main()
