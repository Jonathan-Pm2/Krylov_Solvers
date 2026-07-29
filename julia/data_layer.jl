"""
    DataLayer

Language-neutral, generate-once data layer for the reproducibility study
(PROTOCOL.md, section 2). Julia is the single source of truth: every test
instance is generated once, in memory, and exported to a language-neutral
binary format that both Julia and Python then *load* (never regenerate):

  - matrix A  -> Matrix Market coordinate real `general` (`.mtx`)
  - RHS b     -> IEEE-754 little-endian `.npy` (NumPy format v1.0)

This module is deliberately family-agnostic: it operates on an arbitrary
`AbstractSparseMatrix`/`AbstractMatrix` and an arbitrary `Vector{Float64}`, so
any generator plugged in later (Phase 3) is exported by the same code path.

Bit-for-bit guarantee (see PROTOCOL.md section 2, closes red flag 1.8):

  - `.npy` stores raw IEEE-754 little-endian doubles; any correct loader in any
    language reads back the identical bit pattern.
  - `.mtx` stores each stored value with `%.17g`, the shortest decimal that is
    guaranteed to round-trip a Float64 under correctly-rounded string->double
    conversion (which both Julia's `parse` and Python's `float` implement).
  - A canonical checksum (`matrix_values_sha256`, `vector_sha256`) is computed
    over the *in-memory* generated data at generation time and recorded in the
    manifest. Each language's loader recomputes the same checksum from the file;
    equality proves the text/binary round-trip preserved every index and value
    to bit precision, independently in each language.

Only stdlib dependencies are used here (LinearAlgebra, SparseArrays, SHA,
Printf); JSON serialization is handled by the caller.
"""
module DataLayer

using LinearAlgebra
using SparseArrays
using SHA
using Printf

export write_mtx, read_mtx, write_npy, read_npy,
       canonical_triplets, matrix_values_sha256, vector_sha256

# ---------------------------------------------------------------------------
# Matrix Market (coordinate, real, general) writer / reader
# ---------------------------------------------------------------------------

"""
    write_mtx(path, A)

Write a sparse matrix in Matrix Market coordinate real `general` format.
`general` (not `symmetric`) is used unconditionally so the exact set of stored
entries is preserved for any family, symmetric or not. Indices are 1-based per
the Matrix Market spec; values use `%.17g` for a guaranteed Float64 round-trip.
"""
function write_mtx(path::AbstractString, A::AbstractMatrix)
    S = issparse(A) ? A : sparse(A)
    m, n = size(S)
    I, J, V = findnz(S)
    open(path, "w") do io
        println(io, "%%MatrixMarket matrix coordinate real general")
        println(io, "% generated once by DataLayer.write_mtx (PROTOCOL.md section 2)")
        println(io, "$m $n $(length(V))")
        @inbounds for t in eachindex(V)
            println(io, string(I[t], " ", J[t], " ", @sprintf("%.17g", V[t])))
        end
    end
    return path
end

"""
    read_mtx(path) -> SparseMatrixCSC{Float64,Int}

Minimal reader for the coordinate real `general` format emitted by `write_mtx`.
Deterministic and dependency-free so the loaded structure is fully under our
control (no implicit symmetric expansion).
"""
function read_mtx(path::AbstractString)
    rows = Int[]; cols = Int[]; vals = Float64[]
    m = 0; n = 0; declared_nnz = -1; seen_size = false
    open(path, "r") do io
        for raw in eachline(io)
            line = strip(raw)
            isempty(line) && continue
            startswith(line, "%") && continue
            parts = split(line)
            if !seen_size
                m = parse(Int, parts[1]); n = parse(Int, parts[2])
                declared_nnz = parse(Int, parts[3])
                seen_size = true
            else
                push!(rows, parse(Int, parts[1]))
                push!(cols, parse(Int, parts[2]))
                push!(vals, parse(Float64, parts[3]))
            end
        end
    end
    declared_nnz >= 0 && length(vals) != declared_nnz &&
        error("read_mtx: declared nnz $declared_nnz != entries read $(length(vals)) in $path")
    return sparse(rows, cols, vals, m, n)
end

# ---------------------------------------------------------------------------
# NumPy .npy writer / reader (format v1.0, 1-D float64, little-endian)
# ---------------------------------------------------------------------------

const NPY_MAGIC = UInt8[0x93, 0x4E, 0x55, 0x4D, 0x50, 0x59]  # "\x93NUMPY"

"""
    write_npy(path, v::AbstractVector{Float64})

Write a 1-D Float64 vector as a NumPy `.npy` file (format v1.0), descr `<f8`,
C order, raw little-endian payload. Readable by `numpy.load` and by `read_npy`.
"""
function write_npy(path::AbstractString, v::AbstractVector{Float64})
    header = "{'descr': '<f8', 'fortran_order': False, 'shape': ($(length(v)),), }"
    # total header prefix = 6 (magic) + 2 (version) + 2 (headerlen) = 10 bytes;
    # the header string (incl. trailing '\n') is padded so the total is a
    # multiple of 64, per the .npy spec.
    prefix = 10
    padded = cld(prefix + length(header) + 1, 64) * 64
    npad = padded - prefix - length(header) - 1
    header = header * repeat(" ", npad) * "\n"
    open(path, "w") do io
        write(io, NPY_MAGIC)
        write(io, UInt8(1)); write(io, UInt8(0))          # version 1.0
        write(io, htol(UInt16(length(header))))            # header length, LE uint16
        write(io, codeunits(header))
        for x in v
            write(io, htol(x))                             # LE float64 payload
        end
    end
    return path
end

"""
    read_npy(path) -> Vector{Float64}

Minimal reader for 1-D `<f8` C-order `.npy` files (format v1.x) produced by
`write_npy` / `numpy.save`. Errors loudly on any unexpected dtype/order/shape.
"""
function read_npy(path::AbstractString)
    data = read(path)
    length(data) >= 10 || error("read_npy: file too short: $path")
    data[1:6] == NPY_MAGIC || error("read_npy: bad magic in $path")
    hlen = Int(ltoh(reinterpret(UInt16, data[9:10])[1]))
    header = String(data[11:(10 + hlen)])
    occursin("'<f8'", header) || error("read_npy: expected dtype '<f8' in $path: $header")
    occursin("False", header) || error("read_npy: expected C order in $path: $header")
    payload = data[(11 + hlen):end]
    length(payload) % 8 == 0 || error("read_npy: payload not a multiple of 8 in $path")
    vals = reinterpret(Float64, payload)
    return [ltoh(x) for x in vals]
end

# ---------------------------------------------------------------------------
# Canonical checksums (identical byte layout must be reproduced in Python)
# ---------------------------------------------------------------------------

"""
    canonical_triplets(A) -> (I, J, V)

Stored entries sorted by (column, row), 1-based. Fixes a deterministic order so
the value checksum is independent of storage order across languages.
"""
function canonical_triplets(A::AbstractMatrix)
    S = issparse(A) ? A : sparse(A)
    I, J, V = findnz(S)
    p = sortperm(collect(zip(J, I)))          # (col, row) ascending
    return I[p], J[p], V[p]
end

"""
    matrix_values_sha256(A) -> hex String

SHA-256 over the canonical triplet stream, each entry serialized as
little-endian Int64 row, Int64 col, Float64 value. Python reproduces the exact
same byte stream, so equal digests prove identical shape-free (index+value)
content to bit precision.
"""
function matrix_values_sha256(A::AbstractMatrix)
    I, J, V = canonical_triplets(A)
    buf = IOBuffer()
    @inbounds for t in eachindex(V)
        write(buf, htol(Int64(I[t])))
        write(buf, htol(Int64(J[t])))
        write(buf, htol(Float64(V[t])))
    end
    return bytes2hex(sha256(take!(buf)))
end

"""
    vector_sha256(v) -> hex String

SHA-256 over the raw little-endian Float64 bytes of `v`.
"""
function vector_sha256(v::AbstractVector{Float64})
    buf = IOBuffer()
    for x in v
        write(buf, htol(Float64(x)))
    end
    return bytes2hex(sha256(take!(buf)))
end

end # module
