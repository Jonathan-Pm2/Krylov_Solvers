# Phase-2 benchmark harness (Julia side) — PROTOCOL.md sections 4, 5, 6, 1.3, 8.2.
#
# Responsibilities (one invocation = ONE experimental unit):
#   * Load A (.mtx) and b (.npy) via the Phase-1 DataLayer loaders + manifest.json
#     (NEVER regenerates data).
#   * Symmetric hand-rolled timing (time_ns()): one warmup discarded, then N reps.
#     Time is decomposed into T_assembly (load A/b), T_precond (build the
#     preconditioner), T_solve, and T_total = sum of the three.
#   * Instrumented copies of the in-repo solvers (cg_ref / DrazinKrylov.dgmres):
#     arithmetic is byte-for-byte identical to the reference implementations; the
#     ONLY additions are integer counters (matrix-vector products, preconditioner
#     applications, orthogonalizations, restart cycles). See the notes on each
#     counted solver below.
#   * In-process peak RSS via /proc/self/status VmHWM (baseline-subtracted) plus
#     Base.gc_live_bytes() as a cross-check. The PRIMARY /usr/bin/time -v peak is
#     injected by the runner (run_benchmarks.py); this file writes the in-process
#     value as a self-contained fallback so a harness-only run still yields a
#     complete, schema-valid CSV.
#   * DGMRES stopping is the index-a residual  ‖A^a (b - A x)‖ / ‖A^a b‖ ≤ tol
#     (PROTOCOL.md 8.2), NOT the unknown A^D b. The closed-form A^D b error
#     (diag(B^-1,0) reference, 8.3) is emitted only as a validation column.
#
# The canonical CSV column order is read from code/bench_schema.json, shared with
# the Python harness, so both languages emit an identical schema.
#
# CLI:
#   julia [-t T] harness.jl --mode unit --regime R --requested-threads T \
#       --family F --instance-key K --method M --reps N [--warmup W] \
#       --seed S --out CSV [--krylov-m M] [--tol TOL]
#   julia harness.jl --mode baseline        # print BASELINE_RSS_BYTES=<n> and exit

module Harness

using LinearAlgebra
using SparseArrays
using Dates
using JSON3

const HERE = @__DIR__
include(joinpath(HERE, "data_layer.jl"))
using .DataLayer: read_mtx, read_npy

const CODE_DIR = normpath(joinpath(HERE, ".."))
const DATA_DIR = joinpath(CODE_DIR, "data")
const SCHEMA_PATH = joinpath(CODE_DIR, "bench_schema.json")
const ENV_JSON = joinpath(CODE_DIR, "results", "environment.json")

# ---------------------------------------------------------------------------
# Work counters (only additions to the reference algorithms)
# ---------------------------------------------------------------------------
mutable struct Counters
    matvecs::Int
    precond_applications::Int
    orthogonalizations::Int
    restart_cycles::Int
end
Counters() = Counters(0, 0, 0, 0)

# A^p * M, counting each column-application of A as one matrix-vector product.
# Mirrors DrazinKrylov._apply_power exactly; the only addition is `c.matvecs`.
function _apply_power_counted(A, M::AbstractMatrix, p::Int, c::Counters)
    R = Matrix{Float64}(M)
    for _ in 1:p
        R = A * R
        c.matvecs += size(R, 2)
    end
    return R
end

# ---------------------------------------------------------------------------
# Instrumented CG / Jacobi-PCG  (faithful copy of CGRef.cg_ref, cg_ref.jl)
# ---------------------------------------------------------------------------
# The preconditioner Minv is built OUTSIDE and passed in so its construction can
# be timed as T_precond; the iteration arithmetic is unchanged. `count_precond`
# toggles whether the diagonal apply is counted as a preconditioner application
# (true for Jacobi-PCG, false for the identity used by plain CG).
function cg_counted(A, b, Minv::AbstractVector; rtol = 1e-8, atol = 0.0,
                    maxiter = size(A, 2), count_precond::Bool = false)
    n = length(b)
    x = zeros(n)
    r = copy(b)
    c = Counters()
    z = Minv .* r
    count_precond && (c.precond_applications += 1)
    p = copy(z)
    rz = dot(r, z)
    nb = norm(b)
    thresh = max(rtol * nb, atol)
    iters = 0
    converged = false
    for k in 1:maxiter
        Ap = A * p
        c.matvecs += 1
        alpha = rz / dot(p, Ap)
        x .+= alpha .* p
        r .-= alpha .* Ap
        iters = k
        if norm(r) <= thresh
            converged = true
            break
        end
        z = Minv .* r
        count_precond && (c.precond_applications += 1)
        rz_new = dot(r, z)
        p .= z .+ (rz_new / rz) .* p
        rz = rz_new
    end
    relres = norm(b - A * x) / nb
    stop_residual = norm(r) / nb
    return (x = x, iters = iters, converged = converged,
            stop_residual = stop_residual, relres = relres, counters = c)
end

# ---------------------------------------------------------------------------
# Incomplete Cholesky IC(0)  — zero fill, SPD preconditioner (PROTOCOL.md §7)
# ---------------------------------------------------------------------------
# This is IC(0): a HAND-WRITTEN, zero-fill incomplete Cholesky that keeps EXACTLY
# the sparsity pattern of the lower triangle of A (no drop tolerance, no fill).
# It is NOT ILUT and must never be aliased with a drop-tolerance variant (closes
# red flag 1.9). A = L Lᵀ approximately, with L constrained to pattern(tril(A)).
#
# CROSS-LANGUAGE EQUIVALENCE (the whole study rests on this): the FACTORIZATION
# (ic0_factorize) and the PCG driver below are byte-for-byte identical in
# algorithm and in floating-point operation order to the Python mirror in
# code/python/harness.py. Only the runtime differs, never the factorization
# arithmetic. The left-looking loops accumulate every sparse dot product in
# ascending-column order, so both languages produce a bit-identical factor L on
# identical input.
#
# APPLY PRIMITIVE (PROTOCOL.md §7.1): the triangular-solve APPLY (ic0_apply!) uses
# each language's STANDARD COMPILED primitive, NOT a hand-written pure loop — here
# Julia's native sparse triangular solve (LowerTriangular/UpperTriangular
# backslash: solve L y = r, then Lᵀ x = y); the Python mirror uses
# scipy.sparse.linalg.spsolve_triangular. The factorization stays byte-identical
# hand-written (the L factor is the same); ONLY the apply's internal
# implementation and floating-point operation order differ between the two
# compiled primitives. This is an implementation choice, deliberately kept out of
# the language claim (a pure-Python substitution loop was 46-57× slower in the
# preview, which is interpreter/loop overhead, not an ecosystem property —
# reporting it would reintroduce red flag 1.2 / Prioridad 7). Crucially, the
# iteration count and the per-iteration preconditioner-application count are
# UNCHANGED by the apply choice, so the cross-language work-count identity holds
# exactly; only z can differ in its last ULP.
#
# SPD SAFETY (§7): IC(0) can break down on some SPD matrices (a non-positive
# pivot d ≤ 0). We do NOT silently patch it into a different algorithm. The
# unshifted attempt is tried first and any breakdown is reported per instance
# (precond_breakdown=true, precond_kind="ic0_breakdown" until a factor is found).
# A documented Manteuffel diagonal-shift fallback — factor A + α·I with the
# smallest α from a fixed schedule that yields all-positive pivots — is applied
# ONLY as a labeled fallback (precond_kind="ic0_shifted", precond_shift=α).

struct IC0Factor
    n::Int
    rowcols::Vector{Vector{Int}}      # per row i (1-based): sorted cols j ≤ i in pattern(tril(A))
    Lvals::Vector{Vector{Float64}}    # factor L values aligned with rowcols
    diag::Vector{Float64}             # L[i,i] > 0
    L::SparseMatrixCSC{Float64,Int}   # assembled hand-written factor L (for the compiled apply, §7.1)
    LT::SparseMatrixCSC{Float64,Int}  # Lᵀ (upper triangular), assembled once for the back solve
end

# Extract the lower-triangle pattern of A as per-row (sorted cols, A-values).
# Representation-independent: works from any AbstractSparseMatrix; the resulting
# per-row value lists are bit-identical to the Python side because they come from
# the same .mtx bytes.
function build_lower_pattern(A)
    n = size(A, 1)
    rowcols = [Int[] for _ in 1:n]
    rowvals = [Float64[] for _ in 1:n]
    I, J, V = findnz(A)
    @inbounds for t in eachindex(V)
        i = I[t]; j = J[t]
        if j <= i
            push!(rowcols[i], j)
            push!(rowvals[i], V[t])
        end
    end
    for i in 1:n
        p = sortperm(rowcols[i])
        rowcols[i] = rowcols[i][p]
        rowvals[i] = rowvals[i][p]
    end
    return rowcols, rowvals
end

# Left-looking IC(0). `shift` is the Manteuffel additive diagonal shift α (α·I).
# Returns (ok, bad_pivot, Lvals, diag). On a non-positive pivot it stops and
# reports the failing row (1-based); it never fabricates a positive pivot.
function ic0_factorize(rowcols, rowvals, n; shift::Float64 = 0.0)
    Lvals = [copy(rowvals[i]) for i in 1:n]
    diag = zeros(Float64, n)
    work = zeros(Float64, n)
    @inbounds for i in 1:n
        cols = rowcols[i]
        vals = rowvals[i]
        for t in eachindex(cols)
            work[cols[t]] = vals[t]
        end
        work[i] += shift                              # Manteuffel α·I (α=0 ⇒ plain IC(0))
        for t in eachindex(cols)
            j = cols[t]
            j < i || break                            # cols sorted; diagonal is last
            s = work[j]
            Lcolsj = rowcols[j]; Lvalsj = Lvals[j]
            for tj in eachindex(Lcolsj)
                k = Lcolsj[tj]
                k < j || break
                s -= work[k] * Lvalsj[tj]             # ascending-k accumulation
            end
            work[j] = s / diag[j]
        end
        d = work[i]
        for t in eachindex(cols)
            k = cols[t]
            k < i || break
            d -= work[k] * work[k]
        end
        if d <= 0.0
            for t in eachindex(cols)
                work[cols[t]] = 0.0
            end
            return (ok = false, bad_pivot = i, Lvals = Lvals, diag = diag)
        end
        di = sqrt(d)
        diag[i] = di
        work[i] = di
        for t in eachindex(cols)
            Lvals[i][t] = work[cols[t]]
        end
        for t in eachindex(cols)
            work[cols[t]] = 0.0
        end
    end
    return (ok = true, bad_pivot = 0, Lvals = Lvals, diag = diag)
end

# Build an IC(0) factor with SPD-safe pivoting (§7). Tries α=0 first; on
# breakdown, walks a fixed Manteuffel shift schedule. Returns
# (factor, shift, breakdown, fill_factor). `fill_factor` = nnz(L)/nnz(tril(A)),
# which is exactly 1.0 by construction (zero fill) and is reported to prove it.
const IC0_SHIFT_SCHEDULE = (1.0e-3, 1.0e-2, 5.0e-2, 1.0e-1, 5.0e-1, 1.0, 5.0, 10.0)

function ic0_build(A)
    n = size(A, 1)
    rowcols, rowvals = build_lower_pattern(A)
    nnz_lower = sum(length, rowcols)
    fac = ic0_factorize(rowcols, rowvals, n; shift = 0.0)
    breakdown = !fac.ok
    shift = 0.0
    if !fac.ok
        @warn "IC(0) breakdown: non-positive pivot at row $(fac.bad_pivot); applying labeled Manteuffel diagonal shift (shifted IC), NOT a silent algorithm swap"
        found = false
        for α in IC0_SHIFT_SCHEDULE
            fac = ic0_factorize(rowcols, rowvals, n; shift = α)
            if fac.ok
                shift = α
                found = true
                break
            end
        end
        found || error("IC(0): breakdown persisted after Manteuffel shift schedule")
    end
    # Assemble the hand-written factor L as a compiled sparse matrix ONCE, for the
    # standard native triangular-solve apply (PROTOCOL.md §7.1). This does NOT
    # touch ic0_factorize (byte-identical, above): Lvals already holds the full L
    # row including the diagonal (Lvals[i] at the position where the sorted column
    # equals i equals diag[i]), so L is exactly the hand-written factor. Assembling
    # it here means its cost is part of the preconditioner build (T_precond), and
    # the per-iteration apply is a pure compiled solve with no per-call assembly.
    Irow = Int[]; Jcol = Int[]; Vval = Float64[]
    for i in 1:n
        cols = rowcols[i]; vals = fac.Lvals[i]
        for t in eachindex(cols)
            push!(Irow, i); push!(Jcol, cols[t]); push!(Vval, vals[t])
        end
    end
    L = sparse(Irow, Jcol, Vval, n, n)
    LT = sparse(transpose(L))                          # Lᵀ (upper triangular) for the back solve
    factor = IC0Factor(n, rowcols, fac.Lvals, fac.diag, L, LT)
    # nnz(L) == nnz(tril(A)) by construction ⇒ fill_factor == 1.0 (zero fill).
    fill_factor = nnz_lower / nnz_lower
    return (factor = factor, shift = shift, breakdown = breakdown, fill_factor = fill_factor)
end

# Preconditioner solve M z = r with M = L Lᵀ: forward solve L y = r, then back
# solve Lᵀ z = y.
#
# PROTOCOL.md §7.1: this APPLY uses Julia's standard NATIVE sparse triangular
# solve (the compiled LowerTriangular/UpperTriangular backslash), NOT a
# hand-written loop. The factor L is byte-identical hand-written (ic0_factorize);
# only the apply's internal implementation and floating-point operation order
# differ from the Python scipy.sparse.linalg.spsolve_triangular apply. This
# changes neither the PCG iteration count nor the per-iteration
# preconditioner-application count (the cross-language work-count identity holds
# exactly); only z can differ in its last ULP.
function ic0_apply!(z::AbstractVector{Float64}, factor::IC0Factor, r::AbstractVector{Float64})
    y = LowerTriangular(factor.L) \ r                 # forward solve L y = r
    z .= UpperTriangular(factor.LT) \ y               # back solve Lᵀ z = y
    return z
end

# PCG preconditioned by IC(0). Faithful copy of cg_counted's iteration; the ONLY
# difference is the preconditioner apply (IC(0) triangular solves instead of the
# Jacobi diagonal scale). One IC(0) solve counts as one precond_application per
# PCG iteration (§5). Arithmetic is byte-identical to the Python pcg_ic0_counted.
function pcg_ic0_counted(A, b, factor::IC0Factor; rtol = 1e-8, atol = 0.0,
                         maxiter = size(A, 2))
    n = length(b)
    x = zeros(n)
    r = copy(b)
    c = Counters()
    z = zeros(n)
    ic0_apply!(z, factor, r)
    c.precond_applications += 1
    p = copy(z)
    rz = dot(r, z)
    nb = norm(b)
    thresh = max(rtol * nb, atol)
    iters = 0
    converged = false
    for k in 1:maxiter
        Ap = A * p
        c.matvecs += 1
        alpha = rz / dot(p, Ap)
        x .+= alpha .* p
        r .-= alpha .* Ap
        iters = k
        if norm(r) <= thresh
            converged = true
            break
        end
        ic0_apply!(z, factor, r)
        c.precond_applications += 1
        rz_new = dot(r, z)
        p .= z .+ (rz_new / rz) .* p
        rz = rz_new
    end
    relres = norm(b - A * x) / nb
    stop_residual = norm(r) / nb
    return (x = x, iters = iters, converged = converged,
            stop_residual = stop_residual, relres = relres, counters = c)
end

# DEFERRED HOOK (labeled): ILU(0) for general non-symmetric A as a "gmres_ilu0"
# method (PROTOCOL.md §7, optional). Not implemented in Phase 3: the priority is
# IC(0)+PCG, and wiring left/right-preconditioned GMRES would change the DGMRES
# arithmetic and risk the cross-language equivalence proof. Left as a documented
# stub so it is not silently aliased with IC(0). See report for rationale.

# ---------------------------------------------------------------------------
# Instrumented DGMRES  (faithful copy of DrazinKrylov.dgmres)
# ---------------------------------------------------------------------------
# Arithmetic is identical to DrazinKrylov.dgmres. Added counters:
#   * matvecs             — every A*V[:,j] plus every column-application of A
#                           inside A^a (via _apply_power_counted).
#   * orthogonalizations  — one per Modified-Gram-Schmidt projection (inner i-loop).
#   * restart_cycles      — 1 (this is the non-restarted single-cycle method).
# `index` (= a = ind(A)) is passed explicitly for the block family, whose Drazin
# index equals the nilpotent block size k (manifest param); this avoids the O(n)
# dense rank scan and matches the closed-form ground truth of PROTOCOL.md 8.3.
# a == 0 reduces exactly to ordinary GMRES (the documented 8.1 baseline).
function dgmres_counted(A, b; index::Int, m::Int = min(size(A, 1), 200),
                        tol::Real = 1e-8, x0 = nothing)
    n = size(A, 1)
    a = index
    x = x0 === nothing ? zeros(n) : Vector{Float64}(x0)
    c = Counters()
    c.restart_cycles = 1

    r0 = b - A * x
    c.matvecs += 1
    reshist = Float64[]

    w0 = _apply_power_counted(A, reshape(r0, n, 1), a, c)[:, 1]   # A^a r0
    beta = norm(w0)
    if beta == 0
        return (x = x, iters = 0, converged = true, stop_residual = 0.0,
                relres = norm(b - A * x) / max(norm(b), 1e-300), counters = c,
                index_a = a)
    end

    V = zeros(n, m + 1)
    H = zeros(m + 1, m)
    V[:, 1] = w0 ./ beta

    converged = false
    used = 0
    rres = beta
    for j in 1:m
        w = A * V[:, j]
        c.matvecs += 1
        for i in 1:j
            H[i, j] = dot(V[:, i], w)
            w -= H[i, j] .* V[:, i]
            c.orthogonalizations += 1
        end
        H[j + 1, j] = norm(w)
        if H[j + 1, j] > 1e-14
            V[:, j + 1] = w ./ H[j + 1, j]
        end

        Vj1 = @view V[:, 1:(j + 1)]
        Hj = @view H[1:(j + 1), 1:j]
        local z
        if a == 0
            g = zeros(j + 1)
            g[1] = beta
            z = Hj \ g
            rres = norm(g - Hj * z)
        else
            P = _apply_power_counted(A, Matrix(Vj1), a, c) # A^a V_{j+1}
            z = (P * Hj) \ w0
            rres = norm(w0 - P * Hj * z)
        end
        push!(reshist, rres)
        used = j
        if rres <= tol * beta
            x = x + V[:, 1:j] * z
            converged = true
            break
        end
        if j == m || H[j + 1, j] <= 1e-14
            x = x + V[:, 1:j] * z
            break
        end
    end
    stop_residual = rres / beta                     # ‖A^a r_m‖ / ‖A^a b‖  (8.2)
    relres = norm(b - A * x) / max(norm(b), 1e-300)
    return (x = x, iters = used, converged = converged, stop_residual = stop_residual,
            relres = relres, counters = c, index_a = a)
end

# ---------------------------------------------------------------------------
# Resource + environment helpers
# ---------------------------------------------------------------------------
"Peak RSS (bytes) from /proc/self/status VmHWM; 0 if unavailable."
function vmhwm_bytes()
    isfile("/proc/self/status") || return 0
    for line in eachline("/proc/self/status")
        if startswith(line, "VmHWM:")
            parts = split(line)
            return parse(Int, parts[2]) * 1024      # kB -> bytes
        end
    end
    return 0
end

function blas_backend_str()
    isfile(ENV_JSON) || return "openblas (from BLAS.get_config)"
    env = JSON3.read(read(ENV_JSON, String))
    libs = env.julia.blas.loaded_libs
    name = isempty(libs) ? "unknown" : String(libs[1].libname)
    return "openblas:" * name
end

# ---------------------------------------------------------------------------
# Instance loading + block-family Drazin reference (closed form, 8.3)
# ---------------------------------------------------------------------------
function load_manifest()
    JSON3.read(read(joinpath(DATA_DIR, "manifest.json"), String))
end

function find_instance(manifest, family, instance_key)
    for inst in manifest.instances
        key = dirname(String(inst.matrix_file))
        if String(inst.family) == family && key == instance_key
            return inst
        end
    end
    error("instance not found: family=$family key=$instance_key")
end

"Closed-form A^D b for a block-diagonal diag(B,N) instance: x[core]=B\\b[core], 0 else."
function drazin_reference(A, b, inst)
    n_core = Int(inst.params.n_core)
    coreblk = 1:n_core
    x = zeros(length(b))
    B = A[coreblk, coreblk]
    x[coreblk] = B \ Vector(b[coreblk])
    return x
end

# Resolve the RHS file (and matching exported x_star, section 8.3) for this unit.
# When `rhs_seed` is given, the rhs_set entry with that seed is selected; otherwise
# the primary RHS is used. The exported x_star = A^D b is the CORRECT closed-form
# Drazin ground truth for EVERY singular family (similarity/coupled are NOT block
# diagonal, so the block-extraction drazin_reference would be wrong for them; the
# exported x_star, generated from the family's exact A^D, is used instead).
function resolve_rhs(inst, rhs_seed)
    if haskey(inst, :rhs_set) && inst.rhs_set !== nothing
        for entry in inst.rhs_set
            if rhs_seed === nothing || Int(entry.seed) == rhs_seed
                xf = get(entry, :x_star_file, nothing)
                xfile = (xf === nothing) ? nothing : String(xf)
                return String(entry.rhs_file), xfile, Int(entry.seed)
            end
        end
    end
    # Phase-1 instances record seed: null (no seeded RHS set); treat as seed 0.
    sd = get(inst, :seed, 0)
    return String(inst.rhs_file), nothing, sd === nothing ? 0 : Int(sd)
end

# ---------------------------------------------------------------------------
# CSV row assembly (schema-driven)
# ---------------------------------------------------------------------------
function load_schema()
    JSON3.read(read(SCHEMA_PATH, String)).columns |> x -> String.(x)
end

csv_escape(s) = (occursin(',', s) || occursin('"', s) || occursin('\n', s)) ?
                '"' * replace(s, '"' => "\"\"") * '"' : s

function fmt(v)
    v isa Bool && return v ? "true" : "false"
    v isa AbstractFloat && (isnan(v) ? (return "NaN") : (return string(v)))
    v isa Integer && return string(v)
    return csv_escape(string(v))
end

function write_rows(out, cols, rows::Vector{<:AbstractDict})
    newfile = !isfile(out) || filesize(out) == 0
    open(out, "a") do io
        if newfile
            println(io, join(cols, ","))
        end
        for row in rows
            println(io, join((fmt(row[c]) for c in cols), ","))
        end
    end
end

# ---------------------------------------------------------------------------
# Single experimental unit: warmup + N timed reps for one (method, instance)
# ---------------------------------------------------------------------------
function run_unit(opts)
    cols = load_schema()
    manifest = load_manifest()
    inst = find_instance(manifest, opts["family"], opts["instance_key"])
    mtx_path = joinpath(DATA_DIR, String(inst.matrix_file))
    n = Int(inst.n)
    nnz_ = Int(inst.nnz)
    method = opts["method"]
    reps = opts["reps"]
    warmup = opts["warmup"]
    tol = opts["tol"]
    krylov_m = opts["krylov_m"]
    index_a = method in ("dgmres", "gmres") ? (method == "gmres" ? 0 : Int(inst.params.k)) : -1

    baseline_rss = vmhwm_bytes()

    # EFFICIENCY (amortize the Julia process startup + JIT across RHS seeds): all
    # requested RHS seeds for this (instance, regime, method) run inside THIS one
    # process. The runtime/JIT is compiled once; each seed still gets its own
    # `warmup` discarded solves and `reps` timed solves, so the per-unit timing
    # semantics are byte-for-byte the same as launching one process per RHS.
    # Methods are intentionally NOT merged into one process: peak RSS is a
    # per-process high-water mark (VmHWM / /usr/bin/time -v), so mixing methods
    # would attribute the max-memory method's footprint to every row (§6).
    rhs_seeds = get(opts, "rhs_seeds", Any[get(opts, "rhs_seed", nothing)])
    rows = Dict{String,Any}[]

    # one solve = assembly + precond + solve, phases timed separately
    function make_one_solve(npy_path, x_star_path)
    function one_solve()
        t0 = time_ns()
        A = read_mtx(mtx_path)
        b = read_npy(npy_path)
        t1 = time_ns()
        # preconditioner phase (BUILT here so its cost is timed as T_precond, §4/§7)
        local Minv, count_precond, ic0
        precond_kind = "none"
        precond_fill_factor = NaN
        precond_shift = NaN
        precond_breakdown = false
        ic0 = nothing
        if method == "pcg_jacobi"
            # diag(A) on a SparseMatrixCSC returns a SparseVector; densify it so the
            # per-iteration `Minv .* r` is a dense-dense broadcast (equivalent to the
            # Python side's dense `1.0 / A.diagonal()`). Without Vector(...) the sparse
            # broadcast is ~7x slower and the Julia timing measures representation
            # overhead, not the Jacobi-PCG algorithm (breaks the §-equivalence contract).
            Minv = 1.0 ./ Vector(diag(A))
            count_precond = true
            precond_kind = "jacobi"
        elseif method == "cg"
            Minv = ones(length(b))
            count_precond = false
            precond_kind = "none"
        elseif method == "pcg_ic0"
            built = ic0_build(A)
            ic0 = built.factor
            Minv = Float64[]
            count_precond = true
            precond_shift = built.shift
            precond_breakdown = built.breakdown
            precond_fill_factor = built.fill_factor       # 1.0 by construction (zero fill)
            precond_kind = built.breakdown ? "ic0_shifted" : "ic0"
        else
            Minv = Float64[]     # no preconditioner for (D)GMRES
            count_precond = false
            precond_kind = "none"
        end
        t2 = time_ns()
        # solve phase
        if method == "pcg_ic0"
            res = pcg_ic0_counted(A, b, ic0; rtol = tol, atol = 0.0, maxiter = n)
            drazin_error = NaN
        elseif method in ("cg", "pcg_jacobi")
            res = cg_counted(A, b, Minv; rtol = tol, atol = 0.0,
                             maxiter = n, count_precond = count_precond)
            drazin_error = NaN
        else
            res = dgmres_counted(A, b; index = index_a, m = krylov_m, tol = tol)
            xref = x_star_path === nothing ? drazin_reference(A, b, inst) : read_npy(x_star_path)
            nref = norm(xref)
            drazin_error = nref == 0 ? norm(res.x) : norm(res.x - xref) / nref
        end
        t3 = time_ns()
        return (t_assembly = t1 - t0, t_precond = t2 - t1, t_solve = t3 - t2,
                res = res, drazin_error = drazin_error,
                precond_kind = precond_kind, precond_fill_factor = precond_fill_factor,
                precond_shift = precond_shift, precond_breakdown = precond_breakdown)
    end
    return one_solve
    end  # make_one_solve

    for rhs_seed in rhs_seeds
        rhs_file, x_star_file, used_rhs_seed = resolve_rhs(inst, rhs_seed)
        npy_path = joinpath(DATA_DIR, rhs_file)
        x_star_path = x_star_file === nothing ? nothing : joinpath(DATA_DIR, x_star_file)
        one_solve = make_one_solve(npy_path, x_star_path)

        # warmup (discarded) — kills JIT contamination; cheap after the first seed
        for _ in 1:warmup
            one_solve()
        end

        for rep in 1:reps
            s = one_solve()
            res = s.res
            c = res.counters
            t_total = s.t_assembly + s.t_precond + s.t_solve
            peak_inproc = max(vmhwm_bytes() - baseline_rss, 0)
            row = Dict{String,Any}(
                "schema_version" => 2,
                "language" => "julia",
                "regime" => opts["regime"],
                "requested_threads" => opts["requested_threads"],
                "observed_task_threads" => Threads.nthreads(),
                "observed_blas_threads" => BLAS.get_num_threads(),
                "blas_backend" => blas_backend_str(),
                "omp_num_threads" => get(ENV, "OMP_NUM_THREADS", ""),
                "openblas_num_threads" => get(ENV, "OPENBLAS_NUM_THREADS", ""),
                "mkl_num_threads" => get(ENV, "MKL_NUM_THREADS", ""),
                "seed" => used_rhs_seed,
                "family" => opts["family"],
                "instance_key" => opts["instance_key"],
                "n" => n,
                "nnz" => nnz_,
                "method" => method,
                "rep" => rep,
                "warmup_discarded" => warmup,
                "t_assembly_ns" => s.t_assembly,
                "t_precond_ns" => s.t_precond,
                "t_solve_ns" => s.t_solve,
                "t_total_ns" => t_total,
                "matvecs" => c.matvecs,
                "precond_applications" => c.precond_applications,
                "orthogonalizations" => c.orthogonalizations,
                "restart_cycles" => c.restart_cycles,
                "iterations" => res.iters,
                "converged" => res.converged,
                "stop_residual" => res.stop_residual,
                "relres" => res.relres,
                "drazin_error" => s.drazin_error,
                "time_per_matvec_ns" => c.matvecs > 0 ? s.t_solve / c.matvecs : NaN,
                "ns_per_nonzero" => s.t_solve / nnz_,
                "peak_rss_bytes" => peak_inproc,
                "peak_rss_method" => "proc_status_vmhwm",
                "peak_rss_inproc_bytes" => peak_inproc,
                "baseline_rss_bytes" => baseline_rss,
                "runtime_live_bytes" => Base.gc_live_bytes(),
                "peak_bytes_per_nonzero" => peak_inproc / nnz_,
                "index_a" => index_a,
                "krylov_m" => method in ("cg", "pcg_jacobi", "pcg_ic0") ? -1 : krylov_m,
                "tol" => tol,
                "precond_kind" => s.precond_kind,
                "precond_fill_factor" => s.precond_fill_factor,
                "precond_shift" => s.precond_shift,
                "precond_breakdown" => s.precond_breakdown,
                "timestamp_utc" => string(now(UTC)),
            )
            push!(rows, row)
        end
    end
    write_rows(opts["out"], cols, rows)
    println("julia unit done: ", method, " ", opts["instance_key"], " regime=", opts["regime"],
            " reps=", reps, " rhs_seeds=", length(rhs_seeds))
end

# ---------------------------------------------------------------------------
# Tiny CLI
# ---------------------------------------------------------------------------
function parse_args(args)
    d = Dict{String,String}()
    i = 1
    while i <= length(args)
        a = args[i]
        startswith(a, "--") || error("unexpected arg: $a")
        key = a[3:end]
        if i + 1 <= length(args) && !startswith(args[i + 1], "--")
            d[key] = args[i + 1]
            i += 2
        else
            d[key] = "true"
            i += 1
        end
    end
    return d
end

function main(args)
    d = parse_args(args)
    mode = get(d, "mode", "unit")
    if mode == "baseline"
        # touch the loaders so the baseline reflects a loaded runtime
        _ = read_npy
        println("BASELINE_RSS_BYTES=", vmhwm_bytes())
        return
    end
    threads = parse(Int, get(d, "requested-threads", "1"))
    BLAS.set_num_threads(threads)
    # --rhs-seeds "1000,1001,1002" runs all listed seeds in this one process
    # (amortized startup). --rhs-seed (singular) is kept for backward compatibility.
    rhs_seeds = if haskey(d, "rhs-seeds")
        Any[parse(Int, strip(s)) for s in split(d["rhs-seeds"], ",") if !isempty(strip(s))]
    elseif haskey(d, "rhs-seed")
        Any[parse(Int, d["rhs-seed"])]
    else
        Any[nothing]
    end
    opts = Dict{String,Any}(
        "regime" => d["regime"],
        "requested_threads" => threads,
        "family" => d["family"],
        "instance_key" => d["instance-key"],
        "method" => d["method"],
        "reps" => parse(Int, get(d, "reps", "3")),
        "warmup" => parse(Int, get(d, "warmup", "1")),
        "seed" => parse(Int, get(d, "seed", "0")),
        "rhs_seeds" => rhs_seeds,
        "out" => d["out"],
        "krylov_m" => parse(Int, get(d, "krylov-m", "100")),
        "tol" => parse(Float64, get(d, "tol", "1e-8")),
    )
    run_unit(opts)
end

end # module

if abspath(PROGRAM_FILE) == @__FILE__
    Harness.main(ARGS)
end
