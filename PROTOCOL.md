# Experimental Protocol — Full Redesign

This document governs the complete experimental rebuild of the study in response
to the reviewer's evaluation (`Evaluación.pdf`, verdict: *reject and resubmit as a
substantially redesigned study*). No benchmark code, table, or figure is produced
outside the rules below. Every rule is traced to the red flag(s) it closes.

## Research question (single, unified)

> To what extent do **equivalent** implementations of Krylov subspace methods —
> including DGMRES for singular systems — exhibit **reproducible** differences in
> time, memory, and scalability between Julia and Python under strictly controlled
> experimental conditions?

Consequences: GMRES **and** DGMRES go through the *same* benchmark protocol as CG/PCG.
The paper is one study, not a language benchmark stapled to a DGMRES validation.
Closes: **1.6, 2.1, 2.6, 2.7** (structure, title, overclaimed novelty).

## Traceability matrix (red flag → protocol section)

| Red flag | Issue | Closed by |
|---|---|---|
| 1.1 | Table 1 holds data the manuscript itself invalidates | §11 (keep table, replace data) |
| 1.2 | Julia/Python parallelism not controlled | §1.3 |
| 1.3 | Timing not comparable (JIT, ILU cost, criteria) | §4 |
| 1.4, 3.1, 3.2, 3.6 | No repetitions / variability / experimental unit | §4, §3.4 |
| 1.5 | Software versions vague | §1.1 |
| 1.7 | Memory/CPU from `htop` | §6 |
| 1.8 | "bit-for-bit" not proven | §2 |
| 1.9 | Preconditioner definition inconsistent | §7 |
| 1.10, 1.11 | Arnoldi/Lanczos on residual curves | §9 |
| 1.12 | SPD family too limited (1D Laplacian only) | §3.1 |
| 1.13 | "fixed density" wrong | §3.1 note + LaTeX phase |
| 1.14 | Singular matrices too favorable | §3.2 |
| 1.15, 1.16 | GMRES–DGMRES objectives not equivalent; stopping criterion | §8 |
| 1.17 | "ground truth" not truly independent | §8.3 |
| 3.4 | No empirical scaling model | §10 |
| 3.5, 3.7 | No normalized metrics; iteration counts not comparable | §5 |
| 3.8 | Inconsistent `n` set vs tables | §3, §11 |
| 3.9, 3.10 | `O(n)` / max-iter claims too categorical | LaTeX phase |
| 2.2, 2.8 | "rigorous" / conclusions exceed evidence | §4, §10, LaTeX phase |

## 1. Environment & reproducibility

### 1.1 Exact versions (record, never ranges) — closes 1.5
Capture and pin in the paper and in `results/environment.json`:
- Julia version (current machine: **1.12.5**), Python version (**3.13.12**).
- NumPy, SciPy, IterativeSolvers.jl / Krylov.jl, IncompleteLU.jl / Preconditioners.jl, LinearAlgebra stdlib.
- BLAS/LAPACK backend and version (OpenBLAS vs MKL) for **both** languages — `LinearAlgebra.BLAS.get_config()` (Julia), `numpy.show_config()` / `threadpoolctl` (Python).
- Compiler, OS/kernel, CPU model, git commit of the code.

### 1.2 CPU description — closes 1.2 (hardware honesty)
Report the true topology: i7-13700 is **hybrid** (P-cores + E-cores). State P/E core counts, base/boost clocks, and thread affinity used. Never report "16 cores" while a table says "24 threads".

### 1.3 Thread-control scenarios — closes 1.2
Every timing runs under **two** regimes, reported separately:
1. **Single-thread controlled**: `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1`, Julia `-t 1`, `BLAS.set_num_threads(1)`.
2. **Matched multi-thread**: identical physical-thread count in both languages, fixed CPU affinity, same BLAS backend, same thread count reported.

> Environment controls requiring **user action** (cannot be set from application code): disable turbo/frequency scaling (`cpupower frequency-set`), pin cores / isolate CPUs (`taskset`, `isolcpus`), disable SMT if desired. These are documented as prerequisites; the harness records the state it observes.

## 2. Data layer — closes 1.8
- Generate every matrix `A` and RHS `b` **once**, in one language, and export to a language-neutral binary format: **Matrix Market** (`.mtx`) for `A`, **IEEE-754 `.npy`** for `b`.
- Both languages **load** the same files. Neither regenerates.
- Verification script asserts, per instance: identical `A` (same nnz, indices, values) and `max_i |b_i^{(Julia)} − b_i^{(Python)}| = 0`. Fail loudly otherwise.

## 3. Matrix families

### 3.1 SPD family — closes 1.12, 1.13
Not only the 1-D Laplacian. Include, at minimum:
- 1-D, **2-D, and 3-D** Laplacians (Dirichlet).
- At least two **SuiteSparse Matrix Collection** SPD problems.
- A parameterized family spanning **condition numbers** across orders of magnitude.
- Varied sparsity **patterns / bandwidths**, incl. an anisotropic-diffusion problem.

Terminology fix (1.13): the 1-D Laplacian does **not** have "fixed density"; density `(3n−2)/n² ~ 3/n → 0`. What is constant is **nnz per row**. Use "fixed nnz per row" everywhere.

### 3.2 Singular family — closes 1.14
Beyond the favorable block-diagonal `A = diag(B, N)`:
- Block-diagonal transformed by a **non-orthogonal similarity** `S A S⁻¹` (couples the subspaces, destroys the trivial split).
- Off-diagonal coupling between the invertible and nilpotent parts.
- **Ill-conditioned** invertible block `B`.
- **Higher Drazin index** (k up to ≥ 4).
- At least one operator from a real **descriptor / DAE** system if feasible.

### 3.3 Multiple RHS — closes 3.1, 3.2
Per matrix and size, use **20–50** distinct seeded RHS vectors `b`. The experimental unit is one `(language, method, matrix, size, RHS-seed, repetition)` tuple.

## 4. Timing protocol — closes 1.3, 1.4, 3.6
- **Warm-up** run, discarded (kills JIT contamination in Julia; page/cache warm-up in Python).
- **N = 30** timed repetitions per experimental unit.
- Julia: `BenchmarkTools.jl` (`@benchmark`, report median). Python: `time.perf_counter_ns`, manual loop, control GC (`gc.disable()` around the timed region, documented).
- **Decompose** total time and report all three plus total:
  `T_total = T_assembly + T_precond + T_solve`  — closes the "ILU cost excluded" artifact.
- **Randomize** method/order execution across repetitions — closes 3.3.
- Report **median, IQR, and 95% bootstrap CI** — never a single wall-clock number.

## 5. Work / iteration metrics — closes 3.5, 3.7
Raw "iterations" are not comparable across methods/languages (inner vs restart-cycle vs callback). Report hardware-independent work counts:
- **matrix–vector products**, preconditioner applications, orthogonalizations, restart cycles.
- Normalized: **time per matvec**, time per iteration, ns per nonzero, peak bytes per nonzero.

## 6. Resource measurement — closes 1.7
Replace `htop`. Use instrumented, reproducible tools:
- Peak **RSS per process**: `/usr/bin/time -v` wrapping each run; cross-check with `psutil` (Python) and `/proc/self/status` VmHWM + `@allocated`/`Base.gc_live_bytes()` (Julia).
- **Subtract the interpreter/runtime baseline** RSS so the number reflects the solver, not the language startup.
- State sampling method and that RSS (not VMS/system/cache) is reported.

## 7. Preconditioner definition — closes 1.9
State exactly, identically across languages:
- Algorithm: **ILU(0)** (zero fill) vs **ILUT** (drop tolerance) — pick one and name it; do not alias them.
- For SPD: use **Incomplete Cholesky** (preserves symmetry/positive-definiteness, required for PCG) — and verify the preconditioner is SPD before use.
- Report fill factor, drop tolerance, pivoting, orientation.

### 7.1 Apply primitive — decided after the Phase-4 preview
The IC(0) **factorization** is hand-written and byte-identical in both languages (equivalence holds at the algorithm level; iteration and preconditioner-application counts stay identical). The triangular-solve **apply** uses each language's standard compiled primitive (Python `scipy` triangular solve; Julia native sparse triangular solve), **not** a hand-written pure-Python loop.

Rationale: a pure-Python substitution loop is 46–57× slower than the compiled Julia loop (measured in the preview), but that gap is interpreter/loop overhead, **not** an ecosystem property — reporting it as "Python is slower" would reintroduce exactly red flag 1.2 / Prioridad 7 (attributing to the language what is an implementation choice). Only the apply's internal implementation/FP order differs; this is stated explicitly in the methods section.

## 8. DGMRES specifics

### 8.1 Objective parity — closes 1.15
GMRES minimizes `‖Ax−b‖`; DGMRES targets `A^D b`. The valid claim is *"DGMRES recovers the Drazin solution; ordinary GMRES need not"* — **not** "DGMRES is faster". Any time comparison is qualified: for the `A^D b` criterion, GMRES is not a competing method.

### 8.2 Operative stopping criterion — closes 1.16
`A^D b` is unknown in practice, so it cannot be the stopping test. Stop on the **index-`a` residual** `‖A^a(b − Ax_m)‖ / ‖A^a b‖ ≤ tol`, with `a = ind(A)`. Document: how `a` is known/estimated, behavior under wrong `a`, cost and error amplification of forming `A^a r`, roundoff sensitivity.

### 8.3 Ground truth — closes 1.17
A numerical pseudoinverse `A^D = A^a (A^{2a+1})^+ A^a` is **not exact** (depends on SVD/rank tolerance). For the block families the natural exact reference is `A^D = diag(B⁻¹, 0)`. Use the closed form as ground truth; use the pseudoinverse identity only as an independent cross-check, and say so.

### 8.4 Stability diagnostics — closes 1.10, 3.11
Report, don't assert: orthogonality drift `δ_m = ‖VₘᵀVₘ − I‖₂`, the Arnoldi relation residual `‖AVₘ − Vₘ₊₁H̄ₘ‖`, exact exception messages, and MGS vs CGS vs reorthogonalization comparison. "Loss of orthogonality" is a measured claim, not a label.

## 9. Spectral vs solver separation — closes 1.10, 1.11
Lanczos and Arnoldi are **spectral diagnostics**, not linear solvers here. They **must not** appear on the same residual/convergence curves as CG/GMRES/PCG. Either define formally how they produce an approximation `x_m`, or plot them separately as eigen-estimate diagnostics.

## 10. Scaling analysis — closes 3.4, 2.8
Do not claim "scalability" from visual time curves. Fit, in log space:
`log T = β₀ + β₁ log n + ε`, per (method, language, phase). Report **β₁** (empirical scaling exponent) with a confidence interval. The word "rigorous" returns to the abstract only after this holds.

## 11. Reporting & artifacts — closes 1.1, 3.8
- **Keep the main results table; replace its data.** The table is retained as a structural element, but every invalidated Table 1 number is removed and regenerated under the matched-criterion protocol. The new results propagate everywhere (abstract, results, discussion, conclusion); the old numbers survive nowhere. This is a substitution, **not** an appended correction section.
- Every table and figure is **generated from CSV** by a script; zero manual transcription; remove the "verify before submission" note.
- One consistent `n` set across methodology, tables, and figures.

## Phase plan

1. **Data layer** (§1.1, §2) — deterministic generation + common binary export + bit-for-bit verification.
2. **Rigorous harness** (§4, §5, §6, §1.3) — warmup, 30 reps, timing decomposition, work counts, resource instrumentation, thread regimes.
3. **Matrix families** (§3, §8.3) — expanded SPD + singular generators, multiple RHS.
4. **Re-run + reporting** (§10, §11) — full sweep, regenerate all tables/figures, scaling fit, normalized metrics.
5. **Manuscript rewrite** — single research question, retitle, moderate all claims, separate spectral/solver, define preconditioner, pin versions, fix terminology; repopulate the main results table with regenerated data.
6. **Verification** — check the rebuilt manuscript against every red flag in this matrix.
