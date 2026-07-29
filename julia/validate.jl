# Validation harness: prove the Drazin inverse + DGMRES are correct BEFORE
# scaling to benchmarks. Run:  julia code/julia/validate.jl
include("DrazinKrylov.jl")
using .DrazinKrylov
using LinearAlgebra
using Printf
using Random: MersenneTwister

pass = Ref(0); fail = Ref(0)
function check(name, cond)
    if cond
        pass[] += 1; @printf("  PASS  %s\n", name)
    else
        fail[] += 1; @printf("  FAIL  %s\n", name)
    end
end

println("== 1. Drazin inverse axioms (block_nilpotent, index k) ==")
for k in 1:3
    A = block_nilpotent(6, k; seed = 100 + k)
    kdet = drazin_index(A)
    Ad = drazin_inverse(A)
    # Drazin defining properties:
    #   (i)  A^{k+1} A^D = A^k
    #   (ii) A^D A A^D = A^D
    #   (iii) A A^D = A^D A
    e1 = norm(A^(k + 1) * Ad - A^k) / norm(A^k)
    e2 = norm(Ad * A * Ad - Ad) / norm(Ad)
    e3 = norm(A * Ad - Ad * A)
    check("index detected == $k (got $kdet)", kdet == k)
    check(@sprintf("A^{k+1} A^D = A^k   (rel err %.2e)", e1), e1 < 1e-8)
    check(@sprintf("A^D A A^D = A^D     (rel err %.2e)", e2), e2 < 1e-8)
    check(@sprintf("A A^D = A^D A       (abs err %.2e)", e3), e3 < 1e-8)
end

println("\n== 2. Singular graph Laplacian (Neumann) has index 1 ==")
for n in (5, 10, 25)
    L = Matrix(graph_laplacian_singular(n))
    check("ind(L_$n) == 1", drazin_index(L) == 1)
    check("L_$n is singular (min sv ~ 0)", minimum(svdvals(L)) < 1e-10)
end

println("\n== 3. DGMRES converges to the Drazin solution A^D b ==")
for k in 1:3
    A = block_nilpotent(8, k; seed = 200 + k)
    n = size(A, 1)
    xtrue = randn(MersenneTwister(7), n)
    b = A * (A^k * (drazin_inverse(A)^k) * xtrue)  # ensure b in range(A^k): b = A x_D-consistent
    b = A * drazin_solution(A, b)                  # canonical consistent RHS
    xD = drazin_solution(A, b)
    res = dgmres(A, b; m = n + k + 2, tol = 1e-10)
    err = norm(res.x - xD) / max(norm(xD), 1e-30)
    check(@sprintf("DGMRES == A^D b, index %d  (rel err %.2e, iters %d, conv %s)",
                   k, err, res.iterations, res.converged), err < 1e-6)
end

println("\n== 4. DGMRES on singular Laplacian vs ground truth ==")
for n in (20, 50)
    L = Matrix(graph_laplacian_singular(n))
    xt = randn(MersenneTwister(3), n)
    b = L * xt                       # consistent by construction (b in range L)
    xD = drazin_solution(L, b)
    res = dgmres(L, b; index = 1, m = n, tol = 1e-10)
    err = norm(res.x - xD) / max(norm(xD), 1e-30)
    check(@sprintf("DGMRES==A^D b on L_%d (rel err %.2e, iters %d)", n, err, res.iterations),
          err < 1e-5)
end

@printf("\n== SUMMARY: %d passed, %d failed ==\n", pass[], fail[])
exit(fail[] == 0 ? 0 : 1)
