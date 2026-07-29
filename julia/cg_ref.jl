"""
Reference Conjugate Gradient (and Jacobi-PCG) with an EXPLICIT, matched stopping
criterion, so the Julia and Python runs solve the identical problem the identical
way. Purpose: test the original paper's claim that "Julia CG converges in ~59
iterations while Python CG fails at ~5000" on the 1-D Laplacian.

Deterministic RHS (same closed form in both languages) removes RNG differences,
so any iteration-count gap would be real rather than an input artefact.
"""
module CGRef

using LinearAlgebra, SparseArrays
export laplacian_1d, rhs_deterministic, cg_ref

"1-D Dirichlet Laplacian tridiag(-1,2,-1): SPD, sparse, condition number ~ n^2."
function laplacian_1d(n::Int)
    d = fill(2.0, n); o = fill(-1.0, n - 1)
    return spdiagm(-1 => o, 0 => d, 1 => o)
end

"Deterministic RHS identical across languages: b_i = sin(0.1 i), normalised."
function rhs_deterministic(n::Int)
    b = [sin(0.1 * i) for i in 1:n]
    return b ./ norm(b)
end

"""
    cg_ref(A, b; rtol, atol, maxiter, jacobi) -> (x, iters, converged, relres)

Preconditioned CG. Stops when ‖r_k‖ ≤ max(rtol·‖b‖, atol). `jacobi=true` uses the
diagonal (Jacobi) preconditioner M = diag(A).
"""
function cg_ref(A, b; rtol = 1e-8, atol = 0.0, maxiter = size(A, 2), jacobi = false)
    n = length(b)
    x = zeros(n)
    r = copy(b)
    Minv = jacobi ? (1.0 ./ diag(A)) : ones(n)
    z = Minv .* r
    p = copy(z)
    rz = dot(r, z)
    nb = norm(b)
    thresh = max(rtol * nb, atol)
    iters = 0
    converged = false
    for k in 1:maxiter
        Ap = A * p
        alpha = rz / dot(p, Ap)
        x .+= alpha .* p
        r .-= alpha .* Ap
        iters = k
        if norm(r) <= thresh
            converged = true
            break
        end
        z = Minv .* r
        rz_new = dot(r, z)
        p .= z .+ (rz_new / rz) .* p
        rz = rz_new
    end
    return x, iters, converged, norm(b - A * x) / nb
end

end # module
