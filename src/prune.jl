module Prune

using Flux
using Random


nnz(model) = sum(x->count(!iszero, x), Flux.params(model))
countparams(model) = sum(length, Flux.params(model))
sparsity(model) = 1.0 - nnz(model)/countparams(model)

function drop!(A::VecOrMat, qty::Int, by::Function)::VecOrMat
    @assert 0 ≤ qty ≤ length(A)

    idx = partialsortperm(vec(A), 1:qty, by=by)
    A[idx] .= zero(eltype(A))

    return A
end

drop(A, qty, by) = drop!(copy(A), qty, by)

function prune!(layer; by::Function, target_sparsity::Real)
    @assert 0 ≤ target_sparsity ≤ 1

    for p in Flux.params(layer)
        p .= drop!(p, round(Int, length(p)*target_sparsity), by)
    end
end


end