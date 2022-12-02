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

"""
    prune!(model; by::Function, target_sparsity::Real)

Prune parameters of `model` with the lowest values of `by` so that only a `target_sparsity` factor of the parameters are non-zero.

It considers the parameter pool the entire model. For example, `prune!(model; by=abs, target_sparsity=0.2)` performs a global magnitude pruning of 80% of the model.
Layer-wise pruning can be achieved through broadcasting, e.g., `prune!.(model; by=abs, target_sparsity=0.1)`.
"""
function prune!(model; by::Function, target_sparsity::Real)
    @assert 0 ≤ target_sparsity ≤ 1

    refs = [Ref(p, i) for p in Flux.params(model) for i in eachindex(p)]
    n_toprune = round(Int, target_sparsity * countparams(model))
    indices = partialsortperm(refs, 1:n_toprune, by=by∘getindex)
    for i ∈ indices
        refs[i][] = zero(refs[i][])
    end

    return model
end


end