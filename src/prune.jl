module Prune

using Flux
using CUDA: CuArray
using Random
using Printf


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

Set to zero the `round(target_sparsity * countparams(model))` parameters of `model` with the lowest values of `by`.

It considers the parameter pool the entire model. For example, `prune!(model; by=abs, target_sparsity=0.2)` performs a global magnitude pruning of 80% of the model.
Layer-wise pruning can be achieved through broadcasting, e.g., `prune!.(model; by=abs, target_sparsity=0.1)`.
"""
function prune!(model; by::Function, target_sparsity::Real, verbose::Bool=false)
    @assert 0 ≤ target_sparsity ≤ 1

    # Check if the model is already sparse enough
    current_sparsity = sparsity(model)
    Δsparsity = target_sparsity - sparsity(model)
    if Δsparsity ≤ 0
        verbose && @info @sprintf("Current sparsity (%.2f%%) is already at least the target sparsity (%.2f%%).\n\tNothing to be done.", 100*current_sparsity, 100*target_sparsity)
        return model
    end

    verbose && @info @sprintf("Current sparsity: %.2f%%. Pruning to target sparsity: %.2f%%.", 100*current_sparsity, 100*target_sparsity)

    # Copy to CPU if necessary
    parameters = cpu.(Flux.params(model))
    refs = [Ref(p, i) for p in parameters for i in eachindex(p) if !iszero(p[i])]
    n_toprune = round(Int, Δsparsity * countparams(model))
    indices = partialsortperm(refs, 1:n_toprune, by=by∘getindex)
    @. setindex!(refs[indices], zero(eltype(refs[indices])))
    # Copy back to GPU if necessary
    for (p_new, p_old) in zip(parameters, Flux.params(model))
        if p_old isa CuArray
            copy!(p_old, p_new)
        end
    end

    verbose && @info @sprintf("Final sparsity: %.2f%%.", 100*sparsity(model))
    return model
end


end
