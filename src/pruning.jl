using Flux
using Random
using SparseArrays


abstract type PruningMethod end

struct PruneByIdentity <: PruningMethod end

struct PruneRandomly{T} <: PruningMethod
    value::T
end

struct PruneByMagnitude{T} <: PruningMethod
    value::T
end

struct PruneByPercentage{T} <: PruningMethod
    value::T
end

struct PruneByQuantity{T} <: PruningMethod
    value::T
end


abstract type PruningSchedule end

struct SequencePruning <: PruningSchedule
    methods::Vector{PruningMethod}
end


# using SparseArrays: dropstored!

# function dropindex!(A::SparseMatrixCSC, v::Vector{T}) where T<:Integer
#     #fkeep!(A, (i, j, x) -> i * lengthA( + j == v))
#     idxs = CartesianIndices(size(A))[v]
#     @show idxs
#     for i ∈ idxs
#         @show i
#         dropstored!(A, i[1], i[2])
#     end
# end

function droprand!(A::SparseMatrixCSC, p::AbstractFloat)
    @assert zero(p) ≤ p ≤ one(p)

    # num of non-zeros to drop
    n = round(Integer, p * length(A))
    k = n - (length(A) - nnz(A))

    if k > 0
        # TODO: add inbounds
        @views idx = shuffle(eachindex(A.nzval))[1:k]
        A.nzval[idx] .= 0
    end

    dropzeros!(A)

    return A
end

function droprand!(A::SparseMatrixCSC, n::Integer)
    @assert n ≥ zero(n)

    k = n - (length(A) - nnz(A))

    @views idx = shuffle(eachindex(A.nzval))[1:k]
    A.nzval[idx] .= 0

    dropzeros!(A)

    return A
end

function droppercentage!(A::SparseMatrixCSC, p::Real)
    @assert zero(p) ≤ p ≤ one(p)
    
    n = round(Integer, p * length(A))
    k = n - (length(A) - nnz(A))

    if k > 0 
        @views idx = partialsortperm(A.nzval, 1:k, by=abs)
        A.nzval[idx] .= 0
    end

    dropzeros!(A)

    return A
end

function dropquantity!(A::SparseMatrixCSC, n::Integer)
    @assert n ≥ 0

    k = n - (length(A) - nnz(A))

    @views idx = partialsortperm(A.nzval, 1:k, by=abs)
    A.nzval[idx] .= 0

    dropzeros!(A)

    return A
end


function prunelayer(layer::Any, pm::PruningMethod)
    @warn "Pruning not implemented for `$(Base.typename(typeof(layer)).wrapper)` layers."
    return layer
end

function prunelayer(chain::Chain, pm::PruningMethod)
    return Chain(prunelayer.(chain.layers, (pm,)))
end

function prunelayer(layer::Dense, pm::PruneByIdentity)::Dense
    w = sparse(layer.weight)
    dropzeros!(w)
    return Dense(w, layer.bias, layer.σ)
end

# unstructured
function prunelayer(layer::Dense, pm::PruneRandomly)::Dense
    w = sparse(layer.weight)
    droprand!(w, pm.value)
    return Dense(w, layer.bias, layer.σ)
end

function prunelayer(layer::Dense, pm::PruneByMagnitude)::Dense
    w = sparse(layer.weight)
    droptol!(w, pm.value)
    return Dense(w, layer.bias, layer.σ)
end

function prunelayer(layer::Dense, pm::PruneByPercentage)::Dense
    w = sparse(layer.weight)
    droppercentage!(w, pm.value)
    return Dense(w, layer.bias, layer.σ)
end

function prunelayer(layer::Dense, pm::PruneByQuantity)::Dense
    w = sparse(layer.weight)
    dropquantity!(w, pm.value)
    return Dense(w, layer.bias, layer.σ)
end


function prunelayer(layer::Any, s::PruningSchedule)::Any
    chain = Chain(Base.Fix2.(prunelayer, s.methods))
    chain(layer)
end

function prunelayer(layer::Chain, s::PruningSchedule)::Chain
    chain = Chain(Base.Fix2.(prunelayer, s.methods))
    chain(layer)
end

function prunelayer(layer::Dense, s::PruningSchedule)::Dense
    chain = Chain(Base.Fix2.(prunelayer, s.methods))
    chain(layer)
end


# begin
#     layer = Conv((3, 3), 2 => 2)

#     display(layer.weight)
#     display(layer.bias)
#     display(layer.σ)

#     s = SequencePruning([PruneByMagnitude(0.7), PruneByQuantity(2)])
    
#     model = prunelayer(layer, s)

#     display(model.weight)
#     display(model.bias)
#     display(model.σ)
# end