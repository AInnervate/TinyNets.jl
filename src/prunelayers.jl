using Flux
using Random
using SparseArrays


abstract type PruningMethod end

struct PruneByIdentity <: PruningMethod end

struct PruneRandomly{T<:Number} <: PruningMethod
    value::T
end

struct PruneByMagnitude{T<:AbstractFloat} <: PruningMethod
    value::T
end

struct PruneByPercentage{T<:AbstractFloat} <: PruningMethod
    value::T
end

struct PruneByQuantity{T<:Integer} <: PruningMethod
    value::T
end


abstract type PruningSchedule end

# TODO: at this moment, the methods vector may contain 
#  PruningMethods with different sizes. not ideal.
struct SequencePruning <: PruningSchedule
    methods::Vector{PruningMethod}
end


numofnonzerostoremove(A::SparseMatrixCSC, n::Integer)::Integer = n - (length(A) - nnz(A))

# TODO: if A contain zeros, the end result could be undesired, since some
#  zeros may remain and will be erased after dropzeros call
# TODO: check which is faster (1) dropzeros!(A) followed by nnz(A) or 
#  (2) count(!iszero, A)
function droprand!(A::SparseMatrixCSC, p::AbstractFloat)::SparseMatrixCSC
    @assert zero(p) ≤ p ≤ one(p)

    # num of non-zeros to drop
    n = round(Integer, p * length(A))
    
    return droprand!(A, n)
end

function droprand!(A::SparseMatrixCSC, n::Integer)::SparseMatrixCSC
    @assert n ≥ zero(n)

    nzr = numofnonzerostoremove(A, n)

    if nzr > 0
        @views idx = shuffle(eachindex(A.nzval))[1:nzr]
        A.nzval[idx] .= 0
    end

    return dropzeros!(A)
end

function droppercentage!(A::SparseMatrixCSC, p::AbstractFloat)::SparseMatrixCSC
    @assert zero(p) ≤ p ≤ one(p)
    
    n = round(Integer, p * length(A))
    nzr = numofnonzerostoremove(A, n)

    if nzr > 0 
        @views idx = partialsortperm(A.nzval, 1:nzr, by=abs)
        A.nzval[idx] .= 0
    end

    return dropzeros!(A)
end

function dropquantity!(A::SparseMatrixCSC, n::Integer)::SparseMatrixCSC
    @assert n ≥ 0

    nzr = numofnonzerostoremove(A, n)

    @views idx = partialsortperm(A.nzval, 1:nzr, by=abs)
    A.nzval[idx] .= 0

    return dropzeros!(A)
end


function prunelayer(layer::Any, pm::PruningMethod)::Any
    @warn "Pruning not implemented for `$(Base.typename(typeof(layer)).wrapper)` layers."
    return layer
end

function prunelayer(chain::Chain, pm::PruningMethod)::Chain
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
