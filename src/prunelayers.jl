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


# TODO: if A contain zeros, the end result could be undesired, since some
#  zeros may remain and will be erased after dropzeros call
function dropquantity!(A::SparseMatrixCSC, value::Integer, f::Function)::SparseMatrixCSC
    @assert value ≥ zero(value)

    S = shuffle(eachindex(A))

    idx = partialsortperm(vec(S), 1:value, by=f)
    A[S[idx]] .= 0

    return dropzeros!(A)
end


function prunelayer(layer::T, pm::PruningMethod)::T where T <: Any
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
function prunelayer(layer::Dense, pm::PruneRandomly{T})::Dense where T <: AbstractFloat
    @assert pm.value ≥ zero(pm.value)
    
    if pm.value ≤ one(pm.value)
        n = round(Integer, pm.value * length(layer.weight))
    else
        n = round(Integer, pm.value)
    end
    
    return prunelayer(layer, PruneRandomly(n))
end

function prunelayer(layer::Dense, pm::PruneRandomly{T})::Dense where T <: Integer
    @assert pm.value ≥ zero(pm.value)

    w = sparse(layer.weight)
    dropquantity!(w, pm.value, zero)
    
    return Dense(w, layer.bias, layer.σ)
end

function prunelayer(layer::Dense, pm::PruneByMagnitude)::Dense
    w = sparse(layer.weight)
    droptol!(w, pm.value)
    return Dense(w, layer.bias, layer.σ)
end

function prunelayer(layer::Dense, pm::PruneByPercentage)::Dense
    @assert zero(pm.value) ≤ pm.value ≤ one(pm.value)
    
    n = round(Integer, pm.value * length(layer.weight))
    return prunelayer(layer, PruneByQuantity(n))
end

function prunelayer(layer::Dense, pm::PruneByQuantity)::Dense
    w = sparse(layer.weight)
    dropquantity!(w, pm.value, i->abs(w[i]))
    return Dense(w, layer.bias, layer.σ)
end


function prunelayer(layer::T, s::PruningSchedule)::T where T <: Any
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
