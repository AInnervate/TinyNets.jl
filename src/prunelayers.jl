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


abstract type FineTuner end

struct TuneByEpochs{T<:Integer} <: FineTuner
    value::T
end

struct TuneByAbsoluteLoss{T<:Number} <: FineTuner
    value::T
end

struct TuneByLossDifference{T<:Number} <: FineTuner
    value::T
end


const PruningSchedule = Vector{<:Tuple{<:PruningMethod, <:FineTuner}}


numofnonzerostoremove(A::SparseMatrixCSC, n::Integer)::Integer = n - (length(A) - nnz(A))

getelementindexes(A, n, pm::PruneRandomly) = shuffle(eachindex(A.nzval))[1:n]
getelementindexes(A, n, pm::PruneByQuantity) = partialsortperm(A.nzval, 1:n, by=abs)


# TODO: if A contain zeros, the end result could be undesired, since some
#  zeros may remain and will be erased after dropzeros call
# TODO: check which is faster (1) dropzeros!(A) followed by nnz(A) or 
#  (2) count(!iszero, A)
function dropquantity!(A::SparseMatrixCSC, pm::PruningMethod)::SparseMatrixCSC
    @assert pm.value isa Integer
    @assert pm.value ≥ zero(pm.value)

    nzr = numofnonzerostoremove(A, pm.value)

    if nzr > 0
        @views idx = getelementindexes(A, nzr, pm)
        A.nzval[idx] .= 0
    end

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
    @assert pm.value ≥ zero(pm.value)
    
    if pm.value isa AbstractFloat
        if pm.value ≤ one(pm.value)
            n = round(Integer, pm.value * length(layer.weight))
        else
            n = round(Integer, pm.value)
        end
        pm = PruneRandomly(n)
    end
    
    w = sparse(layer.weight)
    dropquantity!(w, pm)
    
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
    dropquantity!(w, pm)
    return Dense(w, layer.bias, layer.σ)
end


function scheduledpruning(model::Any, schedule::PruningSchedule, losstype::Function, opt::Flux.Optimise.AbstractOptimiser, data; verbose=false)
    for (pm, s) ∈ schedule
        verbose && println("Applying ", typeof(pm))
        display(length(model[1].weight))
        display(length(model[2].weight))
        model = prunelayer(model, pm)
        display(nnz(model[1].weight))
        display(nnz(model[2].weight))

        ps = Flux.params(model)
        loss(x, y) = losstype(model(x), y)
        finetune(s, loss, ps, opt, data)
    end
    return model
end

function finetune(strategy::TuneByEpochs, loss::Function, ps, opt::Flux.Optimise.AbstractOptimiser, data)
    # @epochs strategy.value train!(loss, ps, data, opt)

    losssum = 0.0
    numsamples = 0

    for i ∈ 1:strategy.value
        for (x, y) in data
            gs = gradient(() -> loss(x,y), ps)
            Flux.Optimise.update!(opt, ps, gs)

            losssum += loss(x, y)
            numsamples += size(x)[end]
        end

        println("epoch: $i - train loss: $(losssum/numsamples)")
    end
end
