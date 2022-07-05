using Flux
using Statistics

include("prunelayers.jl")


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
struct TuneByAcDifference{T<:Number} <: FineTuner
    value::T
end


const PruningSchedule = Vector{<:Tuple{<:PruningMethod, <:FineTuner}}

function scheduledpruning(model::Any, schedule::PruningSchedule, losstype::Function, optimiser::Flux.Optimise.AbstractOptimiser, data::Any; verbose::Bool=false)
    for (pruningmethod, strategy) ∈ schedule
        verbose && println("Applying ", typeof(pruningmethod))
        verbose && println("Old sparsity: ", sparsity(model))
        
        model = prunelayer(model, pruningmethod)
        
        verbose && println("Current sparsity: ", sparsity(model))

        parameters = Flux.params(model)
        
        loss(x, y) = losstype(model(x), y)
        
        finetune(strategy, loss, parameters, optimiser, data, verbose=verbose)
    end

    return model
end

function trainandgetloss!(loss::Function, parameters::Any, data::Any, optimiser::Flux.Optimise.AbstractOptimiser)
    losssum = 0.0
    numsamples = 0

    for (x, y) in data
        gradients = gradient(() -> loss(x,y), parameters)
        Flux.Optimise.update!(optimiser, parameters, gradients)

        losssum += loss(x, y)
        numsamples += size(x)[end]
    end

    return (losssum / numsamples)
end

function finetune(strategy::TuneByEpochs, loss::Function, parameters::Any, optimiser::Flux.Optimise.AbstractOptimiser, data::Any; verbose::Bool=false)
    for epoch ∈ 1:strategy.value
        lossvalue = trainandgetloss!(loss, parameters, data, optimiser)
        verbose && println("epoch: $epoch - train loss: $lossvalue")
    end
end

function finetune(strategy::TuneByAbsoluteLoss, loss::Function, parameters::Any, optimiser::Flux.Optimise.AbstractOptimiser, data::Any; maxepochs::Integer=100, verbose::Bool=false)
    lossvalue = strategy.value + one(strategy.value)

    epoch = 0
    
    while (lossvalue > strategy.value) && (epoch < maxepochs)
        lossvalue = trainandgetloss!(loss, parameters, data, optimiser)

        epoch += 1
        verbose && println("epoch: $epoch - train loss: $(lossvalue)")
    end
end
function finetune(strategy::TuneByLossDifference, loss::Function, parameters::Any, optimiser::Flux.Optimise.AbstractOptimiser, data::Any; maxepochs::Integer=100, verbose::Bool=false)
    lossdiff = strategy.value + one(strategy.value)

    oldloss = 0.0
    epoch = 0
    
    while (lossdiff > strategy.value) && (epoch < maxepochs)
        lossvalue = trainandgetloss!(loss, parameters,data, optimiser)
        
        lossdiff = abs(oldloss - lossvalue)
        oldloss = lossvalue

        epoch += 1
        verbose && println("epoch: $epoch - train loss: $(oldloss)")
    end
end

function finetune(strategy::TuneByAcDifference, loss::Function, parameters::Any, optimiser::Flux.Optimise.AbstractOptimiser, data::Any; maxepochs::Integer=100, verbose::Bool=false)
    
    acdiff = strategy.value + one(strategy.value)

    oldac = 1.0
    epoch = 0
    
    while (acdiff > strategy.value) && (epoch < maxepochs)
        train!(loss, parameters,data, optimiser)
        acvalue=mean([accuracy(x,y) for (x,y) in train_loader])
        
        acdiff = abs(oldac - acvalue)
        oldac = acvalue

        epoch += 1
        verbose && println("epoch: $epoch - train accuracy: $(oldac)")
    end
end
