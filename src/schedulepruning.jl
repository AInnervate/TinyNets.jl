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
struct TuneByAccuracyDifference{T<:Number} <: FineTuner
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

        finetune(model, strategy, loss, parameters, optimiser, data, verbose=verbose)
    end

    return model
end


function datasetloss(data::Any, loss::Function)
    losssum = 0.0
    numsamples = 0

    for (x, y) in data
        losssum += loss(x, y)
        numsamples += size(x)[end]
    end

    return (losssum / numsamples)
end

function datasetaccuracy(data::Any, model::Any)
    accuracysum = 0.0
    numsamples = 0

    for (x, y) in data
        accuracysum += sum(Flux.onecold(model(x)) .== Flux.onecold(y))
        numsamples += size(x)[end]
    end

    return (accuracysum / numsamples)
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

function trainandgetlossandaccuracy!(loss::Function, parameters::Any, data::Any, optimiser::Flux.Optimise.AbstractOptimiser)
    losssum = 0.0
    accuracysum = 0.0
    numsamples = 0

    for (x, y) in data
        gradients = gradient(() -> loss(x,y), parameters)
        Flux.Optimise.update!(optimiser, parameters, gradients)

        losssum += loss(x, y)
        accuracysum += sum(Flux.onecold(model(x)) .== Flux.onecold(y))
        numsamples += size(x)[end]
    end

    return (losssum / numsamples), (accuracysum / numsamples)
end


function finetune(model::Any, strategy::TuneByEpochs, loss::Function, parameters::Any, optimiser::Flux.Optimise.AbstractOptimiser, data::Any; verbose::Bool=false)
    for epoch ∈ 1:strategy.value
        train!(loss, parameters, data, optimiser)

        lossvalue = datasetloss(data, loss)
        accuracyvalue = datasetaccuracy(data, model)

        verbose && println("epoch: $epoch - train loss: $lossvalue")
    end
end

function finetune(model::Any, strategy::TuneByAbsoluteLoss, loss::Function, parameters::Any, optimiser::Flux.Optimise.AbstractOptimiser, data::Any; maxepochs::Integer=100, verbose::Bool=false)
    lossvalue = strategy.value + one(strategy.value)

    epoch = 0

    while (lossvalue > strategy.value) && (epoch < maxepochs)
        train!(loss, parameters, data, optimiser)

        lossvalue = datasetloss(data, loss)
        accuracyvalue = datasetaccuracy(data, model)

        epoch += 1
        verbose && println("epoch: $epoch - train loss: $(lossvalue)")
    end
end

function finetune(model::Any, strategy::TuneByLossDifference, loss::Function, parameters::Any, optimiser::Flux.Optimise.AbstractOptimiser, data::Any; maxepochs::Integer=100, verbose::Bool=false)
    lossdiff = strategy.value + one(strategy.value)

    oldloss = 0.0
    epoch = 0

    while (lossdiff > strategy.value) && (epoch < maxepochs)
        train!(loss, parameters, data, optimiser)

        lossvalue = datasetloss(data, loss)
        accuracyvalue = datasetaccuracy(data, model)

        lossdiff = abs(oldloss - lossvalue)
        oldloss = lossvalue

        epoch += 1
        verbose && println("epoch: $epoch - train loss: $(oldloss)")
    end
end

function finetune(model::Any, strategy::TuneByAccuracyDifference, loss::Function, parameters::Any, optimiser::Flux.Optimise.AbstractOptimiser, data::Any; maxepochs::Integer=100, verbose::Bool=false)
    accuracydiff = strategy.value + one(strategy.value)

    oldaccuracy = 0.0
    epoch = 0

    while (accuracydiff > strategy.value) && (epoch < maxepochs)
        train!(loss, parameters, data, optimiser)

        lossvalue = datasetloss(data, loss)
        accuracyvalue = datasetaccuracy(data, model)

        accuracydiff = accuracyvalue - oldaccuracy
        oldaccuracy = accuracyvalue

        epoch += 1
        verbose && println("epoch: $epoch - train accuracy: $(accuracyvalue) - train loss: $(lossvalue)")
    end
end
