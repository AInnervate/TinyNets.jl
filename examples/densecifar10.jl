include("../src/prunelayers.jl")
include("../src/schedulepruning.jl")

using Flux
using Flux.Data: DataLoader
using Flux: train!, onehotbatch, loadmodel!
using Flux.Losses: logitcrossentropy
using MLDatasets
using Random: seed!
seed!(0x35c88aa0a17d0e83)


function traintoconvergence!(
    model;
    optimizer,
    train_data,
    loss,
    batchsize = 128,
    max_epochs = 100,
    patience = 3,
)
    train_loader = DataLoader(train_data, batchsize=batchsize, shuffle=true)

    loss′(x, y) = loss(model(x), y)
    loss_current = loss′(x_train, y_train)

    loss_best = loss_current
    model_best = deepcopy(model)

    trigger_noimprovement = Flux.early_stopping(identity, patience; init_score=loss_best)

    for epoch ∈ 1:max_epochs
        train!(loss′, Flux.params(model), train_loader, optimizer)

        loss_current = loss′(x_train, y_train)

        @info "Epoch $epoch - loss: $loss_current"

        if loss_current < loss_best
            loss_best = loss_current
            model_best = loadmodel!(model_best, model)
        end
        if trigger_noimprovement(loss_current)
            @info "No improvement for $patience epochs. Stopping early."
            break
        end
    end

    @info "Best loss: $loss_best\n"
    return loadmodel!(model, model_best)
end

@timev begin
    x_train, y_train = MLDatasets.MNIST(Float32, split=:train)[:]
    x_train = Flux.flatten(x_train)
    y_train = onehotbatch(y_train, 0:9)

    model = Chain(Dense(784, 32, relu, init=rand), Dense(32, 10, init=rand))

    traintoconvergence!(model, optimizer=ADAM(3e-4), train_data=(x_train, y_train), loss=logitcrossentropy, patience=2)

    sparsemodel = deepcopy(model)
    for target_sparsity ∈ (0.9, 0.95)
        @info "Sparsity:" current=sparsity(sparsemodel) target=target_sparsity
        sparsemodel = prunelayer(model, PruneByPercentage(target_sparsity))
        traintoconvergence!(sparsemodel, optimizer=ADAM(3e-4), train_data=(x_train, y_train), loss=logitcrossentropy, patience=2)
    end

    @info "End results:" Δloss=(logitcrossentropy(model(x_train), y_train) - logitcrossentropy(sparsemodel(x_train), y_train)) sparsity=sparsity(sparsemodel)
end
