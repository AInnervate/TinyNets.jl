include("../src/prunelayers.jl")
include("../src/schedulepruning.jl")

using Flux
using Flux.Data: DataLoader
using Flux: train!, onehotbatch
using Flux.Losses: logitcrossentropy
using MLDatasets
using Random: seed!
seed!(0x35c88aa0a17d0e83)


function traintoconvergence!(
    model;
    optimizer,
    train_data,
    loss,
    max_epochs = 100,
    patience = 3,
)
    train_loader = DataLoader(train_data, batchsize=256, shuffle=true)

    loss′(x, y) = loss(model(x), y)
    loss_current = loss′(x_train, y_train)

    trigger_noimprovement = Flux.early_stopping(identity, patience; init_score=loss_current)

    for epoch ∈ 1:max_epochs
        train!(loss′, Flux.params(model), train_loader, optimizer)

        loss_current = loss′(x_train, y_train)

        @info "Epoch $epoch - loss: $loss_current"

        if trigger_noimprovement(loss_current)
            @info "No improvement for $patience epochs. Stopping early."
            break
        end
    end

    return model
end


x_train, y_train = MLDatasets.MNIST(Float32, split=:train)[:]
x_train = Flux.flatten(x_train)
y_train = onehotbatch(y_train, 0:9)

model = Chain(Dense(784, 32, relu, init=rand), Dense(32, 10, init=rand))

traintoconvergence!(model, optimizer=ADAM(3e-4), train_data=(x_train, y_train), loss=logitcrossentropy)


schedule = [
    (PruneByPercentage(0.50), TuneByLossDifference(0.001)),
    (PruneByPercentage(0.75), TuneByLossDifference(0.001)),
    (PruneByPercentage(0.90), TuneByLossDifference(0.001))
]

sparsemodel = scheduledpruning(model, schedule, logitcrossentropy, opt, train_loader, verbose=true)
