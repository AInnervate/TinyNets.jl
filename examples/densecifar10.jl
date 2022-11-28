include("../src/prunelayers.jl")
include("../src/schedulepruning.jl")

using Flux
using Flux.Data: DataLoader
using Flux: train!, onehotbatch
using Flux.Losses: logitcrossentropy
using MLDatasets
using Random: seed!
seed!(0x35c88aa0a17d0e83)


x_train, y_train = MLDatasets.MNIST(Float32, split=:train)[:]
x_train = Flux.flatten(x_train)
y_train = onehotbatch(y_train, 0:9)
train_loader = DataLoader((x_train, y_train), batchsize=256, shuffle=true)


model = Chain(Dense(784, 32, relu, init=rand), Dense(32, 10, init=rand))

loss(x, y) = logitcrossentropy(model(x), y)
opt = ADAM(3e-4)

for epoch ∈ 1:20
    train!(loss, Flux.params(model), train_loader, opt)
    @info "Epoch $epoch - loss: $(loss(x_train, y_train))"
end


schedule = [
    (PruneByPercentage(0.50), TuneByLossDifference(0.001)),
    (PruneByPercentage(0.75), TuneByLossDifference(0.001)),
    (PruneByPercentage(0.90), TuneByLossDifference(0.001))
]

sparsemodel = scheduledpruning(model, schedule, logitcrossentropy, opt, train_loader, verbose=true)
