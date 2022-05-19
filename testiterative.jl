include("src/prunelayers.jl")

using Flux
using Flux.Data: DataLoader
using Flux: @epochs, train!, onehotbatch, onecold
using Flux.Losses: logitcrossentropy
using MLDatasets


x_train, y_train = MLDatasets.MNIST.traindata(Float32)
x_train = Flux.flatten(x_train)
y_train = onehotbatch(y_train, 0:9)
train_loader = DataLoader((x_train, y_train), batchsize=256, shuffle=true)


opt = ADAM(3e-4)

model = Chain(Dense(784, 32, relu, init=rand), Dense(32, 10, init=rand))

loss1(x, y) = logitcrossentropy(model(x), y)
@epochs 20 train!(loss1, Flux.params(model), train_loader, opt)

ls = 0.0
num = 0
for (x, y) in train_loader
    ls += loss1(x, y)
    num += size(x)[end]
end
println("train loss: $(ls/num)")


schedule = [
    (PruneByPercentage(0.50), TuneByEpochs(1)),
    (PruneByPercentage(0.75), TuneByEpochs(3)),
    (PruneByPercentage(0.90), TuneByEpochs(5))
]

schedule = [
    (PruneByPercentage(0.50), TuneByAbsoluteLoss(0.01)),
    (PruneByPercentage(0.75), TuneByAbsoluteLoss(0.01)),
    (PruneByPercentage(0.90), TuneByAbsoluteLoss(0.01))
]

schedule = [
    (PruneByPercentage(0.50), TuneByLossDifference(0.001)),
    (PruneByPercentage(0.75), TuneByLossDifference(0.001)),
    (PruneByPercentage(0.90), TuneByLossDifference(0.001))
]

sparsemodel = scheduledpruning(model, schedule, logitcrossentropy, opt, train_loader, verbose=true)

println(nnz(sparsemodel[1].weight))


loss2(x, y) = logitcrossentropy(sparsemodel(x), y)

ls = 0.0
num = 0
for (x, y) in train_loader
    ls += loss2(x, y)
    num += size(x)[end]
end
println("train loss: $(ls/num)")
