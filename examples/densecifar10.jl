include("../src/prunelayers.jl")
include("../src/schedulepruning.jl")

using Flux
using Flux.Data: DataLoader
using Flux: train!, loadmodel!, onehotbatch, onecold
using Flux.Losses: logitcrossentropy
using MLDatasets
using Random
Random.seed!(0x35c88aa0a17d0e83)


function traintoconvergence!(
    model;
    optimizer,
    train_data,
    loss,
    batch_size = 128,
    max_epochs = 100,
    patience = 3,
    validation_proportion = 0.1,
)
    x_data, y_data = train_data
    # Split the data into training and validation sets
    # NOTE: No shuffling is performed! Preshuffled data is assumed.
    n_samples = x_data |> size |> last
    n_val = round(Int, n_samples * validation_proportion)
    n_train = n_samples - n_val
    x_train, y_train = selectdim(x_data, ndims(x_data), 1:n_train), selectdim(y_data, ndims(y_data), 1:n_train)
    x_val, y_val = selectdim(x_data, ndims(x_data), n_train+1:n_samples), selectdim(y_data, ndims(y_data), n_train+1:n_samples)

    train_loader = DataLoader((x_train, y_train), batchsize=batch_size, shuffle=false)

    loss′(x, y) = loss(model(x), y)
    valloss_current = loss′(x_val, y_val)

    valloss_best = valloss_current
    model_best = deepcopy(model)

    trigger_noimprovement = Flux.early_stopping(identity, patience; init_score=valloss_best)

    for epoch ∈ 1:max_epochs
        train!(loss′, Flux.params(model), train_loader, optimizer)

        valloss_current = loss′(x_val, y_val)

        @info "Epoch $epoch - loss (validation/train): $valloss_current / $(loss′(x_train, y_train))"

        if valloss_current < valloss_best
            valloss_best = valloss_current
            model_best = loadmodel!(model_best, model)
        end
        if trigger_noimprovement(valloss_current)
            @info "No improvement for $patience epochs. Stopping early."
            break
        end
    end

    loadmodel!(model, model_best)
    @info "Best loss (validation/train): $valloss_best / $(loss′(x_train, y_train))"
    return model
end

accuracy(model, x, y) = sum(onecold(cpu(model(x))) .== onecold(cpu(y))) / size(x)[end]


@timev begin
    train_data = MLDatasets.MNIST(Float32, split=:train)
    x_train, y_train = train_data[:]
    x_train = Flux.flatten(x_train)
    y_train = onehotbatch(y_train, 0:9)

    test_data = MLDatasets.MNIST(Float32, split=:test)
    x_test, y_test = test_data[:]
    x_test = Flux.flatten(x_test)
    y_test = onehotbatch(y_test, 0:9)

    # Preshuffle train data (to have the same validation set accross training rounds)
    shuffled_indices = shuffle(1:length(train_data))
    x_train = selectdim(x_train, ndims(x_train), shuffled_indices) |> collect
    y_train = selectdim(y_train, ndims(y_train), shuffled_indices) |> collect

    model = Chain(Dense(784, 32, relu), Dense(32, 10))

    traintoconvergence!(model, optimizer=ADAM(3e-4), train_data=(x_train, y_train), loss=logitcrossentropy, max_epochs=2, patience=2)
    @info "Accuracy:" test=accuracy(model, x_test, y_test) train=accuracy(model, x_train, y_train)

    println()
    @info "Pruning..."
    sparsemodel = deepcopy(model)
    for target_sparsity ∈ (0.9, 0.95)
        @info "Sparsity:" current=sparsity(sparsemodel) target=target_sparsity
        sparsemodel = prunelayer(model, PruneByPercentage(target_sparsity))
        traintoconvergence!(sparsemodel, optimizer=ADAM(3e-4), train_data=(x_train, y_train), loss=logitcrossentropy, max_epochs=2, patience=2)
        @info "Accuracy:" test=accuracy(sparsemodel, x_test, y_test) train=accuracy(sparsemodel, x_train, y_train)
    end

    @info("End results:",
        sparsity=sparsity(sparsemodel),
        Δaccuracy_test=accuracy(model, x_test, y_test) - accuracy(sparsemodel, x_test, y_test),
        Δaccuracy_train=accuracy(model, x_train, y_train) - accuracy(sparsemodel, x_train, y_train),
        Δloss_test=(logitcrossentropy(model(x_test), y_test) - logitcrossentropy(sparsemodel(x_test), y_test)),
        Δloss_train=(logitcrossentropy(model(x_train), y_train) - logitcrossentropy(sparsemodel(x_train), y_train)),
    )
end
