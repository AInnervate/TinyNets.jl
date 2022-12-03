include("../src/prune.jl")
include("../src/maskedlayers.jl")
using .Prune: prune!, sparsity
using .MaskedLayers

using Flux
using Flux.Data: DataLoader
using Flux: train!, loadmodel!, onehotbatch, onecold
using Printf
using MLDatasets
using Random
Random.seed!(0x35c88aa0a17d0e83)


accuracy(model, x, y) = count(onecold(cpu(model(x))) .== onecold(cpu(y))) / size(x)[end]

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
    n_samples = size(x_data)[end]
    n_val = round(Int, n_samples * validation_proportion)
    n_train = n_samples - n_val
    x_train = collect(selectdim(x_data, ndims(x_data), 1:n_train))
    y_train = collect(selectdim(y_data, ndims(y_data), 1:n_train))
    x_val = collect(selectdim(x_data, ndims(x_data), n_train+1:n_samples))
    y_val = collect(selectdim(y_data, ndims(y_data), n_train+1:n_samples))

    if x_data isa CuArray
        x_train = x_train |> gpu
        y_train = y_train |> gpu
        x_val = x_val |> gpu
        y_val = y_val |> gpu
    end

    train_loader = DataLoader((x_train, y_train), batchsize=batch_size, shuffle=false)

    loss′(x, y) = loss(model(x), y)
    valloss_current = loss′(x_val, y_val)

    valloss_best = valloss_current
    model_best = deepcopy(model)

    trigger_noimprovement = Flux.early_stopping(identity, patience; init_score=valloss_best)

    for epoch ∈ 1:max_epochs
        train!(loss′, Flux.params(model), train_loader, optimizer)

        valloss_current = loss′(x_val, y_val)

        @info @sprintf("Epoch %3d - loss (val/train): %7.4f / %7.4f\e[F", epoch, valloss_current, loss′(x_train, y_train))

        if valloss_current < valloss_best
            valloss_best = valloss_current
            model_best = loadmodel!(model_best, model)
        end
        if trigger_noimprovement(valloss_current)
            println()
            @info "No improvement for $patience epochs. Stopping early.\e[F"
            break
        end
    end
    println()

    loadmodel!(model, model_best)
    @info "Best loss (val/train): $valloss_best / $(loss′(x_train, y_train))"
    return model
end



function main()
    data_train = MLDatasets.MNIST(Float32, split=:train)
    x_train, y_train = data_train[:]
    x_train = Flux.flatten(x_train)
    y_train = onehotbatch(y_train, 0:9)

    data_test = MLDatasets.MNIST(Float32, split=:test)
    x_test, y_test = data_test[:]
    x_test = Flux.flatten(x_test)
    y_test = onehotbatch(y_test, 0:9)

    # Preshuffle train data (to have the same validation set accross training rounds)
    shuffled_indices = shuffle(1:length(data_train))
    x_train = selectdim(x_train, ndims(x_train), shuffled_indices) |> collect
    y_train = selectdim(y_train, ndims(y_train), shuffled_indices) |> collect

    model = Chain(Dense(784, 32, relu), Dense(32, 10))

    traintoconvergence!(model, optimizer=ADAM(3e-4), train_data=(x_train, y_train), loss=logitcrossentropy, max_epochs=2, patience=2)
    @info "Accuracy:" test=accuracy(model, x_test, y_test) train=accuracy(model, x_train, y_train)

    println()
    @info "Pruning..."
    maskedmodel = mask(model)
    for target_sparsity ∈ (0.9, 0.95)
        prune!(maskedmodel, target_sparsity=target_sparsity, by=abs)
        MaskedLayers.updatemask!.(maskedmodel)
        traintoconvergence!(maskedmodel, optimizer=ADAM(3e-4), train_data=(x_train, y_train), loss=logitcrossentropy, max_epochs=100, patience=3)
        @info "Accuracy:" test=accuracy(maskedmodel, x_test, y_test) train=accuracy(maskedmodel, x_train, y_train)
    end

    @info("End results:",
        sparsity=sparsity(maskedmodel),
        Δaccuracy_test=accuracy(model, x_test, y_test) - accuracy(maskedmodel, x_test, y_test),
        Δaccuracy_train=accuracy(model, x_train, y_train) - accuracy(maskedmodel, x_train, y_train),
        Δloss_test=(logitcrossentropy(model(x_test), y_test) - logitcrossentropy(maskedmodel(x_test), y_test)),
        Δloss_train=(logitcrossentropy(model(x_train), y_train) - logitcrossentropy(maskedmodel(x_train), y_train)),
    )
end

