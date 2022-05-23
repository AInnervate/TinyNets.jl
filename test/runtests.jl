include("../src/prunelayers.jl")

using Test
using Flux
using SparseArrays


@testset "identity pruning" begin
    W = [[-1, 2, -3] [0, -5, 6] [-7, 8, 0] [0, -11, 12]]
    b = [1,2,3]
    model = Dense(W, b)
    @test prunelayer(model, PruneByIdentity()).weight == sparse(W)
    @test prunelayer(model, PruneByIdentity()).bias == model.bias
    @test prunelayer(model, PruneByIdentity()).σ == model.σ

    W = [0 -1 2]
    b = [3]
    model = Dense(W, b)
    @test prunelayer(model, PruneByIdentity()).weight == sparse(W)
    @test prunelayer(model, PruneByIdentity()).bias == model.bias
    @test prunelayer(model, PruneByIdentity()).σ == model.σ
end

@testset "random pruning" begin
    model = Dense(10, 10)
    @test nnz(prunelayer(model, PruneRandomly(0)).weight) == length(model.weight)
    @test nnz(prunelayer(model, PruneRandomly(4)).weight) == (length(model.weight) - 4)
    @test nnz(prunelayer(model, PruneRandomly(100)).weight) == 0
    @test nnz(prunelayer(model, PruneRandomly(0.0)).weight) == length(model.weight)
    @test nnz(prunelayer(model, PruneRandomly(0.1)).weight) == round(0.9 * length(model.weight))
    @test nnz(prunelayer(model, PruneRandomly(1.0)).weight) == 0

    @test_throws AssertionError prunelayer(model, PruneRandomly(-0.5))
    @test_throws AssertionError prunelayer(model, PruneRandomly(-10))

    @test_throws MethodError prunelayer(model, PruneByPercentage("abc"))
    @test_throws MethodError prunelayer(model, PruneByPercentage([0.2, 1.4]))
end

@testset "magnitude pruning" begin
    W = [[-0.1, 0.2, -0.3] [0.0, -0.5, 0.06] [-0.7, 0.08, 0.0] [0.0, -0.11, 0.12]]
    model = Dense(W)
    W2 = sparse(copy(W))
    droptol!(W2, 0.1)
    W3 = sparse([[0.0, 0.2, -0.3] [0.0, -0.5, 0.0] [-0.7, 0.0, 0.0] [0.0, -0.11, 0.12]])
    new_model = prunelayer(model, PruneByMagnitude(0.1))
    @test new_model.weight == W2
    @test new_model.weight == W3
    @test new_model.bias == model.bias
    @test new_model.σ == model.σ

    @test_throws MethodError prunelayer(model, PruneByPercentage(21))
    @test_throws MethodError prunelayer(model, PruneByPercentage(0x03))
    @test_throws MethodError prunelayer(model, PruneByPercentage("abc"))
    @test_throws MethodError prunelayer(model, PruneByPercentage([0.2, 1.4]))
end

@testset "percentage pruning" begin
    W = [[-0.1, 0.2, -0.3] [0.0, -0.5, 0.06] [-0.7, 0.08, 0.0] [0.0, -0.11, 0.12]]
    model = Dense(W)
    W2 = sparse([[-0.1, 0.2, -0.3] [0.0, -0.5, 0.0] [-0.7, 0.0, 0.0] [0.0, -0.11, 0.12]])
    new_model = prunelayer(model, PruneByPercentage(0.4))
    @test new_model.weight == W2
    @test new_model.bias == model.bias
    @test new_model.σ == model.σ

    W = [[0.0, 0.2, -0.3] [0.0, -0.5, 0.0]]
    model = Dense(W)
    new_model = prunelayer(model, PruneByPercentage(0.3))
    @test new_model.weight == sparse(W)
    @test new_model.bias == model.bias
    @test new_model.σ == model.σ

    @test_throws AssertionError prunelayer(model, PruneByPercentage(4.0))
    @test_throws AssertionError prunelayer(model, PruneByPercentage(1.5))
    @test_throws AssertionError prunelayer(model, PruneByPercentage(-0.5))

    @test_throws MethodError prunelayer(model, PruneByPercentage(21))
    @test_throws MethodError prunelayer(model, PruneByPercentage(0x03))
    @test_throws MethodError prunelayer(model, PruneByPercentage("abc"))
    @test_throws MethodError prunelayer(model, PruneByPercentage([0.2, 1.4]))
end

@testset "quantity pruning" begin
    W = [[-0.1, 0.2, -0.3] [0.0, -0.5, 0.06] [-0.7, 0.08, 0.0] [0.0, -0.11, 0.12]]
    model = Dense(W)
    W2 = sparse([[-0.1, 0.2, -0.3] [0.0, -0.5, 0.0] [-0.7, 0.0, 0.0] [0.0, -0.11, 0.12]])
    new_model = prunelayer(model, PruneByQuantity(5))
    @test new_model.weight == W2
    @test new_model.bias == model.bias
    @test new_model.σ == model.σ

    W = [[0.0, 0.2, -0.3] [0.0, -0.5, 0.0]]
    model = Dense(W)
    new_model = prunelayer(model, PruneByQuantity(2))
    @test new_model.weight == sparse(W)
    @test new_model.bias == model.bias
    @test new_model.σ == model.σ

    @test_throws AssertionError prunelayer(model, PruneByQuantity(-5))

    @test_throws MethodError prunelayer(model, PruneByPercentage("abc"))
    @test_throws MethodError prunelayer(model, PruneByPercentage([0.2, 1.4]))
end

@testset "layers other than Dense" begin
    @test_nowarn prunelayer(Dense(2, 2), PruneByIdentity())
    @test_nowarn prunelayer(Chain(Dense(10 => 5), Dense(5 => 2)), PruneByIdentity())
    
    @test_logs (:warn, "Pruning not implemented for `Conv` layers.") prunelayer(Conv((5, 5), 2 => 2), PruneByIdentity())
    @test_logs (:warn, "Pruning not implemented for `MaxPool` layers.") prunelayer(MaxPool((5, 5)), PruneByIdentity())
    @test_logs (:warn, "Pruning not implemented for `Conv` layers.") prunelayer(Chain(Dense(4, 4), Conv((2, 2), 2 => 2)), PruneByIdentity())

    W = [[-0.1, 0.2, -0.3] [0.0, -0.5, 0.06] [-0.7, 0.08, 0.0] [0.0, -0.11, 0.12]]
    new_model = (@test_logs (:warn, "Pruning not implemented for `Conv` layers.") prunelayer(Chain(Dense(W), Conv((2, 2), 2 => 2)), PruneByIdentity()))
    @test (new_model[1].weight == sparse(W))
end
