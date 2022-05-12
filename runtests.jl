include("pruning.jl")

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

    @test_throws AssertionError prunelayer(model, PruneByPercentage(4))
    @test_throws AssertionError prunelayer(model, PruneByPercentage(1.5))
    @test_throws AssertionError prunelayer(model, PruneByPercentage(-0.5))
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

@testset "sequence pruning dense" begin
    W = [[-0.1, 0.2, -0.3] [0.0, -0.5, 0.06] [-0.7, 0.08, 0.0] [0.0, -0.11, 0.12]]
    model = Dense(W)
    sequence = SequencePruning([PruneByIdentity(), PruneByIdentity()])
    new_model = prunelayer(model, sequence)
    @test new_model.weight == sparse(W)
    @test new_model.bias == model.bias
    @test new_model.σ == model.σ
    
    sequence = SequencePruning([PruneByQuantity(4), PruneByMagnitude(0.15)])
    new_model = prunelayer(model, sequence)
    W2 = sparse([[0.0, 0.2, -0.3] [0.0, -0.5, 0.0] [-0.7, 0.0, 0.0] [0.0, 0.0, 0.0]])
    @test new_model.weight == W2
    @test new_model.bias == model.bias
    @test new_model.σ == model.σ
end

@testset "sequence pruning layers other than Dense" begin
    W = [-0.1; 0.2;; 0.0; -0.04;;;; 0.5; 0.06;; -0.7; 0.0]
    model = Conv(W)
    new_model = (@test_logs (:warn, "Pruning not implemented for `Conv` layers.") (:warn, "Pruning not implemented for `Conv` layers.") prunelayer(model, SequencePruning([PruneByIdentity(), PruneByIdentity()])))
    @test model == new_model
    new_model = (@test_logs (:warn, "Pruning not implemented for `Conv` layers.") (:warn, "Pruning not implemented for `Conv` layers.") prunelayer(model, SequencePruning([PruneByQuantity(4), PruneByMagnitude(0.15)])))
    @test model == new_model

    model = MaxPool((3, 3))
    new_model = (@test_logs (:warn, "Pruning not implemented for `MaxPool` layers.") (:warn, "Pruning not implemented for `MaxPool` layers.") prunelayer(model, SequencePruning([PruneByIdentity(), PruneByIdentity()])))
    @test model == new_model
    new_model = (@test_logs (:warn, "Pruning not implemented for `MaxPool` layers.") (:warn, "Pruning not implemented for `MaxPool` layers.") prunelayer(model, SequencePruning([PruneByQuantity(4), PruneByMagnitude(0.15)])))
    @test model == new_model
end

@testset "sequence pruning Chain" begin
    W = [[-0.1, 0.2, -0.3] [0.0, -0.5, 0.06] [-0.7, 0.08, 0.0] [0.0, -0.11, 0.12]]
    K = [-0.1; 0.2;; 0.0; -0.04;;;; 0.5; 0.06;; -0.7; 0.0]
    model = Chain(Dense(W), Conv(K))
    sequence = SequencePruning([PruneByQuantity(4), PruneByMagnitude(0.15)])
    W2 = sparse([[0.0, 0.2, -0.3] [0.0, -0.5, 0.0] [-0.7, 0.0, 0.0] [0.0, 0.0, 0.0]])

    new_model = (@test_logs (:warn, "Pruning not implemented for `Conv` layers.") (:warn, "Pruning not implemented for `Conv` layers.") prunelayer(model, sequence))
    @test new_model[1].weight == W2
    @test new_model[1].bias == model[1].bias
    @test new_model[1].σ == model[1].σ
    @test new_model[2] == model[2]
end
