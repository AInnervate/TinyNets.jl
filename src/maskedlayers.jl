module MaskedLayers

using Flux
using ChainRulesCore


export mask, unmask


struct MaskedLayer{T, Tm}
    layer::T
    mask::Tm
    # TODO: add custom constructor with proper checks
end

function MaskedLayer(layer::T) where T
    if !(T <: Dense || T <: Conv)
        @warn "MaskedLayer not implemented for `$(Base.typename(typeof(layer)).wrapper)` layers. Returning input layer as is."
        return layer
    end
    mask = copy.(Flux.params(layer))
    for m ∈ mask
        m .= oneunit(eltype(m))
    end
    return MaskedLayer{T, typeof(mask)}(layer, mask)
end

Flux.@functor MaskedLayer
# Keeping the mask as non-parameter makes sure that training won't modify it
Flux.trainable(mlayer::MaskedLayer) = (mlayer.layer,)
# Since the mask is not a parameter, we need to say explicitly how to also move it to GPU
Flux.gpu(mlayer::MaskedLayer) = mask(Flux.gpu(mlayer.layer))


mask(layer) = MaskedLayer(deepcopy(layer))
mask(ch::Chain) = Chain(mask.(ch))

unmask(mlayer::MaskedLayer) = mlayer.layer
unmask(ch::Chain) = Chain(unmask.(ch))
unmask(layer) = layer

function applymask!(mlayer::MaskedLayer)
    # NOTE: using bitwise logic should be meaningfully faster
    for (p, m) ∈ zip(Flux.params(mlayer.layer), mlayer.mask)
        p .*= m
    end
end

function (mlayer::MaskedLayer)(x...)
    @ignore_derivatives applymask!(mlayer)
    return mlayer.layer(x...)
end

end
