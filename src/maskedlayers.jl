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
        @warn "MaskedLayer not implemented for `$(Base.typename(typeof(layer)).wrapper)` layers. Returning original layer."
        return layer
    end
    mask = deepcopy(Flux.params(layer))
    for m ∈ mask
        m .= oneunit(eltype(m))
    end
    return MaskedLayer{T, typeof(mask)}(layer, mask)
end

Flux.@functor MaskedLayer
# TODO: masks are not properly moved to GPU
#       Masking a layer that is already on GPU does work, though
Flux.trainable(m::MaskedLayer) = (m.layer,)


mask(layer) = MaskedLayer(deepcopy(layer))
mask(ch::Chain) = Chain(mask.(ch))

unmask(mlayer::MaskedLayer) = mlayer.layer
unmask(ch::Chain) = Chain(unmask.(ch))

function applymask!(mlayer::MaskedLayer)
    for (p, m) ∈ zip(Flux.params(mlayer.layer), mlayer.mask)
        p .*= m
    end
end

function (mlayer::MaskedLayer)(x...)
    @ignore_derivatives applymask!(mlayer)
    return mlayer.layer(x...)
end

end


