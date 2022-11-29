module MaskedLayers

using Flux
using ChainRulesCore


export mask, unmask


struct MaskedLayer{T}
    layer::T
    mask::T
    # TODO: add custom constructor with proper checks
end

function MaskedLayer(layer::T) where T
    if !(T <: Dense || T <: Conv)
        @warn "MaskedLayer not implemented for `$(Base.typename(typeof(layer)).wrapper)` layers. Returning original layer."
        return layer
    end
    mask = deepcopy(layer)
    for p ∈ Flux.params(mask)
        p .= oneunit(eltype(p))
    end
    return MaskedLayer{T}(layer, mask)
end

Flux.@functor MaskedLayer
Flux.trainable(m::MaskedLayer) = (m.layer,)


mask(layer) = MaskedLayer(deepcopy(layer))
mask(ch::Chain) = Chain(mask.(ch))

unmask(mlayer::MaskedLayer) = mlayer.layer
unmask(ch::Chain) = Chain(unmask.(ch))

function applymask!(mlayer::MaskedLayer)
    for (p, m) ∈ zip(Flux.params(mlayer.layer), Flux.params(mlayer.mask))
        p .*= m
    end
    return nothing
end

function (mlayer::MaskedLayer)(x...)
    @ignore_derivatives applymask!(mlayer)
    return mlayer.layer(x...)
end

end


