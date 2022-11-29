module MaskedLayers

using Flux


export MaskedLayer


struct MaskedLayer{T}
    layer::T
    mask::T
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


function applymask!(mlayer::MaskedLayer)
    for (p, m) ∈ zip(Flux.params(mlayer.layer), Flux.params(mlayer.mask))
        p .*= m
    end
    return nothing
end

function (mlayer::MaskedLayer)(x...)
    applymask!(mlayer)
    return mlayer.layer(x...)
end

end


