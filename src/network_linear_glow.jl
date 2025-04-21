
import InvertibleNetworks

export NetworkConditionalLinearGlow

struct NetworkConditionalLinearGlow <: InvertibleNetwork
    LN::NetworkConditionalLinear
    GN::NetworkConditionalGlow
end

@Flux.functor NetworkConditionalLinearGlow

function NetworkConditionalLinearGlow(ln_config::ConditionalLinearOptions, ndims, gn_config::ConditionalGlowOptions)
    GN = NetworkConditionalGlow(ndims, gn_config)
    LN = NetworkConditionalLinear(ln_config)
    return NetworkConditionalLinearGlow(LN, GN)
end

function InvertibleNetworks.forward(X::AbstractArray{T, NX}, Y::AbstractArray{T, NY}, G::NetworkConditionalLinearGlow) where {T, NX, NY}
    X, Y, lgdet_total = InvertibleNetworks.forward(X, Y, G.LN)
    if ndims(X) < 4
        X = reshape(X, ones(Int64, 4 - ndims(X))..., size(X)...)
    end
    if ndims(Y) < 4
        Y = reshape(Y, ones(Int64, 4 - ndims(Y))..., size(Y)...)
    end
    X, Y, lgdet = InvertibleNetworks.forward(X, Y, G.GN)
    lgdet_total += lgdet
    return X, Y, lgdet_total
end

function InvertibleNetworks.inverse(X::AbstractArray{T, NX}, Y::AbstractArray{T, NY}, G::NetworkConditionalLinearGlow) where {T, NX, NY}
    if NX < 4
        X = reshape(X, ones(Int64, 4 - NX)..., size(X)...)
    end
    if NY < 4
        Y = reshape(Y, ones(Int64, 4 - NY)..., size(Y)...)
    end
    X = InvertibleNetworks.inverse(X, Y, G.GN)
    X = InvertibleNetworks.inverse(X, Y, G.LN)
    return X
end

function InvertibleNetworks.backward(ΔX::AbstractArray{T, NX}, X::AbstractArray{T, NX}, Y::AbstractArray{T, NY}, G::NetworkConditionalLinearGlow;) where {T, NX, NY}
    if NX < 4
        X = reshape(X, ones(Int64, 4 - NX)..., size(X)...)
        ΔX = reshape(ΔX, ones(Int64, 4 - NX)..., size(ΔX)...)
    end
    if NY < 4
        Y = reshape(Y, ones(Int64, 4 - NY)..., size(Y)...)
    end
    ΔY_total = zero(Y)
    ΔX, X, ΔY = InvertibleNetworks.backward(ΔX, X, Y, G.GN)
    ΔY_total += ΔY
    ΔX, X, ΔY = InvertibleNetworks.backward(ΔX, X, Y, G.LN)
    ΔY_total += ΔY
    return ΔX, X, ΔY_total
end

function NetworkConditionalLinearGlow(ndims, config::ConditionalLinearGlowOptions)
    NetworkConditionalLinearGlow(config.ln_config, ndims, config.gn_config)
end

function reset_network(network::NetworkConditionalLinearGlow)
    return NetworkConditionalLinearGlow(reset_network(network.LN), reset_network(network.GN))
end
