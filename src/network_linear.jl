
using InvertibleNetworks: InvertibleNetworks, InvertibleNetwork

import Flux

export NetworkConditionalLinear

struct NetworkConditionalLinear <: InvertibleNetwork
    CN::ConditionalLinearLayer
end

@Flux.functor NetworkConditionalLinear

# Constructor
function NetworkConditionalLinear()
    CN = ConditionalLinearLayer(logdet=true)
    return NetworkConditionalLinear(CN)
end

# Forward pass and compute logdet
function InvertibleNetworks.forward(X::AbstractArray{T, NX}, Y::AbstractArray{T, NY}, G::NetworkConditionalLinear) where {T, NX, NY}
    Z, logdet = G.CN.forward(X, Y)
    return Z, Y, logdet
end

# Inverse pass 
function InvertibleNetworks.inverse(X::AbstractArray{T, NX}, Y::AbstractArray{T, NY}, G::NetworkConditionalLinear) where {T, NX, NY}
    Z = G.CN.inverse(X, Y)
    return Z
end

# Backward pass and compute gradients
function InvertibleNetworks.backward(ΔX::AbstractArray{T, NX}, X::AbstractArray{T, NX}, Y::AbstractArray{T, NY}, G::NetworkConditionalLinear;) where {T, NX, NY}
    ΔZ, Z = G.CN.backward(ΔX, X, Y)
    ΔY = zero(Y)
    return ΔZ, Z, ΔY
end

function NetworkConditionalLinear(config::ConditionalLinearOptions)
    return NetworkConditionalLinear()
end

function reset_network(network::NetworkConditionalLinear)
    return NetworkConditionalLinear()
end
