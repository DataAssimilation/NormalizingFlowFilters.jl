
using InvertibleNetworks: InvertibleNetworks, InvertibleNetwork

import Flux

export NetworkConditionalLinear

struct NetworkConditionalLinear <: InvertibleNetwork
    LN::ConditionalLinearLayer
end

@Flux.functor NetworkConditionalLinear

# Constructor
function NetworkConditionalLinear(; random_init=true)
    LN = ConditionalLinearLayer(; random_init, logdet=true)
    return NetworkConditionalLinear(LN)
end

function initialize!(G::NetworkConditionalLinear, X::AbstractArray{T, Nx}, Y::AbstractArray{T, Ny}) where {T, Nx, Ny}
    initialize!(G.LN, X, Y)
end

# Forward pass and compute logdet
function InvertibleNetworks.forward(X::AbstractArray{T, NX}, Y::AbstractArray{T, NY}, G::NetworkConditionalLinear) where {T, NX, NY}
    Z, logdet = InvertibleNetworks.forward(X, Y, G.LN)
    return Z, Y, logdet
end

# Inverse pass 
function InvertibleNetworks.inverse(X::AbstractArray{T, NX}, Y::AbstractArray{T, NY}, G::NetworkConditionalLinear) where {T, NX, NY}
    Z = InvertibleNetworks.inverse(X, Y, G.LN)
    return Z
end

# Backward pass and compute gradients
function InvertibleNetworks.backward(ΔX::AbstractArray{T, NX}, X::AbstractArray{T, NX}, Y::AbstractArray{T, NY}, G::NetworkConditionalLinear;) where {T, NX, NY}
    ΔZ, Z, ΔY = InvertibleNetworks.backward(ΔX, X, Y, G.LN)
    return ΔZ, Z, ΔY
end

function NetworkConditionalLinear(config::ConditionalLinearOptions)
    return NetworkConditionalLinear(; config.random_init)
end

function reset_network(network::NetworkConditionalLinear)
    return NetworkConditionalLinear(; network.LN.random_init)
end
