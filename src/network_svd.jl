
using InvertibleNetworks: InvertibleNetworks, InvertibleNetwork

import Flux

export NetworkConditionalSVD

struct NetworkConditionalSVD <: InvertibleNetwork
    LN::ConditionalSVDLayer
end

@Flux.functor NetworkConditionalSVD

# Constructor
function NetworkConditionalSVD()
    LN = ConditionalSVDLayer(logdet=true)
    return NetworkConditionalSVD(LN)
end

# Forward pass and compute logdet
function InvertibleNetworks.forward(X::AbstractArray{T, NX}, Y::AbstractArray{T, NY}, G::NetworkConditionalSVD) where {T, NX, NY}
    Z, logdet = G.LN.forward(X, Y)
    return Z, Y, logdet
end

# Inverse pass 
function InvertibleNetworks.inverse(X::AbstractArray{T, NX}, Y::AbstractArray{T, NY}, G::NetworkConditionalSVD) where {T, NX, NY}
    Z = G.LN.inverse(X, Y)
    return Z
end

# Backward pass and compute gradients
function InvertibleNetworks.backward(ΔX::AbstractArray{T, NX}, X::AbstractArray{T, NX}, Y::AbstractArray{T, NY}, G::NetworkConditionalSVD;) where {T, NX, NY}
    ΔZ, Z = G.LN.backward(ΔX, X, Y)
    ΔY = zero(Y)
    return ΔZ, Z, ΔY
end

function NetworkConditionalSVD(config::ConditionalSVDOptions)
    return NetworkConditionalSVD()
end

function reset_network(network::NetworkConditionalSVD)
    return NetworkConditionalSVD()
end
