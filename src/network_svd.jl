
using InvertibleNetworks: InvertibleNetworks, InvertibleNetwork

import Flux

export NetworkConditionalSVD

struct NetworkConditionalSVD <: InvertibleNetwork
    CN::ConditionalSVDLayer
end

@Flux.functor NetworkConditionalSVD

# Constructor
function NetworkConditionalSVD()
    CN = ConditionalSVDLayer(logdet=true)
    return NetworkConditionalSVD(CN)
end

# Forward pass and compute logdet
function InvertibleNetworks.forward(X::AbstractArray{T, NX}, Y::AbstractArray{T, NY}, G::NetworkConditionalSVD) where {T, NX, NY}
    Z, logdet = G.CN.forward(X, Y)
    return Z, Y, logdet
end

# Inverse pass 
function InvertibleNetworks.inverse(X::AbstractArray{T, NX}, Y::AbstractArray{T, NY}, G::NetworkConditionalSVD) where {T, NX, NY}
    Z = G.CN.inverse(X, Y)
    return Z
end

# Backward pass and compute gradients
function InvertibleNetworks.backward(ΔX::AbstractArray{T, NX}, X::AbstractArray{T, NX}, Y::AbstractArray{T, NY}, G::NetworkConditionalSVD;) where {T, NX, NY}
    ΔZ, Z = G.CN.backward(ΔX, X, Y)
    ΔY = zero(Y)
    return ΔZ, Z, ΔY
end

function NetworkConditionalSVD(config::ConditionalLinearOptions)
    return NetworkConditionalSVD()
end

function reset_network(network::NetworkConditionalSVD)
    return NetworkConditionalSVD()
end
