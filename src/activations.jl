using InvertibleNetworks: ActivationFunction

export IdentityActivation

function IdentityActivation()
    return ActivationFunction(identity, identity, Identitygrad)
end

Identitygrad(Δy::AbstractArray{T, N}, x::AbstractArray{T, N}) where {T, N} = Δy
