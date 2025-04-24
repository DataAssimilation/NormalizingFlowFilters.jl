using InvertibleNetworks: ActivationFunction

export IdentityActivation, SoftplusLayer

function IdentityActivation()
    return ActivationFunction(identity, identity, Identitygrad)
end

Identitygrad(Δy::AbstractArray{T, N}, x::AbstractArray{T, N}) where {T, N} = Δy

SoftplusLayer() = ActivationFunction(Softplus, SoftplusInv, SoftplusGrad)

function Softplus(x::AbstractArray{T, N}) where {T, N}
    return log.(1 .+ exp.(x))
end

function SoftplusInv(y::AbstractArray{T, N}) where {T, N}
    if any(y .≈ 0)
        throw(InputError("Input contains zeros."))
    else
        return log.(exp.(y) .- 1)
    end
end

function SoftplusGrad(Δy::AbstractArray{T, N}, y::AbstractArray{T, N}) where {T, N}
    return (exp.(y) .- 1) ./ exp.(y) .* Δy
end
