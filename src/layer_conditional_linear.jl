export ConditionalLinearLayer

using InvertibleNetworks: InvertibleNetworks, NeuralNetLayer, Parameter
using LinearAlgebra: svd
import Flux

mutable struct ConditionalLinearLayer <: NeuralNetLayer
    Ux::Parameter
    Vtx::Parameter
    Sx::Parameter
    Uy::Parameter
    Vty::Parameter
    Sy::Parameter
    logdet::Bool
    is_reversed::Bool
end

@Flux.functor ConditionalLinearLayer

function ConditionalLinearLayer(; logdet=false)
    s = Parameter(nothing)
    b = Parameter(nothing)
    Ux = Parameter(nothing)
    Vtx = Parameter(nothing)
    Sx = Parameter(nothing)
    Uy = Parameter(nothing)
    Vty = Parameter(nothing)
    Sy = Parameter(nothing)
    return ConditionalLinearLayer(Ux, Vtx, Sx, Uy, Vty, Sy, logdet, false)
end

function initialize!(LN::ConditionalLinearLayer, X::AbstractArray{T, Nx}, Y::AbstractArray{T, Ny}) where {T, Nx, Ny}
    # Initialize during first pass such that output is conditionally
    # independent of the condition.
    # We're going to treat X as a set of vectors Xi = X[..., i].
    # We're going to treat Y as a set of vectors Yi = Y[..., i].
    N = size(X, Nx)
    X_vecs = reshape(X, :, N)
    Y_vecs = reshape(Y, :, N)
    ny = size(Y_vecs, 1)
    dX_vecs = X_vecs .- mean(X_vecs; dims=2)
    dY_vecs = Y_vecs .- mean(Y_vecs; dims=2)
    Fx = svd(dX_vecs)
    Fy = svd(dY_vecs)
    rank = 1:(min(N-1, ny))
    LN.Ux.data = Fx.U[:, rank]
    LN.Vtx.data = Fx.Vt[rank, :]
    LN.Sx.data = Fx.S[rank] ./ sqrt(N-1)
    LN.Uy.data = Fy.U[:, rank]
    LN.Vty.data = Fy.Vt[rank, :]
    LN.Sy.data = Fy.S[rank] ./ sqrt(N-1)
    return nothing
end

function compute_bias(LN::ConditionalLinearLayer, Y::AbstractArray{T, Ny}) where {T, Ny}
    Ux = LN.Ux.data
    Vtx = LN.Vtx.data
    Sx = LN.Sx.data
    Uy = LN.Uy.data
    Vty = LN.Vty.data
    Sy = LN.Sy.data
    N = size(Y, Ny)
    Y_vecs = reshape(Y, :, N)
    ny = size(Y_vecs, 1)

    a = Uy' * Y_vecs
    if ny > N
        a ./= Sy
    else
        a .= Sy .* (
            Uy' * (
                Uy * (
                    a ./ (Sy .^ 2)
                )
            )
        )
    end
    return Ux * (
        Sx .* (
            Vtx * (
                Vty' * a
            )
        )
    )
end

function InvertibleNetworks.forward(X::AbstractArray{T, Nx}, Y::AbstractArray{T, Ny}, LN::ConditionalLinearLayer; logdet=nothing) where {T, Nx, Ny}
    isnothing(logdet) ? logdet = (LN.logdet && ~LN.is_reversed) : logdet = logdet
    if isnothing(LN.Ux.data) && !LN.is_reversed
        initialize!(LN, X, Y)
    end
    bias = compute_bias(LN, Y)
    Z = X .- reshape(bias, size(X))
    if logdet
        return Z, zero(T)
    end
    return Z
end

function InvertibleNetworks.inverse(Z::AbstractArray{T, N}, Y::AbstractArray{T, N}, LN::ConditionalLinearLayer) where {T, N}
    bias = compute_bias(LN, Y)
    X = Z .+ reshape(bias, size(Z))
    return X
end

function InvertibleNetworks.backward(ΔZ::AbstractArray{T, N}, Z::AbstractArray{T, N}, Y::AbstractArray{T, N}, LN::ConditionalLinearLayer; set_grad::Bool = true) where {T, N}
    Z = InvertibleNetworks.inverse(Z, Y, LN)
    if set_grad
        return ΔZ, Z
    end
    if LN.logdet
        return ΔZ, [], Z, []
    end
    return ΔZ, [], Z
end
