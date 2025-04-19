export ConditionalLinearLayer

using InvertibleNetworks: InvertibleNetworks, NeuralNetLayer, Parameter, glorot_uniform
using Statistics: cov
import Flux
using LinearAlgebra: pinv, cholesky, tr, tril, diagind, Diagonal


"""
This layer computes Ax + By + c where we A is lower triangular with positive diagonal.
"""
mutable struct ConditionalLinearLayer <: NeuralNetLayer
    A_free::Parameter
    B::Parameter
    c::Parameter
    logdet::Bool
    random_init::Bool
    is_reversed::Bool
end

@Flux.functor ConditionalLinearLayer

function ConditionalLinearLayer(; random_init=true, logdet=false)
    A_free = Parameter(nothing)
    B = Parameter(nothing)
    c = Parameter(nothing)
    return ConditionalLinearLayer(A_free, B, c, logdet, random_init, false)
end

function ConditionalLinearLayer(nx, ny; random_init=true, logdet=false)
    LN = ConditionalLinearLayer(; random_init, logdet)
    if random_init
        initialize!(LN, nx, ny)
    end
    return LN
end

function initialize!(LN::ConditionalLinearLayer, nx::Int, ny::Int)
    LN.A_free.data = glorot_uniform(nx, nx)
    LN.B.data = glorot_uniform(nx, ny)
    LN.c.data = zeros(Float32, nx)
    return nothing
end

function initialize!(LN::ConditionalLinearLayer, X::AbstractArray{T, Nx}, Y::AbstractArray{T, Ny}) where {T, Nx, Ny}
    # Initialize such that output is conditionally independent of the condition.
    # We're going to treat X as a set of vectors Xi = X[..., i].
    # We're going to treat Y as a set of vectors Yi = Y[..., i].
    N = size(X, Nx)
    X_vecs = reshape(X, :, N)
    Y_vecs = reshape(Y, :, N)
    if LN.random_init
        initialize!(LN, size(X_vecs, 1), size(Y_vecs, 1))
        return nothing
    end

    μ_x = mean(X_vecs; dims=2)
    μ_y = mean(Y_vecs; dims=2)

    B_x = cov(X_vecs; dims=2)
    B_xy = cov(X_vecs, Y_vecs; dims=2)
    B_y = cov(Y_vecs; dims=2)
    P_y = pinv(B_y)
    A = pinv(cholesky(B_x - B_xy * P_y * B_xy').U)
    LN.B.data = - A * B_xy * P_y
    LN.c.data = - A * μ_x + LN.B.data * μ_y
    LN.A_free.data = A
    LN.A_free.data[diagind(A)] .= log.(A[diagind(A)])
    return nothing
end

function InvertibleNetworks.forward(X::AbstractArray{T, Nx}, Y::AbstractArray{T, Ny}, LN::ConditionalLinearLayer; logdet=nothing) where {T, Nx, Ny}
    isnothing(logdet) ? logdet = (LN.logdet && ~LN.is_reversed) : logdet = logdet
    if isnothing(LN.A_free.data) && !LN.is_reversed
        initialize!(LN, X, Y)
    end
    return ConditionalLinearLayer_forward(X, Y, LN.A_free.data, LN.B.data, LN.c.data; logdet)
end

function ConditionalLinearLayer_forward(X::AbstractArray{T, Nx}, Y::AbstractArray{T, Ny}, A_free, B, c; logdet=true) where {T, Nx, Ny}
    Z, lgdet = ConditionalLinearLayer_forward_logdet(X, Y, A_free, B, c)
    if logdet
        return Z, lgdet
    end
    return Z
end

function ConditionalLinearLayer_forward_logdet(X::AbstractArray{T, Nx}, Y::AbstractArray{T, Ny}, A_free, B, c) where {T, Nx, Ny}
    N = size(X, Nx)
    X_vecs = reshape(X, :, N)
    Y_vecs = reshape(Y, :, N)
    A = tril(A_free, -1) + Diagonal(exp.(A_free[diagind(A_free)]))
    Z = A * X_vecs .+ B * Y_vecs .+ c
    return Z, tr(A_free) / N
end

function InvertibleNetworks.inverse(Z::AbstractArray{T, Nx}, Y::AbstractArray{T, Ny}, LN::ConditionalLinearLayer) where {T, Nx, Ny}
    N = size(Z, Nx)
    Z_vecs = reshape(Z, :, N)
    Y_vecs = reshape(Y, :, N)
    A_free = LN.A_free.data
    A = tril(A_free, -1) + Diagonal(exp.(A_free[diagind(A_free)]))
    X = A \ (Z_vecs .- LN.B.data * Y_vecs .- LN.c.data)
    return X
end

function InvertibleNetworks.backward(ΔZ::AbstractArray{T, Nx}, Z::AbstractArray{T, Nx}, Y::AbstractArray{T, Ny}, LN::ConditionalLinearLayer) where {T, Nx, Ny}
    X = InvertibleNetworks.inverse(Z, Y, LN)
    if LN.logdet
        # log det terms are summed, negated, and averaged over batch in loss function.
        # So the derivative of the loss with respect to this log det term is -1/N,
        #  but we already took care of the N in forward_logdet. 
        Δlgdet = T(-1)
        _, forward_pullback = Flux.pullback(ConditionalLinearLayer_forward_logdet, X, Y, LN.A_free.data, LN.B.data, LN.c.data)
        ΔX, ΔY, ΔA_free, ΔB, Δc = forward_pullback((ΔZ, Δlgdet))
        @show "Doing logdet" ΔA_free
    else
        _, forward_pullback = Flux.pullback(ConditionalLinearLayer_forward, X, Y, LN.A_free.data, LN.B.data, LN.c.data)
        ΔX, ΔY, ΔA_free, ΔB, Δc = forward_pullback(ΔZ)
    end

    LN.A_free.grad = ΔA_free
    LN.B.grad = ΔB
    LN.c.grad = Δc

    return ΔX, X, ΔY
end
