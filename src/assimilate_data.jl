using Flux: cpu, gpu
using LinearAlgebra: norm
using Random: randn
using Statistics: mean, std

export assimilate_data, draw_posterior_samples, normalize_samples

T_LOG = Union{<:AbstractDict,<:Nothing}

function normalize_samples(
    G, X, Y, size_x; device=cpu, num_samples, batch_size, log_data::T_LOG=nothing
)
    batch_size = min(batch_size, num_samples)
    Zx = zeros(Float32, size_x[1:(end - 1)]..., num_samples)
    for i in 1:div(num_samples, batch_size)
        X_forward_i = X[:, :, :, ((i - 1) * batch_size + 1):(i * batch_size)]
        Y_forward_i = Y[:, :, :, ((i - 1) * batch_size + 1):(i * batch_size)]
        Zx_fixed_train_i, _, _ = G.forward(device(X_forward_i), device(Y_forward_i))
        Zx[:, :, :, ((i - 1) * batch_size + 1):(i * batch_size)] = cpu(Zx_fixed_train_i)
    end
    return Zx
end

function draw_posterior_samples(
    G, y, X, Y, size_x; device=cpu, num_samples, batch_size, log_data::T_LOG=nothing
)
    batch_size = min(batch_size, num_samples)
    X_forward = device(randn(Float64, size_x[1:(end - 1)]..., batch_size))
    Y_train_latent_repeat = device(repeat(y, 1, 1, 1, batch_size))
    Zx_fixed_train, Zy_fixed_train, _ = G.forward(X_forward, Y_train_latent_repeat)

    X_post = zeros(eltype(X), size_x[1:(end - 1)]..., num_samples)
    for i in 1:cld(num_samples, batch_size)
        bi_begin = (i - 1) * batch_size + 1
        bi_end = min((i * batch_size), num_samples)
        bis = bi_begin:bi_end
        X_forward_i = X[:, :, :, bis]
        Y_forward_i = Y[:, :, :, bis]
        Zx_fixed_train_i, _, _ = G.forward(device(X_forward_i), device(Y_forward_i))
        Zy_fixed_train_i = Zy_fixed_train[:, :, :, 1:(bi_end-bi_begin+1)]
        X_post[:, :, :,bis] = cpu(
            G.inverse(Zx_fixed_train_i, Zy_fixed_train_i)
        )
    end
    return X_post
end

function draw_posterior_samples(
    G, y, size_x; device=cpu, num_samples, batch_size, log_data::T_LOG=nothing
)
    batch_size = min(batch_size, num_samples)
    X_forward = device(randn(Float64, size_x[1:(end - 1)]..., batch_size))
    y_r = reshape(cpu(y), 1, 1, :, 1)
    Y_train_latent_repeat = device(repeat(y_r, 1, 1, 1, batch_size))
    Zx_fixed_train, Zy_fixed_train, _ = G.forward(X_forward, Y_train_latent_repeat)

    X_post = zeros(Float32, size_x[1:(end - 1)]..., num_samples)
    for i in 1:div(num_samples, batch_size)
        ZX_noise_i = device(randn(Float64, size_x[1:(end - 1)]..., batch_size))
        X_post[:, :, :, ((i - 1) * batch_size + 1):(i * batch_size)] = cpu(
            G.inverse(ZX_noise_i, Zy_fixed_train)
        )
    end
    return X_post
end

_ensure_1d(a::T) where {T<:Number} = T[a]
_ensure_1d(a::AbstractArray{T,N}) where {T,N} = a

_ensure_2d(a::T) where {T<:Number} = T[a;;]
_ensure_2d(a::AbstractArray{T,1}) where {T} = reshape(a, (1, size(a)...))
_ensure_2d(a::AbstractArray{T,N}) where {T,N} = a

function assimilate_data(
    filter::NormalizingFlowFilter,
    prior_state::T1,
    prior_obs::T2,
    y_obs::Ty,
    log_data::T_LOG=nothing;
    train=true,
) where {T1<:AbstractArray,T2<:AbstractArray,Ty<:Union{<:AbstractArray,<:Number}}
    X = reshape(prior_state, filter.state_shape..., :)
    Y = reshape(prior_obs, filter.obs_shape..., :)
    y_obs = reshape(y_obs, filter.obs_shape...)
    X_post = assimilate_data(filter, X, Y, y_obs, log_data; train)
    return reshape(X_post, size(prior_state))
end

"""

- `prior_state` has shape `(s..., N)` for state shape `s` and number of ensemble members `N`.
- `prior_obs` has shape `(r..., N)` for observation shape `r` and number of ensemble members `N`.
- `y_obs` has shape `r` for observation shape `r`.

"""
# function assimilate_data(
#     filter::NormalizingFlowFilter,
#     prior_state::AbstractArray{T1,2},
#     prior_obs::AbstractArray{T2,2},
#     y_obs::AbstractArray{T3,1},
#     log_data::T_LOG=nothing;
#     train=true,
# ) where {T1<:Number,T2<:Number,T3<:Number}
#     X = prior_state
#     Y = prior_obs

#     X = reshape(X, (1, 1, size(X, 1), size(X, 2)))
#     Y = reshape(Y, (1, 1, size(Y, 1), size(Y, 2)))
#     y_obs = reshape(y_obs, (1, 1, size(y_obs, 1), size(y_obs, 2)))
#     X = assimilate_data(filter, prior_state, prior_obs, y_obs, log_data; train)
#     posterior = X[1, 1, :, :]
#     return posterior
# end

function assimilate_data(
    filter::NormalizingFlowFilter,
    prior_state::AbstractArray{T1,4},
    prior_obs::AbstractArray{T2,4},
    y_obs::AbstractArray{T3,3},
    log_data::T_LOG=nothing;
    train=true,
) where {T1<:Number,T2<:Number,T3<:Number}
    X = prior_state
    Y = prior_obs

    if filter.training_config.normalize_initial
        X_scale = std(X; dims=(1,2,4))
        X_scale .= ifelse.(X_scale .== 0, 1, X_scale)
        X_shift = mean(X; dims=(1,2,4))
        X = (X .- X_shift) ./ X_scale

        Y_scale = std(Y; dims=(1,2,4))
        Y_scale .= ifelse.(Y_scale .== 0, 1, Y_scale)
        Y_shift = mean(Y; dims=(1,2,4))
        Y = (Y .- Y_shift) ./ Y_scale
        y_obs = (y_obs .- Y_shift) ./ Y_scale
    end

    if train
        train_network!(filter, X, Y; log_data)
    end

    batch_size = get_batch_size(filter.training_config.batch, size(X, 4))
    X = draw_posterior_samples(
        filter.coupling_network_device,
        y_obs,
        X,
        Y,
        size(X);
        device=filter.device,
        num_samples=size(X, 4),
        batch_size,
        log_data,
    )
    if filter.training_config.normalize_initial
        X .= X_scale .* X .+ X_shift
    end
    return X
end
