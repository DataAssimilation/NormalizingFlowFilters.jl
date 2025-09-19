using Statistics: mean, cov, std
using LinearAlgebra: norm, Diagonal, svd
using Random
using NormalizingFlowFilters
using NormalizingFlowFilters.InvertibleNetworks: get_params, set_params!, get_grads, NetworkConditionalCorrelation, forward, backward
using Test

include("grad_test.jl")

@testset "conditional_correlation gradient" begin
    N = 12
    Nx = 1

    Random.seed!(8237)

    network_config = ConditionalCorrelationOptions(
        chan_x = Nx,
        chan_y = Nx,
        L = 1,
        K = 1,
        subnetwork = CouplingLayerOptions(
            subnetwork = ResidualBlockOptions(n_hidden=1, k1 = 1, p1=0)
        )
    )
    in_shape = (1, 1, Nx)
    cond_shape = (1, 1, Nx)
    network = NetworkConditionalCorrelation(in_shape, cond_shape, network_config)

    forward_full = function (X, Y, params; with_grad=false)
        if !isnothing(params)
            set_params!(network, params)
        end
        sizeX0 = size(X)
        if ndims(X) < 4
            X = reshape(X, ones(Int64, 4 - ndims(X))..., size(X)...)
        end
        if ndims(Y) < 4
            Y = reshape(Y, ones(Int64, 4 - ndims(Y))..., size(Y)...)
        end
        Zx, Zy, logdet = forward(X, Y, network)
        J = sum(0.5 * (Zx .^ 2))/ size(X)[end] - logdet
        if with_grad
            dJ_dZx = Zx / size(X)[end]
            dJ_dX, _, dJ_dY = backward(dJ_dZx, Zx, Zy, network)
            dJ_dparams = get_grads(network)
            dJ_dX = reshape(dJ_dX, sizeX0)
            return J, Zx, dJ_dX, dJ_dparams
        end
        return J
    end
    forward_X_params = (X, params; with_grad=false) -> forward_full(X, Yinit, params; with_grad)

    forward_params = function (X)
        return function (params; with_grad=false)
            return forward_X_params(Xinit, params; with_grad)
        end
    end

    forward_input = function (params)
        function (X; with_grad=false)
            return forward_X_params(X, params; with_grad)
        end
    end

    Xinit = randn(Nx,N)
    Xinit .-= mean(Xinit; dims=2)
    Xinit ./= std(Xinit; dims=2)

    noise = randn(Nx,N)
    noise .-= mean(noise; dims=2)
    noise ./= std(noise; dims=2)
    Yinit = Xinit .+ noise

    # Initialize weights
    network0 = NetworkConditionalCorrelation(in_shape, cond_shape, network_config)

    params0 = get_params_as_type(network0, Xinit, Yinit, Float64)
    params = get_params_as_type(network, Xinit, Yinit, Float64)
    Δparams = params0 - params

    println("Testing gradient with respect to params")
    J, Zx, dJ_dX, dJ_dparams = forward_X_params(Xinit, params; with_grad=true)
    grad_test(forward_params(Xinit), params, Δparams, dJ_dparams; ΔJ=nothing, maxiter=20, h0=4e0, stol=1e-1, hfactor=5e-1, unittest=:test)

    println("Testing gradient with respect to X")
    J, Zx, dJ_dX, dJ_dparams = forward_X_params(Xinit, params; with_grad=true)
    ΔX = randn(Nx,N)
    grad_test(forward_input(params), Xinit, ΔX, dJ_dX; ΔJ=nothing, maxiter=20, h0=4e0, stol=1e-1, hfactor=5e-1, unittest=:test)

    # Now use random weights.
    params1 = deepcopy(params0)
    for p in params1
        p.data .= randn(size(p.data))
    end
    Δparams = deepcopy(params1)
    for Δparams_i in Δparams
        target_norm = norm(Δparams_i) * 1e-1
        Δparams_i.data .= randn(size(target_norm))
        Δparams_i.data .*= target_norm ./ norm(Δparams_i)
    end


    println("Testing gradient with respect to params")
    J, Zx, dJ_dX, dJ_dparams = forward_X_params(Xinit, params1; with_grad=true)
    grad_test(forward_params(Xinit), params1, Δparams, dJ_dparams; ΔJ=nothing, maxiter=20, h0=4e0, stol=1e-1, hfactor=5e-1, unittest=:test)

    println("Testing gradient with respect to X")
    J, Zx, dJ_dX, dJ_dparams = forward_X_params(Xinit, params1; with_grad=true)
    ΔX = randn(Nx,N)
    grad_test(forward_input(params1), Xinit, ΔX, dJ_dX; ΔJ=nothing, maxiter=20, h0=4e0, stol=1e-1, hfactor=5e-1, unittest=:test)
end


@testset "conditional_correlation assimilate" begin
    N = 1000
    Nx = 1

    # Ensemble members sampled from a unit normal.
    Random.seed!(834)
    prior_state = randn(Nx, N)

    # Identity observation operator with noise.
    prior_obs = deepcopy(prior_state) .+ randn(Nx, N)

    # Covariance should be nonzero.
    B_xy = cov(prior_state, prior_obs; dims=2)
    B_y = cov(prior_obs; dims=2)
    @test norm(B_xy) ≈ 1 atol=0.5

    # True state is the mean of the prior.
    y_obs = zeros(Nx)

    # Set up estimator.
    network_config = ConditionalCorrelationOptions(
        chan_x = Nx,
        chan_y = Nx,
        L = 1,
        K = 3,
        subnetwork = CouplingLayerOptions(
            subnetwork = ResidualBlockOptions(n_hidden=1, k1 = 1, p1=0,
                final_activation = ActivationOptions(type="identity")
            )
        )
    )

    in_shape = (1, 1, Nx)
    cond_shape = (1, 1, Nx)
    network = NetworkConditionalCorrelation(in_shape, cond_shape, network_config)

    optimizer_config = OptimizerOptions(; lr=2e-3, method="adam")
    optimizer = create_optimizer(optimizer_config)

    device = cpu
    training_config = TrainingOptions(;
        n_epochs=1500,
        num_post_samples=2,
        noise_lev_y=1e-3,
        noise_lev_x=1e-3,
        batch_size=N,
        validation_perc=1.0,
        reset_weights=false,
        reset_optimizer=true,
        print_every = 200,
        early_stopping_training_loss=EarlyStoppingOptions(active=true),
        early_stopping_validation_loss=EarlyStoppingOptions(active=true),
    )
    estimator = NormalizingFlowFilter(network, optimizer; device, training_config)

    # Assimilate.
    posterior = assimilate_data(estimator, prior_state, prior_obs, y_obs)

    # Cross-covariance with observation should be zero.
    B_zy = cov(posterior, prior_obs; dims=2)
    @test norm(B_zy) < 1e-2

    # Compare to expected solution.
    z = prior_state .- B_xy * (B_y \ prior_obs)
    @test norm(cov(posterior; dims=2)) ≈ norm(cov(z; dims=2)) rtol=1e-2 atol=1e-1
end
