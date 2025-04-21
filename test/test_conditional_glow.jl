using Statistics: mean, cov, std
using LinearAlgebra: norm, Diagonal, svd
using Random


@testset "conditional_glow gradient $activation" for activation in ("exp_clamp", "sigmoid")
    N = 12
    Nx = 1
    network_config = ConditionalGlowOptions(
        chan_x = 1,
        chan_y = 1,
        L = 3,
        K = 1,
        n_hidden = 1,
        residual = ResidualBlockOptions(k1 = 1, p1=0),
        positive_activation = ActivationOptions(type=activation),
    )
    network = NetworkConditionalGlow(2, network_config)

    forward_full = function (X, Y, params; with_grad=false)
        if !isnothing(params)
            set_params!(network, params)
        end
        if ndims(X) < 4
            X = reshape(X, ones(Int64, 4 - ndims(X))..., size(X)...)
        end
        if ndims(Y) < 4
            Y = reshape(Y, ones(Int64, 4 - ndims(Y))..., size(Y)...)
        end
        Zx, Zy, logdet = network.forward(X, Y)
        J = sum(0.5 * (Zx .^ 2))/ size(X)[end] - logdet
        if with_grad
            dJ_dZx = Zx / size(X)[end]
            dJ_dX, _, dJ_dY = network.backward(dJ_dZx, Zx, Zy)
            dJ_dparams = get_grads(network)
            return J, Zx, dJ_dX, dJ_dparams
        end
        return J
    end
    forward = (X, params; with_grad=false) -> forward_full(X, Yinit, params; with_grad)

    forward_params = function (X)
        return function (params; with_grad=false)
            return forward(Xinit, params; with_grad)
        end
    end

    forward_input = function (params)
        function (X; with_grad=false)
            return forward(X, params; with_grad)
        end
    end

    Random.seed!(8237)
    Xinit = randn(Nx,N)
    Xinit .-= mean(Xinit; dims=2)
    Xinit ./= std(Xinit; dims=2)

    noise = randn(Nx,N)
    noise .-= mean(noise; dims=2)
    noise ./= std(noise; dims=2)
    Yinit = Xinit .+ noise

    # This initializes the weights to the optimum value.
    _ = forward_full(Xinit, Yinit, nothing)

    params0 = deepcopy(get_params(network))
    Δparams = deepcopy(params0)
    for Δparams_i in Δparams
        target_norm = norm(Δparams_i) * 1e-1
        Δparams_i.data .= randn(size(target_norm))
        Δparams_i.data .*= target_norm ./ norm(Δparams_i)
    end

    # Test gradient with respect to params.
    J, Zx, dJ_dX, dJ_dparams = forward(Xinit, params0; with_grad=true)
    grad_test(forward_params(Xinit), params0, Δparams, dJ_dparams; ΔJ=nothing, maxiter=6, h0=1e-1, stol=1e-1, hfactor=5e-1, unittest=:test)

    # Test gradient with respect to X.
    J, Zx, dJ_dX, dJ_dparams = forward(Xinit, params0; with_grad=true)
    ΔX = 1e-3 .* randn(Nx,N)
    grad_test(forward_input(params0), Xinit, ΔX, dJ_dX; ΔJ=nothing, maxiter=6, h0=1e0, stol=1e-1, hfactor=8e-1, unittest=:test)

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

    # Test gradient with respect to params.
    J, Zx, dJ_dX, dJ_dparams = forward(Xinit, params1; with_grad=true)
    grad_test(forward_params(Xinit), params1, Δparams, dJ_dparams; ΔJ=nothing, maxiter=6, h0=1e0, stol=1e-1, hfactor=8e-1, unittest=:test)

    # Test gradient with respect to X.
    J, Zx, dJ_dX, dJ_dparams = forward(Xinit, params1; with_grad=true)
    ΔX = 1e-3 .* randn(Nx,N)
    grad_test(forward_input(params1), Xinit, ΔX, dJ_dX; ΔJ=nothing, maxiter=6, h0=1e0, stol=1e-1, hfactor=8e-1, unittest=:test)
end


@testset "conditional_glow assimilate" begin
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
    network_config = ConditionalGlowOptions(
        chan_x = 1,
        chan_y = 1,
        L = 3,
        K = 1,
        n_hidden = 1,
        residual = ResidualBlockOptions(k1 = 1, p1=0),
        positive_activation = ActivationOptions(type="sigmoid"),
    )

    network = NetworkConditionalGlow(2, network_config)

    optimizer_config = OptimizerOptions(; lr=1e-3, method="adam")
    optimizer = create_optimizer(optimizer_config)

    device = cpu
    training_config = TrainingOptions(;
        n_epochs=1000,
        num_post_samples=2,
        noise_lev_y=0e-3,
        noise_lev_x=0e-3,
        batch_size=N,
        validation_perc=1.0,
        reset_weights=true,
        reset_optimizer=true,
        print_every = 200,
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
