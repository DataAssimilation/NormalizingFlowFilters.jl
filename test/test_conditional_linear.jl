using Statistics: mean, cov, std
using LinearAlgebra: norm, Diagonal, svd
using Random

using NormalizingFlowFilters
using NormalizingFlowFilters.InvertibleNetworks: get_params, set_params!, get_grads

include("grad_test.jl")

@testset "conditional_linear gradient" begin
    N = 12
    Nx = 1
    network_config = ConditionalLinearOptions()
    network = NetworkConditionalLinear(network_config)

    forward = function (X, params; with_grad=false)
        set_params!(network, params)
        Zx, Zy, logdet = network.forward(X, Yinit)
        J = sum(0.5 * (Zx .^ 2))/ size(X)[end] - logdet
        if with_grad
            dJ_dZx = Zx / size(X)[end]
            dJ_dX, _, dJ_dY = network.backward(dJ_dZx, Zx, Zy)
            dJ_dparams = get_grads(network)
            return J, Zx, dJ_dX, dJ_dparams
        end
        return J
    end

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
    Z, _, logdet = network.forward(Xinit, Yinit)

    params0 = deepcopy(get_params(network))
    Δparams = deepcopy(params0)
    for Δparams_i in Δparams
        target_norm = norm(Δparams_i) * 1e-1
        Δparams_i.data .= randn(size(target_norm))
        Δparams_i.data .*= target_norm ./ norm(Δparams_i)
    end

    # Test gradient with respect to params.
    J, Zx, dJ_dX, dJ_dparams = forward(Xinit, params0; with_grad=true)
    @test norm(dJ_dparams) < 1/N + 1e-5
    grad_test(forward_params(Xinit), params0, Δparams, dJ_dparams; ΔJ=nothing, maxiter=6, h0=1e1, stol=1e-1, hfactor=5e-1, unittest=:test)

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

@testset "conditional_linear assimilate" begin
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
    network_config = ConditionalLinearOptions()
    network = NetworkConditionalLinear(network_config)

    optimizer_config = OptimizerOptions(; lr=1e-3)
    optimizer = create_optimizer(optimizer_config)

    device = cpu
    training_config = TrainingOptions(;
        n_epochs=3,
        num_post_samples=7,
        noise_lev_y=1e-3,
        noise_lev_x=1e-3,
        batch_size=N,
        validation_perc=1.0,
        reset_weights=true,
        reset_optimizer=true,
    )
    estimator = NormalizingFlowFilter(network, optimizer; device, training_config)

    # Get parameters.
    data_initial = deepcopy(get_data(estimator))
    data_initial2 = deepcopy(get_data(estimator))
    @test all(p.data == p2.data for (p, p2) in zip(data_initial, data_initial2))

    # Assimilate.
    posterior = assimilate_data(estimator, prior_state, prior_obs, y_obs)

    # Cross-covariance with observation should be zero.
    B_zy = cov(posterior, prior_obs; dims=2)
    @test norm(B_zy) < 1e-3

    # Compare to expected solution.
    z = prior_state .- B_xy * (B_y \ prior_obs)
    @test norm(cov(posterior; dims=2)) ≈ norm(cov(z; dims=2))
    @test posterior ≈ z

    B_zy = cov(z, prior_obs; dims=2)
    @test norm(B_zy) < 1e-3

    # Get and set parameters.
    data_final = deepcopy(get_data(estimator))
    @test any(p.data != p2.data for (p, p2) in zip(data_initial, data_final))

    set_data!(estimator, data_initial)
    @test all(
        p.data == p2.data for (p, p2) in zip(data_initial, deepcopy(get_data(estimator)))
    )

    set_data!(estimator, data_final)
    @test all(p.data == p2.data for (p, p2) in zip(data_final, deepcopy(get_data(estimator))))
end

@testset "conditional_linear assimilate: random init" begin
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
    network_config = ConditionalLinearOptions(; random_init=true)
    network = NetworkConditionalLinear(network_config)

    optimizer_config = OptimizerOptions(; method="descent", lr=1e-1)
    optimizer = create_optimizer(optimizer_config)

    device = cpu
    training_config = TrainingOptions(;
        n_epochs=100,
        num_post_samples=7,
        noise_lev_y=0e-3,
        noise_lev_x=0e-3,
        batch_size=N,
        validation_perc=1.0,
        reset_weights=true,
        reset_optimizer=true,
    )
    estimator = NormalizingFlowFilter(network, optimizer; device, training_config)

    # Get parameters.
    data_initial = deepcopy(get_data(estimator))
    data_initial2 = deepcopy(get_data(estimator))
    @test all(p.data == p2.data for (p, p2) in zip(data_initial, data_initial2))

    # Assimilate.
    posterior = assimilate_data(estimator, prior_state, prior_obs, y_obs)

    # Cross-covariance with observation should be zero.
    B_zy = cov(posterior, prior_obs; dims=2)
    @test norm(B_zy) < 1e-3

    # Compare to expected solution.
    z = prior_state .- B_xy * (B_y \ prior_obs)
    @test norm(cov(posterior; dims=2)) ≈ norm(cov(z; dims=2))
    @test posterior ≈ z

    B_zy = cov(z, prior_obs; dims=2)
    @test norm(B_zy) < 1e-3

    # Get and set parameters.
    data_final = deepcopy(get_data(estimator))
    @test any(p.data != p2.data for (p, p2) in zip(data_initial, data_final))

    set_data!(estimator, data_initial)
    @test all(
        p.data == p2.data for (p, p2) in zip(data_initial, deepcopy(get_data(estimator)))
    )

    set_data!(estimator, data_final)
    @test all(p.data == p2.data for (p, p2) in zip(data_final, deepcopy(get_data(estimator))))
end
