using Statistics: mean, cov
using LinearAlgebra: norm, Diagonal, svd
using Random

@testset "conditional_svd" begin
    N = 37
    Nx = 2
    network_config = ConditionalSVDOptions()
    network = NetworkConditionalSVD(network_config)

    optimizer_config = OptimizerOptions(; lr=1e-3)
    optimizer = create_optimizer(optimizer_config)

    device = cpu
    training_config = TrainingOptions(;
        n_epochs=1,
        num_post_samples=7,
        noise_lev_y=1e-3,
        noise_lev_x=1e-3,
        batch_size=11,
        validation_perc=1.0,
        reset_weights=true,
        reset_optimizer=true,
    )

    estimator = NormalizingFlowFilter(network, optimizer; device, training_config)

    # Get parameters.
    data_initial = deepcopy(get_data(estimator))
    data_initial2 = deepcopy(get_data(estimator))
    @test all(p.data == p2.data for (p, p2) in zip(data_initial, data_initial2))

    # Ensemble members sampled from a unit normal.
    Random.seed!(834)
    prior_state = randn(Nx, N)

    # Identity observation operator with noise.
    prior_obs = deepcopy(prior_state) .+ randn(Nx, N)

    # Covariance should be nonzero.
    B_xy = cov(prior_state, prior_obs; dims=2)
    B_y = cov(prior_obs; dims=2)
    @test norm(B_xy) > 1

    # True state is the mean of the prior.
    y_obs = zeros(Nx)

    # Assimilate.
    posterior = assimilate_data(estimator, prior_state, prior_obs, y_obs)

    # Cross-covariance with observation should be zero.
    B_zy = cov(posterior, prior_obs; dims=2)
    @test norm(B_zy) < 1e-5

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
