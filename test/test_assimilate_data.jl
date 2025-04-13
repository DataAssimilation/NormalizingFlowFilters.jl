using Statistics: mean, var

@testset "assimilate_data" begin
    N = 37
    Nx = 2
    glow_config = ConditionalGlowOptions(; chan_x=Nx, chan_y=Nx)
    network = NetworkConditionalGlow(2, glow_config)

    optimizer_config = OptimizerOptions(; lr=1e-3)
    optimizer = create_optimizer(optimizer_config)

    device = cpu
    training_config = TrainingOptions(;
        n_epochs=3,
        num_post_samples=7,
        noise_lev_y=1e-3,
        noise_lev_x=1e-3,
        batch_size=11,
        validation_perc=0.63,
    )

    filter = NormalizingFlowFilter(network, optimizer; device, training_config)

    # Ensemble members sampled from a unit normal.
    prior_state = randn(Nx, N)

    # Identity observation operator with no noise.
    prior_obs = deepcopy(prior_state)

    # True state is the mean of the prior.
    y_obs = zeros(Nx)

    # Assimilate.
    posterior = assimilate_data(filter, prior_state, prior_obs, y_obs)
end

@testset "assimilate_data 1D" begin
    N = 37
    Nx = 1
    glow_config = ConditionalGlowOptions(; chan_x=Nx, chan_y=Nx)
    network = NetworkConditionalGlow(2, glow_config)

    optimizer_config = OptimizerOptions(; lr=1e-3)
    optimizer = create_optimizer(optimizer_config)

    device = cpu
    training_config = TrainingOptions(;
        n_epochs=3,
        num_post_samples=7,
        noise_lev_y=1e-3,
        noise_lev_x=1e-3,
        batch_size=11,
        validation_perc=0.63,
    )

    filter = NormalizingFlowFilter(network, optimizer; device, training_config)

    # Ensemble members sampled from a unit normal.
    prior_state = randn(Nx, N)

    # Identity observation operator with no noise.
    prior_obs = deepcopy(prior_state)

    # True state is the mean of the prior.
    y_obs = zeros(Nx)

    # Assimilate.
    posterior = assimilate_data(filter, prior_state, prior_obs, y_obs)
end
