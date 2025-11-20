# # Description
# This example shows a simple use case for NormalizingFlowFilters.
#
# First, we import the necessary packages.

using Pkg: Pkg
# Pkg.activate(@__DIR__)
# Pkg.instantiate()

using TerminalLoggers: TerminalLogger
using Logging: global_logger
global_logger(TerminalLogger())

using CairoMakie
using GLMakie
Makie.set_active_backend!(CairoMakie)
using NormalizingFlowFilters
using NormalizingFlowFilters: get_batch_size
using Random: randn, seed!
using Statistics: mean, std, cov, var
using Test
using Printf
using Pkg: Pkg
using JLD2
using Flux

using NormalizingFlowFilters: SigmoidLayer, norm
using NormalizingFlowFilters.InvertibleNetworks: get_params, set_params!, Parameter, ResidualBlock, ResidualBlockSkip, LayerStack

@static if VERSION >= v"1.10"
    using PairPlots: PairPlots, pairplot
end

function display_interactive(fig)
    if isinteractive()
        display(fig)
    else
        fig
    end
end

smalltest = get(ENV, "NormalizingFlowFilters_smalltest", "false")
if !(smalltest in ("true", "false"))
    error("Invalid environment variable value NormalizingFlowFilters_smalltest: $smalltest")
end
smalltest = smalltest == "true"


# Then define the filter.
N = smalltest ? 2^4 : 2^9
Nx = 2
in_shape = (1, 1, Nx)
cond_shape = (1, 1, Nx)
seed!(0x84fb4b3c)


# glow_config = ConditionalGlowOptions(;
#     L=1, K=4, chan_x=in_shape[1], chan_y=in_shape[2],
#     residual = ResidualBlockOptions(
#         n_hidden = 64,
#         k1 = 1,
#         p1 = 0,
#         activation = ActivationOptions(type="softplus"),
#         final_activation = ActivationOptions(type="softplus"),
#     )
# )
# network = NetworkConditionalGlow(length(in_shape) - 1, glow_config)

coupling_stack_config = ConditionalCouplingStackOptions(;
    L=1, K=2, chan_x=in_shape[1], chan_y=in_shape[2],
    coupling_network = CouplingLayerOptions(
        split = false,
        subnetwork = ResidualBlockOptions(
            n_hidden = 4,
            k1 = 1,
            p1 = 0,
            # activation = ActivationOptions(type="tanh"),
            # final_activation = ActivationOptions("identity"),
            activation = RQSpline1ActivationOptions(),
            final_activation = RQSpline1ActivationOptions(),
        ),
        # subnetwork = LayerStackOptions(
        #     subnetworks = (
        #         (
        #             # network = LayerConstantOptions(),
        #             network = ResidualBlockOptions(
        #                 n_hidden = 4,
        #                 k1 = 1,
        #                 p1 = 0,
        #                 # activation = ActivationOptions("identity"),
        #                 # final_activation = ActivationOptions("identity"),
        #                 activation = RQSpline1ActivationOptions(),
        #                 final_activation = RQSpline1ActivationOptions(),
        #                 # final_activation = ActivationOptions("tanh"),
        #             ),
        #         ),
        #         (
        #             network = ResidualBlockSkipOptions(
        #                 n_hidden = 4,
        #                 k1 = 1,
        #                 p1 = 0,
        #                 # activation = ActivationOptions("identity"),
        #                 # final_activation = ActivationOptions("identity"),
        #                 activation = RQSpline1ActivationOptions(),
        #                 final_activation = RQSpline1ActivationOptions(),
        #             ),
        #         ),
        #     )
        # ),
        # invertible_network = ConditionalDecorrelationOperatorOptions(),
        invertible_network = AffineCouplingOperatorOptions(
            scale_activation = ActivationOptions("damped_cosh"),
            shift_activation = ActivationOptions("damped_sinh"),
            shift_cond_scalar = false,
            joint_correlation = true,
        ),
        # invertible_network = RQSpline1OperatorOptions(;
        #     affine = AffineCouplingOperatorOptions(
        #         scale_activation = ActivationOptions("damped_cosh"),
        #         shift_activation = ActivationOptions("damped_sinh"),
        #         shift_cond_scalar = false,
        #         joint_correlation = true,
        #         just_shift = true,
        #     ),
        #     constrained_params=true
        # ),
        # invertible_network = RQSpline1OperatorOptions(),
        # subnetwork = LayerConstantOptions(),
    ),
    # cond_network = ActNormOptions(),
    # state_initial_network = ActNormOptions(),
    cond_network = nothing,
    # state_initial_network = nothing,
    state_middle_network = ActNormOptions(),
    state_final_network = ActNormOptions(),
    # state_final_network = nothing,
    # prenetwork = Conv1x1Options(),
    prenetwork = nothing,
)

s = 1f-6 # initialization.scale
initialize!(L) = @warn "Don't know how to initialize something of type $(typeof(L))"
initialize!(L::Parameter) = initialize!(L, size(L.data))

function initialize!(L::Parameter, d_size)
    L.data = s .* randn(Float32, d_size)
end

function initialize!(L::Union{ResidualBlock, ResidualBlockSkip})
    initialize!(L.W1)
    initialize!(L.W2)
    initialize!(L.W3)
    for p in get_params(L.activation)
        initialize!(p, (1, 1, size(L.W2.data)[end]))
    end
    for p in get_params(L.final_activation)
        initialize!(p, (1, 1, size(L.W3.data)[end-1]))
    end
end

function initialize!(L::LayerStack)
    for l in L.subnetworks
        initialize!(l)
    end
end

function my_network_generator(; complexity=coupling_stack_config.K)
    network = NetworkConditionalCouplingStack(in_shape, cond_shape, coupling_stack_config)
    for cl in network.CL
        initialize!(cl.subnetwork)
    end
    return network
end

function my_network_generator(old_network, X_train, Y_train; complexity=coupling_stack_config.K)
    @show complexity
    config = remake_options(coupling_stack_config; K=complexity)
    network = NetworkConditionalCouplingStack(in_shape, cond_shape, config)
    @show network.K

    if !isnothing(old_network)
        # Copy over params from old network.
        copy_i = min(size(old_network.CL, 1), size(network.CL, 1))
        copy_j = min(size(old_network.CL, 2), size(network.CL, 2))
        for i in 1:copy_i
            for j in 1:copy_j
                set_params!(network.CL[i, j], deepcopy(get_params(old_network.CL[i, j])))
                if !isnothing(network.state_networks[i, j])
                    set_params!(network.state_networks[i, j], deepcopy(get_params(old_network.state_networks[i, j])))
                end
            end
        end
        if !isnothing(network.state_final_network)
            set_params!(network.state_final_network, deepcopy(get_params(old_network.state_final_network)))
        end
        if !isnothing(network.cond_network)
            set_params!(network.cond_network, deepcopy(get_params(old_network.cond_network)))
        end
    else
        copy_i = 0
        copy_j = 0
    end

    for i in (copy_i+1):size(network.CL, 1)
        for j in (copy_j+1):size(network.CL, 2)
            initialize!(network.CL[i, j].subnetwork)
        end
    end
    return network
end

@assert in_shape[1:2] == (1, 1)

optimizer_config = OptimizerOptions(;
    lr=1e-3,
    clipnorm_val = 3.0f-1,
    weight_decay = WeightDecayOptions(; active=true, factor=1f-5),
    learning_rate_decay = LearningRateDecayOptions(; active=true, factor=(0.1^0.5), step=3000, minimum=1e-2),
)
optimizer = create_optimizer(optimizer_config)

device = cpu
training_config = TrainingOptions(;
    # n_epochs=smalltest ? 10 : 40000,
    n_epochs=smalltest ? 10 : 20000,
    num_post_samples=1,
    # noise = UnitGaussianNoiseOptions(;
    #     # Uh oh. Independent noise smooths prior (and therefore posterior) but also biases x and y to be unrelated.
    #     x=1e-6,
    #     y=1e-6/sqrt(2),
    # ),
    noise = DataCorrelatedGaussianNoiseOptions(;
        x=1e-1,
        y=1e-1,
        x_correlated = 1f-1,
        y_correlated = 1f-1,
    ),
    # batch=FixedBatchSizeOptions(batch_size=smalltest ? 2^2 : 2^18),
    # batch=FixedBatchSizeOptions(batch_size=2^4),
    batch=FixedNumBatchesOptions(num_batches=2),
    validation_perc=2^(-1),
    early_stopping_training_loss = EarlyStoppingOptions(
        active = true,
        look_backs = ((5000, -1, -1f-6),),
    ),
    early_stopping_validation_loss = EarlyStoppingOptions(
        active = true,
        look_backs = ((5000, -1, -1f-6),),
    ),
    save_best = true,
    hypersearcher = nothing,
    # hypersearcher = HyperComplexitySearcherOptions(;
    #     min_complexity = 1,
    #     max_complexity = 32,
    #     keep_going = 2
    # ),
    reset_optimizer = true,
    reset_weights = false,
)

target_distribution_opts = TargetUnitNormalOptions()
# target_distribution_opts = TargetUnitNormalPlusUniformOptions(; uniform_weight=1e-5)

target_distribution = make_target_distribution(target_distribution_opts)

filter = NormalizingFlowFilter(my_network_generator, target_distribution, optimizer; device, training_config)

function to_table(a; prefix=:x)
    return (; (Symbol(prefix, i) => row for (i, row) in enumerate(eachrow(a)))...)
end
combine_tables(a, b) = (; a..., b...)
combine_tables(a, b...) = combine_tables(combine_tables(a, b[1]), b[2:end]...)

# @static if VERSION >= v"1.10"
#     fig = pairplot(
#         table_prior_state => (
#             PairPlots.Hist(; colormap=:Blues),
#             PairPlots.MarginDensity(;
#                 bandwidth=kde_bandwidth, color=RGBf((49, 130, 189) ./ 255...)
#             ),
#             PairPlots.TrendLine(; color=:red),
#             PairPlots.Correlation(),
#             PairPlots.Scatter(),
#         ),
#         PairPlots.Truth(
#             table_prior_state_mean; label="Mean Values", color=(:black, 0.5), linewidth=4
#         ),
#     )
#     supertitle = Label(fig[0, :], "prior state"; fontsize=30)
#     resize_to_layout!(fig)
#     display_interactive(fig)
# end
# error("done")

# We generate an ensemble.

prior_state = randn(Float64, Nx, N) # ./ sqrt(2) .+ 2
# prior_state[2, :] .*= 1e-3
# prior_state[:, 1:ceil(Int, N/2)] .= randn(Float64, Nx, ceil(Int, N/2)) ./ sqrt(2) .- 2

# Apply observation operator.
obs_func(x) = x .+ randn(Float64, size(x))
# obs_func(x) = (1 .+ x) .^ 2 .+ randn(Float64, size(x)) ./ 2
# obs_func(x) = x .^ 2 .+ randn(Float64, size(x))
prior_obs = obs_func.(prior_state)
# prior_obs[1, :] = obs_func.(prior_state[1, :])
# prior_obs[2, :] = prior_state[2, :] + randn(Float64, size(prior_state[2, :]))

prior_state .-= mean(prior_state; dims=2)
prior_state ./= std(prior_state; dims=2)

prior_obs .-= mean(prior_obs; dims=2)
prior_obs ./= std(prior_obs; dims=2)

table_prior_state = to_table(prior_state)

table_prior_obs = to_table(prior_obs; prefix=:y)

kde_bandwidth = training_config.noise.x / PairPlots.KernelDensity.default_bandwidth(prior_state[1, :])
table_prior_state_mean = to_table(mean(prior_state; dims=2)[:, 1])

# @static if VERSION >= v"1.10"
#     fig = pairplot(
#         table_prior_obs => (
#             PairPlots.Hist(; colormap=:Blues),
#             PairPlots.MarginDensity(;
#                 bandwidth=kde_bandwidth, color=RGBf((49, 130, 189) ./ 255...)
#             ),
#             PairPlots.TrendLine(; color=:red),
#             PairPlots.Correlation(),
#             PairPlots.Scatter(),
#         ),
#         PairPlots.Truth(
#             table_prior_obs_mean; label="Mean Values", color=(:black, 0.5), linewidth=4
#         ),
#     )
#     supertitle = Label(fig[0, :], "prior observation"; fontsize=30)
#     resize_to_layout!(fig)
#     display_interactive(fig)
# end

X = reshape(prior_state, (1, 1, size(prior_state, 1), size(prior_state, 2)))
Y = reshape(prior_obs, (1, 1, size(prior_obs, 1), size(prior_obs, 2)))
Z_initial, ZY_initial, logdet_initial = filter.coupling_network_device.forward(X, Y)

Z_initial = Z_initial[1, 1, :, :]
table_Z_initial = to_table(Z_initial; prefix=:z)
combo_table = combine_tables(table_prior_state, table_prior_obs, table_Z_initial)

combo_table_mean = (; (k => mean(v) for (k,v) in pairs(combo_table))...)
fig = pairplot(
    combo_table => (
        PairPlots.Hist(; colormap=:Blues),
        PairPlots.MarginDensity(;
            bandwidth=kde_bandwidth, color=RGBf((49, 130, 189) ./ 255...)
        ),
        PairPlots.TrendLine(; color=:red),
        PairPlots.Correlation(),
        PairPlots.Scatter(),
    ),
    PairPlots.Truth(
        combo_table_mean; label="Mean Values", color=(:black, 0.5), linewidth=4
    ),
)
supertitle = Label(fig[0, :], "Initial distributions"; fontsize=30)
resize_to_layout!(fig)
display_interactive(fig)

# error("done")

# Look at how observations correlate to data.
exx_state = extrema(prior_state; dims=2)
exx_obs = extrema(prior_obs; dims=2)
if true
    for x_idx in 1:Nx
        for y_idx in 1:Nx
            # linspace_x = range(first(exx_state[x_idx, 1]), last(exx_state[x_idx,1]), length=1000)
            # linspace_y = range(first(exx_obs[y_idx, 1]), last(exx_obs[y_idx,1]), length=500)
            linspace_x = range(-6, 6, length=100)
            linspace_y = range(-6, 6, length=50)

            matrix_initial = [
                let
                    x = reshape(zeros(Nx), (1, 1, Nx, 1))
                    y = reshape(zeros(Nx), (1, 1, Nx, 1))
                    x[x_idx] = x_scalar
                    y[y_idx] = y_scalar
                    z, zy, logdet = filter.coupling_network_device.forward(x, y)
                    nlog_p_z, _ = compute_negative_log_density_with_gradient(filter.target_distribution, z, zy)
                    z, zy, logdet, -only(nlog_p_z) + logdet # -norm(z)^2/2 + logdet - log(2*pi)/2
                end
                for (x_scalar,  y_scalar) in Iterators.product(linspace_x, linspace_y)
            ]
            linspace_initial_z = getindex.(getindex.(matrix_initial, 1), x_idx) .|> only
            linspace_initial_zy = getindex.(getindex.(matrix_initial, 2), y_idx) .|> only
            linspace_initial_logdet = getindex.(matrix_initial, 3) .|> only
            linspace_initial_log_p_xy = getindex.(matrix_initial, 4) .|> only

            linspace_initial_p_xy = exp.(linspace_initial_log_p_xy)
            linspace_initial_p_xy ./= maximum(linspace_initial_p_xy; dims=1)

            fig = Figure()

            ax = Axis(fig[1,1])
            ax.xlabel = "x"
            ax.ylabel = "y"
            hm = heatmap!(ax, linspace_x, linspace_y, linspace_initial_z; colormap=:cividis)
            Colorbar(fig[1, 2], hm)
            supertitle = Label(fig[1, 1:2, Top()], "z"; fontsize=16)

            ax = Axis(fig[1,3])
            ax.xlabel = "x"
            ax.ylabel = "y"
            hm = heatmap!(ax, linspace_x, linspace_y, linspace_initial_zy; colormap=:cividis)
            Colorbar(fig[1, 4], hm)
            supertitle = Label(fig[1, 3:4, Top()], "zy"; fontsize=16)

            ax = Axis(fig[2,1])
            ax.xlabel = "x"
            ax.ylabel = "y"
            hm = heatmap!(ax, linspace_x, linspace_y, linspace_initial_logdet; colormap=:magma)

            Colorbar(fig[2, 2], hm)
            supertitle = Label(fig[2, 1:2, Top()], "log det dz/dx"; fontsize=16)

            ax = Axis(fig[2,3])
            ax.xlabel = "x"
            ax.ylabel = "y"
            hm = heatmap!(ax, linspace_x, linspace_y, linspace_initial_p_xy; colormap=:magma, colorrange=(0, 1))
            Colorbar(fig[2, 4], hm)
            supertitle = Label(fig[2, 3:4, Top()], "p(x|y) (rescaled)"; fontsize=16)
            supertitle = Label(fig[0, :], "network initial x_$x_idx, y_$y_idx"; fontsize=20)
            resize_to_layout!(fig)
            display_interactive(fig)
        end
    end
end

# error("done")

# error("stop")

# @static if VERSION >= v"1.10"
#     combo_table = combine_tables(table_prior_state, table_prior_obs)
#     combo_table_mean = combine_tables(table_prior_state_mean, table_prior_obs_mean)
#     fig = pairplot(
#         combo_table => (
#             PairPlots.Hist(; colormap=:Blues),
#             PairPlots.MarginDensity(;
#                 bandwidth=kde_bandwidth, color=RGBf((49, 130, 189) ./ 255...)
#             ),
#             PairPlots.TrendLine(; color=:red),
#             PairPlots.Correlation(),
#             PairPlots.Scatter(),
#         ),
#         PairPlots.Truth(
#             combo_table_mean; label="Mean Values", color=(:black, 0.5), linewidth=4
#         ),
#     )
#     supertitle = Label(fig[0, :], "prior state-observation"; fontsize=30)
#     resize_to_layout!(fig)
#     display_interactive(fig)
# end

# Then we assimilate an observation. Here, we just pick an arbitrary one.
y_obs = zeros(Nx)
# y_obs = obs_func(prior_state[:, rand(1:size(prior_state, 2))])
log_data = Dict{Symbol,Any}()
posterior = assimilate_data(filter, prior_state, prior_obs, y_obs, log_data)

stats = [
    let
        posterior = assimilate_data(filter, prior_state, prior_obs, fill(Float64(y_obs_i), Nx); train=false)
        (; mean = mean(posterior), var = var(posterior))
    end
    for y_obs_i in -3:7
]

for i in 1:length(stats)
    println(stats[i])
end

# Plot some training metrics.
if length(log_data[:coupling_network][:training][:loss]) > 0
    common_kwargs = (; linewidth=3)

    fig = Figure()

    ax = Axis(fig[1, 1])

    misfit_train = log_data[:coupling_network][:training][:loss]
    misfit_test = log_data[:coupling_network][:testing][:loss]

    ## The training metrics are recorded for each batch, but test metrics are computed each
    ## epoch.
    num_batches = length(misfit_train)
    num_epochs = length(misfit_test)
    batches_per_epochs = div(num_batches, num_epochs)

    test_epochs = 1:num_epochs
    train_epochs = (1:num_batches) ./ batches_per_epochs

    lines_train = lines!(ax, train_epochs, misfit_train; label="train", common_kwargs...)
    lines_test = lines!(ax, test_epochs, misfit_test; label="test", common_kwargs...)

    ax.xlabel = "epoch number"
    ax.ylabel = "loss: 2-norm"
    fig[1, end + 1] = Legend(fig, ax; labelsize=14, unique=true)

    N_train = round(Int, N * training_config.validation_perc)
    N_valid = N - N_train
    ax.title = "Training: $N_train, Validation: $N_valid, Batch size: $(min(get_batch_size(training_config.batch, N_train), max(N_train, N_valid)))"

    ax = Axis(fig[2, 1])

    x = log_data[:coupling_network][:training][:logdet]
    lines!(ax, train_epochs, x; color=lines_train.color, common_kwargs...)

    x = log_data[:coupling_network][:testing][:logdet]
    lines!(ax, test_epochs, x; color=lines_test.color, common_kwargs...)

    ax.xlabel = "epoch number"
    ax.ylabel = "loss: log determinant"


    ax = Axis(fig[3, 1])

    x = log_data[:coupling_network][:training][:logdet] .+ misfit_train
    lines!(ax, train_epochs, x; color=lines_train.color, common_kwargs...)

    x = log_data[:coupling_network][:testing][:logdet] .+ misfit_test
    lines!(ax, test_epochs, x; color=lines_test.color, common_kwargs...)

    ax.xlabel = "epoch number"
    ax.ylabel = "loss: total"

    supertitle = Label(fig[0, :], "Training log"; fontsize=30)
    resize_to_layout!(fig)
    display_interactive(fig)
end

@show y_obs

# Visualize conditionally normalized state.
X = prior_state
Y = prior_obs
X = reshape(X, (1, 1, size(X, 1), size(X, 2)))
Y = reshape(Y, (1, 1, size(Y, 1), size(Y, 2)))
Z = normalize_samples(
    filter.coupling_network_device,
    X,
    Y,
    size(X);
    device=filter.device,
    num_samples=N,
    batch_size=get_batch_size(filter.training_config.batch, size(prior_state)[end]),
)
Z = Z[1, 1, :, :]
table_Z_trained = to_table(Z; prefix=:z)

for y_obs_i in -3:6
    posterior = assimilate_data(filter, prior_state, prior_obs, fill(Float64(y_obs_i), Nx); train=false)

    # Z .+= randn(size(Z)) .* 1e-15

    table_posterior = to_table(posterior; prefix=:xp)

    combo_table = combine_tables(table_prior_state, table_prior_obs, table_Z_trained, table_posterior)
    combo_table_mean = (; (k => mean(v) for (k,v) in pairs(combo_table))...)

    fig = pairplot(
        combo_table => (
            PairPlots.Hist(; colormap=:Blues),
            PairPlots.MarginDensity(;
                bandwidth=kde_bandwidth, color=RGBf((49, 130, 189) ./ 255...)
            ),
            PairPlots.TrendLine(; color=:red),
            PairPlots.Correlation(),
            PairPlots.Scatter(),
        ),
        PairPlots.Truth(
            combo_table_mean; label="Mean Values", color=(:black, 0.5), linewidth=4
        ),
    )
    supertitle = Label(fig[0, :], "Trained distributions for y=$y_obs_i"; fontsize=30)
    resize_to_layout!(fig)
    display_interactive(fig)
end

posterior = assimilate_data(filter, prior_state, prior_obs, fill(Float64(0), Nx); train=false)

# Look at prior mean.
display_interactive(mean(prior_state; dims=2))

# Look at prior covariance.
display_interactive(cov(prior_state; dims=2))

# Look at latent mean.
display_interactive(mean(Z; dims=2))

# Look at latent covariance.
display_interactive(cov(Z; dims=2))

# Look at posterior mean.
display_interactive(mean(posterior; dims=2))

# Look at posterior covariance.
display_interactive(cov(posterior; dims=2))

import Statistics
K = Statistics.cov(prior_state, prior_obs; dims=2) ./ Statistics.var(prior_obs; dims=2)
posterior_kf = prior_state .+ K * (y_obs .- prior_obs)

# Visualize posterior.
for x_idx in 1:Nx
    for y_idx in 1:Nx
        # linspace_x = range(first(exx_state[x_idx, 1]), last(exx_state[x_idx,1]), length=1000)
        # linspace_y = range(first(exx_obs[y_idx, 1]), last(exx_obs[y_idx,1]), length=500)
        linspace_x = range(-6, 6, length=100)
        linspace_y = range(-6, 6, length=50)

        matrix_trained = [
            let
                x = reshape(zeros(Nx), (1, 1, Nx, 1))
                y = reshape(zeros(Nx), (1, 1, Nx, 1))
                x[x_idx] = x_scalar
                y[y_idx] = y_scalar
                z, zy, logdet = filter.coupling_network_device.forward(x, y)
                nlog_p_z, _ = compute_negative_log_density_with_gradient(filter.target_distribution, z, zy)
                z, zy, logdet, -only(nlog_p_z) + logdet # -norm(z)^2/2 + logdet - log(2*pi)/2
            end
            for (x_scalar,  y_scalar) in Iterators.product(linspace_x, linspace_y)
        ]
        linspace_trained_z = getindex.(getindex.(matrix_trained, 1), x_idx) .|> only
        linspace_trained_zy = getindex.(getindex.(matrix_trained, 2), y_idx) .|> only
        linspace_trained_logdet = getindex.(matrix_trained, 3) .|> only
        linspace_trained_log_p_xy = getindex.(matrix_trained, 4) .|> only

        linspace_trained_p_xy = exp.(linspace_trained_log_p_xy)
        linspace_trained_p_xy ./= maximum(linspace_trained_p_xy; dims=1)

        fig = Figure()

        ax = Axis(fig[1,1])
        ax.xlabel = "x"
        ax.ylabel = "y"
        hm = heatmap!(ax, linspace_x, linspace_y, linspace_trained_z; colormap=:cividis)
        Colorbar(fig[1, 2], hm)
        supertitle = Label(fig[1, 1:2, Top()], "z"; fontsize=16)

        ax = Axis(fig[1,3])
        ax.xlabel = "x"
        ax.ylabel = "y"
        hm = heatmap!(ax, linspace_x, linspace_y, linspace_trained_zy; colormap=:cividis)
        Colorbar(fig[1, 4], hm)
        supertitle = Label(fig[1, 3:4, Top()], "zy"; fontsize=16)

        ax = Axis(fig[2,1])
        ax.xlabel = "x"
        ax.ylabel = "y"
        hm = heatmap!(ax, linspace_x, linspace_y, linspace_trained_logdet; colormap=:magma)
        Colorbar(fig[2, 2], hm)
        supertitle = Label(fig[2, 1:2, Top()], "log det dz/dx"; fontsize=16)

        ax = Axis(fig[2,3])
        ax.xlabel = "x"
        ax.ylabel = "y"
        hm = heatmap!(ax, linspace_x, linspace_y, linspace_trained_p_xy; colormap=:magma, colorrange=(0, 1))
        Colorbar(fig[2, 4], hm)
        supertitle = Label(fig[2, 3:4, Top()], "p(x|y) (rescaled)"; fontsize=16)
        supertitle = Label(fig[0, :], "network trained x_$x_idx, y_$y_idx"; fontsize=20)
        resize_to_layout!(fig)
        display_interactive(fig)
    end
end

if Nx == 1
    # fig = Figure();

    # ax = Axis(fig[1,1])
    # y = exp.(linspace_trained_log_p_xy)
    # y ./= maximum(y; dims=1)
    # hm = heatmap!(ax, linspace_x, linspace_y, y; colormap=:magma)
    # ax.xlabel = "x"
    # ax.ylabel = "y"
    # Colorbar(fig[1, 2], hm)

    # ax = Axis(fig[1, 3])
    # ax.xlabel = "x"
    # ax.ylabel = "p(x|y)"
    # linspace_trained_p_xy = exp.(linspace_trained_log_p_xy)
    # n = min(10, length(linspace_y))
    # colors = Makie.cgrad(:viridis, n)
    # for i in 1:n
    #     j = Int64(round(i * length(linspace_y) / n))
    #     label = @sprintf("y=%.2f", linspace_y[j])
    #     color = colors[i/n]
    #     lines!(ax, linspace_x, linspace_trained_p_xy[:, j]; label, color)
    # end
    # fig[1, 4] = Legend(fig, ax; labelsize=14, unique=true)
    # supertitle = Label(fig[0, :], "trained network p(x|y)"; fontsize=20)
    # resize_to_layout!(fig)
    # display_interactive(fig)
end

@test true;
