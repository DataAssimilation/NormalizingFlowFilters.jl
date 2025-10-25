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
using Statistics: mean, std, cov
using Test
using Printf
using Pkg: Pkg
using JLD2
using Flux

using NormalizingFlowFilters: SigmoidLayer, norm
using NormalizingFlowFilters.InvertibleNetworks: get_params, set_params!

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


using NormalizingFlowFilters.InvertibleNetworks: ActNorm, CouplingLayerGlow, Parameter, sample_banana, log_likelihood, ∇log_likelihood, clear_grad!, tensor_cat
seed!(11)

# # Define network
# nx = 1; ny = 1; n_in = 2
# n_hidden = 64
# batchsize = 2^6
# depth = 4
# AN = Array{ActNorm}(undef, depth)
# L = Array{CouplingLayerGlow}(undef, depth)
# Params = Array{Parameter}(undef, 0)

# # Create layers
# for j=1:depth
#     AN[j] = ActNorm(n_in; logdet=true)
#     L[j] = CouplingLayerGlow(n_in, n_hidden; k1=1, k2=1, p1=0, p2=0, logdet=true, freeze_conv=true)
#     L[j].C.v1.data .= 1
#     L[j].C.v2.data .= [0,1]
#     L[j].C.v3.data .= [1,0]
#     # Collect parameters
#     global Params = cat(Params, get_params(AN[j]); dims=1)
#     global Params = cat(Params, get_params(L[j]); dims=1)
# end

# # Forward pass
# function forward(X)
#     logdet = 0f0
#     for j=1:depth
#         X_, logdet2 = L[j].forward(X)
#         X, logdet1 = AN[j].forward(X_)
#         logdet += (logdet1 + logdet2)
#     end
#     return X, logdet
# end

# # Backward pass
# function backward(ΔX, X)
#     for j=depth:-1:1
#         ΔX_, X_ = AN[j].backward(ΔX, X)
#         ΔX, X = L[j].backward(ΔX_, X_)
#     end
#     return ΔX, X
# end

# # Loss
# function loss(X)
#     Y, logdet = forward(X)
#     # Y = sum(Y; dims=3)
#     f = -log_likelihood(Y) - logdet
#     ΔY = -∇log_likelihood(Y)
#     ΔX = backward(ΔY, Y)[1]
#     return f, ΔX
# end

# tmp = sample_banana(2^20)
# b2_mean = mean(tmp[:, :, 2:2, :]; dims=4)
# b2_std = std(tmp[:, :, 2:2, :]; dims=4)
# @show b2_mean b2_std
# tmp = nothing

# function sample_banana2(n)
#     X = sample_banana(n)
#     X[:, :, 1, :] = sample_banana(n)[:, :, 2, :]
#     X = cat(X, cat(X[:, :, 2:2, :], X[:, :, 1:1, :]; dims=3); dims=4)
#     X .+= 1e-3 * randn(size(X))
#     X .-= b2_mean
#     X ./= b2_std
#     return X
# end

# # Initialize parameters
# X = sample_banana2(batchsize)
# loss(X);

# # Training
# maxiter = 10000
# opt = Flux.Optimise.Adam(1f-4)
# fval = zeros(Float32, maxiter)


# for j=1:maxiter

#     # Evaluate objective and gradients
#     X = sample_banana2(batchsize)
#     # X[1, 1, 1, :] .*= 0
#     fval[j] = loss(X)[1]

#     println(j, ", ", fval[j])

#     # Update params
#     for p in Params
#         Flux.update!(opt, p.data, p.grad)
#     end
#     clear_grad!(Params)
# end

# ####################################################################################################

# # Testing
# test_size = 2^14
# X = sample_banana2(test_size)
# # X[1, 1, 1, :] .= 1e-10 .* randn(size(X[1, 1, 1, :]))
# Y_ = forward(X)[1]
# # Y_[1, 1, 1, :] .= 1e-10 .* randn(size(X[1, 1, 1, :]))
# Y = randn(Float32, 1, 1, 2, test_size)
# # Y[1, 1, 1, :] .= 1e-10 .* randn(size(X[1, 1, 1, :]))
# X_ = backward(Y, Y)[2]
# # X_[1, 1, 1, :] .= 1e-10 .* randn(size(X[1, 1, 1, :]))

# # fig = Figure()

# # ax1 = Axis(fig[1,1])
# # scatter!(ax1, X[1, 1, 1, :], X[1, 1, 2, :])
# # ax1.title = L"Data space: $x \sim \hat{p}_X$"
# # # xlims!(ax1, [-3.5, 3.5])
# # xlims!(ax1, [0,50])
# # ylims!(ax1, [0,50])

# # ax2 = Axis(fig[1,2])
# # scatter!(ax2, Y_[1, 1, 1, :], Y_[1, 1, 2, :])
# # ax2.title = L"Latent space: $z = f(x)$"
# # xlims!(ax2, [-3.5, 3.5])
# # ylims!(ax2, [-3.5, 3.5])

# # ax3 = Axis(fig[2,1])
# # scatter!(ax3, X_[1, 1, 1, :], X_[1, 1, 2, :])
# # ax3.title = L"Data space: $x = f^{-1}(z)$"
# # # xlims!(ax3, [-3.5, 3.5])
# # xlims!(ax3, [0,50])
# # ylims!(ax3, [0,50])

# # ax4 = Axis(fig[2,2])
# # scatter!(ax4, Y[1, 1, 1, :], Y[1, 1, 2, :])
# # ax4.title = L"Latent space: $z \sim \hat{p}_Z$"
# # xlims!(ax4, [-3.5, 3.5])
# # ylims!(ax4, [-3.5, 3.5])

# # display_interactive(fig)


# # X = sample_banana2(test_size)
# # # X[1, 1, 1, :] .= 1e-10 .* randn(size(X[1, 1, 1, :]))
# # Y_ = forward(X)[1]
# # # Y_[1, 1, 1, :] .= 1e-10 .* randn(size(X[1, 1, 1, :]))
# # Y = randn(Float32, 1, 1, 2, test_size)
# # # Y[1, 1, 1, :] .= 1e-10 .* randn(size(X[1, 1, 1, :]))
# # X_ = backward(Y, Y)[2]
# # # X_[1, 1, 1, :] .= 1e-10 .* randn(size(X[1, 1, 1, :]))

# combo_table = (;
#     X = X[1, 1, 1, :],
#     Y = X[1, 1, 2, :],
#     ZX = Y_[1, 1, 1, :],
#     ZY = Y_[1, 1, 2, :],
# )

# combo_table_mean = (;(k => mean(v) for (k,v) in pairs(combo_table))...)
# fig = pairplot(
#     combo_table => (
#         PairPlots.Hist(; colormap=:Blues),
#         PairPlots.MarginDensity(;
#             bandwidth=1e-3, color=RGBf((49, 130, 189) ./ 255...)
#         ),
#         PairPlots.TrendLine(; color=:red),
#         PairPlots.Correlation(),
#         PairPlots.Scatter(),
#     ),
#     PairPlots.Truth(
#         combo_table_mean; label="Mean Values", color=(:black, 0.5), linewidth=4
#     ),
# );
# supertitle = Label(fig[0, :], "Forward distributions"; fontsize=30)
# resize_to_layout!(fig)
# display_interactive(fig)

# combo_table = (;
#     X = X_[1, 1, 1, :],
#     Y = X_[1, 1, 2, :],
#     ZX = Y[1, 1, 1, :],
#     ZY = Y[1, 1, 2, :],
# )

# combo_table_mean = (;(k => mean(v) for (k,v) in pairs(combo_table))...)
# fig = pairplot(
#     combo_table => (
#         PairPlots.Hist(; colormap=:Blues),
#         PairPlots.MarginDensity(;
#             bandwidth=1e-3, color=RGBf((49, 130, 189) ./ 255...)
#         ),
#         PairPlots.TrendLine(; color=:red),
#         PairPlots.Correlation(),
#         PairPlots.Scatter(),
#     ),
#     PairPlots.Truth(
#         combo_table_mean; label="Mean Values", color=(:black, 0.5), linewidth=4
#     ),
# );
# supertitle = Label(fig[0, :], "Reverse distributions"; fontsize=30)
# resize_to_layout!(fig)
# display_interactive(fig)

# # error("done")

# error("done with that")

# Then define the filter.
N = smalltest ? 2^4 : 2^12
Nx = 1
in_shape = (1, 1, Nx)
cond_shape = (1, 1, Nx)
seed!(0x84fb4b2c)


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

correlation_config = ConditionalCouplingStackOptions(;
    L=1, K=4, chan_x=in_shape[1], chan_y=in_shape[2],
    coupling_network = CouplingLayerOptions(
        subnetwork = ResidualBlockOptions(
            n_hidden = 2,
            k1 = 1,
            p1 = 0,
            activation = ActivationOptions(type="softplus"),
            final_activation = ActivationOptions("identity"),
        ),
        invertible_network = RQSpline1OperatorOptions(;
            affine = AffineCouplingOperatorOptions(
                scale_activation = ActivationOptions("damped_cosh"),
                shift_activation = ActivationOptions("damped_sinh"),
                shift_cond_scalar = false,
                joint_correlation = true,
            ),
            constrained_params=true
        ),
        # subnetwork = LayerConstantOptions(),
    ),
    # cond_network = ActNormOptions(),
    # state_initial_network = ActNormOptions(),
    cond_network = nothing,
    state_initial_network = nothing,
    state_middle_network = ActNormOptions(),
    state_final_network = ActNormOptions(),
    # prenetwork = Conv1x1Options(),
    prenetwork = nothing,
)

network = NetworkConditionalCouplingStack(in_shape, cond_shape, correlation_config)

# Initialize network to do nothing.
for cl in network.CL
    # println(cl)
    cl.subnetwork.W1.data .= 1e-4 * randn(size(cl.subnetwork.W1.data))
    cl.subnetwork.W2.data .= 1e-4 * randn(size(cl.subnetwork.W2.data))
    cl.subnetwork.W3.data .= 1e-4 * randn(size(cl.subnetwork.W3.data))
    # cl.subnetwork.b1.data .= 1e-3 * randn(size(cl.subnetwork.b1.data))
    # cl.subnetwork.b2.data .= 1e-3 * randn(size(cl.subnetwork.b2.data))
    # cl.subnetwork.val.data .= 1e-3 * randn(size(cl.subnetwork.val.data))
end


@show get_params(network)

optimizer_config = OptimizerOptions(; lr=1e-3)
optimizer = create_optimizer(optimizer_config)

device = cpu
training_config = TrainingOptions(;
    # n_epochs=smalltest ? 10 : 40000,
    n_epochs=smalltest ? 10 : 1000,
    num_post_samples=1,
    noise_lev_y=1e-6,
    noise_lev_x=1e-6,
    # batch=FixedBatchSizeOptions(batch_size=smalltest ? 2^2 : 2^18),
    # batch=FixedBatchSizeOptions(batch_size=2^4),
    batch=FixedNumBatchesOptions(num_batches=2),
    validation_perc=2^(-1),
    early_stopping_training_loss = EarlyStoppingOptions(
        active = false,
        look_backs = ((20, 0.5, 0.1),(100, 0.5, -1f-6)),
    ),
    early_stopping_validation_loss = EarlyStoppingOptions(
        active = false,
        look_backs = ((20, 0.5, 0.1),(100, 0.5, -1f-6)),
    ),
)

filter = NormalizingFlowFilter(network, optimizer; device, training_config)

# save_file = "res11_4.jld2"
# save_file = "simpler3.jld2"
# save_file = "good4.jld2"
# save_file = "gaussian_square_nojoint_6.jld2"
# save_file = "banana_4.jld2"
# save_file = "banana_glow_2.jld2"
# save_file = "banana2d_conditional_3.jld2"
# if isfile(save_file)
#     d = load(save_file)
#     set_data!(filter, d["data"])
# else
#     error("Couldn't find $save_file")
# end


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

prior_state = randn(Float64, Nx, N) ./ sqrt(2)
# prior_state[2, :] .*= 1e-3
# prior_state[:, 1:ceil(Int, N/2)] .= randn(Float64, Nx, ceil(Int, N/2)) ./ sqrt(2) .- 2

# Apply observation operator.
obs_func(x) = x .+ randn(Float64, size(x))
# obs_func(x) = x .^ 2 .+ randn(Float64, size(x)) ./ 2
prior_obs = obs_func.(prior_state)
# prior_obs[1, :] = obs_func.(prior_state[1, :])
# prior_obs[2, :] = prior_state[2, :] + randn(Float64, size(prior_state[2, :]))

prior_state .-= mean(prior_state; dims=2)
prior_state ./= std(prior_state; dims=2)

prior_obs .-= mean(prior_obs; dims=2)
prior_obs ./= std(prior_obs; dims=2)

table_prior_state = to_table(prior_state)

table_prior_obs = to_table(prior_obs; prefix=:y)

kde_bandwidth = training_config.noise_lev_x / PairPlots.KernelDensity.default_bandwidth(prior_state[1, :])
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
Z_initial, ZY_initial, logdet_initial = filter.network_device.forward(X, Y)

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

if Nx == 1
    # Look at how observations correlate to data.
    exx = extrema(prior_state; dims=2)
    linspace_x = range(first(exx[1, 1]), last(exx[1,1]), length=1000)

    exx = extrema(prior_obs; dims=2)
    linspace_y = range(first(exx[1, 1]), last(exx[1,1]), length=500)
    matrix_initial = [
        let
            x = reshape([x], (1, 1, size(x, 1), size(x, 2)))
            y = reshape([y], (1, 1, size(y, 1), size(y, 2)))
            z, zy, logdet = filter.network_device.forward(x, y)
            z, zy, logdet, -norm(z)^2/2 + logdet - log(2*pi)/2
        end
        for (x,  y) in Iterators.product(linspace_x, linspace_y)
    ]
    linspace_initial_z = getindex.(matrix_initial, 1) .|> only
    linspace_initial_zy = getindex.(matrix_initial, 2) .|> only
    linspace_initial_logdet = getindex.(matrix_initial, 3) .|> only
    linspace_initial_log_p_xy = getindex.(matrix_initial, 4) .|> only

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
    hm = heatmap!(ax, linspace_x, linspace_y, exp.(linspace_initial_log_p_xy); colormap=:magma)
    Colorbar(fig[2, 4], hm)
    supertitle = Label(fig[2, 3:4, Top()], "p(x|y)"; fontsize=16)
    supertitle = Label(fig[0, :], "initial network"; fontsize=20)
    resize_to_layout!(fig)
    display_interactive(fig)
end



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

# Plot some training metrics.
if length(log_data[:network_training][:training][:loss]) > 0
    common_kwargs = (; linewidth=3)

    fig = Figure()

    ax = Axis(fig[1, 1])

    misfit_train = log_data[:network_training][:training][:loss]
    misfit_test = log_data[:network_training][:testing][:loss]

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

    x = log_data[:network_training][:training][:logdet]
    lines!(ax, train_epochs, x; color=lines_train.color, common_kwargs...)

    x = log_data[:network_training][:testing][:logdet]
    lines!(ax, test_epochs, x; color=lines_test.color, common_kwargs...)

    ax.xlabel = "epoch number"
    ax.ylabel = "loss: log determinant"


    ax = Axis(fig[3, 1])

    x = log_data[:network_training][:training][:logdet] .+ misfit_train
    lines!(ax, train_epochs, x; color=lines_train.color, common_kwargs...)

    x = log_data[:network_training][:testing][:logdet] .+ misfit_test
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
    filter.network_device,
    X,
    Y,
    size(X);
    device=filter.device,
    num_samples=N,
    batch_size=get_batch_size(filter.training_config.batch, size(prior_state)[end]),
)
Z = Z[1, 1, :, :]
# Z .+= randn(size(Z)) .* 1e-15

table_Z_trained = to_table(Z; prefix=:z)
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
supertitle = Label(fig[0, :], "Trained distributions"; fontsize=30)
resize_to_layout!(fig)
display_interactive(fig)

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

if Nx == 1
    # Visualize posterior.
    matrix_trained = [
        let
            x = reshape([x], (1, 1, size(x, 1), size(x, 2)))
            y = reshape([y], (1, 1, size(y, 1), size(y, 2)))
            z, zy, logdet = filter.network_device.forward(x, y)
            z, zy, logdet, -norm(z)^2/2 + logdet - log(2*pi)/2
        end
        for (x,  y) in Iterators.product(linspace_x, linspace_y)
    ]
    linspace_trained_z = getindex.(matrix_trained, 1) .|> only
    linspace_trained_zy = getindex.(matrix_trained, 2) .|> only
    linspace_trained_logdet = getindex.(matrix_trained, 3) .|> only
    linspace_trained_log_p_xy = getindex.(matrix_trained, 4) .|> only

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
    hm = heatmap!(ax, linspace_x, linspace_y, linspace_trained_log_p_xy; colormap=:magma)
    ax.xlabel = "x"
    ax.ylabel = "y"
    Colorbar(fig[2, 4], hm)
    supertitle = Label(fig[2, 3:4, Top()], "log p(x|y)"; fontsize=16)
    supertitle = Label(fig[0, :], "trained network"; fontsize=20)
    resize_to_layout!(fig)
    display_interactive(fig)


    fig = Figure();

    ax = Axis(fig[1,1])
    y = exp.(linspace_trained_log_p_xy)
    y ./= maximum(y; dims=1)
    hm = heatmap!(ax, linspace_x, linspace_y, y; colormap=:magma)
    ax.xlabel = "x"
    ax.ylabel = "y"
    Colorbar(fig[1, 2], hm)

    ax = Axis(fig[1, 3])
    ax.xlabel = "x"
    ax.ylabel = "p(x|y)"
    linspace_trained_p_xy = exp.(linspace_trained_log_p_xy)
    n = min(10, length(linspace_y))
    colors = Makie.cgrad(:viridis, n)
    for i in 1:n
        j = Int64(round(i * length(linspace_y) / n))
        label = @sprintf("y=%.2f", linspace_y[j])
        color = colors[i/n]
        lines!(ax, linspace_x, linspace_trained_p_xy[:, j]; label, color)
    end
    fig[1, 4] = Legend(fig, ax; labelsize=14, unique=true)
    supertitle = Label(fig[0, :], "trained network p(x|y)"; fontsize=20)
    resize_to_layout!(fig)
    display_interactive(fig)
end

@test true;
