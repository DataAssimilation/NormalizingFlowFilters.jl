# # Description
# This example shows a simple use case for NormalizingFlowFilters.
#
# First, we import the necessary packages.

using Pkg: Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

using TerminalLoggers: TerminalLogger
using Logging: global_logger
global_logger(TerminalLogger())

using CairoMakie
using NormalizingFlowFilters
using Random: randn, seed!
using Statistics: mean, std, cov
using Test
using Pkg: Pkg

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
N = smalltest ? 2^4 : 2^11
Nx = 2
seed!(0x84fb4b2c)
glow_config = ConditionalGlowOptions(; chan_x=Nx, chan_y=Nx)
network = NetworkConditionalGlow(2, glow_config)

optimizer_config = OptimizerOptions(; lr=1e-3)
optimizer = create_optimizer(optimizer_config)

device = cpu
training_config = TrainingOptions(;
    n_epochs=smalltest ? 10 : 45,
    num_post_samples=1,
    noise_lev_y=1e-3,
    noise_lev_x=1e-3,
    batch_size=smalltest ? 2^2 : 2^9,
    validation_perc=2^(-1),
)

filter = NormalizingFlowFilter(network, optimizer; device, training_config)

# We generate an ensemble.

## N ensemble members from a unit normal.
prior_state = randn(Float64, Nx, N)
prior_state .-= mean(prior_state; dims=2)
prior_state ./= std(prior_state; dims=2)

function to_table(a; prefix=:x)
    return (; (Symbol(prefix, i) => row for (i, row) in enumerate(eachrow(a)))...)
end
combine_tables(a, b) = (; a..., b...)
table_prior_state = to_table(prior_state)

@static if VERSION >= v"1.10"
    kde_bandwidth =
        training_config.noise_lev_x /
        PairPlots.KernelDensity.default_bandwidth(prior_state[1, :])
    table_prior_state_mean = to_table(mean(prior_state; dims=2)[:, 1])
    fig = pairplot(
        table_prior_state => (
            PairPlots.Hist(; colormap=:Blues),
            PairPlots.MarginDensity(;
                bandwidth=kde_bandwidth, color=RGBf((49, 130, 189) ./ 255...)
            ),
            PairPlots.TrendLine(; color=:red),
            PairPlots.Correlation(),
            PairPlots.Scatter(),
        ),
        PairPlots.Truth(
            table_prior_state_mean; label="Mean Values", color=(:black, 0.5), linewidth=4
        ),
    )
    supertitle = Label(fig[0, :], "prior state"; fontsize=30)
    resize_to_layout!(fig)
    display_interactive(fig)
end

# Apply observation operator.

## Identity observation operator with some noise.
prior_obs = 0.5 .* deepcopy(prior_state) .+ 0.5 .* randn(Float64, size(prior_state))

table_prior_obs = to_table(prior_obs; prefix=:y)

@static if VERSION >= v"1.10"
    table_prior_obs_mean = to_table(mean(prior_obs; dims=2)[:, 1]; prefix=:y)
    fig = pairplot(
        table_prior_obs => (
            PairPlots.Hist(; colormap=:Blues),
            PairPlots.MarginDensity(;
                bandwidth=kde_bandwidth, color=RGBf((49, 130, 189) ./ 255...)
            ),
            PairPlots.TrendLine(; color=:red),
            PairPlots.Correlation(),
            PairPlots.Scatter(),
        ),
        PairPlots.Truth(
            table_prior_obs_mean; label="Mean Values", color=(:black, 0.5), linewidth=4
        ),
    )
    supertitle = Label(fig[0, :], "prior observation"; fontsize=30)
    resize_to_layout!(fig)
    display_interactive(fig)
end

# Look at how observations correlate to data.

@static if VERSION >= v"1.10"
    combo_table = combine_tables(table_prior_state, table_prior_obs)
    combo_table_mean = combine_tables(table_prior_state_mean, table_prior_obs_mean)
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
    supertitle = Label(fig[0, :], "prior state-observation"; fontsize=30)
    resize_to_layout!(fig)
    display_interactive(fig)
end

# Then we assimilate an observation. Here, we just pick an arbitrary one.
y_obs = zeros(Nx)
log_data = Dict{Symbol,Any}()
posterior = assimilate_data(filter, prior_state, prior_obs, y_obs, log_data)

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
    batch_size=filter.training_config.batch_size,
)
Z = Z[1, 1, :, :]

table_Z = to_table(Z; prefix=:z)

@static if VERSION >= v"1.10"
    table_Z_mean = to_table(mean(Z; dims=2)[:, 1]; prefix=:z)
    fig = pairplot(
        table_Z => (
            PairPlots.Hist(; colormap=:Blues),
            PairPlots.MarginDensity(;
                bandwidth=kde_bandwidth, color=RGBf((49, 130, 189) ./ 255...)
            ),
            PairPlots.TrendLine(; color=:red),
            PairPlots.Correlation(),
            PairPlots.Scatter(),
        ),
        PairPlots.Truth(
            table_Z_mean; label="Mean Values", color=(:black, 0.5), linewidth=4
        ),
    )
    supertitle = Label(fig[0, :], "latent state"; fontsize=30)
    resize_to_layout!(fig)
    display_interactive(fig)
end

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

# Visualize posterior.
table_posterior = to_table(posterior)

@static if VERSION >= v"1.10"
    table_posterior_mean = to_table(mean(posterior; dims=2)[:, 1])
    fig = pairplot(
        table_posterior => (
            PairPlots.Hist(; colormap=:Blues),
            PairPlots.MarginDensity(;
                bandwidth=kde_bandwidth, color=RGBf((49, 130, 189) ./ 255...)
            ),
            PairPlots.TrendLine(; color=:red),
            PairPlots.Correlation(),
            PairPlots.Scatter(),
        ),
        PairPlots.Truth(
            table_posterior_mean; label="Mean Values", color=(:black, 0.5), linewidth=4
        ),
    )
    supertitle = Label(fig[0, :], "posterior state"; fontsize=30)
    resize_to_layout!(fig)
    display_interactive(fig)
end

# The posterior should have mean 0 and some TBD variance.

X = reshape(prior_state, (1, 1, size(prior_state, 1), size(prior_state, 2)))
Y = reshape(prior_obs, (1, 1, size(prior_obs, 1), size(prior_obs, 2)))
y_obs_r = reshape(y_obs, (1, 1, size(y_obs, 1), size(y_obs, 2)))

fresh_samples = draw_posterior_samples(
    filter.network_device,
    y_obs_r,
    size(X);
    device=filter.device,
    num_samples=32,
    batch_size=filter.training_config.batch_size,
    log_data=nothing,
)[
    1, 1, :, :,
]

@test true;

# Plot some training metrics.

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
ax.title = "Training: $N_train, Validation: $N_valid, Batch size: $(training_config.batch_size)"

ax = Axis(fig[2, 1])

x = log_data[:network_training][:training][:logdet]
lines!(ax, train_epochs, x; color=lines_train.color, common_kwargs...)

x = log_data[:network_training][:testing][:logdet]
lines!(ax, test_epochs, x; color=lines_test.color, common_kwargs...)

ax.xlabel = "epoch number"
ax.ylabel = "loss: log determinant"

supertitle = Label(fig[0, :], "Training log"; fontsize=30)
resize_to_layout!(fig)
display_interactive(fig)
