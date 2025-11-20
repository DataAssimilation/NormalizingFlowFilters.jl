using Configurations: @option

export ConditionalGlowOptions, ConditionalSVDOptions, ConditionalLinearOptions, TrainingOptions,
        OptimizerOptions, ActivationOptions, ResidualBlockOptions, ConditionalLinearGlowOptions,
        EarlyStoppingOptions, ConditionalCouplingStackOptions, CouplingLayerOptions, Conv1x1Options,
        ActNormOptions, LayerConstantOptions, FixedNumBatchesOptions, FixedBatchSizeOptions,
        RQSpline1OperatorOptions, AffineCouplingOperatorOptions, ConditionalDecorrelationOperatorOptions,
        RQSpline1ActivationOptions, LayerStackOptions, ResidualBlockSkipOptions,
        LearningRateDecayOptions, WeightDecayOptions, TargetUnitNormalOptions, TargetUnitNormalPlusUniformOptions,
        HyperComplexitySearcherOptions, UnitGaussianNoiseOptions, DataCorrelatedGaussianNoiseOptions,
        CovarianceInflationGaussianNoiseOptions, remake_options

function remake_options(opts::T; kwargs...) where {T}
    opts_kwargs = (f => getfield(opts, f) for f in fieldnames(T))
    return T(; opts_kwargs..., kwargs...)
end

@option struct ConditionalGlowOptions
    chan_x = 3
    chan_y = 3

    "Number of multiscale levels"
    L = 3

    "Number of Real-NVP layers per multiscale level"
    K = 9

    split_scales = false

    residual = ResidualBlockOptions()

    positive_activation = ActivationOptions(type="sigmoid")
end

@option struct CouplingLayerOptions
    subnetwork = ResidualBlockOptions(
        final_activation = ActivationOptions("identity")
    )
    invertible_network = AffineCouplingOperatorOptions()
    split = true
end

@option struct LayerStackOptions
    subnetworks = (
        (
            network = LayerConstantOptions(),
        ),
        (
            network = ResidualBlockOptions(
                final_activation = ActivationOptions("identity")
            ),
        ),
    )
end

@option struct AffineCouplingOperatorOptions
    scale_activation = ActivationOptions("damped_cosh")
    shift_activation = ActivationOptions("damped_sinh")
    shift_cond_scalar = false
    joint_correlation = true
    just_shift = false
end

@option struct RQSpline1ActivationOptions
end

@option struct ConditionalDecorrelationOperatorOptions
end

@option struct RQSpline1OperatorOptions
    constrained_params = true
    affine = AffineCouplingOperatorOptions()
end

@option struct ConditionalCouplingStackOptions
    chan_x = 3
    chan_y = 3

    "Number of multiscale levels"
    L = 3

    "Number of Real-NVP layers per multiscale level"
    K = 9

    split_scales = false

    coupling_network::CouplingLayerOptions = CouplingLayerOptions()
    cond_network = ActNormOptions()
    state_initial_network = ActNormOptions()
    state_middle_network = ActNormOptions()
    state_final_network = ActNormOptions()
    prenetwork = Conv1x1Options()
end

@option struct Conv1x1Options
end

@option struct ActNormOptions
end

@option struct LayerConstantOptions
end

@option struct ActivationOptions
    type = "relu"
end

@option struct ResidualBlockOptions
    activation = ActivationOptions(type="softplus")
    final_activation = ActivationOptions(type="softplus")

    "Number of hidden channels in convolutional residual blocks"
    n_hidden = 8

    k1 = 3 # kernel size along each dimension for first and third convolutions.
    p1 = 1 # padding for first and third convolutions.
    s1 = 1 # strides for the first and third convolutions.

    k2 = 1 # kernel size along each dimension for second convolution.
    p2 = 0 # padding for second convolution.
    s2 = 1 # strides for the second convolution.
end

@option struct ResidualBlockSkipOptions
    activation = ActivationOptions(type="sigmoid")
    final_activation = ActivationOptions(type="identity")

    "Number of hidden channels in convolutional residual blocks"
    n_hidden = 8

    k1 = 3 # kernel size along each dimension for first and third convolutions.
    p1 = 1 # padding for first and third convolutions.
    s1 = 1 # strides for the first and third convolutions.

    k2 = 1 # kernel size along each dimension for second convolution.
    p2 = 0 # padding for second convolution.
    s2 = 1 # strides for the second convolution.

    k13 = 1 # kernel size along each dimension for skip convolution (from input to output).
    p13 = 0 # padding for skip convolution.
    s13 = 1 # strides for the skip convolution.
end

@option struct ConditionalSVDOptions
end

@option struct ConditionalLinearOptions
    random_init=false
end

@option struct ConditionalLinearGlowOptions
    ln_config = ConditionalLinearOptions()
    gn_config = ConditionalGlowOptions()
    post_actnorm = false
end

@option struct TrainingOptions
    n_epochs = 32
    normalize_initial = true
    batch = FixedNumBatchesOptions()
    noise = UnitGaussianNoiseOptions()
    num_post_samples = 10
    validation_perc = 0.8
    n_condmean = 2
    reset_optimizer = false
    reset_weights = false
    print_every = 1
    save_best = true
    early_stopping_training_loss = EarlyStoppingOptions()
    early_stopping_validation_loss = EarlyStoppingOptions()
    hypersearcher = nothing
    cm_metrics = false
end

@option struct UnitGaussianNoiseOptions
    x = 0.005f0
    y = 0.005f0
end

@option struct DataCorrelatedGaussianNoiseOptions
    x = 0.001f0
    y = 0.001f0
    x_correlated = 0.005f0
    y_correlated = 0.005f0
end

@option struct CovarianceInflationGaussianNoiseOptions
    x = 0.001f0
    y = 0.001f0
    inflation = 0.01f0
end

@option struct WeightDecayOptions
    active = false
    factor = 1e-5
end

@option struct LearningRateDecayOptions
    active = false
    factor = 0.9
    step = 500
    minimum = 1e-1
end

@option struct FixedBatchSizeOptions
    batch_size = 2
end

@option struct FixedNumBatchesOptions
    min_batch_size = 2
    num_batches = 1
end

@option struct EarlyStoppingOptions
    active = false

    # Lookback form:
    #   - (epochs to look back on, proportion to check of last epochs, max allowed increase in loss)
    #       I.e., after the first 100 epochs have passed, stop if the loss has increased by
    #       more than the max allowed amount for more than 60% of the past 100 epochs.
    #   - (epochs to look back on, -1, max allowed increase in loss compared to best loss)
    #       I.e., when a best loss is achieved, wait 100 epochs, and stop if the loss has increased by more than the max allowed amount.
    look_backs = ((20, 0.5, 0.1),(100, -1, -1f-6))
end

@option struct OptimizerOptions
    lr = 1.0f-3
    momentum = (0.9, 0.999)
    epsilon = 1.0f-8
    method = "adam"
    clipnorm_val = 3.0f0
    weight_decay = WeightDecayOptions()
    learning_rate_decay = LearningRateDecayOptions()
end

@option struct TargetUnitNormalOptions
end

@option struct TargetUnitNormalPlusUniformOptions
    uniform_weight = 0.001
end

@option struct HyperComplexitySearcherOptions
    min_complexity = 1
    max_complexity = 32
    keep_going = 4
end