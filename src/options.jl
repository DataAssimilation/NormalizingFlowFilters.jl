using Configurations: @option

export ConditionalGlowOptions, TrainingOptions, OptimizerOptions

@option struct ConditionalGlowOptions
    chan_x = 3
    chan_y = 3

    "Number of multiscale levels"
    L = 3

    "Number of Real-NVP layers per multiscale level"
    K = 9

    "Number of hidden channels in convolutional residual blocks"
    n_hidden = 8

    split_scales = false
end

@option struct TrainingOptions
    n_epochs = 32
    batch_size = 2
    noise_lev_x = 0.005f0
    noise_lev_y = 0.0f0
    num_post_samples = 10
    validation_perc = 0.8
    n_condmean = 2
    reset_optimizer = false
    reset_weights = false
end

@option struct OptimizerOptions
    lr = 1.0f-3
    clipnorm_val = 3.0f0
end
