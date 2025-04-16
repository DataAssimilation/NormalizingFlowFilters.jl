using InvertibleNetworks: InvertibleNetworks, NetworkConditionalGlow
using Flux: Flux, ClipNorm, cpu, gpu

export NormalizingFlowFilter,
    NetworkConditionalGlow, create_optimizer, cpu, gpu, get_data, set_data!

struct NormalizingFlowFilter
    network
    network_device
    opt
    device
    training_config
end

function NormalizingFlowFilter(
    network, optimizer; device=cpu, training_config=TrainingOptions()
)
    return NormalizingFlowFilter(
        network, device(network), optimizer, device, training_config
    )
end

function InvertibleNetworks.NetworkConditionalGlow(ndims, config::ConditionalGlowOptions)
    return NetworkConditionalGlow(
        config.chan_x,
        config.chan_y,
        config.n_hidden,
        config.L,
        config.K;
        split_scales=config.split_scales,
        ndims,
    )
end

function create_optimizer(config)
    return Flux.Optimiser(ClipNorm(config.clipnorm_val), Flux.Optimise.Adam(config.lr))
end

function get_data(filter::NormalizingFlowFilter)
    return InvertibleNetworks.get_params(filter.network_device)
end

function set_data!(filter::NormalizingFlowFilter, params)
    InvertibleNetworks.set_params!(filter.network, params)
    InvertibleNetworks.set_params!(filter.network_device, params)
end
