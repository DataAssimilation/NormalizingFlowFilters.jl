using InvertibleNetworks: InvertibleNetworks, NetworkConditionalGlow
using Flux: Flux, ClipNorm, cpu, gpu

export NormalizingFlowFilter,
    NetworkConditionalGlow, create_optimizer, reset_optimizer, cpu, gpu, get_data, set_data!

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
    return  NetworkConditionalGlow(
        config.chan_x,
        config.chan_y,
        config.n_hidden,
        config.L,
        config.K;
        split_scales=config.split_scales,
        ndims,
    )
end

function reset_network(network::NetworkConditionalGlow)
    n_hidden = size(network.CL[1,1].RB.W1.data)[end]
    n_in = network.CL[1,1].C.k
    in_split_plus_n_cond = size(network.CL[1,1].RB.W1.data)[end-1]
    out_chan = size(network.CL[1,1].RB.W3.data)[end-1]
    split_num = out_chan ÷ 2
    in_split = n_in - split_num
    n_cond = in_split_plus_n_cond - in_split
    return NetworkConditionalGlow(
        n_in,
        n_cond,
        n_hidden,
        network.L,
        network.K;
        split_scales=network.split_scales,
        ndims= ndims(network.CL[1,1].RB.W1.data) - 2,
    )
end

function create_optimizer(config)
    adam = Flux.Optimise.Adam(config.lr, config.momentum, config.epsilon)
    return Flux.Optimiser(ClipNorm(config.clipnorm_val), adam)
end

function reset_optimizer(opt)
    c, a = opt.os
    @assert c isa ClipNorm
    @assert a isa Flux.Optimise.Adam
    return Flux.Optimiser(ClipNorm(c.thresh), Flux.Optimise.Adam(a.eta, a.beta, a.epsilon))
end

function get_data(filter::NormalizingFlowFilter)
    return InvertibleNetworks.get_params(filter.network_device)
end

function set_data!(filter::NormalizingFlowFilter, params)
    InvertibleNetworks.set_params!(filter.network, params)
    InvertibleNetworks.set_params!(filter.network_device, params)
end
