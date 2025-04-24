using InvertibleNetworks: InvertibleNetworks, NetworkConditionalGlow, ReLUlayer, SigmoidLayer, LeakyReLUlayer, GaLUlayer
using InvertibleNetworks: ActivationFunction, ExpClamp, ExpClampInv, ExpClampGrad
using Flux: Flux, ClipNorm, cpu, gpu

export NormalizingFlowFilter,
    NetworkConditionalGlow, create_optimizer, reset_optimizer, cpu, gpu, get_data, set_data!, get_activation

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


function get_activation(config::ActivationOptions)
    if config.type == "relu"
        return ReLUlayer()
    elseif config.type == "sigmoid"
        return SigmoidLayer()
    elseif config.type == "leaky_relu"
        return LeakyReLUlayer()
    elseif config.type == "galu"
        return GaLUlayer()
    elseif config.type == "exp_clamp"
        return ActivationFunction(x -> ExpClamp(x), y -> ExpClampInv(y), (Δy, y) -> ExpClampGrad(Δy, y))
    elseif config.type == "softplus"
        return SoftplusLayer()
    elseif config.type == "identity"
        return IdentityActivation()
    else
        error("I don't know what this activation is: $(config.type)")
    end
end


function InvertibleNetworks.NetworkConditionalGlow(ndims, config::ConditionalGlowOptions)
    r = config.residual
    activation = get_activation(config.positive_activation)
    rb_activation = get_activation(r.activation)
    return NetworkConditionalGlow(
        config.chan_x,
        config.chan_y,
        config.n_hidden,
        config.L,
        config.K;
        split_scales=config.split_scales,
        ndims,
        r.k1,
        r.k2,
        r.p1,
        r.p2,
        r.s1,
        r.s2,
        activation,
        rb_activation
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
    cl = network.CL[1,1]
    return NetworkConditionalGlow(
        n_in,
        n_cond,
        n_hidden,
        network.L,
        network.K;
        split_scales=network.split_scales,
        ndims= ndims(cl.RB.W1.data) - 2,
        k1=size(cl.RB.W1.data, 1),
        k2=size(cl.RB.W2.data, 1),
        p1=cl.RB.pad[1],
        p2=cl.RB.pad[2],
        s1=cl.RB.strides[1],
        s2=cl.RB.strides[2],
        activation=cl.activation,
        rb_activation=cl.RB.activation
    )
end

function create_optimizer(config)
    if config.method == "adam"
        a = Flux.Optimise.Adam(config.lr, config.momentum, config.epsilon)
    elseif config.method == "descent"
        a = Flux.Optimise.Descent(config.lr)
    else
        error("Unknown optimizer method: $(config.method)")
    end
    return Flux.Optimiser(ClipNorm(config.clipnorm_val), a)
end

function reset_optimizer(opt)
    c, a = opt.os
    @assert c isa ClipNorm
    c = ClipNorm(c.thresh)
    if a isa Flux.Optimise.Adam
        a = Flux.Optimise.Adam(a.eta, a.beta, a.epsilon)
    elseif a isa Flux.Optimise.Descent
        a = Flux.Optimise.Descent(a.eta)
    else
        error("Unknown optimizer type: $(typeof(opt))")
    end
    return Flux.Optimiser(c, a)
end

function get_data(filter::NormalizingFlowFilter)
    return InvertibleNetworks.get_params(filter.network_device)
end

function set_data!(filter::NormalizingFlowFilter, params)
    InvertibleNetworks.set_params!(filter.network, params)
    InvertibleNetworks.set_params!(filter.network_device, params)
end

function get_network_gradients(filter::NormalizingFlowFilter)
    return InvertibleNetworks.get_grads(filter.network_device)
end