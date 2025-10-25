using InvertibleNetworks: InvertibleNetworks,
    ActivationFunction, ExpClamp, ExpClampInv, ExpClampGrad,
    NetworkConditionalGlow, NetworkConditionalCouplingStack,
    ResidualBlock, Conv1x1, ActNorm, LayerConstant,
    RQSpline1Operator, AffineCouplingOperator,
    IdentityActivation,
    ReLUlayer, SigmoidLayer, LeakyReLUlayer, GaLUlayer,
    SoftplusLayer, TanhLayer, SinhLayer, CoshLayer,
    DampedSinhLayer, DampedCoshLayer, ScaledTanhLayer
using Flux: Flux, ClipNorm, cpu, gpu

export NormalizingFlowFilter, NetworkConditionalCouplingStack,
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
    elseif config.type == "tanh"
        return TanhLayer()
    elseif config.type == "cosh"
        return CoshLayer()
    elseif config.type == "sinh"
        return SinhLayer()
    elseif config.type == "damped_sinh"
        return DampedSinhLayer()
    elseif config.type == "damped_cosh"
        return DampedCoshLayer()
    elseif config.type == "scaled_tanh"
        return ScaledTanhLayer()
    else
        error("I don't know what this activation is: $(config.type)")
    end
end

function get_network_generator(::Conv1x1Options; kwargs...)
    return in_shape -> Conv1x1(in_shape[end]; kwargs...)
end

function get_network_generator(::ActNormOptions; kwargs...)
    return in_shape -> ActNorm(in_shape[end]; kwargs...)
end

function get_network_generator(::Nothing; kwargs...)
    return nothing
end

function get_network_generator(opt::LayerConstantOptions; kwargs...)
    return function (in_shape, out_shape=in_shape)
        p = Parameter(zeros(out_shape))
        return LayerConstant(p)
    end
end

function get_network_generator(opt::ResidualBlockOptions; kwargs...)
    activation = get_activation(opt.activation)
    final_activation = get_activation(opt.final_activation)
    kwargs = (; k1=opt.k1, k2=opt.k2, p1=opt.p1, p2=opt.p2, s1=opt.s1, s2=opt.s2, activation, final_activation, fan=true)
    n_hidden = opt.n_hidden
    return function (in_shape, out_shape)
        ndims = length(in_shape) - 1
        n_out = out_shape[end]
        ResidualBlock(in_shape[end], n_hidden; n_out, ndims, kwargs...)
    end
end

function get_coupling_operator_generator(opt::AffineCouplingOperatorOptions; kwargs...)
    scale_activation = get_activation(opt.scale_activation)
    shift_activation = get_activation(opt.shift_activation)
    shift_cond_scalar = opt.shift_cond_scalar
    joint_correlation = opt.joint_correlation
    return function (inv_shape, sub_shape)
        return AffineCouplingOperator(;
            scale_activation,
            shift_activation,
            shift_cond_scalar,
            joint_correlation,
        )
    end
end

function get_coupling_operator_generator(opt::RQSpline1OperatorOptions; kwargs...)
    constrained_params = opt.constrained_params
    # @show opt
    affine_gen = get_coupling_operator_generator(opt.affine)
    return function (inv_shape, sub_shape)
        affine = affine_gen(inv_shape, sub_shape)
        return RQSpline1Operator(; constrained_params, affine)
    end
end

function get_coupling_operator_generator(opt::ConditionalDecorrelationOperatorOptions; kwargs...)
    return function (inv_shape, sub_shape)
        return ConditionalDecorrelationOperator()
    end
end

function InvertibleNetworks.NetworkConditionalCouplingStack(in_shape, cond_shape, config::ConditionalCouplingStackOptions)
    invertible_coupling_operator_generator = get_coupling_operator_generator(config.coupling_network.invertible_network; logdet=true)
    subnetwork_generator = get_network_generator(config.coupling_network.subnetwork; logdet=false)

    cond_network_generator = get_network_generator(config.cond_network; logdet=false)
    state_initial_network_generator = get_network_generator(config.state_initial_network; logdet=true)
    state_middle_network_generator = get_network_generator(config.state_middle_network; logdet=true)
    state_final_network_generator = get_network_generator(config.state_final_network; logdet=true)
    prenetwork_generator = get_network_generator(config.prenetwork; logdet=true)

    return NetworkConditionalCouplingStack(in_shape, cond_shape, config.L, config.K;
        cond_network_generator,
        state_initial_network_generator,
        state_middle_network_generator,
        state_final_network_generator,
        subnetwork_generator,
        prenetwork_generator,
        invertible_coupling_operator_generator,
    )
end

function InvertibleNetworks.NetworkConditionalGlow(ndims, config::ConditionalGlowOptions)
    r = config.residual
    activation = get_activation(config.positive_activation)
    rb_activation = get_activation(r.activation)
    if r.final_activation != r.activation
        error("NetworkConditionalGlow does not support `final_activation` parameter yet.")
    end
    return NetworkConditionalGlow(
        config.chan_x,
        config.chan_y,
        r.n_hidden,
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

function reset_network(network::NetworkConditionalCouplingStack)
    error("not implemented")
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
