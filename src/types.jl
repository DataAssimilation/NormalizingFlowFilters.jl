using InvertibleNetworks: InvertibleNetworks,
    ActivationFunction, ExpClamp, ExpClampInv, ExpClampGrad,
    NetworkConditionalGlow, NetworkConditionalCouplingStack,
    ResidualBlock, ResidualBlockSkip, Conv1x1, ActNorm, LayerConstant,
    RQSpline1Operator, AffineCouplingOperator,
    IdentityActivation, ConditionalDecorrelationOperator,
    ReLUlayer, SigmoidLayer, LeakyReLUlayer, GaLUlayer,
    SoftplusLayer, TanhLayer, SinhLayer, CoshLayer, LayerStack,
    DampedSinhLayer, DampedCoshLayer, ScaledTanhLayer, RQSpline1
using Flux: Flux, ClipNorm, cpu, gpu

export NormalizingFlowFilter, NetworkConditionalCouplingStack,
    NetworkConditionalGlow, NormalizingFlowOptimizer,
    create_optimizer, cpu, gpu, get_data, set_data!, get_activation

mutable struct NormalizingFlowFilter
    coupling_network
    coupling_network_device
    coupling_network_generator
    target_distribution
    state_shape
    obs_shape
    opt
    device
    training_config
end

function NormalizingFlowFilter(
    coupling_network_generator, target_distribution, state_shape, obs_shape, optimizer; device=cpu, training_config=TrainingOptions()
)
    coupling_network = coupling_network_generator()
    return NormalizingFlowFilter(
        coupling_network, device(coupling_network), coupling_network_generator, target_distribution, state_shape, obs_shape, optimizer, device, training_config
    )
end

function get_activation(config::RQSpline1ActivationOptions; kwargs...)
    return RQSpline1(; kwargs...)
end

function get_activation(config::ActivationOptions; kwargs...)
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
        return LayerConstant(p; kwargs...)
    end
end


function get_network_generator(opt::LayerStackOptions; kwargs...)
    gens = [get_network_generator(sub.network; kwargs...) for sub in opt.subnetworks]
    return function (in_shape, out_shape=in_shape)
        nchan = out_shape[end]
        share, rem = divrem(nchan, length(gens))
        if rem != 0
            error("Tried to split channels $nchan into $(length(gens)) inputs. I can't do it.")
        end
        out_shapes = [(out_shape[1:end-1]..., share) for i in 1:length(gens)]
        subnetworks = [gen(in_shape, out_shape_i) for (gen, out_shape_i) in zip(gens, out_shapes)]
        return LayerStack(subnetworks)
    end
end

function get_network_generator(opt::ResidualBlockOptions; kwargs...)
    activation = get_activation(opt.activation; logdet=false)
    final_activation = get_activation(opt.final_activation; logdet=false)
    kwargs = (; k1=opt.k1, k2=opt.k2, p1=opt.p1, p2=opt.p2, s1=opt.s1, s2=opt.s2, activation, final_activation, fan=true)
    n_hidden = opt.n_hidden
    return function (in_shape, out_shape)
        ndims = length(in_shape) - 1
        n_out = out_shape[end]
        ResidualBlock(in_shape[end], n_hidden; n_out, ndims, kwargs...)
    end
end

function get_network_generator(opt::ResidualBlockSkipOptions; kwargs...)
    activation = get_activation(opt.activation; logdet=false)
    final_activation = get_activation(opt.final_activation; logdet=false)
    kwargs = (;
        k1=opt.k1, k2=opt.k2,p1=opt.p1, p2=opt.p2, s1=opt.s1, s2=opt.s2,
        k13=opt.k13, p13=opt.p13, s13=opt.s13, activation, final_activation
    )
    n_hidden = opt.n_hidden
    return function (in_shape, out_shape)
        ndims = length(in_shape) - 1
        n_out = out_shape[end]
        ResidualBlockSkip(in_shape[end], n_hidden; n_out, ndims, kwargs...)
    end
end

function get_coupling_operator_generator(opt::AffineCouplingOperatorOptions; kwargs...)
    scale_activation = get_activation(opt.scale_activation)
    shift_activation = get_activation(opt.shift_activation)
    shift_cond_scalar = opt.shift_cond_scalar
    joint_correlation = opt.joint_correlation
    just_shift = opt.just_shift
    return function (inv_shape, sub_shape)
        return AffineCouplingOperator(;
            scale_activation,
            shift_activation,
            shift_cond_scalar,
            joint_correlation,
            just_shift,
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
        coupling_layer_params = (; split = config.coupling_network.split),
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

struct NormalizingFlowOptimizer
    flux
    config
end

function create_optimizer(config)
    opts = []
    push!(opts, ClipNorm(config.clipnorm_val))
    if config.method == "adam"
        a = Flux.Optimise.Adam(config.lr, config.momentum, config.epsilon)
    elseif config.method == "descent"
        a = Flux.Optimise.Descent(config.lr)
    else
        error("Unknown optimizer method: $(config.method)")
    end
    push!(opts, a)
    push!(opts, get_weight_decay(config.weight_decay))
    push!(opts, get_learning_rate_decay(config.learning_rate_decay))
    flux = Flux.Optimiser([o for o in opts if !isnothing(o)]...)
    return NormalizingFlowOptimizer(flux, config)
end

function get_weight_decay(opt::WeightDecayOptions)
    if opt.active
        return Flux.WeightDecay(opt.factor)
    end
    return nothing
end

function get_learning_rate_decay(opt::LearningRateDecayOptions)
    if opt.active
        return Flux.ExpDecay(1.0, opt.factor, opt.step, opt.minimum)
    end
    return nothing
end

function get_data(filter::NormalizingFlowFilter)
    return InvertibleNetworks.get_params(filter.coupling_network_device) |> cpu
end

function set_data!(filter::NormalizingFlowFilter, params)
    InvertibleNetworks.set_params!(filter.coupling_network, params)
    InvertibleNetworks.set_params!(filter.coupling_network_device, params)
end

function get_network_gradients(filter::NormalizingFlowFilter)
    return InvertibleNetworks.get_grads(filter.coupling_network_device)
end
