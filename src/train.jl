using Flux: Flux
using LinearAlgebra: norm
using MLUtils: splitobs, obsview
using ImageQualityIndexes: assess_ssim
using Random: randn, randperm
using InvertibleNetworks: InvertibleNetworks, reset!, clear_grad!, get_params
using Statistics: mean, var
using ProgressLogging: @withprogress, @logprogress, @progressid
using SpecialFunctions: logfactorial

export train_network!, get_cm_l2_ssim, get_loss

function get_batch_size(batch_options::FixedBatchSizeOptions, N)
    return batch_options.batch_size
end

function get_batch_size(batch_options::FixedNumBatchesOptions, N)
    batch_size = Int64(cld(N, batch_options.num_batches))
    batch_size = max(batch_size, batch_options.min_batch_size)
    return batch_size
end

function get_cm_l2_ssim(G, X, Y, X_batch, Y_batch; device=gpu, num_samples, batch_size)
    num_test = size(Y_batch)[end]
    l2_total = 0
    ssim_total = 0
    #get cm for each element in batch
    for i in 1:num_test
        y_i = Y_batch[:, :, :, i:i]
        x_i = X_batch[:, :, :, i:i]
        X_post_test = draw_posterior_samples(
            G, y_i, X, Y, size(x_i); device, num_samples, batch_size
        )
        X_post_mean_test = mean(X_post_test; dims=4)
        ssim_total += assess_ssim(X_post_mean_test[:, :, 1, 1], cpu(x_i[:, :, 1, 1]))
        l2_total += norm(X_post_mean_test[:, :, 1, 1] - (cpu(x_i[:, :, 1, 1])))^2
    end
    return l2_total / num_test, ssim_total / num_test
end

function get_loss(G, X_batch, Y_batch; device=gpu, batch_size, N, target_distribution)
    num_test = size(Y_batch)[end]
    if num_test == 0
        return NaN, NaN, NaN
    end
    weighted_misfit_total = 0
    loss_nlogpz_total = 0
    logdet_total = 0
    num_batches = cld(num_test, batch_size)
    batch_idxs = collect(1:batch_size:(num_test + 1))
    if batch_idxs[end] != num_test+1
        push!(batch_idxs, num_test+1)
    end
    for i in 1:num_batches
        idx = batch_idxs[i]:(batch_idxs[i + 1] - 1)
        n_batch = length(idx)
        x_i = X_batch[:, :, :, idx]
        y_i = Y_batch[:, :, :, idx]

        Zx, Zy, lgdet = cpu(G.forward(device(x_i), device(y_i)))
        nlogpz, dnlogpz_dz = compute_negative_log_density_with_gradient(target_distribution, Zx, Zy)
        loss_nlogpz_total += nlogpz
        weighted_misfit_total += sum(dnlogpz_dz .* Zx) / prod(N)
        logdet_total += n_batch * lgdet / prod(N)
    end

    return loss_nlogpz_total / num_test, logdet_total / num_test, weighted_misfit_total / num_test
end

function add_training_noise(cfg::UnitGaussianNoiseOptions, X::AbstractArray{T, Nx}, Y::AbstractArray{T, Ny}) where {T, Nx, Ny}
    X = X .+ T.(cfg.x) * randn(T, size(X))
    Y = Y .+ T.(cfg.y) * randn(T, size(Y))
    return X, Y
end

function add_training_noise(cfg::DataCorrelatedGaussianNoiseOptions, X::AbstractArray{T, Nx}, Y::AbstractArray{T, Ny}) where {T, Nx, Ny}
    Nb = size(X)[end]
    batch_noise = randn(T, Nb, Nb)
    x_scale = cfg.x_correlated / sqrt(Nb - 1)
    y_scale = cfg.y_correlated / sqrt(Nb - 1)
    X = X .+ reshape(reshape(X .- mean(X; dims=Nx), :, Nb) * (batch_noise .* x_scale), size(X)) .+ cfg.x * randn(T, size(X))
    Y = Y .+ reshape(reshape(Y .- mean(Y; dims=Ny), :, Nb) * (batch_noise .* y_scale), size(Y)) .+ cfg.y * randn(T, size(Y))
    return X, Y
end

function add_training_noise(cfg::CovarianceInflationGaussianNoiseOptions, X::AbstractArray{T, Nx}, Y::AbstractArray{T, Ny}) where {T, Nx, Ny}
    Nb = size(X)[end]
    x_scale = cfg.inflation / sqrt(Nb - 1)
    y_scale = cfg.inflation / sqrt(Nb - 1)
    X = X .+ (X .- mean(X; dims=Nx)) .* T.(x_scale) .+ T.(cfg.x) * randn(T, size(X))
    Y = Y .+ (Y .- mean(Y; dims=Ny)) .* T.(y_scale) .+ T.(cfg.y) * randn(T, size(Y))
    return X, Y
end

function train_network!(filter::NormalizingFlowFilter, Xs, Ys; log_data=nothing)
    Xs = Float32.(Xs)
    Ys = Float32.(Ys)
    target_distribution = filter.target_distribution
    device = filter.device
    cfg = filter.training_config
    opt = filter.opt

    # Use MLutils to split into training and validation set
    num_samples = size(Xs)[end]
    shuffle_idxs = randperm(num_samples)

    train_split, test_split = splitobs(num_samples; at=cfg.validation_perc)

    train_split = shuffle_idxs[train_split]
    test_split = shuffle_idxs[test_split]

    X_train = obsview(Xs, train_split)
    Y_train = obsview(Ys, train_split)

    if cfg.reset_weights
        InvertibleNetworks.set_params!(filter.coupling_network_device, get_params(filter.coupling_network_generator(filter.coupling_network_device, X_train, Y_train)) |> device)
    end

    if filter.coupling_network isa NetworkConditionalLinear || filter.coupling_network isa NetworkConditionalSVD || filter.coupling_network isa NetworkConditionalLinearGlow
        Base.depwarn("This `initialize!` call will be taken out in future versions. Please initialize within the `coupling_network_generator`.", :train_network!)
        initialize!(filter.coupling_network.LN, X_train, Y_train)
        initialize!(filter.coupling_network_device.LN, device(X_train), device(Y_train))
    end

    X_test = obsview(Xs, test_split)
    Y_test = obsview(Ys, test_split)

    filter.coupling_network_device = filter.coupling_network_device |> device
    if isnothing(cfg.hypersearcher)
        train_network!(filter.coupling_network_device, target_distribution, cfg, opt, device, X_train, X_test, Y_train, Y_test, log_data)
        log_data[:coupling_network][:training][:split] = train_split
        log_data[:coupling_network][:testing][:split] = test_split
        return
    end
    network = train_network!(cfg.hypersearcher, filter.coupling_network_generator, filter.coupling_network_device, target_distribution, cfg, opt, device, X_train, X_test, Y_train, Y_test, log_data)
    log_data[:coupling_network][:training][:split] = train_split
    log_data[:coupling_network][:testing][:split] = test_split
    filter.coupling_network_device = network |> device
    filter.coupling_network = network |> cpu
end

function train_network!(searcher::HyperComplexitySearcherOptions, coupling_network_generator, coupling_network_device, target_distribution, cfg, opt, device, X_train, X_test, Y_train, Y_test, log_data)
    if isnothing(log_data)
        my_log_data = Dict{Symbol, Any}()
    else
        my_log_data = log_data
    end

    my_log_data[:complexity_search] = Dict{Symbol, Any}()

    f = function (l, smaller_network)
        k = Symbol(l)
        println("Testing complexity == $l")
        if haskey(my_log_data[:complexity_search], k)
            println("   Found in cache: $(my_log_data[:complexity_search][k][:best_loss]) for complexity = $l")
            return my_log_data[:complexity_search][k][:best_loss], my_log_data[:complexity_search][k][:network]
        end
        layer_log_data = Dict{Symbol, Any}()
        my_log_data[:complexity_search][Symbol(l)] = layer_log_data
        network = coupling_network_generator(smaller_network, X_train, Y_train; complexity=l) |> device
        train_network!(network, target_distribution, cfg, opt, device, X_train, X_test, Y_train, Y_test, layer_log_data)
        layer_log_data[:best_loss] = minimum(layer_log_data[:coupling_network][:testing][:loss_total])
        layer_log_data[:network] = get_params(network)
        println("  Got $(layer_log_data[:best_loss]) for complexity = $l")
        return layer_log_data[:best_loss], network
    end

    o_best, network_best = f(searcher.min_complexity, nothing)
    network_closest = network_best
    best_l = searcher.min_complexity
    for l in (searcher.min_complexity+1):searcher.max_complexity
        o, network = f(l, network_closest)
        if o < o_best
            println("New best is at complexity $l")
            o_best = o
            network_best = network
            best_l = l
        end
        if o > o_best && l - best_l >= searcher.keep_going
            println("quitting here because o > o_best && $l - $best_l > $(searcher.keep_going)")
            break
        end
        network_closest = network
    end
    my_log_data[:coupling_network] = my_log_data[:complexity_search][Symbol(best_l)][:coupling_network]
    return network_best
end

function train_network!(coupling_network_device, target_distribution, cfg, opt, device, X_train, X_test, Y_train, Y_test, log_data)
    N = size(X_train)[1:(end - 1)]

    # Training logs
    loss = Vector{Float64}()
    loss_weighted_misfit_train = Vector{Float64}()
    loss_weighted_misfit_valid = Vector{Float64}()
    logdet_train = Vector{Float64}()
    ssim = Vector{Float64}()
    l2_cm = Vector{Float64}()

    loss_test = Vector{Float64}()
    logdet_test = Vector{Float64}()
    ssim_test = Vector{Float64}()
    l2_cm_test = Vector{Float64}()

    loss_total_train_epochs = Vector{Float64}()
    loss_total_valid_epochs = Vector{Float64}()

    n_train = size(X_train)[end]
    n_test = size(X_test)[end]
    batch_size = get_batch_size(cfg.batch, n_train)
    n_batches = cld(n_train, batch_size)

    batch_idxs = collect(1:batch_size:(n_train + 1))
    if batch_idxs[end] != n_train+1
        push!(batch_idxs, n_train+1)
    end

    if cfg.reset_optimizer
        opt = create_optimizer(opt.config).flux
    else
        opt = opt.flux
    end

    best_params = deepcopy(get_params(coupling_network_device) |> cpu)
    best_train_loss = Inf
    best_valid_loss = Inf

    best_train_epoch = 0
    best_valid_epoch = 0

    @withprogress name="Epochs" for e in 1:(cfg.n_epochs) # epoch loop
        train_idxs = randperm(n_train)

        _epoch_logid = @progressid

        # @withprogress name="Batches" for b in 1:n_batches # batch loop
        for b in 1:n_batches # batch loop
            _batch_logid = @progressid
            begin
                idx = train_idxs[batch_idxs[b]:(batch_idxs[b + 1] - 1)]
                n_batch = length(idx)
                X = X_train[:, :, :, idx]
                Y = Y_train[:, :, :, idx]
                X, Y = add_training_noise(cfg.noise, X, Y)

                for i in 1:n_batch
                    if rand() > 0.5
                        X[:, :, :, i:i] = X[end:-1:1, :, :, i:i]
                        Y[:, :, :, i:i] = Y[end:-1:1, :, :, i:i]
                    end
                end

                # Forward pass of normalizing flow
                Zx, Zy, lgdet = coupling_network_device.forward(device(X), device(Y))

                # Loss function comes from target_distribution.
                nlogpz, dnlogpz_dz = compute_negative_log_density_with_gradient(target_distribution, Zx, Zy)
                push!(loss, nlogpz / n_batch)
                push!(loss_weighted_misfit_train, sum(dnlogpz_dz .* Zx) / prod(N) / n_batch)
                push!(logdet_train, -lgdet / prod(N)) # logdet is internally normalized by batch size

                # Set gradients of flow and summary network
                ΔZx = dnlogpz_dz
                coupling_network_device.backward(Float32.(ΔZx) / n_batch, Zx, Zy)

                for p in get_params(coupling_network_device)
                    if isnothing(p.grad)
                        continue
                    end
                    Flux.update!(opt, p.data, p.grad)
                end
                clear_grad!(coupling_network_device)

                if cfg.print_every != 0 && e % cfg.print_every == 0
                    message = string(
                        "Iter:",
                        "\n    epoch = ",
                        e,
                        "/",
                        cfg.n_epochs,
                        "\n    batch = ",
                        b,
                        "/",
                        n_batches,
                        "\n    negative log p(z) =  ",
                        loss[end],
                        "\n    weighted l2 =  ",
                        loss_weighted_misfit_train[end],
                        "\n    lgdet = ",
                        logdet_train[end],
                        "\n    f =     ",
                        loss[end] + logdet_train[end],
                        "\n",
                    )
                    # @logprogress message b/n_batches _id=_batch_logid
                    # if b == n_batches
                    #     print(message)
                    # end
                end
            end
        end
        push!(loss_total_train_epochs, mean(loss[end-n_batches+1:end] .+ logdet_train[end-n_batches+1:end]))

        # get objective mean metrics over testing batch
        nlogpz_valid, lgdet_test_val, weighted_misfit_valid = get_loss(
            coupling_network_device,
            X_test,
            Y_test;
            device,
            batch_size,
            N,
            target_distribution,
        )
        push!(logdet_test, -lgdet_test_val)
        push!(loss_test, nlogpz_valid)
        push!(loss_total_valid_epochs, loss_test[end] + logdet_test[end])
        push!(loss_weighted_misfit_valid, weighted_misfit_valid)

        if cfg.save_best
            if isnothing(best_train_loss) || loss_total_train_epochs[end] < best_train_loss
                best_train_loss = loss_total_train_epochs[end]
                best_train_epoch = e
                best_params = deepcopy(get_params(coupling_network_device) |> cpu)
            end
            if isnothing(best_valid_loss) || loss_total_valid_epochs[end] < best_valid_loss
                best_valid_loss = loss_total_valid_epochs[end]
                best_valid_epoch = e
                best_params = deepcopy(get_params(coupling_network_device) |> cpu)
            end
        end

        if cfg.cm_metrics
            # get conditional mean metrics over training batch
            cm_l2_train, cm_ssim_train = get_cm_l2_ssim(
                coupling_network_device,
                Xs,
                Ys,
                X_train[:, :, :, 1:(cfg.n_condmean)],
                Y_train[:, :, :, 1:(cfg.n_condmean)];
                device,
                num_samples=cfg.num_post_samples,
                batch_size,
            )
            push!(ssim, cm_ssim_train)
            push!(l2_cm, cm_l2_train)

            if size(X_test, 4) > 0
                # get conditional mean metrics over testing batch
                cm_l2_test, cm_ssim_test = get_cm_l2_ssim(
                    coupling_network_device,
                    Xs,
                    Ys,
                    X_test[:, :, :, 1:(cfg.n_condmean)],
                    Y_test[:, :, :, 1:(cfg.n_condmean)];
                    device,
                    num_samples=cfg.num_post_samples,
                    batch_size,
                )
                push!(ssim_test, cm_ssim_test)
                push!(l2_cm_test, cm_l2_test)
            end
        end

        if cfg.print_every != 0 && e % cfg.print_every == 0
            message = string(
                "Iter:",
                "\n    epoch = ",
                e,
                "/",
                cfg.n_epochs,
                "\nTraining batch average:",
                "\n    negative log p(z) =  ",
                mean(loss[(end - n_batches + 1):end]),
                "\n    weighted l2 =  ",
                mean(loss_weighted_misfit_train[(end - n_batches + 1):end]),
                "\n    lgdet = ",
                mean(logdet_train[(end - n_batches + 1):end]),
                "\n    f =     ",
                mean(
                    loss[(end - n_batches + 1):end] .+ logdet_train[(end - n_batches + 1):end]
                ),
                "\nValidation:",
                "\n    negative log p(z) =  ",
                loss_test[end],
                "\n    weighted l2 =  ",
                loss_weighted_misfit_valid[end],
                "\n    lgdet = ",
                logdet_test[end],
                "\n    f =     ",
                loss_test[end] + logdet_test[end],
                "\n",
            )
            @logprogress message e/cfg.n_epochs _id=_epoch_logid
            if e == cfg.n_epochs
                print(message)
            end
        end

        if cfg.early_stopping_training_loss.active
            early_stop = false
            for (de, prop, delta) in cfg.early_stopping_training_loss.look_backs
                if e > de
                    cur_loss = loss_total_train_epochs[end]
                    if prop == -1
                        # For instance, for de=200 and delta=-1f-6, find the most recent epoch where the loss is worse by at least 1f-6.
                        # If that is more than 200 epochs away, then we'll stop now, because the loss isn't improving fast enough.
                        most_recent_epoch = let
                            ep = 0
                            for (li, l) in Iterators.reverse(enumerate(loss_total_train_epochs))
                                if l >= cur_loss - delta
                                    ep = li
                                    break
                                end
                            end
                            ep
                        end
                        if e - most_recent_epoch > de 
                            early_stop = true
                            break
                        end
                    else
                        # Check for loss not improving for some number of epochs.
                        if mean(cur_loss - delta .>= loss_total_train_epochs[end-de:end-1]) > prop
                            early_stop = true
                            break
                        end
                    end
                end
            end
            if early_stop
                break
            end
        end

        if size(X_test, 4) > 0
            if cfg.early_stopping_validation_loss.active
                early_stop = false
                for (de, prop, delta) in cfg.early_stopping_validation_loss.look_backs
                    if e > de
                        cur_loss = loss_total_valid_epochs[end]
                        if prop == -1
                            # For instance, for de=200 and delta=-1f-6, find the most recent epoch where the loss is worse by at least 1f-6.
                            # If that is more than 200 epochs away, then we'll stop now, because the loss isn't improving fast enough.
                            most_recent_epoch = let
                                ep = 0
                                for (li, l) in Iterators.reverse(enumerate(loss_total_valid_epochs))
                                    if l >= cur_loss - delta
                                        ep = li
                                        break
                                    end
                                end
                                ep
                            end
                            if e - most_recent_epoch > de 
                                early_stop = true
                                break
                            end
                        else
                            # Check for loss not improving for some number of epochs.
                            if mean(cur_loss - delta .>= loss_total_valid_epochs[end-de:end-1]) > prop
                                early_stop = true
                                break
                            end
                        end
                    end
                end
                if early_stop
                    break
                end
            end
        end
    end
    if !isnothing(log_data)
        log_data[:coupling_network] = Dict{Symbol,Any}(
            :training => Dict{Symbol,Any}(
                :loss => loss,
                :logdet => logdet_train,
                :loss_total => loss_total_train_epochs,
                :loss_total_batches => loss .+ logdet_train,
                :ssim_cm => ssim,
                :l2_cm => l2_cm,
                :loss_weighted_misfit_train => loss_weighted_misfit_train,
            ),
            :testing => Dict{Symbol,Any}(
                :loss => loss_test,
                :logdet => logdet_test,
                :loss_total => loss_test .+ logdet_test,
                :ssim_cm => ssim_test,
                :l2_cm => l2_cm_test,
                :loss_weighted_misfit_valid => loss_weighted_misfit_valid,
            ),
        )
    end

    if cfg.save_best
        InvertibleNetworks.set_params!(coupling_network_device, best_params |> device)
    end
    return nothing
end
