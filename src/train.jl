using Flux: Flux
using LinearAlgebra: norm
using MLUtils: splitobs, obsview
using ImageQualityIndexes: assess_ssim
using Random: randn, randperm
using InvertibleNetworks: InvertibleNetworks, reset!, clear_grad!, get_params
using Statistics: mean
using ProgressLogging: @withprogress, @logprogress, @progressid

export train_network!, get_cm_l2_ssim, get_loss

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

function get_loss(G, X_batch, Y_batch; device=gpu, batch_size, N, noise_lev_x, noise_lev_y)
    num_test = size(Y_batch)[end]
    if num_test == 0
        return NaN, NaN
    end
    l2_total = 0
    logdet_total = 0
    num_batches = div(num_test, batch_size)
    for i in 1:num_batches
        x_i = X_batch[:, :, :, ((i - 1) * batch_size + 1):(i * batch_size)]
        y_i = Y_batch[:, :, :, ((i - 1) * batch_size + 1):(i * batch_size)]

        x_i .+= noise_lev_x * randn(Float32, size(x_i))
        y_i .+= noise_lev_y * randn(Float32, size(y_i))

        Zx, Zy, lgdet = cpu(G.forward(device(x_i), device(y_i)))
        l2_total += norm(Zx)^2 / (prod(N) * batch_size)
        logdet_total += lgdet / prod(N)
    end
    return l2_total / (num_batches), logdet_total / (num_batches)
end

function train_network!(filter::NormalizingFlowFilter, Xs, Ys; log_data=nothing)
    device = filter.device
    cfg = filter.training_config

    if cfg.reset_weights
        InvertibleNetworks.set_params!(filter.network_device, get_params(reset_network(filter.network_device)))
    end

    if cfg.reset_optimizer
        opt = reset_optimizer(filter.opt)
    else
        opt = filter.opt
    end

    N = size(Xs)[1:(end - 1)]

    # Training logs
    loss = Vector{Float64}()
    logdet_train = Vector{Float64}()
    ssim = Vector{Float64}()
    l2_cm = Vector{Float64}()

    loss_test = Vector{Float64}()
    logdet_test = Vector{Float64}()
    ssim_test = Vector{Float64}()
    l2_cm_test = Vector{Float64}()

    # Use MLutils to split into training and validation set
    num_samples = size(Xs)[end]
    shuffle_idxs = randperm(num_samples)

    train_split, test_split = splitobs(num_samples; at=cfg.validation_perc)

    train_split = shuffle_idxs[train_split]
    test_split = shuffle_idxs[test_split]

    X_train = obsview(Xs, train_split)
    Y_train = obsview(Ys, train_split)

    X_test = obsview(Xs, test_split)
    Y_test = obsview(Ys, test_split)

    # train_loader = DataLoader(XY_train, batchsize=cfg.batch_size, shuffle=true, partial=false);

    # training & test indexes
    n_train = size(X_train)[end]
    n_test = size(X_test)[end]
    n_batches = cld(n_train, cfg.batch_size)
    # n_batches_test = cld(n_test, cfg.batch_size)

    batch_idxs = collect(1:cfg.batch_size:(n_train + 1))
    if batch_idxs[end] != n_train+1
        append!(batch_idxs, n_train+1)
    end

    @withprogress name="Epochs" for e in 1:(cfg.n_epochs) # epoch loop
        train_idxs = randperm(n_train)

        _epoch_logid = @progressid

        @withprogress name="Batches" for b in 1:n_batches # batch loop
            _batch_logid = @progressid
            begin
                idx = train_idxs[batch_idxs[b]:(batch_idxs[b + 1] - 1)]
                n_batch = length(idx)
                X = X_train[:, :, :, idx]
                Y = Y_train[:, :, :, idx]
                X .+= cfg.noise_lev_x * randn(Float32, size(X))
                Y .+= cfg.noise_lev_y * randn(Float32, size(Y))

                for i in 1:n_batch
                    if rand() > 0.5
                        X[:, :, :, i:i] = X[end:-1:1, :, :, i:i]
                        Y[:, :, :, i:i] = Y[end:-1:1, :, :, i:i]
                    end
                end

                # Forward pass of normalizing flow
                Zx, Zy, lgdet = filter.network_device.forward(device(X), device(Y))

                # Loss function is l2 norm
                append!(loss, norm(Zx)^2 / (prod(N) * n_batch))  # normalize by image size and batch size
                append!(logdet_train, -lgdet / prod(N)) # logdet is internally normalized by batch size

                # Set gradients of flow and summary network
                #filter.network_device.backward(Zx / cfg.batch_size, Zx, Zy; Y_save=Y|> device)
                filter.network_device.backward(Zx / n_batch, Zx, Zy)

                for p in get_params(filter.network_device)
                    Flux.update!(opt, p.data, p.grad)
                end
                clear_grad!(filter.network_device)

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
                        "\n    f l2 =  ",
                        loss[end],
                        "\n    lgdet = ",
                        logdet_train[end],
                        "\n    f =     ",
                        loss[end] + logdet_train[end],
                        "\n",
                    )
                    @logprogress message b/n_batches _id=_batch_logid
                    if b == n_batches
                        print(message)
                    end
                end
            end
        end
        # get objective mean metrics over testing batch
        l2_test_val, lgdet_test_val = get_loss(
            filter.network_device,
            X_test,
            Y_test;
            device,
            batch_size=cfg.batch_size,
            N,
            noise_lev_x=cfg.noise_lev_x,
            noise_lev_y=cfg.noise_lev_y,
        )
        append!(logdet_test, -lgdet_test_val)
        append!(loss_test, l2_test_val)

        # get conditional mean metrics over training batch
        cm_l2_train, cm_ssim_train = get_cm_l2_ssim(
            filter.network_device,
            Xs,
            Ys,
            X_train[:, :, :, 1:(cfg.n_condmean)],
            Y_train[:, :, :, 1:(cfg.n_condmean)];
            device,
            num_samples=cfg.num_post_samples,
            batch_size=cfg.batch_size,
        )
        append!(ssim, cm_ssim_train)
        append!(l2_cm, cm_l2_train)

        if size(X_test, 4) > 0
            # get conditional mean metrics over testing batch
            cm_l2_test, cm_ssim_test = get_cm_l2_ssim(
                filter.network_device,
                Xs,
                Ys,
                X_test[:, :, :, 1:(cfg.n_condmean)],
                Y_test[:, :, :, 1:(cfg.n_condmean)];
                device,
                num_samples=cfg.num_post_samples,
                batch_size=cfg.batch_size,
            )
            append!(ssim_test, cm_ssim_test)
            append!(l2_cm_test, cm_l2_test)
        end

        if cfg.print_every != 0 && e % cfg.print_every == 0
            message = string(
                "Iter:",
                "\n    epoch = ",
                e,
                "/",
                cfg.n_epochs,
                "\nTraining batch average:",
                "\n    f l2 =  ",
                mean(loss[(end - n_batches + 1):end]),
                "\n    lgdet = ",
                mean(logdet_train[(end - n_batches + 1):end]),
                "\n    f =     ",
                mean(
                    loss[(end - n_batches + 1):end] .+ logdet_train[(end - n_batches + 1):end]
                ),
                "\nValidation:",
                "\n    f l2 =  ",
                loss_test[end],
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
    end
    if !isnothing(log_data)
        log_data[:network_training] = Dict{Symbol,Any}(
            :training => Dict{Symbol,Any}(
                :loss => loss,
                :logdet => logdet_train,
                :ssim_cm => ssim,
                :l2_cm => l2_cm,
                :split => train_split,
            ),
            :testing => Dict{Symbol,Any}(
                :loss => loss_test,
                :logdet => logdet_test,
                :ssim_cm => ssim_test,
                :l2_cm => l2_cm_test,
                :split => test_split,
            ),
        )
    end
end
