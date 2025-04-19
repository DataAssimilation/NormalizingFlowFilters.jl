
include("grad_test.jl")

using NormalizingFlowFilters.InvertibleNetworks: get_grads, get_params, set_params!

image_shape = (1, 1, 1)

forward = function (x; with_grad=false)
    G = estimator.network_device
    set_params!(G, x)
    Zx, Zy, logdet = G.forward(X, Y)
    J = sum(0.5 * (Zx .^ 2))/ size(X, 4)  - logdet
    J = sum(0.5 * (Zx .^ 2))/ size(X, 4)  - logdet
    if with_grad
        G.backward(Zx / size(X, 4), Zx, Zy)
        dJdx = get_grads(G)
        return J, dJdx
    end
    return J
end

forward_theirs = function (x; with_grad=false)
    G = estimator.network_device
    set_params!(G, x)
    Zx, Zy, logdet = G.forward(X, Y)
    f = sum(0.5 * (Zx .^ 2))/size(Zx, 4) - logdet
    ΔZx = Zx / size(Zx, 4)
    G.backward(ΔZx, Zx, Zy)
    return f, get_grads(G)
end

Random.seed!(10)
x0 = deepcopy(get_params(estimator.network_device))
Δx = deepcopy(x0)
for Δx_i in Δx
    target_norm = norm(Δx_i) * 1e-3
    Δx_i.data .= randn(size(target_norm))
    Δx_i.data .*= target_norm ./ norm(Δx_i)
end

n_batch = 1
X = randn(image_shape...,n_batch)
Y = randn(image_shape...,n_batch)
J, dJdx = forward(x0; with_grad=true)

grad_test(forward, x0, Δx, dJdx; ΔJ=nothing, maxiter=6, h0=1e-1, stol=1e-1, hfactor=8e-1, unittest=:test)
