
export TargetUnitNormal, TargetUnitNormalPlusUniform
export make_target_distribution
export compute_negative_log_density_with_gradient

"""Target density is Gaussian with zero mean and identity covariance.""" 
struct TargetUnitNormal
end

"""Target density is proportional to a Gaussian with zero mean and identity
covariance plus alpha times an improper uniform prior.""" 
struct TargetUnitNormalPlusUniform
    alpha
end

"""Returns an object that can compute a negative log density and derivative of log density with respective to the input."""
function make_target_distribution end

function compute_negative_log_density_with_gradient end

make_target_distribution(::TargetUnitNormalOptions) = TargetUnitNormal()
make_target_distribution(a::TargetUnitNormalPlusUniformOptions) = TargetUnitNormalPlusUniform(a.uniform_weight)


function compute_negative_log_density_with_gradient(::TargetUnitNormal, Zx::AbstractArray{T,N}, Zy=nothing) where {T, N}
    nz = prod(size(Zx)[1:end-1])
    return 0.5 * norm(Zx)^2 / nz, Zx
end

function log_gamma_n2p1(n)
    return logfactorial(2*n) - logfactorial(n) - n*log(4) + 0.5 * log(pi)
end

function log_nsphere_volume_diff(a, b, n)
    return n/2 * log(pi) - log_gamma_n2p1(n) + log(b^n - a^n)
end

function compute_negative_log_density_with_gradient(target::TargetUnitNormalPlusUniform, Zx::AbstractArray{T,N}, Zy=nothing) where {T, N}
    nz = prod(size(Zx)[1:end-1])
    ny = prod(size(Zx)[1:end-1])
    alpha = target.alpha
    if alpha == 0
        return 0.5 * norm(Zx)^2 / nz, Zx
    end

    Zx_norm = [norm(Zxi) for Zxi in eachslice(Zx; dims=ndims(Zx))]
    Zx_norm = reshape(Zx_norm, ones(Int, ndims(Zx) - 1)..., size(Zx)[end])

    if isnothing(Zy)
        Zy_norm = T(0)
    else
        Zy_norm = [norm(Zyi) for Zyi in eachslice(Zy; dims=ndims(Zy))]
        Zy_norm = reshape(Zy_norm, ones(Int, ndims(Zy) - 1)..., size(Zy)[end])
    end

    pz_normal = exp.(-Zx_norm .^ 2 ./ nz ./ 2 .- Zy_norm .^ 2 ./ ny ./ 2)
    pz = pz_normal .+ alpha
    nlogpz = -log.(pz)
    dnlogpz_dz = Zx .* Flux.sigmoid(-Zx_norm .^ 2 / nz / 2 .- Zy_norm .^ 2 / ny / 2 .- log(alpha)) ./ nz
    offset = Zy_norm .^ 2 / ny / 2
    return sum(nlogpz) - sum(offset), dnlogpz_dz
end
