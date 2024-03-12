"""
Generate an observation kernel matrix corresponding to discrete delay interval `delay_int`
and with size `time_horizon`.

## Mathematical specification

A common delay model for case data ``y_t`` with the discrete delay distribution ``d`` and
unobserved infections ``I_t`` is,

```math
y_t = \sum_{j\geq 0} I_{t-j} d_j.
```
NB:

- The first position in the `delay_int` vector corresponds to zero time lag between
unobserved infection and becoming a case.
- This model assumes that all infections become a case; this is only appropriate for models
with no population size.

The action of the discrete delay distribution can be replicated for a time series of length
`time_horizon` by constructing a matrix `K` such that,

```math
y = K_d I.
```

# Returns
- `K::SparseMatrixCSC{Float64, Int}`: The observation kernel matrix.
"""
function _generate_observation_kernel(delay_int, time_horizon)
    K = zeros(eltype(delay_int), time_horizon, time_horizon) |> SparseMatrixCSC
    for i in 1:time_horizon, j in 1:time_horizon
        m = i - j
        if m >= 0 && m <= (length(delay_int) - 1)
            K[i, j] = delay_int[m + 1]
        end
    end
    return K
end

struct DelayObservations{T <: AbstractFloat, S <: Sampleable} <: AbstractObservationModel
    delay_kernel::SparseMatrixCSC{T, Integer}
    neg_bin_cluster_factor_prior::S

    function DelayObservations(delay_int,
            time_horizon,
            neg_bin_cluster_factor_prior)
        @assert all(delay_int .>= 0) "Delay interval must be non-negative"
        @assert sum(delay_int)≈1 "Delay interval must sum to 1"

        K = _generate_observation_kernel(delay_int, time_horizon)

        new{eltype(K), typeof(neg_bin_cluster_factor_prior)}(K,
            neg_bin_cluster_factor_prior)
    end

    function DelayObservations(;
            delay_distribution::ContinuousDistribution,
            time_horizon::Integer,
            neg_bin_cluster_factor_prior::Sampleable,
            D_delay,
            Δd = 1.0)
        delay_int = create_discrete_pmf(delay_distribution; Δd = Δd, D = D_delay)
        return DelayObservations(delay_int, time_horizon, neg_bin_cluster_factor_prior)
    end
end

function default_delay_obs_priors()
    return (:neg_bin_cluster_factor_prior => truncated(
        Normal(0, 0.1 * sqrt(pi) / sqrt(2)), 0.0, Inf),) |> Dict
end

function generate_observations(observation_model::AbstractObservationModel,
        y_t,
        I_t;
        pos_shift)
    @info "No concrete implementation for generate_observations is defined."
    return nothing
end

@model function generate_observations(observation_model::DelayObservations,
        y_t,
        I_t;
        pos_shift)

    #Parameters
    neg_bin_cluster_factor ~ observation_model.neg_bin_cluster_factor_prior

    #Predictive distribution
    expected_obs = observation_model.delay_kernel * I_t .+ pos_shift

    if ismissing(y_t)
        y_t = Vector{Int}(undef, length(expected_obs))
    end

    for i in eachindex(y_t)
        y_t[i] ~ NegativeBinomialMeanClust(
            expected_obs[i], neg_bin_cluster_factor^2
        )
    end

    return y_t, (; neg_bin_cluster_factor,)
end
