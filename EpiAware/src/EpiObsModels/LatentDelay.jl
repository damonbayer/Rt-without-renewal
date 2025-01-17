@doc raw"
The `LatentDelay` struct represents an observation model that introduces a latent delay in the observations. It is a subtype of `AbstractTuringObservationModel`.

## Fields
- `model::M`: The underlying observation model.
- `pmf::T`: The probability mass function (PMF) representing the delay distribution.

## Constructors
- `LatentDelay(model::M, distribution::C; D = 15, Δd = 1.0) where {M <: AbstractTuringObservationModel, C <: ContinuousDistribution}`: Constructs a `LatentDelay` object with the given underlying observation model and continuous distribution. The `D` parameter specifies the number of discrete delay intervals, and the `Δd` parameter specifies the width of each delay interval.

- `LatentDelay(model::M, pmf::T) where {M <: AbstractTuringObservationModel, T <: AbstractVector{<:Real}}`: Constructs a `LatentDelay` object with the given underlying observation model and delay PMF.

## Examples
```julia
using Distributions, Turing, EpiAware
obs = LatentDelay(NegativeBinomialError(), truncated(Normal(5.0, 2.0), 0.0, Inf))
obs_model = generate_observations(obs, missing, fill(10, 10))
rand(obs_model)
```
"
struct LatentDelay{M <: AbstractTuringObservationModel, T <: AbstractVector{<:Real}} <:
       AbstractTuringObservationModel
    model::M
    pmf::T

    function LatentDelay(model::M, distribution::C; D = 15,
            Δd = 1.0) where {
            M <: AbstractTuringObservationModel, C <: ContinuousDistribution}
        pmf = censored_pmf(distribution; Δd = Δd, D = D)
        return LatentDelay(model, pmf)
    end

    function LatentDelay(model::M,
            pmf::T) where {M <: AbstractTuringObservationModel, T <: AbstractVector{<:Real}}
        @assert all(pmf .>= 0) "Delay interval must be non-negative"
        @assert isapprox(sum(pmf), 1) "Delay interval must sum to 1"
        new{typeof(model), typeof(pmf)}(model, pmf)
    end
end

@doc raw"
Generates observations based on the `LatentDelay` observation model.

## Arguments
- `obs_model::LatentDelay`: The `LatentDelay` observation model.
- `y_t`: The current observations.
- `I_t`: The current infection indicator.

## Returns
- `y_t`: The updated observations.
- `obs_aux`: Additional observation-related variables.

"
@model function EpiAwareBase.generate_observations(obs_model::LatentDelay, y_t, Y_t)
    kernel = generate_observation_kernel(obs_model.pmf, length(Y_t))
    expected_obs = kernel * Y_t

    @submodel y_t, obs_aux = generate_observations(obs_model.model, y_t, expected_obs)

    return y_t, (; obs_aux...)
end
