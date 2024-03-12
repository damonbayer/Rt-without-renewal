abstract type AbstractModel end

"""
    abstract type AbstractEpiModel <: AbstractModel end

The abstract supertype for all structs that define a model for generating unobserved/latent
    infections.
"""
abstract type AbstractEpiModel <: AbstractModel end

abstract type AbstractLatentModel <: AbstractModel end

"""
The abstract supertype for all structs that define a model for generating observed case data
and/or generative modeling of case data.
"""
abstract type AbstractObservationModel <: AbstractModel end

@doc raw"""
Generate unobserved/latent infections based on the given `epi_model <: AbstractEpimodel`
    and a latent process path ``Z_t``.

The `generate_latent_infs` function implements a model of generating unobserved/latent
infections conditional on a latent process. Which model of generating unobserved/latent
infections to be implemented is set by the type of `epi_model`. If no implemention is
defined for the given `epi_model`, then `EpiAware` will return a warning and return
`nothing`.

## Interface to `Turing.jl` probablilistic programming language (PPL)

Apart from the no implementation fallback method, the `generate_latent_infs` implementation
function should be a constructor function for a
    [`DynamicPPL.Model`](https://turinglang.org/DynamicPPL.jl/stable/api/#DynamicPPL.Model)
    object.
"""
function generate_latent_infs(epi_model::AbstractEpiModel, Z_t)
    @warn "No concrete implementation for `generate_latent_infs` is defined."
    return nothing
end

function generate_latent(latent_model::AbstractLatentModel, n)
    @info "No concrete implementation for generate_latent is defined."
    return nothing
end

@doc raw"""
Constructor function for a observation model with case data `y_t` and unobserved infections
`I_t`.

The `generate_observations` function implements a model of generating and/or scoring
observered cast data `y_t`. Which model for observations is implemented is set by the type
of `observation_model`. If no implemention is defined for the type of `observation_model`,
then `EpiAware` will pass a warning and return `nothing`.

## Interface to `Turing.jl` probablilistic programming language (PPL)

Apart from the no implementation fallback method, the `generate_observations` implementation
function should return a constructor function for a
    [`DynamicPPL.Model`](https://turinglang.org/DynamicPPL.jl/stable/api/#DynamicPPL.Model)
object. Observed data `y_t` passed to the observation model constructor will accumulate
log-likelihood in Bayesian inference, whereas `y_t = missing` will generate case data
conditional on unobserved infections `I_t`. Priors for model parameters are fields of
`observation_model`.
"""
function generate_observations(observation_model::AbstractObservationModel,
        y_t,
        I_t;
        pos_shift)
    @info "No concrete implementation for `generate_observations` is defined."
    return nothing
end
