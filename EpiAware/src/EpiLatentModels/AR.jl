@doc raw"
The autoregressive (AR) model struct.

# Constructors
- `AR(damp_prior::Distribution, std_prior::Distribution, init_prior::Distribution; p::Int = 1)`: Constructs an AR model with the specified prior distributions for damping coefficients, standard deviation, and initial conditions. The order of the AR model can also be specified.

- `AR(; damp_priors::Vector{D} = [truncated(Normal(0.0, 0.05))], std_prior::Distribution = truncated(Normal(0.0, 0.05), 0.0, Inf), init_priors::Vector{I} = [Normal()]) where {D <: Distribution, I <: Distribution}`: Constructs an AR model with the specified prior distributions for damping coefficients, standard deviation, and initial conditions. The order of the AR model is determined by the length of the `damp_priors` vector.

- `AR(damp_prior::Distribution, std_prior::Distribution, init_prior::Distribution, p::Int)`: Constructs an AR model with the specified prior distributions for damping coefficients, standard deviation, and initial conditions. The order of the AR model is explicitly specified.

# Examples

```julia
using Distributions
using EpiAware
ar = AR()
ar_model = generate_latent(ar, 10)
rand(ar_model)
```
"
struct AR{D <: Sampleable, S <: Sampleable, I <: Sampleable, P <: Int} <:
       AbstractTuringLatentModel
    "Prior distribution for the damping coefficients."
    damp_prior::D
    "Prior distribution for the standard deviation."
    std_prior::S
    "Prior distribution for the initial conditions"
    init_prior::I
    "Order of the AR model."
    p::P
    function AR(damp_prior::Distribution, std_prior::Distribution,
            init_prior::Distribution; p::Int = 1)
        damp_priors = fill(damp_prior, p)
        init_priors = fill(init_prior, p)
        return AR(; damp_priors = damp_priors, std_prior = std_prior,
            init_priors = init_priors)
    end

    function AR(; damp_priors::Vector{D} = [truncated(Normal(0.0, 0.05), 0, 1)],
            std_prior::Distribution = HalfNormal(0.1),
            init_priors::Vector{I} = [Normal()]) where {
            D <: Distribution, I <: Distribution}
        p = length(damp_priors)
        damp_prior = _expand_dist(damp_priors)
        init_prior = _expand_dist(init_priors)
        return AR(damp_prior, std_prior, init_prior, p)
    end

    function AR(damp_prior::Distribution, std_prior::Distribution,
            init_prior::Distribution, p::Int)
        @assert p>0 "p must be greater than 0"
        @assert length(damp_prior)==length(init_prior) "damp_prior and init_prior must have the same length"
        @assert p==length(damp_prior) "p must be equal to the length of damp_prior"
        new{typeof(damp_prior), typeof(std_prior), typeof(init_prior), typeof(p)}(
            damp_prior, std_prior, init_prior, p
        )
    end
end

@doc raw"
Generate a latent AR series.

# Arguments

- `latent_model::AR`: The AR model.
- `n::Int`: The length of the AR series.

# Returns
- `ar::Vector{Float64}`: The generated AR series.
- `params::NamedTuple`: A named tuple containing the generated parameters (`σ_AR`, `ar_init`, `damp_AR`).

# Notes
- The length of `damp_prior` and `init_prior` must be the same.
- `n` must be longer than the order of the autoregressive process.
"
@model function EpiAwareBase.generate_latent(latent_model::AR, n)
    p = latent_model.p
    ϵ_t ~ MvNormal(I(n - p))
    σ_AR ~ latent_model.std_prior
    ar_init ~ latent_model.init_prior
    damp_AR ~ latent_model.damp_prior

    @assert n>p "n must be longer than order of the autoregressive process"

    # Initialize the AR series with the initial values
    ar = Vector{eltype(ϵ_t)}(undef, n)
    ar[1:p] = ar_init

    # Generate the rest of the AR series
    for t in (p + 1):n
        ar[t] = damp_AR' * ar[(t - p):(t - 1)] + σ_AR * ϵ_t[t - p]
    end

    return ar, (; σ_AR, ar_init, damp_AR)
end
