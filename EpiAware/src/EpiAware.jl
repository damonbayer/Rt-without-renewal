"""
    EpiAware

This module provides functionality for calculating Rt (effective reproduction number) with and without
    considering renewal processes.

# Dependencies

- Distributions: for working with probability distributions.
- Turing: for probabilistic programming.
- LogExpFunctions: for working with logarithmic, logistic and exponential functions.
- LinearAlgebra: for linear algebra operations.
- SparseArrays: for working with sparse arrays.
- Random: for generating random numbers.
- ReverseDiff: for automatic differentiation.
- Optim: for optimization algorithms.
- Zygote: for automatic differentiation.

"""
module EpiAware

using Distributions,
    Turing,
    LogExpFunctions,
    LinearAlgebra,
    SparseArrays,
    Random,
    ReverseDiff,
    Optim,
    Parameters,
    QuadGK

# Exported utilities
export create_discrete_pmf, default_rw_priors, default_delay_obs_priors

# Exported types
export EpiData, Renewal, ExpGrowthRate, DirectInfections

# Exported Turing model constructors
export make_epi_inference_model, random_walk, delay_observations

include("epimodel.jl")
include("utilities.jl")
include("models.jl")
include("latent-processes.jl")
include("observation-processes.jl")

end
