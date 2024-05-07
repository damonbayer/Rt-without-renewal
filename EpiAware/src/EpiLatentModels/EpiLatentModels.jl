"""
Module for defining latent models.
"""
module EpiLatentModels

using ..EpiAwareBase

using ..EpiAwareUtils: HalfNormal

using Turing, Distributions, DocStringExtensions, LinearAlgebra

#Export models
export RandomWalk, AR, DiffLatentModel, BroadcastLatentModel

# Export broadcast rules
export RepeatEach, RepeatBlock

# Export helper functions
export dayofweek, weekly

include("docstrings.jl")
include("RandomWalk.jl")
include("AR.jl")
include("DiffLatentModel.jl")
include("broadcast/LatentModel.jl")
include("broadcast/rules.jl")
include("broadcast/helpers.jl")
include("utils.jl")

end