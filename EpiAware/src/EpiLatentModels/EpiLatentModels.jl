module EpiLatentModels

"""
Module for defining latent models.
"""

using ..EpiAwareBase

using Turing, Distributions, DocStringExtensions

export RandomWalk, AR, ThinPlateSpline, DiffLatentModel

include("docstrings.jl")
include("randomwalk.jl")
include("autoregressive.jl")
include("thinplatespline.jl")
include("difflatentmodel.jl")
include("utils.jl")

end
