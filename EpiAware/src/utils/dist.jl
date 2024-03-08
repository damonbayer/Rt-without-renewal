"""
Create a discrete probability mass function (PMF) from a given distribution, assuming that the
primary event happens at `primary_approximation_point * Δd` within an intial censoring interval. Common
single-censoring approximations are `primary_approximation_point = 0` (left-hand approximation),
`primary_approximation_point = 1` (right-hand) and `primary_approximation_point = 0.5` (midpoint).

Arguments:
- `dist`: The distribution from which to create the PMF.
- ::Val{:single_censored}: A dummy argument to dispatch to this method. The purpose of the `Val`
type argument is that to use `single-censored` approximation is an active decision.
- `primary_approximation_point`: A approximation point for the primary time in its censoring interval.
Default is 0.5 for midpoint approximation.
- `Δd`: The step size for discretizing the domain. Default is 1.0.
- `D`: The upper bound of the domain. Must be greater than `Δd`.

Returns:
- A vector representing the PMF.

Raises:
- `AssertionError` if the minimum value of `dist` is negative.
- `AssertionError` if `Δd` is not positive.
- `AssertionError` if `D` is not greater than `Δd`.
"""
function create_discrete_pmf(dist::Distribution,
        ::Val{:single_censored};
        primary_approximation_point = 0.5,
        Δd = 1.0,
        D)
    @assert minimum(dist)>=0.0 "Distribution must be non-negative"
    @assert Δd>0.0 "Δd must be positive"
    @assert D>Δd "D must be greater than Δd"
    @assert primary_approximation_point >= 0.0&&primary_approximation_point <= 1.0 "`primary_approximation_point` must be in [0,1]."

    ts = Δd:Δd:D |> collect
    @assert ts[end]==D "D must be a multiple of Δd."
    ts = [primary_approximation_point * Δd; ts] #This covers situation where primary_approximation_point == 1

    ts .|> (t -> cdf(dist, t)) |> diff |> p -> p ./ sum(p)
end

"""
Create a discrete probability mass function (PMF) from a given distribution, assuming
a uniform distribution over primary event times with censoring intervals of width `Δd` for
both primary and secondary events. The CDF for the time from the left edge of the interval
containing the primary event to the secondary event is created by direct numerical integration
of the convolution of the CDF of `dist` with the uniform density on `[0,Δd)`, the discrete PMF
for double censored delays is then found using simple differencing on the CDF.


Arguments:
- `dist`: The distribution from which to create the PMF.
- `Δd`: The step size for discretizing the domain. Default is 1.0.
- `D`: The upper bound of the domain. Must be greater than `Δd`.

Returns:
- A vector representing the PMF.

Raises:
- `AssertionError` if the minimum value of `dist` is negative.
- `AssertionError` if `Δd` is not positive.
- `AssertionError` if `D` is not greater than `Δd`.
"""
function create_discrete_pmf(dist::Distribution; Δd = 1.0, D)
    @assert minimum(dist)>=0.0 "Distribution must be non-negative."
    @assert Δd>0.0 "Δd must be positive."
    @assert D>Δd "D must be greater than Δd."

    ts = 0.0:Δd:D |> collect

    @assert ts[end]==D "D must be a multiple of Δd."

    ∫F(dist, t, Δd) = quadgk(u -> cdf(dist, t - u) / Δd, 0.0, Δd)[1]

    ts .|> (t -> ∫F(dist, t, Δd)) |> diff |> p -> p ./ sum(p)
end

"""
Compute the negative moment generating function (MGF) for a given rate `r` and weights `w`.

# Arguments
- `r`: The rate parameter.
- `w`: An abstract vector of weights.

# Returns
The value of the negative MGF.

"""
function neg_MGF(r, w::AbstractVector)
    return sum([w[i] * exp(-r * i) for i in 1:length(w)])
end

function dneg_MGF_dr(r, w::AbstractVector)
    return -sum([w[i] * i * exp(-r * i) for i in 1:length(w)])
end

"""
This function computes an approximation to the exponential growth rate `r`
given the reproductive ratio `R₀` and the discretized generation interval `w` with
discretized interval width `Δd`. This is based on the implicit solution of

```math
G(r) - {1 \\over R_0} = 0.
```

where

```math
G(r) = \\sum_{i=1}^n w_i e^{-r i}.
```

is the negative moment generating function (MGF) of the generation interval distribution.

The two step approximation is based on:
    1. Direct solution of implicit equation for a small `r` approximation.
    2. Improving the approximation using Newton's method for a fixed number of steps `newton_steps`.

Returns:
- The approximate value of `r`.
"""
function R_to_r(R₀, w::Vector{T}; newton_steps = 2, Δd = 1.0) where {T <: AbstractFloat}
    mean_gen_time = dot(w, 1:length(w)) * Δd
    # Small r approximation as initial guess
    r_approx = (R₀ - 1) / (R₀ * mean_gen_time)
    # Newton's method
    for _ in 1:newton_steps
        r_approx -= (R₀ * neg_MGF(r_approx, w) - 1) / (R₀ * dneg_MGF_dr(r_approx, w))
    end
    return r_approx
end

function R_to_r(R₀, epi_model::AbstractEpiModel; newton_steps = 2, Δd = 1.0)
    R_to_r(R₀, epi_model.data.gen_int; newton_steps = newton_steps, Δd = Δd)
end

"""
    r_to_R(r, w)

Compute the reproductive ratio given exponential growth rate `r`
    and discretized generation interval `w`.

# Arguments
- `r`: The exponential growth rate.
- `w`: discretized generation interval.

# Returns
- The reproductive ratio.
"""
function r_to_R(r, w::AbstractVector)
    return 1 / neg_MGF(r, w::AbstractVector)
end

"""
    NegativeBinomialMeanClust(μ, α)

Compute the mean-cluster factor negative binomial distribution.

# Arguments
- `μ`: The mean of the distribution.
- `α`: The clustering factor parameter.

# Returns
A `NegativeBinomial` distribution object.
"""
function NegativeBinomialMeanClust(μ, α)
    ex_σ² = (α * μ^2) + 1e-6
    p = μ / (μ + ex_σ² + 1e-6)
    r = μ^2 / ex_σ²
    return NegativeBinomial(r, p)
end