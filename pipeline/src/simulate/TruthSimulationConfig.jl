"""
A configuration struct for truth simulation to compare to inference.

NB: This config assumes Gamma distributed generation interval, specified by
mean and standard deviation.
"""
@kwdef struct TruthSimulationConfig{T <: Real, F <: Function}
    "True time-varying process. Default this is the time-varing reproductive
    number"
    truth_process::Vector{T}
    "True generation interval distribution mean."
    gi_mean::T
    "True generation interval distribution std."
    gi_std::T
    "Infection-generating model type. Default is `:Renewal`."
    igp::UnionAll = Renewal
    "Maximum next generation interval when discretized. Default is 21 days."
    D_gen::T = 60.0
    "Transformation function"
    transformation::F = exp
    "Delay distribution: Default is Gamma(4, 5/4)."
    delay_distribution::Distribution = Gamma(4, 5 / 4)
    "Maximum delay when discretized. Default is 15 days."
    D_obs::T = 15.0
    "Prior for log initial infections. Default is Normal(4.6, 1e-5)."
    log_I0_prior::Distribution = Normal(log(100.0), 1e-5)
    "Prior for negative binomial cluster factor. Default is HalfNormal(0.1)."
    cluster_factor_prior::Distribution = HalfNormal(0.1)
end

"""
Simulates or infers the truth process and observations based on the given configuration.

# Arguments
- `config::TruthSimulationConfig`: The configuration object containing the parameters for the simulation.

# Returns
A dictionary containing the sampled infections and observations, along with other relevant information.

"""
function simulate(config::TruthSimulationConfig)
    #Define infection-generating model
    shape = (config.gi_mean / config.gi_std)^2
    scale = config.gi_std^2 / config.gi_mean
    gen_distribution = Gamma(shape, scale)

    model_data = EpiData(gen_distribution = gen_distribution,
        D_gen = config.D_gen,
        transformation = config.transformation)

    epi = config.igp(model_data, config.log_I0_prior)

    # Sample infections
    inf_mdl = generate_latent_infs(epi, log.(config.truth_process))
    I_t = inf_mdl()

    #Define the infection conditional observation distribution
    obs = LatentDelay(
        NegativeBinomialError(cluster_factor_prior = config.cluster_factor_prior),
        config.delay_distribution; D = config.D_obs)

    #Sample observations
    obs_model = generate_observations(obs, missing, I_t)
    yt, θ = obs_model()

    #Return the sampled infections and observations

    return Dict("I_t" => I_t, "y_t" => yt, "cluster_factor" => θ.cluster_factor,
        "truth_process" => config.truth_process,
        "truth_gi_mean" => config.gi_mean,
        "truth_gi_std" => config.gi_std)
end
