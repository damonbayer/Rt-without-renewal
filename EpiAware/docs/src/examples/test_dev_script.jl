let
    docs_dir = dirname(dirname(@__DIR__))
    pkg_dir = dirname(docs_dir)

    using Pkg: Pkg
    Pkg.activate(docs_dir)
    Pkg.develop(; path = pkg_dir)
    Pkg.instantiate()
end;

##
begin
    using EpiAware
    using Turing
    using Distributions
    using StatsPlots
    using Random
    using DynamicPPL
    using Statistics
    using DataFramesMeta
    using LinearAlgebra
    using Transducers
    using ReverseDiff
    Random.seed!(1)
end

## Model definition

rwp = EpiAware.RandomWalk(
    init_prior = Normal(),
    std_prior = HalfNormal(0.05)
)

weekly_rwp = BroadcastLatentModel(rwp, 7, RepeatBlock())
ar = AR(; damp_priors = [truncated(Normal(0.8, .05), 0, 1), truncated(Normal(0.1, .05), 0, 1)],
            std_prior = truncated(Normal(), 0, Inf),
            init_priors = [Normal(-1.0, 0.1), Normal(-1.0, 0.1)],
)

weekly_ar = BroadcastLatentModel(ar, 7, RepeatBlock())

truth_GI = Gamma(2, 5)
model_data = EpiData(gen_distribution = truth_GI,
    D_gen = 10.0)

log_I0_prior = Normal(log(100.0), 1.0)
epi = DirectInfections(model_data, log_I0_prior)

obs = LatentDelay(
    NegativeBinomialError(cluster_factor_prior = Gamma(10, 0.05 / 10)),
    fill(0.25, 4),
)

obs_direct = NegativeBinomialError(cluster_factor_prior = Gamma(10, 0.05 / 10))

##
n_latent_inference = 20

@model function test_latent_inference(rwp, n)
    @submodel Z, _ = generate_latent(rwp, n)
    y_t ~ MvNormal(Z, 0.1)
    return (Z = Z, y_t = y_t)
end

chn_rad_f = sample(cond_mdl, NUTS(adtype = AutoReverseDiff(false)), 2_000)
chn_rad_t = sample(cond_mdl, NUTS(adtype = AutoReverseDiff(true)), 2_000)

##
function test_inference(adtype; n = 28)
    mdl = test_latent_inference(weekly_rwp, n)
    test_y_t = vcat(randn(n ÷ 2) .+ 10 , randn(n ÷ 2) .+ 20)
    cond_mdl = mdl | (y_t = test_y_t,)
    chn_fwd = sample(cond_mdl, NUTS(adtype = adtype), 2_000)

    gens = generated_quantities(cond_mdl, chn_fwd)
    plt = plot()
    for gen in gens[1:10:end]
        plot!(plt, gen.Z, lab = "", alpha = 0.1, c = :grey)
    end
    plot!(plt, test_y_t, lab = "Observed", lw = 2, c = :red)
    plt
end

inference_plts = map([
    AutoForwardDiff(), AutoReverseDiff(true), AutoReverseDiff(false)]) do adtype
    test_inference(adtype)
end

plts = plot(inference_plts..., layout = (3, 1), size = (500, 400),
    title = "rwp only: " .* ["Forward" "Reverse (true)" "Reverse (false)"] .* " AD")
##

n = 20
@model function test_latent_and_latent_inf_inference(rwp, epi, n)
    @submodel Z, _ = generate_latent(rwp, n)
    @submodel I_t = generate_latent_infs(epi, Z)
    y_t ~ MvNormal(log.(I_t), 0.1)
    return (Z = Z, y_t = y_t, I_t)
end

##

function test_inference_lat_and_lat_inf(adtype; n = 20)
    mdl = test_latent_and_latent_inf_inference(ar, epi, n)
    test_y_t = randn(n) .+ 1
    cond_mdl = mdl | (y_t = test_y_t,)
    chn = sample(cond_mdl, NUTS(adtype = adtype), 2_000)

    gens = generated_quantities(cond_mdl, chn)
    plt = plot()
    for gen in gens[1:10:end]
        plot!(plt, gen.I_t, lab = "", alpha = 0.1, c = :grey)
    end
    plot!(plt, exp.(test_y_t), lab = "Observed", lw = 2, c = :red)
    plt
end

inference_plts_lat_and_lat_inf = map([
    AutoForwardDiff(), AutoReverseDiff(true), AutoReverseDiff(false)]) do adtype
    test_inference_lat_and_lat_inf(adtype)
end

plts = plot(inference_plts_lat_and_lat_inf...,
    layout = (3, 1),
    size = (500, 400),
    title = "rwp + lat inf only: " .* ["Forward" "Reverse (true)" "Reverse (false)"] .*
            " AD")

##

@model function test_inference_lat_and_lat_inf_obs(rwp, epi, obs, n, obs_yt = missing)
    @submodel Z, _ = generate_latent(rwp, n)
    @submodel I_t = generate_latent_infs(epi, Z)
    @submodel gen_y_t, _ = generate_observations(obs, obs_yt, I_t)

    return (; Z, I_t, gen_y_t)
end

##

function test_inference_lat_and_lat_inf_obs(adtype; n = 20)
    test_y_t = randn(n) .+ 1 .|> exp .|> _y -> round(Int64, _y)
    cond_mdl = test_inference_lat_and_lat_inf_obs(rwp, epi, obs, n, test_y_t)

    # cond_mdl = mdl | (y_t = test_y_t,)
    chn = sample(cond_mdl, NUTS(adtype = adtype), 2_000)

    gens = generated_quantities(cond_mdl, chn)
    plt = plot()
    for gen in gens[1:10:end]
        plot!(plt, gen.I_t, lab = "", alpha = 0.1, c = :grey)
    end
    scatter!(plt, test_y_t, lab = "Observed", lw = 2, c = :red)
    hline!(plt, [(exp(1))], lab = "", c = :black)

    plt
end

inference_plts_lat_and_lat_inf_obs = map([
    AutoForwardDiff(), AutoReverseDiff(true), AutoReverseDiff(false)]) do adtype
        test_inference_lat_and_lat_inf_obs(adtype)
end

plts = plot(inference_plts_lat_and_lat_inf_obs...,
    layout = (3, 1),
    size = (500, 400),
    title = "rwp + lat inf + obs only: " .* ["Forward" "Reverse (true)" "Reverse (false)"] .*
            " AD")


##

function test_inference_gen_epiaware(adtype; n = 20)
    test_y_t = randn(n) .+ 1 .|> exp .|> _y -> round(Int64, _y)

    cond_mdl = generate_epiaware(test_y_t, n, epi;latent_model = rwp, observation_model = obs)

    # cond_mdl = mdl | (y_t = test_y_t,)
    chn = sample(cond_mdl, NUTS(adtype = adtype), 2_000)

    gens = generated_quantities(cond_mdl, chn)
    plt = plot()
    for gen in gens[1:10:end]
        plot!(plt, gen.I_t, lab = "", alpha = 0.1, c = :grey)
    end
    scatter!(plt, test_y_t, lab = "Observed", lw = 2, c = :red)
    hline!(plt, [(exp(1))], lab = "", c = :black)

    plt
end

inference_plts_gen_epiaware = map([
    AutoForwardDiff(), AutoReverseDiff(true), AutoReverseDiff(false)]) do adtype
        test_inference_gen_epiaware(adtype)
end

plts = plot(inference_plts_gen_epiaware...,
    layout = (3, 1),
    size = (500, 400),
    title = "epiaware: " .* ["Forward" "Reverse (true)" "Reverse (false)"] .*
            " AD")


##

function test_inference_gen_epiaware2(adtype; n = 20)
    test_y_t = randn(n) .+ 1 .|> exp .|> _y -> round(Int64, _y)
    epi_prob = EpiProblem(epi, rwp, obs, (1, n))
    cond_mdl = generate_epiaware(epi_prob, (y_t = test_y_t,))

    # cond_mdl = mdl | (y_t = test_y_t,)
    chn = sample(cond_mdl, NUTS(adtype = adtype), 2_000)

    gens = generated_quantities(cond_mdl, chn)
    plt = plot()
    for gen in gens[1:10:end]
        plot!(plt, gen.I_t, lab = "", alpha = 0.1, c = :grey)
    end
    scatter!(plt, test_y_t, lab = "Observed", lw = 2, c = :red)
    hline!(plt, [(exp(1))], lab = "", c = :black)

    plt
end


inference_plts_gen_epiaware = map([
    AutoForwardDiff(), AutoReverseDiff(true), AutoReverseDiff(false)]) do adtype
        test_inference_gen_epiaware2(adtype)
end

plts = plot(inference_plts_gen_epiaware...,
    layout = (3, 1),
    size = (500, 400),
    title = "epiprob: " .* ["Forward" "Reverse (true)" "Reverse (false)"] .*
            " AD")


## Multiple chains


function test_inference_gen_epiaware_multi(adtype; n = 20)
    test_y_t = randn(n) .+ 1 .|> exp .|> _y -> round(Int64, _y)
    epi_prob = EpiProblem(epi, rwp, obs, (1, n))
    cond_mdl = generate_epiaware(epi_prob, (y_t = test_y_t,))

    # cond_mdl = mdl | (y_t = test_y_t,)
    chn = sample(cond_mdl, NUTS(adtype = adtype), MCMCThreads(), 500, 4)

    gens = generated_quantities(cond_mdl, chn)
    plt = plot()
    for gen in gens[1:10:end]
        plot!(plt, gen.I_t, lab = "", alpha = 0.1, c = :grey)
    end
    scatter!(plt, test_y_t, lab = "Observed", lw = 2, c = :red)
    hline!(plt, [(exp(1))], lab = "", c = :black)

    plt
end


inference_plts_gen_epiaware = map([
    AutoForwardDiff(), AutoReverseDiff(true), AutoReverseDiff(false)]) do adtype
        test_inference_gen_epiaware_multi(adtype)
end

plts = plot(inference_plts_gen_epiaware...,
    layout = (3, 1),
    size = (500, 400),
    title = "epiprob multi: " .* ["Forward" "Reverse (true)" "Reverse (false)"] .*
            " AD")

## Raw latent delay
##

function test_inference_gen_epiaware_direct(adtype; n = 20)
    test_y_t = randn(n) .+ 1 .|> exp .|> _y -> round(Int64, _y)

    cond_mdl = generate_epiaware(test_y_t, n, epi;latent_model = weekly_ar, observation_model = obs_direct)

    # cond_mdl = mdl | (y_t = test_y_t,)
    chn = sample(cond_mdl, NUTS(adtype = adtype), 2_000)

    gens = generated_quantities(cond_mdl, chn)
    plt = plot()
    for gen in gens[1:10:end]
        plot!(plt, gen.I_t, lab = "", alpha = 0.1, c = :grey)
    end
    scatter!(plt, test_y_t, lab = "Observed", lw = 2, c = :red)
    hline!(plt, [(exp(1))], lab = "", c = :black)

    plt
end

inference_plts_gen_epiaware_dir = map([
    AutoForwardDiff(), AutoReverseDiff(true), AutoReverseDiff(false)]) do adtype
        test_inference_gen_epiaware_direct(adtype)
end

plts = plot(inference_plts_gen_epiaware_dir...,
    layout = (3, 1),
    size = (500, 400),
    title = "epiaware direct obs: " .* ["Forward" "Reverse (true)" "Reverse (false)"] .*
            " AD")


##
