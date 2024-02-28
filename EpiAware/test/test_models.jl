
@testitem "`make_epi_aware` with direct infections and RW latent process runs" begin
    using Distributions, Turing, DynamicPPL
    # Define test inputs
    y_t = missing # Data will be generated from the model
    data = EpiData([0.2, 0.3, 0.5], exp)
    pos_shift = 1e-6

    #Define the epi_model
    epi_model = DirectInfections(data, Normal())

    #Define the latent process model
    rwp = EpiAware.RandomWalk(Normal(0.0, 1.0),
        truncated(Normal(0.0, 0.05), 0.0, Inf))

    #Define the observation model
    delay_distribution = Gamma(2.0, 5 / 2)
    time_horizon = 365
    D_delay = 14.0
    Δd = 1.0

    obs_model = EpiAware.DelayObservations(delay_distribution = delay_distribution,
        time_horizon = time_horizon,
        neg_bin_cluster_factor_prior = Gamma(5, 0.05 / 5),
        D_delay = D_delay,
        Δd = Δd)

    # Create full epi model and sample from it
    test_mdl = make_epi_aware(y_t, time_horizon; epi_model = epi_model,
        latent_model_model = rwp,
        observation_model = obs_model, pos_shift)
    gen = generated_quantities(test_mdl, rand(test_mdl))

    #Check model sampled
    @test eltype(gen.generated_y_t) <: Integer
    @test eltype(gen.I_t) <: AbstractFloat
    @test length(gen.I_t) == time_horizon
end

@testitem "`make_epi_aware` with Exp growth rate and RW latent process runs" begin
    using Distributions, Turing, DynamicPPL
    # Define test inputs
    y_t = missing# rand(1:10, 365) # Data will be generated from the model
    data = EpiData([0.2, 0.3, 0.5], exp)
    pos_shift = 1e-6

    #Define the epi_model
    epi_model = EpiAware.ExpGrowthRate(data, Normal())

    #Define the latent process model
    r_3 = log(2) / 3.0
    rwp = EpiAware.RandomWalk(
        truncated(Normal(0.0, r_3 / 3), -r_3, r_3), # 3 day doubling time at 3 sigmas in prior
        truncated(Normal(0.0, 0.01), 0.0, 0.1))

    #Define the observation model - no delay model
    time_horizon = 5
    obs_model = EpiAware.DelayObservations([1.0],
        time_horizon,
        truncated(Gamma(5, 0.05 / 5), 1e-3, 1.0))

    # Create full epi model and sample from it
    test_mdl = make_epi_aware(y_t,
        time_horizon;
        epi_model = epi_model,
        latent_model_model = rwp,
        observation_model = obs_model,
        pos_shift)

    chn = sample(test_mdl, Prior(), 1000)
    gens = generated_quantities(test_mdl, chn)

    #Check model sampled
    @test eltype(gens[1].generated_y_t) <: Integer
    @test eltype(gens[1].I_t) <: AbstractFloat
    @test length(gens[1].I_t) == time_horizon
end

@testitem "`make_epi_aware` with Renewal and RW latent process runs" begin
    using Distributions, Turing, DynamicPPL
    # Define test inputs
    y_t = missing# rand(1:10, 365) # Data will be generated from the model
    data = EpiData([0.2, 0.3, 0.5], exp)
    pos_shift = 1e-6

    #Define the epi_model
    epi_model = EpiAware.Renewal(data, Normal())

    #Define the latent process model
    r_3 = log(2) / 3.0
    rwp = EpiAware.RandomWalk(
        truncated(Normal(0.0, r_3 / 3), -r_3, r_3), # 3 day doubling time at 3 sigmas in prior
        truncated(Normal(0.0, 0.01), 0.0, 0.1))

    #Define the observation model - no delay model
    time_horizon = 5
    obs_model = EpiAware.DelayObservations([1.0],
        time_horizon,
        truncated(Gamma(5, 0.05 / 5), 1e-3, 1.0))

    # Create full epi model and sample from it
    test_mdl = make_epi_aware(y_t,
        time_horizon;
        epi_model = epi_model,
        latent_model_model = rwp,
        observation_model = obs_model,
        pos_shift)

    chn = sample(test_mdl, Prior(), 1000)
    gens = generated_quantities(test_mdl, chn)

    #Check model sampled
    @test eltype(gens[1].generated_y_t) <: Integer
    @test eltype(gens[1].I_t) <: AbstractFloat
    @test length(gens[1].I_t) == time_horizon
end
