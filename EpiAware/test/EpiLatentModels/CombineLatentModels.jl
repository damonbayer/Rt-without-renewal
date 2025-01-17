@testitem "CombineLatentModels constructor works as expected" begin
    using Distributions: Normal
    int = Intercept(Normal(0, 1))
    ar = AR()
    comb = CombineLatentModels([int, ar])
    @test typeof(comb) <: AbstractTuringLatentModel
    @test comb.models == [int, ar]
end

@testitem "CombineLatentModels generate_latent method works as expected: FixedIntecept + custom" begin
    using Turing

    struct NextScale <: AbstractTuringLatentModel end

    @model function EpiAware.EpiAwareBase.generate_latent(model::NextScale, n::Int)
        scale = 2
        return scale_vect = fill(scale, n), (; nscale = scale)
    end

    s = FixedIntercept(1)
    ns = NextScale()
    comb = CombineLatentModels([s, ns])
    comb_model = generate_latent(comb, 5)
    comb_model_out = comb_model()

    @test typeof(comb_model) <: DynamicPPL.Model
    @test length(comb_model_out[1]) == 5
    @test all(comb_model_out[1] .== fill(3.0, 5))
    @test comb_model_out[2].intercept == 1.0
    @test comb_model_out[2].nscale == 2.0
end

@testitem "CombineLatentModels generate_latent method works as expected: Intercept + AR" begin
    using Turing
    using Distributions: Normal
    using HypothesisTests: ExactOneSampleKSTest, pvalue

    int = Intercept(Normal(0, 1))
    ar = AR()
    n = 10
    comb = CombineLatentModels([int, ar])
    comb_model = generate_latent(comb, n)

    # Test constant if conditioning on zero residuals
    no_residual_mdl = comb_model | (ϵ_t = zeros(n - 1), ar_init = [0.0])
    y_const, θ_const = no_residual_mdl()

    @test all(y_const .== fill(θ_const.intercept, n))

    # Check against linear regression by conditioning on normal residuals
    # Generate data
    fix_intercept = 0.5
    normal_res_mdl = comb_model | (damp_AR = [0.0], σ_AR = 1.0, intercept = fix_intercept)
    y, θ = normal_res_mdl()

    # Fit no-slope linear regression as a model test
    @model function no_slope_linear_regression(y)
        @submodel y_pred, θ = generate_latent(comb, n)
        y ~ MvNormal(y_pred, ones(n))
    end

    ns_regression_mdl = no_slope_linear_regression(y) |
                        (damp_AR = [0.0], σ_AR = 1.0, ϵ_t = zeros(n - 1), ar_init = [0.0])
    chain = sample(ns_regression_mdl, NUTS(), 5000, progress = false)

    # Theoretical posterior distribution for intercept
    # if \beta ~ int.intercept_prior = N(\mu_0, \sigma_0) and \sigma^2 = 1 for
    #    the white noise
    # then the posterior distribution for the intercept is Normal
    # \mathcal{N}(\text{mean} = (n * \sigma_0^2 * ȳ + \mu_0) / (n * \sigma_0^2 + 1),
    #             \text{var} = \sigma_0^2 / (n * \sigma_0^2 + 1))

    post_mean = (n * var(int.intercept_prior) * mean(y) + mean(int.intercept_prior)) /
                (n * var(int.intercept_prior) + 1)
    post_var = var(int.intercept_prior) / (n * var(int.intercept_prior) + 1)
    post_dist = Normal(post_mean, sqrt(post_var))

    samples = get(chain, :intercept).intercept |> vec
    ks_test_pval = ExactOneSampleKSTest(samples, post_dist) |> pvalue
    @test ks_test_pval > 1e-6
end
