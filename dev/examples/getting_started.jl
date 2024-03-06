### A Pluto.jl notebook ###
# v0.19.40

using Markdown
using InteractiveUtils

# ╔═╡ c593a2a0-d7f5-11ee-0931-d9f65ae84a72
# hideall
let
    docs_dir = dirname(dirname(@__DIR__))
    pkg_dir = dirname(docs_dir)

    using Pkg: Pkg
    Pkg.activate(docs_dir)
    Pkg.develop(; path = pkg_dir)
    Pkg.instantiate()
end;

# ╔═╡ da479d8d-1312-4b98-b0af-5be52dffaf3f
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
end

# ╔═╡ 3ebc8384-f73d-4597-83a7-07a3744fed61
md"
# Getting stated with `EpiAware`

This tutorial introduces the basic functionality of `EpiAware`. `EpiAware` is a package for making inferences on epidemiological case/determined infection data using a model-based approach.

## `EpiAware` models
The models we consider are discrete-time $t = 1,\dots, T$ with a latent random process, $Z_t$ generating stochasticity in the number of new infections $I_t$ at each time step. Observations are treated as downstream random variables determined by the actual infections and a model of infection to observation delay.

#### Mathematical definition
```math
\begin{align}
Z_\cdot &\sim \mathcal{P}(\mathbb{R}^T) | \theta_Z, \\
I_0 &\sim f_0(\theta_I), \\
I_t &\sim g_I(\{I_s, Z_s\}_{s < t}, \theta_{I}), \\
y_t &\sim f_O(\{I_s\}_{s \leq t}, \theta_{O}).
\end{align}
```
Where, $\mathcal{P}(\mathbb{R}^T) | \theta_Z$ is a parametric process on $\mathbb{R}^T$. $f_0$ and $f_O$ are parametric distributions on, respectively, the initial number of infections and the observed case data conditional on underlying infections. $g_I$ is distribution of new infections conditional on infections and latent process in the past. Note that we assume that new infections are conditional on the strict past, whereas new observations can depend on infections on the same time step.

#### Code structure outline

An `EpiAware` model in code is created from three modular components:

- A `LatentModel`: This defines the model for $Z_\cdot$. This chooses $\mathcal{P}(\mathbb{R}^T) | \theta_Z$.
- An `EpiModel`: This defines a generative process for infections conditional on the latent process. This chooses $f_0(\theta_I)$, and $g_I(\{I_s, Z_s\}_{s < t}, \theta_{I})$.
- An `ObservationModel`: This defines the observation model. This chooses $f_O({I_s}_{s \leq t}, \theta_{O})$

#### Reproductive number
`EpiAware` models do not need to specify a time-varying reproductive number $\mathcal{R}_t$ to generate $I_\cdot$, however, this is often a quantity of interest. When not directly used we will typically back-calculate $\mathcal{R}_t$ from the generated infections:

```math
\mathcal{R}_t = {I_t \over \sum_{s \geq 1} g_s I_{t-s} }.
```

Where $g_s$ is a discrete generation interval. For this reason, even when not using a reproductive number approach directly, we ask for a generation interval.
"

# ╔═╡ 5a0d5ab8-e985-4126-a1ac-58fe08beee38
md"
## Random walk `LatentModel`

As an example, we choose the latent process as a random walk with parameters $\theta_Z$:

- ``Z_0``: Initial position.
- ``\sigma^2_{Z}``: The step-size variance.

Conditional on the parameters the random walk is then generated by white noise:
```math
\begin{align}
Z_t &= Z_0 + \sigma_{RW} \sum_{t= 1}^T \epsilon_t, \\
\epsilon_t &\sim \mathcal{N}(0,1).
\end{align}
```

In `EpiAware` we provide a constructor for random walk latent models with priors for $\theta_Z$. We choose priors,
```math
\begin{align}
Z_0 &\sim \mathcal{N}(0,1),\\
\sigma^2_Z &\sim \text{HalfNormal}(0.01).
\end{align}
```
"

# ╔═╡ 56ae496b-0094-460b-89cb-526627991717
rwp = EpiAware.RandomWalk(Normal(),
    truncated(Normal(0.0, 0.02), 0.0, Inf))

# ╔═╡ 767beffd-1ef5-4e6c-9ac6-edb52e60fb44
md"
## Direct infection `EpiModel`

This is a simple model where the unobserved log-infections are directly generated by the latent process $Z$.
```math
\log I_t = \log I_0 + Z_t.
```

As discussed above, we still ask for a defined generation interval, which can be used to calculate $\mathcal{R}_t$.

"

# ╔═╡ 9e43cbe3-94de-44fc-a788-b9c7adb34218
truth_GI = Gamma(2, 5)

# ╔═╡ f067284f-a1a6-44a6-9b79-f8c2de447673
md"
The `EpiData` constructor performs double interval censoring to convert our _continuous_ estimate of the generation interval into a discretized version. We also implement right truncation using the keyword `D_gen`.
"

# ╔═╡ c0662d48-4b54-4b6d-8c91-ddf4b0e3aa43
model_data = EpiData(truth_GI, D_gen = 10.0)

# ╔═╡ fd72094f-1b95-4d07-a8b0-ef47dc560dfc
md"
We can supply a prior for the initial log_infections.
"

# ╔═╡ 6639e66f-7725-4976-81b2-6472419d1a62
log_I0_prior = Normal(log(100.0), 1.0)

# ╔═╡ df5e59f8-3185-4bed-9cca-7c266df17cec
md"
And construct the `EpiModel`.
"

# ╔═╡ 6fbdd8e6-2323-4352-9185-1f31a9cf9012
epi_model = DirectInfections(model_data, log_I0_prior)

# ╔═╡ 5e62a50a-71f4-4902-b1c9-fdf51fe145fa
md"


### Delayed Observations `ObservationModel`

The observation model is a negative binomial distribution with mean `μ` and cluster factor `1 / r`. Delays are implemented
as the action of a sparse kernel on the infections $I(t)$.

```math
\begin{align}
y_t &\sim \text{NegBinomial}(\mu = \sum_{s\geq 0} K[t, t-s] I(s), r), \\
1 / r &\sim \text{Gamma}(3, 0.05/3).
\end{align}
```
"

# ╔═╡ e813d547-6100-4c43-b84c-8cebe306bda8
md"
We also set up the inference to occur over 100 days.
"

# ╔═╡ c7580ae6-0db5-448e-8b20-4dd6fcdb1ae0
time_horizon = 100

# ╔═╡ 0aa3fcbd-0831-45b8-9a2c-7ffbabf5895f
md"
We choose a simple observation model where infections are observed 0, 1, 2, 3 days later with equal probability.
"

# ╔═╡ 448669bc-99f4-4823-b15e-fcc9040ba31b
obs_model = EpiAware.DelayObservations(
    fill(0.25, 4),
    time_horizon,
    truncated(Gamma(5, 0.05 / 5), 1e-3, 1.0)
)

# ╔═╡ e49713e8-4840-4083-8e3f-fc52d791be7b
md"
## Generate cases from the `EpiAware` model

Having chosen an `EpiModel`, `LatentModel` and `ObservationModel`, we can implement the model as a [`Turing`](https://turinglang.org/dev/) model using  `make_epi_aware`.

By giving `missing` to the first argument, we indicate that case data will be _generated_ from the model rather than treated as fixed.
"

# ╔═╡ abeff860-58c3-4644-9325-66ffd4446b6d
full_epi_aware_mdl = make_epi_aware(missing, time_horizon;
    epi_model = epi_model,
    latent_model = rwp,
    observation_model = obs_model)

# ╔═╡ 821628fb-8044-48b0-aa4f-0b7b57a2f45a
md"
We choose some fixed parameters:
- Initial incidence is 100.
- In the direct infection model, the initial incidence and in the initial value of the random walk form a non-identifiable pair. Therefore, we fix $Z_0 = 0$.
"

# ╔═╡ 36b34fd2-2891-42ca-b5dc-abb482e516ee
fixed_parameters = (rw_init = 0.0, init_incidence = log(100.0))

# ╔═╡ 0aadd9e3-7f91-4b45-9663-67d11335f0d0
md"
We fix these parameters using `fix`, and generate a random epidemic.
"

# ╔═╡ 7e0e6012-8648-4f84-a25a-8b0138c4b72a
cond_generative_model = fix(full_epi_aware_mdl, fixed_parameters)

# ╔═╡ b20c28be-7b07-410c-a33b-ea5ad6828c12
random_epidemic = rand(cond_generative_model)

# ╔═╡ d073e63b-62da-4743-ace0-78ef7806bc0b
true_infections = generated_quantities(cond_generative_model, random_epidemic).I_t

# ╔═╡ f68b4e41-ac5c-42cd-a8c2-8761d66f7543
let
    plot(true_infections,
        label = "I_t",
        xlabel = "Time",
        ylabel = "Infections",
        title = "Generated Infections")
    scatter!(random_epidemic.y_t, lab = "generated cases")
end

# ╔═╡ b5bc8f05-b538-4abf-aa84-450bf2dff3d9
md"
## Inference
Fixing $Z_0 = 0$ for the random walk was based on inference principles; in this model $Z_0$ and $\log I_0$ are non-identifiable.

However, we now treat the generated data as `truth_data` and make inference without fixing any other parameters.

We do the inference by MCMC/NUTS using the `Turing` NUTS sampler with default warm-up steps.
"

# ╔═╡ c8ce0d46-a160-4c40-a055-69b3d10d1770
truth_data = random_epidemic.y_t

# ╔═╡ b4033728-b321-4100-8194-1fd9fe2d268d
inference_mdl = fix(
    make_epi_aware(truth_data, time_horizon; epi_model = epi_model,
        latent_model = rwp, observation_model = obs_model),
    (rw_init = 0.0,)
)

# ╔═╡ 3eb5ec5e-aae7-478e-84fb-80f2e9f85b4c
chn = sample(inference_mdl,
    NUTS(; adtype = AutoReverseDiff(true)),
    MCMCThreads(),
    250,
    4;
    drop_warmup = true)

# ╔═╡ 30498cc7-16a5-441a-b8cd-c19b220c60c1
md"
### Predictive plotting

We can spaghetti plot generated case data from the version of the model _which hasn't conditioned on case data_ using posterior parameters inferred from the version conditioned on observed data. This is known as _posterior predictive checking_, and is a useful diagnostic tool for Bayesian inference (see [here](http://www.stat.columbia.edu/~gelman/book/BDA3.pdf)).

Because we are using synthetic data we can also plot the model predictions for the _unobserved_ infections and check that (at least in this example) we were able to capture some unobserved/latent variables in the process accurate.
"

# ╔═╡ e9df22b8-8e4d-4ab7-91ea-c01f2239b3e5
let
    post_check_mdl = fix(full_epi_aware_mdl, (rw_init = 0.0,))
    post_check_y_t = mapreduce(hcat, generated_quantities(post_check_mdl, chn)) do gen
        gen.generated_y_t
    end

    predicted_I_t = mapreduce(hcat, generated_quantities(inference_mdl, chn)) do gen
        gen.I_t
    end

    p1 = plot(post_check_y_t, c = :grey, alpha = 0.05, lab = "")
    scatter!(p1, truth_data,
        lab = "Observed cases",
        xlabel = "Time",
        ylabel = "Cases",
        title = "Post. predictive checking: cases",
        ylims = (-0.5, maximum(truth_data) * 1.5),
        c = :green)

    p2 = plot(predicted_I_t, c = :grey, alpha = 0.05, lab = "")
    scatter!(p2, true_infections,
        lab = "Actual infections",
        xlabel = "Time",
        ylabel = "Unobserved Infections",
        title = "Post. predictions: infections",
        ylims = (-0.5, maximum(true_infections) * 1.5),
        c = :red)

    plot(p1, p2,
        layout = (1, 2),
        size = (700, 400))
end

# ╔═╡ fd6321b1-4c3a-4123-b0dc-c45b951e0b80
md"
As well as checking the posterior predictions for latent infections, we can also check how well inference recovered unknown parameters, such as the random walk variance or the cluster factor of the negative binomial observations.
"

# ╔═╡ 10d8fe24-83a6-47ac-97b7-a374481473d3
let
    parameters_to_plot = (:σ²_RW, :neg_bin_cluster_factor)

    plts = map(parameters_to_plot) do name
        var_samples = chn[name] |> vec
        histogram(var_samples,
            bins = 50,
            norm = :pdf,
            lw = 0,
            fillalpha = 0.5,
            lab = "MCMC")
        vline!([getfield(random_epidemic, name)], lab = "True value")
        title!(string(name))
    end
    plot(plts..., layout = (2, 1))
end

# ╔═╡ 81efe8ca-b753-4a12-bafc-a887a999377b
md"
## Reproductive number back-calculation

As mentioned at the top, we _don't_ directly use the concept of reproductive numbers in this note. However, we can back-calculate the implied $\mathcal{R}(t)$ values, conditional on the specified generation interval being correct.

Here we spaghetti plot posterior sampled time-varying reproductive numbers against the actual.
"

# ╔═╡ 15b9f37f-8d5f-460d-8c28-d7f2271fd099
let
    n = epi_model.data.len_gen_int
    Rt_denom = [dot(reverse(epi_model.data.gen_int), true_infections[(t - n):(t - 1)])
                for t in (n + 1):length(true_infections)]
    true_Rt = true_infections[(n + 1):end] ./ Rt_denom

    predicted_Rt = mapreduce(hcat, generated_quantities(inference_mdl, chn)) do gen
        _It = gen.I_t
        _Rt_denom = [dot(reverse(epi_model.data.gen_int), _It[(t - n):(t - 1)])
                     for t in (n + 1):length(_It)]
        Rt = _It[(n + 1):end] ./ _Rt_denom
    end

    plt = plot((n + 1):time_horizon, predicted_Rt, c = :grey, alpha = 0.05, lab = "")
    plot!(plt, (n + 1):time_horizon, true_Rt,
        lab = "true Rt",
        xlabel = "Time",
        ylabel = "Rt",
        title = "Post. predictions: reproductive number",
        c = :red,
        lw = 2)
end

# ╔═╡ Cell order:
# ╟─c593a2a0-d7f5-11ee-0931-d9f65ae84a72
# ╟─3ebc8384-f73d-4597-83a7-07a3744fed61
# ╠═da479d8d-1312-4b98-b0af-5be52dffaf3f
# ╟─5a0d5ab8-e985-4126-a1ac-58fe08beee38
# ╠═56ae496b-0094-460b-89cb-526627991717
# ╟─767beffd-1ef5-4e6c-9ac6-edb52e60fb44
# ╠═9e43cbe3-94de-44fc-a788-b9c7adb34218
# ╟─f067284f-a1a6-44a6-9b79-f8c2de447673
# ╠═c0662d48-4b54-4b6d-8c91-ddf4b0e3aa43
# ╟─fd72094f-1b95-4d07-a8b0-ef47dc560dfc
# ╠═6639e66f-7725-4976-81b2-6472419d1a62
# ╟─df5e59f8-3185-4bed-9cca-7c266df17cec
# ╠═6fbdd8e6-2323-4352-9185-1f31a9cf9012
# ╟─5e62a50a-71f4-4902-b1c9-fdf51fe145fa
# ╟─e813d547-6100-4c43-b84c-8cebe306bda8
# ╠═c7580ae6-0db5-448e-8b20-4dd6fcdb1ae0
# ╟─0aa3fcbd-0831-45b8-9a2c-7ffbabf5895f
# ╠═448669bc-99f4-4823-b15e-fcc9040ba31b
# ╟─e49713e8-4840-4083-8e3f-fc52d791be7b
# ╠═abeff860-58c3-4644-9325-66ffd4446b6d
# ╟─821628fb-8044-48b0-aa4f-0b7b57a2f45a
# ╠═36b34fd2-2891-42ca-b5dc-abb482e516ee
# ╟─0aadd9e3-7f91-4b45-9663-67d11335f0d0
# ╠═7e0e6012-8648-4f84-a25a-8b0138c4b72a
# ╠═b20c28be-7b07-410c-a33b-ea5ad6828c12
# ╠═d073e63b-62da-4743-ace0-78ef7806bc0b
# ╟─f68b4e41-ac5c-42cd-a8c2-8761d66f7543
# ╟─b5bc8f05-b538-4abf-aa84-450bf2dff3d9
# ╠═c8ce0d46-a160-4c40-a055-69b3d10d1770
# ╠═b4033728-b321-4100-8194-1fd9fe2d268d
# ╠═3eb5ec5e-aae7-478e-84fb-80f2e9f85b4c
# ╟─30498cc7-16a5-441a-b8cd-c19b220c60c1
# ╠═e9df22b8-8e4d-4ab7-91ea-c01f2239b3e5
# ╟─fd6321b1-4c3a-4123-b0dc-c45b951e0b80
# ╠═10d8fe24-83a6-47ac-97b7-a374481473d3
# ╟─81efe8ca-b753-4a12-bafc-a887a999377b
# ╠═15b9f37f-8d5f-460d-8c28-d7f2271fd099
