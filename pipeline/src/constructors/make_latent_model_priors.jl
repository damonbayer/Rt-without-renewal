"""
Constructs and returns a dictionary of prior distributions for the latent model
parameters. This is the default method.

# Arguments
- `pipeline`: An instance of the `AbstractEpiAwarePipeline` type.

# Returns
A dictionary containing the following prior distributions:
- `"transformed_process_init_prior"`: A normal distribution with mean 0.0 and
standard deviation 0.25.
- `"std_prior"`: A half-normal distribution with standard deviation 0.25.
- `"damp_param_prior"`: A beta distribution with shape parameters 0.5 and 0.5.

"""
function make_latent_model_priors(pipeline::AbstractEpiAwarePipeline)
    transformed_process_init_prior = Normal(0.0, 0.25)
    std_prior = HalfNormal(0.25)
    damp_param_prior = Beta(0.5, 0.5)

    return Dict(
        "transformed_process_init_prior" => transformed_process_init_prior,
        "std_prior" => std_prior,
        "damp_param_prior" => damp_param_prior
    )
end
