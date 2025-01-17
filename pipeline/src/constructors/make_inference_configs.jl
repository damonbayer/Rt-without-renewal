"""
Create inference configurations for the given pipeline. This is the default method.

# Arguments
- `pipeline`: An instance of `AbstractEpiAwarePipeline`.

# Returns
- An object representing the inference configurations.

"""
function make_inference_configs(pipeline::AbstractEpiAwarePipeline)
    gi_param_dict = make_gi_params(pipeline)
    namemodel_vect = make_epiaware_name_model_pairs(pipeline)
    igps = make_inf_generating_processes(pipeline)

    inference_configs = Dict("igp" => igps, "latent_namemodels" => namemodel_vect,
        "gi_mean" => gi_param_dict["gi_means"], "gi_std" => gi_param_dict["gi_stds"]) |>
                        dict_list

    return inference_configs
end
