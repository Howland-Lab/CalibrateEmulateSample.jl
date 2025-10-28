module ParameterPriors

# Export functions
export construct_priors

# Import modules
using ..Functions: create_prior_dict
using Distributions
using CalibrateEmulateSample
const EKP = CalibrateEmulateSample.EnsembleKalmanProcesses
const PD = EKP.ParameterDistributions


# Construct prior distributions for parameters
function construct_priors(turb_model::String)
    # Standard k-epsilon
    if turb_model == "stdke"
        params_true = [0.09, 1.44, 1.92, 1.0, 1.0, 1.3]
        param_names = ["C_mu", "C_1", "C_2", "C_3", "sig_k", "sig_e"]
        prior_lower = [0.045, 0.72, 0.96, 0.5, 0.5, 0.65]
        prior_upper = [0.135, 2.16, 2.88, 1.5, 1.5, 1.95]
    # MOST k-epsilon
    elseif turb_model == "mostke"
        params_true = [0.03, 1.21, 1.92, 1.0, 1.3]
        param_names = ["C_mu", "C_1", "C_2", "sig_k", "sig_e"]
        prior_lower = [0.015, 0.605, 0.96, 0.5, 0.65]
        prior_upper = [0.045, 1.815, 2.88, 1.5, 1.95]
    else
        error("Unsupported turbulence model: $turb_model")
    end

    n_param = length(param_names)
    params_true = reshape(params_true, (n_param, 1))

    # Generate prior dictionaries
    priors_list = [create_prior_dict(name) for name in param_names]
    priors = PD.ParameterDistribution(priors_list)

    return params_true, param_names, n_param, prior_lower, prior_upper, priors
end

end # module