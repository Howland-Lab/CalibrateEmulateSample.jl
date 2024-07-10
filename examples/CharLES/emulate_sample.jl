# Import modules
include(joinpath(@__DIR__, "..", "ci", "linkfig.jl"))

# Import modules
using Distributions  # probability distributions and associated functions
using LinearAlgebra
ENV["GKSwstype"] = "100"
using StatsPlots
using Plots
using Random
using JLD2
using DelimitedFiles

# CES
using CalibrateEmulateSample.Emulators
using CalibrateEmulateSample.MarkovChainMonteCarlo
using CalibrateEmulateSample.Utilities
using CalibrateEmulateSample.EnsembleKalmanProcesses
using CalibrateEmulateSample.ParameterDistributions
using CalibrateEmulateSample.DataContainers
using CalibrateEmulateSample.Observations

using CSV, Tables, DataFrames



################### FUNCTIONS ####################

function get_standardizing_factors(data::Array{FT, 2}) where {FT}
    # Input: data size: N_data x N_ensembles
    # Ensemble median of the data
    norm_factor = median(data, dims = 2)
    return norm_factor
end

function get_standardizing_factors(data::Array{FT, 1}) where {FT}
    # Input: data size: N_data*N_ensembles (splatted)
    # Ensemble median of the data
    norm_factor = median(data)
    return norm_factor
end

function comp2phys_transform(comp_params, constrain_method)
    physical_params = similar(comp_params);
    for j in 1:size(physical_params,1)
        if constrain_method == "log"
            physical_params[j,:] = inverse_log_transformation(comp_params[j,:]);
        else
            throw("Not implemented")
        end
    end
    return physical_params;
end

# Transform parameters from physical to computational space
function phys2comp_transform(physical_params, constrain_method)
    comp_params = similar(physical_params)
    for j in 1:size(physical_params,1)
        if constrain_method == "log"
            comp_params[j,:] = log_transformation(physical_params[j,:])
        else
            throw("not implemented")
        end
    end
    return comp_params;
end

# 1. log transformation function (physical --> computational)
log_transformation(x) = log.(x);
# inverse log transformation function (computational --> physical)
inverse_log_transformation(x) = exp.(x);


# Check if enough arguments are passed
if length(ARGS) < 2
    println("Please provide the required inputs.")
else
    yt_setting = ARGS[1]  
    script_dir = ARGS[2]

    # Print the received arguments
    println("Received yt_setting: $yt_setting")
    println("Received script_dir: $script_dir")
end


cases = [
    "GP" # diagonalize, train scalar GP, assume diag inputs
]

println("case: ", cases[1])
N_ens = 20
N_iter = 5 

# Should be loaded:
z0_true = 1   # reference 'truth' roughness length

rng_seed = 44011
rng = Random.MersenneTwister(rng_seed)

# loading relevant data
data_save_directory = joinpath(script_dir, "output_data/")
data_save_file = joinpath(data_save_directory, "calibrate_results.jld2")

if !isfile(data_save_file)
    throw(
        ErrorException(
            "data file $data_save_file not found. \n First run: \n > julia --project calibrate.jl \n and store results $data_save_file",
        ),
    )
end

inputs = load(data_save_file)["inputs"]
outputs = load(data_save_file)["outputs"]
priors = load(data_save_file)["priors"]
ekiobj = load(data_save_file)["eki"]
truth_sample = load(data_save_file)["truth_sample"]
truth_params_constrained = load(data_save_file)["truth_input_constrained"] #true parameters in constrained space
truth_params = phys2comp_transform(truth_params_constrained, "log")
Gy = ekiobj.obs_noise_cov

n_params = length(truth_params_constrained); # "input dim"
output_dim = size(Gy, 1);


################# Emulate: Gaussian Process Regression ###########################

# Emulate-sample settings
# choice of machine-learning tool in the emulation stage
if cases[1] == "GP"
    gppackage = Emulators.SKLJL()
    # gppackage = Emulators.GPJL()
    pred_type = Emulators.YType()
    mlt = GaussianProcess(
        gppackage;
        kernel = nothing, # using default squared exponential kernel
        prediction_type = pred_type,
        noise_learn = false)
end


# Standardize the output data
# Use median over all data since all data are the same type
truth_sample_norm = vcat(truth_sample...)
norm_factor = get_standardizing_factors(truth_sample_norm)
#norm_factor = vcat(norm_factor...)
norm_factor = fill(norm_factor, size(truth_sample))

# Make training points
comp_gp_inputs = phys2comp_transform(inputs[1:N_iter,:], "log")
gp_inputs = reshape(comp_gp_inputs', 1, :)

println(size(gp_inputs))
 
# quick fix for now
g_ens_csv = joinpath(data_save_directory, "g_ens.csv")
g_ens = CSV.File(g_ens_csv; header=false); g_ens = DataFrame(g_ens); g_ens = Matrix(g_ens); g_ens =g_ens';

gp_outputs = g_ens

println(size(gp_outputs))

input_output_pairs = PairedDataContainer(gp_inputs, gp_outputs, data_are_columns = true)
# Save data
@save joinpath(data_save_directory, "input_output_pairs.jld2") input_output_pairs

# # plot training points in constrained space
# if case == cases[mask[1]]
#     gr(dpi = 300, size = (400, 400))
#     inputs_unconstrained = get_inputs(input_output_pairs)
#     inputs_constrained = transform_unconstrained_to_constrained(priors, inputs_unconstrained)
#     p = plot(
#         title = "training points",
#         xlims = extrema(inputs_constrained[1, :]),
#         xaxis = "z0",
#     )
#     histogram!(p, inputs_constrained[1, :], color = :magenta, label = false)
#     inputs_test_unconstrained = get_inputs(input_output_pairs_test)
#     inputs_test_constrained = transform_unconstrained_to_constrained(priors, inputs_test_unconstrained)
#     histogram!(p, inputs_test_constrained[1, :], color = :black, label = false)
#     vline!(p, [truth_params_constrained[1]], linestyle = :dash, linecolor = :red, label = false)
#     savefig(p, joinpath(figure_save_directory, "training_points.pdf"))
#     savefig(p, joinpath(figure_save_directory, "training_points.png"))
# end

standardize = true               # this normalizes outputs (requires norm factor)
retained_svd_frac = 1.0           # retaining 100% of the svd
normalized = true                 # this normalizes inputs (parameters)
decorrelate = true                # decorrelate output using SVD (true default)

emulator = Emulator(
    mlt,
    input_output_pairs;
    obs_noise_cov = Gy,
    normalize_inputs = normalized,
    standardize_outputs = standardize,
    standardize_outputs_factors = norm_factor,
    retained_svd_frac = retained_svd_frac,
    decorrelate = decorrelate,
)
optimize_hyperparameters!(emulator)


# # Check how well the Gaussian Process regression predicts on the true parameters
# y_mean, y_var = Emulators.predict(emulator, reshape(truth_params, :, 1), transform_to_real = true)

# println("ML prediction on true parameters: ")
# println(vec(y_mean))
# println("true data: ")
# println(truth_sample) # what was used as truth
# println(" ML predicted standard deviation")
# println(sqrt.(diag(y_var[1], 0)))
# println("ML MSE (truth): ")
# println(mean((truth_sample - vec(y_mean)) .^ 2));
# 
# # Write out into csv file for later
# y_mean_csv = joinpath(data_save_directory, "gp_y_mean.csv")
# writedlm(y_mean_csv, y_mean, ',');
# y_var_csv = joinpath(data_save_directory, "gp_y_var.csv")
# writedlm(y_var_csv, y_var, ',');



#################  Sample: Markov Chain Monte Carlo ###########################
# initial values
u0 = vec(mean(get_inputs(input_output_pairs), dims = 2))
println("initial parameters: ", u0);

# First let's run a short chain to determine a good step size
mcmc = MCMCWrapper(RWMHSampling(), truth_sample, priors, emulator; init_params = u0)
new_step = optimize_stepsize(mcmc; init_stepsize = 0.1, N = 2000, discard_initial = 0)

# Now begin the actual MCMC
println("Begin MCMC - with step size ", new_step)
chain = MarkovChainMonteCarlo.sample(mcmc, 200_000; stepsize = new_step, discard_initial = 2_000)

posterior = MarkovChainMonteCarlo.get_posterior(mcmc, chain)

post_mean = mean(posterior)
post_cov = cov(posterior)
println("post_mean")
println(post_mean)
println("post_cov")
println(post_cov)
println("D util")
println(det(inv(post_cov)))
println(" ")

param_names = get_name(posterior)

# posterior_samples = vcat([get_distribution(posterior)[name] for name in get_name(posterior)]...) #samples are columns
# constrained_posterior_samples =
# mapslices(x -> transform_unconstrained_to_constrained(posterior, x), posterior_samples, dims = 1)


# Save data
save(
    joinpath(data_save_directory, "posterior.jld2"),
    "posterior",
    posterior,
    "input_output_pairs",
    input_output_pairs,
)

# For matlab postprocessing
comp_posterior_dist = zeros(1,200000)
comp_posterior_dist[1,:] = get_distribution(posterior)[param_names[1]][:]
posterior_dist = comp2phys_transform(comp_posterior_dist, "log");

posterior_file = joinpath(data_save_directory, "posterior_dist.csv")
writedlm(posterior_file, posterior_dist, ',');




