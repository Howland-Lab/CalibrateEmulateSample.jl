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


function comp2phys_transform(comp_params, prior_lower, prior_upper)
    physical_params = similar(comp_params);
    for j in 1:size(physical_params,1)
        if j == 1
            physical_params[j,:] = inverse_log_transformation(comp_params[j,:]);
        else
            physical_params[j,:] = pi_transformation(comp_params[j,:]);
            physical_params[j,:] = bounding_function(physical_params[j,:],prior_lower[j],prior_upper[j]);
        end
    end
    return physical_params;
end

# Transform parameters from physical to computational space
function phys2comp_transform(physical_params, prior_lower, prior_upper)
    comp_params = similar(physical_params)
    for j in 1:size(physical_params,1)
        if j == 1
            comp_params[j,:] = log_transformation(physical_params[j,:])
        else
            comp_params[j,:] = unbounding_function(physical_params[j,:],prior_lower[j],prior_upper[j]);
            comp_params[j,:] = inverse_pi_transformation(comp_params[j,:]);
        end
    end
    return comp_params;
end


# 3. probability integral transformation function (computational --> physical)
function pi_transformation(x)
    uniform_cdf_values = cdf.(Normal(0,1), x)
    uniform_distribution = Uniform(0,1)
    uniform_samples = quantile.(uniform_distribution, uniform_cdf_values)
    return uniform_samples
end

# Inverse probability integral transformation function (physical --> computational)
function inverse_pi_transformation(x)
    uniform_distribution = Uniform(0,1)
    uniform_cdf_values = cdf.(uniform_distribution,x)
    normal_distribution = Normal(0,1)
    normal_samples = quantile.(normal_distribution, uniform_cdf_values)
    return normal_samples
end

# Bounding function between [l,u]
function bounding_function(unbounded,l,u)
    bounded = (u - l)*unbounded .+ l;
end

function unbounding_function(bounded,l,u)
    unbounded = (bounded .- l) / (u - l);
end


# 1. log transformation function (physical --> computational)
log_transformation(x) = log.(x);
# inverse log transformation function (computational --> physical)
inverse_log_transformation(x) = exp.(x);


function success_run_filter(g_ens, params)
    # Delete failed forward model statistics and parameter values
    failed_cols = zeros(1,size(g_ens,2))
    for j in 1:size(g_ens,2)
        if any(isnan.(g_ens[:,j]))
            failed_cols[j] = 1;
        end
    end
    for j in reverse(1:size(g_ens,2))
        if failed_cols[:,j][1] == 1
            g_ens = g_ens[:, Not(j)];
            params = params[:, Not(j)];
        end
    end

    # Count number of fails
    count_of_fails = count(failed_cols -> failed_cols == 1, failed_cols)
    println("Number of fails: ", count_of_fails);

    return g_ens, params
end



# Check if enough arguments are passed
if length(ARGS) < 2
    println("Please provide the required inputs.")
else
    script_dir = ARGS[1]
    run = ARGS[2] 
end


if run == "z0zd"
    prior_lower = [0, 0.0]   # first index is dummy value
    prior_upper = [0, 8.0]
elseif run == "all"
    prior_lower = [0, 0, 0.03, 1.0, 1.15, 0.5, 0.5, 1.0]   # first index is dummy value
    prior_upper = [0, 8.0, 0.135, 2.0, 2.88, 1.5, 1.5, 2.0]
elseif run == "z0zdC1C2"
    prior_lower = [0, 0, 1.0, 1.15]   # first index is dummy value
    prior_upper = [0, 8.0, 2.0, 2.88] 
end

println("prior lower: ", prior_lower);
println("prior upper: ", prior_upper);


cases = [
    "GP" # diagonalize, train scalar GP, assume diag inputs
]

println("case: ", cases[1])
N_ens = 100
N_iter = 5

rng_seed = 44011
rng = Random.MersenneTwister(rng_seed)

# loading relevant data
data_save_directory = script_dir;
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
truth_sample_mean = load(data_save_file)["truth_sample_mean"]
truth_params_constrained = load(data_save_file)["truth_input_constrained"] #true parameters in constrained space
Gy = ekiobj.obs_noise_cov

n_params = length(truth_params_constrained); # "input dim"
output_dim = size(Gy, 1);



################# Emulate: Gaussian Process Regression ###########################

# Emulate-sample settings
# choice of machine-learning tool in the emulation stage
if cases[1] == "GP"
    gppackage = Emulators.SKLJL()
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



# quick fix for now
g_ens_csv = joinpath(data_save_directory, "g_ens.csv")
g_ens = CSV.File(g_ens_csv; header=false); g_ens = DataFrame(g_ens); g_ens = Matrix(g_ens); g_ens = g_ens';


params_csv = joinpath(data_save_directory, "params.csv")
params = CSV.File(params_csv; header=false); params = DataFrame(params); params = Matrix(params); params = params';
comp_params = phys2comp_transform(params, prior_lower, prior_upper);
inputs = comp_params[:,1:500]

# Filter out the columns with NaNs from both `outputs` and `inputs`
gp_outputs, gp_inputs = success_run_filter(g_ens, inputs)

# Make training points
println("inputs size: ", size(gp_inputs));
println("outputs size: ", size(gp_outputs));

input_output_pairs = PairedDataContainer(gp_inputs, gp_outputs, data_are_columns = true)
# Save data
@save joinpath(data_save_directory, "input_output_pairs.jld2") input_output_pairs


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
comp_posterior_dist = zeros(n_params,200000)
comp_posterior_dist[1,:] = get_distribution(posterior)[param_names[1]][:]
comp_posterior_dist[2,:] = get_distribution(posterior)[param_names[2]][:]
posterior_dist = comp2phys_transform(comp_posterior_dist, prior_lower, prior_upper);

posterior_file = joinpath(data_save_directory, "posterior_dist.csv")
writedlm(posterior_file, posterior_dist, ',');



