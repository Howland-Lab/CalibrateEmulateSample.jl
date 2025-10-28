# calibrate.jl — Calibration step in CES
# author: Ethan Shin


# Include forward model and utils
include(joinpath(dirname(@__DIR__), "GModel", "GModel.jl"))
include(joinpath(dirname(@__DIR__), "utils", "utils.jl"))
using .utils: construct_priors, problem_settings, build_observations, comp2phys
using .utils: scale_statistics, failed_runs, success_run_filter, shrinkage_cov, interpolate_zaxis

# Import modules
using Distributions
using LinearAlgebra
using StatsPlots
using Plots
using Random
using JLD2
using Statistics
using CSV, Tables, DataFrames
using NPZ, MAT
using Interpolations
using DelimitedFiles
using CalibrateEmulateSample
using CalibrateEmulateSample.Emulators
using CalibrateEmulateSample.MarkovChainMonteCarlo
using CalibrateEmulateSample.DataContainers
const EKP = CalibrateEmulateSample.EnsembleKalmanProcesses
const PD = EKP.ParameterDistributions


################################################## Input settings ################################################################

# Check if enough arguments are passed
if length(ARGS) < 8
    println("Please provide the required inputs.")
else
    # Input arguments
    casename = ARGS[1]  
    end_time = parse(Int, ARGS[2])
    timeavg = parse(Int, ARGS[3])
    script_dir = ARGS[4]
    statistics = parse(Int, ARGS[5])
    geo_U = parse(Int, ARGS[6])
    z0 = parse(Float64, ARGS[7])
    turb_model = ARGS[8]

    # Print the received arguments
    println("Received case name: $casename")
    println("Received end time: $end_time")
    println("Received timeavg: $timeavg")
    println("Received script directory path: $script_dir")
    println("Received statistics number: $statistics")
    println("Received geostrophic wind speed: $geo_U")
    println("Received roughness length: $z0")
    println("Received turbulence model: $turb_model")
end


# Fix random seed
rng_seed = 413
rng = Random.MersenneTwister(rng_seed)

# Start experiment in new folder
experiment_num = 1;
while true
    if isdir(joinpath(script_dir, string(experiment_num)))
	    global experiment_num = experiment_num + 1;
    else
	    mkdir(joinpath(script_dir, string(experiment_num)))
	    break;
    end
end
output_dir = joinpath(script_dir, string(experiment_num))


########################################### Construct parameter priors for STD k-e ##################################################

params_true = [0.09, 1.44, 1.92, 1.0, 1.0, 1.3]
param_names = ["C_mu", "C_1", "C_2", "C_3", "sig_k", "sig_e"]
n_param = length(param_names)
params_true = reshape(params_true, (n_param, 1))

prior_C_mu = Dict(
    "distribution" => PD.Parameterized(Normal(0.09, 0.05*0.09)),
    "constraint" => PD.no_constraint(),
    "name" => param_names[1]
)
prior_C_1 = Dict(
    "distribution" => PD.Parameterized(Normal(1.44, 0.05*1.44)),
    "constraint" => PD.no_constraint(),
    "name" => param_names[2]
)
prior_C_2 = Dict(
    "distribution" => PD.Parameterized(Normal(1.92, 0.05*1.92)),
    "constraint" => PD.no_constraint(),
    "name" => param_names[3]
)
prior_C_3 = Dict(
    "distribution" => PD.Parameterized(Normal(1.0, 0.05*1.0)),
    "constraint" => PD.no_constraint(),
    "name" => param_names[4]
)
prior_sig_k = Dict(
    "distribution" => PD.Parameterized(Normal(1.0, 0.05*1.0)),
    "constraint" => PD.no_constraint(),
    "name" => param_names[5]
)
prior_sig_e = Dict(
    "distribution" => PD.Parameterized(Normal(1.3, 0.05*1.3)),
    "constraint" => PD.no_constraint(),
    "name" => param_names[6]
)

priors = PD.ParameterDistribution([prior_C_mu, prior_C_1, prior_C_2, prior_C_3, prior_sig_k, prior_sig_e])


points_mask = collect(1:360)
N_ens = 100

# Construct prior distribution for parameters and run
synthetic_params = EKP.construct_initial_ensemble(rng, priors, N_ens)
scm_settings = GModel.Settings(end_time, casename, statistics, timeavg, points_mask, geo_U, z0, turb_model)
g_ens = GModel.run_G_ensemble(synthetic_params, scm_settings, experiment_num, statistics)

# Save files
save_synthetic_params = string(output_dir, "/synthetic_params.csv")
CSV.write(save_synthetic_params,  Tables.table(synthetic_params), writeheader=false)
save_synthetic_output = string(output_dir, "/synthetic_yt.csv")
CSV.write(save_synthetic_output,  Tables.table(g_ens), writeheader=false)
