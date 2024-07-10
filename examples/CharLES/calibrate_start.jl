# Include forward model script
include(joinpath(@__DIR__, "gmodel_charles.jl"))

# Import modules
using Distributions  # probability distributions and associated functions
using LinearAlgebra
using StatsPlots
using Plots
using Random
using JLD2
# Import modules
using Distributions  # probability distributions and associated functions
using LinearAlgebra
using StatsPlots
using Plots
using Random
using JLD2
# CES
using CalibrateEmulateSample
using CalibrateEmulateSample.Emulators
using CalibrateEmulateSample.MarkovChainMonteCarlo
using CalibrateEmulateSample.Utilities
using CalibrateEmulateSample.EnsembleKalmanProcesses
using CalibrateEmulateSample.ParameterDistributions 
using CalibrateEmulateSample.DataContainers
using CalibrateEmulateSample.Observations
const EKP = CalibrateEmulateSample.EnsembleKalmanProcesses
const PD = EKP.ParameterDistributions
# Often useful modules
using Statistics
using Printf
using CSV, Tables, DataFrames



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

function failed_runs(g_ens, physical_params)
    # Loop over each column to check the conditions (we can add more relevant filtering conditions for failed runs)
    for j in 1:size(g_ens, 2)
        if all(g_ens[:,j] .== 0) || any(isnan.(g_ens[:,j])) || any(.!isfinite.(g_ens[:,j]))
           g_ens[:,j] .= NaN;
        end
    end
    return g_ens
end


function success_run_filter(g_ens, physical_params)
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
            physical_params = physical_params[:, Not(j)];
        end
    end

    # Count number of fails
    count_of_fails = count(failed_cols -> failed_cols == 1, failed_cols)
    println("Number of fails: ", count_of_fails);
    return g_ens, physical_params;
end


# 1. log transformation function (physical --> computational)
log_transformation(x) = log.(x);
# inverse log transformation function (computational --> physical)
inverse_log_transformation(x) = exp.(x);


# Check if enough arguments are passed
if length(ARGS) < 2
    println("Please provide the required inputs.")
else
    # Convert the first argument to an integer
    avg = ARGS[1]  # Directly capture the second argument as a string
    yt_setting = ARGS[2]

    # Print the received arguments
    println("Received averaging parameter: $avg")
    println("Received yt setting: $yt_setting")
end

rng_seed = 4137
rng = Random.MersenneTwister(rng_seed)
# Output save directory
homedir = pwd()
println(homedir)
data_save_directory = homedir * "/output_data/"
if !isdir(data_save_directory)
    mkdir(data_save_directory)
end

###
###  Define the (true) parameters
###
# Define the parameters that we want to learn
z0_true = 1;      # prescribed roughness length for perfect model 
params_true = [z0_true];                             # Vector{Float64}
param_names = ["z0"];                                # Vector{String}
n_param = length(param_names);
params_true = reshape(params_true, (n_param, 1));    # Matrix{Float64} 

###
### Define the parameter prior
###
prior_z0 = Dict(
    "distribution" => Parameterized(Normal(0, 1)),
    "constraint" => no_constraint(),
    "name" => param_names[1],
)

# ParameterDistribution takes in dictionaries as input
priors = ParameterDistribution([prior_z0])

###
###  Define the data from which we want to learn the parameters
###
data_names = ["y0"]


### 
### Settings for mid-fidelity model (refered to as gmodel)
### e.g., time averaging window, start time, end time, etc.
###
statistics = 1;
if avg == "t" 
    scratch_dir = "/home/ctrsp-2024/mjchan/charles_data_t";      # Has to be in the PadeOps directory
elseif avg == "txz"
    scratch_dir = "/home/ctrsp-2024/mjchan/charles_data_txz"; 
elseif avg == "tz"
    scratch_dir = "/home/ctrsp-2024/mjchan/charles_data_tz";
elseif avg == "txz2"
    scratch_dir = "/home/ctrsp-2024/mjchan/charles_data_txz2";
end
points_mask = collect(1:20);
gmodel_settings = gmodel.Settings(statistics, scratch_dir, points_mask, avg);

###
### Choose observational data and create internal variability covariance matrix
###
data_path = @__DIR__
if yt_setting == "perfect"
    # If running in perfect model setting, observational data is generated by the low/mid fidelity model in the CES framework
    # Example script below.
    if avg == "t"
        yt_csv = joinpath(data_path, "yt_data/perfect_model_U_t_avg.csv");
    elseif avg == "txz"
        yt_csv = joinpath(data_path, "yt_data/perfect_model_U_txz_avg.csv");
    elseif avg == "tz"
        yt_csv = joinpath(data_path, "yt_data/perfect_model_U_tz_avg.csv");
    elseif avg == "txz2"
        yt_csv = joinpath(data_path, "yt_data/perfect_model_U_txz2_avg.csv");
    end
    yt = CSV.File(yt_csv; header=false); yt = DataFrame(yt); yt = Matrix(yt);
elseif yt_setting == "imperfect"
    # If running in imperfect model setting, observational data is the geometry-resolving CharLES data.
    # Example script below.
    if avg == "t"
        yt_csv = joinpath(data_path, "yt_data/imperfect_model_U_t_avg.csv");
    elseif avg == "txz"
        yt_csv = joinpath(data_path, "yt_data/imperfect_model_U_txz_avg.csv");
    elseif avg == "tz"
        yt_csv = joinpath(data_path, "yt_data/imperfect_model_U_tz_avg.csv");	
    elseif avg == "txz2"
        yt_csv = joinpath(data_path, "yt_data/imperfect_model_U_txz2_avg.csv");
    end
    yt = CSV.File(yt_csv; header=false); yt = DataFrame(yt); yt = Matrix(yt);
end

# Covariance: check for rank deficiency
Gy = cov(yt, dims = 2)
println(size(Gy), " ", rank(Gy))
if rank(Gy) < size(Gy,1)
    @warn "Rank deficient."
else
    println("Full rank")
end 

# Save locally for future postprocessing
save_yt = string(data_save_directory, "/yt.csv");
save_Gy = string(data_save_directory, "/Gy.csv");
CSV.write(save_yt,  Tables.table(yt), writeheader=false);
CSV.write(save_Gy,  Tables.table(Gy), writeheader=false);

# Store data in Observations.Observation object
truth = Observations.Observation(yt, Gy, data_names)
truth_sample = truth.mean


# Transformation settings
# EKI requires a normal distribution for the parameter ensemble, while the actual physical parameters going into the forward model may not necessarily be a normal distribution. So, we transform between computational (EKI) and physical (model) space.
constrain = true;   # Will you transform parameters from computational to physical
constrain_method = "log";


########################################### Calibrate: Ensemble Kalman Inversion #############################################
# EKI settings
N_ens = 20;             # number of ensemble members
N_iter = 5;             # number of EKI iterations

gmodel.make_folders(scratch_dir, N_ens, N_iter);

# initial parameters: N_params x N_ens

initial_params = EKP.construct_initial_ensemble(rng, priors, N_ens)

ekiobj = EKP.EnsembleKalmanProcess(
    initial_params,
    truth_sample,
    truth.obs_noise_cov,
    EKP.Inversion(),
    scheduler = EKP.DataMisfitController(),
    verbose = true,
    failure_handler_method=SampleSuccGauss()
)

# First iteration of EKI
if constrain == true
    physical_params = comp2phys_transform(get_u_final(ekiobj), constrain_method);
else
    physical_params = get_u_final(ekiobj);
end

# Run ensemble of forward models
println("Now running ensemble of forward model runs.");
g_ens = gmodel.run_G_ensemble(physical_params, gmodel_settings, 1);

save(
    joinpath(data_save_directory, "calibrate_0.jld2"),
    "priors",
    priors,
    "eki",
    ekiobj,
    "truth_sample",
    truth_sample,
    "truth_input_constrained",
    params_true, #constrained here, as these are in a physically constrained space (unlike the u inputs),
    )

