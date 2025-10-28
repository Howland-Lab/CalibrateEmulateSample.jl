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
if length(ARGS) < 12
    println("Please provide the required inputs.")
else
    # Input arguments
    casename = ARGS[1]  
    end_time = parse(Int, ARGS[2])
    timeavg = parse(Int, ARGS[3])
    script_dir = ARGS[4]
    statistics = parse(Int, ARGS[5])
    regime = ARGS[6]
    geo_U = parse(Int, ARGS[7])
    z0 = parse(Float64, ARGS[8])
    turb_model = ARGS[9]
    yt_setting = ARGS[10]
    sample_num = parse(Int, ARGS[11])
    reg = ARGS[12]

    # Print the received arguments
    println("Received case name: $casename")
    println("Received end time: $end_time")
    println("Received timeavg: $timeavg")
    println("Received script directory path: $script_dir")
    println("Received statistics number: $statistics")
    println("Received height regime: $regime")
    println("Received geostrophic wind speed: $geo_U")
    println("Received roughness length: $z0")
    println("Received turbulence model: $turb_model")
    println("Received yt_setting: $yt_setting");
    println("Received sample_num: $sample_num");
    println("Received regularization: $reg");
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


################################################### Construct parameter priors ###################################################

params_true, param_names, n_param, prior_lower, prior_upper, priors = construct_priors(turb_model)

# Transformation settings
constrain = true   # Need to transform parameters from computational to physical?
constrain_method = 3   # 1. log transformation, 2. logitnormal transformation, 3. inverse probability integral transformation


################################################### Problem-specific settings ####################################################

ps = problem_settings(statistics, casename, regime)
points_mask        = collect(ps.points_mask)
zCell              = collect(ps.zCell)
interp_points_mask = collect(ps.interp_z)

# Settings for the forward model 
scm_settings = GModel.Settings(end_time, casename, statistics, timeavg, points_mask, geo_U, z0, turb_model)


####################################################### Construct observation #####################################################

data_path = "../"
yt, mins, maxs = build_observations(data_path, yt_setting;
    statistics=statistics,
    points_mask=points_mask,
    casename=casename,
    interp_z=interp_points_mask,
    zCell = zCell,
    sample_num=sample_num,
    nugget=0.0)

# Treating sample covariance matrix
if reg == "none"
    # Raw sample covariance matrix
    Gy = cov(yt, dims=2)
elseif reg == "ledoitwolf"
    # Ledoit-Wolf shrinkage estimator
    Gy = shrinkage_cov(yt)
end
println("Size: ", size(Gy))
println("Rank: ", rank(Gy))
if rank(Gy) < size(Gy,1)
    # Break if sample covariance matrix is rank-deficient
    @warn "Rank deficient."      
else
    println("Full rank")
end
println("Condition number: ", cond(Gy))

# Construct observation object
truth = EKP.Observation(Dict("samples" => yt[:,sample_num], "covariances" => Gy, "names" => "sample"))

# Save locally for debugging
save_yt = string(output_dir, "/yt.csv");
save_Gy = string(output_dir, "/Gy.csv");
CSV.write(save_yt,  Tables.table(yt), writeheader=false);
CSV.write(save_Gy,  Tables.table(Gy), writeheader=false);


########################################### Calibrate: Ensemble Kalman Inversion #############################################

# EKI settings
N_ens = 100;             # number of ensemble members
N_iter = 5;              # number of EKI iterations

# Construct prior distribution for parameters
initial_params = EKP.construct_initial_ensemble(rng, priors, N_ens)
println("initial params: ", initial_params);

# Define Ensemble Kalman Process (Note: currently, uses most basic settings)
ekiobj = EKP.EnsembleKalmanProcess(
    initial_params,
    truth,
    EKP.Inversion(),
    scheduler = EKP.DefaultScheduler(),
    localization_method = EKP.NoLocalization(),
    accelerator = EKP.DefaultAccelerator(),
    verbose = false,
    failure_handler_method = EKP.SampleSuccGauss()
)


# Helpers
@inline function scale_block!(A::AbstractMatrix, r::UnitRange, mn, mx)
    @views A[r, :] .= scale_statistics(A[r, :], mn, mx)
    return A
end

@inline function label_nonpositive!(A::AbstractMatrix, top::Int)
    @views bad = vec(all(A[1:top, :] .<= 0, dims=1))
    @views A[1:top, bad] .= NaN
    return A
end

# Predefine paths
params_csv        = string(output_dir, "/params.csv")
data_err_csv      = string(output_dir, "/data_err.csv")
g_ens_csv         = string(output_dir, "/g_ens.csv")
g_ens_debug_csv   = string(output_dir, "/g_ens_debug.csv")

# Precompute sizes
err        = zeros(N_iter)
final_iter = N_iter
top    = length(interp_points_mask)              
ncol_p = size(EKP.get_u_final(ekiobj), 2)   

# Ranges for scaling variables
range_U = 1:top
range_dir = top+1:2top
range_k = 2top+1:3top

# Initialize containers
phys_per_iter = Matrix{Float64}[]
gens_per_iter = Matrix{Float64}[]

println("EKI inversion error:")
for i in 1:N_iter
    # Current physical parameters (n_param × N_ens)
    phys = constrain ?
        comp2phys(EKP.get_u_final(ekiobj), constrain_method, prior_lower, prior_upper) :
        EKP.get_u_final(ekiobj)
    println("physical_params: ", phys)
    CSV.write(params_csv, Tables.table(phys'), writeheader=false, append=true)

    # Run ensemble of forward models
    println("Now running ensemble of forward model runs.");
    g_ens = GModel.run_G_ensemble(phys, scm_settings, experiment_num, statistics)
    # Save model outputs before filtering failed runs
    CSV.write(g_ens_debug_csv, Tables.table(g_ens'), writeheader=false, append=true);

    # Interpolate to points mask
    g_ens = interpolate_zaxis(g_ens, points_mask, zCell, interp_points_mask)

    # Filter failed forward model runs
    label_nonpositive!(g_ens, top)

    #  Scale efficiently per statistics
    @views begin
        if statistics == 1
            scale_block!(g_ens, range_U, mins[1], maxs[1])
        elseif statistics == 2
            scale_block!(g_ens, range_U,   mins[1], maxs[1])
            scale_block!(g_ens, range_dir, mins[2], maxs[2])
        elseif statistics == 3
            scale_block!(g_ens, range_U,   mins[1], maxs[1])
            scale_block!(g_ens, range_dir, mins[2], maxs[2])
            scale_block!(g_ens, range_k,   mins[3], maxs[3])
        end
    end

    # Debug dump
    CSV.write(g_ens_debug_csv, Tables.table(g_ens'), writeheader=false, append=true)

    # # Failure checks on scaled values
    @views begin
        all_zero  = vec(all(g_ens .== 0, dims=1))                 
        bad_nan   = vec(any(isnan.(g_ens), dims=1))                   
        bad_inf   = vec(any(isinf.(g_ens), dims=1))                  
        bad_range = vec(any((g_ens .>= 5.0) .| (g_ens .<= -1.0), dims=1))
    end

    # Parameters at bounds
    at_upper = vec(any(phys .== prior_upper, dims=1))
    at_lower = vec(any(phys .== prior_lower, dims=1))

    # Handle failed runs
    if any(all_zero) || any(bad_nan) || any(bad_inf) || any(bad_range) || any(at_upper) || any(at_lower)
        g_ens = failed_runs(g_ens, phys, prior_lower, prior_upper)
    end

    # Collect inputs and outputs
    push!(phys_per_iter, phys)
    push!(gens_per_iter, g_ens)

    # Success filter & data misfit
    g_ens_success, _ = success_run_filter(g_ens, phys)
    μ_pred = mean(g_ens_success, dims=2)
    data_err = mean((EKP.get_obs(truth) .- μ_pred).^2)
    err[i] = data_err

    # Append CSVs
    CSV.write(data_err_csv, Tables.table([data_err]),  writeheader=false, append=true)
    CSV.write(g_ens_csv,    Tables.table(g_ens'),      writeheader=false, append=true)

    # EKI update
    terminated = EKP.update_ensemble!(ekiobj, g_ens)
    if terminated !== nothing
        final_iter = i - 1            # previous iteration was the last update
        break
    end
end
# Collect all inputs and outputs
inputs  = hcat(phys_per_iter...)
outputs = hcat(gens_per_iter...)


# EKI results
println("True parameters: ")
println(params_true)

# Constrain parameters from computational to physical space for saving
eki_params = constrain ?
    comp2phys(EKP.get_u_final(ekiobj), constrain_method, prior_lower, prior_upper) :
    EKP.get_u_final(ekiobj)
println("Final EKI mean", mean(eki_params, dims = 2));
CSV.write(params_csv, Tables.table(eki_params'), writeheader=false, append=true)

# Save calibration results
save(
    joinpath(output_dir, "calibrate_results.jld2"),
    "inputs",
    inputs,
    "outputs",
    outputs,
    "eki_params",
    eki_params,
    "truth_sample",
    EKP.get_obs(truth),
    "truth_input_constrained",
    params_true,
)
