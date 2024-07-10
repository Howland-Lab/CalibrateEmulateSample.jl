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
if length(ARGS) < 3
    println("Please provide the required inputs.");
else
    # Convert the first argument to an integer
    iteration = parse(Int, ARGS[1])
    N_ens = parse(Int, ARGS[2])
    avg = ARGS[3]

    # Print the received arguments
    println("Received iteration number: $iteration");
    println("Received ensemble size number: $N_ens");
    println("Received averaging parameter: $avg");
end


# A few redundant things that need to be loaded
rng_seed = 4137
rng = Random.MersenneTwister(rng_seed)
# Output figure save directory
homedir = pwd()
println(homedir)
data_save_directory = homedir * "/output_data/"
constrain = true;   # Will you transform parameters from computational to physical
constrain_method = "log";
N_iter = 5;             # number of EKI iterations
final_iter = [N_iter];
points_mask = collect(1:32);   # for now
statistics = 1;   # Just wind speed for now


########################################### Calibrate: Ensemble Kalman Inversion #############################################

# Load my files and my g_ens 
data_file = joinpath(data_save_directory, "calibrate_$(iteration-1).jld2")
ekiobj = load(data_file)["eki"]
priors = load(data_file)["priors"]
truth_sample = load(data_file)["truth_sample"]
truth_params_constrained = load(data_file)["truth_input_constrained"] #true parameters in constrained space

# Postprocess g_ens (written by Miles Chan)
g_ens = gmodel.postprocess(N_ens, statistics, iteration, avg);
g_ens_csv = "g_ens_temp.csv";   # this is postprocessed file
g_ens = CSV.File(g_ens_csv; header=false); g_ens = DataFrame(g_ens); g_ens = Matrix(g_ens); g_ens =g_ens'; 

println("g_ens: ", g_ens);

# # Deal with forward model runs we consider 'failed' runs (add conditions as we see fit)
# if any(all(g_ens .== 0, dims=1)) || any(isnan.(g_ens)) || any(.!isfinite.(g_ens))
#     # Change failed runs into NaNs. The SampleSuccGauss method will handle these.
#     g_ens = failed_runs(g_ens, get_u_final(ekiobj));
# end

# # Store data for debugging purposes
g_ens_success, params_success = success_run_filter(g_ens, get_u_final(ekiobj)); 

println("size g_ens ", size(g_ens_success));
println("size params ", size(params_success));
println("size truth_sample ", size(truth_sample));

data_err = mean((truth_sample - mean(g_ens_success, dims=2)).^2);

if constrain == true
    physical_params_success = comp2phys_transform(params_success, constrain_method);
else
    physical_params_success = params_success;
end

# To save QoIs (filtered input and output)
data_err_csv = string(data_save_directory, "/data_err.csv");
g_ens_csv = string(data_save_directory, "/g_ens.csv");
params_csv = string(data_save_directory, "/params.csv");
CSV.write(data_err_csv, Tables.table([data_err]), writeheader=false, append=true); 
CSV.write(g_ens_csv, Tables.table(g_ens'), writeheader=false, append=true);           # what will be used for EKI
CSV.write(params_csv, Tables.table(physical_params_success), writeheader=false, append=true); # what will be used for EKI

# Update EKI object
terminated = EnsembleKalmanProcesses.update_ensemble!(ekiobj, g_ens);
if !isnothing(terminated)
    final_iter = iteration # final update was previous iteration
end

println("Parameters before update: ", params_success);
println("Physical parameters before update: ", physical_params_success);

println("Parameters after update: ", get_u_final(ekiobj));

if iteration < N_iter
    if constrain == true
        new_physical_params = comp2phys_transform(get_u_final(ekiobj), constrain_method);
    else
        new_physical_params = get_u_final(ekiobj);
    end

    println("Physical parameters after update: ", new_physical_params);

    # Run ensemble of forward models (will need to change with Jianyu)
    println("Now running ensemble of forward model runs.");
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
    points_mask = collect(1:32);
    gmodel_settings = gmodel.Settings(statistics, scratch_dir, points_mask, avg);
    g_ens = gmodel.run_G_ensemble(new_physical_params, gmodel_settings, iteration+1);
        
    save(
        joinpath(data_save_directory, "calibrate_$(iteration).jld2"),
        "priors",
        priors,
        "eki",
        ekiobj,   # the updated one
        "truth_sample",
        truth_sample,
        "truth_input_constrained",
        truth_params_constrained, #constrained here, as these are in a physically constrained space (unlike the u inputs),
        )
else
    N_iter = final_iter #in case it terminated early 
    
    # EKI results: Has the ensemble collapsed toward the truth?
    println("True parameters: ")
    println(truth_params_constrained)
    println("\nEKI results:")
    println(get_u_final(ekiobj))

    # the final ensemble of updated parameters
    if constrain == true
        new_physical_params = comp2phys_transform(get_u_final(ekiobj), constrain_method);
    else
        new_physical_params = get_u_final(ekiobj);
    end
    CSV.write(params_csv, Tables.table(new_physical_params), writeheader=false, append=true); # what will be used for EKI

    # Load my final input output pairs from csv files
    params_csv = joinpath(data_save_directory, "params.csv");
    params = CSV.File(params_csv; header=false); params = DataFrame(params); params = Matrix(params); # params = params';
    g_ens_csv = joinpath(data_save_directory, "g_ens.csv");
    g_ens = CSV.File(g_ens_csv; header=false); g_ens = DataFrame(g_ens); g_ens = Matrix(g_ens); # g_ens = g_ens';
    data_err_csv = joinpath(data_save_directory, "data_err.csv");
    data_err = CSV.File(data_err_csv; header=false); data_err = DataFrame(data_err); data_err = Matrix(data_err); # g_ens = g_ens';

    save(
        joinpath(data_save_directory, "calibrate_results.jld2"),
        "inputs",   # in computational space (this will also contain a N_iter+1 ensemble that is the final ensemble after update)
        params,
        "outputs",
        g_ens,
        "priors",
        priors,
        "eki",
        ekiobj,
        "truth_sample",
        truth_sample,
        "truth_input_constrained",
        truth_params_constrained, #constrained here, as these are in a physically constrained space (unlike the u inputs),
        "output_error",
        data_err,
        )
end








