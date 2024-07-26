# Template of calibrate phase, based off of Lorenz 96 example in CalibrateEmulateSample.jl

# Include forward model script
include(joinpath(@__DIR__, "gmodel", "gmodel_z0zdC1C2.jl"))

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



function failed_runs(g_ens, physical_params)
    # Loop over each column to check the conditions (we can add more relevant filtering conditions for failed runs)
    for j in 1:size(g_ens, 2)
        if all(g_ens[:,j] .== 0) || any(isnan.(g_ens[:,j])) || any(.!isfinite.(g_ens[:,j])) || any(abs.(g_ens[:,j]) .> 20)
           g_ens[:,j] .= NaN;
        end
    end
    return g_ens
end


function success_run_filter(g_ens)
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
        end
    end

    # Count number of fails
    count_of_fails = count(failed_cols -> failed_cols == 1, failed_cols)
    println("Number of fails: ", count_of_fails);
    return g_ens
end

# 1. log transformation function (physical --> computational)
log_transformation(x) = log.(x);
# inverse log transformation function (computational --> physical)
inverse_log_transformation(x) = exp.(x);
    

# Check if enough arguments are passed
if length(ARGS) < 8
    println("Please provide the required inputs.")
else
    # Convert the first argument to an integer
    casename = ARGS[1]  # Directly capture the second argument as a string
    end_time = parse(Int, ARGS[2])
    timeavg = parse(Float64, ARGS[3])
    script_dir = ARGS[4]  # Third argument is the directory path
    statistics = parse(Int, ARGS[5])
    turb_model = ARGS[6]
    avg = ARGS[7]
    yt_setting = ARGS[8]

    # Print the received arguments
    println("Received case name: $casename")
    println("Received end time: $end_time")
    println("Received time avg: $timeavg");
    println("Received script directory path: $script_dir")
    println("Received statistics number: $statistics")
    println("Received turbulence model: $turb_model")
    println("Received avg: $avg");
    println("Received yt_setting: $yt_setting");
end


function main()
    rng_seed = 4137
    rng = Random.MersenneTwister(rng_seed)
    # Output figure save directory
    homedir = pwd()
    println(homedir)

    if avg == "t"
        data_save_directory = homedir * "/run_t_z0zdC1C2/"
    elseif avg == "tz"
        data_save_directory = homedir * "/run_tz_z0zdC1C2/"
    elseif avg == "txz"
        data_save_directory = homedir * "/run_txz_z0zdC1C2/"
    end

    if !isdir(data_save_directory)
        mkdir(data_save_directory)
    end


    ###
    ###  Define the (true) parameters
    ###
    # Define the parameters that we want to learn
    z0_true = 1000 * 6.8E-5;      # Roughness length
    zd_true = 1;
    C_1_true = 1.44; 
    C_2_true = 1.92;     

    params_true = [z0_true, zd_true, C_1_true, C_2_true];                             # Vector{Float64}
    param_names = ["z0", "zd", "C_1", "C_2"];                                # Vector{String}
    n_param = length(param_names);
    params_true = reshape(params_true, (n_param, 1));    # Matrix{Float64} 

    # Define the parameter priors (physical prior space)
    prior_lower = [0, 0, 1.0, 1.15]   # first index is dummy value
    prior_upper = [0, 8.0, 2.0, 2.88]
    println("prior lower: ", prior_lower);
    println("prior upper: ", prior_upper);

    ###
    ### Define the parameter prior
    ###
    prior_z0 = Dict(
        "distribution" => Parameterized(Normal(0, 1)),
        "constraint" => no_constraint(),
        "name" => param_names[1],
    )

    prior_zd = Dict(
        "distribution" => Parameterized(Normal(0, 1)),
        "constraint" => no_constraint(),
        "name" => param_names[2],
    )

    prior_C_1 = Dict(
        "distribution" => Parameterized(Normal(0, 1)),
        "constraint" => no_constraint(),
        "name" => param_names[3],
    )
    prior_C_2 = Dict(
        "distribution" => Parameterized(Normal(0, 1)),
        "constraint" => no_constraint(),
        "name" => param_names[4],
    )
    # ParameterDistribution takes in dictionaries as input
    # priors = PD([prior_z0])
    priors = ParameterDistribution([prior_z0, prior_zd, prior_C_1, prior_C_2])

    ###
    ###  Define the data from which we want to learn the parameters
    ###
    data_names = ["y0"]

    
    ### 
    ### Settings for mid-fidelity model (refered to as gmodel)
    ### e.g., time averaging window, start time, end time, etc.
    ###
    # Settings for the forward model (specific to SCM here)
    points_mask = collect(2:9);
    gmodel_settings = gmodel.Settings(end_time, casename, statistics, timeavg, points_mask, turb_model, avg);


    ###
    ### Create the internal variability covariance matrix
    ###
    data_path = @__DIR__
    if yt_setting == "perfect"
        # If running in perfect model setting, observadtional data is generated by the low/mid fidelity model in the CES framework
        yt_csv = joinpath(data_path, "les_data/perfect_model_data.csv");
        yt = CSV.File(yt_csv; header=false); yt = DataFrame(yt); yt = Matrix(yt);
    elseif yt_setting == "imperfect"
        # If running in imperfect model setting, observational data is the geometry-resolving CharLES data.
        if avg == "t"
            yt_csv = joinpath(data_path, "les_data/hf_U_t_interp_8pt.csv");
        elseif avg == "tz"
            yt_csv = joinpath(data_path, "les_data/hf_U_tz_26_interp_8pt.csv");
        elseif avg == "txz"
            yt_csv = joinpath(data_path, "les_data/hf_U_txz_interp_8pt.csv");
        end

        yt = CSV.File(yt_csv; header=false); yt = DataFrame(yt); yt = Matrix(yt);
        yt=yt/maximum(mean(yt,dims=2));
    end


    # Covariance: check for rank deficiency
    Γy = cov(yt, dims = 2)
    println(size(Γy), " ", rank(Γy))
    if rank(Γy) < size(Γy,1)
        @warn "Rank deficient."
    else
        println("Full rank")
    end  

    # Save locally for future postprocessing
    save_yt = string(data_save_directory, "/yt.csv");
    save_Γy = string(data_save_directory, "/Γy.csv");
    CSV.write(save_yt,  Tables.table(yt), writeheader=false);
    CSV.write(save_Γy,  Tables.table(Γy), writeheader=false);

    # Store data in Observations.Observation object
    truth = Observations.Observation(yt, Γy, data_names)
    truth_sample = yt[:,end]


    # Transformation settings
    # EKI requires a normal distribution for the parameter ensemble, while the actual physical parameters going into the forward model may not necessarily be a normal distribution. So, we transform between computational (EKI) and physical (model) space.
    constrain = true;   # Will you transform parameters from computational to physical
    
    ########################################### Calibrate: Ensemble Kalman Inversion #############################################
    # EKI settings
    N_ens = 100;             # number of ensemble members
    N_iter = 5;             # number of EKI iterations
    
    # initial parameters: N_params x N_ens
    initial_params = EKP.construct_initial_ensemble(rng, priors, N_ens)

    println("initial params: ", initial_params);

    ekiobj = EKP.EnsembleKalmanProcess(
        initial_params,
        truth_sample,
        truth.obs_noise_cov,
        EKP.Inversion(),
        failure_handler_method=SampleSuccGauss()
    )

    # To save QoIs
    params_csv = string(data_save_directory, "/params.csv");
    data_err_csv = string(data_save_directory, "/data_err.csv");
    g_ens_csv = string(data_save_directory, "/g_ens.csv");
    
    # EKI iterations
    println("EKP inversion error:")
    err = zeros(N_iter)
    final_iter = [N_iter]
    for i in 1:N_iter
        if constrain == true
            physical_params = comp2phys_transform(get_u_final(ekiobj), prior_lower, prior_upper);
        else
            physical_params = get_u_final(ekiobj);
        end
       
	println("physical_params: ", physical_params);

        # Run ensemble of forward models
        println("Now running ensemble of forward model runs.");
        g_ens = gmodel.run_G_ensemble(physical_params, gmodel_settings, 1, statistics);
        
        # Deal with forward model runs we consider 'failed' runs (add conditions as we see fit)
        if any(all(g_ens .== 0, dims=1)) || any(isnan.(g_ens)) || any(.!isfinite.(g_ens))
            # Change failed runs into NaNs. The SampleSuccGauss method will handle these.
            g_ens = failed_runs(g_ens, physical_params);
        end
        
        # Store data for debugging purposes
        g_ens_success = success_run_filter(g_ens); 
        data_err = mean((truth_sample - mean(g_ens_success, dims=2)).^2);
        CSV.write(data_err_csv, Tables.table([data_err]), writeheader=false, append=true);
        CSV.write(params_csv, Tables.table(physical_params'), writeheader=false, append=true);    
        CSV.write(g_ens_csv, Tables.table(g_ens'), writeheader=false, append=true);           

        # Update EKI object
        terminated = EnsembleKalmanProcesses.update_ensemble!(ekiobj, g_ens);
        if !isnothing(terminated)
            final_iter = i - 1 # final update was previous iteration
            break
        end
    end

    N_iter = final_iter[1]; #in case it terminated early 
    
    # EKI results: Has the ensemble collapsed toward the truth?
    println("True parameters: ")
    println(params_true)
    println("\nEKI results:")
    println(EKP.get_ϕ_mean_final(priors, ekiobj))

    if constrain == true
        physical_params = comp2phys_transform(get_u_final(ekiobj), prior_lower, prior_upper);
    else
        physical_params = get_u_final(ekiobj);
    end
    CSV.write(params_csv, Tables.table(physical_params'), writeheader=false, append=true);


    u_stored = EKP.get_u(ekiobj, return_array = false)
    g_stored = EKP.get_g(ekiobj, return_array = false)

    save(
        joinpath(data_save_directory, "calibrate_results.jld2"),
        "inputs",
        u_stored,
        "outputs",
        g_stored,
        "priors",
        priors,
        "eki",
        ekiobj,
        "truth_sample",
        truth_sample,
        "truth_sample_mean",
        truth.mean,
        "truth_input_constrained",
        params_true, #constrained here, as in physically constrained space (unlike the u inputs), 
    )

end

main()

