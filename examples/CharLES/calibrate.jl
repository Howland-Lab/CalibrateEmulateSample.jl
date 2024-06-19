# Template of calibrate phase, based off of Lorenz 96 example in CalibrateEmulateSample.jl

# Include forward model script
include("/path/gmodel_template.jl")

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
    # Loop over each column to check the conditions
    for j in 1:size(g_ens, 2)
        if all(g_ens[:,j] .== 0) || any(isnan.(g_ens[:,j])) || any(.!isfinite.(g_ens[:,j]))
           g_ens[:,j] .= NaN;
        end
    end
    return g_ens
end



function main()
    rng_seed = 4137
    rng = Random.MersenneTwister(rng_seed)
    # Output figure save directory
    homedir = pwd()
    println(homedir)
    figure_save_directory = homedir * "/output_fig/"
    data_save_directory = homedir * "/output_data/"
    if !isdir(figure_save_directory)
        mkdir(figure_save_directory)
    end
    if !isdir(data_save_directory)
        mkdir(data_save_directory)
    end

    
    # Settings
    perfect = false;    # perfect model vs imperfect model setting

    ###
    ###  Define the (true) parameters
    ###
    # Define the parameters that we want to learn
    z0_true = 0.1;     # Roughness length
    params_true = [z0_true];                             # Vector{Float64}
    param_names = ["z0"];                                # Vector{String}
    n_param = length(param_names);
    params_true = reshape(params_true, (n_param, 1));    # Matrix{Float64} 

    ###
    ### Define the parameter prior
    ###
    prior_z0 => Dict(
        "distribution" => Parameterized(Normal(0, 1)),
        "constraint" => no_constraint(),
        "name" => param_names[1],
    )

    # ParameterDistribution takes in dictionaries as input
    priors = PD([prior_z0])

    ###
    ###  Define the data from which we want to learn the parameters
    ###
    data_names = ["y0", "y1"]

    
    ### 
    ### Settings for mid-fidelity model (refered to as gmodel)
    ### e.g., time averaging window, start time, end time, etc.
    ###
    gmodel_settings = gmodel.Settings(...);

    ###
    ### Choose observational data and create internal variability covariance matrix
    ###
    if perfect
        # If running in perfect model setting, observational data is generated by the low/mid fidelity model in the CES framework
        # Example script below.
        yt_csv = "path/perfect_model_data.csv";
        yt = CSV.File(yt_csv; header=false); yt = DataFrame(yt); yt = Matrix(yt);
    else
        # If running in imperfect model setting, observational data is the geometry-resolving CharLES data.
        # Example script below.
        yt_csv = "path/imperfect_model_data.csv";
        yt = CSV.File(yt_csv; header=false); yt = DataFrame(yt); yt = Matrix(yt);
    end
    
    # Covariance: check for rank deficiency
    Γy = cov(yt, dims = 2)
    println(size(Γy), " ", rank(Γy))
    if rank(Γy) < size(Γy,1)
        @warn "Rank deficient."
    end    

    # Save locally for future postprocessing
    save_yt = string(data_save_directory, "/yt.csv");
    save_Γy = string(data_save_directory, "/Γy.csv");
    CSV.write(save_yt,  Tables.table(yt), writeheader=false);
    CSV.write(save_Γy,  Tables.table(Γy), writeheader=false);

    # Store data in Observations.Observation object
    truth = Observations.Observation(yt, Γy, data_names)
    truth_sample = truth.mean


    # Transformation settings
    # EKI requires a normal distribution for the parameter ensemble, while the actual physical parameters going into the forward model may not necessarily be a normal distribution. So, we transform between computational (EKI) and physical (model) space.
    constrain = true;   # Will you transform parameters from computational to physical
    constrain_method = "log";
    
    # 1. log transformation function (physical --> computational)
    log_transformation(x) = log.(x);
    # inverse log transformation function (computational --> physical)
    inverse_log_transformation(x) = exp.(x);
    

    ###
    ###  Calibrate: Ensemble Kalman Inversion
    ###
    # EKI settings
    N_ens = 20;             # number of ensemble members
    N_iter = 5;             # number of EKI iterations
    
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

    # To save QoIs
    params_csv = string(data_save_directory, "/params.csv");
    data_err_csv = string(data_save_directory, "/data_err.csv");
    g_ens_csv = string(data_save_directory, "/g_ens_", experiment_num, ".csv");
    
    # EKI iterations
    println("EKP inversion error:")
    err = zeros(N_iter)
    final_iter = [N_iter]
    for i in 1:N_iter
        if constrain == true
            physical_params = comp2phys_transform(get_u_final(ekiobj), constrain_method);
        else
            physical_params = get_u_final(ekiobj);
        end
        
        # Run ensemble of forward models
        g_ens = gmodel.run_G_ensemble(physical_params, gmodel_settings, num_statistics) 
        
        # Deal with forward model runs we consider 'failed' runs (add conditions as we see fit)
        if any(all(g_ens .== 0, dims=1)) || any(isnan.(g_ens)) || any(.!isfinite.(g_ens))
            # Change failed runs into NaNs. The SampleSuccGauss method will handle these.
            g_ens = failed_runs(g_ens, physical_params);
        end
        
        err[i] = EKP.get_error(ekiobj)[end] #mean((params_true - mean(params_i,dims=2)).^2)
        println("Iteration: " * string(i) * ", Error: " * string(err[i]))
        # Store data for debugging purposes
        CSV.write(params_csv, Tables.table(physical_params), writeheader=false, append=true);    
        CSV.write(g_ens_csv, Tables.table(g_ens), writeheader=false, append=true);           

        # Update EKI object
        terminated = EKP.update_ensemble!(ekiobj, g_ens);
        if !isnothing(terminated)
            final_iter = i - 1 # final update was previous iteration
            break
        end
    end
    N_iter = final_iter[1] #in case it terminated early 
    
    # EKI results: Has the ensemble collapsed toward the truth?
    println("True parameters: ")
    println(params_true)
    println("\nEKI results:")
    println(EKP.get_ϕ_mean_final(priors, ekiobj))

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
        params_true, #constrained here, as these are in a physically constrained space (unlike the u inputs),
        "parameter_error",
        err,  
    )

end

main()
