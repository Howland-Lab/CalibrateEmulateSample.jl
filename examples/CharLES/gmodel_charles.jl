module gmodel

# Import useful modules
using Random
using Distributions
using LinearAlgebra
using FFTW
using Statistics
using Printf
using Plots
using Distributed

# Functions to be exported to main_template.jl
export run_G_ensemble
export run

# Settings to be inputted into the forward model run
mutable struct Settings
    # I have included examples of settings that may be needed
    
    # Choice of forward model statistics
    statistics::Int64
    # Averaging time period in hrs
    timeavg::Float64
    # Vertical grid points at which we obtain statistics
    points_mask::Vector{Int64}

    # Any settings that are needed and should be easily modifiable inputs for CharLES
end

# Function to transfer an ensemble of parameters into an ensemble of forward model runs
# INPUTS
# params: ensemble of parameters to be run in the forward model
# settings_local: any settings that the mid-fidelity runs may require
# statistics: Total number of output statistics from a single run  (e.g., 64 vector U + 64 vector TKE = 128)
# OUTPUTS
# g_ens: an ensemble of outputs that can be connected back to the input ensemble of parameters
function run_G_ensemble(params::Array{FT, 2}, settings_local::Settings, num_statistics) where {FT <: AbstractFloat}

    # Initialize ensemble
    N_ens = size(params, 2); # params of size (N_params x N_ens)
    g_ens = zeros(num_statistics, N_ens);    # ensemble of the outputs from the model runs is size (num_statistics x N_ens)

    # Parallel run settings: 0. Sequential, 1. Multithread, 2. Parallel map
    parallel_runs = 0;
    if parallel_runs == 0
        # Option 1. Run the model for each ensemble member sequentially
        for i in 1:N_ens
            println("Run ", i);
            g_ens[:, i] = run(params[:,i], settings_local);
            GC.gc();   # Manual memory dump to not run out of memory
        end
    elseif parallel_runs == 1
        # Option 2. run using multithreads
        Threads.@threads for i in 1:N_ens
            # Run model with the current parameters
            g_ens[:, i] = run(params[:,i], settings_local);
        end
    elseif parallel_runs == 2
        # Option 3. run using parallel map
        # Convert to an array of vectors and then run
        params_ensemble = [params[:, i] for i in 1:size(params, 2)]
        results = pmap(x -> run(x, settings_local), params_ensemble)
        g_ens = hcat(results...)
    end

    return g_ens;
end


# Function to run a single forward model run
# INPUTS
# params: parameter / set of parameters to be inputted (e.g., z0) into single run
# settings_local: (optional) any settings that may be relevant and needed for the run 
# OUTPUTS
# A vector of size (1xnum_statistics) that is the total output of a single run
function run(params::Array{FT, 1}, settings_local::Settings) where {FT <: AbstractFloat}

    # Code to interface with mid-fidelity CharLES
    # Needed features:
    # 1. To take the given parameter set (and settings) and input into CharLES
    # 2. To run CharLES
    # 3. To return output from CharLES when the run is over
    # And so, will need an automated way to check whether the run is over


end




end   # end of module